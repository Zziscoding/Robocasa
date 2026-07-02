from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions_gcc11")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
# Comfree requires MuJoCo 3.6.0 while the general RoboCasa package pins 3.3.1.
os.environ.setdefault("ROBOCASA_ALLOW_VERSION_MISMATCH", "1")
python_bin_dir = str(Path(sys.executable).resolve().parent)
path_entries = os.environ.get("PATH", "").split(os.pathsep)
if python_bin_dir not in path_entries:
    os.environ["PATH"] = os.pathsep.join([python_bin_dir] + path_entries)
if Path("/usr/bin/g++-11").exists():
    os.environ.setdefault("CXX", "/usr/bin/g++-11")
elif shutil.which("c++") is None and Path("/usr/bin/g++").exists():
    os.environ.setdefault("CXX", "/usr/bin/g++")
if Path("/usr/bin/gcc-11").exists():
    os.environ.setdefault("CC", "/usr/bin/gcc-11")
elif shutil.which("cc") is None and Path("/usr/bin/gcc").exists():
    os.environ.setdefault("CC", "/usr/bin/gcc")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-c",
            (
                "from robocasa.demos."
                "demo_close_toaster_oven_door_contact_curobo import main; main()"
            ),
            *sys.argv[1:],
        ],
    )

import mujoco
import numpy as np
import robosuite
import trimesh
from robosuite.controllers import load_composite_controller_config
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import robocasa  # noqa: F401
from robocasa.demos import demo_close_drawer_contact_curobo as D
from robocasa.demos import demo_open_drawer_contact_curobo as O
from robocasa.demos.mlqp_point_cabinet import LambdaContactControlOptimizer
from robocasa.demos.scene_process import MJWarpVisibility, build_or_load_scene_points


@dataclass
class ToasterOvenDoorFrame:
    center_world: np.ndarray
    rotation_world: np.ndarray
    half_size: np.ndarray
    geom_name: str
    fixture_name: str
    door_joint_name: str
    joint_min: float
    joint_max: float
    current_q: float
    target_q: float
    object_body_id: int
    frame_body_id: int
    tracked_point_object: np.ndarray


@dataclass(frozen=True)
class DoorTaskSpec:
    object_body_id: int
    frame_body_id: int
    tracked_point_object: np.ndarray
    current_position_local: np.ndarray
    target_position_local: np.ndarray
    object_position_world: np.ndarray
    object_rotation_world: np.ndarray
    cost: object


@dataclass
class CloseDoorStage:
    start_door_q: float
    target_door_q: float
    selected_contact_index: int
    selected_contact_world: np.ndarray
    selected_contact_local: np.ndarray
    selected_contact_cost: float
    dream_solution: object
    candidates: list
    dream_reports: list
    feasible_points_world: np.ndarray
    feasible_costs: np.ndarray
    feasible_tree: object
    task_current_position_local: np.ndarray
    task_target_position_local: np.ndarray
    dream_best_cost: float


@dataclass
class DoorTrajectoryFrame:
    q_arm: np.ndarray
    door_q: float


def _log(message, args=None):
    if args is not None and not getattr(args, "verbose", True):
        return
    print(message, flush=True)


def _raw_model(model):
    return getattr(model, "_model", model)


def _raw_data(data):
    return getattr(data, "_data", data)


def create_close_toaster_oven_door_env(args):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=args.robot,
    )

    def _make(has_offscreen_renderer):
        env = robosuite.make(
            env_name="CloseToasterOvenDoor",
            robots=args.robot,
            controller_configs=controller_config,
            has_renderer=bool(args.render or args.visualize_contact),
            has_offscreen_renderer=bool(has_offscreen_renderer),
            render_camera=None,
            renderer="mjviewer",
            use_camera_obs=False,
            ignore_done=True,
            reward_shaping=True,
            control_freq=args.control_freq,
            layout_ids=args.layout,
            style_ids=args.style,
            seed=args.seed,
        )
        env.reset()
        return env

    try:
        return _make(args.save_trajectory_videos)
    except ImportError as exc:
        if not args.save_trajectory_videos or args.require_video_renderer:
            raise
        warnings.warn(
            "Offscreen renderer initialization failed; trajectory video is disabled. "
            f"Original error: {exc}",
            RuntimeWarning,
        )
        args.save_trajectory_videos = False
        return _make(False)


def _toaster_oven(env):
    if not hasattr(env, "toaster_oven"):
        raise RuntimeError(
            "Expected CloseToasterOvenDoor environment with env.toaster_oven."
        )
    return env.toaster_oven


def _door_joint_info(env):
    fixture = _toaster_oven(env)
    joint_name = fixture.door_joint_names[0]
    joint_info = fixture._joint_infos.get(joint_name)
    if joint_info is None:
        raise RuntimeError(f"Cannot find door joint info for {joint_name!r}.")
    joint_min, joint_max = np.asarray(joint_info["range"], dtype=np.float64)
    return joint_name, float(joint_min), float(joint_max)


def _door_joint_value(env):
    joint_name, _, _ = _door_joint_info(env)
    return float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(joint_name)])


def _set_door_joint_value(env, value):
    joint_name, _, _ = _door_joint_info(env)
    env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(joint_name)] = float(value)


def _door_panel_geom_name(env):
    fixture_name = _toaster_oven(env).name
    model = env.sim.model
    preferred = (
        f"{fixture_name}_door_main",
        f"{fixture_name}_door_reg_main",
        f"{fixture_name}_door",
    )
    for name in preferred:
        if name in model._geom_name2id:
            return name
    candidates = [
        name
        for name in model._geom_name2id
        if name.startswith(f"{fixture_name}_door")
        and "handle" not in name
        and "visual" not in name
    ]
    if not candidates:
        raise RuntimeError(
            f"Cannot find toaster-oven door panel geom for {fixture_name!r}."
        )
    candidates.sort(
        key=lambda name: float(np.prod(model.geom_size[model.geom_name2id(name)])),
        reverse=True,
    )
    return candidates[0]


def _body_descendant_ids(model, root_body_id):
    descendants = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if (
                int(model.body_parentid[body_id]) in descendants
                and body_id not in descendants
            ):
                descendants.add(body_id)
                changed = True
    return descendants


def get_panel_frame(env, args=None):
    model = _raw_model(env.sim.model)
    data = _raw_data(env.sim.data)
    fixture = _toaster_oven(env)
    geom_name = _door_panel_geom_name(env)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    joint_name, joint_min, joint_max = _door_joint_info(env)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if geom_id < 0 or joint_id < 0:
        raise RuntimeError("Cannot resolve toaster-oven panel geom or door joint.")
    object_body_id = int(model.jnt_bodyid[joint_id])
    frame_body_id = int(model.body_parentid[object_body_id])
    center_world = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy()
    rotation_world = (
        np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3).copy()
    )
    object_position = np.asarray(data.xpos[object_body_id], dtype=np.float64)
    object_rotation = np.asarray(data.xmat[object_body_id], dtype=np.float64).reshape(
        3, 3
    )
    tracked_point_object = object_rotation.T @ (center_world - object_position)
    current_q = _door_joint_value(env)
    target_fraction = 0.0 if args is None else float(args.target_open_fraction)
    target_fraction = float(np.clip(target_fraction, 0.0, 1.0))
    target_q = joint_min + target_fraction * (joint_max - joint_min)
    target_q = min(float(current_q), float(target_q))
    return ToasterOvenDoorFrame(
        center_world=center_world,
        rotation_world=rotation_world,
        half_size=np.asarray(model.geom_size[geom_id], dtype=np.float64).copy(),
        geom_name=geom_name,
        fixture_name=fixture.name,
        door_joint_name=joint_name,
        joint_min=joint_min,
        joint_max=joint_max,
        current_q=float(current_q),
        target_q=float(target_q),
        object_body_id=object_body_id,
        frame_body_id=frame_body_id,
        tracked_point_object=tracked_point_object,
    )


def _point_local_in_frame(data, body_id, frame_id, point_object):
    object_position = np.asarray(data.xpos[body_id], dtype=np.float64)
    object_rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    point_world = object_position + object_rotation @ np.asarray(
        point_object, dtype=np.float64
    )
    frame_position = np.asarray(data.xpos[frame_id], dtype=np.float64)
    frame_rotation = np.asarray(data.xmat[frame_id], dtype=np.float64).reshape(3, 3)
    return frame_rotation.T @ (point_world - frame_position)


def _make_door_task_spec(env, panel, args):
    from robocasa.demos.dream import ObjectFramePointPositionTaskCost

    model = _raw_model(env.sim.model)
    data = _raw_data(env.sim.data)
    qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(data.qvel, dtype=np.float64).copy()
    current_local = _point_local_in_frame(
        data,
        panel.object_body_id,
        panel.frame_body_id,
        panel.tracked_point_object,
    )
    object_position = np.asarray(
        data.xpos[panel.object_body_id], dtype=np.float64
    ).copy()
    object_rotation = (
        np.asarray(data.xmat[panel.object_body_id], dtype=np.float64)
        .reshape(3, 3)
        .copy()
    )
    try:
        _set_door_joint_value(env, panel.target_q)
        env.sim.forward()
        target_local = _point_local_in_frame(
            _raw_data(env.sim.data),
            panel.object_body_id,
            panel.frame_body_id,
            panel.tracked_point_object,
        )
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
    cost = ObjectFramePointPositionTaskCost.from_mujoco(
        model,
        data,
        object_body_id=panel.object_body_id,
        frame_body_id=panel.frame_body_id,
        point_position_object=panel.tracked_point_object,
        target_position_local=target_local,
        weight=args.door_task_cost_weight,
    )
    return DoorTaskSpec(
        object_body_id=panel.object_body_id,
        frame_body_id=panel.frame_body_id,
        tracked_point_object=panel.tracked_point_object.copy(),
        current_position_local=current_local,
        target_position_local=target_local,
        object_position_world=object_position,
        object_rotation_world=object_rotation,
        cost=cost,
    )


def _door_geom_names(env, panel):
    model = _raw_model(env.sim.model)
    body_ids = _body_descendant_ids(model, panel.object_body_id)
    names = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) not in body_ids:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name:
            continue
        if "handle" in name and not getattr(env, "_include_door_handle_points", False):
            continue
        names.append(name)
    return tuple(names)


def _initialize_scene_processing(env, panel, args):
    args._scene_point_cloud = None
    args._scene_visibility = None
    args._scene_runtime_data = env.sim.data
    env._scene_point_cloud = None
    if not args.scene_process:
        return
    env._include_door_handle_points = bool(args.include_door_handle_points)
    point_cloud = build_or_load_scene_points(
        _toaster_oven(env).get_xml(),
        env.sim.model,
        args.scene_cache_dir,
        points_per_link=args.scene_points_per_link,
        seed=args.seed,
        force=args.scene_force_rebuild,
    )
    visibility = MJWarpVisibility(
        env.sim.model,
        env.sim.data,
        point_cloud,
        device=args.scene_visibility_device,
        hit_tolerance=args.scene_visibility_hit_tolerance,
        use_bvh=not args.scene_disable_bvh,
        allow_cpu_fallback=not args.scene_require_mjwarp,
    )
    args._scene_point_cloud = point_cloud
    args._scene_visibility = visibility
    env._scene_point_cloud = point_cloud
    env._scene_visibility = visibility
    _refresh_scene_visibility(env, args)
    _log(
        "Scene point cache: "
        f"{point_cloud.cache_dir} | links={len(point_cloud.links)} "
        f"points={point_cloud.size} visible={int(point_cloud.mask.sum())} "
        f"backend={visibility.backend_name}",
        args,
    )


def _refresh_scene_visibility(env, args):
    visibility = getattr(args, "_scene_visibility", None)
    if visibility is None:
        return None
    env.sim.forward()
    return visibility.update_from_camera(
        args.scene_visibility_camera,
        width=args.scene_visibility_width,
        height=args.scene_visibility_height,
    )


def _fallback_panel_samples(panel, args):
    x_lim = max(float(panel.half_size[0]) - float(args.panel_margin), 1e-4)
    z_lim = max(float(panel.half_size[2]) - float(args.panel_margin), 1e-4)
    xs = np.linspace(-x_lim, x_lim, max(int(args.grid_x), 1))
    zs = np.linspace(-z_lim, z_lim, max(int(args.grid_z), 1))
    points_panel = np.asarray(
        [[x, -float(panel.half_size[1]), z] for z in zs for x in xs],
        dtype=np.float64,
    )
    world = points_panel @ panel.rotation_world.T + panel.center_world
    outward = np.repeat(
        (-panel.rotation_world[:, 1]).reshape(1, 3),
        world.shape[0],
        axis=0,
    )
    return (
        world,
        outward,
        np.asarray([""] * len(world), dtype=object),
        np.full(len(world), -1, dtype=np.int64),
    )


def _visible_door_samples(env, panel, task_spec, args):
    point_cloud = getattr(args, "_scene_point_cloud", None)
    allowed_names = set(_door_geom_names(env, panel))
    if point_cloud is None:
        world, outward_world, link_names, point_indices = _fallback_panel_samples(
            panel, args
        )
    else:
        data = getattr(args, "_scene_runtime_data")
        world_chunks = []
        normal_chunks = []
        link_names = []
        point_indices = []
        for link in point_cloud.links:
            if not allowed_names.intersection(link.geom_names):
                continue
            if not args.include_door_handle_points and any(
                "handle" in name for name in link.geom_names
            ):
                continue
            visible = np.flatnonzero(np.asarray(link.mask, dtype=np.uint8) == 1)
            if visible.size == 0:
                continue
            world_chunks.append(link.world_points(data)[visible])
            normal_chunks.append(link.world_normals(data)[visible])
            link_names.extend([link.name] * visible.size)
            point_indices.extend(visible.tolist())
        if not world_chunks:
            raise RuntimeError(
                "No visible mask=1 toaster-oven door points are available. "
                "Check the visibility camera or rebuild/increase the scene point cache."
            )
        world = np.concatenate(world_chunks, axis=0).astype(np.float64)
        outward_world = O._normalize_rows(
            np.concatenate(normal_chunks, axis=0),
            -panel.rotation_world[:, 1],
        )
        link_names = np.asarray(link_names, dtype=object)
        point_indices = np.asarray(point_indices, dtype=np.int64)
    local_object = (
        world
        - np.asarray(task_spec.object_position_world, dtype=np.float64).reshape(1, 3)
    ) @ np.asarray(task_spec.object_rotation_world, dtype=np.float64)
    outward_local = outward_world @ np.asarray(
        task_spec.object_rotation_world, dtype=np.float64
    )
    force_normal_local = -O._normalize_rows(outward_local, [0.0, 1.0, 0.0])
    return (
        local_object,
        world,
        force_normal_local,
        O._normalize_rows(outward_world, -panel.rotation_world[:, 1]),
        np.asarray(link_names, dtype=object),
        np.asarray(point_indices, dtype=np.int64),
    )


def _contact_face(normal_object):
    normal = np.asarray(normal_object, dtype=np.float64)
    axis = int(np.argmax(np.abs(normal)))
    axis_name = ("x", "y", "z")[axis]
    sign = "positive" if normal[axis] >= 0.0 else "negative"
    return f"{sign}_{axis_name}"


def _build_contact_optimizer(panel, sample_count, args):
    mesh = trimesh.creation.box(
        extents=2.0 * np.asarray(panel.half_size, dtype=np.float64)
    )
    mesh_path = tempfile.NamedTemporaryFile(
        prefix="robocasa_toaster_oven_door_",
        suffix=".stl",
        delete=False,
    ).name
    mesh.export(mesh_path)
    optimizer = LambdaContactControlOptimizer(
        mesh_path,
        obj_mass=args.contact_obj_mass,
        arm_friction=args.contact_friction,
        contact_stiffness=args.contact_stiffness,
        time_step=args.contact_dt,
        max_contacts=1,
        sample_num=max(8, int(sample_count)),
        pos_coef=args.door_task_cost_weight,
        ori_coef=args.contact_ori_coef,
        task_position_weights=(1.0, 1.0, 1.0),
        qp_device=args.contact_qp_device,
        qp_maxiter=args.contact_qp_maxiter,
        qp_tol=args.contact_qp_tol,
        qp_regularization=args.contact_qp_regularization,
        qp_objective_scale=args.contact_qp_objective_scale,
        qp_batch_size=args.contact_qp_batch_size,
    )
    return optimizer, mesh_path


def evaluate_contacts(env, panel, task_spec, args):
    stage_t0 = time.time()
    (
        local_points,
        world_points,
        force_normals,
        outward_world,
        scene_link_names,
        scene_point_indices,
    ) = _visible_door_samples(env, panel, task_spec, args)
    tangent1, tangent2 = O._contact_tangent_frames(force_normals)
    optimizer, mesh_path = _build_contact_optimizer(panel, len(local_points), args)
    try:
        current_x = np.concatenate(
            [
                task_spec.current_position_local,
                np.array([1.0, 0.0, 0.0, 0.0]),
            ]
        )
        target_x = np.concatenate(
            [
                task_spec.target_position_local,
                np.array([1.0, 0.0, 0.0, 0.0]),
            ]
        )
        lam, x_plus, costs, statuses = optimizer._solve_once(
            x_d=target_x,
            current_x=current_x,
            tau_o=np.zeros(6, dtype=np.float64),
            n_arm=force_normals,
            t1=tangent1,
            t2=tangent2,
            p_arm=local_points,
            curr_ori_coef=0.0,
            lam_upper_bound=args.contact_lam_upper_bound,
            task_target_position=task_spec.target_position_local,
            task_current_position=task_spec.current_position_local,
        )
        candidates = []
        for index in range(len(local_points)):
            cost = float(costs[index])
            feasible = bool(np.isfinite(cost) and cost <= args.contact_cost_threshold)
            candidate = D.ContactCandidate(
                local_point=np.asarray(local_points[index], dtype=np.float64),
                world_point=np.asarray(world_points[index], dtype=np.float64),
                cost=cost,
                lam=np.asarray(lam[index], dtype=np.float64),
                resulting_pose=np.asarray(x_plus[index], dtype=np.float64),
                solver_status=str(statuses[index]),
                feasible=feasible,
            )
            candidate.force_normal_local = np.asarray(
                force_normals[index], dtype=np.float64
            )
            candidate.outward_local = -candidate.force_normal_local
            candidate.approach_world = np.asarray(
                outward_world[index], dtype=np.float64
            )
            candidate.contact_surface = _contact_face(candidate.outward_local)
            candidate.visible = True
            candidate.scene_link_name = str(scene_link_names[index])
            candidate.scene_point_index = int(scene_point_indices[index])
            candidate.scene_point_distance = (
                0.0 if candidate.scene_point_index >= 0 else float("nan")
            )
            candidates.append(candidate)
        candidates.sort(key=lambda candidate: candidate.cost)
        feasible_candidates = [
            candidate for candidate in candidates if candidate.feasible
        ]
        if not feasible_candidates and args.require_feasible_contact:
            best = candidates[0]
            raise RuntimeError(
                "No feasible visible toaster-oven door contact point found. "
                f"Best cost={best.cost:.6f}, "
                f"threshold={args.contact_cost_threshold:.6f}, "
                f"status={best.solver_status}."
            )
        selected = feasible_candidates[0] if feasible_candidates else candidates[0]
        feasible_points = np.asarray(
            [candidate.world_point for candidate in feasible_candidates],
            dtype=np.float64,
        ).reshape(-1, 3)
        feasible_costs = np.asarray(
            [candidate.cost for candidate in feasible_candidates],
            dtype=np.float64,
        )
        feasible_tree = cKDTree(feasible_points) if feasible_points.shape[0] else None
        _log(
            f"[contact] GPU QP solved {len(candidates)} visible points in "
            f"{time.time() - stage_t0:.3f}s; feasible={len(feasible_candidates)}",
            args,
        )
        return (
            candidates,
            selected,
            feasible_points,
            feasible_costs,
            feasible_tree,
        )
    finally:
        try:
            os.unlink(mesh_path)
        except OSError:
            pass


def _candidate_pose_at_q(env, panel, candidate, q_value):
    qpos = env.sim.data.qpos.copy()
    qvel = env.sim.data.qvel.copy()
    try:
        _set_door_joint_value(env, q_value)
        env.sim.forward()
        data = _raw_data(env.sim.data)
        object_position = np.asarray(data.xpos[panel.object_body_id], dtype=np.float64)
        object_rotation = np.asarray(
            data.xmat[panel.object_body_id], dtype=np.float64
        ).reshape(3, 3)
        point_world = object_position + object_rotation @ np.asarray(
            candidate.local_point, dtype=np.float64
        )
        outward_world = D._normalize(
            object_rotation @ np.asarray(candidate.outward_local, dtype=np.float64)
        )
        return point_world.copy(), outward_world.copy()
    finally:
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()


def _gripper_rotation(outward_world, roll_angle):
    push_world = -D._normalize(outward_world)
    return D._make_gripper_contact_rotation(push_world) @ D._rot_about_axis(
        [1.0, 0.0, 0.0],
        roll_angle,
    )


def _make_dream_report(
    candidate,
    candidate_index,
    status,
    reason,
    position_error,
    collision_free,
):
    return D.MinkContactAttemptReport(
        drawer_candidate_index=int(candidate_index),
        drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
        drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
        drawer_contact_cost=float(candidate.cost),
        contact_feasible=bool(candidate.feasible),
        status=str(status),
        reason=str(reason),
        best_ee_sample_index=0,
        best_ee_sample_name="dream_gripper_proxy",
        best_position_error=float(position_error),
        best_collision_free=bool(collision_free),
    )


def _door_nominal_ee_sequence(
    env,
    panel,
    candidate,
    contact_offset,
    horizon_steps,
    roll_angle,
    args,
):
    q_values = np.linspace(
        panel.current_q,
        panel.target_q,
        int(horizon_steps),
        dtype=np.float64,
    )
    poses = []
    qpos = env.sim.data.qpos.copy()
    qvel = env.sim.data.qvel.copy()
    try:
        for q_value in q_values:
            _set_door_joint_value(env, q_value)
            env.sim.forward()
            data = _raw_data(env.sim.data)
            object_position = np.asarray(
                data.xpos[panel.object_body_id], dtype=np.float64
            )
            object_rotation = np.asarray(
                data.xmat[panel.object_body_id], dtype=np.float64
            ).reshape(3, 3)
            point_world = object_position + object_rotation @ np.asarray(
                candidate.local_point, dtype=np.float64
            )
            outward_world = D._normalize(
                object_rotation @ np.asarray(candidate.outward_local, dtype=np.float64)
            )
            target_rotation = _gripper_rotation(outward_world, roll_angle)
            desired_contact = point_world - outward_world * float(
                args.dream_initial_penetration
            )
            frame_position = desired_contact - target_rotation @ np.asarray(
                contact_offset, dtype=np.float64
            )
            poses.append(
                np.concatenate(
                    [
                        frame_position,
                        D._quat_wxyz_from_matrix(target_rotation),
                    ]
                )
            )
    finally:
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
    return np.asarray(poses, dtype=np.float64)


def _optimize_seed_pose_with_dream(
    env,
    panel,
    candidate,
    task_spec,
    robot_state,
    q_seed_robot,
    contact_offset,
    args,
):
    from robocasa.demos.dream import (
        BatchedLMIKCost,
        ComfreeEEMPPI,
        CompositeRolloutCost,
        DreamConfig,
        NonPenetrationCost,
    )

    device = args.dream_device
    if device == "auto":
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = _raw_model(env.sim.model)
    data = _raw_data(env.sim.data)
    qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(data.qvel, dtype=np.float64).copy()
    arm_joint_names = robot_state["robocasa_joint_names"]
    try:
        D._set_env_arm_q(
            env,
            arm_joint_names,
            D._arm_q_from_robot_model_q(
                env.robots[0].robot_model.mujoco_model,
                q_seed_robot,
                arm_joint_names,
            ),
        )
        _set_door_joint_value(env, panel.current_q)
        env.sim.forward()
        horizon_steps = max(int(args.dream_horizon_steps), 2)
        sim_dt = max(float(args.dream_dt), 1e-4)
        knot_steps = min(max(int(args.dream_knot_steps), 1), horizon_steps)
        config = DreamConfig(
            device=device,
            seed=int(args.seed),
            num_samples=max(int(args.dream_num_samples), 1),
            num_perturb_samples=max(int(args.dream_num_perturb_samples), 1),
            sim_dt=sim_dt,
            horizon=horizon_steps * sim_dt,
            knot_dt=knot_steps * sim_dt,
            max_num_iterations=max(int(args.dream_iterations), 1),
            pos_noise_scale=float(args.dream_position_noise),
            rot_noise_scale=float(args.dream_rotation_noise),
            zero_first_knot_noise=False,
            contact_stiffness=float(args.dream_contact_stiffness),
            contact_damping=float(args.dream_contact_damping),
            nconmax_per_env=max(int(args.dream_nconmax_per_env), 1),
            njmax_per_env=max(int(args.dream_njmax_per_env), 1),
            compile_cuda_graph=not args.disable_dream_cuda_graph,
            cost_aggregation="sum",
        )
        robot_geoms, environment_geoms = O._dream_penetration_geom_sets(model)
        rollout_cost = CompositeRolloutCost(
            [
                task_spec.cost,
                NonPenetrationCost(
                    robot_geoms,
                    environment_geoms,
                    margin=float(args.dream_penetration_margin),
                    weight=float(args.dream_penetration_weight),
                ),
                BatchedLMIKCost(
                    damping=float(args.dream_lm_damping),
                    residual_weight=float(args.dream_ik_weight),
                    step_weight=float(args.dream_lm_step_weight),
                    joint_limit_weight=float(args.dream_joint_limit_weight),
                ),
            ]
        )
        solver = ComfreeEEMPPI(
            model,
            data,
            ee_site=args.mink_contact_frame,
            cost_fn=rollout_cost,
            config=config,
            arm_joint_names=arm_joint_names,
        )
        nominal = _door_nominal_ee_sequence(
            env,
            panel,
            candidate,
            contact_offset,
            horizon_steps,
            float(args.dream_initial_roll),
            args,
        )
        solver.sync_from_mujoco(data)
        return solver.solve(nominal)
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)


def solve_contact_pose_with_dream(
    env,
    panel,
    candidates,
    task_spec,
    robot_state,
    args,
):
    region_indices, seed_index = O._sample_dense_feasible_region(candidates, args)
    if seed_index < 0:
        best_index = int(np.argmin([candidate.cost for candidate in candidates]))
        return None, [
            _make_dream_report(
                candidates[best_index],
                best_index,
                "failed",
                "dream:no_feasible_dense_contact_region",
                float("inf"),
                False,
            )
        ]
    candidate = candidates[seed_index]
    robot_model = env.robots[0].robot_model.mujoco_model
    arm_joint_names = robot_state["robocasa_joint_names"]
    frame_name = args.mink_contact_frame
    q_initial = D._current_robot_model_q(env, robot_model)
    q_posture = q_initial.copy()
    posture_cost = D._make_mink_posture_cost(robot_model, arm_joint_names, args)
    ee_sample_name, contact_offset, ee_contact_geom_name = O._dream_contact_offset(
        env, frame_name, args
    )
    point_world, outward_world = _candidate_pose_at_q(
        env, panel, candidate, panel.current_q
    )
    target_rotation = _gripper_rotation(outward_world, float(args.dream_initial_roll))
    initial_contact = point_world - outward_world * float(
        args.dream_initial_penetration
    )
    target_frame_position = initial_contact - target_rotation @ np.asarray(
        contact_offset
    )
    reports = []
    try:
        q_seed, seed_error = D._solve_mink_frame_pose(
            env,
            frame_name,
            target_frame_position,
            target_rotation,
            q_initial,
            q_posture,
            posture_cost,
            args,
        )
        result = _optimize_seed_pose_with_dream(
            env,
            panel,
            candidate,
            task_spec,
            robot_state,
            q_seed,
            contact_offset,
            args,
        )
        sequence = np.asarray(result.ee_pose_sequence.detach().cpu(), dtype=np.float64)
        waypoint_count = min(
            max(int(args.dream_curobo_waypoint_count), 2),
            sequence.shape[0],
        )
        sequence_indices = np.unique(
            np.linspace(
                0,
                sequence.shape[0] - 1,
                waypoint_count,
                dtype=np.int64,
            )
        )
        precontact_frame_position = (
            point_world
            + outward_world * float(args.precontact_distance)
            - target_rotation @ np.asarray(contact_offset)
        )
        target_gripper_poses = [
            ("precontact", precontact_frame_position, target_rotation)
        ]
        total = len(sequence_indices)
        for target_index, sequence_index in enumerate(sequence_indices, start=1):
            pose = sequence[int(sequence_index)]
            rotation = Rotation.from_quat(np.roll(pose[3:7], -1)).as_matrix()
            target_gripper_poses.append(
                (
                    f"dream_{target_index:03d}_of_{total:03d}",
                    pose[:3].copy(),
                    rotation,
                )
            )
        dream_feasible = bool(
            np.isfinite(result.best_cost)
            and result.best_cost <= float(args.dream_feasible_cost_threshold)
        )
        seed_arm_q = D._arm_q_from_robot_model_q(robot_model, q_seed, arm_joint_names)
        solution = D.MinkContactPoseSolution(
            drawer_candidate_index=int(seed_index),
            drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
            drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
            drawer_contact_cost=float(candidate.cost),
            ee_sample_index=0,
            ee_sample_name=f"dream_dense:{ee_sample_name}",
            ee_contact_geom_name=str(ee_contact_geom_name),
            contact_frame=f"{frame_name}:dream_dense",
            contact_offset_local=np.asarray(contact_offset, dtype=np.float64),
            roll_angle=float(args.dream_initial_roll),
            q_waypoints=np.asarray([seed_arm_q], dtype=np.float64),
            target_gripper_poses=target_gripper_poses,
            contact_position_error=float(seed_error),
            collision_free=dream_feasible,
        )
        solution.contact_surface = candidate.contact_surface
        reports.append(
            _make_dream_report(
                candidate,
                seed_index,
                "success" if dream_feasible else "failed",
                (
                    f"dream:dense_region={len(region_indices)}"
                    f"|seed_error={seed_error:.5f}"
                    f"|best_cost={result.best_cost:.6f}"
                    f"|cost_threshold={args.dream_feasible_cost_threshold:.6f}"
                ),
                seed_error,
                dream_feasible,
            )
        )
        args._last_dream_best_cost = float(result.best_cost)
        # Keep Dream's final sequence even when it misses the configured
        # feasibility threshold so it can still be executed and inspected.
        return solution, reports
    except Exception as exc:
        reports.append(
            _make_dream_report(
                candidate,
                seed_index,
                "failed",
                f"dream:exception:{exc.__class__.__name__}:{exc}",
                float("inf"),
                False,
            )
        )
        return None, reports


def _door_trajectory_from_segments(panel, q_traj, segments):
    arm = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    door_values = []
    current_q = float(panel.current_q)
    for segment in segments or []:
        steps = max(int(segment.get("steps", 0)), 0)
        if steps <= 0:
            continue
        name = str(segment.get("name", ""))
        if name.startswith("dream_"):
            parts = name.split("_")
            try:
                progress = np.clip(int(parts[1]) / max(int(parts[3]), 1), 0.0, 1.0)
            except (IndexError, ValueError):
                progress = 1.0
            target_q = panel.current_q + progress * (panel.target_q - panel.current_q)
            values = np.linspace(current_q, target_q, steps, dtype=np.float64)
            current_q = float(target_q)
        else:
            values = np.full(steps, current_q, dtype=np.float64)
        door_values.extend(values.tolist())
    if len(door_values) < arm.shape[0]:
        door_values.extend([current_q] * (arm.shape[0] - len(door_values)))
    return np.asarray(door_values[: arm.shape[0]], dtype=np.float64)


def _trajectory_frames(panel, q_traj, segments, stage=None, robot_state=None):
    del stage, robot_state
    if q_traj is None or not np.asarray(q_traj).size:
        return []
    arm = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    door = _door_trajectory_from_segments(panel, arm, segments)
    return [
        DoorTrajectoryFrame(q_arm=q_arm, door_q=float(q_door))
        for q_arm, q_door in zip(arm, door)
    ]


def _final_curobo_target(target_hand_poses):
    if not target_hand_poses:
        return []
    name, position, rotation = target_hand_poses[-1]
    return [(f"{name}:final_only", position, rotation)]


def _final_dream_ee_pose(stage):
    solution = stage.dream_solution
    if solution is None or not solution.target_gripper_poses:
        return None
    _, position, rotation = solution.target_gripper_poses[-1]
    return np.asarray(position), np.asarray(rotation)


def _contact_marker_at_q(env, panel, stage, door_q):
    candidate = stage.candidates[stage.selected_contact_index]
    point, _ = _candidate_pose_at_q(env, panel, candidate, door_q)
    return point


def save_trajectory_video(
    env,
    panel,
    stage,
    robot_state,
    q_traj,
    segments,
    args,
):
    if not args.save_trajectory_videos:
        return None
    if env.sim._render_context_offscreen is None:
        _log("Trajectory video skipped: offscreen renderer is unavailable.", args)
        return None
    frames = _trajectory_frames(
        panel, q_traj, segments, stage=stage, robot_state=robot_state
    )
    if not frames:
        _log("Trajectory video skipped: no cuRobo trajectory is available.", args)
        return None
    output_dir = Path(args.video_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.video_prefix}_trajectory.mp4"
    qpos = env.sim.data.qpos.copy()
    qvel = env.sim.data.qvel.copy()
    rendered = []
    try:
        for frame in frames:
            D._set_env_arm_q(env, robot_state["robocasa_joint_names"], frame.q_arm)
            _set_door_joint_value(env, frame.door_q)
            env.sim.forward()
            marker = _contact_marker_at_q(env, panel, stage, frame.door_q)
            marker_specs = [
                (
                    marker,
                    args.video_marker_size,
                    (1.0, 0.0, 0.0, 1.0),
                )
            ]
            if stage.dream_solution is not None:
                try:
                    ee_marker = D._ee_sample_world(
                        env,
                        args.mink_contact_frame,
                        stage.dream_solution.contact_offset_local,
                    )
                    marker_specs.append(
                        (
                            ee_marker,
                            args.video_marker_size * 0.75,
                            (0.0, 1.0, 0.15, 1.0),
                        )
                    )
                except Exception:
                    pass
            current_panel = get_panel_frame(env, args)
            rendered.append(
                D._render_frame(
                    env,
                    current_panel,
                    stage.dream_solution,
                    args,
                    marker_specs=marker_specs,
                )
            )
    finally:
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
    D._write_mp4(output_path, rendered, args.video_fps, args)
    _log(f"Saved cuRobo trajectory video to {output_path}", args)
    return str(output_path)


def visualize_contact(
    env,
    panel,
    stage,
    robot_state,
    q_traj,
    segments,
    args,
):
    if not args.visualize_contact:
        return
    # mjviewer is initialized lazily by robosuite.
    try:
        env.render()
    except Exception as exc:
        _log(f"Contact visualization skipped: renderer launch failed: {exc}", args)
        return
    if env.viewer is None:
        _log("Contact visualization skipped: renderer is unavailable.", args)
        return
    frames = _trajectory_frames(
        panel, q_traj, segments, stage=stage, robot_state=robot_state
    )
    ghost_only = not frames
    ghost_pose = _final_dream_ee_pose(stage)
    if ghost_only and ghost_pose is None:
        _log(
            "Contact visualization skipped: no cuRobo trajectory or Dream EE result is available.",
            args,
        )
        return
    try:
        env.viewer.update()
    except Exception as exc:
        _log(f"Contact visualization skipped: {exc}", args)
        return
    viewer = getattr(env.viewer, "viewer", None)
    if viewer is None:
        return
    marker_rgba = np.array([1.0, 0.0, 0.0, args.contact_marker_alpha], dtype=np.float32)
    frame_index = 0
    started = time.time()
    camera_initialized = False
    try:
        while True:
            if hasattr(viewer, "is_running") and not viewer.is_running():
                break
            if ghost_only:
                marker = stage.selected_contact_world
            else:
                frame = frames[frame_index]
                D._set_env_arm_q(env, robot_state["robocasa_joint_names"], frame.q_arm)
                _set_door_joint_value(env, frame.door_q)
                env.sim.forward()
                marker = _contact_marker_at_q(env, panel, stage, frame.door_q)
            if hasattr(viewer, "user_scn"):
                viewer.user_scn.ngeom = 0
                D._draw_viewer_sphere(
                    viewer,
                    marker,
                    args.contact_marker_size,
                    marker_rgba,
                )
                if ghost_only:
                    O._draw_ghost_ee(viewer, ghost_pose[0], ghost_pose[1], args)
            if hasattr(viewer, "cam") and not camera_initialized:
                viewer.cam.lookat[:] = marker
                viewer.cam.distance = args.contact_camera_distance
                viewer.cam.azimuth = args.contact_camera_azimuth
                viewer.cam.elevation = args.contact_camera_elevation
                camera_initialized = True
            if hasattr(viewer, "sync"):
                viewer.sync()
            if not ghost_only:
                if args.visualize_loop_trajectory:
                    frame_index = (frame_index + 1) % len(frames)
                else:
                    frame_index = min(frame_index + 1, len(frames) - 1)
            if (
                args.visualize_contact_seconds > 0
                and time.time() - started >= args.visualize_contact_seconds
            ):
                break
            time.sleep(1.0 / max(args.contact_marker_fps, 1.0))
    except KeyboardInterrupt:
        pass


def save_outputs(
    path,
    env,
    panel,
    stage,
    target_hand_poses,
    q_traj,
    segments,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree_path = path.with_name(f"{path.stem}_feasible_kdtree.pkl")
    scene_point_cloud = getattr(env, "_scene_point_cloud", None)
    selected = stage.candidates[stage.selected_contact_index]
    metadata = {
        "task": "CloseToasterOvenDoor",
        "toaster_oven_name": _toaster_oven(env).name,
        "door_geom": panel.geom_name,
        "door_joint_name": panel.door_joint_name,
        "door_joint_range": [panel.joint_min, panel.joint_max],
        "door_q_start": stage.start_door_q,
        "door_q_target": stage.target_door_q,
        "task_current_position_local": stage.task_current_position_local.tolist(),
        "task_target_position_local": stage.task_target_position_local.tolist(),
        "candidate_count": len(stage.candidates),
        "visible_candidate_count": int(
            sum(bool(getattr(c, "visible", True)) for c in stage.candidates)
        ),
        "feasible_candidate_count": int(
            sum(bool(c.feasible) for c in stage.candidates)
        ),
        "feasible_kdtree_size": int(stage.feasible_points_world.shape[0]),
        "feasible_kdtree_path": str(tree_path),
        "selected_contact": {
            "index": stage.selected_contact_index,
            "world_point": stage.selected_contact_world.tolist(),
            "object_local_point": stage.selected_contact_local.tolist(),
            "cost": stage.selected_contact_cost,
            "surface": str(getattr(selected, "contact_surface", "")),
            "scene_link": str(getattr(selected, "scene_link_name", "")),
            "scene_point_index": int(getattr(selected, "scene_point_index", -1)),
        },
        "dream_feasible": bool(
            stage.dream_solution is not None and stage.dream_solution.collision_free
        ),
        "dream_best_cost": stage.dream_best_cost,
        "dream_attempt_reason_counts": O._report_reason_counts(stage.dream_reports),
        "has_executable_arm_trajectory": bool(
            q_traj is not None and np.asarray(q_traj).size
        ),
        "segments": segments,
        "scene_processing": (
            scene_point_cloud.summary() if scene_point_cloud is not None else None
        ),
    }
    np.savez(
        path,
        q_traj=(
            np.asarray(q_traj, dtype=np.float64)
            if q_traj is not None
            else np.zeros((0, 7), dtype=np.float64)
        ),
        door_q_trajectory=(
            _door_trajectory_from_segments(panel, q_traj, segments)
            if q_traj is not None and np.asarray(q_traj).size
            else np.zeros(0, dtype=np.float64)
        ),
        door_q_endpoints=np.asarray(
            [stage.start_door_q, stage.target_door_q], dtype=np.float64
        ),
        selected_contact_world=stage.selected_contact_world,
        selected_contact_object_local=stage.selected_contact_local,
        selected_contact_cost=np.asarray(stage.selected_contact_cost),
        candidate_world_points=np.asarray(
            [c.world_point for c in stage.candidates], dtype=np.float64
        ),
        candidate_object_local_points=np.asarray(
            [c.local_point for c in stage.candidates], dtype=np.float64
        ),
        candidate_costs=np.asarray(
            [c.cost for c in stage.candidates], dtype=np.float64
        ),
        candidate_feasible=np.asarray(
            [c.feasible for c in stage.candidates], dtype=bool
        ),
        feasible_contact_points_world=stage.feasible_points_world,
        feasible_contact_costs=stage.feasible_costs,
        task_current_position_local=stage.task_current_position_local,
        task_target_position_local=stage.task_target_position_local,
        dream_best_cost=np.asarray(stage.dream_best_cost),
        dream_seed_q_waypoints=(
            np.asarray(stage.dream_solution.q_waypoints, dtype=np.float64)
            if stage.dream_solution is not None
            else np.zeros((0, 7), dtype=np.float64)
        ),
        target_hand_pos_base=np.asarray(
            [pose[1] for pose in target_hand_poses], dtype=np.float64
        ).reshape(-1, 3),
        target_hand_quat_wxyz_base=np.asarray(
            [D._quat_wxyz_from_matrix(pose[2]) for pose in target_hand_poses],
            dtype=np.float64,
        ).reshape(-1, 4),
        scene_points_world=(
            scene_point_cloud.world_points(env.sim.data)
            if scene_point_cloud is not None
            else np.zeros((0, 3), dtype=np.float32)
        ),
        scene_visibility_mask=(
            scene_point_cloud.mask
            if scene_point_cloud is not None
            else np.zeros(0, dtype=np.uint8)
        ),
        metadata_json=json.dumps(metadata, indent=2),
    )
    with tree_path.open("wb") as stream:
        pickle.dump(
            {
                "points_world": stage.feasible_points_world,
                "costs": stage.feasible_costs,
                "tree": stage.feasible_tree,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "CloseToasterOvenDoor contact demo: visible scene points -> GPU QP "
            "contact filtering -> dense-region Mink seed -> Dream/Comfree EE MPC "
            "-> cuRobo Franka execution trajectory."
        )
    )
    parser.add_argument("--robot", type=str, default="PandaOmron")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=int, default=-1)
    parser.add_argument("--style", type=int, default=-1)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)

    parser.add_argument("--target-open-fraction", type=float, default=0.02)
    parser.add_argument("--precontact-distance", type=float, default=0.08)
    parser.add_argument("--grid-x", type=int, default=5)
    parser.add_argument("--grid-z", type=int, default=3)
    parser.add_argument("--panel-margin", type=float, default=0.018)

    parser.add_argument("--contact-cost-threshold", type=float, default=0.25)
    parser.add_argument("--require-feasible-contact", action="store_true")
    parser.add_argument("--contact-lam-upper-bound", type=float, default=2.0)
    parser.add_argument("--contact-obj-mass", type=float, default=0.26)
    parser.add_argument("--contact-friction", type=float, default=0.9)
    parser.add_argument("--contact-stiffness", type=float, default=12.5)
    parser.add_argument("--contact-dt", type=float, default=0.01)
    parser.add_argument("--contact-ori-coef", type=float, default=0.0005)
    parser.add_argument("--door-task-cost-weight", type=float, default=1.0)
    parser.add_argument("--contact-qp-device", type=str, default="auto")
    parser.add_argument("--contact-qp-maxiter", type=int, default=1000)
    parser.add_argument("--contact-qp-tol", type=float, default=1e-5)
    parser.add_argument("--contact-qp-regularization", type=float, default=1e-6)
    parser.add_argument("--contact-qp-objective-scale", type=float, default=100.0)
    parser.add_argument("--contact-qp-batch-size", type=int, default=8192)

    parser.add_argument("--scene-process", dest="scene_process", action="store_true")
    parser.add_argument(
        "--disable-scene-process",
        dest="scene_process",
        action="store_false",
    )
    parser.set_defaults(scene_process=True)
    parser.add_argument(
        "--scene-cache-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "scene_point_cache"),
    )
    parser.add_argument("--scene-points-per-link", type=int, default=2048)
    parser.add_argument("--scene-force-rebuild", action="store_true")
    parser.add_argument(
        "--scene-visibility-camera",
        type=str,
        default="robot0_agentview_left",
    )
    parser.add_argument("--scene-visibility-device", type=str, default="cuda:0")
    parser.add_argument("--scene-visibility-width", type=int, default=640)
    parser.add_argument("--scene-visibility-height", type=int, default=480)
    parser.add_argument("--scene-visibility-hit-tolerance", type=float, default=0.012)
    parser.add_argument("--scene-disable-bvh", action="store_true")
    parser.add_argument("--scene-require-mjwarp", action="store_true")
    parser.add_argument("--include-door-handle-points", action="store_true")

    parser.add_argument("--dream-device", type=str, default="auto")
    parser.add_argument("--dream-num-samples", type=int, default=256)
    parser.add_argument("--dream-num-perturb-samples", type=int, default=1)
    parser.add_argument("--dream-horizon-steps", type=int, default=24)
    parser.add_argument("--dream-dt", type=float, default=0.02)
    parser.add_argument("--dream-knot-steps", type=int, default=4)
    parser.add_argument("--dream-iterations", type=int, default=6)
    parser.add_argument("--dream-position-noise", type=float, default=0.012)
    parser.add_argument("--dream-rotation-noise", type=float, default=0.035)
    parser.add_argument("--dream-contact-stiffness", type=float, default=0.2)
    parser.add_argument("--dream-contact-damping", type=float, default=0.001)
    parser.add_argument("--dream-nconmax-per-env", type=int, default=120)
    parser.add_argument("--dream-njmax-per-env", type=int, default=500)
    parser.add_argument("--disable-dream-cuda-graph", action="store_true")
    parser.add_argument("--dream-dense-region-radius", type=float, default=0.035)
    parser.add_argument("--dream-dense-region-quantile", type=float, default=0.7)
    parser.add_argument("--dream-initial-penetration", type=float, default=0.003)
    parser.add_argument("--dream-penetration-margin", type=float, default=0.003)
    parser.add_argument("--dream-penetration-weight", type=float, default=5000.0)
    parser.add_argument("--dream-lm-damping", type=float, default=0.02)
    parser.add_argument("--dream-ik-weight", type=float, default=10.0)
    parser.add_argument("--dream-lm-step-weight", type=float, default=0.001)
    parser.add_argument("--dream-joint-limit-weight", type=float, default=50.0)
    parser.add_argument("--dream-feasible-cost-threshold", type=float, default=1000.0)
    parser.add_argument("--dream-curobo-waypoint-count", type=int, default=8)
    parser.add_argument("--dream-initial-roll", type=float, default=0.0)
    parser.add_argument(
        "--dream-gripper-contact-mode",
        type=str,
        default="front_center",
        choices=("front_center", "grip_site"),
    )

    parser.add_argument(
        "--mink-contact-frame",
        type=str,
        default="gripper0_right_grip_site",
    )
    parser.add_argument("--gripper-front-ring-forward-offset", type=float, default=0.0)
    parser.add_argument("--gripper-front-ring-radius-scale", type=float, default=1.0)
    parser.add_argument("--mink-ee-sample-count", type=int, default=15)
    parser.add_argument("--mink-max-iters", type=int, default=120)
    parser.add_argument("--mink-dt", type=float, default=0.02)
    parser.add_argument("--mink-solver", type=str, default="quadprog")
    parser.add_argument("--mink-damping", type=float, default=1e-5)
    parser.add_argument("--mink-position-cost", type=float, default=80.0)
    parser.add_argument("--mink-orientation-cost", type=float, default=0.8)
    parser.add_argument("--mink-arm-posture-cost", type=float, default=0.02)
    parser.add_argument("--mink-locked-dof-cost", type=float, default=200.0)
    parser.add_argument("--mink-posture-lm-damping", type=float, default=2.0)
    parser.add_argument("--mink-frame-lm-damping", type=float, default=1.0)
    parser.add_argument("--mink-position-tolerance", type=float, default=0.008)

    parser.add_argument("--curobo-robot-cfg", type=str, default="franka.yml")
    parser.add_argument("--curobo-trajopt-tsteps", type=int, default=32)
    parser.add_argument("--curobo-interpolation-dt", type=float, default=0.02)
    parser.add_argument("--curobo-ik-seeds", type=int, default=16)
    parser.add_argument("--curobo-graph-seeds", type=int, default=2)
    parser.add_argument("--curobo-trajopt-seeds", type=int, default=2)
    parser.add_argument("--curobo-max-attempts", type=int, default=2)
    parser.add_argument("--curobo-enable-graph-attempt", type=int, default=1)
    parser.add_argument("--disable-curobo-self-collision", action="store_true")
    parser.add_argument("--disable-curobo-cuda-graph", action="store_true")

    parser.add_argument(
        "--visualize-contact",
        dest="visualize_contact",
        action="store_true",
    )
    parser.add_argument(
        "--no-visualize-contact",
        dest="visualize_contact",
        action="store_false",
    )
    parser.set_defaults(visualize_contact=True)
    parser.add_argument("--visualize-contact-seconds", type=float, default=0.0)
    parser.add_argument("--contact-marker-size", type=float, default=0.012)
    parser.add_argument("--contact-marker-alpha", type=float, default=1.0)
    parser.add_argument("--contact-marker-fps", type=float, default=30.0)
    parser.add_argument("--ghost-ee-alpha", type=float, default=0.35)
    parser.add_argument("--ghost-ee-scale", type=float, default=1.0)
    parser.add_argument("--contact-camera-distance", type=float, default=0.9)
    parser.add_argument("--contact-camera-azimuth", type=float, default=145.0)
    parser.add_argument("--contact-camera-elevation", type=float, default=-18.0)
    parser.add_argument(
        "--visualize-loop-trajectory",
        dest="visualize_loop_trajectory",
        action="store_true",
    )
    parser.add_argument(
        "--no-visualize-loop-trajectory",
        dest="visualize_loop_trajectory",
        action="store_false",
    )
    parser.set_defaults(visualize_loop_trajectory=True)
    parser.add_argument(
        "--save-trajectory-videos",
        dest="save_trajectory_videos",
        action="store_true",
    )
    parser.add_argument(
        "--no-save-trajectory-videos",
        dest="save_trajectory_videos",
        action="store_false",
    )
    parser.set_defaults(save_trajectory_videos=True)
    parser.add_argument("--require-video-renderer", action="store_true")
    parser.add_argument(
        "--video-output-dir",
        type=str,
        default=str(REPO_ROOT / "outputs"),
    )
    parser.add_argument("--video-prefix", type=str, default="close_toaster_oven_door")
    parser.add_argument("--video-camera", type=str, default="free")
    parser.add_argument("--video-camera-distance", type=float, default=1.1)
    parser.add_argument("--video-camera-azimuth", type=float, default=145.0)
    parser.add_argument("--video-camera-elevation", type=float, default=-18.0)
    parser.add_argument(
        "--video-camera-lookat",
        type=str,
        default="panel",
        choices=("panel", "contact", "ee"),
    )
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=720)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-marker-size", type=float, default=0.03)
    parser.add_argument(
        "--video-encoder",
        type=str,
        default="ffmpeg",
        choices=("ffmpeg", "opencv"),
    )
    parser.add_argument("--ffmpeg-path", type=str, default="")
    parser.add_argument("--video-codec", type=str, default="libx264")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", type=str, default="slow")
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/robocasa_close_toaster_oven_door_contact_curobo.npz",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args._contact_rng = np.random.default_rng(int(args.seed))
    env = create_close_toaster_oven_door_env(args)
    try:
        panel = get_panel_frame(env, args)
        _initialize_scene_processing(env, panel, args)
        _refresh_scene_visibility(env, args)
        task_spec = _make_door_task_spec(env, panel, args)
        (
            candidates,
            selected,
            feasible_points,
            feasible_costs,
            feasible_tree,
        ) = evaluate_contacts(env, panel, task_spec, args)
        robot_state = D.get_robot_arm_state(env)
        args._last_dream_best_cost = float("nan")
        dream_solution, reports = solve_contact_pose_with_dream(
            env,
            panel,
            candidates,
            task_spec,
            robot_state,
            args,
        )
        if dream_solution is not None:
            selected_index = int(dream_solution.drawer_candidate_index)
            selected = candidates[selected_index]
            target_gripper_poses = dream_solution.target_gripper_poses
            if not bool(dream_solution.collision_free):
                warnings.warn(
                    "Dream final result exceeds the feasibility threshold. "
                    "It will still be sent to cuRobo and executed for inspection.",
                    RuntimeWarning,
                )
        else:
            selected_index = next(
                index
                for index, candidate in enumerate(candidates)
                if candidate is selected
            )
            target_gripper_poses = []
            warnings.warn(
                "Dream found no feasible EE trajectory for closing the "
                "toaster-oven door. cuRobo planning is skipped. Reasons: "
                f"{O._report_reason_counts(reports)}",
                RuntimeWarning,
            )
        stage = CloseDoorStage(
            start_door_q=panel.current_q,
            target_door_q=panel.target_q,
            selected_contact_index=selected_index,
            selected_contact_world=np.asarray(selected.world_point, dtype=np.float64),
            selected_contact_local=np.asarray(selected.local_point, dtype=np.float64),
            selected_contact_cost=float(selected.cost),
            dream_solution=dream_solution,
            candidates=candidates,
            dream_reports=reports,
            feasible_points_world=feasible_points,
            feasible_costs=feasible_costs,
            feasible_tree=feasible_tree,
            task_current_position_local=task_spec.current_position_local,
            task_target_position_local=task_spec.target_position_local,
            dream_best_cost=float(args._last_dream_best_cost),
        )
        target_hand_poses = [
            (name, *D.gripper_pose_to_curobo_hand_pose(pos, rot, robot_state))
            for name, pos, rot in target_gripper_poses
        ]
        q_traj = None
        segments = []
        if target_hand_poses:
            try:
                D.preload_curobo_runtime()
                q_traj, segments = D.plan_with_curobo(
                    robot_state, target_hand_poses, args
                )
                if q_traj is None or not np.asarray(q_traj).size:
                    q_traj, segments = D.plan_with_curobo(
                        robot_state,
                        _final_curobo_target(target_hand_poses),
                        args,
                    )
            except Exception as exc:
                warnings.warn(
                    "cuRobo could not plan the complete Dream EE sequence; "
                    f"retrying the final optimized EE pose only: {exc}",
                    RuntimeWarning,
                )
                try:
                    q_traj, segments = D.plan_with_curobo(
                        robot_state,
                        _final_curobo_target(target_hand_poses),
                        args,
                    )
                except Exception as final_exc:
                    warnings.warn(
                        "cuRobo could not move the Franka to Dream's final "
                        f"optimized EE pose: {final_exc}",
                        RuntimeWarning,
                    )
                    q_traj = None
                    segments = []
        metadata = save_outputs(
            args.output,
            env,
            panel,
            stage,
            target_hand_poses,
            q_traj,
            segments,
        )
        print(json.dumps(metadata, indent=2))
        print(f"Saved: {args.output}")
        save_trajectory_video(
            env,
            panel,
            stage,
            robot_state,
            q_traj,
            segments,
            args,
        )
        visualize_contact(
            env,
            panel,
            stage,
            robot_state,
            q_traj,
            segments,
            args,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
