import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions_gcc11")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
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
                "import contextlib, os\n"
                "_sink = open(os.devnull, 'w')\n"
                "with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):\n"
                "    from robocasa.demos.demo_open_drawer_contact_curobo import main\n"
                "main()"
            ),
            *sys.argv[1:],
        ],
    )

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from robocasa.demos.scene_process import sample_inner_surface_candidates
from robocasa.demos.object_cso import (
    point_sdf_for_spheres_numpy as _point_sdf_for_spheres_numpy,
    sphere_centers_world_from_pose as _sphere_centers_world_from_pose,
)
from robocasa.demos.franka_collision_model import (
    load_curobo_ee_collision_spheres as _load_curobo_ee_collision_spheres,
    solve_collision_sphere_contact_with_q_mpc as _solve_collision_sphere_contact_with_q_mpc,
)
from robocasa.demos.open_drawer.collision import (
    check_arm_q_collision_for_surface as _check_arm_q_collision_for_surface_base,
    robot_contact_geom_sets_for_surface as _robot_contact_geom_sets_for_surface,
)
from robocasa.demos.open_drawer.math import (
    _distance_summary,
    _nearest_point_distances,
    _rotation_matrix_to_quat_wxyz,
    _score_ee_pose_contact_targets,
    _score_ee_pose_contacts,
)
from robocasa.demos.open_drawer.scene_process import (
    _project_contact_samples_to_actual_geoms,
    _surface_contact_sample_count,
)
from robocasa.demos.open_drawer.scene import (
    initialize_scene_processing as _initialize_scene_processing,
    make_handle_inner_surface as _make_handle_inner_surface,
    refresh_scene_visibility as _refresh_scene_visibility,
)
from robocasa.demos.open_drawer.utils import (
    _configure_cuda_memory_limit,
    _empty_cuda_caches,
    load_yaml_config as _load_yaml_config,
)
from robocasa.demos.open_drawer.viewer import (
    _play_open_trajectory_viewer,
    _save_open_outputs,
    _validate_curobo_reaches_gripper_targets,
    _validate_joint_space_trajectory_collisions,
    create_open_drawer_env,
)
from robocasa.demos.dream import solve_plan_once_mppi_for_q_configs, solve_stages

_QUIET_IMPORT_SINK = open(os.devnull, "w")
with contextlib.redirect_stdout(_QUIET_IMPORT_SINK), contextlib.redirect_stderr(
    _QUIET_IMPORT_SINK
):
    import robosuite
    import robocasa  # noqa: F401
    import robocasa.utils.lerobot_utils as LU
    import robocasa.demos.demo_close_drawer_contact_curobo as close_demo
    import robocasa.demos.mink_q as mink_q
    from robocasa.demos.demo_tasks import get_ds_path_any_split
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

_ENABLE_STAGE_BANNERS = False


@dataclass
class OpenDrawerStage:
    name: str
    surface_name: str
    start_drawer_q: float
    pull_distance: float
    selected_contact_index: int
    selected_contact_world: np.ndarray
    selected_contact_local: np.ndarray
    selected_contact_cost: float
    mink_solution: object
    candidates: list
    mink_reports: list


@dataclass
class DemonstrationContactSeed:
    dataset: str
    episode_name: str
    frame_index: int
    arm_q: np.ndarray
    ee_position_object: np.ndarray
    ee_rotation_object: np.ndarray
    contact_position_object: np.ndarray
    contact_offset_ee: np.ndarray
    ee_contact_geom_name: str
    projected_ee_position_world: np.ndarray
    projected_ee_rotation_world: np.ndarray
    projected_contact_world: np.ndarray
    source_position_roundtrip_error: float
    source_rotation_roundtrip_error: float
    projected_position_roundtrip_error: float
    projected_rotation_roundtrip_error: float


def _stage_result(name, elapsed_seconds, success_count):
    print(
        f"[{name}] time={float(elapsed_seconds):.6f}s "
        f"successful_trajectories={int(success_count)}",
        flush=True,
    )


def _count_result(name, elapsed_seconds, count_name, count):
    print(
        f"[{name}] time={float(elapsed_seconds):.6f}s " f"{count_name}={int(count)}",
        flush=True,
    )


@contextlib.contextmanager
def _suppress_solver_output():
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _target_pull_distance(env, args):
    if args.pull_distance is not None:
        return max(float(args.pull_distance), 1e-4)
    return max(
        float(env.drawer.size[1]) * float(args.target_open_fraction) * 0.55, 1e-4
    )


def _create_demonstration_env(dataset):
    env_meta = LU.get_env_metadata(Path(dataset))
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = False
    env_kwargs["use_camera_obs"] = False
    env_kwargs["renderer"] = "mjviewer"
    return robosuite.make(**env_kwargs)


def _load_demonstration_episode(args):
    dataset = (
        Path(args.demonstration_dataset)
        if args.demonstration_dataset
        else get_ds_path_any_split(args.demonstration_task, source="human")
    )
    if dataset is None:
        raise RuntimeError(
            f"No registered human dataset found for {args.demonstration_task}."
        )
    dataset = Path(dataset)
    if not dataset.exists():
        raise FileNotFoundError(f"Demonstration dataset does not exist: {dataset}")
    episodes = LU.get_episodes(dataset)
    index = int(args.demonstration_index)
    if index < 0 or index >= len(episodes):
        raise IndexError(
            f"demonstration index {index} is outside [0, {len(episodes) - 1}]"
        )
    states = LU.get_episode_states(dataset, index)
    initial_state = {
        "states": states[0],
        "model": LU.get_episode_model_xml(dataset, index),
        "ep_meta": json.dumps(LU.get_episode_meta(dataset, index)),
    }
    return dataset, episodes[index].stem, initial_state, states


def _site_pose_world(env, site_name):
    model = env.sim.model
    data = env.sim.data
    if site_name not in model._site_name2id:
        raise RuntimeError(
            f"EE site {site_name!r} is not present in demonstration model"
        )
    site_id = model.site_name2id(site_name)
    return (
        np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
        np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def _first_effective_demonstration_contact(source_env, states, args):
    drawer_q = np.empty(len(states), dtype=np.float64)
    for frame_index, state in enumerate(states):
        reset_to(source_env, {"states": state})
        drawer_q[frame_index] = close_demo._drawer_joint_value(source_env)

    lookahead = max(int(args.demonstration_motion_lookahead), 1)
    motion_threshold = max(float(args.demonstration_drawer_motion_threshold), 0.0)
    for frame_index, state in enumerate(states):
        future_index = min(frame_index + lookahead, len(states) - 1)
        if (
            abs(float(drawer_q[future_index] - drawer_q[frame_index]))
            < motion_threshold
        ):
            continue
        reset_to(source_env, {"states": state})
        panel = close_demo.get_panel_frame(source_env)
        surface = _make_handle_inner_surface(source_env, panel)
        _, ee_geoms, drawer_geoms = _robot_contact_geom_sets_for_surface(
            source_env,
            surface,
        )
        contact_world = None
        ee_contact_geom_id = -1
        for contact_index in range(int(source_env.sim.data.ncon)):
            contact = source_env.sim.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in ee_geoms and geom2 in drawer_geoms:
                ee_contact_geom_id = geom1
            elif geom2 in ee_geoms and geom1 in drawer_geoms:
                ee_contact_geom_id = geom2
            else:
                continue
            contact_world = np.asarray(contact.pos, dtype=np.float64).copy()
            break
        if contact_world is None:
            continue

        ee_position_world, ee_rotation_world = _site_pose_world(
            source_env,
            args.mink_contact_frame,
        )
        arm_joint_names = tuple(source_env.robots[0].robot_model.joints[:7])
        arm_q = np.asarray(
            [
                source_env.sim.data.qpos[
                    source_env.sim.model.get_joint_qpos_addr(joint_name)
                ]
                for joint_name in arm_joint_names
            ],
            dtype=np.float64,
        )
        geom_id_to_name = {
            int(geom_id): name
            for name, geom_id in source_env.sim.model._geom_name2id.items()
        }
        return {
            "frame_index": frame_index,
            "arm_q": arm_q,
            "ee_position_object": panel.rotation_world.T
            @ (ee_position_world - panel.center_world),
            "ee_rotation_object": panel.rotation_world.T @ ee_rotation_world,
            "contact_position_object": panel.rotation_world.T
            @ (contact_world - panel.center_world),
            "contact_offset_ee": ee_rotation_world.T
            @ (contact_world - ee_position_world),
            "ee_contact_geom_name": geom_id_to_name.get(ee_contact_geom_id, ""),
            "source_position_roundtrip_error": float(
                np.linalg.norm(
                    panel.center_world
                    + panel.rotation_world
                    @ (
                        panel.rotation_world.T
                        @ (ee_position_world - panel.center_world)
                    )
                    - ee_position_world
                )
            ),
            "source_rotation_roundtrip_error": float(
                np.linalg.norm(
                    panel.rotation_world @ (panel.rotation_world.T @ ee_rotation_world)
                    - ee_rotation_world
                )
            ),
        }
    raise RuntimeError(
        "No demonstration frame contains both EE-drawer contact and subsequent "
        "drawer displacement."
    )


def _load_and_project_demonstration_seed(env, args):
    dataset, episode_name, initial_state, states = _load_demonstration_episode(args)
    source_env = _create_demonstration_env(dataset)
    try:
        reset_to(source_env, initial_state)
        extracted = _first_effective_demonstration_contact(source_env, states, args)
    finally:
        source_env.close()
    panel = close_demo.get_panel_frame(env)
    projected_ee_position_world = (
        panel.center_world + panel.rotation_world @ extracted["ee_position_object"]
    )
    projected_ee_rotation_world = panel.rotation_world @ extracted["ee_rotation_object"]
    projected_contact_world = (
        panel.center_world + panel.rotation_world @ extracted["contact_position_object"]
    )
    projected_position_roundtrip_error = float(
        np.linalg.norm(
            panel.rotation_world.T @ (projected_ee_position_world - panel.center_world)
            - extracted["ee_position_object"]
        )
    )
    projected_rotation_roundtrip_error = float(
        np.linalg.norm(
            panel.rotation_world.T @ projected_ee_rotation_world
            - extracted["ee_rotation_object"]
        )
    )
    maximum_error = max(
        float(extracted["source_position_roundtrip_error"]),
        float(extracted["source_rotation_roundtrip_error"]),
        projected_position_roundtrip_error,
        projected_rotation_roundtrip_error,
    )
    if maximum_error > float(args.demonstration_transform_tolerance):
        raise RuntimeError(
            f"Demonstration object-frame projection failed round-trip validation: "
            f"error={maximum_error:.3e}"
        )
    return DemonstrationContactSeed(
        dataset=str(dataset),
        episode_name=str(episode_name),
        frame_index=int(extracted["frame_index"]),
        arm_q=np.asarray(extracted["arm_q"], dtype=np.float64),
        ee_position_object=np.asarray(
            extracted["ee_position_object"],
            dtype=np.float64,
        ),
        ee_rotation_object=np.asarray(
            extracted["ee_rotation_object"],
            dtype=np.float64,
        ),
        contact_position_object=np.asarray(
            extracted["contact_position_object"],
            dtype=np.float64,
        ),
        contact_offset_ee=np.asarray(
            extracted["contact_offset_ee"],
            dtype=np.float64,
        ),
        ee_contact_geom_name=str(extracted["ee_contact_geom_name"]),
        projected_ee_position_world=np.asarray(
            projected_ee_position_world,
            dtype=np.float64,
        ),
        projected_ee_rotation_world=np.asarray(
            projected_ee_rotation_world,
            dtype=np.float64,
        ),
        projected_contact_world=np.asarray(
            projected_contact_world,
            dtype=np.float64,
        ),
        source_position_roundtrip_error=float(
            extracted["source_position_roundtrip_error"]
        ),
        source_rotation_roundtrip_error=float(
            extracted["source_rotation_roundtrip_error"]
        ),
        projected_position_roundtrip_error=projected_position_roundtrip_error,
        projected_rotation_roundtrip_error=projected_rotation_roundtrip_error,
    )


def _build_surface_contact_optimizer(surface, args):
    LambdaContactControlOptimizer = close_demo._import_contact_optimizer()
    mesh = trimesh.creation.box(
        extents=2.0 * np.asarray(surface.half_size, dtype=np.float64)
    )
    mesh_path = tempfile.NamedTemporaryFile(
        prefix=f"robocasa_{surface.name}_", suffix=".stl", delete=False
    ).name
    mesh.export(mesh_path)
    sample_count = _surface_contact_sample_count(surface, args)
    optimizer = LambdaContactControlOptimizer(
        mesh_path,
        obj_mass=args.contact_obj_mass,
        arm_friction=args.contact_friction,
        contact_stiffness=args.contact_stiffness,
        time_step=args.contact_dt,
        max_contacts=1,
        sample_num=max(8, int(sample_count)),
        pos_coef=args.contact_pos_coef,
        ori_coef=args.contact_ori_coef,
        nlp_solver=args.contact_solver,
    )
    return optimizer, mesh_path


def _stage_banner(message):
    """Print verbose stage banners when explicitly enabled."""
    if not _ENABLE_STAGE_BANNERS:
        return
    stream = sys.__stdout__
    try:
        stream.write(message + "\n")
        stream.flush()
    except Exception:
        print(message, flush=True)


def _build_obstacle_point_cloud(env, args):
    """Sample an obstacle point cloud from every collision geom that is NOT
    the drawer (contact target) and NOT the robot/gripper.

    The existing `_scene_point_cloud` is built from `env.drawer.get_xml()` only
    (drawer-local), and the MPPI nonpenetration cost samples points from
    geoms with the `env.drawer.*` prefix only. Neither sees the cabinet body
    or the counter top above the drawer, so EE poses that penetrate those
    surfaces go undetected. This cloud closes that gap for stage 1 candidate
    pose validation.

    Cached on `args`.
    """
    cached_xyz = getattr(args, "_obstacle_point_cloud_xyz", None)
    cached_tree = getattr(args, "_obstacle_point_cloud_tree", None)
    if cached_xyz is not None and cached_tree is not None:
        return cached_xyz, cached_tree
    from robocasa.demos.scene_process import (
        _geom_mesh_in_body,
        _uniform_surface_samples,
    )

    raw_model = (
        env.sim.model._model if hasattr(env.sim.model, "_model") else env.sim.model
    )
    raw_data = env.sim.data._data if hasattr(env.sim.data, "_data") else env.sim.data
    drawer_prefix = f"{env.drawer.name}_"
    points_per_geom = int(getattr(args, "obstacle_cloud_points_per_geom", 200))
    geom_name_by_id = {
        int(gid): str(name) for name, gid in env.sim.model._geom_name2id.items()
    }
    rng = np.random.default_rng(int(args.seed) + 7919)
    all_pts = []
    for gid in range(int(raw_model.ngeom)):
        name = geom_name_by_id.get(gid, "")
        if not name:
            continue
        if name.startswith(drawer_prefix):
            continue
        if name.startswith("robot0_") or name.startswith("gripper0_"):
            continue
        if int(raw_model.geom_group[gid]) != 0:
            continue
        if float(raw_model.geom_rgba[gid, 3]) <= 1e-5:
            continue
        mesh_body = _geom_mesh_in_body(raw_model, int(gid))
        if mesh_body is None or not mesh_body.faces.size:
            continue
        body_id = int(raw_model.geom_bodyid[int(gid)])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(raw_data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.asarray(raw_data.xpos[body_id], dtype=np.float64).reshape(3)
        mesh_w = mesh_body.copy()
        mesh_w.apply_transform(T)
        try:
            pts, _ = _uniform_surface_samples(mesh_w, points_per_geom, rng)
        except Exception:
            continue
        all_pts.append(np.asarray(pts, dtype=np.float64))
    if not all_pts:
        cloud = np.zeros((0, 3), dtype=np.float64)
        tree = None
    else:
        cloud = np.concatenate(all_pts, axis=0)
        tree = cKDTree(cloud)
    args._obstacle_point_cloud_xyz = cloud
    args._obstacle_point_cloud_tree = tree
    return cloud, tree


def _candidate_pose_penetrates_obstacles(pose_wxyz, sphere_model, tree, tolerance):
    """Return (penetrates, worst_signed_margin) for an EE pose against an
    obstacle KDTree. `worst_signed_margin = min(dist_to_nearest - sphere_radius)`;
    negative means the sphere is inside the nearest obstacle point."""
    if tree is None or sphere_model is None:
        return False, float("nan")
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(-1)
    if pose.size < 7 or not np.all(np.isfinite(pose[:7])):
        return False, float("nan")
    pos = pose[:3]
    quat = pose[3:7]
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return False, float("nan")
    R = close_demo._matrix_from_quat_wxyz(quat / norm)
    centers_ee = np.asarray(sphere_model.centers_ee, dtype=np.float64)
    centers_world = centers_ee @ R.T + pos
    distances, _ = tree.query(centers_world)
    radii = np.asarray(sphere_model.radii, dtype=np.float64)
    margins = distances - radii
    worst = float(np.min(margins))
    penetrates = bool(worst < -float(tolerance))
    return penetrates, worst


def evaluate_open_contacts(env, surface, pull_distance, args):
    stage1_t0 = time.time()
    _stage_banner(
        f"=== Stage 1: sampling-MPC initial contact search "
        f"(surface={surface.name}) ==="
    )
    sample_count = _surface_contact_sample_count(surface, args)
    (
        local_points,
        world_points,
        force_normals,
        approach_dirs,
        face_names,
    ) = sample_inner_surface_candidates(
        surface,
        sample_count=sample_count,
        margin=args.contact_margin,
    )
    (
        local_points,
        world_points,
        approach_dirs,
        projection_distances,
    ) = _project_contact_samples_to_actual_geoms(
        env,
        surface,
        local_points,
        world_points,
        approach_dirs,
    )
    optimizer, mesh_path = _build_surface_contact_optimizer(surface, args)
    try:
        current_x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        x_d = np.array(
            [0.0, -float(pull_distance), 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        tau_o = np.zeros(6, dtype=np.float64)
        tangent_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tangent_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # ---- Batched GPU QP solve (vmap'd JAXopt OSQP) over ALL candidates ----
        # Previously this looped over candidates calling `_solve_once` once each,
        # which serialized the JAX dispatch (one host->device launch per point and
        # no vmap parallelism). We now build (N,3) tangent/normal/point batches and
        # call `_solve_batch` exactly once so JAX runs the chunked vmapped solver on
        # the configured device. Semantics are identical to the previous per-sample
        # call (same x_d, current_x, tau_o, curr_ori_coef, lam_upper_bound) — only
        # the dispatch shape changes.
        local_points_arr = np.asarray(local_points, dtype=np.float64).reshape(-1, 3)
        n_candidates = local_points_arr.shape[0]
        if n_candidates == 0:
            return [], None
        n_arm_batch = np.asarray(force_normals, dtype=np.float64).reshape(-1, 3)
        t1_batch = (
            np.broadcast_to(tangent_x, (n_candidates, 3)).astype(np.float64).copy()
        )
        t2_batch = (
            np.broadcast_to(tangent_z, (n_candidates, 3)).astype(np.float64).copy()
        )
        lam_batch, x_plus_batch, cost_batch, status_batch = optimizer._solve_batch(
            x_d=x_d,
            current_x=current_x,
            tau_o=tau_o,
            n_arm=n_arm_batch,
            t1=t1_batch,
            t2=t2_batch,
            p_arm=local_points_arr,
            curr_ori_coef=1.0,
            lam_upper_bound=args.contact_lam_upper_bound,
        )

        # ---- EE-pose vs scene-obstacle penetration check (closes a gap left
        # by the MPPI nonpenetration cost, which only samples drawer geoms and
        # therefore can't see the cabinet body / counter top above the drawer).
        scene_check_enabled = bool(getattr(args, "contact_stage1_scene_check", True))
        scene_tol = float(
            getattr(args, "contact_stage1_scene_penetration_tolerance", 0.005)
        )
        scene_sphere_model = None
        scene_obstacle_tree = None
        if scene_check_enabled:
            try:
                _scene_robot_state = close_demo.get_robot_arm_state(env)
                scene_sphere_model = _load_curobo_ee_collision_spheres(
                    env, _scene_robot_state, args
                )
                _, scene_obstacle_tree = _build_obstacle_point_cloud(env, args)
            except Exception as exc:
                scene_sphere_model = None
                scene_obstacle_tree = None
                _stage_banner(
                    f"stage 1: scene-penetration check disabled "
                    f"(failed to build EE spheres / obstacle cloud: {exc})"
                )
        stage1_scene_penetration_count = 0

        candidates = []
        for idx, (
            p_local,
            p_world,
            force_normal,
            approach_dir,
            face_name,
            projection_distance,
        ) in enumerate(
            zip(
                local_points,
                world_points,
                force_normals,
                approach_dirs,
                face_names,
                projection_distances,
            )
        ):
            lam = lam_batch[idx]
            x_plus = x_plus_batch[idx]
            cost = float(cost_batch[idx])
            status = str(status_batch[idx])
            # Friction-cone test for pull feasibility.
            # `approach_dir` is the outward face normal in world frame; the EE
            # pushes along -approach_dir. To pull along surface.pull_world, the
            # angle between pull_world and -approach_dir must be inside the
            # friction cone (cos >= 1/sqrt(1+mu^2)).
            mu = max(float(args.contact_friction), 1e-6)
            pull_world = np.asarray(surface.pull_world, dtype=np.float64)
            pull_world = pull_world / max(float(np.linalg.norm(pull_world)), 1e-9)
            approach_unit = np.asarray(approach_dir, dtype=np.float64)
            approach_unit = approach_unit / max(
                float(np.linalg.norm(approach_unit)), 1e-9
            )
            pull_along_inward_normal = -float(np.dot(pull_world, approach_unit))
            friction_cone_cos_threshold = 1.0 / float(np.sqrt(1.0 + mu * mu))
            friction_cone_ok = bool(
                pull_along_inward_normal >= friction_cone_cos_threshold
            )
            feasible = bool(
                np.isfinite(cost)
                and cost <= args.contact_cost_threshold
                and float(projection_distance)
                <= float(args.contact_surface_projection_max_distance)
                and (friction_cone_ok or not bool(args.require_friction_cone_pull))
            )
            scene_penetrates = False
            scene_margin = float("nan")
            if (
                feasible
                and scene_sphere_model is not None
                and scene_obstacle_tree is not None
            ):
                scene_penetrates, scene_margin = _candidate_pose_penetrates_obstacles(
                    x_plus,
                    scene_sphere_model,
                    scene_obstacle_tree,
                    scene_tol,
                )
                if scene_penetrates:
                    feasible = False
                    stage1_scene_penetration_count += 1
            candidate = close_demo.ContactCandidate(
                local_point=np.asarray(p_local, dtype=np.float64),
                world_point=np.asarray(p_world, dtype=np.float64),
                cost=cost,
                lam=np.asarray(lam, dtype=np.float64),
                resulting_pose=np.asarray(x_plus, dtype=np.float64),
                solver_status=str(status),
                feasible=feasible,
            )
            candidate.force_normal_local = np.asarray(force_normal, dtype=np.float64)
            candidate.approach_world = np.asarray(approach_dir, dtype=np.float64)
            candidate.contact_face = str(face_name)
            candidate.surface_projection_distance = float(projection_distance)
            candidate.friction_cone_ok = bool(friction_cone_ok)
            candidate.friction_cone_pull_ratio = float(pull_along_inward_normal)
            candidate.scene_penetration = bool(scene_penetrates)
            candidate.scene_min_margin = float(scene_margin)
            _annotate_candidate_visibility(candidate, surface, args)
            candidates.append(candidate)
        stage1_elapsed = time.time() - stage1_t0
        stage1_feasible_count = sum(1 for c in candidates if c.feasible)
        contact_stage_stats = {
            "stage1_success_count": int(stage1_feasible_count),
            "stage1_elapsed": float(stage1_elapsed),
        }
        setattr(args, "_last_contact_stage_stats", contact_stage_stats)
        _stage_banner(
            f"=== Stage 1 done: produced {len(candidates)} candidates "
            f"({stage1_feasible_count} feasible, "
            f"{stage1_scene_penetration_count} rejected for scene penetration), "
            f"elapsed {stage1_elapsed:.3f}s ==="
        )

        if args.require_visible_contact:
            candidates = [candidate for candidate in candidates if candidate.visible]
            if not candidates:
                raise RuntimeError(
                    f"No camera-visible {surface.name} contact candidates remain. "
                    "Increase --scene-points-per-link or "
                    "--scene-visible-point-max-distance, change the visibility camera, "
                    "or disable --require-visible-contact."
                )
        candidates.sort(key=lambda c: c.cost)
        if surface.name == "handle" and args.handle_mink_candidate_faces:
            preferred_faces = [
                face.strip()
                for face in args.handle_mink_candidate_faces.split(",")
                if face.strip()
            ]
            preferred_rank = {face: rank for rank, face in enumerate(preferred_faces)}
            candidates.sort(
                key=lambda c: (
                    preferred_rank.get(
                        str(getattr(c, "contact_face", "")), len(preferred_rank)
                    ),
                    c.cost,
                )
            )
        feasible_candidates = [c for c in candidates if c.feasible]
        if not feasible_candidates and args.require_feasible_contact:
            best = candidates[0]
            raise RuntimeError(
                f"No feasible {surface.name} contact point found. "
                f"Best cost={best.cost:.6f}, threshold={args.contact_cost_threshold:.6f}, "
                f"status={best.solver_status}."
            )
        return candidates, (
            feasible_candidates[0] if feasible_candidates else candidates[0]
        )
    finally:
        try:
            os.unlink(mesh_path)
        except OSError:
            pass


def _annotate_candidate_visibility(candidate, surface, args):
    point_cloud = getattr(args, "_scene_point_cloud", None)
    if point_cloud is None:
        candidate.visible = True
        candidate.scene_link_name = ""
        candidate.scene_point_index = -1
        candidate.scene_point_distance = float("nan")
        return
    nearest = point_cloud.nearest(
        candidate.world_point,
        getattr(args, "_scene_runtime_data"),
        allowed_geom_names=surface.allowed_geom_names,
    )
    if nearest is None:
        candidate.visible = False
        candidate.scene_link_name = ""
        candidate.scene_point_index = -1
        candidate.scene_point_distance = float("inf")
        return
    link, point_index, distance = nearest
    candidate.scene_link_name = link.name
    candidate.scene_point_index = int(point_index)
    candidate.scene_point_distance = float(distance)
    candidate.visible = bool(
        link.mask[point_index]
        and distance <= float(args.scene_visible_point_max_distance)
    )


def _demonstration_seed_pose_wxyz(demonstration_seed):
    return np.concatenate(
        [
            np.asarray(
                demonstration_seed.projected_ee_position_world,
                dtype=np.float64,
            ).reshape(3),
            _rotation_matrix_to_quat_wxyz(
                demonstration_seed.projected_ee_rotation_world
            ),
        ]
    )


def _linear_contact_target_points(anchor_contact_world, pull_delta_world, count):
    count = max(int(count), 1)
    anchor = np.asarray(anchor_contact_world, dtype=np.float64).reshape(3)
    pull_delta = np.asarray(pull_delta_world, dtype=np.float64).reshape(3)
    alphas = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return anchor.reshape(1, 3) + alphas[:, None] * pull_delta.reshape(1, 3)


def _solve_mink_precontact_seed(
    env,
    surface,
    candidates,
    demonstration_seed,
    robot_state,
    args,
):
    feasible_candidates = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if bool(candidate.feasible)
    ]
    if not feasible_candidates:
        raise RuntimeError(
            "mink precontact seed requires at least one feasible contact candidate"
        )
    projected_contact = np.asarray(
        demonstration_seed.projected_contact_world, dtype=np.float64
    ).reshape(3)
    distances = np.asarray(
        [
            np.linalg.norm(
                np.asarray(candidate.world_point, dtype=np.float64) - projected_contact
            )
            for _, candidate in feasible_candidates
        ],
        dtype=np.float64,
    )
    selected_local = int(np.argmin(distances))
    selected_candidate_index, selected_candidate = feasible_candidates[selected_local]
    projected_ee_position = np.asarray(
        demonstration_seed.projected_ee_position_world, dtype=np.float64
    ).reshape(3)
    demo_contact_offset_world = (
        np.asarray(demonstration_seed.projected_contact_world, dtype=np.float64)
        - projected_ee_position
    )
    retreat = -close_demo._normalize(
        demo_contact_offset_world,
        fallback=np.asarray(
            getattr(selected_candidate, "approach_world", surface.approach_world),
            dtype=np.float64,
        ),
    )
    precontact_rotation = np.asarray(
        demonstration_seed.projected_ee_rotation_world, dtype=np.float64
    ).reshape(3, 3)
    panel = close_demo.get_panel_frame(env)
    distance_multipliers = tuple(
        float(value)
        for value in getattr(
            args,
            "mink_precontact_distance_multipliers",
            (1.0, 1.5, 2.0, 3.0),
        )
    )
    if not distance_multipliers:
        distance_multipliers = (1.0,)
    attempts = []
    solution = None
    for multiplier in distance_multipliers:
        precontact_position = projected_ee_position + retreat * (
            float(args.precontact_distance) * float(multiplier)
        )
        candidate_solution = mink_q.solve_precontact_pose(
            env,
            panel,
            robot_state,
            precontact_position,
            precontact_rotation,
            np.asarray(demonstration_seed.arm_q, dtype=np.float64),
            args,
            collision_checker=lambda q_arm: _check_arm_q_collision_for_surface_base(
                env,
                surface,
                robot_state["robocasa_joint_names"],
                q_arm,
                close_demo._drawer_joint_value(env),
                set_arm_q=close_demo._set_env_arm_q,
                set_drawer_joint_value=close_demo._set_drawer_joint_value,
                allowed_ee_geom_name=None,
                penetration_tolerance=float(args.mink_collision_penetration_tolerance),
                collision_scope="arm",
            ),
        )
        attempts.append(
            (
                float(multiplier),
                float(candidate_solution.position_error),
                bool(candidate_solution.collision_free),
                str(candidate_solution.collision_reason),
            )
        )
        if (
            candidate_solution.position_error <= float(args.mink_position_tolerance)
            and candidate_solution.collision_free
        ):
            solution = candidate_solution
            break
        if solution is None or (
            candidate_solution.collision_free and not solution.collision_free
        ):
            solution = candidate_solution
    if bool(getattr(args, "require_mink_precontact", True)):
        if solution.position_error > float(args.mink_position_tolerance):
            raise RuntimeError(
                "mink precontact seed did not reach target: "
                f"position_error={solution.position_error:.6f}, "
                f"tolerance={float(args.mink_position_tolerance):.6f}, "
                f"attempts={attempts}"
            )
        if not solution.collision_free:
            raise RuntimeError(
                "mink precontact seed is not collision-free: "
                f"{solution.collision_reason}; attempts={attempts}"
            )
    return solution, int(selected_candidate_index)


def _site_pose_for_arm_q(env, robot_state, q_arm, frame_name, drawer_q=None):
    model = env.sim.model
    data = env.sim.data
    if frame_name not in model._site_name2id:
        raise RuntimeError(f"Cannot read site pose: site '{frame_name}' not found")
    qpos_saved = data.qpos.copy()
    qvel_saved = data.qvel.copy()
    try:
        close_demo._set_env_arm_q(
            env,
            robot_state["robocasa_joint_names"],
            np.asarray(q_arm, dtype=np.float64).reshape(7),
        )
        if drawer_q is not None:
            close_demo._set_drawer_joint_value(env, float(drawer_q))
        env.sim.forward()
        site_id = model.site_name2id(frame_name)
        pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
        rot = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
        return pos, rot
    finally:
        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        env.sim.forward()


def _solve_stage(env, surface, pull_distance, args, stage_name):
    stage_solve_started = time.perf_counter()
    _refresh_scene_visibility(env, args)
    with _suppress_solver_output():
        demonstration_seed = _load_and_project_demonstration_seed(env, args)
    args._demonstration_seed = demonstration_seed
    robot_state = close_demo.get_robot_arm_state(env)
    base_alignment = mink_q.align_base_to_demo_ee_pose(
        env,
        demonstration_seed,
        robot_state,
        args,
    )
    robot_state = close_demo.get_robot_arm_state(env)
    qp_started = time.perf_counter()
    with _suppress_solver_output():
        candidates, selected = evaluate_open_contacts(
            env,
            surface,
            pull_distance,
            args,
        )
    _ = time.perf_counter() - qp_started
    demonstration_pose = _demonstration_seed_pose_wxyz(demonstration_seed)
    feasible_points = np.asarray(
        [candidate.world_point for candidate in candidates if bool(candidate.feasible)],
        dtype=np.float64,
    )
    projected_seed_contact_world = (
        demonstration_seed.projected_ee_position_world
        + demonstration_seed.projected_ee_rotation_world
        @ demonstration_seed.contact_offset_ee
    )
    (
        projected_seed_contact_distance,
        projected_seed_contact_index,
    ) = _nearest_point_distances(
        projected_seed_contact_world.reshape(1, 3),
        feasible_points,
    )
    dream_result = None
    dream_sequence = np.repeat(demonstration_pose[None], 2, axis=0)
    dream_diagnostics = {}
    selected_mode = ""
    selected_device = ""
    selected_contact_candidate_indices = np.zeros(0, dtype=np.int64)
    sphere_model = None
    representative_points = None
    selected_sphere_index = -1
    sphere_sdf = np.zeros(0, dtype=np.float64)
    contact_offset_ee = np.asarray(
        demonstration_seed.contact_offset_ee, dtype=np.float64
    )
    contact_position_error = float("nan")
    q_config_solution = None
    q_waypoints = np.zeros((0, 7), dtype=np.float64)
    frame_name = str(getattr(args, "mink_contact_frame", "gripper0_right_grip_site"))
    precontact_solution, precontact_seed_candidate_index = _solve_mink_precontact_seed(
        env,
        surface,
        candidates,
        demonstration_seed,
        robot_state,
        args,
    )
    if getattr(args, "use_q_config_mpc", False):
        mpc_started = time.perf_counter()
        try:
            with _suppress_solver_output():
                q_config_solution = _solve_collision_sphere_contact_with_q_mpc(
                    env,
                    surface,
                    candidates,
                    demonstration_seed,
                    robot_state,
                    precontact_solution,
                    args,
                    pull_distance=pull_distance,
                )
            selected = candidates[int(q_config_solution["selected_candidate_index"])]
            q_waypoints = np.asarray(
                q_config_solution["q_waypoints"], dtype=np.float64
            ).reshape(-1, 7)
            selected_mode = "q_config_mpc"
            selected_device = str(getattr(args, "q_config_mpc_device", "cuda:0"))
            selected_contact_candidate_indices = np.asarray(
                q_config_solution.get(
                    "contact_set_candidate_indices",
                    [int(q_config_solution["selected_candidate_index"])],
                ),
                dtype=np.int64,
            )
            sphere_model = q_config_solution["sphere_model"]
            representative_points = q_config_solution["representative_points"]
            selected_sphere_index = int(q_config_solution["selected_sphere_index"])
            waypoint_drawer_q = np.full(
                3,
                float(close_demo._drawer_joint_value(env)),
                dtype=np.float64,
            )
            waypoint_poses = [
                _site_pose_for_arm_q(
                    env,
                    robot_state,
                    q_arm,
                    frame_name,
                    drawer_q=drawer_q,
                )
                for q_arm, drawer_q in zip(q_waypoints, waypoint_drawer_q)
            ]
            dream_sequence = np.asarray(
                [
                    np.concatenate(
                        [pos, _rotation_matrix_to_quat_wxyz(rot)],
                        axis=0,
                    )
                    for pos, rot in waypoint_poses[1:]
                ],
                dtype=np.float64,
            )
            contact_pos, contact_rotation = waypoint_poses[1]
            pull_pos, pull_rotation = waypoint_poses[-1]
            try:
                import sys as _sys

                _dbg_out = _sys.__stdout__
                _sel_world = np.asarray(selected.world_point, dtype=np.float64).reshape(
                    3
                )
                _sel_normal = np.asarray(
                    getattr(selected, "approach_world", np.zeros(3)),
                    dtype=np.float64,
                ).reshape(3)
                _delta = (
                    np.asarray(contact_pos, dtype=np.float64).reshape(3) - _sel_world
                )
                _delta_norm = float(np.linalg.norm(_delta))
                _signed = (
                    float(np.dot(_delta, _sel_normal))
                    if np.linalg.norm(_sel_normal) > 1e-9
                    else float("nan")
                )
                print(
                    "[contact_debug] "
                    f"selected_world_point={_sel_world.tolist()} "
                    f"grip_site_world={np.asarray(contact_pos, dtype=np.float64).tolist()} "
                    f"delta={_delta.tolist()} "
                    f"|delta|={_delta_norm:.6f} "
                    f"approach_dot_delta={_signed:.6f} "
                    "(>0: grip outward of surface; <0: grip inward / penetrating)",
                    file=_dbg_out,
                    flush=True,
                )
                _link_field = str(getattr(args, "curobo_contact_sphere_links", ""))
                print(
                    f"[contact_debug] curobo_contact_sphere_links={_link_field!r} "
                    f"drawer_action={getattr(args, 'drawer_action', 'open')!r} "
                    f"execute_pull_stage={getattr(args, 'execute_pull_stage', True)!r}",
                    file=_dbg_out,
                    flush=True,
                )
            except Exception as _dbg_exc:
                print(
                    f"[contact_debug] print_failed:{_dbg_exc!r}",
                    file=_sys.__stdout__,
                    flush=True,
                )
            contact_offset_ee = contact_rotation.T @ (
                np.asarray(selected.world_point, dtype=np.float64) - contact_pos
            )
            contact_position_error = float(
                np.linalg.norm(
                    contact_pos
                    + contact_rotation @ contact_offset_ee
                    - np.asarray(selected.world_point, dtype=np.float64)
                )
            )
            plan_once_started = time.perf_counter()
            contact_pose = np.concatenate(
                [contact_pos, _rotation_matrix_to_quat_wxyz(contact_rotation)],
                axis=0,
            )
            sphere_centers_world = _sphere_centers_world_from_pose(
                contact_pose, sphere_model
            )
            sphere_sdf = _point_sdf_for_spheres_numpy(
                sphere_centers_world,
                sphere_model.radii,
                representative_points,
            )
            dream_diagnostics = {
                "selected_mode": selected_mode,
                "selected_device": selected_device,
                "best_cost": float(q_config_solution["best_cost"]),
                "pull_best_cost": float(q_config_solution["pull_best_cost"]),
                "reachable_solution_count": int(
                    q_config_solution["successful_hypotheses"]
                ),
                "evaluated_hypotheses": int(q_config_solution["evaluated_hypotheses"]),
                "selected_sphere_index": int(selected_sphere_index),
                "selected_sphere_link": str(
                    sphere_model.link_names[selected_sphere_index]
                ),
                "selected_sphere_radius": float(
                    sphere_model.radii[selected_sphere_index]
                ),
                "selected_feasible_index": int(
                    q_config_solution["selected_feasible_index"]
                ),
                "contact_position_error": float(contact_position_error),
                "minimum_sphere_sdf": float(np.min(sphere_sdf)),
                "selected_sphere_sdf": float(sphere_sdf[selected_sphere_index]),
                "action_distance": float(q_config_solution["action_distance"]),
                "q_config_mpc_time": float(time.perf_counter() - mpc_started),
                "plan_once_mppi_time": float(time.perf_counter() - plan_once_started),
                "plan_once_mppi_best_cost": float("nan"),
                "plan_once_mppi_best_q_index": -1,
                "plan_once_mppi_candidate_count": 0,
                "demo_q_position_error": float(
                    q_config_solution["demonstration_q_mapping"].get(
                        "demo_position_error", float("nan")
                    )
                ),
                "demo_q_rotation_error": float(
                    q_config_solution["demonstration_q_mapping"].get(
                        "demo_rotation_error", float("nan")
                    )
                ),
                "mapped_demo_q_position_error": float(
                    q_config_solution["demonstration_q_mapping"].get(
                        "mapped_position_error", float("nan")
                    )
                ),
                "mapped_demo_q_rotation_error": float(
                    q_config_solution["demonstration_q_mapping"].get(
                        "mapped_rotation_error", float("nan")
                    )
                ),
                "mapped_demo_q_used_mink": bool(
                    q_config_solution["demonstration_q_mapping"].get(
                        "used_mink_projection", False
                    )
                ),
                "mapped_demo_q_reason": str(
                    q_config_solution["demonstration_q_mapping"].get("reason", "")
                ),
                "base_alignment_applied": bool(base_alignment.applied),
                "base_alignment_reason": str(base_alignment.reason),
                "base_alignment_yaw_delta": float(base_alignment.yaw_delta),
                "base_alignment_translation_delta": np.asarray(
                    base_alignment.translation_delta, dtype=np.float64
                ).tolist(),
                "base_alignment_initial_position_error": float(
                    base_alignment.initial_position_error
                ),
                "base_alignment_initial_rotation_error": float(
                    base_alignment.initial_rotation_error
                ),
                "base_alignment_final_position_error": float(
                    base_alignment.final_position_error
                ),
                "base_alignment_final_rotation_error": float(
                    base_alignment.final_rotation_error
                ),
                "precontact_mink_candidate_index": int(precontact_seed_candidate_index),
                "precontact_mink_position_error": float(
                    precontact_solution.position_error
                ),
                "precontact_mink_rotation_error": float(
                    precontact_solution.rotation_error
                ),
                "precontact_mink_collision_free": bool(
                    precontact_solution.collision_free
                ),
                "precontact_mink_collision_reason": str(
                    precontact_solution.collision_reason
                ),
            }
            _stage_result(
                "stage1",
                time.perf_counter() - stage_solve_started,
                int(q_config_solution["successful_hypotheses"]),
            )
        except Exception:
            _stage_result(
                "stage1",
                time.perf_counter() - stage_solve_started,
                0,
            )
            raise
    else:
        raise RuntimeError(
            "q-config MPC is required in the simplified OpenDrawer pipeline"
        )

    execute_pull = bool(getattr(args, "execute_pull_stage", True))
    if q_waypoints.size:
        waypoint_names = ("precontact", "contact", "pull")
        target_gripper_poses = [
            (name, pos, rot) for name, (pos, rot) in zip(waypoint_names, waypoint_poses)
        ]
        if not execute_pull:
            target_gripper_poses = [
                wp for wp in target_gripper_poses if wp[0] != "pull"
            ]
    else:
        contact_pose = dream_sequence[0]
        pull_pose = dream_sequence[-1]
        contact_rotation = close_demo._matrix_from_quat_wxyz(contact_pose[3:])
        pull_rotation = close_demo._matrix_from_quat_wxyz(pull_pose[3:])
        precontact_position = contact_pose[:3] + close_demo._normalize(
            selected.approach_world
        ) * float(args.precontact_distance)
        target_gripper_poses = [
            ("precontact", precontact_position, contact_rotation),
            ("contact", contact_pose[:3], contact_rotation),
        ]
        if execute_pull:
            target_gripper_poses.append(("pull", pull_pose[:3], pull_rotation))
    selected_index = next(
        index for index, candidate in enumerate(candidates) if candidate is selected
    )
    mink_solution = close_demo.MinkContactPoseSolution(
        drawer_candidate_index=int(selected_index),
        drawer_contact_world=selected.world_point,
        drawer_contact_local=selected.local_point,
        drawer_contact_cost=float(selected.cost),
        ee_sample_index=-1,
        ee_sample_name=(
            (
                f"{sphere_model.link_names[selected_sphere_index]}"
                f":sphere_{selected_sphere_index}"
            )
            if sphere_model is not None and selected_sphere_index >= 0
            else selected_mode or "demonstration_contact_pose"
        ),
        ee_contact_geom_name=(
            str(sphere_model.link_names[selected_sphere_index])
            if sphere_model is not None and selected_sphere_index >= 0
            else ""
        ),
        contact_frame=args.mink_contact_frame,
        contact_offset_local=contact_offset_ee,
        roll_angle=0.0,
        q_waypoints=q_waypoints,
        target_gripper_poses=target_gripper_poses,
        contact_position_error=contact_position_error,
        collision_free=bool(
            sphere_sdf.size
            and float(np.min(sphere_sdf))
            >= -float(args.sphere_contact_penetration_tolerance)
        ),
    )
    stage = OpenDrawerStage(
        name=stage_name,
        surface_name=surface.name,
        start_drawer_q=close_demo._drawer_joint_value(env),
        pull_distance=float(pull_distance),
        selected_contact_index=int(selected_index),
        selected_contact_world=np.asarray(selected.world_point, dtype=np.float64),
        selected_contact_local=np.asarray(selected.local_point, dtype=np.float64),
        selected_contact_cost=float(selected.cost),
        mink_solution=mink_solution,
        candidates=candidates,
        mink_reports=[],
    )
    initial_score = _score_ee_pose_contacts(
        demonstration_pose.reshape(1, 7),
        contact_offset_ee,
        feasible_points,
        max_distance=float(args.dream_initial_contact_feasible_distance),
    )
    target_contact_points = _linear_contact_target_points(
        selected.world_point,
        np.asarray(surface.pull_world, dtype=np.float64)
        * float(
            dream_diagnostics.get(
                "action_distance",
                min(float(pull_distance), float(args.dream_max_action_distance)),
            )
        ),
        dream_sequence.shape[0],
    )
    target_score = _score_ee_pose_contact_targets(
        dream_sequence,
        contact_offset_ee,
        target_contact_points,
        max_distance=float(args.dream_contact_target_tolerance),
    )
    stage.demonstration_seed = demonstration_seed
    stage.projected_seed_contact_world = projected_seed_contact_world
    stage.projected_seed_contact_offset_error = float(
        np.linalg.norm(
            projected_seed_contact_world - demonstration_seed.projected_contact_world
        )
    )
    stage.projected_seed_contact_feasible_distance = float(
        projected_seed_contact_distance[0]
    )
    stage.projected_seed_contact_feasible_index = int(projected_seed_contact_index[0])
    stage.dream_initial_poses = np.repeat(
        demonstration_pose.reshape(1, 7),
        2,
        axis=0,
    )
    stage.dream_initial_candidate_indices = np.full(
        2,
        int(selected_index),
        dtype=np.int64,
    )
    stage.dream_initial_contact_points = initial_score["contact_points_world"]
    stage.dream_initial_contact_distances = initial_score["nearest_feasible_distances"]
    stage.dream_initial_contact_mask = initial_score["feasible_mask"]
    stage.dream_initial_contact_fraction = float(initial_score["feasible_fraction"])
    stage.dream_ee_sequence = dream_sequence
    stage.dream_contact_points_world = target_score["contact_points_world"]
    stage.dream_contact_target_points_world = target_score["target_points_world"]
    stage.dream_contact_target_distances = target_score["target_distances"]
    stage.dream_contact_target_mask = target_score["target_mask"]
    stage.dream_contact_target_fraction = float(target_score["target_fraction"])
    first_distance, first_index = _nearest_point_distances(
        target_score["contact_points_world"][:1],
        feasible_points,
    )
    stage.dream_first_feasible_contact_distance = float(first_distance[0])
    stage.dream_first_feasible_contact_index = int(first_index[0])
    stage.dream_target_snap_applied = False
    stage.target_contact_points_world = target_score["contact_points_world"]
    stage.target_contact_target_points_world = target_score["target_points_world"]
    stage.target_contact_distances = target_score["target_distances"]
    stage.target_contact_mask = target_score["target_mask"]
    stage.curobo_collision_sphere_centers_ee = (
        np.zeros((0, 3), dtype=np.float64)
        if sphere_model is None
        else np.asarray(sphere_model.centers_ee, dtype=np.float64)
    )
    stage.curobo_collision_sphere_radii = (
        np.zeros(0, dtype=np.float64)
        if sphere_model is None
        else np.asarray(sphere_model.radii, dtype=np.float64)
    )
    stage.curobo_collision_sphere_links = (
        tuple() if sphere_model is None else sphere_model.link_names
    )
    stage.selected_curobo_sphere_index = int(selected_sphere_index)
    stage.curobo_sphere_sdf = sphere_sdf
    stage.object_representative_points_world = (
        np.zeros((0, 3), dtype=np.float64)
        if representative_points is None
        else np.asarray(representative_points.points_world, dtype=np.float64)
    )
    stage.object_representative_normals_world = (
        np.zeros((0, 3), dtype=np.float64)
        if representative_points is None
        else np.asarray(representative_points.normals_world, dtype=np.float64)
    )
    stage.dream_result = dream_result
    stage.dream_diagnostics = dream_diagnostics
    stage.dream_initial_pose_tolerance = float(args.dream_initial_pose_tolerance)
    stage.dream_failure_detail = {
        "reason": "collision_sphere_point_sdf_mpc",
        "terminal_open_distance": float("nan"),
        "required_open_distance": float(pull_distance)
        * float(args.dream_open_success_fraction),
        "candidate_open_success_count": 0,
        "contact_target_fraction": float("nan"),
        "candidate_contact_target_fraction": float("nan"),
        "initial_pose_error": {"min": float("nan")},
    }
    stage.gripper_mode = selected_mode
    stage.dream_device = selected_device
    stage.contact_geometry_cache = ""
    stage.feasible_graph_edges = np.zeros((0, 2), dtype=np.int64)
    stage.selected_contact_set_candidate_indices = selected_contact_candidate_indices
    if q_config_solution is not None:
        feasible_cache = q_config_solution.get("feasible_contact_cache")
        stage.feasible_contact_positions_object = (
            np.zeros((0, 3), dtype=np.float64)
            if feasible_cache is None
            else np.asarray(feasible_cache.positions_object, dtype=np.float64)
        )
        stage.feasible_contact_normals_object = (
            np.zeros((0, 3), dtype=np.float64)
            if feasible_cache is None
            else np.asarray(feasible_cache.normals_object, dtype=np.float64)
        )
        stage.feasible_contact_tangents_object = (
            np.zeros((0, 3), dtype=np.float64)
            if feasible_cache is None
            else np.asarray(feasible_cache.tangents_object, dtype=np.float64)
        )
    # Stash receding-horizon replan context so the viewer can re-solve from
    # the live joint state when the current trajectory is exhausted.
    stage._replan_context = {
        "surface_name": surface.name,
        "surface": surface,
        "q_config_solution": q_config_solution,
        "precontact_solution": precontact_solution,
        "contact_offset_ee": np.asarray(contact_offset_ee, dtype=np.float64),
        "robot_state": robot_state,
    }
    return stage, [], robot_state


def _all_target_gripper_poses(stages, exclude_phases=()):
    """Flatten per-stage gripper waypoint targets.

    `exclude_phases` lets the caller drop specific phases (e.g. "pull") so
    they are NOT handed to cuRobo. The dropped waypoints' arm-q values are
    still kept on the mink solution (`stage.mink_solution.q_waypoints[-1]`)
    so a separate post-processing pass can append a linear-DOF pull tail.
    """
    exclude = {str(name) for name in exclude_phases}
    poses = []
    for stage in stages:
        for name, pos, rot in stage.mink_solution.target_gripper_poses:
            if name in exclude:
                continue
            poses.append((f"{stage.name}:{name}", pos, rot))
    return poses


def _joint_space_trajectory_from_stage_waypoints(
    robot_state, stages, steps_per_segment
):
    current_q = np.asarray(robot_state["q"], dtype=np.float64).reshape(7)
    chunks = []
    segments = []
    phase_names = ("precontact", "contact", "pull")
    steps = max(int(steps_per_segment), 2)
    for stage in stages:
        q_waypoints = np.asarray(stage.mink_solution.q_waypoints, dtype=np.float64)
        if q_waypoints.size == 0:
            raise RuntimeError(
                f"Stage '{stage.name}' has no q waypoints for joint interpolation"
            )
        q_waypoints = q_waypoints.reshape(-1, 7)
        for phase, goal_q in zip(phase_names, q_waypoints):
            alphas = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)[1:, None]
            segment = (1.0 - alphas) * current_q[None, :] + alphas * goal_q[None, :]
            chunks.append(segment)
            segments.append(
                {
                    "name": f"{stage.name}:{phase}",
                    "steps": int(segment.shape[0]),
                    "planner": "joint_space_interpolation",
                }
            )
            current_q = goal_q.copy()
    return np.concatenate(chunks, axis=0), segments


def _mppi_joint_trajectory_from_stage_rollouts(robot_state, stages):
    current_q = np.asarray(robot_state["q"], dtype=np.float64).reshape(1, 7)
    chunks = []
    segments = []
    for stage in stages:
        result = getattr(stage, "dream_result", None)
        q_sequence = getattr(result, "arm_qpos_sequence", None)
        if q_sequence is None:
            raise RuntimeError(
                f"Stage '{stage.name}' has no MPPI arm_qpos_sequence. "
                "CompositeRolloutCost must run successfully before MPPI execution."
            )
        if hasattr(q_sequence, "detach"):
            q_sequence = q_sequence.detach().cpu().numpy()
        q_sequence = np.asarray(q_sequence, dtype=np.float64).reshape(-1, 7)
        if q_sequence.shape[0] == 0:
            raise RuntimeError(
                f"Stage '{stage.name}' has an empty MPPI arm_qpos_sequence"
            )
        if not chunks:
            if not np.allclose(q_sequence[0], current_q[0]):
                q_sequence = np.concatenate([current_q, q_sequence], axis=0)
        else:
            q_sequence = q_sequence[1:]
        chunks.append(q_sequence)
        contact_steps = max(1, int(np.ceil(q_sequence.shape[0] * 0.5)))
        pull_steps = max(0, int(q_sequence.shape[0]) - contact_steps)
        segments.append(
            {
                "name": f"{stage.name}:contact",
                "steps": int(contact_steps),
                "planner": "mppi_composite_rollout_cost",
            }
        )
        if pull_steps > 0:
            segments.append(
                {
                    "name": f"{stage.name}:pull",
                    "steps": int(pull_steps),
                    "planner": "mppi_composite_rollout_cost",
                }
            )
        current_q = q_sequence[-1:].copy()
    if not chunks:
        raise RuntimeError("No MPPI rollout joint trajectory is available")
    q_traj = np.concatenate(chunks, axis=0)
    q_delta = np.linalg.norm(q_traj - q_traj[:1], axis=1)
    print(
        "[mppi_execution] "
        f"frames={q_traj.shape[0]} "
        f"max_joint_delta={float(np.max(q_delta)):.6f} "
        f"final_joint_delta={float(q_delta[-1]):.6f}",
        flush=True,
    )
    return q_traj, segments


def _parse_config_value(value):
    import yaml

    parsed = yaml.safe_load(value)
    return (
        value
        if parsed is None and value.lower() not in ("null", "none", "~")
        else parsed
    )


def _apply_config_overrides(config, overrides):
    for item in overrides:
        key, sep, raw_value = str(item).partition("=")
        if not sep or not key:
            raise ValueError(f"Override must use key=value syntax: {item!r}")
        config[key.replace("-", "_")] = _parse_config_value(raw_value)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenDrawer contact demo configured by YAML."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("open_drawer_contact_curobo.yaml")),
        help="YAML config file containing demo arguments.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value. May be passed multiple times.",
    )
    cli = parser.parse_args()
    config = _load_yaml_config(cli.config)
    config["scene_cache_dir"] = str(
        config.get("scene_cache_dir") or (REPO_ROOT / "outputs" / "scene_point_cache")
    )
    config.setdefault("mink_arm_posture_cost", 0.02)
    config.setdefault("mink_locked_dof_cost", 200.0)
    config.setdefault("object_representative_point_count", 2048)
    config.setdefault("object_representative_min_per_geom", 16)
    config.setdefault("curobo_trajopt_tsteps", 32)
    config.setdefault("curobo_interpolation_dt", 0.02)
    config.setdefault("curobo_ik_seeds", 16)
    config.setdefault("curobo_graph_seeds", 2)
    config.setdefault("curobo_trajopt_seeds", 2)
    config.setdefault("curobo_max_attempts", 2)
    config.setdefault("curobo_enable_graph_attempt", 1)
    config.setdefault("disable_curobo_self_collision", False)
    config.setdefault("disable_curobo_cuda_graph", False)
    config.setdefault("curobo_world_padding", 0.005)
    config.setdefault("curobo_world_exclude_geoms", "")
    config.setdefault("curobo_world_exclude_bodies", "")
    config.setdefault("curobo_world_max_obstacles", None)
    config.setdefault("curobo_joint_enable_graph", False)
    config.setdefault("curobo_joint_enable_graph_attempt", None)
    config.setdefault("curobo_joint_disable_graph_attempt", None)
    config.setdefault("curobo_joint_max_attempts", 6)
    config.setdefault("curobo_joint_timeout", 5.0)
    config.setdefault("curobo_joint_retry_graph", True)
    config.setdefault("curobo_joint_graph_max_attempts", 2)
    config.setdefault("curobo_joint_graph_timeout", 8.0)
    config.setdefault("curobo_joint_enable_finetune_trajopt", False)
    config.setdefault("curobo_joint_check_start_validity", False)
    config.setdefault("execute_pull_stage", True)
    _apply_config_overrides(config, cli.overrides)
    return argparse.Namespace(**config)


def main():
    args = parse_args()
    _configure_cuda_memory_limit(args)
    with _suppress_solver_output():
        env = create_open_drawer_env(args)
    try:
        with _suppress_solver_output():
            _initialize_scene_processing(env, args)
        initial_robot_state = close_demo.get_robot_arm_state(env)
        total_pull_distance = _target_pull_distance(env, args)
        initial_panel = close_demo.get_panel_frame(env)
        handle_surface = _make_handle_inner_surface(env, initial_panel)

        stages = []
        target_hand_poses = []
        first_stage, _, _ = _solve_stage(
            env,
            handle_surface,
            total_pull_distance,
            args,
            stage_name="handle_inner_pull",
        )
        stages.append(first_stage)

        # Plan from the original arm configuration through all contact poses, but use
        # the CURRENT robot base pose (mink/base alignment may have moved the base
        # during _solve_stage; target_gripper_poses are recorded in that aligned
        # world frame, so cuRobo must transform into the same base).
        initial_q = np.asarray(initial_robot_state["q"], dtype=np.float64).copy()
        _qpos_saved = env.sim.data.qpos.copy()
        _qvel_saved = env.sim.data.qvel.copy()
        try:
            close_demo._set_env_arm_q(
                env, initial_robot_state["robocasa_joint_names"], initial_q
            )
            env.sim.forward()
            robot_state = close_demo.get_robot_arm_state(env)
        finally:
            env.sim.data.qpos[:] = _qpos_saved
            env.sim.data.qvel[:] = _qvel_saved
            env.sim.forward()
        target_gripper_poses = _all_target_gripper_poses(stages)
        target_hand_poses = [
            (name, *close_demo.gripper_pose_to_curobo_hand_pose(pos, rot, robot_state))
            for name, pos, rot in target_gripper_poses
        ]

        q_traj = None
        segments = []
        execution_planner = str(getattr(args, "execution_planner", "mppi")).lower()
        if execution_planner == "mppi":
            staged_solution = solve_stages(
                env,
                stages,
                args,
                robot_state=robot_state,
                frame_name=str(
                    getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
                ),
            )
            q_traj = staged_solution["q_traj"]
            segments = staged_solution["segments"]
            args._execution_planner_source = "solve_stages_curobo_approach_mppi_contact"
            args._curobo_target_diagnostics = []
            _count_result(
                "solve_stages",
                0.0,
                "successful_segments",
                int(len(segments)),
            )
        elif execution_planner == "curobo" and not args.skip_curobo:
            curobo_started = time.perf_counter()
            try:
                can_use_joint_interpolation = (
                    bool(getattr(args, "allow_joint_interpolation_fallback", False))
                    and bool(getattr(args, "use_joint_interpolation", False))
                ) and all(
                    np.asarray(stage.mink_solution.q_waypoints).size for stage in stages
                )
                if can_use_joint_interpolation:
                    q_traj, segments = _joint_space_trajectory_from_stage_waypoints(
                        robot_state,
                        stages,
                        int(args.joint_interpolation_steps_per_segment),
                    )
                    args._execution_planner_source = "joint_space_interpolation"
                    joint_validation = _validate_joint_space_trajectory_collisions(
                        env,
                        stages,
                        q_traj,
                        segments,
                        args,
                    )
                    args._joint_space_validation = joint_validation
                    if not bool(joint_validation["valid"]):
                        first_failure = joint_validation["failures"][0]
                        raise RuntimeError(
                            "Joint-space interpolation failed collision validation: "
                            f"step={first_failure['step']}, "
                            f"stage={first_failure['stage']}, "
                            f"reason={first_failure['reason']}"
                        )
                    args._curobo_target_diagnostics = []
                else:
                    with _suppress_solver_output():
                        close_demo.preload_curobo_runtime()
                    with _suppress_solver_output():
                        q_traj, segments = close_demo.plan_with_curobo(
                            robot_state,
                            target_hand_poses,
                            args,
                        )
                    for segment in segments:
                        segment.setdefault("planner", "curobo")
                    args._execution_planner_source = "curobo"
                    args._curobo_target_diagnostics = (
                        _validate_curobo_reaches_gripper_targets(
                            env,
                            robot_state,
                            target_gripper_poses,
                            q_traj,
                            segments,
                            args,
                        )
                    )
            except Exception as exc:
                _count_result(
                    "curobo",
                    time.perf_counter() - curobo_started,
                    "successful_segments",
                    0,
                )
                metadata = _save_open_outputs(
                    args.output,
                    env,
                    stages,
                    target_hand_poses,
                    q_traj,
                    segments,
                    args,
                )
                raise RuntimeError(
                    "cuRobo planning or target validation failed after OpenDrawer contact sampling. "
                    "Rerun with --skip-curobo to inspect contact and mink outputs only."
                ) from exc
            curobo_success_count = (
                int(len(segments))
                if (q_traj is not None and np.asarray(q_traj).shape[0] > 0)
                else 0
            )
            _count_result(
                "curobo",
                time.perf_counter() - curobo_started,
                "successful_segments",
                curobo_success_count,
            )
        elif execution_planner == "curobo":
            print("[curobo] skipped by --skip-curobo", flush=True)
        else:
            raise RuntimeError(
                f"Unsupported execution planner '{execution_planner}'. "
                "Use --execution-planner mppi or --execution-planner curobo."
            )

        if (
            bool(getattr(args, "allow_joint_interpolation_fallback", False))
            and (q_traj is None or not np.asarray(q_traj).size)
            and all(
                np.asarray(stage.mink_solution.q_waypoints).size for stage in stages
            )
        ):
            q_traj, segments = _joint_space_trajectory_from_stage_waypoints(
                robot_state,
                stages,
                int(args.joint_interpolation_steps_per_segment),
            )
            args._joint_space_validation = {
                "valid": True,
                "evaluated_steps": int(np.asarray(q_traj).reshape(-1, 7).shape[0]),
                "failures": [],
                "source": "execution_viewer_fallback",
            }
            args._execution_planner_source = "joint_space_interpolation_fallback"
            print(
                "[execution_viewer] using stage q_waypoints joint interpolation fallback",
                flush=True,
            )
        elif q_traj is None or not np.asarray(q_traj).size:
            raise RuntimeError(
                "No executable arm trajectory was produced. "
                "By default this demo requires MPPI CompositeRolloutCost execution; "
                "use --execution-planner curobo for cuRobo execution or "
                "--allow-joint-interpolation-fallback only for diagnostics."
            )

        # Receding-horizon replan: when the viewer finishes the current
        # trajectory, call back into MPPI from the LIVE joint state.
        # Re-uses the LAST stage's solve context (object goal + surface).
        replan_state = {
            "iterations": 0,
            "max_iterations": int(getattr(args, "execution_max_replans", 20)),
            "last_drawer_q": float(close_demo._drawer_joint_value(env)),
            "stall_count": 0,
        }

        def _replan_fn(current_q):
            replan_state["iterations"] += 1
            if replan_state["iterations"] > replan_state["max_iterations"]:
                print(
                    f"[replan] reached max_iterations="
                    f"{replan_state['max_iterations']}; stopping.",
                    flush=True,
                )
                return None
            last_stage = stages[-1]
            ctx = getattr(last_stage, "_replan_context", None)
            if ctx is None:
                return None
            # Snap MuJoCo state to the live joint config the viewer just
            # finished executing so MPPI rolls out from the right place.
            close_demo._set_env_arm_q(
                env, ctx["robot_state"]["robocasa_joint_names"], current_q
            )
            env.sim.forward()
            # Reconstruct the handle surface from the CURRENT panel pose (the
            # drawer has moved, so panel.center_world drifted).
            panel = close_demo.get_panel_frame(env)
            surface_live = _make_handle_inner_surface(env, panel)
            try:
                with _suppress_solver_output():
                    sol = solve_plan_once_mppi_for_q_configs(
                        env,
                        surface_live,
                        ctx["q_config_solution"],
                        ctx["contact_offset_ee"],
                        args,
                        robot_state=ctx["robot_state"],
                    )
            except Exception as exc:
                print(f"[replan] solve failed: {exc}", flush=True)
                return None
            result = sol.get("result")
            arm_seq = getattr(result, "arm_qpos_sequence", None)
            if arm_seq is None:
                return None
            if hasattr(arm_seq, "detach"):
                arm_seq = arm_seq.detach().cpu().numpy()
            arm_seq = np.asarray(arm_seq, dtype=np.float64).reshape(-1, 7)
            # Stall detection: stop if the drawer joint barely changed across
            # replans (avoids spinning forever in a local minimum).
            new_drawer_q = float(close_demo._drawer_joint_value(env))
            drawer_delta = abs(new_drawer_q - replan_state["last_drawer_q"])
            replan_state["last_drawer_q"] = new_drawer_q
            if drawer_delta < float(
                getattr(args, "execution_replan_stall_threshold", 1e-3)
            ):
                replan_state["stall_count"] += 1
            else:
                replan_state["stall_count"] = 0
            if replan_state["stall_count"] >= int(
                getattr(args, "execution_replan_stall_limit", 3)
            ):
                print(
                    f"[replan] drawer stalled for "
                    f"{replan_state['stall_count']} replans; stopping.",
                    flush=True,
                )
                return None
            print(
                f"[replan] iter={replan_state['iterations']} "
                f"drawer_q={new_drawer_q:.4f} (Δ={drawer_delta:.4f}) "
                f"arm_traj={arm_seq.shape[0]}f cost={sol['best_cost']:.3f}",
                flush=True,
            )
            return arm_seq

        replan_fn = _replan_fn if execution_planner == "mppi" else None

        _play_open_trajectory_viewer(
            env,
            stages,
            q_traj,
            segments,
            robot_state,
            args,
            replan_fn=replan_fn,
        )

        metadata = _save_open_outputs(
            args.output,
            env,
            stages,
            target_hand_poses,
            q_traj,
            segments,
            args,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
