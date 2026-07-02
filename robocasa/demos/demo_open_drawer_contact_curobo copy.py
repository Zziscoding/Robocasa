import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
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
from robocasa.demos.object_cso import (
    allocate_surface_samples as _allocate_surface_samples,
    point_sdf_for_spheres_numpy as _point_sdf_for_spheres_numpy,
    sphere_centers_world_from_pose as _sphere_centers_world_from_pose,
)
from robocasa.demos.franka_collision_model import (
    load_curobo_ee_collision_spheres as _load_curobo_ee_collision_spheres,
    solve_collision_sphere_contact_with_q_mpc as _solve_collision_sphere_contact_with_q_mpc,
)
from robocasa.demos.open_drawer.math import (
    _array_value_or_nan,
    _distance_summary,
    _nearest_point_distances,
    _rotation_angle_error,
    _rotation_matrix_to_quat_wxyz,
    _score_ee_pose_contact_targets,
    _score_ee_pose_contacts,
)
from robocasa.demos.open_drawer.utils import (
    _configure_cuda_memory_limit,
    _cuda_memory_limit_bytes,
    _round_up,
)

_QUIET_IMPORT_SINK = open(os.devnull, "w")
with contextlib.redirect_stdout(_QUIET_IMPORT_SINK), contextlib.redirect_stderr(
    _QUIET_IMPORT_SINK
):
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    import robocasa  # noqa: F401
    import robocasa.utils.lerobot_utils as LU
    import robocasa.demos.demo_close_drawer_contact_curobo as close_demo
    import robocasa.demos.mink_solver as mink_solver
    from robocasa.demos.demo_tasks import get_ds_path_any_split
    from robocasa.demos.scene_process import (
        MJWarpVisibility,
        build_or_load_scene_points,
    )
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to


@dataclass
class OpenContactSurface:
    name: str
    center_world: np.ndarray
    rotation_world: np.ndarray
    half_size: np.ndarray
    approach_world: np.ndarray
    pull_world: np.ndarray
    geom_name: str
    allowed_geom_names: tuple[str, ...]
    contact_local_y: float
    force_normal_local: np.ndarray


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


def _limit_q_config_samples_for_memory(config, raw_model, args):
    limit_bytes = _cuda_memory_limit_bytes(args)
    if limit_bytes <= 0:
        return config
    budget_fraction = float(getattr(args, "q_config_mpc_memory_budget_fraction", 0.20))
    budget_bytes = int(limit_bytes * max(min(budget_fraction, 1.0), 0.05))
    nv = int(getattr(raw_model, "nv", 0))
    njmax = int(config.njmax_per_env)
    njmax_pad = _round_up(njmax, 16)
    nv_pad = _round_up(nv, 16) if nv > 32 else _round_up(nv, 4)
    # Dense Comfree constraint Jacobian dominates q-config memory:
    # efc.J shape is (num_samples, njmax_pad, nv_pad) float64.
    # The dense constraint Jacobian is only the dominant persistent buffer. Comfree
    # allocates additional temporary buffers during stepping, so keep a conservative
    # headroom factor here instead of filling the nominal CUDA budget.
    bytes_per_sample = max(njmax_pad * nv_pad * 8 * 4, 1)
    max_samples = max(1, budget_bytes // bytes_per_sample)
    if int(config.num_samples) <= max_samples:
        return config
    bounded = max(1, int(max_samples))
    print(
        "[q_config_mpc] reducing num_samples from "
        f"{int(config.num_samples)} to {bounded} to stay within "
        f"{float(getattr(args, 'cuda_memory_limit_gb', 20.0)):.2f}GB CUDA budget",
        flush=True,
    )
    return replace(config, num_samples=bounded)


def _q_config_result_to_cpu(result):
    return replace(
        result,
        best_q=result.best_q.detach().cpu(),
        candidate_q=result.candidate_q.detach().cpu(),
        candidate_costs=result.candidate_costs.detach().cpu(),
        best_sphere_centers_world=result.best_sphere_centers_world.detach().cpu(),
        best_contact_distances=result.best_contact_distances.detach().cpu(),
    )


@contextlib.contextmanager
def _suppress_solver_output():
    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def create_open_drawer_env(args):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=args.robot,
    )
    has_offscreen_renderer = False
    try:
        env = robosuite.make(
            env_name="OpenDrawer",
            robots=args.robot,
            controller_configs=controller_config,
            has_renderer=bool(args.render),
            has_offscreen_renderer=has_offscreen_renderer,
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
    except ImportError:
        if not has_offscreen_renderer:
            raise
        pass
    env = robosuite.make(
        env_name="OpenDrawer",
        robots=args.robot,
        controller_configs=controller_config,
        has_renderer=bool(args.render),
        has_offscreen_renderer=False,
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


def _target_pull_distance(env, args):
    if args.pull_distance is not None:
        return max(float(args.pull_distance), 1e-4)
    return max(
        float(env.drawer.size[1]) * float(args.target_open_fraction) * 0.55, 1e-4
    )


def _handle_reg_geom_name(env):
    model = env.sim.model
    drawer_name = env.drawer.name
    preferred = f"{drawer_name}_door_handle_reg_main"
    if preferred in model._geom_name2id:
        return preferred
    handle_names = sorted(
        name
        for name in model._geom_name2id
        if name.startswith(f"{drawer_name}_door_handle_g")
    )
    if not handle_names:
        raise RuntimeError(f"Cannot find handle geoms for drawer '{drawer_name}'.")
    return handle_names[0]


def _handle_geom_names(env):
    model = env.sim.model
    drawer_name = env.drawer.name
    names = sorted(
        name
        for name in model._geom_name2id
        if name.startswith(f"{drawer_name}_door_handle_g")
    )
    if not names:
        raise RuntimeError(f"Cannot find handle geoms for drawer '{drawer_name}'.")
    return tuple(names)


def _handle_allowed_geom_names(env):
    model = env.sim.model
    drawer_name = env.drawer.name
    names = sorted(
        name
        for name in model._geom_name2id
        if name.startswith(f"{drawer_name}_door_handle")
    )
    if not names:
        raise RuntimeError(f"Cannot find handle geoms for drawer '{drawer_name}'.")
    return tuple(names)


def _make_handle_inner_surface(env, panel):
    model = env.sim.model
    data = env.sim.data
    handle_names = _handle_geom_names(env)
    corners_panel = []
    for name in handle_names:
        geom_id = model.geom_name2id(name)
        geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        geom_rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        geom_size = np.asarray(model.geom_size[geom_id], dtype=np.float64).copy()
        geom_size = np.maximum(
            geom_size, np.array([0.002, 0.002, 0.002], dtype=np.float64)
        )
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corner_world = geom_pos + geom_rot @ (
                        geom_size * np.array([sx, sy, sz], dtype=np.float64)
                    )
                    corners_panel.append(
                        panel.rotation_world.T @ (corner_world - panel.center_world)
                    )
    corners_panel = np.asarray(corners_panel, dtype=np.float64)
    min_corner = np.min(corners_panel, axis=0)
    max_corner = np.max(corners_panel, axis=0)
    center_local = 0.5 * (min_corner + max_corner)
    half_size = np.maximum(
        0.5 * (max_corner - min_corner),
        np.array([0.006, 0.006, 0.006], dtype=np.float64),
    )
    center_world = panel.center_world + panel.rotation_world @ center_local
    return OpenContactSurface(
        name="handle",
        center_world=center_world,
        rotation_world=np.asarray(panel.rotation_world, dtype=np.float64).copy(),
        half_size=half_size,
        approach_world=np.asarray(panel.push_world, dtype=np.float64).copy(),
        pull_world=np.asarray(panel.outward_world, dtype=np.float64).copy(),
        geom_name=handle_names[0],
        allowed_geom_names=_handle_allowed_geom_names(env),
        contact_local_y=float(half_size[1]),
        force_normal_local=np.array([0.0, -1.0, 0.0], dtype=np.float64),
    )


def _make_panel_inner_surface(env, panel):
    drawer_name = env.drawer.name
    panel_geom_names = tuple(
        sorted(
            name
            for name in env.sim.model._geom_name2id
            if name == panel.geom_name or name.startswith(f"{drawer_name}_door_g")
        )
    )
    return OpenContactSurface(
        name="panel_inner",
        center_world=np.asarray(panel.center_world, dtype=np.float64).copy(),
        rotation_world=np.asarray(panel.rotation_world, dtype=np.float64).copy(),
        half_size=np.asarray(panel.half_size, dtype=np.float64).copy(),
        approach_world=np.asarray(panel.push_world, dtype=np.float64).copy(),
        pull_world=np.asarray(panel.outward_world, dtype=np.float64).copy(),
        geom_name=panel.geom_name,
        allowed_geom_names=panel_geom_names,
        contact_local_y=float(panel.half_size[1]),
        force_normal_local=np.array([0.0, -1.0, 0.0], dtype=np.float64),
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


def _grid_shape_for_count(sample_count):
    sample_count = max(int(sample_count), 1)
    if sample_count == 15:
        return 5, 3
    best = (sample_count, 1)
    best_score = sample_count
    for rows in range(1, int(np.sqrt(sample_count)) + 2):
        cols = int(np.ceil(sample_count / rows))
        score = abs(cols - rows) + (cols * rows - sample_count)
        if score < best_score:
            best = (cols, rows)
            best_score = score
    return best


def _face_grid_points(count):
    base = np.array(
        [
            [0.0, 0.0],
            [-0.75, -0.75],
            [0.75, -0.35],
            [-0.35, 0.75],
            [0.75, 0.75],
        ],
        dtype=np.float64,
    )
    if count <= len(base):
        return base[:count]
    # Uniform tensor grid covering [-0.85, 0.85]^2; pick the smallest n with n*n >= count
    # to avoid the previous sqrt(count) rounding leaving large gaps near edges.
    n = max(int(np.ceil(np.sqrt(count))), 2)
    values = np.linspace(-0.85, 0.85, n)
    grid = np.asarray([(u, v) for v in values for u in values], dtype=np.float64)
    return grid[:count]


def sample_handle_surface_candidates(surface, sample_count, margin):
    per_face = max(int(sample_count) // 4, 1)
    remainder = max(int(sample_count) - 4 * per_face, 0)
    face_counts = [
        per_face + (1 if face_idx < remainder else 0) for face_idx in range(4)
    ]
    hx = max(float(surface.half_size[0]) - float(margin), 1e-4)
    hy = max(float(surface.half_size[1]) - float(margin), 1e-4)
    hz = max(float(surface.half_size[2]) - float(margin), 1e-4)
    face_specs = (
        (
            "inner",
            1,
            hy,
            (0, hx),
            (2, hz),
            np.array([0.0, -1.0, 0.0]),
            surface.rotation_world[:, 1],
        ),
        (
            "outer",
            1,
            -hy,
            (0, hx),
            (2, hz),
            np.array([0.0, 1.0, 0.0]),
            -surface.rotation_world[:, 1],
        ),
        (
            "top",
            2,
            hz,
            (0, hx),
            (1, hy),
            np.array([0.0, 0.0, -1.0]),
            surface.rotation_world[:, 2],
        ),
        (
            "bottom",
            2,
            -hz,
            (0, hx),
            (1, hy),
            np.array([0.0, 0.0, 1.0]),
            -surface.rotation_world[:, 2],
        ),
    )
    local_points = []
    world_points = []
    force_normals = []
    approach_dirs = []
    face_names = []
    for face_count, (
        face_name,
        fixed_axis,
        fixed_value,
        axis_a,
        axis_b,
        normal,
        approach,
    ) in zip(face_counts, face_specs):
        for u, v in _face_grid_points(face_count):
            p_local = np.zeros(3, dtype=np.float64)
            p_local[fixed_axis] = fixed_value
            p_local[axis_a[0]] = float(u) * axis_a[1]
            p_local[axis_b[0]] = float(v) * axis_b[1]
            local_points.append(p_local)
            world_points.append(surface.center_world + surface.rotation_world @ p_local)
            force_normals.append(np.asarray(normal, dtype=np.float64))
            approach_dirs.append(close_demo._normalize(approach))
            face_names.append(face_name)
    return (
        np.asarray(local_points, dtype=np.float64),
        np.asarray(world_points, dtype=np.float64),
        np.asarray(force_normals, dtype=np.float64),
        np.asarray(approach_dirs, dtype=np.float64),
        face_names,
    )


def sample_inner_surface_candidates(surface, sample_count, margin):
    if surface.name == "handle":
        return sample_handle_surface_candidates(surface, sample_count, margin)
    grid_x, grid_z = _grid_shape_for_count(sample_count)
    x_lim = max(float(surface.half_size[0]) - float(margin), 1e-4)
    z_lim = max(float(surface.half_size[2]) - float(margin), 1e-4)
    xs = np.linspace(-x_lim, x_lim, grid_x)
    zs = np.linspace(-z_lim, z_lim, grid_z)
    local = []
    world = []
    for z in zs:
        for x in xs:
            if len(local) >= sample_count:
                break
            p_local = np.array([x, float(surface.contact_local_y), z], dtype=np.float64)
            p_world = surface.center_world + surface.rotation_world @ p_local
            local.append(p_local)
            world.append(p_world)
    count = len(local)
    force_normals = np.repeat(surface.force_normal_local.reshape(1, 3), count, axis=0)
    approach_dirs = np.repeat(surface.approach_world.reshape(1, 3), count, axis=0)
    face_names = ["inner"] * count
    return (
        np.asarray(local, dtype=np.float64),
        np.asarray(world, dtype=np.float64),
        force_normals,
        approach_dirs,
        face_names,
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


def _surface_contact_sample_count(surface, args):
    if args.contact_sample_count is not None:
        return max(int(args.contact_sample_count), 1)
    if surface.name == "handle":
        return max(int(args.handle_contact_sample_count), 1)
    return max(int(args.panel_contact_sample_count), 1)


def _make_open_gripper_contact_rotation(contact_axis_world):
    z_axis = close_demo._normalize(contact_axis_world)
    ref_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(z_axis, ref_axis))) > 0.95:
        ref_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(ref_axis, z_axis)
    x_axis = close_demo._normalize(
        x_axis, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    y_axis = close_demo._normalize(
        np.cross(z_axis, x_axis), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    x_axis = close_demo._normalize(np.cross(y_axis, z_axis), fallback=x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _project_contact_samples_to_actual_geoms(
    env,
    surface,
    local_points,
    world_points,
    approach_dirs,
):
    """Snap proxy-box samples onto the actual MuJoCo drawer geometry."""
    from robocasa.demos.scene_process import _geom_mesh_in_body

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    preferred_names = [
        name
        for name in surface.allowed_geom_names
        if "_g" in str(name) and name in env.sim.model._geom_name2id
    ]
    geom_names = preferred_names or [
        name
        for name in surface.allowed_geom_names
        if name in env.sim.model._geom_name2id
    ]
    meshes = []
    for name in geom_names:
        geom_id = int(env.sim.model.geom_name2id(name))
        if float(raw_model.geom_rgba[geom_id, 3]) <= 1e-5:
            continue
        mesh_body = _geom_mesh_in_body(raw_model, geom_id)
        if mesh_body is None or not mesh_body.faces.size:
            continue
        body_id = int(raw_model.geom_bodyid[geom_id])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            raw_data.xmat[body_id], dtype=np.float64
        ).reshape(3, 3)
        transform[:3, 3] = np.asarray(raw_data.xpos[body_id], dtype=np.float64).reshape(
            3
        )
        mesh_world = mesh_body.copy()
        mesh_world.apply_transform(transform)
        meshes.append(mesh_world)
    if not meshes:
        return (
            np.asarray(local_points, dtype=np.float64),
            np.asarray(world_points, dtype=np.float64),
            np.asarray(approach_dirs, dtype=np.float64),
            np.zeros(len(world_points), dtype=np.float64),
        )

    query = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    best_distance = np.full(query.shape[0], np.inf, dtype=np.float64)
    best_points = query.copy()
    best_normals = np.asarray(approach_dirs, dtype=np.float64).copy()
    for mesh in meshes:
        closest, distance, face_ids = trimesh.proximity.closest_point_naive(mesh, query)
        better = np.asarray(distance) < best_distance
        if not np.any(better):
            continue
        normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_ids]
        best_distance[better] = np.asarray(distance)[better]
        best_points[better] = np.asarray(closest)[better]
        best_normals[better] = normals[better]

    original_approach = np.asarray(approach_dirs, dtype=np.float64)
    flip = np.sum(best_normals * original_approach, axis=1) < 0.0
    best_normals[flip] *= -1.0
    best_normals /= np.linalg.norm(best_normals, axis=1, keepdims=True).clip(min=1e-12)
    projected_local = (
        best_points - np.asarray(surface.center_world, dtype=np.float64)
    ) @ np.asarray(surface.rotation_world, dtype=np.float64)
    return projected_local, best_points, best_normals, best_distance


_ENABLE_STAGE_BANNERS = False


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


def _initialize_scene_processing(env, args):
    args._scene_point_cloud = None
    args._scene_visibility = None
    args._scene_runtime_data = env.sim.data
    if not args.scene_process:
        return
    point_cloud = build_or_load_scene_points(
        env.drawer.get_xml(),
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


def _robot_contact_geom_sets_for_surface(env, surface):
    model = env.sim.model
    robot_geoms = {
        geom_id
        for name, geom_id in model._geom_name2id.items()
        if name.startswith("robot0_") and "collision" in name
    }
    ee_geoms = {
        geom_id
        for name, geom_id in model._geom_name2id.items()
        if name.startswith("gripper0_right_") and "collision" in name
    }
    robot_geoms.update(ee_geoms)
    drawer_name = surface.geom_name.split("_door")[0]
    allowed_target_geoms = {
        geom_id
        for name, geom_id in model._geom_name2id.items()
        if name.startswith(f"{drawer_name}_")
    }
    return robot_geoms, ee_geoms, allowed_target_geoms


def _check_arm_q_collision_for_surface(
    env,
    surface,
    arm_joint_names,
    q_arm,
    drawer_q,
    allowed_ee_geom_name=None,
    penetration_tolerance=0.0,
    collision_scope="ee",
):
    model = env.sim.model
    data = env.sim.data
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    geom_id_to_name = {geom_id: name for name, geom_id in model._geom_name2id.items()}
    try:
        close_demo._set_env_arm_q(env, arm_joint_names, q_arm)
        close_demo._set_drawer_joint_value(env, drawer_q)
        env.sim.forward()
        (
            robot_geoms,
            ee_geoms,
            allowed_target_geoms,
        ) = _robot_contact_geom_sets_for_surface(env, surface)
        checked_robot_geoms = ee_geoms if collision_scope == "ee" else robot_geoms
        allowed_ee_geoms = set(ee_geoms)
        if (
            allowed_ee_geom_name is not None
            and allowed_ee_geom_name in model._geom_name2id
        ):
            allowed_ee_geoms = {model.geom_name2id(allowed_ee_geom_name)}
        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in checked_robot_geoms and geom2 not in checked_robot_geoms:
                continue
            if geom1 in robot_geoms and geom2 in robot_geoms:
                continue
            robot_geom = geom1 if geom1 in checked_robot_geoms else geom2
            other_geom = geom2 if robot_geom == geom1 else geom1
            contact_dist = float(contact.dist)
            if other_geom not in allowed_target_geoms:
                if robot_geom in ee_geoms and contact_dist < -max(
                    float(penetration_tolerance), 0.0
                ):
                    robot_name = geom_id_to_name.get(robot_geom, str(robot_geom))
                    other_name = geom_id_to_name.get(other_geom, str(other_geom))
                    return (
                        False,
                        f"ee_env_penetration:{robot_name}--{other_name}:dist={contact_dist:.6f}",
                    )
                continue
            if robot_geom in allowed_ee_geoms:
                if contact_dist >= -max(float(penetration_tolerance), 0.0):
                    continue
                robot_name = geom_id_to_name.get(robot_geom, str(robot_geom))
                other_name = geom_id_to_name.get(other_geom, str(other_geom))
                return (
                    False,
                    f"target_penetration:{robot_name}--{other_name}:dist={contact_dist:.6f}",
                )
            if contact_dist >= -max(float(penetration_tolerance), 0.0):
                continue
            robot_name = geom_id_to_name.get(robot_geom, str(robot_geom))
            other_name = geom_id_to_name.get(other_geom, str(other_geom))
            return (
                False,
                f"collision:{robot_name}--{other_name}:dist={contact_dist:.6f}",
            )
        return True, "collision_free"
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        env.sim.forward()


def _extract_geom_points(
    env,
    name_filter,
    total_surface_points=2000,
    color_override=None,
    min_points_per_geom=64,
    geom_ids=None,
):
    """Sample a fixed-budget colored cloud from selected MuJoCo geoms.

    The previous implementation multiplied the requested count by mesh area in
    square metres. Most kitchen and gripper geoms therefore collapsed to the
    64-point floor. Here the requested count is a total point budget, allocated
    by surface area while retaining a per-geom minimum.
    """
    import trimesh

    model = env.sim.model
    raw_model = model._model if hasattr(model, "_model") else model
    data = env.sim.data
    raw_data = data._data if hasattr(data, "_data") else data
    selected_geom_ids = (
        None if geom_ids is None else {int(geom_id) for geom_id in geom_ids}
    )
    sampled_geoms = []
    geom_id_to_name = {int(v): k for k, v in model._geom_name2id.items()}
    for geom_id in range(int(raw_model.ngeom)):
        name = geom_id_to_name.get(geom_id, "")
        if selected_geom_ids is not None:
            if geom_id not in selected_geom_ids:
                continue
        elif not name_filter(name):
            continue
        try:
            if int(raw_model.geom_group[geom_id]) not in (0, 1, 2):
                continue
            geom_type = int(raw_model.geom_type[geom_id])
            size = np.asarray(raw_model.geom_size[geom_id], dtype=np.float64)
        except Exception:
            continue
        try:
            if geom_type == 7:  # mjGEOM_MESH
                mesh_id = int(raw_model.geom_dataid[geom_id])
                vadr = int(raw_model.mesh_vertadr[mesh_id])
                vcnt = int(raw_model.mesh_vertnum[mesh_id])
                fadr = int(raw_model.mesh_faceadr[mesh_id])
                fcnt = int(raw_model.mesh_facenum[mesh_id])
                verts = np.array(
                    raw_model.mesh_vert[vadr : vadr + vcnt], dtype=np.float64
                )
                faces = np.array(
                    raw_model.mesh_face[fadr : fadr + fcnt], dtype=np.int64
                )
                mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            elif geom_type == 6:  # mjGEOM_BOX
                mesh = trimesh.creation.box(extents=2.0 * size)
            elif geom_type == 2:  # mjGEOM_SPHERE
                mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
            elif geom_type == 4:  # mjGEOM_ELLIPSOID on newer MuJoCo builds
                mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
                mesh.apply_scale(size)
            elif geom_type == 5:  # mjGEOM_CYLINDER
                mesh = trimesh.creation.cylinder(
                    radius=float(size[0]), height=2.0 * float(size[1])
                )
            elif geom_type == 3:  # mjGEOM_CAPSULE
                mesh = trimesh.creation.capsule(
                    radius=float(size[0]), height=2.0 * float(size[1])
                )
            else:
                continue
        except Exception:
            continue
        if mesh.faces.size == 0:
            continue
        try:
            xpos = np.asarray(raw_data.geom_xpos[geom_id], dtype=np.float64)
            xmat = np.asarray(raw_data.geom_xmat[geom_id], dtype=np.float64).reshape(
                3, 3
            )
            rgba = np.asarray(raw_model.geom_rgba[geom_id], dtype=np.float32)
        except Exception:
            continue
        T = np.eye(4)
        T[:3, :3] = xmat
        T[:3, 3] = xpos
        mesh_w = mesh.copy()
        mesh_w.apply_transform(T)
        sampled_geoms.append((mesh_w, rgba))

    if not sampled_geoms:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    sample_counts = _allocate_surface_samples(
        [float(mesh.area) for mesh, _ in sampled_geoms],
        total_surface_points=total_surface_points,
        min_points_per_geom=min_points_per_geom,
    )
    points_world = []
    colors = []
    for (mesh_w, rgba), n in zip(sampled_geoms, sample_counts):
        try:
            pts, _ = trimesh.sample.sample_surface(mesh_w, int(n))
        except Exception:
            pts = np.asarray(
                mesh_w.vertices[: min(int(n), len(mesh_w.vertices))], dtype=np.float64
            )
        if color_override is not None:
            col = np.asarray(color_override, dtype=np.uint8)
        else:
            col = (rgba[:3] * 255).clip(0, 255).astype(np.uint8)
        pts_f = np.asarray(pts, dtype=np.float32)
        points_world.append(pts_f)
        colors.append(np.tile(col, (pts_f.shape[0], 1)).astype(np.uint8))
    return np.concatenate(points_world, axis=0), np.concatenate(colors, axis=0)


def _ee_geom_name_filter(name):
    return (
        name.startswith("gripper0_right_")
        or name.startswith("robot0_link7")
        or "panda_hand" in name
        or "panda_leftfinger" in name
        or "panda_rightfinger" in name
    )


def _ee_geom_ids(env):
    """Return link7 plus the complete hand/finger body subtree."""
    model = env.sim.model
    raw_model = model._model if hasattr(model, "_model") else model
    body_name_to_id = {
        name: int(body_id) for name, body_id in model._body_name2id.items()
    }
    root_body_ids = {
        body_name_to_id[name]
        for name in ("robot0_link7", "robot0_right_hand")
        if name in body_name_to_id
    }
    if not root_body_ids:
        return {
            int(geom_id)
            for name, geom_id in model._geom_name2id.items()
            if _ee_geom_name_filter(name)
        }
    ee_body_ids = set(root_body_ids)
    changed = True
    while changed:
        changed = False
        for body_id in range(1, int(raw_model.nbody)):
            if (
                int(raw_model.body_parentid[body_id]) in ee_body_ids
                and body_id not in ee_body_ids
            ):
                ee_body_ids.add(body_id)
                changed = True
    return {
        int(geom_id)
        for geom_id in range(int(raw_model.ngeom))
        if int(raw_model.geom_bodyid[geom_id]) in ee_body_ids
    }


def _drawer_geom_name_filter(env):
    drawer_prefix = f"{env.drawer.name}_"
    return lambda name: name.startswith(drawer_prefix)


def _scene_geom_name_filter(env):
    drawer_prefix = f"{env.drawer.name}_"

    def _filter(name):
        if not name:
            return False
        if (
            name.startswith("gripper0_")
            or name.startswith("robot0_")
            or name.startswith("mount0_")
        ):
            return False
        if name.startswith(drawer_prefix):
            return False
        return True

    return _filter


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
    precontact_position = projected_ee_position + retreat * float(
        args.precontact_distance
    )
    precontact_rotation = np.asarray(
        demonstration_seed.projected_ee_rotation_world, dtype=np.float64
    ).reshape(3, 3)
    panel = close_demo.get_panel_frame(env)
    solution = mink_solver.solve_precontact_pose(
        env,
        panel,
        robot_state,
        precontact_position,
        precontact_rotation,
        np.asarray(demonstration_seed.arm_q, dtype=np.float64),
        args,
        collision_checker=lambda q_arm: _check_arm_q_collision_for_surface(
            env,
            surface,
            robot_state["robocasa_joint_names"],
            q_arm,
            close_demo._drawer_joint_value(env),
            allowed_ee_geom_name=None,
            penetration_tolerance=float(args.mink_collision_penetration_tolerance),
            collision_scope=str(getattr(args, "mink_collision_scope", "ee")),
        ),
    )
    if bool(getattr(args, "require_mink_precontact", True)):
        if solution.position_error > float(args.mink_position_tolerance):
            raise RuntimeError(
                "mink precontact seed did not reach target: "
                f"position_error={solution.position_error:.6f}, "
                f"tolerance={float(args.mink_position_tolerance):.6f}"
            )
        if not solution.collision_free:
            raise RuntimeError(
                "mink precontact seed is not collision-free: "
                f"{solution.collision_reason}"
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


def _raw_model(model):
    return getattr(model, "_model", model)


def _raw_data(data):
    return getattr(data, "_data", data)


def _object_body_id_for_surface(env, surface):
    model = env.sim.model
    geom_name = str(getattr(surface, "geom_name", ""))
    if geom_name in model._geom_name2id:
        return int(model.geom_bodyid[model.geom_name2id(geom_name)])
    drawer_body = getattr(getattr(env, "drawer", None), "root_body", "")
    if drawer_body and drawer_body in model._body_name2id:
        return int(model.body_name2id(drawer_body))
    return 0


def _linear_pose_nominal(start_pose, target_pose, steps):
    start_pose = np.asarray(start_pose, dtype=np.float64).reshape(7)
    target_pose = np.asarray(target_pose, dtype=np.float64).reshape(7)
    steps = max(int(steps), 2)
    alpha = np.linspace(0.0, 1.0, steps, dtype=np.float64)[:, None]
    poses = (1.0 - alpha) * start_pose + alpha * target_pose
    quat = poses[:, 3:]
    if np.dot(start_pose[3:], target_pose[3:]) < 0.0:
        target_quat = -target_pose[3:]
        poses[:, 3:] = (1.0 - alpha) * start_pose[3:] + alpha * target_quat
        quat = poses[:, 3:]
    poses[:, 3:] = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
    return poses


def _solve_plan_once_mppi_for_q_configs(
    env,
    surface,
    candidates,
    robot_state,
    q_config_solution,
    contact_offset_ee,
    args,
):
    del candidates
    from robocasa.demos.dream import solve_plan_once_mppi_for_q_configs

    return solve_plan_once_mppi_for_q_configs(
        env,
        surface,
        q_config_solution,
        contact_offset_ee,
        args,
        robot_state=robot_state,
    )


def _solve_stage(env, surface, pull_distance, args, stage_name):
    stage_solve_started = time.perf_counter()
    _refresh_scene_visibility(env, args)
    with _suppress_solver_output():
        demonstration_seed = _load_and_project_demonstration_seed(env, args)
    args._demonstration_seed = demonstration_seed
    robot_state = close_demo.get_robot_arm_state(env)
    base_alignment = mink_solver.align_base_to_demo_ee_pose(
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
            waypoint_drawer_q = np.asarray(
                [
                    float(close_demo._drawer_joint_value(env)),
                    float(close_demo._drawer_joint_value(env)),
                    float(close_demo._drawer_joint_value(env))
                    - float(q_config_solution["action_distance"]),
                ],
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
            with _suppress_solver_output():
                plan_once_solution = _solve_plan_once_mppi_for_q_configs(
                    env,
                    surface,
                    candidates,
                    robot_state,
                    q_config_solution,
                    contact_offset_ee,
                    args,
                )
            dream_result = plan_once_solution["result"]
            dream_sequence = np.asarray(
                plan_once_solution["sequence"], dtype=np.float64
            )
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
                "plan_once_mppi_best_cost": float(plan_once_solution["best_cost"]),
                "plan_once_mppi_best_q_index": int(plan_once_solution["best_q_index"]),
                "plan_once_mppi_candidate_count": int(
                    plan_once_solution["candidate_count"]
                ),
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
            _visualize_q_config_mpc_successes(env, q_config_solution, args)
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

    if q_waypoints.size:
        waypoint_names = ("precontact", "contact", "pull")
        target_gripper_poses = [
            (name, pos, rot) for name, (pos, rot) in zip(waypoint_names, waypoint_poses)
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
            ("pull", pull_pose[:3], pull_rotation),
        ]
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
    return stage, [], robot_state


def _set_stage_end_state(env, stage, robot_state):
    solution = stage.mink_solution
    if solution is None or not np.asarray(solution.q_waypoints).size:
        return False
    close_demo._set_env_arm_q(
        env, robot_state["robocasa_joint_names"], solution.q_waypoints[-1]
    )
    close_demo._set_drawer_joint_value(
        env, float(stage.start_drawer_q) - float(stage.pull_distance)
    )
    env.sim.forward()
    return True


def _choose_switch_distance(total_pull_distance, rng, args):
    min_dist = min(float(args.switch_min_open_distance), total_pull_distance)
    max_dist = min(float(args.switch_max_open_distance), total_pull_distance)
    if max_dist <= min_dist:
        return max(total_pull_distance * 0.5, 1e-4)
    return float(rng.uniform(min_dist, max_dist))


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


def _append_drawer_pull_tail(q_traj, segments, stages, tail_steps):
    """Append a pull tail to a cuRobo plan that did not include 'pull'.

    The arm-joint trajectory is a linear interpolation from the last cuRobo
    output (end of `contact`) to the mink-IK arm-q solved for the pull
    waypoint. The drawer DOF is then linearly interpolated from
    `start_drawer_q` to `start_drawer_q - pull_distance` over the same
    number of steps (handled by `_drawer_trajectory_from_curobo_segments`,
    which reads `phase == "pull"`).
    """
    if q_traj is None:
        return q_traj, segments
    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    if arm_traj.shape[0] == 0 or not stages:
        return q_traj, segments
    new_segments = list(segments)
    appended_chunks = [arm_traj]
    steps_each = max(int(tail_steps), 2)
    for stage in stages:
        sol = stage.mink_solution
        if sol is None:
            continue
        q_wp = np.asarray(sol.q_waypoints, dtype=np.float64).reshape(-1, 7)
        if q_wp.shape[0] < 1:
            continue
        # mink waypoints were built as [precontact, contact, pull]; the last
        # entry is the pull arm-q.
        pull_arm_q = q_wp[-1]
        start_arm_q = appended_chunks[-1][-1]
        alphas = np.linspace(0.0, 1.0, steps_each, dtype=np.float64)[1:, None]
        tail = (1.0 - alphas) * start_arm_q[None, :] + alphas * pull_arm_q[None, :]
        appended_chunks.append(tail)
        new_segments.append(
            {
                "name": f"{stage.name}:pull",
                "steps": int(tail.shape[0]),
                "synthesized": True,
            }
        )
    return np.concatenate(appended_chunks, axis=0), new_segments


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


def _validate_joint_space_trajectory_collisions(env, stages, q_traj, segments, args):
    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    drawer_q, stage_indices = _drawer_trajectory_from_curobo_segments(
        stages,
        segments,
        arm_traj,
    )
    arm_joint_names = tuple(env.robots[0].robot_model.joints[:7])
    failures = []
    for step_index, (q_arm, drawer_value, stage_index) in enumerate(
        zip(arm_traj, drawer_q, stage_indices)
    ):
        stage = stages[int(np.clip(stage_index, 0, len(stages) - 1))]
        surface = _current_surface_for_stage(env, stage)
        ok, reason = _check_arm_q_collision_for_surface(
            env,
            surface,
            arm_joint_names,
            q_arm,
            float(drawer_value),
            allowed_ee_geom_name=str(stage.mink_solution.ee_contact_geom_name),
            penetration_tolerance=float(args.mink_collision_penetration_tolerance),
            collision_scope=str(args.mink_collision_scope),
        )
        if not ok:
            failures.append(
                {
                    "step": int(step_index),
                    "stage": str(stage.name),
                    "drawer_q": float(drawer_value),
                    "reason": str(reason),
                }
            )
            if len(failures) >= int(getattr(args, "joint_validation_max_failures", 8)):
                break
    return {
        "valid": not failures,
        "evaluated_steps": int(arm_traj.shape[0]),
        "failures": failures,
    }


def _validate_curobo_reaches_gripper_targets(
    env,
    robot_state,
    target_gripper_poses,
    q_traj,
    segments,
    args,
):
    """Check cuRobo joint output against robosuite grip-site targets in MuJoCo FK."""
    if q_traj is None or not np.asarray(q_traj).size:
        return []
    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    if not segments:
        return []

    model = env.sim.model
    data = env.sim.data
    frame_name = str(getattr(args, "mink_contact_frame", "gripper0_right_grip_site"))
    if frame_name not in model._site_name2id:
        raise RuntimeError(
            f"Cannot validate cuRobo targets: site '{frame_name}' not found."
        )
    site_id = model.site_name2id(frame_name)
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    diagnostics = []
    cumulative_steps = 0
    try:
        for (target_name, target_pos, target_rot), segment in zip(
            target_gripper_poses,
            segments,
        ):
            steps = max(int(segment.get("steps", 0)), 0)
            if steps <= 0:
                continue
            cumulative_steps += steps
            q_index = min(cumulative_steps - 1, arm_traj.shape[0] - 1)
            close_demo._set_env_arm_q(
                env,
                robot_state["robocasa_joint_names"],
                arm_traj[q_index],
            )
            actual_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
            actual_rot = (
                np.asarray(data.site_xmat[site_id], dtype=np.float64)
                .reshape(3, 3)
                .copy()
            )
            position_error = float(
                np.linalg.norm(
                    actual_pos - np.asarray(target_pos, dtype=np.float64).reshape(3)
                )
            )
            rotation_error = _rotation_angle_error(actual_rot, target_rot)
            diagnostics.append(
                {
                    "name": str(target_name),
                    "segment_name": str(segment.get("name", "")),
                    "q_index": int(q_index),
                    "position_error": position_error,
                    "rotation_error": rotation_error,
                    "target_pos": np.asarray(target_pos, dtype=np.float64).tolist(),
                    "actual_pos": actual_pos.tolist(),
                }
            )
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        env.sim.forward()

    max_pos_error = max((entry["position_error"] for entry in diagnostics), default=0.0)
    max_rot_error = max((entry["rotation_error"] for entry in diagnostics), default=0.0)
    args._curobo_target_diagnostics = diagnostics
    if max_pos_error > float(
        args.curobo_target_position_tolerance
    ) or max_rot_error > float(args.curobo_target_rotation_tolerance):
        worst = max(
            diagnostics,
            key=lambda entry: (
                float(entry["position_error"]),
                float(entry["rotation_error"]),
            ),
        )
        raise RuntimeError(
            "cuRobo reported success, but the MuJoCo grip site does not reach "
            "the requested MPC target. "
            f"worst_segment='{worst['name']}', "
            f"position_error={worst['position_error']:.6f} m, "
            f"rotation_error={worst['rotation_error']:.6f} rad. "
            "This indicates a frame or joint-order mismatch between cuRobo and robosuite."
        )
    return diagnostics


def _interpolate_segment(start, goal, steps):
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    steps = max(int(steps), 2)
    return np.asarray(
        [
            (1.0 - alpha) * start + alpha * goal
            for alpha in np.linspace(0.0, 1.0, steps, endpoint=False)
        ],
        dtype=np.float64,
    )


def _stage_index_by_name(stages):
    return {stage.name: index for index, stage in enumerate(stages)}


def _drawer_trajectory_from_curobo_segments(stages, segments, q_traj):
    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    if not segments:
        return np.full(
            arm_traj.shape[0], float(stages[0].start_drawer_q), dtype=np.float64
        ), np.zeros(arm_traj.shape[0], dtype=np.int64)
    stage_by_name = _stage_index_by_name(stages)
    drawer_values = []
    stage_indices = []
    current_drawer_q = float(stages[0].start_drawer_q)
    for segment in segments:
        steps = max(int(segment.get("steps", 0)), 0)
        if steps <= 0:
            continue
        name = str(segment.get("name", ""))
        stage_name, _, phase = name.partition(":")
        stage_index = stage_by_name.get(
            stage_name, min(len(stages) - 1, len(stage_indices))
        )
        stage = stages[stage_index]
        start_q = float(stage.start_drawer_q)
        end_q = float(stage.start_drawer_q - stage.pull_distance)
        if phase == "pull":
            segment_drawer = np.linspace(start_q, end_q, steps, dtype=np.float64)
            current_drawer_q = end_q
        else:
            segment_drawer = np.full(
                steps,
                start_q if phase in ("precontact", "contact") else current_drawer_q,
            )
        drawer_values.extend(float(value) for value in segment_drawer)
        stage_indices.extend([stage_index] * steps)
    if len(drawer_values) < arm_traj.shape[0]:
        drawer_values.extend(
            [current_drawer_q] * (arm_traj.shape[0] - len(drawer_values))
        )
        stage_indices.extend(
            [len(stages) - 1] * (arm_traj.shape[0] - len(stage_indices))
        )
    drawer_values = np.asarray(drawer_values[: arm_traj.shape[0]], dtype=np.float64)
    stage_indices = np.asarray(stage_indices[: arm_traj.shape[0]], dtype=np.int64)
    return drawer_values, stage_indices


def _play_open_trajectory_viewer(env, stages, q_traj, segments, robot_state, args):
    if not bool(getattr(args, "execution_viewer", True)):
        return
    if q_traj is None or not np.asarray(q_traj).size:
        print("[execution_viewer] skipped: no arm trajectory to play", flush=True)
        return
    try:
        import mujoco
        import mujoco.viewer
    except Exception as exc:
        print(
            f"[execution_viewer] skipped: failed to import mujoco.viewer: {exc}",
            flush=True,
        )
        return

    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    try:
        from robocasa.demos import visualize_mujoco as viz_mj

        ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(
            env, str(args.mink_contact_frame)
        )
    except Exception as exc:
        ghost_geoms = ()
        print(f"[execution_viewer] ghost EE overlay disabled: {exc}", flush=True)
    ghost_poses = []
    if ghost_geoms:
        try:
            site_id = env.sim.model.site_name2id(str(args.mink_contact_frame))
            for stage_index, stage in enumerate(stages):
                q_waypoints = np.asarray(
                    stage.mink_solution.q_waypoints, dtype=np.float64
                ).reshape(-1, 7)
                for waypoint_index, q_arm in enumerate(q_waypoints):
                    close_demo._set_env_arm_q(env, arm_joint_names, q_arm)
                    if waypoint_index == q_waypoints.shape[0] - 1:
                        close_demo._set_drawer_joint_value(
                            env,
                            float(stage.start_drawer_q) - float(stage.pull_distance),
                        )
                    else:
                        close_demo._set_drawer_joint_value(
                            env, float(stage.start_drawer_q)
                        )
                    env.sim.forward()
                    ghost_poses.append(
                        (
                            stage_index,
                            waypoint_index,
                            np.asarray(
                                env.sim.data.site_xpos[site_id], dtype=np.float64
                            ).copy(),
                            np.asarray(
                                env.sim.data.site_xmat[site_id], dtype=np.float64
                            )
                            .reshape(3, 3)
                            .copy(),
                        )
                    )
        finally:
            env.sim.data.qpos[:] = qpos_saved
            env.sim.data.qvel[:] = qvel_saved
            env.sim.forward()
    frame_dt = 1.0 / max(float(getattr(args, "execution_viewer_fps", 30.0)), 1.0)
    sim_dt = float(getattr(raw_model, "opt", raw_model).timestep)
    substeps = max(
        int(getattr(args, "execution_viewer_pd_substeps", 0) or 0),
        int(np.ceil(frame_dt / max(sim_dt, 1e-6))),
        1,
    )
    kp = float(getattr(args, "execution_viewer_pd_kp", 350.0))
    kd = float(getattr(args, "execution_viewer_pd_kd", 35.0))
    loop = bool(getattr(args, "execution_viewer_loop", False))
    qpos_addrs = np.asarray(
        [env.sim.model.get_joint_qpos_addr(name) for name in arm_joint_names],
        dtype=np.int64,
    )
    dof_addrs = np.asarray(
        [int(raw_model.joint(name).dofadr[0]) for name in arm_joint_names],
        dtype=np.int64,
    )
    print(
        f"[execution_viewer] playing {arm_traj.shape[0]} frames with PD control; "
        f"planner={str(getattr(args, '_execution_planner_source', 'unknown'))}; "
        "close the MuJoCo window to finish",
        flush=True,
    )
    try:
        env.sim.data.qpos[qpos_addrs] = arm_traj[0]
        env.sim.data.qvel[dof_addrs] = 0.0
        close_demo._set_drawer_joint_value(env, float(stages[0].start_drawer_q))
        env.sim.forward()
        lookat = np.asarray(stages[0].selected_contact_world, dtype=np.float64)
        with mujoco.viewer.launch_passive(raw_model, raw_data) as viewer:
            _configure_native_viewer_camera(viewer, lookat)
            ghost_rgba = np.asarray(
                [
                    float(getattr(args, "execution_viewer_ghost_r", 0.05)),
                    float(getattr(args, "execution_viewer_ghost_g", 0.45)),
                    float(getattr(args, "execution_viewer_ghost_b", 1.0)),
                    float(getattr(args, "execution_viewer_ghost_alpha", 0.26)),
                ],
                dtype=np.float32,
            )
            while viewer.is_running():
                for q_des in arm_traj:
                    if not viewer.is_running():
                        break
                    for _ in range(substeps):
                        q = np.asarray(env.sim.data.qpos[qpos_addrs], dtype=np.float64)
                        qd = np.asarray(env.sim.data.qvel[dof_addrs], dtype=np.float64)
                        raw_data.qfrc_applied[:] = 0.0
                        # print("pd force  ", np.max(np.abs(q_des - q)), np.max(np.abs(raw_data.qfrc_applied[dof_addrs])))
                        raw_data.qfrc_applied[dof_addrs] = kp * (q_des - q) - kd * qd
                        # print("raw_data.qfrc_actuator[dof_addrs] = ", raw_data.qfrc_actuator[dof_addrs])
                        mujoco.mj_step(raw_model, raw_data)
                    if hasattr(viewer, "user_scn") and ghost_geoms:
                        viewer.user_scn.ngeom = 0
                        for (
                            _stage_index,
                            waypoint_index,
                            target_pos,
                            target_rot,
                        ) in ghost_poses:
                            rgba = ghost_rgba.copy()
                            if waypoint_index == 1:
                                rgba[:3] = np.asarray(
                                    [1.0, 0.62, 0.05], dtype=np.float32
                                )
                            elif waypoint_index >= 2:
                                rgba[:3] = np.asarray(
                                    [0.05, 0.75, 0.35], dtype=np.float32
                                )
                            for ghost in ghost_geoms:
                                viz_mj._add_ghost_geom(
                                    viewer.user_scn,
                                    ghost,
                                    target_pos,
                                    target_rot,
                                    rgba,
                                )
                    viewer.sync()
                if not loop:
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(frame_dt)
                    break
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()


def _current_surface_for_stage(env, stage):
    panel = close_demo.get_panel_frame(env)
    if stage.surface_name == "handle":
        return _make_handle_inner_surface(env, panel)
    return _make_panel_inner_surface(env, panel)


def _diagnose_current_stage_contact(env, stage, args):
    model = env.sim.model
    data = env.sim.data
    surface = _current_surface_for_stage(env, stage)
    expected_contact_world = surface.center_world + surface.rotation_world @ np.asarray(
        stage.selected_contact_local, dtype=np.float64
    )
    frame_name = str(args.mink_contact_frame)
    ee_contact_point = np.full(3, np.nan, dtype=np.float64)
    ee_contact_point_error = float("inf")
    if frame_name in model._site_name2id:
        site_id = model.site_name2id(frame_name)
        site_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        site_rot = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        ee_contact_point = site_pos + site_rot @ np.asarray(
            stage.mink_solution.contact_offset_local,
            dtype=np.float64,
        )
        ee_contact_point_error = float(
            np.linalg.norm(ee_contact_point - expected_contact_world)
        )

    _, ee_geom_ids, target_geom_ids = _robot_contact_geom_sets_for_surface(env, surface)
    best_contact_distance = float("inf")
    best_contact_point_error = float("inf")
    best_contact_world = np.full(3, np.nan, dtype=np.float64)
    geom_id_to_name = {
        int(geom_id): name for name, geom_id in model._geom_name2id.items()
    }
    best_contact_geom_name = ""
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        pair_matches = (geom1 in ee_geom_ids and geom2 in target_geom_ids) or (
            geom2 in ee_geom_ids and geom1 in target_geom_ids
        )
        if not pair_matches:
            continue
        contact_world = np.asarray(contact.pos, dtype=np.float64).copy()
        point_error = float(np.linalg.norm(contact_world - expected_contact_world))
        contact_distance = float(contact.dist)
        rank = (point_error, abs(contact_distance))
        best_rank = (best_contact_point_error, abs(best_contact_distance))
        if rank < best_rank:
            contact_ee_geom_id = geom1 if geom1 in ee_geom_ids else geom2
            best_contact_distance = contact_distance
            best_contact_point_error = point_error
            best_contact_world = contact_world
            best_contact_geom_name = geom_id_to_name.get(
                int(contact_ee_geom_id),
                str(contact_ee_geom_id),
            )
    actual_contact = bool(
        np.isfinite(best_contact_distance)
        and best_contact_distance <= float(args.mink_actual_contact_max_distance)
    )
    actual_contact_near_selected = bool(
        actual_contact
        and best_contact_point_error <= float(args.mink_actual_contact_point_tolerance)
    )
    return {
        "expected_contact_world": expected_contact_world,
        "ee_contact_point_world": ee_contact_point,
        "ee_contact_point_error": ee_contact_point_error,
        "actual_contact": actual_contact,
        "actual_contact_near_selected": actual_contact_near_selected,
        "actual_contact_distance": best_contact_distance,
        "actual_contact_point_error": best_contact_point_error,
        "actual_contact_world": best_contact_world,
        "actual_contact_geom_name": best_contact_geom_name,
    }


def _diagnose_curobo_trajectory_contacts(env, stages, q_traj, segments, args):
    if q_traj is None or not np.asarray(q_traj).size:
        return {
            "summary": {
                "evaluated_steps": 0,
                "actual_contact_step_count": 0,
                "actual_contact_near_selected_step_count": 0,
                "actual_contact_fraction": 0.0,
                "actual_contact_near_selected_fraction": 0.0,
                "ee_contact_point_error_min": float("inf"),
                "ee_contact_point_error_median": float("inf"),
                "ee_contact_point_error_max": float("inf"),
            },
            "stage_indices": np.zeros(0, dtype=np.int64),
            "drawer_q": np.zeros(0, dtype=np.float64),
            "ee_contact_point_error": np.zeros(0, dtype=np.float64),
            "actual_contact": np.zeros(0, dtype=bool),
            "actual_contact_near_selected": np.zeros(0, dtype=bool),
            "actual_contact_distance": np.zeros(0, dtype=np.float64),
            "actual_contact_point_error": np.zeros(0, dtype=np.float64),
        }

    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    drawer_q, stage_indices = _drawer_trajectory_from_curobo_segments(
        stages,
        segments,
        arm_traj,
    )
    data = env.sim.data
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    arm_joint_names = tuple(env.robots[0].robot_model.joints[:7])
    ee_errors = []
    actual_contacts = []
    actual_near = []
    actual_distances = []
    actual_point_errors = []
    try:
        for q_arm, drawer_value, stage_index in zip(arm_traj, drawer_q, stage_indices):
            close_demo._set_env_arm_q(env, arm_joint_names, q_arm)
            close_demo._set_drawer_joint_value(env, float(drawer_value))
            env.sim.forward()
            stage = stages[int(np.clip(stage_index, 0, len(stages) - 1))]
            diag = _diagnose_current_stage_contact(env, stage, args)
            ee_errors.append(float(diag["ee_contact_point_error"]))
            actual_contacts.append(bool(diag["actual_contact"]))
            actual_near.append(bool(diag["actual_contact_near_selected"]))
            actual_distances.append(float(diag["actual_contact_distance"]))
            actual_point_errors.append(float(diag["actual_contact_point_error"]))
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        env.sim.forward()

    ee_errors = np.asarray(ee_errors, dtype=np.float64)
    actual_contacts = np.asarray(actual_contacts, dtype=bool)
    actual_near = np.asarray(actual_near, dtype=bool)
    actual_distances = np.asarray(actual_distances, dtype=np.float64)
    actual_point_errors = np.asarray(actual_point_errors, dtype=np.float64)
    error_summary = _distance_summary(ee_errors)
    evaluated_steps = int(arm_traj.shape[0])
    summary = {
        "evaluated_steps": evaluated_steps,
        "actual_contact_step_count": int(actual_contacts.sum()),
        "actual_contact_near_selected_step_count": int(actual_near.sum()),
        "actual_contact_fraction": float(actual_contacts.mean())
        if evaluated_steps
        else 0.0,
        "actual_contact_near_selected_fraction": float(actual_near.mean())
        if evaluated_steps
        else 0.0,
        "ee_contact_point_error_min": float(error_summary["min"]),
        "ee_contact_point_error_median": float(error_summary["median"]),
        "ee_contact_point_error_max": float(error_summary["max"]),
        "actual_contact_distance_min": float(
            _distance_summary(actual_distances)["min"]
        ),
        "actual_contact_point_error_min": float(
            _distance_summary(actual_point_errors)["min"]
        ),
    }
    return {
        "summary": summary,
        "stage_indices": stage_indices,
        "drawer_q": drawer_q,
        "ee_contact_point_error": ee_errors,
        "actual_contact": actual_contacts,
        "actual_contact_near_selected": actual_near,
        "actual_contact_distance": actual_distances,
        "actual_contact_point_error": actual_point_errors,
    }


def _save_open_outputs(
    path,
    env,
    stages,
    target_hand_poses,
    q_traj,
    segments,
    should_switch,
    switch_distance,
    args,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene_point_cloud = getattr(env, "_scene_point_cloud", None)
    scene_summary = (
        scene_point_cloud.summary() if scene_point_cloud is not None else None
    )
    curobo_contact = _diagnose_curobo_trajectory_contacts(
        env,
        stages,
        q_traj,
        segments,
        args,
    )
    metadata = {
        "task": "OpenDrawer",
        "drawer_name": env.drawer.name,
        "layout_id": int(env.layout_id)
        if hasattr(env, "layout_id") and not isinstance(env.layout_id, dict)
        else str(getattr(env, "layout_id", "")),
        "style_id": int(env.style_id)
        if hasattr(env, "style_id") and not isinstance(env.style_id, dict)
        else str(getattr(env, "style_id", "")),
        "drawer_size": np.asarray(env.drawer.size, dtype=np.float64).tolist(),
        "drawer_state": {
            k: float(v) for k, v in env.drawer.get_door_state(env).items()
        },
        "should_switch_to_panel_inner": bool(should_switch),
        "switch_distance": None if switch_distance is None else float(switch_distance),
        "planned_total_pull_distance": float(
            sum(stage.pull_distance for stage in stages)
        ),
        "execution_planner_source": str(
            getattr(args, "_execution_planner_source", "none")
        ),
        "segments": segments,
        "curobo_target_diagnostics": getattr(args, "_curobo_target_diagnostics", []),
        "joint_space_validation": getattr(args, "_joint_space_validation", None),
        "scene_processing": scene_summary,
        "curobo_contact_diagnostics": curobo_contact["summary"],
        "stages": [
            {
                "name": stage.name,
                "surface": stage.surface_name,
                "start_drawer_q": float(stage.start_drawer_q),
                "end_drawer_q": float(stage.start_drawer_q - stage.pull_distance),
                "pull_distance": float(stage.pull_distance),
                "has_executable_arm_trajectory": bool(
                    np.asarray(stage.mink_solution.q_waypoints).size
                ),
                "selected_contact_index": int(stage.selected_contact_index),
                "selected_contact_face": str(
                    getattr(
                        stage.candidates[stage.selected_contact_index],
                        "contact_face",
                        "",
                    )
                ),
                "selected_contact_world": stage.selected_contact_world.tolist(),
                "selected_contact_local": stage.selected_contact_local.tolist(),
                "selected_contact_cost": float(stage.selected_contact_cost),
                "gripper_mode": str(getattr(stage, "gripper_mode", "")),
                "dream_device": str(getattr(stage, "dream_device", "")),
                "contact_geometry_cache": str(
                    getattr(stage, "contact_geometry_cache", "")
                ),
                "feasible_graph_edge_count": int(
                    np.asarray(
                        getattr(
                            stage,
                            "feasible_graph_edges",
                            np.zeros((0, 2), dtype=np.int64),
                        )
                    ).shape[0]
                ),
                "representative_point_count": int(
                    np.asarray(
                        getattr(
                            stage,
                            "object_representative_points_world",
                            np.zeros((0, 3), dtype=np.float64),
                        )
                    ).shape[0]
                ),
                "curobo_collision_sphere_count": int(
                    np.asarray(
                        getattr(
                            stage,
                            "curobo_collision_sphere_radii",
                            np.zeros(0, dtype=np.float64),
                        )
                    ).size
                ),
                "selected_curobo_sphere_index": int(
                    getattr(stage, "selected_curobo_sphere_index", -1)
                ),
                "selected_curobo_sphere_link": (
                    str(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "selected_sphere_link", ""
                        )
                    )
                ),
                "minimum_curobo_sphere_sdf": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "minimum_sphere_sdf", float("nan")
                    )
                ),
                "selected_contact_set_candidate_indices": np.asarray(
                    getattr(
                        stage,
                        "selected_contact_set_candidate_indices",
                        np.zeros(0, dtype=np.int64),
                    ),
                    dtype=np.int64,
                ).tolist(),
                "candidate_count": len(stage.candidates),
                "feasible_candidate_count": int(
                    sum(candidate.feasible for candidate in stage.candidates)
                ),
                "visible_candidate_count": int(
                    sum(
                        bool(getattr(candidate, "visible", True))
                        for candidate in stage.candidates
                    )
                ),
                "selected_contact_visible": bool(
                    getattr(
                        stage.candidates[stage.selected_contact_index], "visible", True
                    )
                ),
                "selected_scene_link": str(
                    getattr(
                        stage.candidates[stage.selected_contact_index],
                        "scene_link_name",
                        "",
                    )
                ),
                "selected_scene_point_index": int(
                    getattr(
                        stage.candidates[stage.selected_contact_index],
                        "scene_point_index",
                        -1,
                    )
                ),
                "selected_scene_point_distance": float(
                    getattr(
                        stage.candidates[stage.selected_contact_index],
                        "scene_point_distance",
                        float("nan"),
                    )
                ),
                "selected_surface_projection_distance": float(
                    getattr(
                        stage.candidates[stage.selected_contact_index],
                        "surface_projection_distance",
                        float("nan"),
                    )
                ),
                "projected_seed_contact_world": getattr(
                    stage,
                    "projected_seed_contact_world",
                    np.full(3, np.nan, dtype=np.float64),
                ).tolist(),
                "projected_seed_contact_offset_error": float(
                    getattr(stage, "projected_seed_contact_offset_error", float("nan"))
                ),
                "projected_seed_contact_feasible_distance": float(
                    getattr(
                        stage, "projected_seed_contact_feasible_distance", float("nan")
                    )
                ),
                "projected_seed_contact_feasible_index": int(
                    getattr(stage, "projected_seed_contact_feasible_index", -1)
                ),
                "initial_pose_source": "demonstration_base_aligned_mink_precontact",
                "base_alignment": {
                    "applied": bool(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_applied", False
                        )
                    ),
                    "reason": str(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_reason", ""
                        )
                    ),
                    "yaw_delta": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_yaw_delta", float("nan")
                        )
                    ),
                    "translation_delta": list(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_translation_delta", []
                        )
                    ),
                    "initial_position_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_initial_position_error", float("nan")
                        )
                    ),
                    "final_position_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_final_position_error", float("nan")
                        )
                    ),
                    "initial_rotation_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_initial_rotation_error", float("nan")
                        )
                    ),
                    "final_rotation_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_final_rotation_error", float("nan")
                        )
                    ),
                },
                "mink_precontact": {
                    "candidate_index": int(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "precontact_mink_candidate_index", -1
                        )
                    ),
                    "position_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "precontact_mink_position_error", float("nan")
                        )
                    ),
                    "rotation_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "precontact_mink_rotation_error", float("nan")
                        )
                    ),
                    "collision_free": bool(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "precontact_mink_collision_free", False
                        )
                    ),
                    "collision_reason": str(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "precontact_mink_collision_reason", ""
                        )
                    ),
                },
                "mink_contact_pose": {
                    "contact_frame": stage.mink_solution.contact_frame,
                    "ee_sample_index": int(stage.mink_solution.ee_sample_index),
                    "ee_sample_name": stage.mink_solution.ee_sample_name,
                    "ee_contact_geom_name": stage.mink_solution.ee_contact_geom_name,
                    "contact_offset_local": np.asarray(
                        stage.mink_solution.contact_offset_local,
                        dtype=np.float64,
                    ).tolist(),
                    "contact_position_error": float(
                        stage.mink_solution.contact_position_error
                    ),
                    "collision_free": bool(stage.mink_solution.collision_free),
                    "ee_sample_point_error": float(
                        getattr(
                            stage.mink_solution, "ee_sample_point_error", float("nan")
                        )
                    ),
                    "actual_contact": bool(
                        getattr(stage.mink_solution, "actual_contact", False)
                    ),
                    "actual_contact_near_candidate": bool(
                        getattr(
                            stage.mink_solution, "actual_contact_near_candidate", False
                        )
                    ),
                    "actual_contact_distance": float(
                        getattr(
                            stage.mink_solution, "actual_contact_distance", float("nan")
                        )
                    ),
                    "actual_contact_point_error": float(
                        getattr(
                            stage.mink_solution,
                            "actual_contact_point_error",
                            float("nan"),
                        )
                    ),
                    "actual_contact_geom_name": str(
                        getattr(stage.mink_solution, "actual_contact_geom_name", "")
                    ),
                    "selected_ee_geom_contact": bool(
                        getattr(stage.mink_solution, "selected_ee_geom_contact", False)
                    ),
                    "contact_standoff": float(
                        getattr(stage.mink_solution, "contact_standoff", float("nan"))
                    ),
                    "roll_angle": float(stage.mink_solution.roll_angle),
                },
                "demonstration_seed": {
                    "dataset": str(stage.demonstration_seed.dataset),
                    "episode_name": str(stage.demonstration_seed.episode_name),
                    "frame_index": int(stage.demonstration_seed.frame_index),
                    "arm_q": stage.demonstration_seed.arm_q.tolist(),
                    "ee_position_object": stage.demonstration_seed.ee_position_object.tolist(),
                    "ee_rotation_object": stage.demonstration_seed.ee_rotation_object.tolist(),
                    "contact_position_object": stage.demonstration_seed.contact_position_object.tolist(),
                    "contact_offset_ee": stage.demonstration_seed.contact_offset_ee.tolist(),
                    "source_position_roundtrip_error": float(
                        stage.demonstration_seed.source_position_roundtrip_error
                    ),
                    "source_rotation_roundtrip_error": float(
                        stage.demonstration_seed.source_rotation_roundtrip_error
                    ),
                    "projected_position_roundtrip_error": float(
                        stage.demonstration_seed.projected_position_roundtrip_error
                    ),
                    "projected_rotation_roundtrip_error": float(
                        stage.demonstration_seed.projected_rotation_roundtrip_error
                    ),
                },
                "dream_initial_pose_count": int(stage.dream_initial_poses.shape[0]),
                "dream_initial_candidate_indices": getattr(
                    stage,
                    "dream_initial_candidate_indices",
                    np.zeros(0, dtype=np.int64),
                ).tolist(),
                "dream_initial_feasible_contact_count": int(
                    np.asarray(
                        getattr(
                            stage, "dream_initial_contact_mask", np.zeros(0, dtype=bool)
                        ),
                        dtype=bool,
                    ).sum()
                ),
                "dream_initial_feasible_contact_fraction": float(
                    getattr(stage, "dream_initial_contact_fraction", float("nan"))
                ),
                "dream_initial_contact_distance_min": float(
                    np.nanmin(
                        np.asarray(
                            getattr(stage, "dream_initial_contact_distances", [np.nan]),
                            dtype=np.float64,
                        )
                    )
                ),
                "dream_initial_contact_distance_median": float(
                    np.nanmedian(
                        np.asarray(
                            getattr(stage, "dream_initial_contact_distances", [np.nan]),
                            dtype=np.float64,
                        )
                    )
                ),
                "dream_contact_target_fraction": float(
                    getattr(stage, "dream_contact_target_fraction", float("nan"))
                ),
                "dream_contact_target_distance_first": float(
                    _array_value_or_nan(
                        getattr(stage, "dream_contact_target_distances", np.zeros(0)),
                        0,
                    )
                ),
                "dream_contact_target_distance_terminal": float(
                    _array_value_or_nan(
                        getattr(stage, "dream_contact_target_distances", np.zeros(0)),
                        -1,
                    )
                ),
                "dream_contact_target_distance_max": float(
                    _distance_summary(
                        getattr(
                            stage,
                            "dream_contact_target_distances",
                            np.asarray([np.nan]),
                        )
                    )["max"]
                ),
                "dream_first_feasible_contact_distance": float(
                    getattr(
                        stage, "dream_first_feasible_contact_distance", float("nan")
                    )
                ),
                "dream_first_feasible_contact_index": int(
                    getattr(stage, "dream_first_feasible_contact_index", -1)
                ),
                "dream_target_snap_applied": bool(
                    getattr(stage, "dream_target_snap_applied", False)
                ),
                "target_contact_distances": np.asarray(
                    getattr(
                        stage, "target_contact_distances", np.zeros(0, dtype=np.float64)
                    ),
                    dtype=np.float64,
                ).tolist(),
                "target_contact_mask": np.asarray(
                    getattr(stage, "target_contact_mask", np.zeros(0, dtype=bool)),
                    dtype=bool,
                ).tolist(),
                "dream_success_count": (
                    int(
                        bool(
                            getattr(stage, "dream_diagnostics", {}).get(
                                "drawer_open_success", False
                            )
                        )
                    )
                ),
                "dream_drawer_open_success": bool(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "drawer_open_success", False
                    )
                ),
                "dream_terminal_drawer_q": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "terminal_drawer_q", float("nan")
                    )
                ),
                "dream_terminal_open_distance": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "terminal_open_distance", float("nan")
                    )
                ),
                "dream_candidate_open_success_count": int(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "candidate_open_success_count", 0
                    )
                ),
                "dream_candidate_open_success_fraction": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "candidate_open_success_fraction", float("nan")
                    )
                ),
                "dream_candidate_open_distance_median": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "candidate_open_distance_median", float("nan")
                    )
                ),
                "dream_candidate_open_distance_max": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "candidate_open_distance_max", float("nan")
                    )
                ),
                "dream_candidate_contact_target_fraction": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "candidate_contact_target_fraction", float("nan")
                    )
                ),
                "dream_best_cost": (
                    float(stage.dream_result.best_cost)
                    if stage.dream_result is not None
                    else float("nan")
                ),
                "dream_best_index": (
                    int(stage.dream_result.best_index)
                    if stage.dream_result is not None
                    else -1
                ),
                "dream_initial_pose_error_min": (
                    float(
                        np.nanmin(
                            stage.dream_result.initial_pose_errors.detach()
                            .cpu()
                            .numpy()
                        )
                    )
                    if stage.dream_result is not None
                    and stage.dream_result.initial_pose_errors is not None
                    else float("nan")
                ),
                "dream_initial_position_error_min": (
                    float(
                        np.nanmin(
                            stage.dream_result.initial_position_errors.detach()
                            .cpu()
                            .numpy()
                        )
                    )
                    if stage.dream_result is not None
                    and stage.dream_result.initial_position_errors is not None
                    else float("nan")
                ),
                "dream_initial_rotation_error_min": (
                    float(
                        np.nanmin(
                            stage.dream_result.initial_rotation_errors.detach()
                            .cpu()
                            .numpy()
                        )
                    )
                    if stage.dream_result is not None
                    and stage.dream_result.initial_rotation_errors is not None
                    else float("nan")
                ),
                "dream_failure_reason": str(
                    getattr(stage, "dream_failure_detail", {}).get("reason", "unknown")
                ),
                "dream_failure_detail": getattr(stage, "dream_failure_detail", {}),
                "mink_attempt_reason_counts": _report_reason_counts(stage.mink_reports),
            }
            for stage in stages
        ],
    }

    np.savez(
        path,
        q_traj=np.asarray(q_traj, dtype=np.float64)
        if q_traj is not None
        else np.zeros((0, 7)),
        target_hand_pos_base=np.asarray(
            [pose[1] for pose in target_hand_poses], dtype=np.float64
        ),
        target_hand_quat_wxyz_base=np.asarray(
            [close_demo._quat_wxyz_from_matrix(pose[2]) for pose in target_hand_poses],
            dtype=np.float64,
        ),
        stage_names=np.asarray([stage.name for stage in stages]),
        stage_surfaces=np.asarray([stage.surface_name for stage in stages]),
        stage_start_drawer_q=np.asarray(
            [stage.start_drawer_q for stage in stages], dtype=np.float64
        ),
        stage_pull_distance=np.asarray(
            [stage.pull_distance for stage in stages], dtype=np.float64
        ),
        selected_contact_world=np.asarray(
            [stage.selected_contact_world for stage in stages], dtype=np.float64
        ),
        selected_contact_local=np.asarray(
            [stage.selected_contact_local for stage in stages], dtype=np.float64
        ),
        selected_contact_cost=np.asarray(
            [stage.selected_contact_cost for stage in stages], dtype=np.float64
        ),
        selected_contact_offset_ee=np.asarray(
            [
                np.asarray(
                    stage.mink_solution.contact_offset_local,
                    dtype=np.float64,
                )
                for stage in stages
            ],
            dtype=np.float64,
        ),
        selected_curobo_sphere_index=np.asarray(
            [getattr(stage, "selected_curobo_sphere_index", -1) for stage in stages],
            dtype=np.int64,
        ),
        curobo_collision_sphere_centers_ee=np.asarray(
            [
                getattr(
                    stage,
                    "curobo_collision_sphere_centers_ee",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        curobo_collision_sphere_radii=np.asarray(
            [
                getattr(
                    stage,
                    "curobo_collision_sphere_radii",
                    np.zeros(0, dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        curobo_collision_sphere_links=np.asarray(
            [
                np.asarray(
                    getattr(
                        stage,
                        "curobo_collision_sphere_links",
                        tuple(),
                    ),
                    dtype=str,
                )
                for stage in stages
            ],
            dtype=object,
        ),
        curobo_sphere_sdf=np.asarray(
            [
                getattr(
                    stage,
                    "curobo_sphere_sdf",
                    np.zeros(0, dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        object_representative_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "object_representative_points_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        object_representative_normals_world=np.asarray(
            [
                getattr(
                    stage,
                    "object_representative_normals_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        mink_q_waypoints=np.asarray(
            [
                stage.mink_solution.q_waypoints
                if np.asarray(stage.mink_solution.q_waypoints).size
                else np.zeros((0, 7), dtype=np.float64)
                for stage in stages
            ],
            dtype=object,
        ),
        demonstration_arm_q=np.asarray(
            [stage.demonstration_seed.arm_q for stage in stages],
            dtype=np.float64,
        ),
        demonstration_ee_position_object=np.asarray(
            [stage.demonstration_seed.ee_position_object for stage in stages],
            dtype=np.float64,
        ),
        demonstration_ee_rotation_object=np.asarray(
            [stage.demonstration_seed.ee_rotation_object for stage in stages],
            dtype=np.float64,
        ),
        demonstration_contact_position_object=np.asarray(
            [stage.demonstration_seed.contact_position_object for stage in stages],
            dtype=np.float64,
        ),
        projected_seed_contact_world=np.asarray(
            [
                getattr(
                    stage,
                    "projected_seed_contact_world",
                    np.full(3, np.nan, dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=np.float64,
        ),
        projected_seed_contact_offset_error=np.asarray(
            [
                getattr(stage, "projected_seed_contact_offset_error", np.nan)
                for stage in stages
            ],
            dtype=np.float64,
        ),
        projected_seed_contact_feasible_distance=np.asarray(
            [
                getattr(stage, "projected_seed_contact_feasible_distance", np.nan)
                for stage in stages
            ],
            dtype=np.float64,
        ),
        dream_initial_ee_poses=np.asarray(
            [stage.dream_initial_poses for stage in stages],
            dtype=object,
        ),
        dream_initial_candidate_indices=np.asarray(
            [
                getattr(
                    stage,
                    "dream_initial_candidate_indices",
                    np.zeros(0, dtype=np.int64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_initial_contact_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "dream_initial_contact_points",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_initial_contact_distances=np.asarray(
            [
                getattr(
                    stage,
                    "dream_initial_contact_distances",
                    np.zeros(0, dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_initial_contact_mask=np.asarray(
            [
                getattr(stage, "dream_initial_contact_mask", np.zeros(0, dtype=bool))
                for stage in stages
            ],
            dtype=object,
        ),
        dream_ee_pose_sequences=np.asarray(
            [
                getattr(stage, "dream_ee_sequence", np.zeros((0, 7), dtype=np.float64))
                for stage in stages
            ],
            dtype=object,
        ),
        dream_contact_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "dream_contact_points_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_contact_target_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "dream_contact_target_points_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_contact_target_distances=np.asarray(
            [
                getattr(
                    stage,
                    "dream_contact_target_distances",
                    np.zeros(0, dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_contact_target_mask=np.asarray(
            [
                getattr(stage, "dream_contact_target_mask", np.zeros(0, dtype=bool))
                for stage in stages
            ],
            dtype=object,
        ),
        target_contact_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "target_contact_points_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        target_contact_target_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "target_contact_target_points_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        target_contact_distances=np.asarray(
            [
                getattr(
                    stage, "target_contact_distances", np.zeros(0, dtype=np.float64)
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_initial_position_errors=np.asarray(
            [
                (
                    stage.dream_result.initial_position_errors.detach().cpu().numpy()
                    if stage.dream_result is not None
                    and stage.dream_result.initial_position_errors is not None
                    else np.zeros(0, dtype=np.float64)
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_initial_rotation_errors=np.asarray(
            [
                (
                    stage.dream_result.initial_rotation_errors.detach().cpu().numpy()
                    if stage.dream_result is not None
                    and stage.dream_result.initial_rotation_errors is not None
                    else np.zeros(0, dtype=np.float64)
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_failure_reasons=np.asarray(
            [
                str(getattr(stage, "dream_failure_detail", {}).get("reason", "unknown"))
                for stage in stages
            ]
        ),
        dream_failure_details=np.asarray(
            [getattr(stage, "dream_failure_detail", {}) for stage in stages],
            dtype=object,
        ),
        curobo_contact_stage_indices=np.asarray(
            curobo_contact["stage_indices"],
            dtype=np.int64,
        ),
        curobo_contact_drawer_q=np.asarray(
            curobo_contact["drawer_q"],
            dtype=np.float64,
        ),
        curobo_ee_contact_point_error=np.asarray(
            curobo_contact["ee_contact_point_error"],
            dtype=np.float64,
        ),
        curobo_actual_contact=np.asarray(
            curobo_contact["actual_contact"],
            dtype=bool,
        ),
        curobo_actual_contact_near_selected=np.asarray(
            curobo_contact["actual_contact_near_selected"],
            dtype=bool,
        ),
        curobo_actual_contact_distance=np.asarray(
            curobo_contact["actual_contact_distance"],
            dtype=np.float64,
        ),
        curobo_actual_contact_point_error=np.asarray(
            curobo_contact["actual_contact_point_error"],
            dtype=np.float64,
        ),
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
        scene_link_names=(
            np.asarray([link.name for link in scene_point_cloud.links])
            if scene_point_cloud is not None
            else np.asarray([], dtype=str)
        ),
        scene_link_point_counts=(
            np.asarray(
                [link.points_local.shape[0] for link in scene_point_cloud.links],
                dtype=np.int64,
            )
            if scene_point_cloud is not None
            else np.zeros(0, dtype=np.int64)
        ),
        metadata_json=json.dumps(metadata, indent=2),
    )
    return metadata


def _report_reason_counts(reports):
    reason_counts = {}
    for report in reports or []:
        key = report.reason if report.status != "success" else "success"
        reason_counts[key] = reason_counts.get(key, 0) + 1
    return reason_counts


def _parse_config_value(value):
    import yaml

    parsed = yaml.safe_load(value)
    return (
        value
        if parsed is None and value.lower() not in ("null", "none", "~")
        else parsed
    )


def _load_yaml_config(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


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
    _apply_config_overrides(config, cli.overrides)
    return argparse.Namespace(**config)


def main():
    args = parse_args()
    _configure_cuda_memory_limit(args)
    rng = np.random.default_rng(int(args.seed))
    with _suppress_solver_output():
        env = create_open_drawer_env(args)
    try:
        with _suppress_solver_output():
            _initialize_scene_processing(env, args)
        initial_robot_state = close_demo.get_robot_arm_state(env)
        total_pull_distance = _target_pull_distance(env, args)
        should_switch = bool(rng.random() < float(args.switch_probability))
        switch_distance = (
            _choose_switch_distance(total_pull_distance, rng, args)
            if should_switch
            else None
        )

        initial_panel = close_demo.get_panel_frame(env)
        handle_surface = _make_handle_inner_surface(env, initial_panel)

        stages = []
        target_hand_poses = []
        first_pull = switch_distance if should_switch else total_pull_distance
        first_stage, _, first_robot_state = _solve_stage(
            env,
            handle_surface,
            first_pull,
            args,
            stage_name="handle_inner_pull",
        )
        stages.append(first_stage)
        first_stage_advanced = _set_stage_end_state(env, first_stage, first_robot_state)

        if should_switch and first_stage_advanced:
            panel_after_switch = close_demo.get_panel_frame(env)
            panel_inner_surface = _make_panel_inner_surface(env, panel_after_switch)
            remaining_pull = max(
                float(total_pull_distance) - float(switch_distance), 1e-4
            )
            second_stage, _, second_robot_state = _solve_stage(
                env,
                panel_inner_surface,
                remaining_pull,
                args,
                stage_name="panel_inner_pull",
            )
            stages.append(second_stage)
            _set_stage_end_state(env, second_stage, second_robot_state)
        elif should_switch:
            pass

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
        exclude_phases = (
            ("pull",) if bool(getattr(args, "curobo_skip_pull_waypoint", False)) else ()
        )
        target_gripper_poses = _all_target_gripper_poses(
            stages, exclude_phases=exclude_phases
        )
        target_hand_poses = [
            (name, *close_demo.gripper_pose_to_curobo_hand_pose(pos, rot, robot_state))
            for name, pos, rot in target_gripper_poses
        ]

        q_traj = None
        segments = []
        execution_planner = str(getattr(args, "execution_planner", "mppi")).lower()
        if execution_planner == "mppi":
            q_traj, segments = _mppi_joint_trajectory_from_stage_rollouts(
                robot_state, stages
            )
            args._execution_planner_source = "mppi_composite_rollout_cost"
            args._curobo_target_diagnostics = []
            _count_result(
                "mppi_execution",
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
                    # If we deliberately withheld the pull waypoint from cuRobo,
                    # append a linear arm-q + drawer-DOF pull tail here so the
                    # pull motion follows the drawer's prismatic DOF rather than
                    # a free-form cuRobo plan.
                    if exclude_phases and q_traj is not None:
                        q_traj, segments = _append_drawer_pull_tail(
                            q_traj, segments, stages, int(args.pull_tail_steps)
                        )
                        for segment in segments:
                            segment.setdefault("planner", "curobo_with_joint_pull_tail")
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
                    should_switch,
                    switch_distance,
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

        _play_open_trajectory_viewer(
            env,
            stages,
            q_traj,
            segments,
            robot_state,
            args,
        )

        metadata = _save_open_outputs(
            args.output,
            env,
            stages,
            target_hand_poses,
            q_traj,
            segments,
            should_switch,
            switch_distance,
            args,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
