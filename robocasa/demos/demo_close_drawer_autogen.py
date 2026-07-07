import argparse
import functools
import concurrent.futures
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from pathlib import Path

import mujoco
from tqdm import tqdm

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions_gcc11")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("ROBOCASA_ALLOW_VERSION_MISMATCH", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
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

import numpy as np
import robosuite
import trimesh
from robosuite.controllers import load_composite_controller_config
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.transform import Rotation

import robocasa  # noqa: F401
import robocasa.utils.lerobot_utils as LU
from robocasa.demos import ee_skelton
from robocasa.demos.demo_tasks import get_ds_path_any_split
from robocasa.demos.ee_floating_mppi import (
    FloatingEEConfig,
    FloatingEEMPPI,
)
from robocasa.demos import mink_q
from robocasa.demos.full_scene_mjwarp import FullSceneCollisionCheckerPool
from robocasa.demos.rollout import (
    FloatingEERollout,
    RolloutConfig,
    RolloutResult,
)
from robocasa.demos.object_cso import farthest_point_subset
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to


CONTACT_REPO = Path("/home/lab423/scsp/Franka-contact-face-detection-manipulation-main")
CUROBO_SRC_CANDIDATES = (
    Path("/home/lab423/scsp/thirdparty/curobo/src"),
    Path("/home/lab423/opt_ws/src/curobo/src"),
)
PANDA_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))


@dataclass
class PanelFrame:
    center_world: np.ndarray
    rotation_world: np.ndarray
    half_size: np.ndarray
    outward_world: np.ndarray
    push_world: np.ndarray
    geom_name: str


@dataclass
class ContactCandidate:
    local_point: np.ndarray
    world_point: np.ndarray
    cost: float
    lam: np.ndarray
    resulting_pose: np.ndarray
    solver_status: str
    feasible: bool


@dataclass
class MinkContactPoseSolution:
    drawer_candidate_index: int
    drawer_contact_world: np.ndarray
    drawer_contact_local: np.ndarray
    drawer_contact_cost: float
    ee_sample_index: int
    ee_sample_name: str
    ee_contact_geom_name: str
    contact_frame: str
    contact_offset_local: np.ndarray
    roll_angle: float
    q_waypoints: np.ndarray
    target_gripper_poses: list
    contact_position_error: float
    collision_free: bool


@dataclass
class MinkContactAttemptReport:
    drawer_candidate_index: int
    drawer_contact_world: np.ndarray
    drawer_contact_local: np.ndarray
    drawer_contact_cost: float
    contact_feasible: bool
    status: str
    reason: str
    best_ee_sample_index: int
    best_ee_sample_name: str
    best_position_error: float
    best_collision_free: bool


@dataclass
class AutogenFeasibleContactCache:
    candidate_indices: np.ndarray
    positions_world: np.ndarray
    positions_object: np.ndarray
    normals_world: np.ndarray
    normals_object: np.ndarray
    tangents1_world: np.ndarray
    tangents2_world: np.ndarray
    tangents1_object: np.ndarray
    tangents2_object: np.ndarray
    is_edge: np.ndarray
    graph_edges: np.ndarray
    tree: cKDTree


def _normalize(vec, fallback=None):
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        if fallback is None:
            raise ValueError("Cannot normalize near-zero vector.")
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return vec / norm


def _orthonormal_tangents(normal):
    normal = _normalize(normal, fallback=np.array([0.0, 1.0, 0.0]))
    ref = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(normal[2])) < 0.9
        else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    t1 = ref - normal * float(np.dot(ref, normal))
    t1 = _normalize(t1, fallback=np.array([1.0, 0.0, 0.0]))
    t2 = _normalize(np.cross(normal, t1), fallback=np.array([0.0, 0.0, 1.0]))
    return t1, t2


def _quat_wxyz_from_matrix(rot):
    quat_xyzw = Rotation.from_matrix(rot).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )


def _matrix_from_quat_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _pose_inv(pos, rot):
    rot_inv = rot.T
    return -rot_inv @ pos, rot_inv


def _pose_mul(pos_a, rot_a, pos_b, rot_b):
    return pos_a + rot_a @ pos_b, rot_a @ rot_b


def _pose_in_frame(pos_world, rot_world, frame_pos_world, frame_rot_world):
    inv_pos, inv_rot = _pose_inv(frame_pos_world, frame_rot_world)
    return _pose_mul(inv_pos, inv_rot, pos_world, rot_world)


def _mj_has_name(model, kind, name):
    private_map = getattr(model, f"_{kind}_name2id", None)
    if private_map is not None:
        return name in private_map
    try:
        getattr(model, kind)(name)
        return True
    except Exception:
        return False


def _mj_named(model, kind, name):
    return getattr(model, kind)(name)


def _mj_id(model, kind, name):
    name2id = getattr(model, f"{kind}_name2id", None)
    if name2id is not None:
        return int(name2id(name))
    private_map = getattr(model, f"_{kind}_name2id", None)
    if private_map is not None:
        return int(private_map[name])
    obj_type = getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}")
    obj_id = int(mujoco.mj_name2id(model, obj_type, name))
    if obj_id < 0:
        raise KeyError(f"Unknown MuJoCo {kind} name: {name}")
    return obj_id


def _make_gripper_contact_rotation(push_world):
    x_axis = _normalize(push_world)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = z_axis - x_axis * np.dot(z_axis, x_axis)
    z_axis = _normalize(z_axis, fallback=np.array([0.0, 0.0, 1.0]))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    z_axis = _normalize(np.cross(x_axis, y_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _rot_about_axis(axis, angle):
    return Rotation.from_rotvec(_normalize(axis) * float(angle)).as_matrix()


def _add_contact_repo_to_path():
    if str(CONTACT_REPO) not in sys.path:
        sys.path.insert(0, str(CONTACT_REPO))


def _import_contact_optimizer():
    _add_contact_repo_to_path()
    from robocasa.demos.mlqp_point_cabinet import LambdaContactControlOptimizer

    return LambdaContactControlOptimizer


def _install_qpsolvers_jaxopt_stub():
    module_name = "qpsolvers.solvers.jaxopt_osqp_"
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)

    def _disabled(*args, **kwargs):
        raise ImportError(
            "jaxopt_osqp is disabled for this demo; using quadprog instead."
        )

    stub.jaxopt_osqp_solve_problem = _disabled
    stub.jaxopt_osqp_solve_qp = _disabled
    sys.modules[module_name] = stub


def _import_mink():
    _install_qpsolvers_jaxopt_stub()
    import mink

    return mink


def _ensure_curobo_importable():
    if "setuptools_scm" not in sys.modules:
        scm = types.ModuleType("setuptools_scm")
        scm.get_version = lambda *args, **kwargs: "0+local"
        sys.modules["setuptools_scm"] = scm

    for candidate in CUROBO_SRC_CANDIDATES:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def preload_curobo_runtime():
    _ensure_curobo_importable()
    import torch

    torch.cuda.init()
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
    )  # noqa: F401


def _assert_cuda_context_usable(stage: str) -> None:
    """Fail early when an earlier Warp/mjwarp kernel poisoned CUDA state."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        device = torch.device("cuda:0")
        probe = torch.empty((1,), device=device)
        probe.fill_(0.0)
        torch.cuda.synchronize(device)
        del probe
    except Exception as exc:
        raise RuntimeError(
            f"CUDA context is unusable before {stage}: {exc}. "
            "This usually means an earlier asynchronous Warp/mjwarp kernel "
            "failed; restart the Python process before retrying cuRobo."
        ) from exc


def _canonical_panda_joint_key(joint_name):
    name = str(joint_name)
    for prefix in ("robot0_joint", "panda_joint", "joint"):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                return f"panda_joint{int(suffix)}"
    return name


def _extract_curobo_joint_names(motion_gen):
    candidates = (
        ("kinematics", "joint_names"),
        ("kinematics", "kinematics_config", "joint_names"),
        ("kinematics", "kinematics_config", "cspace", "joint_names"),
        ("kinematics", "robot_cfg", "kinematics", "cspace", "joint_names"),
        ("rollout_fn", "kinematics", "joint_names"),
        ("rollout_fn", "kinematics", "kinematics_config", "joint_names"),
    )
    for path in candidates:
        obj = motion_gen
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj:
            names = tuple(str(name) for name in obj)
            if len(names) >= 7:
                return names[:7]
    return PANDA_JOINT_NAMES


def _joint_order_indices(source_joint_names, target_joint_names):
    source_keys = [_canonical_panda_joint_key(name) for name in source_joint_names]
    target_keys = [_canonical_panda_joint_key(name) for name in target_joint_names]
    indices = []
    for key in target_keys:
        if key not in source_keys:
            raise RuntimeError(
                "Cannot map cuRobo and robosuite joint orders: "
                f"missing joint '{key}' in source joints {tuple(source_joint_names)}"
            )
        indices.append(source_keys.index(key))
    return np.asarray(indices, dtype=np.int64)


def _reorder_q(q, source_joint_names, target_joint_names):
    q_array = np.asarray(q, dtype=np.float64)
    leading_shape = q_array.shape[:-1]
    q_array = q_array.reshape(-1, q_array.shape[-1])
    if q_array.shape[-1] != len(tuple(source_joint_names)):
        raise RuntimeError(
            "Cannot reorder joint vector: "
            f"q has {q_array.shape[-1]} joints, source order has {len(tuple(source_joint_names))}"
        )
    indices = _joint_order_indices(source_joint_names, target_joint_names)
    reordered = q_array[:, indices]
    return reordered.reshape(*leading_shape, len(tuple(target_joint_names)))


def create_close_drawer_env(args):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=args.robot,
    )
    env = robosuite.make(
        env_name="CloseDrawer",
        robots=args.robot,
        controller_configs=controller_config,
        has_renderer=bool(args.render or args.visualize_contact),
        has_offscreen_renderer=bool(
            args.save_trajectory_videos or getattr(args, "origin_demo", False)
        ),
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


def get_panel_frame(env):
    model = env.sim.model
    data = env.sim.data
    drawer_name = env.drawer.name
    reg_name = f"{drawer_name}_door_reg_main"
    if reg_name in model._geom_name2id:
        panel_geom = reg_name
    else:
        panel_geom = f"{drawer_name}_door_g1"
    if panel_geom not in model._geom_name2id:
        raise RuntimeError(f"Cannot find drawer panel geom for drawer '{drawer_name}'.")

    geom_id = model.geom_name2id(panel_geom)
    center_world = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy()
    rotation_world = (
        np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3).copy()
    )
    half_size = np.asarray(model.geom_size[geom_id], dtype=np.float64).copy()

    outward = None
    handle_names = [
        name
        for name in model._geom_name2id
        if name.startswith(f"{drawer_name}_door_handle_g")
    ]
    if handle_names:
        handle_pos = np.mean(
            [
                np.asarray(data.geom_xpos[model.geom_name2id(name)], dtype=np.float64)
                for name in handle_names
            ],
            axis=0,
        )
        outward = handle_pos - center_world
        outward = outward - rotation_world[:, 0] * np.dot(outward, rotation_world[:, 0])
        outward = outward - rotation_world[:, 2] * np.dot(outward, rotation_world[:, 2])

    outward = _normalize(outward, fallback=-rotation_world[:, 1])
    push_world = -outward
    if np.dot(push_world, rotation_world[:, 1]) < 0.0:
        rotation_world[:, 1] *= -1.0

    return PanelFrame(
        center_world=center_world,
        rotation_world=rotation_world,
        half_size=half_size,
        outward_world=outward,
        push_world=push_world,
        geom_name=panel_geom,
    )


def sample_panel_candidates(panel, grid_x, grid_z, margin):
    x_lim = max(float(panel.half_size[0]) - margin, 1e-4)
    z_lim = max(float(panel.half_size[2]) - margin, 1e-4)
    xs = np.linspace(-x_lim, x_lim, int(grid_x))
    zs = np.linspace(-z_lim, z_lim, int(grid_z))
    local = []
    world = []
    for z in zs:
        for x in xs:
            p_local = np.array([x, -float(panel.half_size[1]), z], dtype=np.float64)
            p_world = panel.center_world + panel.rotation_world @ p_local
            local.append(p_local)
            world.append(p_world)
    return np.asarray(local), np.asarray(world)


def build_contact_optimizer(panel, args):
    LambdaContactControlOptimizer = _import_contact_optimizer()
    mesh = trimesh.creation.box(extents=2.0 * panel.half_size)
    mesh_path = tempfile.NamedTemporaryFile(
        prefix="robocasa_drawer_panel_", suffix=".stl", delete=False
    ).name
    mesh.export(mesh_path)
    optimizer = LambdaContactControlOptimizer(
        mesh_path,
        obj_mass=args.contact_obj_mass,
        arm_friction=args.contact_friction,
        contact_stiffness=args.contact_stiffness,
        time_step=args.contact_dt,
        max_contacts=1,
        sample_num=max(8, args.grid_x * args.grid_z),
        pos_coef=args.contact_pos_coef,
        ori_coef=args.contact_ori_coef,
        nlp_solver=args.contact_solver,
    )
    return optimizer, mesh_path


def _mesh_from_geom_world(env, geom_id):
    from robocasa.demos.scene_process import _geom_mesh_in_body

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    mesh_body = _geom_mesh_in_body(raw_model, int(geom_id))
    if mesh_body is None or mesh_body.faces.size == 0:
        return None
    body_id = int(raw_model.geom_bodyid[int(geom_id)])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(raw_data.xmat[body_id], dtype=np.float64).reshape(
        3, 3
    )
    transform[:3, 3] = np.asarray(raw_data.xpos[body_id], dtype=np.float64).reshape(3)
    mesh_world = mesh_body.copy()
    mesh_world.apply_transform(transform)
    return mesh_world


def _subdivide_mesh(mesh, max_edge):
    if max_edge is None or float(max_edge) <= 0.0:
        return mesh
    try:
        vertices, faces = trimesh.remesh.subdivide_to_size(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            max_edge=float(max_edge),
        )
        if vertices.size == 0 or faces.size == 0:
            return mesh
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except Exception:
        return mesh


def _coacd_parts(mesh, args, seed):
    try:
        import coacd
    except ImportError as exc:
        raise RuntimeError(
            "autogen close drawer contact sampling requires the `coacd` package"
        ) from exc

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    if mesh.faces.shape[0] < 4:
        return [mesh]
    source = coacd.Mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    kwargs = {
        "threshold": float(getattr(args, "autogen_coacd_threshold", 0.05)),
        "max_convex_hull": int(getattr(args, "autogen_coacd_max_convex_hull", 32)),
        "preprocess_mode": str(getattr(args, "autogen_coacd_preprocess_mode", "auto")),
        "preprocess_resolution": int(
            getattr(args, "autogen_coacd_preprocess_resolution", 30)
        ),
        "resolution": int(getattr(args, "autogen_coacd_resolution", 2000)),
        "mcts_nodes": int(getattr(args, "autogen_coacd_mcts_nodes", 20)),
        "mcts_iterations": int(getattr(args, "autogen_coacd_mcts_iterations", 100)),
        "mcts_max_depth": int(getattr(args, "autogen_coacd_mcts_max_depth", 3)),
        "max_ch_vertex": int(getattr(args, "autogen_coacd_max_ch_vertex", 256)),
        "seed": int(seed),
    }
    try:
        parts = coacd.run_coacd(source, **kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        parts = coacd.run_coacd(source, **kwargs)
    return [
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        for vertices, faces in parts
    ]


def _convex_part_equations(parts):
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return np.zeros((0, 1, 4), dtype=np.float64)

    equations = []
    for part in parts:
        vertices = np.asarray(part.vertices, dtype=np.float64)
        if vertices.shape[0] < 4:
            continue
        try:
            equations.append(
                np.asarray(ConvexHull(vertices).equations, dtype=np.float64)
            )
        except Exception:
            continue
    if not equations:
        return np.zeros((0, 1, 4), dtype=np.float64)
    max_rows = max(int(eq.shape[0]) for eq in equations)
    padded = np.zeros((len(equations), max_rows, 4), dtype=np.float64)
    padded[..., 3] = -1e6
    for idx, eq in enumerate(equations):
        padded[idx, : eq.shape[0], :] = eq
    return padded


def _polygon_records_from_mesh(mesh, geom_name, center_hint):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        return []
    center_hint = np.asarray(center_hint, dtype=np.float64).reshape(3)
    records = []
    for face in faces:
        tri = vertices[face]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if float(np.linalg.norm(normal)) < 1e-10:
            continue
        normal = _normalize(normal)
        center = np.mean(tri, axis=0)
        if float(np.dot(center - center_hint, normal)) < 0.0:
            normal = -normal
        t1, t2 = _orthonormal_tangents(normal)
        records.append(
            {
                "center": center,
                "normal": normal,
                "tangent1": t1,
                "tangent2": t2,
                "geom_name": str(geom_name),
            }
        )
    return records


def _inflating_handle_geom_names(env):
    """Names of fixture geoms that protrude toward the robot and inflate the
    panel's collision volume. These are handle knobs / connectors / screws /
    pre-contact bars that hang in front of the panel surface. When pushed against,
    the robot only contacts the flat panel; these upstream fixtures should NOT
    be part of the contact target or the COACD hull (the hull built from them
    extends out into free air and causes every in-front EE pose to be flagged as
    penetration by ``_signed_distance_to_convex``)."""
    model = env.sim.model
    drawer_name = env.drawer.name
    names = set()
    _name2id = getattr(model, "_geom_name2id", None)
    if not _name2id:
        return names
    for name in _name2id:
        if not (
            name.startswith(f"{drawer_name}_door_g")
            or name.startswith(f"{drawer_name}_door")
        ):
            continue
        low = name.lower()
        # Handle fixtures project out of the panel along the handle axis. KNobs,
        # connectors, hang-bars, thumb-screws, etc. all share a "handle" or
        # "knob" substring (see robocasa.models.fixtures.handles).
        if "handle" in low or "knob" in low:
            names.add(name)
    return names


def _target_panel_geom_ids(env, panel):
    model = env.sim.model
    ids = []
    drawer_name = env.drawer.name
    inflating = _inflating_handle_geom_names(env)
    for name, geom_id in model._geom_name2id.items():
        if name in inflating:
            continue
        if name == panel.geom_name:
            ids.append(int(geom_id))
        elif bool(
            getattr(panel, "include_drawer_door_geoms", False)
        ) and name.startswith(f"{drawer_name}_door_g"):
            ids.append(int(geom_id))
    return tuple(sorted(set(ids)))


def _build_autogen_contact_candidates(env, panel, push_distance, args):
    cached = getattr(args, "_autogen_candidates_by_drawer_q", None)
    cache_key = float(_drawer_joint_value(env))
    if cached is not None and cache_key in cached:
        return cached[cache_key]

    rng = np.random.default_rng(int(args.seed) + 17011)
    records = []
    convex_parts_world = []
    target_geom_ids = _target_panel_geom_ids(env, panel)
    max_edge = float(getattr(args, "autogen_handle_subdivide_max_edge", 0.005))
    for geom_id in target_geom_ids:
        mesh_world = _mesh_from_geom_world(env, geom_id)
        if mesh_world is None:
            continue
        geom_name = next(
            (
                name
                for name, gid in env.sim.model._geom_name2id.items()
                if int(gid) == int(geom_id)
            ),
            str(geom_id),
        )
        for part_index, part in enumerate(
            _coacd_parts(mesh_world, args, int(args.seed) + int(geom_id))
        ):
            convex_parts_world.append(part)
            dense_part = _subdivide_mesh(part, max_edge)
            records.extend(
                _polygon_records_from_mesh(
                    dense_part,
                    f"{geom_name}:coacd_{part_index}",
                    center_hint=np.asarray(
                        env.sim.data.geom_xpos[geom_id], dtype=np.float64
                    ),
                )
            )
    if not records:
        raise RuntimeError(
            f"No autogen drawer contact polygons were produced for {panel.geom_name!r}"
        )

    centers = np.asarray([r["center"] for r in records], dtype=np.float64)
    limit = int(getattr(args, "autogen_object_point_count", 384))
    if centers.shape[0] > limit:
        subset = farthest_point_subset(
            centers, limit, initial_index=int(rng.integers(centers.shape[0]))
        )
        records = [records[int(i)] for i in subset]
        centers = centers[subset]

    rotation_world = np.asarray(panel.rotation_world, dtype=np.float64)
    local_points = (
        centers - np.asarray(panel.center_world, dtype=np.float64)
    ) @ rotation_world
    normals_world = np.asarray(
        [_normalize(r["normal"]) for r in records], dtype=np.float64
    )
    normals_local = normals_world @ rotation_world

    optimizer, mesh_path = build_contact_optimizer(panel, args)
    try:
        current_x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        x_d = np.array(
            [0.0, float(push_distance), 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        tau_o = np.zeros(6, dtype=np.float64)
        t1_batch = np.asarray(
            [r["tangent1"] @ rotation_world for r in records], dtype=np.float64
        )
        t2_batch = np.asarray(
            [r["tangent2"] @ rotation_world for r in records], dtype=np.float64
        )
        lam_batch, x_plus_batch, cost_batch, status_batch = optimizer._solve_batch(
            x_d=x_d,
            current_x=current_x,
            tau_o=tau_o,
            n_arm=normals_local,
            t1=t1_batch,
            t2=t2_batch,
            p_arm=local_points,
            curr_ori_coef=1.0,
            lam_upper_bound=args.contact_lam_upper_bound,
        )
    finally:
        try:
            os.unlink(mesh_path)
        except OSError:
            pass

    push_world = _normalize(panel.push_world)
    mu = max(float(args.contact_friction), 1e-6)
    friction_threshold = 1.0 / float(np.sqrt(1.0 + mu * mu))
    candidates = []
    for idx, record in enumerate(records):
        normal_world = _normalize(normals_world[idx])
        push_along_inward_normal = -float(np.dot(push_world, normal_world))
        friction_ok = push_along_inward_normal >= friction_threshold
        feasible = bool(
            np.isfinite(float(cost_batch[idx]))
            and float(cost_batch[idx]) <= float(args.contact_cost_threshold)
            and (
                friction_ok
                or not bool(getattr(args, "require_friction_cone_push", False))
            )
        )
        candidate = ContactCandidate(
            local_point=np.asarray(local_points[idx], dtype=np.float64),
            world_point=np.asarray(record["center"], dtype=np.float64),
            cost=float(cost_batch[idx]),
            lam=np.asarray(lam_batch[idx], dtype=np.float64),
            resulting_pose=np.asarray(x_plus_batch[idx], dtype=np.float64),
            solver_status=str(status_batch[idx]),
            feasible=feasible,
        )
        candidate.force_normal_local = np.asarray(normals_local[idx], dtype=np.float64)
        candidate.approach_world = normal_world
        candidate.contact_face = str(record["geom_name"])
        candidate.friction_cone_ok = bool(friction_ok)
        candidate.friction_cone_push_ratio = float(push_along_inward_normal)
        candidate.is_edge = False
        candidates.append(candidate)

    feasible_indices = np.asarray(
        [i for i, c in enumerate(candidates) if bool(c.feasible)], dtype=np.int64
    )
    if feasible_indices.size == 0 and args.require_feasible_contact:
        best = min(candidates, key=lambda c: c.cost)
        raise RuntimeError(
            "Autogen contact search found no feasible point. "
            f"Best cost={best.cost:.6f}, threshold={args.contact_cost_threshold:.6f}, "
            f"status={best.solver_status}."
        )

    graph_edges = np.zeros((0, 2), dtype=np.int64)
    is_edge = np.zeros(feasible_indices.size, dtype=bool)
    if feasible_indices.size >= 4:
        points = np.asarray(
            [candidates[i].world_point for i in feasible_indices], dtype=np.float64
        )
        try:
            delaunay = Delaunay(points, qhull_options="QJ")
            edges = set()
            for simplex in np.asarray(delaunay.simplices, dtype=np.int64):
                for a in range(simplex.size):
                    for b in range(a + 1, simplex.size):
                        edges.add(tuple(sorted((int(simplex[a]), int(simplex[b])))))
            graph_edges = np.asarray(sorted(edges), dtype=np.int64)
            is_edge[
                np.unique(np.asarray(delaunay.convex_hull, dtype=np.int64).reshape(-1))
            ] = True
        except Exception:
            graph_edges = np.zeros((0, 2), dtype=np.int64)
    elif feasible_indices.size:
        is_edge[:] = True
    for local_index, candidate_index in enumerate(feasible_indices):
        candidates[int(candidate_index)].is_edge = bool(is_edge[int(local_index)])

    feasible_positions_world = np.asarray(
        [candidates[i].world_point for i in feasible_indices], dtype=np.float64
    ).reshape(-1, 3)
    feasible_normals_world = np.asarray(
        [_normalize(candidates[i].approach_world) for i in feasible_indices],
        dtype=np.float64,
    ).reshape(-1, 3)
    tangents1_world = []
    tangents2_world = []
    for normal in feasible_normals_world:
        t1, t2 = _orthonormal_tangents(normal)
        tangents1_world.append(t1)
        tangents2_world.append(t2)
    tree_points = (
        (feasible_positions_world - panel.center_world) @ rotation_world
        if feasible_positions_world.shape[0]
        else np.zeros((0, 3), dtype=np.float64)
    )
    tree = cKDTree(
        tree_points if tree_points.shape[0] else np.zeros((1, 3), dtype=np.float64)
    )
    feasible_cache = AutogenFeasibleContactCache(
        candidate_indices=feasible_indices,
        positions_world=feasible_positions_world,
        positions_object=tree_points,
        normals_world=feasible_normals_world,
        normals_object=feasible_normals_world @ rotation_world,
        tangents1_world=np.asarray(tangents1_world, dtype=np.float64).reshape(-1, 3),
        tangents2_world=np.asarray(tangents2_world, dtype=np.float64).reshape(-1, 3),
        tangents1_object=np.asarray(tangents1_world, dtype=np.float64).reshape(-1, 3)
        @ rotation_world,
        tangents2_object=np.asarray(tangents2_world, dtype=np.float64).reshape(-1, 3)
        @ rotation_world,
        is_edge=is_edge,
        graph_edges=graph_edges,
        tree=tree,
    )
    if bool(getattr(args, "autogen_skeleton_disable_handle_convex", False)):
        # Close-drawer pushes a flat panel head-on. The panel is a single convex
        # box, so COACD returns 1 part = the whole drawer body. With that whole-
        # body convex hull, any hand pose pushing the front face has the hand
        # body landing "outside" the front plane but still flagged by other
        # planes — the convex check is geometrically meaningless here and was
        # silently rejecting every R0 candidate. The tangent half-plane (A_obj)
        # already enforces "hand on +b_w side of the contact face".
        feasible_cache.handle_convex_equations = np.zeros((0, 1, 4), dtype=np.float64)
    else:
        feasible_cache.handle_convex_equations = _convex_part_equations(
            convex_parts_world
        )
    selected = min(
        (c for c in candidates if bool(c.feasible)),
        key=lambda c: c.cost,
        default=min(candidates, key=lambda c: c.cost),
    )
    if cached is None:
        cached = {}
        args._autogen_candidates_by_drawer_q = cached
    cached[cache_key] = (candidates, selected, feasible_cache)
    args._autogen_feasible_cache = feasible_cache
    return cached[cache_key]


def evaluate_contacts(env, panel, args):
    qpos = float(
        env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr(env.drawer.door_joint_names[0])
        ]
    )
    current_open_distance = max(-qpos, 0.0)
    joint_close_distance = max(
        current_open_distance - env.drawer.size[1] * 0.55 * args.target_open_fraction,
        0.0,
    )
    push_distance = min(float(args.push_distance), joint_close_distance)
    push_distance = max(push_distance, 1e-4)
    if bool(getattr(args, "use_autogen_contact", True)):
        candidates, selected, _ = _build_autogen_contact_candidates(
            env, panel, push_distance, args
        )
        return candidates, selected, push_distance

    local_points, world_points = sample_panel_candidates(
        panel,
        grid_x=args.grid_x,
        grid_z=args.grid_z,
        margin=args.panel_margin,
    )
    optimizer, mesh_path = build_contact_optimizer(panel, args)
    try:
        qpos = float(
            env.sim.data.qpos[
                env.sim.model.get_joint_qpos_addr(env.drawer.door_joint_names[0])
            ]
        )
        current_open_distance = max(-qpos, 0.0)
        joint_close_distance = max(
            current_open_distance
            - env.drawer.size[1] * 0.55 * args.target_open_fraction,
            0.0,
        )
        push_distance = min(float(args.push_distance), joint_close_distance)
        push_distance = max(push_distance, 1e-4)

        current_x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        x_d = np.array([0.0, push_distance, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        tau_o = np.zeros(6, dtype=np.float64)
        force_normal_local = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        tangent_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        candidates = []
        for p_local, p_world in zip(local_points, world_points):
            lam, x_plus, cost, status = optimizer._solve_once(
                x_d=x_d,
                current_x=current_x,
                tau_o=tau_o,
                n_arm=force_normal_local,
                t1=tangent_x,
                t2=tangent_z,
                p_arm=p_local,
                curr_ori_coef=1.0,
                lam_upper_bound=args.contact_lam_upper_bound,
            )
            cost = float(cost)
            feasible = bool(np.isfinite(cost) and cost <= args.contact_cost_threshold)
            candidates.append(
                ContactCandidate(
                    local_point=np.asarray(p_local, dtype=np.float64),
                    world_point=np.asarray(p_world, dtype=np.float64),
                    cost=cost,
                    lam=np.asarray(lam, dtype=np.float64),
                    resulting_pose=np.asarray(x_plus, dtype=np.float64),
                    solver_status=str(status),
                    feasible=feasible,
                )
            )

        candidates.sort(key=lambda c: c.cost)
        feasible_candidates = [c for c in candidates if c.feasible]
        if not feasible_candidates and args.require_feasible_contact:
            best = candidates[0]
            raise RuntimeError(
                "No feasible contact point found. "
                f"Best cost={best.cost:.6f}, threshold={args.contact_cost_threshold:.6f}, "
                f"status={best.solver_status}."
            )
        return (
            candidates,
            (feasible_candidates[0] if feasible_candidates else candidates[0]),
            push_distance,
        )
    finally:
        try:
            os.unlink(mesh_path)
        except OSError:
            pass


def get_robot_arm_state(env):
    robot = env.robots[0]
    model = env.sim.model
    data = env.sim.data
    robocasa_joint_names = tuple(robot.robot_model.joints[:7])
    q = np.array(
        [
            float(data.qpos[model.get_joint_qpos_addr(joint_name)])
            for joint_name in robocasa_joint_names
        ],
        dtype=np.float64,
    )
    base_body = "robot0_link0"
    if base_body not in model._body_name2id:
        raise RuntimeError("Expected robot0_link0 body for Panda arm base.")
    base_id = model.body_name2id(base_body)
    base_pos = np.asarray(data.body_xpos[base_id], dtype=np.float64).copy()
    base_rot = (
        np.asarray(data.body_xmat[base_id], dtype=np.float64).reshape(3, 3).copy()
    )

    hand_body = "robot0_right_hand"
    grip_site_id = robot.eef_site_id["right"]
    hand_id = model.body_name2id(hand_body)
    hand_pos = np.asarray(data.body_xpos[hand_id], dtype=np.float64).copy()
    hand_rot = (
        np.asarray(data.body_xmat[hand_id], dtype=np.float64).reshape(3, 3).copy()
    )
    grip_pos = np.asarray(data.site_xpos[grip_site_id], dtype=np.float64).copy()
    grip_rot = (
        np.asarray(data.site_xmat[grip_site_id], dtype=np.float64).reshape(3, 3).copy()
    )
    hand_pos_base, hand_rot_base = _pose_in_frame(
        hand_pos, hand_rot, base_pos, base_rot
    )
    hand_to_grip_pos, hand_to_grip_rot = _pose_in_frame(
        grip_pos, grip_rot, hand_pos, hand_rot
    )
    return {
        "robocasa_joint_names": robocasa_joint_names,
        "curobo_joint_names": PANDA_JOINT_NAMES,
        "q": q,
        "base_pos": base_pos,
        "base_rot": base_rot,
        "hand_pos_base": hand_pos_base,
        "hand_rot_base": hand_rot_base,
        "hand_to_grip_pos": hand_to_grip_pos,
        "hand_to_grip_rot": hand_to_grip_rot,
    }


def gripper_pose_to_curobo_hand_pose(pos_world, rot_world, robot_state):
    inv_grip_pos, inv_grip_rot = _pose_inv(
        robot_state["hand_to_grip_pos"],
        robot_state["hand_to_grip_rot"],
    )
    hand_pos_world, hand_rot_world = _pose_mul(
        pos_world, rot_world, inv_grip_pos, inv_grip_rot
    )
    return _pose_in_frame(
        hand_pos_world,
        hand_rot_world,
        robot_state["base_pos"],
        robot_state["base_rot"],
    )


def build_target_gripper_poses(panel, selected, push_distance, args):
    grip_rot = _make_gripper_contact_rotation(panel.push_world)
    contact_pos = selected.world_point + panel.outward_world * args.contact_standoff
    precontact_pos = (
        selected.world_point + panel.outward_world * args.precontact_distance
    )
    poses = [
        ("precontact", precontact_pos, grip_rot),
        ("contact", contact_pos, grip_rot),
    ]
    if bool(getattr(args, "execute_push_stage", False)):
        push_pos = contact_pos + panel.push_world * push_distance
        poses.append(("push", push_pos, grip_rot))
    return poses


def _current_robot_model_q(env, robot_model):
    full_model = env.sim.model
    full_data = env.sim.data
    q = np.asarray(robot_model.qpos0, dtype=np.float64).copy()
    for joint_id in range(robot_model.njnt):
        joint_name = robot_model.joint(joint_id).name
        if not _mj_has_name(full_model, "joint", joint_name):
            continue
        src = int(_mj_named(full_model, "joint", joint_name).qposadr[0])
        dst = int(robot_model.joint(joint_name).qposadr[0])
        width = len(robot_model.joint(joint_name).qpos0)
        q[dst : dst + width] = full_data.qpos[src : src + width]
    return q


def _sync_mink_base_pose(env, robot_model):
    full_model = env.sim.model
    for body_name in ("robot0_base", "robot0_link0"):
        if _mj_has_name(robot_model, "body", body_name) and _mj_has_name(
            full_model, "body", body_name
        ):
            robot_model.body(body_name).pos = _mj_named(
                full_model, "body", body_name
            ).pos
            robot_model.body(body_name).quat = _mj_named(
                full_model, "body", body_name
            ).quat


def _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names):
    q = []
    for joint_name in arm_joint_names:
        addr = int(robot_model.joint(joint_name).qposadr[0])
        q.append(float(q_robot[addr]))
    return np.asarray(q, dtype=np.float64)


def _make_mink_posture_cost(robot_model, arm_joint_names, args):
    arm_dof_indices = set()
    for joint_name in arm_joint_names:
        if not _mj_has_name(robot_model, "joint", joint_name):
            continue
        joint = robot_model.joint(joint_name)
        dof_start = int(joint.dofadr[0])
        dof_width = len(joint.qpos0)
        arm_dof_indices.update(range(dof_start, dof_start + dof_width))
    costs = np.full(robot_model.nv, float(args.mink_locked_dof_cost), dtype=np.float64)
    for dof_idx in arm_dof_indices:
        costs[dof_idx] = float(args.mink_arm_posture_cost)
    return costs


def _make_pose_matrix(pos, rot):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return pose


def _frame_pose_from_configuration(configuration, frame_name, frame_type):
    transform = configuration.get_transform_frame_to_world(frame_name, frame_type)
    matrix = transform.as_matrix()
    return np.asarray(matrix[:3, 3], dtype=np.float64), np.asarray(
        matrix[:3, :3], dtype=np.float64
    )


def _solve_mink_frame_pose(
    env, frame_name, target_pos, target_rot, q_start, q_posture, posture_cost, args
):
    mink = _import_mink()
    robot_model = env.robots[0].robot_model.mujoco_model
    _sync_mink_base_pose(env, robot_model)

    configuration = mink.Configuration(robot_model)
    configuration.update(np.asarray(q_start, dtype=np.float64).copy())

    posture_task = mink.PostureTask(
        robot_model,
        cost=posture_cost,
        lm_damping=args.mink_posture_lm_damping,
    )
    posture_task.set_target(q_posture)

    frame_task = mink.FrameTask(
        frame_name=frame_name,
        frame_type="site",
        position_cost=args.mink_position_cost,
        orientation_cost=args.mink_orientation_cost,
        lm_damping=args.mink_frame_lm_damping,
    )
    frame_task.set_target(
        mink.SE3.from_matrix(_make_pose_matrix(target_pos, target_rot))
    )
    tasks = [posture_task, frame_task]

    last_pos_error = np.inf
    for _ in range(args.mink_max_iters):
        velocity = mink.solve_ik(
            configuration,
            tasks,
            args.mink_dt,
            args.mink_solver,
            args.mink_damping,
        )
        configuration.integrate_inplace(velocity, args.mink_dt)
        actual_pos, _ = _frame_pose_from_configuration(
            configuration, frame_name, "site"
        )
        last_pos_error = float(np.linalg.norm(actual_pos - target_pos))
        if last_pos_error <= args.mink_position_tolerance:
            break
    return configuration.q.copy(), last_pos_error


def _contact_offsets_from_gripper(env, frame_name):
    model = env.sim.model
    data = env.sim.data
    site_id = model.site_name2id(frame_name)
    site_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    site_rot = (
        np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    )
    offsets = [
        (
            "gripper_center",
            np.zeros(3, dtype=np.float64),
            "gripper0_right_hand_collision",
        )
    ]
    pad_specs = (
        ("gripper0_right_finger1_pad_collision", -1.0),
        ("gripper0_right_finger2_pad_collision", 1.0),
    )
    contour_uv = (
        (0.0, 0.0),
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
    )
    for geom_name, x_sign in pad_specs:
        if geom_name not in model._geom_name2id:
            continue
        geom_id = model.geom_name2id(geom_name)
        geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        geom_rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        offset, rot = _pose_in_frame(geom_pos, geom_rot, site_pos, site_rot)
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        for sample_idx, (u, v) in enumerate(contour_uv):
            local_surface = np.array(
                [x_sign * size[0], u * size[1], v * size[2]], dtype=np.float64
            )
            sample_name = "surface_center" if sample_idx == 0 else f"edge_{sample_idx}"
            offsets.append(
                (f"{geom_name}:{sample_name}", offset + rot @ local_surface, geom_name)
            )
    return offsets[:15]


def _set_env_arm_q(env, arm_joint_names, q_arm):
    model = env.sim.model
    data = env.sim.data
    for joint_name, value in zip(arm_joint_names, q_arm):
        data.qpos[model.get_joint_qpos_addr(joint_name)] = float(value)
    env.sim.forward()


def _robot_contact_geom_sets(env, panel):
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
    drawer_name = env.drawer.name
    allowed_drawer_geoms = {
        geom_id
        for name, geom_id in model._geom_name2id.items()
        if name == panel.geom_name or name.startswith(f"{drawer_name}_door_g")
    }
    return robot_geoms, ee_geoms, allowed_drawer_geoms


def _check_arm_q_collision(
    env,
    panel,
    arm_joint_names,
    q_arm,
    allowed_ee_geom_name=None,
    penetration_tolerance=0.0,
):
    model = env.sim.model
    data = env.sim.data
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    geom_id_to_name = {geom_id: name for name, geom_id in model._geom_name2id.items()}
    try:
        _set_env_arm_q(env, arm_joint_names, q_arm)
        robot_geoms, ee_geoms, allowed_drawer_geoms = _robot_contact_geom_sets(
            env, panel
        )
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
            if geom1 not in robot_geoms and geom2 not in robot_geoms:
                continue
            if geom1 in robot_geoms and geom2 in robot_geoms:
                continue
            robot_geom = geom1 if geom1 in robot_geoms else geom2
            other_geom = geom2 if robot_geom == geom1 else geom1
            if robot_geom in allowed_ee_geoms and other_geom in allowed_drawer_geoms:
                continue
            contact_dist = float(contact.dist)
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


def _is_arm_q_collision_free(env, panel, arm_joint_names, q_arm):
    return _check_arm_q_collision(env, panel, arm_joint_names, q_arm)[0]


def _build_mink_solution(
    env,
    panel,
    candidate,
    candidate_index,
    push_distance,
    robot_model,
    arm_joint_names,
    frame_name,
    ee_sample_index,
    ee_sample_name,
    ee_contact_geom_name,
    contact_offset,
    target_rot,
    roll_angle,
    q_initial,
    q_posture,
    posture_cost,
    contact_pos_error,
    collision_free,
    args,
):
    drawer_q0 = _drawer_joint_value(env)
    drawer_q_closed = min(drawer_q0 + float(push_distance), 0.0)
    waypoint_specs = [
        (
            "precontact",
            candidate.world_point + panel.outward_world * args.precontact_distance,
            drawer_q0,
        ),
        (
            "contact",
            candidate.world_point + panel.outward_world * args.contact_standoff,
            drawer_q0,
        ),
        (
            "push",
            candidate.world_point
            + panel.outward_world * args.contact_standoff
            + panel.push_world * push_distance,
            drawer_q_closed,
        ),
    ]
    q_robot = q_initial.copy()
    q_waypoints = []
    target_gripper_poses = []
    max_pos_error = float(contact_pos_error)
    for name, desired_contact_point, _ in waypoint_specs:
        target_frame_pos = desired_contact_point - target_rot @ contact_offset
        q_robot, pos_error = _solve_mink_frame_pose(
            env,
            frame_name,
            target_frame_pos,
            target_rot,
            q_robot,
            q_posture,
            posture_cost,
            args,
        )
        max_pos_error = max(max_pos_error, float(pos_error))
        frame_pos, frame_rot = _frame_pose_from_configuration(
            _make_mink_configuration(robot_model, q_robot),
            frame_name,
            "site",
        )
        q_waypoints.append(
            _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
        )
        target_gripper_poses.append((name, frame_pos, frame_rot))

    waypoint_collision_free = bool(collision_free)
    try:
        for q_arm, (_, _, drawer_q) in zip(q_waypoints, waypoint_specs):
            _set_drawer_joint_value(env, drawer_q)
            env.sim.forward()
            collision_ok, _ = _check_arm_q_collision(
                env,
                panel,
                arm_joint_names,
                q_arm,
                allowed_ee_geom_name=ee_contact_geom_name,
                penetration_tolerance=args.mink_collision_penetration_tolerance,
            )
            waypoint_collision_free = waypoint_collision_free and bool(collision_ok)
    finally:
        _set_drawer_joint_value(env, drawer_q0)
        env.sim.forward()

    return MinkContactPoseSolution(
        drawer_candidate_index=int(candidate_index),
        drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
        drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
        drawer_contact_cost=float(candidate.cost),
        ee_sample_index=int(ee_sample_index),
        ee_sample_name=str(ee_sample_name),
        ee_contact_geom_name=str(ee_contact_geom_name),
        contact_frame=f"{frame_name}:{ee_sample_name}",
        contact_offset_local=np.asarray(contact_offset, dtype=np.float64),
        roll_angle=float(roll_angle),
        q_waypoints=np.asarray(q_waypoints, dtype=np.float64),
        target_gripper_poses=target_gripper_poses,
        contact_position_error=float(max_pos_error),
        collision_free=bool(waypoint_collision_free),
    )


def _solve_mink_for_drawer_candidate(
    env,
    panel,
    candidate,
    candidate_index,
    push_distance,
    robot_model,
    arm_joint_names,
    frame_name,
    q_initial,
    q_posture,
    posture_cost,
    contact_offsets,
    roll_angles,
    base_rot,
    args,
):
    if args.require_feasible_contact and not candidate.feasible:
        report = MinkContactAttemptReport(
            drawer_candidate_index=int(candidate_index),
            drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
            drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
            drawer_contact_cost=float(candidate.cost),
            contact_feasible=bool(candidate.feasible),
            status="failed",
            reason="contact_cost_above_threshold",
            best_ee_sample_index=-1,
            best_ee_sample_name="",
            best_position_error=float("inf"),
            best_collision_free=False,
        )
        return None, report

    best = None
    best_score = np.inf
    best_reason = "not_attempted"
    for ee_sample_index, (
        ee_sample_name,
        contact_offset,
        ee_contact_geom_name,
    ) in enumerate(contact_offsets):
        if ee_sample_name == "gripper_center" and not args.mink_include_grip_site:
            continue
        for roll_angle in roll_angles:
            target_rot = base_rot @ _rot_about_axis([1.0, 0.0, 0.0], roll_angle)
            desired_contact_point = (
                candidate.world_point + panel.outward_world * args.contact_standoff
            )
            target_frame_pos = desired_contact_point - target_rot @ contact_offset
            try:
                q_contact, pos_error = _solve_mink_frame_pose(
                    env,
                    frame_name,
                    target_frame_pos,
                    target_rot,
                    q_initial,
                    q_posture,
                    posture_cost,
                    args,
                )
            except Exception as exc:
                best_reason = f"ik_exception:{exc.__class__.__name__}"
                continue

            q_arm = _arm_q_from_robot_model_q(robot_model, q_contact, arm_joint_names)
            collision_free, collision_reason = _check_arm_q_collision(
                env,
                panel,
                arm_joint_names,
                q_arm,
                allowed_ee_geom_name=ee_contact_geom_name,
                penetration_tolerance=args.mink_collision_penetration_tolerance,
            )
            within_tolerance = bool(pos_error <= args.mink_position_tolerance)
            score = float(pos_error) + (
                0.0 if collision_free else args.mink_collision_penalty
            )
            if score < best_score:
                best_score = score
                best = (
                    ee_sample_index,
                    ee_sample_name,
                    ee_contact_geom_name,
                    contact_offset,
                    roll_angle,
                    target_rot,
                    q_contact,
                    float(pos_error),
                    bool(collision_free),
                    collision_reason,
                )
                if not within_tolerance:
                    best_reason = "position_error"
                elif not collision_free:
                    best_reason = collision_reason
                else:
                    best_reason = "success"

            if within_tolerance and collision_free:
                solution = _build_mink_solution(
                    env,
                    panel,
                    candidate,
                    candidate_index,
                    push_distance,
                    robot_model,
                    arm_joint_names,
                    frame_name,
                    ee_sample_index,
                    ee_sample_name,
                    ee_contact_geom_name,
                    contact_offset,
                    target_rot,
                    roll_angle,
                    q_initial,
                    q_posture,
                    posture_cost,
                    pos_error,
                    collision_free,
                    args,
                )
                if (
                    solution.contact_position_error <= args.mink_position_tolerance
                    and solution.collision_free
                ):
                    report = MinkContactAttemptReport(
                        drawer_candidate_index=int(candidate_index),
                        drawer_contact_world=np.asarray(
                            candidate.world_point, dtype=np.float64
                        ),
                        drawer_contact_local=np.asarray(
                            candidate.local_point, dtype=np.float64
                        ),
                        drawer_contact_cost=float(candidate.cost),
                        contact_feasible=bool(candidate.feasible),
                        status="success",
                        reason="success",
                        best_ee_sample_index=int(ee_sample_index),
                        best_ee_sample_name=str(ee_sample_name),
                        best_position_error=float(solution.contact_position_error),
                        best_collision_free=True,
                    )
                    return solution, report
                best_reason = (
                    "position_error"
                    if solution.contact_position_error > args.mink_position_tolerance
                    else "trajectory_collision"
                )

    if best is None:
        report = MinkContactAttemptReport(
            drawer_candidate_index=int(candidate_index),
            drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
            drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
            drawer_contact_cost=float(candidate.cost),
            contact_feasible=bool(candidate.feasible),
            status="failed",
            reason=best_reason,
            best_ee_sample_index=-1,
            best_ee_sample_name="",
            best_position_error=float("inf"),
            best_collision_free=False,
        )
        return None, report

    (
        ee_sample_index,
        ee_sample_name,
        _,
        _,
        _,
        _,
        _,
        pos_error,
        collision_free,
        collision_reason,
    ) = best
    if pos_error > args.mink_position_tolerance:
        reason = "position_error"
    elif collision_free:
        reason = best_reason if best_reason != "success" else "trajectory_collision"
    else:
        reason = collision_reason
    report = MinkContactAttemptReport(
        drawer_candidate_index=int(candidate_index),
        drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
        drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
        drawer_contact_cost=float(candidate.cost),
        contact_feasible=bool(candidate.feasible),
        status="failed",
        reason=reason,
        best_ee_sample_index=int(ee_sample_index),
        best_ee_sample_name=str(ee_sample_name),
        best_position_error=float(pos_error),
        best_collision_free=bool(collision_free),
    )
    return None, report


def _drawer_body_geom_ids(env, *, exclude_inflating=True):
    model = env.sim.model
    drawer_name = env.drawer.name
    inflating = _inflating_handle_geom_names(env) if exclude_inflating else set()
    return {
        int(gid)
        for name, gid in model._geom_name2id.items()
        if (
            name.startswith(f"{drawer_name}_door_")
            or name.startswith(f"{drawer_name}_door")
        )
        and name not in inflating
    }


def _scene_geom_ids_for_skeleton(env, panel):
    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    excluded = set(_target_panel_geom_ids(env, panel)) | _drawer_body_geom_ids(env)
    return tuple(gid for gid in range(int(raw_model.ngeom)) if int(gid) not in excluded)


def _strict_robot_drawer_penetration(env, arm_joint_names, q_arm):
    model = env.sim.model
    data = env.sim.data
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    robot_geoms = {
        int(geom_id)
        for name, geom_id in model._geom_name2id.items()
        if (name.startswith("robot0_") or name.startswith("gripper0_"))
        and "collision" in name
    }
    drawer_geoms = set(_drawer_body_geom_ids(env))
    geom_id_to_name = {geom_id: name for name, geom_id in model._geom_name2id.items()}
    max_pen = 0.0
    worst = "collision_free"
    try:
        _set_env_arm_q(env, arm_joint_names, q_arm)
        env.sim.forward()
        for contact_idx in range(int(data.ncon)):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            pair = (geom1 in robot_geoms and geom2 in drawer_geoms) or (
                geom2 in robot_geoms and geom1 in drawer_geoms
            )
            if not pair:
                continue
            penetration = max(0.0, -float(contact.dist))
            if penetration > max_pen:
                max_pen = penetration
                worst = (
                    f"{geom_id_to_name.get(geom1, str(geom1))}"
                    f"--{geom_id_to_name.get(geom2, str(geom2))}:"
                    f"dist={float(contact.dist):.6f}"
                )
        return float(max_pen), worst
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        env.sim.forward()


def _current_site_pose(env, site_name):
    site_id = int(env.sim.model.site_name2id(site_name))
    return (
        np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).reshape(3).copy(),
        np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64)
        .reshape(3, 3)
        .copy(),
    )


def _select_close_panel_interior_feasible_points(
    feasible_cache, panel, n_select, seed, args
):
    positions = np.asarray(feasible_cache.positions_world, dtype=np.float64).reshape(
        -1, 3
    )
    normals = np.asarray(feasible_cache.normals_world, dtype=np.float64).reshape(-1, 3)
    positions_object = np.asarray(
        feasible_cache.positions_object, dtype=np.float64
    ).reshape(-1, 3)
    if positions.shape[0] == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    # Drop Delaunay convex-hull boundary points (the drawer's outer rim) before
    # any further filtering. Matches demo_open_drawer_autogen.py's
    # ee_skelton.select_interior_feasible_points behaviour, which the close
    # path otherwise skipped — leaving top-edge points in the candidate set.
    is_edge = np.asarray(
        getattr(feasible_cache, "is_edge", np.zeros(positions.shape[0], dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if is_edge.shape[0] != positions.shape[0]:
        is_edge = np.zeros(positions.shape[0], dtype=bool)
    drop_edge = bool(getattr(args, "autogen_drop_feasible_edge_points", True))
    interior_idx = (
        np.flatnonzero(~is_edge)
        if drop_edge
        else np.arange(positions.shape[0], dtype=np.int64)
    )
    if interior_idx.size == 0:
        interior_idx = np.arange(positions.shape[0], dtype=np.int64)
    positions_pool = positions[interior_idx]
    normals_pool = normals[interior_idx]
    positions_object_pool = positions_object[interior_idx]

    half_size = np.asarray(panel.half_size, dtype=np.float64).reshape(3)
    edge_fraction = float(getattr(args, "autogen_panel_edge_fraction", 0.28))
    edge_fraction = float(np.clip(edge_fraction, 0.0, 0.45))
    top_edge_fraction = float(getattr(args, "autogen_panel_top_edge_fraction", 0.38))
    top_edge_fraction = float(np.clip(top_edge_fraction, edge_fraction, 0.48))
    base_margin = float(getattr(args, "autogen_panel_edge_margin", 0.05))
    x_margin = max(base_margin, edge_fraction * max(float(half_size[0]), 1e-6))
    z_margin_bottom = max(base_margin, edge_fraction * max(float(half_size[2]), 1e-6))
    z_margin_top = max(base_margin, top_edge_fraction * max(float(half_size[2]), 1e-6))
    x_lim = max(float(half_size[0]) - x_margin, 0.0)
    z_low = -float(half_size[2]) + z_margin_bottom
    z_high = float(half_size[2]) - z_margin_top
    mask = (
        (np.abs(positions_object_pool[:, 0]) <= x_lim)
        & (positions_object_pool[:, 2] >= z_low)
        & (positions_object_pool[:, 2] <= z_high)
    )
    selected_pool = np.flatnonzero(mask)
    if selected_pool.size == 0:
        # If the physical panel is very small, fall back to percentile trimming
        # rather than accepting all hull/edge points.
        x_abs = np.abs(positions_object_pool[:, 0])
        x_cut = float(np.quantile(x_abs, max(0.0, 1.0 - edge_fraction)))
        z_low_cut = float(np.quantile(positions_object_pool[:, 2], edge_fraction))
        z_high_cut = float(
            np.quantile(positions_object_pool[:, 2], 1.0 - top_edge_fraction)
        )
        selected_pool = np.flatnonzero(
            (x_abs <= x_cut)
            & (positions_object_pool[:, 2] >= z_low_cut)
            & (positions_object_pool[:, 2] <= z_high_cut)
        )
    selected_ids = interior_idx[selected_pool]
    if selected_ids.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    n_select = int(max(1, min(int(n_select), selected_ids.size)))
    pts = positions[selected_ids]
    rng = np.random.default_rng(int(seed))
    init = int(rng.integers(pts.shape[0]))
    fps_local = farthest_point_subset(pts, n_select, initial_index=init)
    local_ids = selected_ids[np.asarray(fps_local, dtype=np.int64)]
    return local_ids.astype(np.int64), positions[local_ids], normals[local_ids]


def _print_skeleton_qp_debug(debug_log, label="qp"):
    if not debug_log:
        print(f"SKELETON_QP_DEBUG label={label} entries=0", flush=True)
        return
    by_key = {}
    for entry in debug_log:
        primitive = str(entry.get("primitive", "unknown"))
        status = int(entry.get("status", -999))
        rejected = str(entry.get("rejected", ""))
        key = (primitive, status, rejected)
        by_key[key] = by_key.get(key, 0) + 1
    parts = []
    for (primitive, status, rejected), count in sorted(by_key.items()):
        suffix = f":{rejected}" if rejected else ""
        parts.append(f"{primitive}/status={status}{suffix}:{count}")
    print(
        f"SKELETON_QP_DEBUG label={label} entries={len(debug_log)} " + " ".join(parts),
        flush=True,
    )


def _theta_diverse_skeleton_pose_order(skeleton_poses):
    if not skeleton_poses:
        return []
    buckets = {}
    for candidate_index, pose in skeleton_poses:
        theta = float(getattr(pose, "theta", 0.0))
        key = int(np.floor(((theta % (2.0 * np.pi)) / (2.0 * np.pi)) * 12.0))
        buckets.setdefault(key, []).append((candidate_index, pose))
    for poses in buckets.values():
        poses.sort(key=lambda item: float(getattr(item[1], "qp_cost", 0.0)))
    ordered = []
    keys = sorted(buckets)
    while keys:
        next_keys = []
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return ordered


def _print_skeleton_theta_histogram(skeleton_poses):
    if not skeleton_poses:
        print("SKELETON_THETA_HIST total=0", flush=True)
        return
    hist = {}
    for _, pose in skeleton_poses:
        primitive = str(
            getattr(pose, "contact_primitive", getattr(pose, "contact_finger", ""))
        )
        bucket = int(
            np.floor(
                ((float(getattr(pose, "theta", 0.0)) % (2.0 * np.pi)) / (2.0 * np.pi))
                * 12.0
            )
        )
        hist[(primitive, bucket)] = hist.get((primitive, bucket), 0) + 1
    parts = [
        f"{prim}[{bucket}]={count}" for (prim, bucket), count in sorted(hist.items())
    ]
    print(
        f"SKELETON_THETA_HIST total={len(skeleton_poses)} " + " ".join(parts),
        flush=True,
    )


def _rotation_angle_between(rot_a, rot_b):
    rel = np.asarray(rot_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        rot_b, dtype=np.float64
    ).reshape(3, 3)
    c = 0.5 * (float(np.trace(rel)) - 1.0)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _print_skeleton_rotation_debug(skeleton_poses, label="poses"):
    counts = {"left_finger": 0, "right_finger": 0, "hand": 0}
    rotations = []
    for _, pose in skeleton_poses:
        primitive = str(
            getattr(pose, "contact_primitive", getattr(pose, "contact_finger", ""))
        )
        if primitive.startswith("hand"):
            bucket = "hand"
        elif primitive.startswith("left_finger"):
            bucket = "left_finger"
        elif primitive.startswith("right_finger"):
            bucket = "right_finger"
        else:
            bucket = primitive
        counts[bucket] = counts.get(bucket, 0) + 1
        rotations.append(np.asarray(pose.ee_rotation, dtype=np.float64).reshape(3, 3))
    if len(rotations) >= 2:
        angles = [
            _rotation_angle_between(rotations[i], rotations[j])
            for i in range(len(rotations))
            for j in range(i + 1, len(rotations))
        ]
        min_angle = float(np.min(angles))
        median_angle = float(np.median(angles))
        max_angle = float(np.max(angles))
    else:
        min_angle = median_angle = max_angle = 0.0
    print(
        f"SKELETON_ROTATION_DEBUG label={label} total={len(skeleton_poses)} "
        f"left_finger={counts.get('left_finger', 0)} "
        f"right_finger={counts.get('right_finger', 0)} "
        f"hand={counts.get('hand', 0)} "
        f"pair_angle_min={min_angle:.6f} "
        f"pair_angle_median={median_angle:.6f} "
        f"pair_angle_max={max_angle:.6f}",
        flush=True,
    )
    for idx, (_, pose) in enumerate(skeleton_poses[:12]):
        rot = np.asarray(pose.ee_rotation, dtype=np.float64).reshape(3, 3)
        primitive = str(
            getattr(pose, "contact_primitive", getattr(pose, "contact_finger", ""))
        )
        print(
            "SKELETON_ROTATION "
            f"idx={idx} primitive={primitive} finger={getattr(pose, 'contact_finger', '')} "
            f"theta={float(getattr(pose, 'theta', 0.0)):.6f} "
            f"cost={float(getattr(pose, 'qp_cost', 0.0)):.6f} "
            f"R={np.array2string(rot, precision=4, suppress_small=True)}",
            flush=True,
        )


def _skeleton_solution_to_contact_solution(
    env,
    panel,
    candidate,
    candidate_index,
    push_distance,
    robot_state,
    frame_name,
    skeleton_pose,
    q_precontact,
    g_best,
    pos_err,
    rot_err,
    max_penetration,
    args,
    contact_anchor_pose=None,
):
    normal = _normalize(candidate.approach_world, fallback=panel.outward_world)
    if float(np.dot(normal, panel.outward_world)) < 0.0:
        normal = -normal
    lift = float(
        getattr(
            skeleton_pose, "lift", getattr(args, "autogen_skeleton_initial_lift", 0.005)
        )
    )
    contact_shift = (float(args.contact_standoff) - lift) * normal
    push_shift = contact_shift + np.asarray(panel.push_world, dtype=np.float64) * float(
        push_distance
    )

    pre_pos = np.asarray(skeleton_pose.ee_position, dtype=np.float64).reshape(3)
    rot = np.asarray(skeleton_pose.ee_rotation, dtype=np.float64).reshape(3, 3)
    execute_push = bool(getattr(args, "execute_push_stage", False))
    target_gripper_poses = [
        ("precontact", pre_pos, rot),
        ("contact", pre_pos + contact_shift, rot),
    ]
    if execute_push:
        target_gripper_poses.append(("push", pre_pos + push_shift, rot))

    q_waypoints = [np.asarray(q_precontact, dtype=np.float64).reshape(7)]
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    try:
        _set_env_arm_q(env, tuple(robot_state["robocasa_joint_names"]), q_waypoints[0])
        _set_drawer_joint_value(env, _drawer_joint_value(env))
        env.sim.forward()
        actual_pre_pos, actual_pre_rot = _current_site_pose(env, frame_name)
        _outward_world_dbg = np.asarray(panel.outward_world, dtype=np.float64).reshape(
            3
        )
        _outward_norm_dbg = float(np.linalg.norm(_outward_world_dbg))
        _outward_unit = (
            _outward_world_dbg / _outward_norm_dbg
            if _outward_norm_dbg > 1e-12
            else np.zeros(3, dtype=np.float64)
        )
        _skel_target = np.asarray(skeleton_pose.ee_position, dtype=np.float64).reshape(
            3
        )
        _actual_outward = float(
            np.dot(actual_pre_pos - candidate.world_point, _outward_unit)
        )
        _penetration_tol = float(
            getattr(
                args, "autogen_skeleton_max_penetration", float(args.contact_standoff)
            )
        )
        _use_actual = _actual_outward >= -_penetration_tol
        if contact_anchor_pose is not None:
            anchor_pos, anchor_rot = contact_anchor_pose
            anchor_pos = np.asarray(anchor_pos, dtype=np.float64).reshape(3)
            anchor_rot = np.asarray(anchor_rot, dtype=np.float64).reshape(3, 3)
            target_gripper_poses[0] = ("precontact", actual_pre_pos, actual_pre_rot)
            target_gripper_poses[1] = (
                "contact",
                anchor_pos + contact_shift,
                anchor_rot,
            )
            if execute_push:
                target_gripper_poses[2] = ("push", anchor_pos + push_shift, anchor_rot)
        elif _use_actual:
            target_gripper_poses[0] = ("precontact", actual_pre_pos, actual_pre_rot)
            rot = actual_pre_rot
            target_gripper_poses[1] = ("contact", actual_pre_pos + contact_shift, rot)
            if execute_push:
                target_gripper_poses[2] = ("push", actual_pre_pos + push_shift, rot)
        else:
            target_gripper_poses[0] = ("precontact", _skel_target, rot)
            target_gripper_poses[1] = ("contact", _skel_target + contact_shift, rot)
            if execute_push:
                target_gripper_poses[2] = ("push", _skel_target + push_shift, rot)
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()

    robot_model = env.robots[0].robot_model.mujoco_model
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    q_robot = _current_robot_model_q(env, robot_model)
    for joint_name, value in zip(arm_joint_names, q_precontact):
        if _mj_has_name(robot_model, "joint", joint_name):
            q_robot[int(robot_model.joint(joint_name).qposadr[0])] = float(value)
    q_posture = q_robot.copy()
    posture_cost = _make_mink_posture_cost(robot_model, arm_joint_names, args)
    for _, target_pos, target_rot in target_gripper_poses[1:]:
        try:
            q_robot, _ = _solve_mink_frame_pose(
                env,
                frame_name,
                target_pos,
                target_rot,
                q_robot,
                q_posture,
                posture_cost,
                args,
            )
            q_waypoints.append(
                _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
            )
        except Exception:
            q_waypoints.append(q_waypoints[-1].copy())

    # cuRobo plans to EE poses, not directly to the q_waypoints above. For the
    # mink route, make those pose goals exactly match the MuJoCo FK of the mink
    # q solutions so the planner is not asked to solve a different skeleton
    # anchor/contact pose than the one mink already found.
    fk_target_gripper_poses = []
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    try:
        for (name, _, _), q_arm in zip(target_gripper_poses, q_waypoints):
            _set_env_arm_q(env, arm_joint_names, q_arm)
            fk_pos, fk_rot = _current_site_pose(env, frame_name)
            fk_target_gripper_poses.append((name, fk_pos, fk_rot))
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()
    if len(fk_target_gripper_poses) == len(target_gripper_poses):
        target_gripper_poses = fk_target_gripper_poses

    collision_free = bool(
        float(max_penetration) <= float(args.mink_collision_penetration_tolerance)
    )
    solution = MinkContactPoseSolution(
        drawer_candidate_index=int(candidate_index),
        drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
        drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
        drawer_contact_cost=float(candidate.cost),
        ee_sample_index=0,
        ee_sample_name=f"skeleton_{getattr(skeleton_pose, 'contact_primitive', skeleton_pose.contact_finger)}",
        ee_contact_geom_name="skeleton",
        contact_frame=f"{frame_name}:skeleton_{getattr(skeleton_pose, 'contact_primitive', skeleton_pose.contact_finger)}",
        contact_offset_local=np.zeros(3, dtype=np.float64),
        roll_angle=float(getattr(skeleton_pose, "theta", 0.0)),
        q_waypoints=np.asarray(q_waypoints, dtype=np.float64),
        target_gripper_poses=target_gripper_poses,
        contact_position_error=float(pos_err),
        collision_free=collision_free,
    )
    solution.rotation_error = float(rot_err)
    if g_best is not None:
        solution.gripper_opening = float(g_best)
    return solution


def _mppi_terminal_ee_solution(
    panel,
    candidate,
    candidate_index,
    frame_name,
    skeleton_pose,
    refined_pos,
    refined_rot,
    refined_g,
    max_penetration,
    args,
):
    primitive = getattr(
        skeleton_pose,
        "contact_primitive",
        getattr(skeleton_pose, "contact_finger", "unknown"),
    )
    solution = MinkContactPoseSolution(
        drawer_candidate_index=int(candidate_index),
        drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
        drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
        drawer_contact_cost=float(candidate.cost),
        ee_sample_index=0,
        ee_sample_name=f"mppi_terminal_{primitive}",
        ee_contact_geom_name="mppi_terminal",
        contact_frame=f"{frame_name}:mppi_terminal_{primitive}",
        contact_offset_local=np.zeros(3, dtype=np.float64),
        roll_angle=float(getattr(skeleton_pose, "theta", 0.0)),
        q_waypoints=np.zeros((0, 7), dtype=np.float64),
        target_gripper_poses=[
            (
                "mppi_terminal",
                np.asarray(refined_pos, dtype=np.float64).reshape(3).copy(),
                np.asarray(refined_rot, dtype=np.float64).reshape(3, 3).copy(),
            )
        ],
        contact_position_error=0.0,
        collision_free=bool(
            bool(getattr(args, "autogen_qmppi_accept_object_improvement_only", True))
            or float(max_penetration)
            <= float(getattr(args, "mink_collision_penetration_tolerance", 0.02))
        ),
    )
    solution.rotation_error = 0.0
    solution.gripper_opening = float(refined_g)
    return solution


# Populated by _solve_contact_poses_with_skeleton with (pos, rot, gripper, candidate_index)
# tuples for every FloatingEEMPPI-accepted candidate. Consumed by autogen wrappers
# that want to render ghost-EE previews of the refined poses.
LAST_FLOATING_REFINED_POSES: list = []


def _solve_contact_poses_with_skeleton(
    env, panel, candidates, push_distance, robot_state, args
):
    LAST_FLOATING_REFINED_POSES.clear()
    frame_name = str(args.mink_contact_frame).split(":")[0]
    if frame_name not in env.sim.model._site_name2id:
        raise RuntimeError(
            f"mink contact frame '{frame_name}' is not present in the simulation model."
        )
    feasible_cache = getattr(args, "_autogen_feasible_cache", None)
    if feasible_cache is None:
        _, _, feasible_cache = _build_autogen_contact_candidates(
            env, panel, push_distance, args
        )

    skeleton = ee_skelton.build_panda_skeleton(env, frame_name)
    if bool(getattr(args, "autogen_visualize_skeleton", False)):
        ee_skelton.visualize_skeleton_and_ee(env, frame_name, skeleton, args)

    select_count = int(getattr(args, "autogen_initial_pose_count", 200))
    (
        local_ids,
        points_world,
        normals_world,
    ) = _select_close_panel_interior_feasible_points(
        feasible_cache,
        panel,
        select_count,
        int(args.seed),
        args,
    )
    feasible_candidate_indices = np.asarray(
        feasible_cache.candidate_indices, dtype=np.int64
    )
    feasible_row_by_candidate = {
        int(candidate_index): int(row_index)
        for row_index, candidate_index in enumerate(feasible_candidate_indices)
    }
    scene_geom_ids = _scene_geom_ids_for_skeleton(env, panel)
    object_eqs = getattr(feasible_cache, "handle_convex_equations", None)
    initial_rot = _current_site_pose(env, frame_name)[1]

    def _collect_skeleton_poses(use_current_rotation):
        collected = []
        variants_per_contact = max(
            1, int(getattr(args, "autogen_skeleton_pose_variants_per_contact", 4))
        )
        theta_sep = float(
            getattr(args, "autogen_skeleton_pose_min_theta_separation", np.pi / 6.0)
        )
        primitive_specs = (
            ("left_finger", "left"),
            ("right_finger", "right"),
            ("hand", "left"),
        )
        jobs = []
        for local_id, point, normal in zip(local_ids, points_world, normals_world):
            candidate_index = int(feasible_candidate_indices[int(local_id)])
            for primitive, finger in primitive_specs:
                jobs.append(
                    (
                        int(candidate_index),
                        np.asarray(point, dtype=np.float64).reshape(3).copy(),
                        np.asarray(normal, dtype=np.float64).reshape(3).copy(),
                        str(primitive),
                        str(finger),
                    )
                )

        skeleton_workers = getattr(args, "autogen_skeleton_parallel_workers", None)
        if skeleton_workers is None:
            skeleton_workers = getattr(args, "autogen_mink_parallel_workers", 1)
        workers = int(skeleton_workers or 1)
        active_workers = max(1, min(workers, len(jobs) or 1))
        verbose_daqp = bool(getattr(args, "autogen_skeleton_daqp_verbose", True))
        scene_pool = getattr(args, "autogen_skeleton_scene_pool", None)
        if scene_pool is not None:
            try:
                scene_pool.reset()
            except Exception as exc:
                print(f"[skeleton_scene_pool] reset failed: {exc}", flush=True)
        if verbose_daqp:
            print(
                "SKELETON_POSE_SOLVER "
                f"backend=daqp workers={active_workers} "
                f"jobs={len(jobs)} "
                f"use_current_rotation={bool(use_current_rotation)}",
                flush=True,
            )

        def _solve_job(job):
            candidate_index, point, normal, primitive, finger = job
            try:
                if scene_pool is not None:
                    with scene_pool.borrow() as raw_model_data:
                        poses = ee_skelton.solve_skeleton_pose_candidates(
                            env,
                            skeleton,
                            point,
                            normal,
                            finger=finger,
                            contact_primitive=primitive,
                            object_convex_equations=object_eqs,
                            object_convex_equation_mask=None,
                            scene_geom_ids=scene_geom_ids,
                            initial_ee_rotation_world=initial_rot
                            if use_current_rotation
                            else None,
                            args=args,
                            max_candidates=variants_per_contact,
                            min_theta_separation=theta_sep,
                            raw_model_data=raw_model_data,
                        )
                else:
                    poses = ee_skelton.solve_skeleton_pose_candidates(
                        env,
                        skeleton,
                        point,
                        normal,
                        finger=finger,
                        contact_primitive=primitive,
                        object_convex_equations=object_eqs,
                        object_convex_equation_mask=None,
                        scene_geom_ids=scene_geom_ids,
                        initial_ee_rotation_world=initial_rot
                        if use_current_rotation
                        else None,
                        args=args,
                        max_candidates=variants_per_contact,
                        min_theta_separation=theta_sep,
                    )
                return int(candidate_index), list(poses), None
            except Exception as exc:
                return int(candidate_index), [], f"{exc.__class__.__name__}:{exc}"

        use_tqdm = (not verbose_daqp) and len(jobs) > 0
        if use_tqdm:
            try:
                from tqdm import tqdm as _tqdm
            except Exception:
                _tqdm = None
        else:
            _tqdm = None
        pbar = (
            _tqdm(
                total=len(jobs),
                desc=f"daqp (workers={active_workers})",
                unit="job",
                file=sys.__stdout__,
                dynamic_ncols=True,
                miniters=1,
                mininterval=0.2,
                leave=True,
            )
            if _tqdm is not None
            else None
        )

        if workers <= 1 or len(jobs) <= 1:
            results = []
            for job in jobs:
                results.append(_solve_job(job))
                if pbar is not None:
                    pbar.update(1)
        else:
            results_by_index = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=active_workers
            ) as executor:
                future_by_index = {
                    executor.submit(_solve_job, job): idx
                    for idx, job in enumerate(jobs)
                }
                for future in concurrent.futures.as_completed(future_by_index):
                    results_by_index[future_by_index[future]] = future.result()
                    if pbar is not None:
                        pbar.update(1)
            results = [results_by_index[idx] for idx in range(len(jobs))]

        if pbar is not None:
            pbar.close()

        failures = {}
        for candidate_index, poses, error in results:
            if error is not None:
                failures[error] = failures.get(error, 0) + 1
                continue
            for pose in poses:
                collected.append((candidate_index, pose))
        if failures and verbose_daqp:
            print(f"SKELETON_POSE_SOLVER_FAILURES {failures}", flush=True)
        return collected

    daqp_debug = bool(getattr(args, "autogen_skeleton_daqp_verbose", True))
    rotation_debug = bool(
        getattr(
            args,
            "autogen_skeleton_rotation_debug",
            daqp_debug,
        )
    )
    args._skeleton_debug_log = [] if daqp_debug else None
    skeleton_poses = _collect_skeleton_poses(
        bool(getattr(args, "autogen_use_current_ee_rotation", False))
    )
    if daqp_debug:
        _print_skeleton_qp_debug(args._skeleton_debug_log, label="free_rotation")
    if rotation_debug:
        _print_skeleton_rotation_debug(skeleton_poses, label="free_rotation_raw")
    allow_rotation_fallback = bool(
        getattr(args, "autogen_allow_current_rotation_fallback", True)
    )
    if (
        not skeleton_poses
        and not bool(getattr(args, "autogen_use_current_ee_rotation", False))
        and allow_rotation_fallback
    ):
        args._skeleton_debug_log = [] if daqp_debug else None
        skeleton_poses = _collect_skeleton_poses(True)
        if daqp_debug:
            _print_skeleton_qp_debug(args._skeleton_debug_log, label="r0_override")
        if rotation_debug:
            _print_skeleton_rotation_debug(skeleton_poses, label="r0_override_raw")
    skeleton_poses = _theta_diverse_skeleton_pose_order(skeleton_poses)
    if rotation_debug:
        _print_skeleton_rotation_debug(skeleton_poses, label="ordered")
    if daqp_debug:
        _print_skeleton_theta_histogram(skeleton_poses)
    if bool(getattr(args, "autogen_visualize_skeleton_poses", False)):
        max_visualized = int(
            getattr(args, "autogen_visualize_skeleton_pose_limit", 120)
        )
        visualized_poses = [p for _, p in skeleton_poses[:max_visualized]]
        ee_skelton.visualize_skeleton_poses(
            env, frame_name, skeleton, visualized_poses, args
        )

    reports = []
    solutions = []
    rng = np.random.default_rng(int(args.seed) + 29003)
    order = np.arange(len(skeleton_poses), dtype=np.int64)
    rng.shuffle(order)
    q_seed = np.asarray(robot_state["q"], dtype=np.float64).reshape(7)
    drawer_q_now = float(_drawer_joint_value(env))
    pos_tol = float(
        getattr(args, "autogen_accept_position_tolerance", args.mink_position_tolerance)
    )
    pen_tol = float(args.mink_collision_penetration_tolerance)
    started = time.perf_counter()
    robot_model = env.robots[0].robot_model.mujoco_model
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    q_robot_start = _current_robot_model_q(env, robot_model)
    q_posture = q_robot_start.copy()
    posture_cost = _make_mink_posture_cost(robot_model, arm_joint_names, args)
    solve_step2 = str(getattr(args, "solve_step2", "MPPI")).strip().lower()
    if solve_step2 not in ("mppi", "mink"):
        raise ValueError(
            f"Unsupported solve_step2={getattr(args, 'solve_step2', None)!r}; expected MPPI or mink"
        )
    if solve_step2 == "mink":
        # The mink step is intended to solve q for every skeleton pose. Keep the
        # MPPI path's historical cap, but do not truncate skeleton q solving.
        max_attempts = len(order)
    else:
        max_attempts = min(
            int(getattr(args, "autogen_mink_max_attempts", len(order))), len(order)
        )
    yellow = "\033[93m"
    reset_color = "\033[0m"

    def _step2_text(step2_name, text):
        if solve_step2 == step2_name:
            return f"{yellow}{text}{reset_color}"
        return text

    # When False (autogen default), the per-pose solving status is shown via a
    # tqdm progress bar rather than one print per skeleton pose; the final
    # summary (elapsed_s + feasible_q) is still printed in yellow below.
    _step2_verbose_poses = bool(getattr(args, "autogen_step2_verbose_poses", True))

    _tqdm_pbar = None
    if not _step2_verbose_poses:
        step2_desc = "mink" if solve_step2 == "mink" else f"step2[{solve_step2}]"
        _tqdm_pbar = tqdm(
            order[:max_attempts],
            desc=step2_desc,
            unit="skel",
            file=sys.__stdout__,
            dynamic_ncols=True,
            miniters=1,
            mininterval=0.2,
            leave=True,
        )

    _step2_last_status = {
        "tag": "",
        "candidate": -1,
        "pen": None,
        "pos": None,
        "obj": None,
    }
    _step2_postfix_last_s = [0.0]
    _step2_postfix_interval_s = float(
        getattr(args, "autogen_step2_tqdm_postfix_interval", 0.25)
    )

    def _step2_pbar_post(tag, candidate_index, pen=None, pos=None, obj=None):
        """Update tqdm postfix (non-verbose) — no-op when verbose printing."""
        _step2_last_status["tag"] = tag
        _step2_last_status["candidate"] = int(candidate_index)
        _step2_last_status["pen"] = pen
        _step2_last_status["pos"] = pos
        _step2_last_status["obj"] = obj
        if _tqdm_pbar is None:
            return
        now_s = time.perf_counter()
        if (
            _step2_postfix_last_s[0] > 0.0
            and now_s - _step2_postfix_last_s[0] < _step2_postfix_interval_s
        ):
            return
        _step2_postfix_last_s[0] = now_s
        if solve_step2 == "mink":
            acc = mink_q_accepted_count
            rej = mink_q_rejected_count
        else:
            acc = successful_floating_ee_count
            rej = rejected_floating_ee_count
        parts = [f"cand={int(candidate_index)}", tag]
        if pen is not None:
            parts.append(f"pen={float(pen):.4f}")
        if pos is not None:
            parts.append(f"pos={float(pos):.4f}")
        if obj is not None:
            try:
                parts.append(f"obj={float(obj):.2e}")
            except (TypeError, ValueError):
                parts.append(f"obj={obj}")
        parts.append(f"acc={acc}")
        parts.append(f"rej={rej}")
        _tqdm_pbar.set_postfix_str(" ".join(parts))

    print(
        f"STEP2_SOLVER "
        f"{_step2_text('mppi', 'MPPI=active' if solve_step2 == 'mppi' else 'MPPI=inactive')} "
        f"{_step2_text('mink', 'mink=active' if solve_step2 == 'mink' else 'mink=inactive')} "
        f"mink_solver={getattr(args, 'mink_solver', None)}",
        flush=True,
    )

    def _candidate_force_direction_world(candidate_index, candidate):
        fallback = _normalize(panel.push_world, fallback=[1.0, 0.0, 0.0])
        row = feasible_row_by_candidate.get(int(candidate_index))
        if row is None:
            return fallback
        try:
            lam = np.asarray(
                getattr(candidate, "lam", np.zeros(3)), dtype=np.float64
            ).reshape(-1)
            if lam.size < 3:
                return fallback
            normal = np.asarray(
                feasible_cache.normals_world[row], dtype=np.float64
            ).reshape(3)
            t1 = np.asarray(
                feasible_cache.tangents1_world[row], dtype=np.float64
            ).reshape(3)
            t2 = np.asarray(
                feasible_cache.tangents2_world[row], dtype=np.float64
            ).reshape(3)
            direction = lam[0] * normal + lam[1] * t1 + lam[2] * t2
            if float(np.dot(direction, panel.push_world)) < 0.0:
                direction = -direction
            return _normalize(direction, fallback=fallback)
        except Exception:
            return fallback

    # --- FloatingEEMPPI setup (single instance per call; reused per candidate)
    # Drawer body / qpos for target object position computation.
    _drawer = getattr(env, "drawer", None)
    _drawer_body_id = -1
    _drawer_slide_axis_world = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    _drawer_qpos_addr_close = None
    if _drawer is not None and len(_drawer.door_joint_names) > 0:
        try:
            _dj = _drawer.door_joint_names[0]
            _drawer_qpos_addr_close = int(env.sim.model.get_joint_qpos_addr(_dj))
            _jid = env.sim.model.joint_name2id(_dj)
            _drawer_slide_axis_world = np.asarray(
                env.sim.model.jnt_axis[_jid], dtype=np.float64
            ).reshape(3)
            _drawer_body_id = int(env.sim.model.jnt_bodyid[_jid])
        except Exception:
            _drawer_body_id = -1
    # Target drawer qpos = current + push_distance (clamped to 0 = fully closed).
    _drawer_q_target = min(float(drawer_q_now) + float(push_distance), 0.0)
    _current_drawer_body_pos = (
        np.asarray(env.sim.data.body_xpos[_drawer_body_id], dtype=np.float64)
        if _drawer_body_id >= 0
        else np.zeros(3)
    )
    _target_object_pos = (
        _current_drawer_body_pos
        + (_drawer_q_target - float(drawer_q_now)) * _drawer_slide_axis_world
    )

    # TODO(refactor): no hand-only xml; FloatingEEMPPI uses the env model and
    # masks contacts to {hand subtree, drawer subtree}. Pass full env model id.
    floating_mppi = None
    floating_mppi_unavailable_reason = ""
    successful_floating_ee_count = 0
    attempted_floating_ee_count = 0
    rejected_floating_ee_count = 0
    floating_reject_reasons: dict[str, int] = {}
    floating_exception_reasons: dict[str, int] = {}
    floating_pen_values: list[float] = []
    floating_contact_values: list[float] = []
    floating_obj_delta_values: list[float] = []
    fallback_skeleton_ee_count = 0
    best_floating_obj_delta = float("inf")
    if solve_step2 == "mppi":
        try:
            # Approach world: use panel push direction as the close direction.
            _approach_world = np.asarray(panel.push_world, dtype=np.float64).reshape(3)
            floating_mppi = FloatingEEMPPI(
                env,
                hand_xml_path=None,
                finger_geom_names=(),
                object_body_id=int(_drawer_body_id),
                ee_site_name=frame_name,
                config=FloatingEEConfig(
                    seed=int(getattr(args, "autogen_qmppi_seed", args.seed)),
                    num_samples=int(getattr(args, "autogen_qmppi_num_samples", 256)),
                    max_num_iterations=int(
                        getattr(args, "autogen_qmppi_num_iterations", 6)
                    ),
                    elite_ratio=float(getattr(args, "autogen_qmppi_elite_ratio", 0.1)),
                    temperature=float(getattr(args, "autogen_qmppi_temperature", 1.0)),
                    gripper_noise_scale=float(
                        getattr(args, "autogen_qmppi_gripper_noise_scale", 0.005)
                    ),
                    gripper_min=float(getattr(args, "autogen_qmppi_gripper_min", 0.0)),
                    gripper_max=float(getattr(args, "autogen_qmppi_gripper_max", 0.08)),
                    pen_threshold=float(
                        getattr(
                            args,
                            "autogen_qmppi_penetration_threshold",
                            min(
                                float(getattr(args, "contact_standoff", 0.005)),
                                float(
                                    getattr(
                                        args,
                                        "mink_collision_penetration_tolerance",
                                        0.02,
                                    )
                                ),
                            ),
                        )
                    ),
                    contact_tolerance=float(
                        getattr(args, "autogen_qmppi_contact_tolerance", 0.002)
                    ),
                    pen_weight=float(
                        getattr(args, "autogen_qmppi_penetration_weight", 0.0)
                    ),
                    contact_weight=float(
                        getattr(args, "autogen_qmppi_contact_weight", 0.0)
                    ),
                    track_pos_weight=float(
                        getattr(args, "autogen_qmppi_pos_weight", 200.0)
                    ),
                    track_rot_weight=float(
                        getattr(args, "autogen_qmppi_rot_weight", 20.0)
                    ),
                    track_gripper_weight=float(
                        getattr(args, "autogen_qmppi_gripper_tracking_weight", 50.0)
                    ),
                    accept_object_improvement_only=bool(
                        getattr(
                            args, "autogen_qmppi_accept_object_improvement_only", True
                        )
                    ),
                ),
                approach_world=_approach_world,
                target_object_position=_target_object_pos,
                drawer_qpos_addr=_drawer_qpos_addr_close,
                drawer_qpos_value=float(drawer_q_now),
            )
            print(
                _step2_text(
                    "mppi",
                    "FLOATING_EE_MPPI_READY "
                    f"backend={floating_mppi.backend_kind} "
                    f"horizon={floating_mppi.config.horizon_steps} "
                    f"approach_total={floating_mppi.config.approach_total_distance:.6f} "
                    f"accept_object_improvement_only={floating_mppi.config.accept_object_improvement_only} "
                    f"pen_weight={floating_mppi.config.pen_weight:.6g} "
                    f"contact_weight={floating_mppi.config.contact_weight:.6g}",
                ),
                flush=True,
            )
        except Exception as exc:
            floating_mppi_unavailable_reason = f"{exc.__class__.__name__}:{exc}"
            sys.stderr.write(
                f"[close_drawer] FloatingEEMPPI unavailable; using skeleton pose as-is: {exc!r}\n"
            )
            sys.stderr.flush()
            floating_mppi = None
    else:
        print(
            _step2_text(
                "mink",
                "MINK_Q_READY "
                f"precontact_distance={float(getattr(args, 'mink_q_precontact_distance', min(max(float(getattr(args, 'contact_standoff', 0.005)), 0.003), 0.008))):.6f} "
                f"penetration_tolerance={getattr(args, 'mink_q_collision_penetration_tolerance', None)} "
                f"retreat_multipliers={getattr(args, 'mink_q_retreat_distance_multipliers', None)}",
            ),
            flush=True,
        )

    # --- FloatingEERollout setup -------------------------------------------------
    # The approach rollout ("translate refined pose along the force direction and
    # report the object-cost delta") lives in demos/rollout.py and is what decides
    # whether a candidate actually pushes the drawer toward the target. One
    # Rollout per distinct force direction (the stripped model is built around the
    # approach axis), lazily constructed and reused across candidates.
    _rollouts_by_direction: dict[tuple[float, ...], FloatingEERollout] = {}
    _rollout_failed: bool = False

    def _get_rollout(force_direction: np.ndarray) -> FloatingEERollout | None:
        nonlocal _rollout_failed
        if _rollout_failed:
            return None
        key = tuple(
            np.round(
                np.asarray(force_direction, dtype=np.float64).reshape(3), 6
            ).tolist()
        )
        cached = _rollouts_by_direction.get(key)
        if cached is not None:
            return cached
        try:
            r = FloatingEERollout(
                env,
                hand_xml_path=None,
                finger_geom_names=(),
                object_body_id=int(_drawer_body_id),
                ee_site_name=frame_name,
                config=RolloutConfig(
                    sim_dt=float(getattr(args, "autogen_qmppi_sim_dt", 0.05)),
                    horizon_steps=int(getattr(args, "autogen_qmppi_horizon_steps", 30)),
                    approach_total_distance=float(
                        getattr(
                            args,
                            "autogen_qmppi_approach_total_distance",
                            float(push_distance),
                        )
                    ),
                    object_improvement_eps=float(
                        getattr(args, "autogen_qmppi_object_improvement_eps", 1e-5)
                    ),
                    score_accept_threshold=float(
                        getattr(args, "autogen_qmppi_score_accept_threshold", 0.15)
                    ),
                ),
                approach_world=_normalize(force_direction, fallback=panel.push_world),
                target_object_position=_target_object_pos,
            )
            if _step2_verbose_poses:
                print(
                    _step2_text(
                        solve_step2,
                        "FLOATING_EE_ROLLOUT_READY "
                        f"backend={r.backend_kind} "
                        f"horizon={r.config.horizon_steps} "
                        f"approach_total={r.config.approach_total_distance:.6g} "
                        f"direction={np.asarray(force_direction, dtype=np.float64).reshape(3)}",
                    ),
                    flush=True,
                )
            _rollouts_by_direction[key] = r
            return r
        except Exception as exc:
            sys.stderr.write(
                f"[close_drawer] FloatingEERollout unavailable ({exc!r}); "
                "object-improvement gating will reject candidates for this direction.\n"
            )
            sys.stderr.flush()
            _rollout_failed = True
            return None

    mink_q_attempted_count = 0
    mink_q_accepted_count = 0
    mink_q_rejected_count = 0
    mink_q_reject_reasons: dict[str, int] = {}
    mink_rollout_rejected_count = 0
    mink_rollout_reject_reasons: dict[str, int] = {}
    mink_q_pen_values: list[float] = []
    mink_q_pos_values: list[float] = []
    mink_q_obj_delta_values: list[float] = []
    best_mink_q_obj_delta = float("inf")
    pending_mink_q_solutions: list[dict] = []
    _mink_q_pen_tol_raw = getattr(args, "mink_q_collision_penetration_tolerance", None)
    _mink_q_scene_pen_tol = (
        min(float(getattr(args, "mink_collision_penetration_tolerance", 0.0)), 1e-4)
        if _mink_q_pen_tol_raw is None
        else float(_mink_q_pen_tol_raw)
    )
    try:
        _mink_q_retreat_count = len(
            mink_q._parse_multipliers(  # type: ignore[attr-defined]
                getattr(args, "mink_q_retreat_distance_multipliers", None)
            )
        )
    except Exception:
        _mink_q_retreat_count = 1
    _mink_q_retreat_count = max(int(_mink_q_retreat_count), 1)

    # Full-scene mjwarp collision checker.  When available, create one
    # independent backend per active mink worker; each backend contains enough
    # worlds to batch that pose's retreat-distance candidates in one forward.
    _mink_q_checker = None
    if (
        solve_step2 == "mink"
        and max_attempts > 0
        and bool(getattr(args, "autogen_mink_q_mjwarp_checker", True))
    ):
        try:
            _mink_q_requested_workers = max(
                int(getattr(args, "autogen_mink_parallel_workers", 1) or 1),
                1,
            )
            _checker_worker_arg = getattr(args, "autogen_mink_q_checker_workers", None)
            if _checker_worker_arg is None:
                # Full-scene mjwarp backend construction is heavy and serialized
                # on a model mutation lock. More than a handful of GPU scene
                # copies tends to slow startup and contact readback more than it
                # helps IK throughput; expose an override for profiling.
                _mink_q_workers = min(_mink_q_requested_workers, max_attempts, 8)
            else:
                _mink_q_workers = min(
                    max(int(_checker_worker_arg), 1),
                    _mink_q_requested_workers,
                    max_attempts,
                )
            _mink_q_workers = max(int(_mink_q_workers), 1)
            _worlds_per_worker_arg = getattr(
                args, "autogen_mink_q_worlds_per_worker", None
            )
            _mink_q_worlds_per_worker = (
                _mink_q_retreat_count
                if _worlds_per_worker_arg is None
                else max(int(_worlds_per_worker_arg), 1)
            )
            _mink_q_checker = FullSceneCollisionCheckerPool.from_env(
                env,
                arm_joint_names=arm_joint_names,
                frame_name=frame_name,
                panel=panel,
                num_workers=_mink_q_workers,
                allowed_ee_geom_name=None,
                penetration_tolerance=_mink_q_scene_pen_tol,
                device=getattr(args, "autogen_mink_q_mjwarp_device", "cuda:0"),
                nconmax_per_env=int(
                    getattr(args, "autogen_mink_q_mjwarp_nconmax", 256)
                ),
                njmax_per_env=int(getattr(args, "autogen_mink_q_mjwarp_njmax", 1024)),
                prefer_comfree=bool(
                    getattr(args, "autogen_mink_q_mjwarp_comfree", True)
                ),
                debug_compare_env=bool(getattr(args, "autogen_mink_q_debug", False)),
                debug_limit=int(getattr(args, "autogen_mink_q_debug_limit", 12)),
                nworld_per_worker=_mink_q_worlds_per_worker,
            )
            print(
                _step2_text(
                    "mink",
                    "MINK_Q_MJWARP_CHECKER_READY "
                    f"requested_workers={_mink_q_requested_workers} "
                    f"checker_workers={_mink_q_workers} "
                    f"worlds_per_worker={_mink_q_worlds_per_worker} "
                    f"retreat_candidates={_mink_q_retreat_count}",
                ),
                file=sys.__stdout__,
                flush=True,
            )
            # Eagerly build every worker's backend BEFORE the outer parallel
            # dispatch. Without this, N outer workers all trigger their first
            # borrow simultaneously and serialize on _BACKEND_BUILD_LOCK inside
            # _build_backend — visually appearing as a hang between DAQP finish
            # and MINK_Q_OUTER_PARALLEL_READY. Warmup here shows a tqdm bar so
            # the wait is attributable.
            try:
                _mink_q_checker.warmup()
            except Exception as _warmup_exc:
                sys.stderr.write(
                    f"[close_drawer] mjwarp pool warmup skipped: {_warmup_exc!r}\n"
                )
                sys.stderr.flush()
        except Exception as exc:
            sys.stderr.write(
                f"[close_drawer] FullSceneMjWarp checker unavailable ({exc!r}); "
                f"falling back to env.sim-backed per-candidate checks.\n"
            )
            sys.stderr.flush()
            _mink_q_checker = None

    def _close_mink_q_checker() -> None:
        nonlocal _mink_q_checker
        checker = _mink_q_checker
        _mink_q_checker = None
        if checker is None:
            return
        close_fn = getattr(checker, "close", None)
        if close_fn is None:
            return
        try:
            close_fn()
        except Exception as exc:
            sys.stderr.write(
                f"[close_drawer] FullSceneMjWarp checker close failed ({exc!r}); continuing.\n"
            )
            sys.stderr.flush()

    def _solve_mink_q_for_pose_index(
        pose_index,
        *,
        scene_checker=None,
        max_workers_override=None,
        serial_env=False,
    ):
        candidate_index, skeleton_pose = skeleton_poses[int(pose_index)]
        candidate = candidates[int(candidate_index)]
        retreat_normal = np.asarray(
            getattr(skeleton_pose, "contact_normal_world", panel.outward_world),
            dtype=np.float64,
        ).reshape(3)
        if float(np.dot(retreat_normal, panel.outward_world)) < 0.0:
            retreat_normal = -retreat_normal
        if scene_checker is not None and not hasattr(
            scene_checker, "evaluate_candidates_threadsafe"
        ):
            scene_checker.reset()
        solver = (
            getattr(
                mink_q,
                "_solve_skeleton_precontact_q_serial_impl",
                mink_q.solve_skeleton_precontact_q,
            )
            if serial_env
            else mink_q.solve_skeleton_precontact_q_parallel
        )
        kwargs = dict(
            env=env,
            robot_model=robot_model,
            arm_joint_names=arm_joint_names,
            frame_name=frame_name,
            skeleton_pose=skeleton_pose,
            q_start=q_robot_start,
            q_posture=q_posture,
            posture_cost=posture_cost,
            args=args,
            retreat_direction_world=retreat_normal,
            penetration_checker=lambda q_arm: _strict_robot_drawer_penetration(
                env, arm_joint_names, q_arm
            ),
            scene_collision_checker=lambda q_arm: _check_arm_q_collision(
                env,
                panel,
                arm_joint_names,
                q_arm,
                allowed_ee_geom_name=None,
                penetration_tolerance=_mink_q_scene_pen_tol,
            ),
        )
        if not serial_env:
            kwargs["scene_checker"] = scene_checker
            kwargs["max_workers"] = max_workers_override
        result = solver(**kwargs)
        return int(candidate_index), candidate, skeleton_pose, result

    _mink_q_debug_success_pose_indices: list[int] = []
    _mink_q_debug_parallel_results: dict[int, object] = {}
    _mink_q_parallel_exceptions: dict[int, Exception] = {}
    _mink_q_outer_precomputed = False
    if solve_step2 == "mink" and bool(getattr(args, "autogen_mink_q_debug", False)):
        debug_limit = max(int(getattr(args, "autogen_mink_q_debug_limit", 20)), 0)
        debug_order = [int(pose_index) for pose_index in order[:max_attempts]]
        print(
            _step2_text(
                "mink",
                "MINK_Q_DEBUG_SERIAL_BEGIN "
                f"poses={len(debug_order)} "
                "backend=mujoco_env",
            ),
            file=sys.__stdout__,
            flush=True,
        )
        debug_serial_results: dict[int, object] = {}
        debug_serial_failures: dict[str, int] = {}
        for pose_index in debug_order:
            try:
                (
                    candidate_index,
                    _,
                    skeleton_pose,
                    serial_result,
                ) = _solve_mink_q_for_pose_index(
                    pose_index,
                    scene_checker=None,
                    max_workers_override=1,
                    serial_env=True,
                )
                debug_serial_results[int(pose_index)] = serial_result
                if bool(serial_result.collision_free):
                    _mink_q_debug_success_pose_indices.append(int(pose_index))
                    print(
                        _step2_text(
                            "mink",
                            "MINK_Q_DEBUG_SERIAL_SUCCESS "
                            f"pose_index={int(pose_index)} "
                            f"candidate={int(candidate_index)} "
                            f"primitive={getattr(skeleton_pose, 'contact_primitive', getattr(skeleton_pose, 'contact_finger', 'unknown'))} "
                            f"pos={float(serial_result.position_error):.6f} "
                            f"rot={float(serial_result.rotation_error):.6f} "
                            f"pen={float(serial_result.max_penetration):.6f} "
                            f"retreat={float(serial_result.retreat_distance):.6f} "
                            f"reason={serial_result.collision_reason}",
                        ),
                        file=sys.__stdout__,
                        flush=True,
                    )
                else:
                    reason = str(serial_result.collision_reason)
                    debug_serial_failures[reason] = (
                        debug_serial_failures.get(reason, 0) + 1
                    )
            except Exception as exc:
                reason = f"{exc.__class__.__name__}:{str(exc)[:120]}"
                debug_serial_failures[reason] = debug_serial_failures.get(reason, 0) + 1
        print(
            _step2_text(
                "mink",
                "MINK_Q_DEBUG_SERIAL_SUMMARY "
                f"attempted={len(debug_order)} "
                f"success={len(_mink_q_debug_success_pose_indices)} "
                f"success_pose_indices={_mink_q_debug_success_pose_indices} "
                f"failures={debug_serial_failures}",
            ),
            file=sys.__stdout__,
            flush=True,
        )

        compare_pose_indices = _mink_q_debug_success_pose_indices[: debug_limit or None]
        print(
            _step2_text(
                "mink",
                "MINK_Q_DEBUG_PARALLEL_BEGIN "
                f"poses={len(compare_pose_indices)} "
                f"workers={getattr(args, 'autogen_mink_parallel_workers', None)} "
                f"mjwarp_checker={_mink_q_checker is not None}",
            ),
            file=sys.__stdout__,
            flush=True,
        )
        parallel_match_count = 0
        parallel_reason_counts: dict[str, int] = {}
        parallel_exception_counts: dict[str, int] = {}
        compare_rows: list[str] = []
        for pose_index in compare_pose_indices:
            serial_result = debug_serial_results[int(pose_index)]
            try:
                (
                    candidate_index,
                    _,
                    skeleton_pose,
                    parallel_result,
                ) = _solve_mink_q_for_pose_index(
                    pose_index,
                    scene_checker=_mink_q_checker,
                    max_workers_override=getattr(
                        args, "autogen_mink_parallel_workers", None
                    ),
                    serial_env=False,
                )
                _mink_q_debug_parallel_results[int(pose_index)] = parallel_result
                q_delta = float(
                    np.linalg.norm(
                        np.asarray(parallel_result.arm_q, dtype=np.float64)
                        - np.asarray(serial_result.arm_q, dtype=np.float64)
                    )
                )
                actual_pos_delta = float(
                    np.linalg.norm(
                        np.asarray(
                            parallel_result.actual_position_world, dtype=np.float64
                        )
                        - np.asarray(
                            serial_result.actual_position_world, dtype=np.float64
                        )
                    )
                )
                pen_delta = float(
                    abs(
                        float(parallel_result.max_penetration)
                        - float(serial_result.max_penetration)
                    )
                )
                match = bool(parallel_result.collision_free)
                if match:
                    parallel_match_count += 1
                parallel_reason = str(parallel_result.collision_reason)
                parallel_reason_counts[parallel_reason] = (
                    parallel_reason_counts.get(parallel_reason, 0) + 1
                )
                serial_last = (
                    str(serial_result.attempts[-1].reason)
                    if serial_result.attempts
                    else str(serial_result.collision_reason)
                )
                parallel_last = (
                    str(parallel_result.attempts[-1].reason)
                    if parallel_result.attempts
                    else str(parallel_result.collision_reason)
                )
                row = (
                    f"pose={int(pose_index)} cand={int(candidate_index)} "
                    f"parallel_ok={bool(parallel_result.collision_free)} "
                    f"q_delta={q_delta:.3g} pos_delta={actual_pos_delta:.3g} "
                    f"pen_delta={pen_delta:.3g} "
                    f"s_pos={float(serial_result.position_error):.4g} "
                    f"p_pos={float(parallel_result.position_error):.4g} "
                    f"s_pen={float(serial_result.max_penetration):.4g} "
                    f"p_pen={float(parallel_result.max_penetration):.4g} "
                    f"s_last={serial_last} p_last={parallel_last} "
                    f"p_reason={parallel_reason}"
                )
                compare_rows.append(row)
                print(
                    _step2_text(
                        "mink",
                        "MINK_Q_DEBUG_COMPARE_SHORT "
                        f"{row} "
                        f"primitive={getattr(skeleton_pose, 'contact_primitive', getattr(skeleton_pose, 'contact_finger', 'unknown'))} ",
                    ),
                    file=sys.__stdout__,
                    flush=True,
                )
            except Exception as exc:
                exc_key = f"{exc.__class__.__name__}:{str(exc)[:120]}"
                parallel_exception_counts[exc_key] = (
                    parallel_exception_counts.get(exc_key, 0) + 1
                )
                print(
                    _step2_text(
                        "mink",
                        "MINK_Q_DEBUG_COMPARE_EXCEPTION "
                        f"pose_index={int(pose_index)} "
                        f"type={exc.__class__.__name__} "
                        f"message={str(exc)[:240]}",
                    ),
                    file=sys.__stdout__,
                    flush=True,
                )
        print(
            _step2_text(
                "mink",
                "MINK_Q_DEBUG_PARALLEL_SUMMARY "
                f"compared={len(compare_pose_indices)} "
                f"parallel_success={parallel_match_count} "
                f"parallel_reasons={parallel_reason_counts} "
                f"parallel_exceptions={parallel_exception_counts} "
                f"first_compare_rows={compare_rows[:min(5, len(compare_rows))]} "
                f"serial_success_pose_indices={_mink_q_debug_success_pose_indices}",
            ),
            file=sys.__stdout__,
            flush=True,
        )
        if _mink_q_debug_success_pose_indices:
            order = np.asarray(_mink_q_debug_success_pose_indices, dtype=np.int64)
            max_attempts = len(order)
        else:
            order = np.asarray([], dtype=np.int64)
            max_attempts = 0
        if _tqdm_pbar is not None:
            _tqdm_pbar.close()
            _tqdm_pbar = tqdm(
                order[:max_attempts],
                desc="mink",
                unit="skel",
                file=sys.__stdout__,
                dynamic_ncols=True,
                miniters=1,
                mininterval=0.2,
                leave=True,
            )

    if (
        solve_step2 == "mink"
        and not bool(getattr(args, "autogen_mink_q_debug", False))
        and int(getattr(args, "autogen_mink_parallel_workers", 1) or 1) > 1
        and max_attempts > 1
    ):
        precompute_pose_indices = [
            int(pose_index) for pose_index in order[:max_attempts]
        ]
        batch_workers = max(
            1,
            min(
                int(getattr(args, "autogen_mink_parallel_workers", 1) or 1),
                len(precompute_pose_indices),
            ),
        )
        if batch_workers > 1:
            _mink_q_outer_precomputed = True
            print(
                _step2_text(
                    "mink",
                    "MINK_Q_BATCH_PARALLEL_READY "
                    f"poses={len(precompute_pose_indices)} "
                    f"ik_workers={batch_workers} "
                    f"mink_solver={getattr(args, 'mink_solver', None)} "
                    f"checker={'full_scene_mjwarp_pool' if _mink_q_checker is not None else 'mujoco_env'} "
                    f"checker_workers={getattr(_mink_q_checker, 'num_workers', None)} "
                    f"worlds_per_worker={locals().get('_mink_q_worlds_per_worker')}",
                ),
                file=sys.__stdout__,
                flush=True,
            )
            try:
                batch_entries = []
                for pose_index in precompute_pose_indices:
                    candidate_index, skeleton_pose = skeleton_poses[int(pose_index)]
                    retreat_normal = np.asarray(
                        getattr(
                            skeleton_pose,
                            "contact_normal_world",
                            panel.outward_world,
                        ),
                        dtype=np.float64,
                    ).reshape(3)
                    if float(np.dot(retreat_normal, panel.outward_world)) < 0.0:
                        retreat_normal = -retreat_normal
                    batch_entries.append(
                        (int(pose_index), skeleton_pose, retreat_normal.copy())
                    )
                batch_results = mink_q.solve_skeleton_precontact_q_batch(
                    env,
                    robot_model=robot_model,
                    arm_joint_names=arm_joint_names,
                    frame_name=frame_name,
                    pose_entries=batch_entries,
                    q_start=q_robot_start,
                    q_posture=q_posture,
                    posture_cost=posture_cost,
                    args=args,
                    scene_checker=_mink_q_checker,
                    penetration_checker=lambda q_arm: _strict_robot_drawer_penetration(
                        env, arm_joint_names, q_arm
                    ),
                    scene_collision_checker=lambda q_arm: _check_arm_q_collision(
                        env,
                        panel,
                        arm_joint_names,
                        q_arm,
                        allowed_ee_geom_name=None,
                        penetration_tolerance=_mink_q_scene_pen_tol,
                    ),
                    max_workers=batch_workers,
                )
                for pose_index in precompute_pose_indices:
                    _mink_q_debug_parallel_results[int(pose_index)] = batch_results[
                        int(pose_index)
                    ]
                    if _tqdm_pbar is not None:
                        _tqdm_pbar.update(1)
            except Exception as exc:
                _mink_q_outer_precomputed = False
                sys.stderr.write(
                    "[close_drawer] mink_q batch precompute failed "
                    f"({exc!r}); falling back to per-pose solver.\n"
                )
                sys.stderr.flush()

    step2_iter = (
        order[:max_attempts]
        if _mink_q_outer_precomputed
        else (_tqdm_pbar if _tqdm_pbar is not None else order[:max_attempts])
    )
    for attempt_id, pose_index in enumerate(step2_iter):
        candidate_index, skeleton_pose = skeleton_poses[int(pose_index)]
        candidate = candidates[int(candidate_index)]
        feasible_row = feasible_row_by_candidate.get(int(candidate_index))
        try:
            # FloatingEEMPPI (single-step, tracking + non-penetration cost) refines
            # the EE pose + gripper in a stripped scene; FloatingEERollout then
            # checks whether driving that pose toward the target reduces the drawer
            # object-cost. Skip candidate if either gate fails.
            _skel_pos = np.asarray(skeleton_pose.ee_position, dtype=np.float64)
            _skel_rot = np.asarray(skeleton_pose.ee_rotation, dtype=np.float64)
            _skel_g = float(
                getattr(
                    skeleton_pose,
                    "gripper_opening",
                    ee_skelton.PANDA_DEFAULT_GRIPPER_OPENING,
                )
            )
            _accept_improvement_only = bool(
                getattr(args, "autogen_qmppi_accept_object_improvement_only", True)
            )
            _obj_eps = float(
                getattr(args, "autogen_qmppi_object_improvement_eps", 1e-5)
            )
            _score_thresh = float(
                getattr(args, "autogen_qmppi_score_accept_threshold", 0.15)
            )
            if solve_step2 == "mppi" and floating_mppi is not None:
                attempted_floating_ee_count += 1
                _skel_quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(_skel_quat, _skel_rot.reshape(9))
                selected_point = (
                    np.asarray(
                        feasible_cache.positions_world[int(feasible_row)],
                        dtype=np.float64,
                    )
                    if feasible_row is not None
                    else np.asarray(candidate.world_point, dtype=np.float64)
                )
                force_direction = _candidate_force_direction_world(
                    candidate_index, candidate
                )
                floating_mppi.selected_contact_point_world = selected_point.reshape(3)
                floating_mppi.approach_world = force_direction.reshape(3)
                _result = floating_mppi.solve(_skel_pos, _skel_quat, _skel_g)
                refined_pos = np.asarray(_result.ee_position, dtype=np.float64)
                refined_rot = np.asarray(_result.ee_rotation, dtype=np.float64)
                refined_g = float(_result.gripper_opening)
                max_pen = float(_result.pen_cost)
                floating_pen_values.append(max_pen)
                floating_contact_values.append(float(_result.contact_distance))

                # --- Approach rollout: does pushing the refined pose actually drive the
                # drawer toward the target? Owned by FloatingEERollout (demos/rollout.py).
                _rollout = _get_rollout(force_direction)
                _obj_delta = 0.0
                _score_norm = float("nan")
                _score_rebound = float("nan")
                if _rollout is not None:
                    _rollout_res = _rollout.run(refined_pos, refined_rot, refined_g)
                    _obj_delta = float(_rollout_res.object_cost_delta)
                    _score_norm = float(_rollout_res.score_normalized)
                    _score_rebound = float(_rollout_res.score_rebound)
                    floating_obj_delta_values.append(_obj_delta)
                    best_floating_obj_delta = min(
                        float(best_floating_obj_delta), _obj_delta
                    )
                    obj_improved = bool(
                        np.isfinite(_score_norm) and _score_norm > _score_thresh
                    )
                else:
                    _obj_delta = float("nan")
                    obj_improved = False
                    floating_obj_delta_values.append(_obj_delta)

                if _accept_improvement_only:
                    final_accepted = obj_improved
                else:
                    final_accepted = bool(_result.accepted) and obj_improved

                if not final_accepted:
                    rejected_floating_ee_count += 1
                    pen_rejected = bool(
                        float(_result.pen_cost)
                        > float(floating_mppi.config.pen_threshold)
                    )
                    contact_rejected = bool(
                        float(_result.contact_distance)
                        > float(floating_mppi.config.contact_tolerance)
                    )
                    if obj_improved and not _result.accepted:
                        if pen_rejected:
                            _reason = "floating_mppi_penetration"
                        elif contact_rejected:
                            _reason = "floating_mppi_contact_distance"
                        else:
                            _reason = "floating_mppi_reject"
                    elif not obj_improved:
                        if _rollout is None:
                            _reason = "floating_rollout_unavailable"
                        elif not np.isfinite(_score_norm):
                            _reason = "floating_rollout_nan"
                        elif _score_rebound > 0.0:
                            _reason = "floating_rollout_rebound"
                        else:
                            _reason = "floating_rollout_low_score"
                    else:
                        _reason = "floating_reject"
                    floating_reject_reasons[_reason] = (
                        floating_reject_reasons.get(_reason, 0) + 1
                    )
                    _step2_pbar_post(
                        "MPPI_REJECT",
                        candidate_index,
                        pen=_result.pen_cost,
                        pos=None,
                        obj=_obj_delta,
                    )
                    if _step2_verbose_poses:
                        print(
                            _step2_text(
                                "mppi",
                                f"MPPI_REJECT candidate={candidate_index} "
                                f"reason={_reason} "
                                f"pen={_result.pen_cost:.4f} "
                                f"pen_threshold={floating_mppi.config.pen_threshold:.4f} "
                                f"contact_dist={_result.contact_distance:.4f} "
                                f"contact_tol={floating_mppi.config.contact_tolerance:.4f} "
                                f"track={_result.track_cost:.4f} "
                                f"obj_delta={_obj_delta:.4e} "
                                f"best_cost={_result.best_cost:.4f} "
                                f"iters={len(_result.iteration_history)}",
                            ),
                            flush=True,
                        )
                    reports.append(
                        MinkContactAttemptReport(
                            drawer_candidate_index=int(candidate_index),
                            drawer_contact_world=np.asarray(
                                candidate.world_point, dtype=np.float64
                            ),
                            drawer_contact_local=np.asarray(
                                candidate.local_point, dtype=np.float64
                            ),
                            drawer_contact_cost=float(candidate.cost),
                            contact_feasible=bool(candidate.feasible),
                            status="failed",
                            reason=_reason,
                            best_ee_sample_index=0,
                            best_ee_sample_name=f"skeleton_{getattr(skeleton_pose, 'contact_primitive', skeleton_pose.contact_finger)}",
                            best_position_error=float("inf"),
                            best_collision_free=False,
                        )
                    )
                    continue
                _step2_pbar_post(
                    "MPPI_ACCEPT",
                    candidate_index,
                    pen=_result.pen_cost,
                    pos=None,
                    obj=_obj_delta,
                )
                if _step2_verbose_poses:
                    print(
                        _step2_text(
                            "mppi",
                            f"MPPI_ACCEPT candidate={candidate_index} "
                            f"pen={_result.pen_cost:.4f} "
                            f"contact_dist={_result.contact_distance:.4f} "
                            f"track={_result.track_cost:.4f} "
                            f"obj_delta={_obj_delta:.4e} "
                            f"best_cost={_result.best_cost:.4f} "
                            f"iters={len(_result.iteration_history)}",
                        ),
                        flush=True,
                    )
                LAST_FLOATING_REFINED_POSES.append(
                    (
                        refined_pos.copy(),
                        refined_rot.copy(),
                        refined_g,
                        int(candidate_index),
                    )
                )
                successful_floating_ee_count += 1
            elif solve_step2 == "mppi":
                fallback_skeleton_ee_count += 1
                rejected_floating_ee_count += 1
                _reason = "floating_mppi_unavailable"
                floating_reject_reasons[_reason] = (
                    floating_reject_reasons.get(_reason, 0) + 1
                )
                _step2_pbar_post("MPPI_UNAVAILABLE_REJECT", candidate_index)
                if _step2_verbose_poses:
                    print(
                        _step2_text(
                            "mppi",
                            f"MPPI_REJECT candidate={candidate_index} "
                            f"reason={_reason} "
                            f"unavailable_reason={floating_mppi_unavailable_reason}",
                        ),
                        flush=True,
                    )
                reports.append(
                    MinkContactAttemptReport(
                        drawer_candidate_index=int(candidate_index),
                        drawer_contact_world=np.asarray(
                            candidate.world_point, dtype=np.float64
                        ),
                        drawer_contact_local=np.asarray(
                            candidate.local_point, dtype=np.float64
                        ),
                        drawer_contact_cost=float(candidate.cost),
                        contact_feasible=bool(candidate.feasible),
                        status="failed",
                        reason=_reason,
                        best_ee_sample_index=0,
                        best_ee_sample_name=f"skeleton_{getattr(skeleton_pose, 'contact_primitive', skeleton_pose.contact_finger)}",
                        best_position_error=float("inf"),
                        best_collision_free=False,
                    )
                )
                continue
            else:
                mink_q_attempted_count += 1
                force_direction = _candidate_force_direction_world(
                    candidate_index, candidate
                )
                _mink_q_result = _mink_q_debug_parallel_results.get(int(pose_index))
                if (
                    _mink_q_result is None
                    and int(pose_index) in _mink_q_parallel_exceptions
                ):
                    raise _mink_q_parallel_exceptions[int(pose_index)]
                if _mink_q_result is None:
                    _, _, _, _mink_q_result = _solve_mink_q_for_pose_index(
                        int(pose_index),
                        scene_checker=_mink_q_checker,
                        max_workers_override=getattr(
                            args, "autogen_mink_parallel_workers", None
                        ),
                        serial_env=False,
                    )
                mink_q_pen_values.append(float(_mink_q_result.max_penetration))
                mink_q_pos_values.append(float(_mink_q_result.position_error))
                if not bool(_mink_q_result.collision_free):
                    mink_q_rejected_count += 1
                    _reason = str(_mink_q_result.collision_reason)
                    mink_q_reject_reasons[_reason] = (
                        mink_q_reject_reasons.get(_reason, 0) + 1
                    )
                    _step2_pbar_post(
                        "MINK_Q_REJECT",
                        candidate_index,
                        pen=float(_mink_q_result.max_penetration),
                        pos=float(_mink_q_result.position_error),
                    )
                    if _step2_verbose_poses:
                        print(
                            _step2_text(
                                "mink",
                                "MINK_Q_REJECT "
                                f"candidate={candidate_index} "
                                f"reason={_reason} "
                                f"pos_err={float(_mink_q_result.position_error):.6f} "
                                f"rot_err={float(_mink_q_result.rotation_error):.6f} "
                                f"pen={float(_mink_q_result.max_penetration):.6f} "
                                f"retreat={float(_mink_q_result.retreat_distance):.6f} "
                                f"attempts={len(_mink_q_result.attempts)}",
                            ),
                            flush=True,
                        )
                    reports.append(
                        MinkContactAttemptReport(
                            drawer_candidate_index=int(candidate_index),
                            drawer_contact_world=np.asarray(
                                candidate.world_point, dtype=np.float64
                            ),
                            drawer_contact_local=np.asarray(
                                candidate.local_point, dtype=np.float64
                            ),
                            drawer_contact_cost=float(candidate.cost),
                            contact_feasible=bool(candidate.feasible),
                            status="failed",
                            reason=f"mink_q:{_reason}",
                            best_ee_sample_index=0,
                            best_ee_sample_name=f"skeleton_{getattr(skeleton_pose, 'contact_primitive', skeleton_pose.contact_finger)}",
                            best_position_error=float(_mink_q_result.position_error),
                            best_collision_free=False,
                        )
                    )
                    continue
                q_best = np.asarray(_mink_q_result.arm_q, dtype=np.float64).reshape(7)
                g_best = _skel_g
                pos_err = float(_mink_q_result.position_error)
                rot_err = float(_mink_q_result.rotation_error)
                max_pen = float(_mink_q_result.max_penetration)
                refined_pos = np.asarray(
                    _mink_q_result.actual_position_world, dtype=np.float64
                )
                refined_rot = np.asarray(
                    _mink_q_result.actual_rotation_world, dtype=np.float64
                )
                refined_g = _skel_g
                mink_q_accepted_count += 1
                pending_mink_q_solutions.append(
                    {
                        "candidate_index": int(candidate_index),
                        "candidate": candidate,
                        "skeleton_pose": skeleton_pose,
                        "q_best": q_best,
                        "g_best": g_best,
                        "pos_err": pos_err,
                        "rot_err": rot_err,
                        "max_pen": max_pen,
                        "refined_pos": refined_pos.copy(),
                        "refined_rot": refined_rot.copy(),
                        "refined_g": refined_g,
                        "force_direction": np.asarray(force_direction, dtype=np.float64)
                        .reshape(3)
                        .copy(),
                        "retreat_distance": float(_mink_q_result.retreat_distance),
                    }
                )
                _step2_pbar_post(
                    "MINK_Q_ACCEPT",
                    candidate_index,
                    pen=float(max_pen),
                    pos=float(pos_err),
                )
                if _step2_verbose_poses:
                    print(
                        _step2_text(
                            "mink",
                            "MINK_Q_ACCEPT "
                            f"candidate={candidate_index} "
                            f"pos_err={pos_err:.6f} "
                            f"rot_err={rot_err:.6f} "
                            f"pen={max_pen:.6f} "
                            f"retreat={float(_mink_q_result.retreat_distance):.6f} "
                            f"attempts={len(_mink_q_result.attempts)}",
                        ),
                        flush=True,
                    )
                continue
            if floating_mppi is not None and bool(
                getattr(args, "autogen_skip_mink_q_after_mppi", False)
            ):
                solution = _mppi_terminal_ee_solution(
                    panel,
                    candidate,
                    candidate_index,
                    frame_name,
                    skeleton_pose,
                    refined_pos,
                    refined_rot,
                    refined_g,
                    max_pen,
                    args,
                )
                ok = bool(solution.collision_free)
                reports.append(
                    MinkContactAttemptReport(
                        drawer_candidate_index=int(candidate_index),
                        drawer_contact_world=np.asarray(
                            candidate.world_point, dtype=np.float64
                        ),
                        drawer_contact_local=np.asarray(
                            candidate.local_point, dtype=np.float64
                        ),
                        drawer_contact_cost=float(candidate.cost),
                        contact_feasible=bool(candidate.feasible),
                        status="success" if ok else "failed",
                        reason="success" if ok else "mppi_penetration",
                        best_ee_sample_index=0,
                        best_ee_sample_name=str(solution.ee_sample_name),
                        best_position_error=0.0,
                        best_collision_free=ok,
                    )
                )
                if ok:
                    solutions.append(solution)
                continue
            if solve_step2 == "mppi":
                # Feed the refined pose into mink IK to obtain a feasible arm-q.
                q_robot_refined, pos_err_val = _solve_mink_frame_pose(
                    env,
                    frame_name,
                    refined_pos,
                    refined_rot,
                    q_robot_start,
                    q_posture,
                    posture_cost,
                    args,
                )
                q_best = _arm_q_from_robot_model_q(
                    robot_model, q_robot_refined, arm_joint_names
                )
                g_best = refined_g
                pos_err = float(pos_err_val)
                rot_err = 0.0
                actual_pen, actual_pen_pair = _strict_robot_drawer_penetration(
                    env, arm_joint_names, q_best
                )
                max_pen = max(float(max_pen), float(actual_pen))
                if actual_pen > pen_tol:
                    _step2_pbar_post(
                        "STRICT_PEN_REJECT",
                        candidate_index,
                        pen=float(actual_pen),
                        pos=float(pos_err),
                    )
                    if _step2_verbose_poses:
                        print(
                            _step2_text(
                                "mppi",
                                "SKELETON_STRICT_PENETRATION_REJECT "
                                f"candidate={candidate_index} "
                                f"primitive={getattr(skeleton_pose, 'contact_primitive', skeleton_pose.contact_finger)} "
                                f"penetration={actual_pen:.6f} pair={actual_pen_pair}",
                            ),
                            flush=True,
                        )
        except Exception as exc:
            _exc_key = f"{exc.__class__.__name__}:{str(exc)[:160]}"
            exception_label = (
                "skeleton_mppi_exception"
                if solve_step2 == "mppi"
                else "mink_q_exception"
            )
            if solve_step2 == "mppi":
                floating_exception_reasons[_exc_key] = (
                    floating_exception_reasons.get(_exc_key, 0) + 1
                )
                rejected_floating_ee_count += 1
                floating_reject_reasons[exception_label] = (
                    floating_reject_reasons.get(exception_label, 0) + 1
                )
            else:
                mink_q_rejected_count += 1
                mink_q_reject_reasons[_exc_key] = (
                    mink_q_reject_reasons.get(_exc_key, 0) + 1
                )
            _step2_pbar_post(
                f"{'MPPI' if solve_step2 == 'mppi' else 'MINK'}_EXCEPTION",
                candidate_index,
            )
            if _step2_verbose_poses:
                print(
                    _step2_text(
                        solve_step2,
                        (
                            "MPPI_EXCEPTION "
                            if solve_step2 == "mppi"
                            else "MINK_Q_EXCEPTION "
                        )
                        + f"candidate={candidate_index} "
                        f"type={exc.__class__.__name__} "
                        f"message={str(exc)[:240]}",
                    ),
                    flush=True,
                )
            reports.append(
                MinkContactAttemptReport(
                    drawer_candidate_index=int(candidate_index),
                    drawer_contact_world=np.asarray(
                        candidate.world_point, dtype=np.float64
                    ),
                    drawer_contact_local=np.asarray(
                        candidate.local_point, dtype=np.float64
                    ),
                    drawer_contact_cost=float(candidate.cost),
                    contact_feasible=bool(candidate.feasible),
                    status="failed",
                    reason=f"{exception_label}:{exc.__class__.__name__}",
                    best_ee_sample_index=0,
                    best_ee_sample_name=f"skeleton_{getattr(skeleton_pose, 'contact_primitive', skeleton_pose.contact_finger)}",
                    best_position_error=float("inf"),
                    best_collision_free=False,
                )
            )
            continue

        collision_free = bool(float(max_pen) <= pen_tol)
        solution = _skeleton_solution_to_contact_solution(
            env,
            panel,
            candidate,
            candidate_index,
            push_distance,
            robot_state,
            frame_name,
            skeleton_pose,
            q_best,
            g_best,
            pos_err,
            rot_err,
            max_pen,
            args,
            contact_anchor_pose=(_skel_pos, _skel_rot)
            if solve_step2 == "mink"
            else None,
        )
        ok = bool(float(pos_err) <= pos_tol and collision_free)
        reports.append(
            MinkContactAttemptReport(
                drawer_candidate_index=int(candidate_index),
                drawer_contact_world=np.asarray(
                    candidate.world_point, dtype=np.float64
                ),
                drawer_contact_local=np.asarray(
                    candidate.local_point, dtype=np.float64
                ),
                drawer_contact_cost=float(candidate.cost),
                contact_feasible=bool(candidate.feasible),
                status="success" if ok else "failed",
                reason=(
                    "success"
                    if ok
                    else (
                        "position_error"
                        if float(pos_err) > pos_tol
                        else (
                            "mppi_penetration"
                            if solve_step2 == "mppi"
                            else "mink_q_penetration"
                        )
                    )
                ),
                best_ee_sample_index=0,
                best_ee_sample_name=f"skeleton_{getattr(skeleton_pose, 'contact_primitive', skeleton_pose.contact_finger)}",
                best_position_error=float(pos_err),
                best_collision_free=collision_free,
            )
        )
        if ok:
            solutions.append(solution)

    q_elapsed_s = time.perf_counter() - started
    feasible_q = (
        mink_q_accepted_count
        if solve_step2 == "mink"
        else sum(1 for r in reports if r.status == "success")
    )
    if _tqdm_pbar is not None:
        if solve_step2 == "mink":
            _tqdm_pbar.set_postfix_str(
                f"done acc={mink_q_accepted_count} rej={mink_q_rejected_count} "
                f"feasible_q={feasible_q} t={q_elapsed_s:.3f}s"
            )
        else:
            _tqdm_pbar.set_postfix_str(
                f"done acc={successful_floating_ee_count} rej={rejected_floating_ee_count} "
                f"feasible_q={feasible_q} t={q_elapsed_s:.3f}s"
            )
        _tqdm_pbar.close()

    rollout_elapsed_s = 0.0
    if solve_step2 == "mink" and pending_mink_q_solutions:
        rollout_started = time.perf_counter()
        rollout_pbar = None
        rollout_iter = pending_mink_q_solutions
        rollout_postfix_last_s = 0.0
        if not _step2_verbose_poses:
            rollout_pbar = tqdm(
                pending_mink_q_solutions,
                desc="rollout[mink]",
                unit="q",
                file=sys.__stdout__,
                dynamic_ncols=True,
                miniters=1,
                mininterval=0.2,
                leave=True,
            )
            rollout_iter = rollout_pbar
        for entry in rollout_iter:
            candidate_index = int(entry["candidate_index"])
            candidate = entry["candidate"]
            skeleton_pose = entry["skeleton_pose"]
            refined_pos = np.asarray(entry["refined_pos"], dtype=np.float64).reshape(3)
            refined_rot = np.asarray(entry["refined_rot"], dtype=np.float64).reshape(
                3, 3
            )
            refined_g = float(entry["refined_g"])
            pos_err = float(entry["pos_err"])
            rot_err = float(entry["rot_err"])
            max_pen = float(entry["max_pen"])
            _rollout = _get_rollout(
                np.asarray(entry["force_direction"], dtype=np.float64)
            )
            _obj_delta = float("nan")
            _score_norm = float("nan")
            _score_rebound = float("nan")
            if _rollout is not None:
                _rollout_res = _rollout.run(refined_pos, refined_rot, refined_g)
                _obj_delta = float(_rollout_res.object_cost_delta)
                _score_norm = float(_rollout_res.score_normalized)
                _score_rebound = float(_rollout_res.score_rebound)
                mink_q_obj_delta_values.append(_obj_delta)
                best_mink_q_obj_delta = min(float(best_mink_q_obj_delta), _obj_delta)
                obj_improved = bool(
                    np.isfinite(_score_norm)
                    and _score_norm
                    > float(getattr(args, "autogen_qmppi_score_accept_threshold", 0.15))
                )
            else:
                mink_q_obj_delta_values.append(_obj_delta)
                obj_improved = False
            if not obj_improved:
                mink_rollout_rejected_count += 1
                if _rollout is None:
                    _reason = "mink_rollout_unavailable"
                elif not np.isfinite(_score_norm):
                    _reason = "mink_rollout_nan"
                elif _score_rebound > 0.0:
                    _reason = "mink_rollout_rebound"
                else:
                    _reason = "mink_rollout_low_score"
                mink_rollout_reject_reasons[_reason] = (
                    mink_rollout_reject_reasons.get(_reason, 0) + 1
                )
                if _step2_verbose_poses:
                    print(
                        _step2_text(
                            "mink",
                            "MINK_Q_ROLLOUT_REJECT "
                            f"candidate={candidate_index} "
                            f"reason={_reason} "
                            f"pos_err={pos_err:.6f} "
                            f"rot_err={rot_err:.6f} "
                            f"pen={max_pen:.6f} "
                            f"obj_delta={_obj_delta:.4e} "
                            f"score_norm={_score_norm:.4e} "
                            f"rebound={_score_rebound:.4e} "
                            f"retreat={float(entry['retreat_distance']):.6f}",
                        ),
                        flush=True,
                    )
                reports.append(
                    MinkContactAttemptReport(
                        drawer_candidate_index=int(candidate_index),
                        drawer_contact_world=np.asarray(
                            candidate.world_point, dtype=np.float64
                        ),
                        drawer_contact_local=np.asarray(
                            candidate.local_point, dtype=np.float64
                        ),
                        drawer_contact_cost=float(candidate.cost),
                        contact_feasible=bool(candidate.feasible),
                        status="failed",
                        reason=f"mink_rollout:{_reason}",
                        best_ee_sample_index=0,
                        best_ee_sample_name=f"skeleton_{getattr(skeleton_pose, 'contact_primitive', skeleton_pose.contact_finger)}",
                        best_position_error=float(pos_err),
                        best_collision_free=True,
                    )
                )
                if rollout_pbar is not None:
                    now_s = time.perf_counter()
                    if now_s - rollout_postfix_last_s >= _step2_postfix_interval_s:
                        rollout_postfix_last_s = now_s
                        rollout_pbar.set_postfix_str(
                            f"cand={candidate_index} reject={_reason} "
                            f"accepted={len(solutions)} "
                            f"rejected={mink_rollout_rejected_count} "
                            f"cache={len(_rollouts_by_direction)}"
                        )
                continue

            LAST_FLOATING_REFINED_POSES.append(
                (
                    refined_pos.copy(),
                    refined_rot.copy(),
                    refined_g,
                    int(candidate_index),
                )
            )
            collision_free = bool(float(max_pen) <= pen_tol)
            solution = _skeleton_solution_to_contact_solution(
                env,
                panel,
                candidate,
                candidate_index,
                push_distance,
                robot_state,
                frame_name,
                skeleton_pose,
                np.asarray(entry["q_best"], dtype=np.float64).reshape(7),
                float(entry["g_best"]),
                pos_err,
                rot_err,
                max_pen,
                args,
                contact_anchor_pose=(
                    np.asarray(skeleton_pose.ee_position, dtype=np.float64).reshape(3),
                    np.asarray(skeleton_pose.ee_rotation, dtype=np.float64).reshape(
                        3, 3
                    ),
                ),
            )
            ok = bool(float(pos_err) <= pos_tol and collision_free)
            reports.append(
                MinkContactAttemptReport(
                    drawer_candidate_index=int(candidate_index),
                    drawer_contact_world=np.asarray(
                        candidate.world_point, dtype=np.float64
                    ),
                    drawer_contact_local=np.asarray(
                        candidate.local_point, dtype=np.float64
                    ),
                    drawer_contact_cost=float(candidate.cost),
                    contact_feasible=bool(candidate.feasible),
                    status="success" if ok else "failed",
                    reason=(
                        "success"
                        if ok
                        else (
                            "position_error"
                            if float(pos_err) > pos_tol
                            else "mink_q_penetration"
                        )
                    ),
                    best_ee_sample_index=0,
                    best_ee_sample_name=f"skeleton_{getattr(skeleton_pose, 'contact_primitive', skeleton_pose.contact_finger)}",
                    best_position_error=float(pos_err),
                    best_collision_free=collision_free,
                )
            )
            if ok:
                solutions.append(solution)
            if rollout_pbar is not None:
                now_s = time.perf_counter()
                if now_s - rollout_postfix_last_s >= _step2_postfix_interval_s:
                    rollout_postfix_last_s = now_s
                    rollout_pbar.set_postfix_str(
                        f"cand={candidate_index} obj={_obj_delta:.2e} "
                        f"accepted={len(solutions)} "
                        f"rejected={mink_rollout_rejected_count} "
                        f"cache={len(_rollouts_by_direction)}"
                    )
        rollout_elapsed_s = time.perf_counter() - rollout_started
        if rollout_pbar is not None:
            rollout_pbar.set_postfix_str(
                f"done accepted={len(solutions)} "
                f"rejected={mink_rollout_rejected_count} "
                f"cache={len(_rollouts_by_direction)} "
                f"t={rollout_elapsed_s:.3f}s"
            )
            rollout_pbar.close()

    elapsed_s = time.perf_counter() - started
    final_success_count = sum(1 for r in reports if r.status == "success")
    if _step2_verbose_poses:
        floating_obj_delta_finite = [
            float(value) for value in floating_obj_delta_values if np.isfinite(value)
        ]
        mink_q_obj_delta_finite = [
            float(value) for value in mink_q_obj_delta_values if np.isfinite(value)
        ]
        if solve_step2 == "mppi":
            print(
                _step2_text(
                    "mppi",
                    f"SUCCESSFUL_FLOATING_EE_POSES={successful_floating_ee_count}",
                ),
                flush=True,
            )
            print(
                _step2_text(
                    "mppi",
                    "FLOATING_EE_MPPI_SUMMARY "
                    f"available={floating_mppi is not None} "
                    f"attempted={attempted_floating_ee_count} "
                    f"accepted={successful_floating_ee_count} "
                    f"rejected={rejected_floating_ee_count} "
                    f"reject_reasons={floating_reject_reasons} "
                    f"exception_reasons={floating_exception_reasons} "
                    f"pen_min={min(floating_pen_values) if floating_pen_values else float('nan'):.6f} "
                    f"pen_max={max(floating_pen_values) if floating_pen_values else float('nan'):.6f} "
                    f"contact_min={min(floating_contact_values) if floating_contact_values else float('nan'):.6f} "
                    f"contact_max={max(floating_contact_values) if floating_contact_values else float('nan'):.6f} "
                    f"obj_delta_min={min(floating_obj_delta_finite) if floating_obj_delta_finite else float('nan'):.4e} "
                    f"obj_delta_max={max(floating_obj_delta_finite) if floating_obj_delta_finite else float('nan'):.4e} "
                    f"fallback_skeleton={fallback_skeleton_ee_count} "
                    f"best_obj_delta={best_floating_obj_delta:.4e} "
                    f"unavailable_reason={floating_mppi_unavailable_reason}",
                ),
                flush=True,
            )
        else:
            print(
                _step2_text("mink", f"SUCCESSFUL_MINK_Q_POSES={mink_q_accepted_count}"),
                flush=True,
            )
            print(
                _step2_text(
                    "mink",
                    "MINK_Q_SUMMARY "
                    f"attempted={mink_q_attempted_count} "
                    f"accepted={mink_q_accepted_count} "
                    f"rejected={mink_q_rejected_count} "
                    f"reject_reasons={mink_q_reject_reasons} "
                    f"rollout_rejected={mink_rollout_rejected_count} "
                    f"rollout_reject_reasons={mink_rollout_reject_reasons} "
                    f"rollout_cache_count={len(_rollouts_by_direction)} "
                    f"rollout_elapsed_s={rollout_elapsed_s:.6f} "
                    f"final_success={final_success_count} "
                    f"pos_min={min(mink_q_pos_values) if mink_q_pos_values else float('nan'):.6f} "
                    f"pos_max={max(mink_q_pos_values) if mink_q_pos_values else float('nan'):.6f} "
                    f"pen_min={min(mink_q_pen_values) if mink_q_pen_values else float('nan'):.6f} "
                    f"pen_max={max(mink_q_pen_values) if mink_q_pen_values else float('nan'):.6f} "
                    f"obj_delta_min={min(mink_q_obj_delta_finite) if mink_q_obj_delta_finite else float('nan'):.4e} "
                    f"obj_delta_max={max(mink_q_obj_delta_finite) if mink_q_obj_delta_finite else float('nan'):.4e} "
                    f"best_obj_delta={best_mink_q_obj_delta:.4e}",
                ),
                flush=True,
            )
        print(
            _step2_text(
                solve_step2,
                f"mink_q_time={q_elapsed_s:.6f} "
                f"successful_pre_contact_q={feasible_q} "
                f"rollout_time={rollout_elapsed_s:.6f} "
                f"final_success={final_success_count}",
            ),
            flush=True,
        )
    else:
        # Non-verbose (autogen): a single yellow line with q duration + feasible q count.
        if solve_step2 == "mink":
            summary = (
                f"step2[mink] q_elapsed_s={q_elapsed_s:.3f} "
                f"feasible_q={feasible_q} "
                f"rollout_elapsed_s={rollout_elapsed_s:.3f} "
                f"rollout_accepted={final_success_count} "
                f"rollout_rejected={mink_rollout_rejected_count} "
                f"rollout_cache_count={len(_rollouts_by_direction)} "
                f"reject_reasons={mink_q_reject_reasons} "
                f"rollout_reject_reasons={mink_rollout_reject_reasons} "
                f"total_elapsed_s={elapsed_s:.3f}"
            )
        else:
            summary = f"step2[{solve_step2}] elapsed_s={elapsed_s:.3f} feasible_q={feasible_q}"
        print(_step2_text(solve_step2, summary), flush=True)

    _close_mink_q_checker()
    if not solutions:
        if args.require_mink_contact_pose:
            reasons = {}
            for report in reports:
                reasons[report.reason] = reasons.get(report.reason, 0) + 1
            raise RuntimeError(
                f"skeleton contact pose solve failed. Reasons: {reasons}"
            )
        return None, [], reports
    selected_solution = min(
        solutions,
        key=lambda s: (
            0 if s.collision_free else 1,
            s.contact_position_error,
            s.drawer_contact_cost,
        ),
    )
    return selected_solution, solutions, reports


def solve_contact_poses_with_mink(
    env, panel, candidates, push_distance, robot_state, args
):
    if not args.use_mink_contact_pose:
        return None, [], []
    if bool(getattr(args, "use_autogen_contact", True)):
        return _solve_contact_poses_with_skeleton(
            env,
            panel,
            candidates,
            push_distance,
            robot_state,
            args,
        )

    mink = _import_mink()
    del mink
    robot = env.robots[0]
    robot_model = robot.robot_model.mujoco_model
    frame_name = args.mink_contact_frame
    if not _mj_has_name(robot_model, "site", frame_name):
        raise RuntimeError(
            f"mink contact frame '{frame_name}' is not present in the robot model."
        )

    arm_joint_names = robot_state["robocasa_joint_names"]
    q_initial = _current_robot_model_q(env, robot_model)
    q_posture = q_initial.copy()
    posture_cost = _make_mink_posture_cost(robot_model, arm_joint_names, args)
    base_rot = _make_gripper_contact_rotation(panel.push_world)
    contact_offsets = _contact_offsets_from_gripper(env, frame_name)
    contact_offsets = contact_offsets[: max(1, int(args.mink_ee_sample_count))]
    roll_angles = np.linspace(0.0, 2.0 * np.pi, args.mink_roll_samples, endpoint=False)

    candidate_limit = min(len(candidates), max(1, int(args.mink_drawer_contact_count)))
    solutions = []
    reports = []
    for candidate_index, candidate in enumerate(candidates[:candidate_limit]):
        solution, report = _solve_mink_for_drawer_candidate(
            env,
            panel,
            candidate,
            candidate_index,
            push_distance,
            robot_model,
            arm_joint_names,
            frame_name,
            q_initial,
            q_posture,
            posture_cost,
            contact_offsets,
            roll_angles,
            base_rot,
            args,
        )
        reports.append(report)
        if solution is not None:
            solutions.append(solution)

    if not solutions:
        if args.require_mink_contact_pose:
            reasons = {}
            for report in reports:
                reasons[report.reason] = reasons.get(report.reason, 0) + 1
            raise RuntimeError(
                f"mink could not find any successful contact pose. Reasons: {reasons}"
            )
        return None, [], reports

    if args.require_mink_collision_free and any(
        not solution.collision_free for solution in solutions
    ):
        raise RuntimeError(
            "mink generated at least one contact pose that is not collision-free."
        )

    selected_solution = min(
        solutions,
        key=lambda s: (
            s.contact_position_error,
            s.drawer_contact_cost,
            np.linalg.norm(s.drawer_contact_local),
        ),
    )
    return selected_solution, solutions, reports


def solve_contact_pose_with_mink(
    env, panel, selected, push_distance, robot_state, args
):
    solution, _, _ = solve_contact_poses_with_mink(
        env, panel, [selected], push_distance, robot_state, args
    )
    return solution


def _make_mink_configuration(robot_model, q):
    mink = _import_mink()
    configuration = mink.Configuration(robot_model)
    configuration.update(np.asarray(q, dtype=np.float64).copy())
    return configuration


def _tensor_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _kinematic_model_ee_components(kinematic_state):
    """Return EE position/quaternion across supported cuRobo state APIs."""
    position = getattr(kinematic_state, "ee_position", None)
    quaternion = getattr(kinematic_state, "ee_quaternion", None)
    if position is None:
        position = getattr(kinematic_state, "ee_pos_seq", None)
    if quaternion is None:
        quaternion = getattr(kinematic_state, "ee_quat_seq", None)
    if position is None or quaternion is None:
        ee_pose = getattr(kinematic_state, "ee_pose", None)
        if ee_pose is not None:
            position = getattr(ee_pose, "position", position)
            quaternion = getattr(ee_pose, "quaternion", quaternion)
    if position is None or quaternion is None:
        raise AttributeError(
            "Unsupported cuRobo kinematic state. Expected ee_position/"
            "ee_quaternion or ee_pos_seq/ee_quat_seq; available fields: "
            f"{sorted(name for name in dir(kinematic_state) if not name.startswith('_'))}"
        )
    return position, quaternion


def _target_drawer_body_excludes(env, args):
    if env is None or not bool(
        getattr(args, "curobo_world_exclude_target_drawer", True)
    ):
        return ()
    drawer = getattr(env, "drawer", None)
    names = []
    root_body = getattr(drawer, "root_body", "") if drawer is not None else ""
    if root_body:
        names.append(str(root_body))
    drawer_name = getattr(drawer, "name", "") if drawer is not None else ""
    if drawer_name:
        names.append(str(drawer_name))
    return tuple(dict.fromkeys(names))


def plan_with_curobo(
    robot_state, target_hand_poses_base, args, env=None, q_waypoints=None
):
    _assert_cuda_context_usable("cuRobo planning")
    _ensure_curobo_importable()
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
    )

    robosuite_joint_names = tuple(
        robot_state.get("robocasa_joint_names", PANDA_JOINT_NAMES)
    )
    use_mujoco_world = (
        bool(getattr(args, "curobo_use_mujoco_world", True)) and env is not None
    )
    world_summary = {
        "world_collision_model": "none",
        "world_obstacle_count": 0,
        "world_signature": "",
        "world_exclude_bodies": (),
    }
    if use_mujoco_world:
        from robocasa.demos.curobo_planning import cached_motion_gen_for_env

        exclude_bodies = _target_drawer_body_excludes(env, args)
        cached = cached_motion_gen_for_env(
            env,
            args,
            robot_state=robot_state,
            extra_exclude_body_names=exclude_bodies,
        )
        tensor_args = cached["tensor_args"]
        motion_gen = cached["motion_gen"]
        curobo_joint_names = tuple(cached["curobo_joint_names"])
        curobo_base_pos_world = np.asarray(
            cached["curobo_base_pos_world"], dtype=np.float64
        ).reshape(3)
        curobo_base_rot_world = _matrix_from_quat_wxyz(
            np.asarray(cached["curobo_base_quat_wxyz_world"], dtype=np.float64)
        )
        world_summary = {
            "world_collision_model": "mujoco",
            "world_obstacle_count": int(len(cached["world_obstacle_names"])),
            "world_signature": str(cached["world_signature"]),
            "world_exclude_bodies": tuple(exclude_bodies),
        }
    else:
        tensor_args = TensorDeviceType()
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            args.curobo_robot_cfg,
            None,
            tensor_args,
            trajopt_tsteps=args.curobo_trajopt_tsteps,
            interpolation_dt=args.curobo_interpolation_dt,
            use_cuda_graph=not args.disable_curobo_cuda_graph,
            self_collision_check=not args.disable_curobo_self_collision,
            self_collision_opt=not args.disable_curobo_self_collision,
            num_ik_seeds=args.curobo_ik_seeds,
            num_graph_seeds=args.curobo_graph_seeds,
            num_trajopt_seeds=args.curobo_trajopt_seeds,
            collision_activation_distance=float(
                getattr(args, "curobo_collision_activation_distance", 0.005)
            ),
            collision_checker_type=None,
        )
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup(enable_graph=not args.disable_curobo_cuda_graph)
        curobo_joint_names = _extract_curobo_joint_names(motion_gen)
    q_current_robosuite = np.asarray(robot_state["q"], dtype=np.float64).reshape(1, 7)
    q_current_curobo = _reorder_q(
        q_current_robosuite,
        robosuite_joint_names,
        curobo_joint_names,
    ).reshape(1, 7)
    if q_waypoints is not None and bool(getattr(args, "curobo_mink_joint_space", True)):
        q_waypoints = np.asarray(q_waypoints, dtype=np.float64).reshape(-1, 7)
        if q_waypoints.shape[0] > 0:
            names = [str(pose[0]) for pose in target_hand_poses_base]
            if len(names) != q_waypoints.shape[0]:
                names = [f"mink_q_{idx}" for idx in range(q_waypoints.shape[0])]
            q_segments = []
            segment_summaries = []
            for name, q_goal_robosuite in zip(names, q_waypoints):
                q_goal_curobo = _reorder_q(
                    np.asarray(q_goal_robosuite, dtype=np.float64).reshape(1, 7),
                    robosuite_joint_names,
                    curobo_joint_names,
                ).reshape(1, 7)
                start_state = JointState.from_position(
                    tensor_args.to_device(q_current_curobo)
                )
                goal_state = JointState.from_position(
                    tensor_args.to_device(q_goal_curobo)
                )
                result = motion_gen.plan_single_js(
                    start_state,
                    goal_state,
                    plan_config=MotionGenPlanConfig(
                        max_attempts=args.curobo_max_attempts,
                        enable_graph_attempt=args.curobo_enable_graph_attempt,
                    ),
                )
                success = bool(_tensor_to_numpy(result.success).reshape(-1)[0])
                if not success:
                    raise RuntimeError(
                        f"cuRobo failed to plan joint-space segment '{name}': {result.status}. "
                        f"world_collision_model={world_summary['world_collision_model']}; "
                        f"world_obstacle_count={world_summary['world_obstacle_count']}; "
                        f"world_exclude_bodies={world_summary['world_exclude_bodies']}. "
                        "The mink q goal is known, so this failure is a joint-space "
                        "trajectory/collision failure rather than an EE IK failure."
                    )
                interpolated = result.get_interpolated_plan()
                if interpolated is None:
                    raw_plan = getattr(result, "raw_plan", None)
                    q_plan = _tensor_to_numpy(raw_plan)
                else:
                    q_plan = _tensor_to_numpy(interpolated.position)
                q_plan = np.asarray(q_plan, dtype=np.float64).reshape(-1, 7)
                q_plan_full = q_plan
                if q_segments and q_plan.shape[0] > 1:
                    q_plan = q_plan[1:]
                q_plan_robosuite = _reorder_q(
                    q_plan,
                    curobo_joint_names,
                    robosuite_joint_names,
                ).reshape(-1, 7)
                q_segments.append(q_plan_robosuite)
                q_current_curobo = q_plan_full[-1:].copy()
                segment_summaries.append(
                    {
                        "name": name,
                        "steps": int(q_plan.shape[0]),
                        "planner": "curobo_joint_space",
                        "curobo_joint_names": tuple(curobo_joint_names),
                        "robosuite_joint_names": tuple(robosuite_joint_names),
                        "goal_q_robosuite": np.asarray(
                            q_goal_robosuite, dtype=np.float64
                        )
                        .reshape(7)
                        .tolist(),
                        "goal_q_curobo": q_goal_curobo.reshape(7).tolist(),
                        **world_summary,
                        "status": str(result.status),
                    }
                )
            return np.concatenate(q_segments, axis=0), segment_summaries

    current_state = JointState.from_position(tensor_args.to_device(q_current_curobo))
    current_kin = motion_gen.compute_kinematics(current_state)
    current_ee_position, current_ee_quaternion = _kinematic_model_ee_components(
        current_kin
    )
    curobo_hand_pos = _tensor_to_numpy(current_ee_position).reshape(-1, 3)[0]
    curobo_hand_quat = _tensor_to_numpy(current_ee_quaternion).reshape(-1, 4)[0]
    curobo_hand_rot = _matrix_from_quat_wxyz(curobo_hand_quat)
    if "hand_pos_base" in robot_state and "hand_rot_base" in robot_state:
        robosuite_hand_pos = np.asarray(
            robot_state["hand_pos_base"], dtype=np.float64
        ).reshape(3)
        robosuite_hand_rot = np.asarray(
            robot_state["hand_rot_base"], dtype=np.float64
        ).reshape(3, 3)
        curobo_base_pos_in_robosuite, curobo_base_rot_in_robosuite = _pose_mul(
            robosuite_hand_pos,
            robosuite_hand_rot,
            *_pose_inv(curobo_hand_pos, curobo_hand_rot),
        )
    else:
        curobo_base_pos_in_robosuite = np.zeros(3, dtype=np.float64)
        curobo_base_rot_in_robosuite = np.eye(3, dtype=np.float64)

    q_segments = []
    segment_summaries = []
    for name, hand_pos_base, hand_rot_base in target_hand_poses_base:
        if use_mujoco_world:
            hand_pos_world, hand_rot_world = _pose_mul(
                np.asarray(robot_state["base_pos"], dtype=np.float64).reshape(3),
                np.asarray(robot_state["base_rot"], dtype=np.float64).reshape(3, 3),
                np.asarray(hand_pos_base, dtype=np.float64).reshape(3),
                np.asarray(hand_rot_base, dtype=np.float64).reshape(3, 3),
            )
            hand_pos_curobo_base, hand_rot_curobo_base = _pose_in_frame(
                hand_pos_world,
                hand_rot_world,
                curobo_base_pos_world,
                curobo_base_rot_world,
            )
        else:
            hand_pos_curobo_base, hand_rot_curobo_base = _pose_in_frame(
                np.asarray(hand_pos_base, dtype=np.float64),
                np.asarray(hand_rot_base, dtype=np.float64).reshape(3, 3),
                curobo_base_pos_in_robosuite,
                curobo_base_rot_in_robosuite,
            )
        start_state = JointState.from_position(tensor_args.to_device(q_current_curobo))
        goal_pose = Pose(
            tensor_args.to_device(hand_pos_curobo_base),
            quaternion=tensor_args.to_device(
                _quat_wxyz_from_matrix(hand_rot_curobo_base)
            ),
        )
        result = motion_gen.plan_single(
            start_state,
            goal_pose,
            plan_config=MotionGenPlanConfig(
                max_attempts=args.curobo_max_attempts,
                enable_graph_attempt=args.curobo_enable_graph_attempt,
            ),
        )
        success = bool(_tensor_to_numpy(result.success).reshape(-1)[0])
        if not success:
            ik_detail = ""
            try:
                if hasattr(motion_gen, "inverse_kinematics"):
                    ik_result = motion_gen.inverse_kinematics(
                        goal_pose,
                        num_seeds=args.curobo_ik_seeds,
                    )
                    ik_success = bool(
                        np.any(_tensor_to_numpy(ik_result.success).reshape(-1))
                    )
                    position_error = float(
                        np.min(_tensor_to_numpy(ik_result.position_error).reshape(-1))
                    )
                    rotation_error = float(
                        np.min(_tensor_to_numpy(ik_result.rotation_error).reshape(-1))
                    )
                    ik_detail = (
                        f", standalone_ik_success={ik_success}, "
                        f"ik_position_error={position_error:.6f}, "
                        f"ik_rotation_error={rotation_error:.6f}"
                    )
                else:
                    ik_detail = ", standalone_ik_diagnostic_unavailable=no_motion_gen_inverse_kinematics"
            except Exception as ik_exc:
                ik_detail = f", standalone_ik_diagnostic_failed={ik_exc!r}"
            raise RuntimeError(
                f"cuRobo failed to plan segment '{name}': {result.status}"
                f"{ik_detail}. world_collision_model={world_summary['world_collision_model']}; "
                f"world_obstacle_count={world_summary['world_obstacle_count']}; "
                f"world_exclude_bodies={world_summary['world_exclude_bodies']}. "
                "Use --disable-curobo-self-collision only as a diagnostic "
                "if the standalone IK fails with self-collision enabled."
            )
        interpolated = result.get_interpolated_plan()
        if interpolated is None:
            raw_plan = getattr(result, "raw_plan", None)
            q_plan = _tensor_to_numpy(raw_plan)
        else:
            q_plan = _tensor_to_numpy(interpolated.position)
        q_plan = np.asarray(q_plan, dtype=np.float64).reshape(-1, 7)
        if q_segments:
            q_plan = q_plan[1:]
        q_plan_robosuite = _reorder_q(
            q_plan,
            curobo_joint_names,
            robosuite_joint_names,
        ).reshape(-1, 7)
        q_segments.append(q_plan_robosuite)
        q_current_curobo = q_plan[-1:].copy()
        segment_summaries.append(
            {
                "name": name,
                "steps": int(q_plan.shape[0]),
                "curobo_joint_names": tuple(curobo_joint_names),
                "robosuite_joint_names": tuple(robosuite_joint_names),
                "goal_pos_base": np.asarray(
                    hand_pos_curobo_base, dtype=np.float64
                ).tolist(),
                "goal_quat_wxyz_base": _quat_wxyz_from_matrix(
                    hand_rot_curobo_base
                ).tolist(),
                "requested_robosuite_goal_pos_base": np.asarray(
                    hand_pos_base, dtype=np.float64
                ).tolist(),
                "requested_robosuite_goal_quat_wxyz_base": _quat_wxyz_from_matrix(
                    hand_rot_base
                ).tolist(),
                "curobo_base_pos_in_robosuite": curobo_base_pos_in_robosuite.tolist(),
                "curobo_base_quat_wxyz_in_robosuite": _quat_wxyz_from_matrix(
                    curobo_base_rot_in_robosuite
                ).tolist(),
                **world_summary,
                "status": str(result.status),
            }
        )

    return np.concatenate(q_segments, axis=0), segment_summaries


def save_outputs(
    path,
    env,
    panel,
    candidates,
    selected,
    push_distance,
    target_gripper_poses,
    target_hand_poses,
    q_traj,
    segments,
    mink_solution=None,
    all_mink_solutions=None,
    mink_reports=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_mink_solutions = all_mink_solutions or []
    mink_reports = mink_reports or []
    metadata = {
        "drawer_name": env.drawer.name,
        "drawer_geom": panel.geom_name,
        "drawer_size": np.asarray(env.drawer.size, dtype=np.float64).tolist(),
        "drawer_state": {
            k: float(v) for k, v in env.drawer.get_door_state(env).items()
        },
        "push_distance": float(push_distance),
        "selected_contact": {
            "world_point": selected.world_point.tolist(),
            "local_point": selected.local_point.tolist(),
            "cost": float(selected.cost),
            "lam": selected.lam.tolist(),
            "solver_status": selected.solver_status,
            "feasible": bool(selected.feasible),
        },
        "segments": segments,
        "candidate_count": len(candidates),
        "feasible_candidate_count": int(sum(c.feasible for c in candidates)),
        "mink_success_count": len(all_mink_solutions),
        "mink_attempt_count": len(mink_reports),
    }
    if mink_solution is not None:
        metadata["mink_contact_pose"] = {
            "contact_frame": mink_solution.contact_frame,
            "contact_offset_local": mink_solution.contact_offset_local.tolist(),
            "roll_angle": float(mink_solution.roll_angle),
            "contact_position_error": float(mink_solution.contact_position_error),
            "collision_free": bool(mink_solution.collision_free),
        }
    if all_mink_solutions:
        metadata["mink_contact_pose_summary"] = [
            {
                "drawer_candidate_index": int(solution.drawer_candidate_index),
                "drawer_contact_world": solution.drawer_contact_world.tolist(),
                "drawer_contact_local": solution.drawer_contact_local.tolist(),
                "drawer_contact_cost": float(solution.drawer_contact_cost),
                "ee_sample_index": int(solution.ee_sample_index),
                "ee_sample_name": solution.ee_sample_name,
                "ee_contact_geom_name": solution.ee_contact_geom_name,
                "contact_frame": solution.contact_frame,
                "contact_offset_local": solution.contact_offset_local.tolist(),
                "roll_angle": float(solution.roll_angle),
                "contact_position_error": float(solution.contact_position_error),
                "collision_free": bool(solution.collision_free),
            }
            for solution in all_mink_solutions
        ]
    if mink_reports:
        metadata["mink_attempt_reports"] = [
            {
                "drawer_candidate_index": int(report.drawer_candidate_index),
                "drawer_contact_world": report.drawer_contact_world.tolist(),
                "drawer_contact_local": report.drawer_contact_local.tolist(),
                "drawer_contact_cost": float(report.drawer_contact_cost),
                "contact_feasible": bool(report.contact_feasible),
                "status": report.status,
                "reason": report.reason,
                "best_ee_sample_index": int(report.best_ee_sample_index),
                "best_ee_sample_name": report.best_ee_sample_name,
                "best_position_error": float(report.best_position_error),
                "best_collision_free": bool(report.best_collision_free),
            }
            for report in mink_reports
        ]
        reason_counts = {}
        for report in mink_reports:
            key = report.reason if report.status != "success" else "success"
            reason_counts[key] = reason_counts.get(key, 0) + 1
        metadata["mink_attempt_reason_counts"] = reason_counts

    if not bool(getattr(save_outputs, "_enabled", True)):
        return metadata

    if all_mink_solutions:
        all_q_waypoints = np.asarray(
            [solution.q_waypoints for solution in all_mink_solutions], dtype=np.float64
        )
        all_drawer_indices = np.asarray(
            [solution.drawer_candidate_index for solution in all_mink_solutions],
            dtype=np.int64,
        )
        all_ee_indices = np.asarray(
            [solution.ee_sample_index for solution in all_mink_solutions],
            dtype=np.int64,
        )
        all_ee_geom_names = np.asarray(
            [solution.ee_contact_geom_name for solution in all_mink_solutions]
        )
        all_contact_world = np.asarray(
            [solution.drawer_contact_world for solution in all_mink_solutions],
            dtype=np.float64,
        )
        all_contact_local = np.asarray(
            [solution.drawer_contact_local for solution in all_mink_solutions],
            dtype=np.float64,
        )
        all_ee_offsets = np.asarray(
            [solution.contact_offset_local for solution in all_mink_solutions],
            dtype=np.float64,
        )
        all_errors = np.asarray(
            [solution.contact_position_error for solution in all_mink_solutions],
            dtype=np.float64,
        )
        all_collision_free = np.asarray(
            [solution.collision_free for solution in all_mink_solutions], dtype=bool
        )
        all_ee_names = np.asarray(
            [solution.ee_sample_name for solution in all_mink_solutions]
        )
    else:
        all_q_waypoints = np.zeros((0, 3, 7), dtype=np.float64)
        all_drawer_indices = np.zeros((0,), dtype=np.int64)
        all_ee_indices = np.zeros((0,), dtype=np.int64)
        all_ee_geom_names = np.asarray([], dtype=str)
        all_contact_world = np.zeros((0, 3), dtype=np.float64)
        all_contact_local = np.zeros((0, 3), dtype=np.float64)
        all_ee_offsets = np.zeros((0, 3), dtype=np.float64)
        all_errors = np.zeros((0,), dtype=np.float64)
        all_collision_free = np.zeros((0,), dtype=bool)
        all_ee_names = np.asarray([], dtype=str)

    report_status = np.asarray([report.status for report in mink_reports])
    report_reasons = np.asarray([report.reason for report in mink_reports])
    report_best_errors = np.asarray(
        [report.best_position_error for report in mink_reports], dtype=np.float64
    )
    report_best_collision_free = np.asarray(
        [report.best_collision_free for report in mink_reports], dtype=bool
    )
    np.savez(
        path,
        q_traj=np.asarray(q_traj, dtype=np.float64)
        if q_traj is not None
        else np.zeros((0, 7)),
        mink_q_waypoints=(
            np.asarray(mink_solution.q_waypoints, dtype=np.float64)
            if mink_solution is not None
            else np.zeros((0, 7), dtype=np.float64)
        ),
        selected_contact_world=selected.world_point,
        selected_contact_local=selected.local_point,
        target_gripper_pos_world=np.asarray(
            [pose[1] for pose in target_gripper_poses], dtype=np.float64
        ),
        target_gripper_quat_wxyz_world=np.asarray(
            [_quat_wxyz_from_matrix(pose[2]) for pose in target_gripper_poses],
            dtype=np.float64,
        ),
        target_hand_pos_base=np.asarray(
            [pose[1] for pose in target_hand_poses], dtype=np.float64
        ),
        target_hand_quat_wxyz_base=np.asarray(
            [_quat_wxyz_from_matrix(pose[2]) for pose in target_hand_poses],
            dtype=np.float64,
        ),
        candidate_world_points=np.asarray(
            [c.world_point for c in candidates], dtype=np.float64
        ),
        candidate_local_points=np.asarray(
            [c.local_point for c in candidates], dtype=np.float64
        ),
        candidate_costs=np.asarray([c.cost for c in candidates], dtype=np.float64),
        candidate_feasible=np.asarray([c.feasible for c in candidates], dtype=bool),
        mink_all_q_waypoints=all_q_waypoints,
        mink_all_drawer_candidate_indices=all_drawer_indices,
        mink_all_ee_sample_indices=all_ee_indices,
        mink_all_ee_sample_names=all_ee_names,
        mink_all_ee_contact_geom_names=all_ee_geom_names,
        mink_all_contact_world=all_contact_world,
        mink_all_contact_local=all_contact_local,
        mink_all_ee_offsets=all_ee_offsets,
        mink_all_position_errors=all_errors,
        mink_all_collision_free=all_collision_free,
        mink_report_status=report_status,
        mink_report_reasons=report_reasons,
        mink_report_best_position_errors=report_best_errors,
        mink_report_best_collision_free=report_best_collision_free,
        metadata_json=json.dumps(metadata, indent=2),
    )
    return metadata


def _draw_viewer_sphere(viewer, pos, radius, rgba):
    import mujoco

    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _interpolate_arm_waypoints(waypoints, steps_per_segment):
    waypoints = np.asarray(waypoints, dtype=np.float64).reshape(-1, 7)
    if waypoints.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64)
    if waypoints.shape[0] == 1:
        return waypoints.copy()
    steps_per_segment = max(int(steps_per_segment), 2)
    segments = []
    for start, goal in zip(waypoints[:-1], waypoints[1:]):
        for alpha in np.linspace(0.0, 1.0, steps_per_segment, endpoint=False):
            segments.append((1.0 - alpha) * start + alpha * goal)
    segments.append(waypoints[-1])
    return np.asarray(segments, dtype=np.float64)


def _drawer_joint_value(env):
    joint_name = env.drawer.door_joint_names[0]
    return float(env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(joint_name)])


def _set_drawer_joint_value(env, value):
    joint_name = env.drawer.door_joint_names[0]
    env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(joint_name)] = float(value)


def _build_mink_visualization_segment(env, robot_state, solution, push_distance, args):
    drawer_q0 = _drawer_joint_value(env)
    waypoints = np.vstack([robot_state["q"].reshape(1, 7), solution.q_waypoints])
    arm_segments = []
    drawer_segments = []
    solution_indices = []
    steps_per_segment = max(int(args.visualize_steps_per_segment), 2)
    drawer_q_closed = min(drawer_q0 + float(push_distance), 0.0)
    drawer_waypoints = np.array(
        [drawer_q0, drawer_q0, drawer_q0, drawer_q_closed], dtype=np.float64
    )
    for segment_idx, (start, goal) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        for alpha in np.linspace(0.0, 1.0, steps_per_segment, endpoint=False):
            arm_segments.append((1.0 - alpha) * start + alpha * goal)
            drawer_segments.append(
                (1.0 - alpha) * drawer_waypoints[segment_idx]
                + alpha * drawer_waypoints[segment_idx + 1]
            )
            solution_indices.append(solution.drawer_candidate_index)
    for _ in range(max(int(args.visualize_pause_frames), 0)):
        arm_segments.append(waypoints[-1])
        drawer_segments.append(drawer_waypoints[-1])
        solution_indices.append(solution.drawer_candidate_index)
    return (
        np.asarray(arm_segments, dtype=np.float64),
        np.asarray(drawer_segments, dtype=np.float64),
        np.asarray(solution_indices, dtype=np.int64),
    )


def _visualization_trajectory(
    env, robot_state, q_traj, mink_solution, all_mink_solutions, push_distance, args
):
    drawer_q0 = _drawer_joint_value(env)
    if q_traj is not None and np.asarray(q_traj).size:
        arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
        drawer_traj = np.full(arm_traj.shape[0], drawer_q0, dtype=np.float64)
        return (
            arm_traj,
            drawer_traj,
            np.full(arm_traj.shape[0], -1, dtype=np.int64),
            "cuRobo",
        )
    if args.visualize_trajectory:
        solutions = list(all_mink_solutions or [])
        if not args.visualize_all_contact_poses and mink_solution is not None:
            solutions = [mink_solution]
        if solutions:
            arm_parts = []
            drawer_parts = []
            solution_index_parts = []
            for solution in solutions:
                (
                    arm_part,
                    drawer_part,
                    solution_indices,
                ) = _build_mink_visualization_segment(
                    env,
                    robot_state,
                    solution,
                    push_distance,
                    args,
                )
                arm_parts.append(arm_part)
                drawer_parts.append(drawer_part)
                solution_index_parts.append(solution_indices)
            return (
                np.concatenate(arm_parts, axis=0),
                np.concatenate(drawer_parts, axis=0),
                np.concatenate(solution_index_parts, axis=0),
                f"mink {len(solutions)} contact poses",
            )
    return (
        np.zeros((0, 7), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.int64),
        "static",
    )


def _ee_sample_world(env, frame_name, contact_offset_local):
    model = env.sim.model
    data = env.sim.data
    site_id = model.site_name2id(frame_name)
    site_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    site_rot = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    return site_pos + site_rot @ np.asarray(
        contact_offset_local, dtype=np.float64
    ).reshape(3)


def _write_mp4(path, frames, fps, args=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError(f"No frames to write for video: {path}")

    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    video_encoder = (
        getattr(args, "video_encoder", "ffmpeg") if args is not None else "ffmpeg"
    )
    ffmpeg = getattr(args, "ffmpeg_path", "") if args is not None else ""
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if ffmpeg is not None and video_encoder == "ffmpeg":
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            str(float(fps)),
            "-i",
            "-",
            "-an",
            "-c:v",
            getattr(args, "video_codec", "libx264"),
            "-preset",
            getattr(args, "video_preset", "slow"),
            "-crf",
            str(int(getattr(args, "video_crf", 18))),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            assert process.stdin is not None
            for frame in frames:
                frame = np.asarray(frame, dtype=np.uint8)
                if frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        f"Frame size mismatch while writing {path}: {frame.shape[:2]} != {(height, width)}"
                    )
                process.stdin.write(np.ascontiguousarray(frame[:, :, :3]).tobytes())
            process.stdin.close()
            stderr = (
                process.stderr.read().decode("utf-8", errors="replace")
                if process.stderr
                else ""
            )
            stdout = (
                process.stdout.read().decode("utf-8", errors="replace")
                if process.stdout
                else ""
            )
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed writing {path} with code {returncode}:\n{stderr or stdout}"
            )
        return

    import cv2

    print("[video] ffmpeg not used; falling back to OpenCV mp4v writer", flush=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open mp4 writer: {path}")
    try:
        for frame in frames:
            writer.write(
                cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            )
    finally:
        writer.release()


def _load_origin_demo_episode(args):
    dataset = get_ds_path_any_split("CloseDrawer", source="human")
    if dataset is None:
        raise RuntimeError("No registered human dataset found for CloseDrawer.")
    dataset = Path(dataset)
    if not dataset.exists():
        raise FileNotFoundError(
            f"CloseDrawer dataset is not available locally: {dataset}. "
            "Run python -m robocasa.demos.demo_tasks --task CloseDrawer once to download/register it."
        )

    episodes = LU.get_episodes(dataset)
    if not episodes:
        raise RuntimeError(f"No episodes found in CloseDrawer dataset: {dataset}")
    demo_index = int(args.origin_demo_index)
    if demo_index < 0 or demo_index >= len(episodes):
        raise IndexError(
            f"--origin-demo-index {demo_index} is out of range [0, {len(episodes) - 1}]."
        )

    states = LU.get_episode_states(dataset, demo_index)
    if args.origin_demo_extend_states > 0:
        states = np.concatenate(
            (states, [states[-1]] * int(args.origin_demo_extend_states))
        )
    initial_state = {
        "states": states[0],
        "model": LU.get_episode_model_xml(dataset, demo_index),
        "ep_meta": json.dumps(LU.get_episode_meta(dataset, demo_index)),
    }
    return dataset, episodes[demo_index].stem, initial_state, states


def _create_origin_demo_env(dataset, has_offscreen_renderer=True):
    env_meta = LU.get_env_metadata(Path(dataset))
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = bool(has_offscreen_renderer)
    env_kwargs["use_camera_obs"] = False
    env_kwargs["renderer"] = "mjviewer"
    env = robosuite.make(**env_kwargs)
    env.reset()  # robocasa defers .drawer init until first reset
    return env


def _joint_qpos_slice(model, joint_name):
    joint_id = _mj_id(model, "joint", joint_name)
    start = int(model.jnt_qposadr[joint_id])
    end = (
        int(model.jnt_qposadr[joint_id + 1])
        if joint_id + 1 < model.njnt
        else int(model.nq)
    )
    return slice(start, end)


def _copy_joint_by_name(src_env, dst_env, src_joint_name, dst_joint_name=None):
    dst_joint_name = src_joint_name if dst_joint_name is None else dst_joint_name
    src_model = src_env.sim.model
    dst_model = dst_env.sim.model
    if (
        src_joint_name not in src_model._joint_name2id
        or dst_joint_name not in dst_model._joint_name2id
    ):
        return
    src_slice = _joint_qpos_slice(src_model, src_joint_name)
    dst_slice = _joint_qpos_slice(dst_model, dst_joint_name)
    width = min(src_slice.stop - src_slice.start, dst_slice.stop - dst_slice.start)
    dst_env.sim.data.qpos[
        dst_slice.start : dst_slice.start + width
    ] = src_env.sim.data.qpos[src_slice.start : src_slice.start + width]


def _apply_origin_demo_joints_to_render_env(source_env, render_env):
    source_robot = source_env.robots[0]
    render_robot = render_env.robots[0]
    source_arm_joints = tuple(source_robot.robot_model.joints[:7])
    render_arm_joints = tuple(render_robot.robot_model.joints[:7])
    for source_joint, render_joint in zip(source_arm_joints, render_arm_joints):
        _copy_joint_by_name(source_env, render_env, source_joint, render_joint)

    for joint_name in source_env.sim.model._joint_name2id:
        if joint_name.startswith("gripper0_"):
            _copy_joint_by_name(source_env, render_env, joint_name)

    if source_env.drawer.door_joint_names and render_env.drawer.door_joint_names:
        _copy_joint_by_name(
            source_env,
            render_env,
            source_env.drawer.door_joint_names[0],
            render_env.drawer.door_joint_names[0],
        )
    render_env.sim.forward()


def _find_drawer_contact_local_from_demo(env, states):
    for frame_index, state in enumerate(states):
        reset_to(env, {"states": state})
        panel = get_panel_frame(env)
        robot_geoms, _, drawer_geoms = _robot_contact_geom_sets(env, panel)
        for contact_idx in range(env.sim.data.ncon):
            contact = env.sim.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in robot_geoms and geom2 in drawer_geoms:
                contact_pos = np.asarray(contact.pos, dtype=np.float64).copy()
            elif geom2 in robot_geoms and geom1 in drawer_geoms:
                contact_pos = np.asarray(contact.pos, dtype=np.float64).copy()
            else:
                continue
            local = panel.rotation_world.T @ (contact_pos - panel.center_world)
            local[0] = np.clip(local[0], -panel.half_size[0], panel.half_size[0])
            local[1] = -float(panel.half_size[1])
            local[2] = np.clip(local[2], -panel.half_size[2], panel.half_size[2])
            return local, frame_index

    reset_to(env, {"states": states[0]})
    panel = get_panel_frame(env)
    return np.array([0.0, -float(panel.half_size[1]), 0.0], dtype=np.float64), -1


def save_origin_demo_video(args):
    dataset, episode_name, initial_state, states = _load_origin_demo_episode(args)
    source_env = _create_origin_demo_env(dataset, has_offscreen_renderer=False)
    render_env = create_close_drawer_env(args)
    render_qpos = render_env.sim.data.qpos.copy()
    render_qvel = render_env.sim.data.qvel.copy()
    try:
        reset_to(source_env, initial_state)
        contact_local, contact_frame = _find_drawer_contact_local_from_demo(
            source_env, states
        )

        output_path = Path(args.origin_demo_output)
        if str(output_path) == "":
            output_path = (
                Path(args.video_output_dir) / f"{args.video_prefix}_origin_demo.mp4"
            )

        frames = []
        video_skip = max(int(args.origin_demo_video_skip), 1)
        for frame_index, state in enumerate(states):
            if frame_index % video_skip != 0 and frame_index != len(states) - 1:
                continue
            reset_to(source_env, {"states": state})
            render_env.sim.data.qpos[:] = render_qpos
            render_env.sim.data.qvel[:] = render_qvel
            _apply_origin_demo_joints_to_render_env(source_env, render_env)
            panel = get_panel_frame(render_env)
            contact_local_render = contact_local.copy()
            contact_local_render[0] = np.clip(
                contact_local_render[0], -panel.half_size[0], panel.half_size[0]
            )
            contact_local_render[1] = -float(panel.half_size[1])
            contact_local_render[2] = np.clip(
                contact_local_render[2], -panel.half_size[2], panel.half_size[2]
            )
            marker_pos = (
                panel.center_world + panel.rotation_world @ contact_local_render
            )
            marker_pos = marker_pos + panel.outward_world * args.contact_marker_offset
            marker_specs = ((marker_pos, args.video_marker_size, (1.0, 0.0, 0.0, 1.0)),)
            frames.append(
                _render_frame(
                    render_env,
                    panel,
                    None,
                    args,
                    marker_specs=marker_specs,
                )
            )

        _write_mp4(output_path, frames, args.video_fps)
        print(
            f"Saved origin demonstration video to {output_path} "
            f"(dataset={dataset}, episode={episode_name}, contact_frame={contact_frame}, frames={len(frames)})"
        )
        return str(output_path)
    finally:
        source_env.close()
        render_env.close()


def _add_scene_sphere(scene, pos, radius, rgba):
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _video_camera_lookat(env, panel, solution=None, args=None):
    robot = env.robots[0]
    ee_pos = np.asarray(
        env.sim.data.site_xpos[robot.eef_site_id["right"]], dtype=np.float64
    )
    mode = getattr(args, "video_camera_lookat", "panel")
    if mode == "panel":
        lookat = np.asarray(panel.center_world, dtype=np.float64).copy()
        lookat[2] += 0.08
        return lookat
    if mode == "contact":
        lookat = np.asarray(
            solution.drawer_contact_world
            if solution is not None
            else panel.center_world,
            dtype=np.float64,
        ).copy()
        lookat[2] += 0.08
        return lookat
    contact_pos = np.asarray(
        solution.drawer_contact_world if solution is not None else panel.center_world,
        dtype=np.float64,
    )
    lookat = 0.55 * contact_pos + 0.45 * ee_pos
    lookat[2] += 0.08
    return lookat


def _yaw_from_xy(vec):
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    return float(np.degrees(np.arctan2(vec[1], vec[0])))


def _render_frame(env, panel, solution, args, marker_specs=None, reference_panel=None):
    import mujoco

    camera_name = args.video_camera
    if camera_name.lower() not in ("none", "free"):
        image = env.sim.render(
            width=int(args.video_width),
            height=int(args.video_height),
            camera_name=camera_name,
        )
        return np.asarray(image[::-1], dtype=np.uint8)

    context = env.sim._render_context_offscreen
    if context is None:
        raise RuntimeError("Offscreen render context is not initialized.")

    width = int(args.video_width)
    height = int(args.video_height)
    context.gl_ctx.make_current()
    if width > context.con.offWidth or height > context.con.offHeight:
        context.update_offscreen_size(
            max(width, context.model.vis.global_.offwidth),
            max(height, context.model.vis.global_.offheight),
        )

    context.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    context.cam.lookat[:] = _video_camera_lookat(env, panel, solution, args)
    context.cam.distance = float(args.video_camera_distance)
    azimuth = float(args.video_camera_azimuth)
    if reference_panel is not None:
        azimuth += _yaw_from_xy(panel.outward_world) - _yaw_from_xy(
            reference_panel.outward_world
        )
    context.cam.azimuth = azimuth
    context.cam.elevation = float(args.video_camera_elevation)

    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjv_updateScene(
        context.model._model,
        context.data._data,
        context.vopt,
        context.pert,
        context.cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        context.scn,
    )
    for pos, radius, rgba in marker_specs or []:
        _add_scene_sphere(context.scn, pos, radius, rgba)
    mujoco.mjr_render(viewport=viewport, scn=context.scn, con=context.con)
    return np.asarray(context.read_pixels(width, height)[::-1], dtype=np.uint8)


def save_mink_solution_videos(
    env, panel, robot_state, all_mink_solutions, push_distance, args
):
    if not args.save_trajectory_videos or not all_mink_solutions:
        return []
    if env.sim._render_context_offscreen is None:
        print("Trajectory videos skipped: offscreen renderer was not initialized.")
        return []

    output_dir = Path(args.video_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in sorted(output_dir.glob(f"{args.video_prefix}_*.mp4")):
        stale_path.unlink()
    arm_joint_names = robot_state["robocasa_joint_names"]
    qpos = env.sim.data.qpos.copy()
    qvel = env.sim.data.qvel.copy()
    saved_paths = []
    try:
        for video_index, solution in enumerate(all_mink_solutions, start=1):
            env.sim.data.qpos[:] = qpos
            env.sim.data.qvel[:] = qvel
            env.sim.forward()
            arm_trajectory, drawer_trajectory, _ = _build_mink_visualization_segment(
                env,
                robot_state,
                solution,
                push_distance,
                args,
            )
            frames = []
            for q_arm, drawer_q in zip(arm_trajectory, drawer_trajectory):
                _set_env_arm_q(env, arm_joint_names, q_arm)
                _set_drawer_joint_value(env, drawer_q)
                env.sim.forward()
                drawer_marker = (
                    solution.drawer_contact_world
                    + panel.outward_world * args.contact_marker_offset
                )
                ee_marker = _ee_sample_world(
                    env, args.mink_contact_frame, solution.contact_offset_local
                )
                marker_specs = (
                    (drawer_marker, args.video_marker_size, (1.0, 0.0, 0.0, 1.0)),
                    (ee_marker, args.video_marker_size * 0.75, (0.0, 1.0, 0.15, 1.0)),
                )
                frames.append(
                    _render_frame(env, panel, solution, args, marker_specs=marker_specs)
                )
            output_path = output_dir / f"{args.video_prefix}_{video_index:03d}.mp4"
            _write_mp4(output_path, frames, args.video_fps, args)
            saved_paths.append(str(output_path))
    finally:
        env.sim.data.qpos[:] = qpos
        env.sim.data.qvel[:] = qvel
        env.sim.forward()
    print(f"Saved {len(saved_paths)} trajectory videos to {output_dir}")
    return saved_paths


def visualize_contact_marker(
    env,
    panel,
    selected,
    push_distance,
    robot_state,
    q_traj,
    mink_solution,
    all_mink_solutions,
    args,
):
    if env.viewer is None:
        print("Contact visualization skipped: renderer was not initialized.")
        return

    solution_by_candidate = {
        solution.drawer_candidate_index: solution
        for solution in (all_mink_solutions or [])
    }
    marker_pos = selected.world_point + panel.outward_world * args.contact_marker_offset
    marker_rgba = np.array([1.0, 0.0, 0.0, args.contact_marker_alpha], dtype=np.float32)
    ee_marker_rgba = np.array([0.0, 1.0, 0.15, 1.0], dtype=np.float32)
    arm_joint_names = robot_state["robocasa_joint_names"]
    (
        arm_trajectory,
        drawer_trajectory,
        solution_indices,
        trajectory_source,
    ) = _visualization_trajectory(
        env,
        robot_state,
        q_traj,
        mink_solution,
        all_mink_solutions,
        push_distance,
        args,
    )
    if arm_trajectory.shape[0] == 0:
        print("Contact visualization skipped: empty trajectory.")
        return

    # --- PD torque control so the arm respects contacts with the drawer
    #     instead of teleporting through it (the old _set_env_arm_q did).
    timestep = float(env.sim.model.opt.timestep)
    kp = np.full(
        7, float(getattr(args, "contact_marker_pd_kp", 350.0)), dtype=np.float64
    )
    kd = np.full(
        7, float(getattr(args, "contact_marker_pd_kd", 35.0)), dtype=np.float64
    )
    qpos_addrs, dof_addrs, actuator_ids = _arm_pd_control_maps(env, arm_joint_names)
    target_dt = float(
        getattr(
            args,
            "contact_marker_pd_target_dt",
            getattr(args, "curobo_interpolation_dt", 0.02),
        )
    )
    sim_steps_per_target = max(int(round(max(target_dt, timestep) / timestep)), 1)

    print(
        f"Showing selected contact point and {trajectory_source} trajectory "
        f"(PD: kp={float(kp[0]):.0f}, kd={float(kd[0]):.0f}, "
        f"steps/target={sim_steps_per_target}). "
        "Close the viewer window or press Ctrl+C to exit."
    )
    try:
        env.viewer.update()
    except Exception as exc:
        print(f"Contact visualization skipped: viewer launch failed: {exc}")
        return
    viewer = getattr(env.viewer, "viewer", None)
    if viewer is None:
        print("Contact visualization skipped: viewer launch failed.")
        return
    if hasattr(viewer, "cam"):
        viewer.cam.type = 0
        viewer.cam.fixedcamid = -1

    started_at = time.time()
    camera_initialized = False
    frame_index = 0
    render_count = 0
    renders_per_target = max(sim_steps_per_target, 1)
    try:
        while True:
            if hasattr(viewer, "is_running") and not viewer.is_running():
                break
            active_solution = None
            if arm_trajectory.shape[0] > 0:
                active_solution = solution_by_candidate.get(
                    int(solution_indices[frame_index])
                )
                q_target = np.asarray(arm_trajectory[frame_index], dtype=np.float64)
                if render_count > 0 and render_count % renders_per_target == 0:
                    frame_index = (frame_index + 1) % arm_trajectory.shape[0]
                    active_solution = solution_by_candidate.get(
                        int(solution_indices[frame_index])
                    )
                    q_target = np.asarray(arm_trajectory[frame_index], dtype=np.float64)
                _set_drawer_joint_value(env, drawer_trajectory[frame_index])
                _step_arm_pd(env, q_target, qpos_addrs, dof_addrs, actuator_ids, kp, kd)
                render_count += 1
                if active_solution is not None:
                    marker_pos = (
                        active_solution.drawer_contact_world
                        + panel.outward_world * args.contact_marker_offset
                    )
            if hasattr(viewer, "user_scn"):
                viewer.user_scn.ngeom = 0
                _draw_viewer_sphere(
                    viewer, marker_pos, args.contact_marker_size, marker_rgba
                )
                if active_solution is not None:
                    ee_marker_pos = _ee_sample_world(
                        env,
                        args.mink_contact_frame,
                        active_solution.contact_offset_local,
                    )
                    _draw_viewer_sphere(
                        viewer,
                        ee_marker_pos,
                        args.contact_marker_size * 0.75,
                        ee_marker_rgba,
                    )
            if hasattr(viewer, "cam"):
                if not camera_initialized:
                    viewer.cam.lookat[:] = marker_pos
                    viewer.cam.distance = args.contact_camera_distance
                    viewer.cam.azimuth = args.contact_camera_azimuth
                    viewer.cam.elevation = args.contact_camera_elevation
                    camera_initialized = True
            if hasattr(viewer, "sync"):
                viewer.sync()
            if (
                args.visualize_contact_seconds > 0
                and time.time() - started_at >= args.visualize_contact_seconds
            ):
                break
            time.sleep(1.0 / max(args.contact_marker_fps, 1.0))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Contact visualization stopped: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a new CloseDrawer front-panel contact point and plan a Panda arm trajectory with cuRobo."
    )
    parser.add_argument("--robot", type=str, default="PandaOmron")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", type=int, default=-1)
    parser.add_argument("--style", type=int, default=-1)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument(
        "--origin_demo", "--origin-demo", dest="origin_demo", action="store_true"
    )
    parser.add_argument("--origin-demo-index", type=int, default=0)
    parser.add_argument("--origin-demo-output", type=str, default="")
    parser.add_argument("--origin-demo-video-skip", type=int, default=5)
    parser.add_argument("--origin-demo-extend-states", type=int, default=50)
    parser.add_argument(
        "--ts-skeleton",
        dest="ts_skeleton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Demo-driven drawer close: extract demo trajectory, solve skeleton "
        "poses at the reference pose, project across 10Hz interpolated drawer "
        "poses, filter penetration, animate in a popup (0.1s / frame). "
        "Enabled by default; disable with --no-ts-skeleton.",
    )
    parser.add_argument("--ts-viz-frame-seconds", type=float, default=0.1)
    parser.add_argument("--ts-viz-render-fps", type=float, default=30.0)
    parser.add_argument(
        "--ts-viz-skeleton-pose-limit",
        type=int,
        default=120,
        help="Maximum skeleton poses drawn per visualization timestep. Use <=0 to draw until scene capacity.",
    )
    parser.add_argument(
        "--ts-sample-hz",
        type=float,
        default=10.0,
        help="Drawer trajectory resampling rate (Hz) for the ts-skeleton pipeline.",
    )
    parser.add_argument(
        "--ts-penetration-workers",
        type=int,
        default=None,
        help="Threads for per-frame mj_multiRay penetration filter. Defaults to cpu_count-2.",
    )
    parser.add_argument(
        "--ts-penetration-mjwarp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mjwarp batched narrowphase for per-frame penetration checking "
        "(~200x faster than CPU mj_multiRay). Disable with --no-ts-penetration-mjwarp.",
    )
    parser.add_argument(
        "--ts-penetration-mjwarp-batch",
        type=int,
        default=128,
        help="Worlds per mjwarp forward call (higher = more GPU memory but fewer launches).",
    )
    parser.add_argument(
        "--ts-skeleton-max-reference-poses",
        type=int,
        default=512,
        help="Randomly downsample reference skeleton poses to this count before projection.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--visualize-contact", dest="visualize_contact", action="store_true"
    )
    parser.add_argument(
        "--no-visualize-contact", dest="visualize_contact", action="store_false"
    )
    parser.set_defaults(visualize_contact=None)
    parser.add_argument("--visualize-contact-seconds", type=float, default=0.0)
    parser.add_argument("--contact-marker-size", type=float, default=0.035)
    parser.add_argument("--contact-marker-offset", type=float, default=0.012)
    parser.add_argument("--contact-marker-alpha", type=float, default=1.0)
    parser.add_argument("--contact-marker-fps", type=float, default=30.0)
    parser.add_argument("--contact-marker-pd-kp", type=float, default=350.0)
    parser.add_argument("--contact-marker-pd-kd", type=float, default=35.0)
    parser.add_argument("--contact-marker-pd-target-dt", type=float, default=0.02)
    parser.add_argument("--contact-camera-distance", type=float, default=0.9)
    parser.add_argument("--contact-camera-azimuth", type=float, default=135.0)
    parser.add_argument("--contact-camera-elevation", type=float, default=-25.0)
    parser.add_argument(
        "--visualize-trajectory", dest="visualize_trajectory", action="store_true"
    )
    parser.add_argument(
        "--no-visualize-trajectory", dest="visualize_trajectory", action="store_false"
    )
    parser.set_defaults(visualize_trajectory=True)
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
    parser.add_argument("--visualize-steps-per-segment", type=int, default=45)
    parser.add_argument(
        "--visualize-all-contact-poses",
        dest="visualize_all_contact_poses",
        action="store_true",
    )
    parser.add_argument(
        "--visualize-selected-contact-pose",
        dest="visualize_all_contact_poses",
        action="store_false",
    )
    parser.set_defaults(visualize_all_contact_poses=True)
    parser.add_argument("--visualize-pause-frames", type=int, default=20)
    parser.add_argument(
        "--save-trajectory-videos", dest="save_trajectory_videos", action="store_true"
    )
    parser.add_argument(
        "--no-save-trajectory-videos",
        dest="save_trajectory_videos",
        action="store_false",
    )
    parser.set_defaults(save_trajectory_videos=True)
    parser.add_argument(
        "--video-output-dir", type=str, default="/home/lab423/wm/vlm/robocasa/outputs"
    )
    parser.add_argument("--video-prefix", type=str, default="close_drawer")
    parser.add_argument("--video-camera", type=str, default="free")
    parser.add_argument("--video-camera-distance", type=float, default=1.35)
    parser.add_argument("--video-camera-azimuth", type=float, default=90.0)
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
        "--video-encoder", type=str, default="ffmpeg", choices=("ffmpeg", "opencv")
    )
    parser.add_argument("--ffmpeg-path", type=str, default="")
    parser.add_argument("--video-codec", type=str, default="libx264")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", type=str, default="slow")

    parser.add_argument("--grid-x", type=int, default=5)
    parser.add_argument("--grid-z", type=int, default=3)
    parser.add_argument("--panel-margin", type=float, default=0.025)
    parser.add_argument("--push-distance", type=float, default=0.12)
    parser.add_argument(
        "--execute-push-stage",
        dest="execute_push_stage",
        action="store_true",
        help="Append the inward push waypoint to the executed trajectory. Disabled by "
        "default so curobo stops at the contact pose and does not penetrate the panel.",
    )
    parser.add_argument(
        "--no-execute-push-stage",
        dest="execute_push_stage",
        action="store_false",
    )
    parser.set_defaults(execute_push_stage=False)
    parser.add_argument("--target-open-fraction", type=float, default=0.02)
    parser.add_argument("--precontact-distance", type=float, default=0.08)
    parser.add_argument("--contact-standoff", type=float, default=0.005)

    parser.add_argument("--contact-cost-threshold", type=float, default=0.10)
    parser.add_argument("--require-feasible-contact", action="store_true")
    parser.add_argument(
        "--contact-solver",
        type=str,
        default="ipopt",
        choices=("ipopt", "snopt", "acados"),
    )
    parser.add_argument("--contact-lam-upper-bound", type=float, default=2.0)
    parser.add_argument("--contact-obj-mass", type=float, default=0.01)
    parser.add_argument("--contact-friction", type=float, default=0.9)
    parser.add_argument("--contact-stiffness", type=float, default=12.5)
    parser.add_argument("--contact-dt", type=float, default=0.01)
    parser.add_argument("--contact-pos-coef", type=float, default=1.0)
    parser.add_argument("--contact-ori-coef", type=float, default=0.0005)
    parser.add_argument(
        "--use-autogen-contact", dest="use_autogen_contact", action="store_true"
    )
    parser.add_argument(
        "--disable-autogen-contact", dest="use_autogen_contact", action="store_false"
    )
    parser.set_defaults(use_autogen_contact=True)
    parser.add_argument(
        "--solve_step2",
        "--solve-step2",
        dest="solve_step2",
        type=str,
        default="MPPI",
        choices=("MPPI", "mink", "mppi", "MINK"),
    )
    parser.add_argument("--require-friction-cone-push", action="store_true")
    parser.add_argument("--autogen-object-point-count", type=int, default=384)
    parser.add_argument(
        "--autogen-handle-subdivide-max-edge", type=float, default=0.005
    )
    parser.add_argument("--autogen-initial-pose-count", type=int, default=200)
    parser.add_argument("--autogen-mink-max-attempts", type=int, default=200)
    parser.add_argument("--autogen-mink-parallel-workers", type=int, default=1)
    parser.add_argument("--autogen-skeleton-parallel-workers", type=int, default=None)
    parser.add_argument("--autogen-accept-position-tolerance", type=float, default=0.06)
    parser.add_argument("--autogen-panel-edge-margin", type=float, default=0.05)
    parser.add_argument("--autogen-panel-edge-fraction", type=float, default=0.28)
    parser.add_argument("--autogen-panel-top-edge-fraction", type=float, default=0.38)
    parser.add_argument("--autogen-coacd-threshold", type=float, default=0.05)
    parser.add_argument("--autogen-coacd-max-convex-hull", type=int, default=32)
    parser.add_argument("--autogen-coacd-preprocess-mode", type=str, default="auto")
    parser.add_argument("--autogen-coacd-preprocess-resolution", type=int, default=30)
    parser.add_argument("--autogen-coacd-resolution", type=int, default=2000)
    parser.add_argument("--autogen-coacd-mcts-nodes", type=int, default=20)
    parser.add_argument("--autogen-coacd-mcts-iterations", type=int, default=100)
    parser.add_argument("--autogen-coacd-mcts-max-depth", type=int, default=3)
    parser.add_argument("--autogen-coacd-max-ch-vertex", type=int, default=256)
    parser.add_argument(
        "--autogen-use-current-ee-rotation",
        dest="autogen_use_current_ee_rotation",
        action="store_true",
    )
    parser.add_argument(
        "--autogen-solve-free-ee-rotation",
        dest="autogen_use_current_ee_rotation",
        action="store_false",
    )
    parser.set_defaults(autogen_use_current_ee_rotation=False)
    parser.add_argument(
        "--autogen-allow-current-rotation-fallback", action="store_true"
    )
    parser.add_argument("--autogen-visualize-skeleton", action="store_true")
    parser.add_argument("--autogen-visualize-skeleton-poses", action="store_true")
    parser.add_argument("--autogen-skeleton-margin", type=float, default=0.002)
    parser.add_argument("--autogen-skeleton-theta-count", type=int, default=12)
    parser.add_argument("--autogen-skeleton-segment-samples", type=int, default=5)
    parser.add_argument("--autogen-skeleton-initial-lift", type=float, default=0.005)
    parser.add_argument("--autogen-skeleton-lift-weight", type=float, default=100.0)
    parser.add_argument("--autogen-skeleton-gripper-default", type=float, default=0.04)
    parser.add_argument("--autogen-skeleton-gripper-weight", type=float, default=10.0)
    parser.add_argument("--autogen-skeleton-gripper-min", type=float, default=0.005)
    parser.add_argument("--autogen-skeleton-gripper-max", type=float, default=0.08)
    parser.add_argument("--autogen-skeleton-reg-weight", type=float, default=1.0)
    parser.add_argument("--autogen-skeleton-motion-bound", type=float, default=0.05)
    parser.add_argument("--autogen-skeleton-rot-bound", type=float, default=0.35)
    parser.add_argument(
        "--autogen-skeleton-object-penetration-tol", type=float, default=0.001
    )
    parser.add_argument(
        "--autogen-skeleton-clearance-tolerance", type=float, default=0.001
    )
    parser.add_argument(
        "--autogen-skeleton-pose-variants-per-contact", type=int, default=8
    )
    parser.add_argument(
        "--autogen-skeleton-pose-min-theta-separation",
        type=float,
        default=float(np.pi / 6.0),
    )
    parser.add_argument(
        "--autogen-visualize-skeleton-pose-limit", type=int, default=120
    )
    parser.add_argument("--autogen-qmppi-num-samples", type=int, default=256)
    parser.add_argument("--autogen-qmppi-num-iterations", type=int, default=6)
    parser.add_argument("--autogen-qmppi-elite-ratio", type=float, default=0.1)
    parser.add_argument("--autogen-qmppi-temperature", type=float, default=1.0)
    parser.add_argument("--autogen-qmppi-q-noise-scale", type=float, default=0.05)
    parser.add_argument("--autogen-qmppi-tracking-weight", type=float, default=1.0)
    parser.add_argument("--autogen-qmppi-penetration-weight", type=float, default=0.0)
    parser.add_argument("--autogen-qmppi-contact-weight", type=float, default=0.0)
    parser.add_argument("--autogen-qmppi-contact-tolerance", type=float, default=0.002)
    parser.add_argument("--autogen-qmppi-pos-weight", type=float, default=200.0)
    parser.add_argument("--autogen-qmppi-rot-weight", type=float, default=20.0)
    parser.add_argument(
        "--autogen-qmppi-gripper-noise-scale", type=float, default=0.015
    )
    parser.add_argument(
        "--autogen-qmppi-gripper-tracking-weight", type=float, default=50.0
    )
    parser.add_argument("--autogen-qmppi-gripper-min", type=float, default=0.0)
    parser.add_argument("--autogen-qmppi-gripper-max", type=float, default=0.08)
    parser.add_argument(
        "--autogen-qmppi-penetration-threshold", type=float, default=0.005
    )
    parser.add_argument("--autogen-qmppi-seed", type=int, default=0)
    parser.add_argument("--autogen-qmppi-horizon-steps", type=int, default=5)
    parser.add_argument(
        "--autogen-qmppi-approach-total-distance", type=float, default=0.01
    )
    parser.add_argument(
        "--autogen-qmppi-object-improvement-eps", type=float, default=1e-5
    )
    parser.add_argument(
        "--autogen-qmppi-accept-object-improvement-only",
        dest="autogen_qmppi_accept_object_improvement_only",
        action="store_true",
    )
    parser.add_argument(
        "--autogen-qmppi-require-contact-constraints",
        dest="autogen_qmppi_accept_object_improvement_only",
        action="store_false",
    )
    parser.set_defaults(autogen_qmppi_accept_object_improvement_only=True)
    parser.add_argument(
        "--autogen-skip-mink-q-after-mppi",
        dest="autogen_skip_mink_q_after_mppi",
        action="store_true",
    )
    parser.add_argument(
        "--autogen-use-mink-q-after-mppi",
        dest="autogen_skip_mink_q_after_mppi",
        action="store_false",
    )
    parser.set_defaults(autogen_skip_mink_q_after_mppi=False)

    parser.add_argument(
        "--use-mink-contact-pose", dest="use_mink_contact_pose", action="store_true"
    )
    parser.add_argument(
        "--disable-mink-contact-pose",
        dest="use_mink_contact_pose",
        action="store_false",
    )
    parser.set_defaults(use_mink_contact_pose=True)
    parser.add_argument(
        "--mink-contact-frame", type=str, default="gripper0_right_grip_site"
    )
    parser.add_argument(
        "--mink-include-grip-site", dest="mink_include_grip_site", action="store_true"
    )
    parser.add_argument(
        "--mink-exclude-grip-site", dest="mink_include_grip_site", action="store_false"
    )
    parser.set_defaults(mink_include_grip_site=False)
    parser.add_argument("--mink-ee-sample-count", type=int, default=15)
    parser.add_argument("--mink-drawer-contact-count", type=int, default=15)
    parser.add_argument("--mink-roll-samples", type=int, default=8)
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
    parser.add_argument(
        "--mink-collision-penetration-tolerance", type=float, default=0.02
    )
    parser.add_argument("--mink-collision-penalty", type=float, default=10.0)
    parser.add_argument("--mink-q-precontact-distance", type=float, default=0.005)
    parser.add_argument(
        "--mink-q-retreat-distance-multipliers", type=str, default="0,0.5,1,1.5,2,3"
    )
    parser.add_argument("--mink-q-position-tolerance", type=float, default=None)
    parser.add_argument(
        "--mink-q-collision-penetration-tolerance", type=float, default=None
    )
    parser.add_argument(
        "--autogen-mink-q-mjwarp-checker",
        dest="autogen_mink_q_mjwarp_checker",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-mink-q-mjwarp-checker",
        dest="autogen_mink_q_mjwarp_checker",
        action="store_false",
    )
    parser.set_defaults(autogen_mink_q_mjwarp_checker=True)
    parser.add_argument("--autogen-mink-q-mjwarp-device", type=str, default="cuda:0")
    parser.add_argument("--autogen-mink-q-mjwarp-nconmax", type=int, default=256)
    parser.add_argument("--autogen-mink-q-mjwarp-njmax", type=int, default=1024)
    parser.add_argument(
        "--autogen-mink-q-mjwarp-comfree",
        dest="autogen_mink_q_mjwarp_comfree",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-mink-q-mjwarp-comfree",
        dest="autogen_mink_q_mjwarp_comfree",
        action="store_false",
    )
    parser.set_defaults(autogen_mink_q_mjwarp_comfree=True)
    parser.add_argument("--autogen-mink-q-checker-workers", type=int, default=None)
    parser.add_argument("--autogen-mink-q-worlds-per-worker", type=int, default=None)
    parser.add_argument(
        "--autogen-mink-q-debug", dest="autogen_mink_q_debug", action="store_true"
    )
    parser.add_argument(
        "--no-autogen-mink-q-debug", dest="autogen_mink_q_debug", action="store_false"
    )
    parser.set_defaults(autogen_mink_q_debug=False)
    parser.add_argument("--autogen-mink-q-debug-limit", type=int, default=12)
    parser.add_argument("--require-mink-contact-pose", action="store_true")
    parser.add_argument("--require-mink-collision-free", action="store_true")
    parser.add_argument(
        "--curobo-mink-joint-space", dest="curobo_mink_joint_space", action="store_true"
    )
    parser.add_argument(
        "--curobo-mink-ee-pose", dest="curobo_mink_joint_space", action="store_false"
    )
    parser.set_defaults(curobo_mink_joint_space=True)

    parser.add_argument("--skip-curobo", action="store_true")
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
        "--curobo-use-mujoco-world", dest="curobo_use_mujoco_world", action="store_true"
    )
    parser.add_argument(
        "--curobo-no-mujoco-world", dest="curobo_use_mujoco_world", action="store_false"
    )
    parser.set_defaults(curobo_use_mujoco_world=True)
    parser.add_argument(
        "--curobo-world-exclude-target-drawer",
        dest="curobo_world_exclude_target_drawer",
        action="store_true",
    )
    parser.add_argument(
        "--curobo-world-include-target-drawer",
        dest="curobo_world_exclude_target_drawer",
        action="store_false",
    )
    parser.set_defaults(curobo_world_exclude_target_drawer=True)
    parser.add_argument("--curobo-world-exclude-geoms", type=str, default="")
    parser.add_argument("--curobo-world-exclude-bodies", type=str, default="")
    parser.add_argument("--curobo-world-padding", type=float, default=0.005)
    parser.add_argument("--curobo-world-max-obstacles", type=int, default=None)
    parser.add_argument(
        "--curobo-collision-activation-distance", type=float, default=0.005
    )
    parser.add_argument(
        "--output", type=str, default="/tmp/robocasa_close_drawer_contact_curobo.npz"
    )
    parser.add_argument("--save-output", dest="save_output", action="store_true")
    parser.add_argument("--no-save-output", dest="save_output", action="store_false")
    parser.set_defaults(save_output=True)
    args = parser.parse_args()
    if args.visualize_contact is None:
        args.visualize_contact = bool(args.skip_curobo)
    return args


def main():
    args = parse_args()
    if args.origin_demo:
        save_origin_demo_video(args)
        return
    if bool(getattr(args, "ts_skeleton", False)):
        run_timeseries_pipeline(args)
        return
    if not args.skip_curobo:
        preload_curobo_runtime()
    env = create_close_drawer_env(args)
    try:
        panel = get_panel_frame(env)
        candidates, selected, push_distance = evaluate_contacts(env, panel, args)
        robot_state = get_robot_arm_state(env)
        mink_solution = None
        all_mink_solutions = []
        mink_reports = []
        if args.use_mink_contact_pose:
            try:
                (
                    mink_solution,
                    all_mink_solutions,
                    mink_reports,
                ) = solve_contact_poses_with_mink(
                    env,
                    panel,
                    candidates,
                    push_distance,
                    robot_state,
                    args,
                )
                if mink_solution is not None:
                    selected = candidates[mink_solution.drawer_candidate_index]
                    target_gripper_poses = mink_solution.target_gripper_poses
                else:
                    target_gripper_poses = build_target_gripper_poses(
                        panel, selected, push_distance, args
                    )
            except Exception as exc:
                if args.require_mink_contact_pose:
                    raise
                print(
                    f"mink contact pose solve failed; falling back to nominal gripper-center pose: {exc}"
                )
                target_gripper_poses = build_target_gripper_poses(
                    panel, selected, push_distance, args
                )
        else:
            target_gripper_poses = build_target_gripper_poses(
                panel, selected, push_distance, args
            )
        target_hand_poses = [
            (name, *gripper_pose_to_curobo_hand_pose(pos, rot, robot_state))
            for name, pos, rot in target_gripper_poses
        ]

        q_traj = None
        segments = []
        if not args.skip_curobo:
            try:
                mink_q_waypoints_for_curobo = None
                if mink_solution is not None and bool(
                    getattr(args, "curobo_mink_joint_space", True)
                ):
                    q_waypoints = np.asarray(
                        getattr(mink_solution, "q_waypoints", np.zeros((0, 7))),
                        dtype=np.float64,
                    )
                    if q_waypoints.size:
                        q_waypoints = q_waypoints.reshape(-1, 7)
                    if q_waypoints.shape[0] > 0:
                        mink_q_waypoints_for_curobo = q_waypoints
                        print(
                            "CUROBO_MINK_JOINT_SPACE "
                            f"enabled=True waypoints={q_waypoints.shape[0]} "
                            f"solve_step2={getattr(args, 'solve_step2', None)!r}",
                            flush=True,
                        )
                q_traj, segments = plan_with_curobo(
                    robot_state,
                    target_hand_poses,
                    args,
                    env=env,
                    q_waypoints=mink_q_waypoints_for_curobo,
                )
            except Exception as exc:
                save_outputs._enabled = bool(getattr(args, "save_output", True))
                metadata = save_outputs(
                    args.output,
                    env,
                    panel,
                    candidates,
                    selected,
                    push_distance,
                    target_gripper_poses,
                    target_hand_poses,
                    q_traj,
                    segments,
                    mink_solution,
                    all_mink_solutions,
                    mink_reports,
                )
                if bool(getattr(args, "save_output", True)):
                    print(
                        f"Saved contact-only output before cuRobo failure: {args.output}"
                    )
                else:
                    print("Contact-only output save skipped by --no-save-output")
                raise RuntimeError(
                    "cuRobo planning failed after contact sampling. "
                    "Install/enable cuRobo runtime dependencies or rerun with --skip-curobo "
                    "to inspect contact outputs only."
                ) from exc

        save_outputs._enabled = bool(getattr(args, "save_output", True))
        metadata = save_outputs(
            args.output,
            env,
            panel,
            candidates,
            selected,
            push_distance,
            target_gripper_poses,
            target_hand_poses,
            q_traj,
            segments,
            mink_solution,
            all_mink_solutions,
            mink_reports,
        )
        if bool(getattr(args, "save_output", True)):
            print(f"Saved: {args.output}")
        else:
            print("Output save skipped by --no-save-output")
        if args.save_trajectory_videos:
            save_mink_solution_videos(
                env, panel, robot_state, all_mink_solutions, push_distance, args
            )
        if args.visualize_contact:
            visualize_contact_marker(
                env,
                panel,
                selected,
                push_distance,
                robot_state,
                q_traj,
                mink_solution,
                all_mink_solutions,
                args,
            )
    finally:
        env.close()


# Keep a handle to the close-drawer implementation copied above. The autogen
# wrapper patches module-level functions through this self alias while preserving
# the original wrapper flow.
_close_contact_main = main
_close_impl = sys.modules[__name__]


def _autogen_print(message):
    print(message, file=sys.__stdout__, flush=True)


def _hand_box_from_ghost_geoms(env, frame_name):
    from robocasa.demos import visualize_mujoco as viz_mj

    geoms = viz_mj._extract_hand_finger_ghost_geoms(env, frame_name)
    best = None
    best_volume = -1.0
    for ghost in geoms:
        half = np.asarray(ghost.size, dtype=np.float64).reshape(3)
        # Fingers are long and thin; the palm/hand mesh is the largest compact
        # box among the EE ghost geoms. Reject implausibly large scene links if
        # the source-body filter ever becomes too broad.
        if float(np.max(half)) > 0.20:
            continue
        volume = float(np.prod(np.maximum(half, 1e-6)))
        if volume > best_volume:
            best = ghost
            best_volume = volume
    if best is None:
        return None
    return (
        np.asarray(best.local_pos, dtype=np.float64).reshape(3).copy(),
        np.asarray(best.local_rot, dtype=np.float64).reshape(3, 3).copy(),
        np.asarray(best.size, dtype=np.float64).reshape(3).copy(),
    )


def _drawer_panel_pose(env):
    """Read the current drawer door panel pose from MuJoCo geom data.

    Returns ``(center_world, rotation_world)`` mirroring the ref point used
    by ``get_panel_frame`` so skeleton poses can be projected onto a moving
    drawer the same way ``_project_skeleton_poses_to_drawer_pose`` does.
    """
    model = env.sim.model
    data = env.sim.data
    drawer_name = env.drawer.name
    reg_name = f"{drawer_name}_door_reg_main"
    if reg_name in model._geom_name2id:
        panel_geom = reg_name
    else:
        panel_geom = f"{drawer_name}_door_g1"
    if panel_geom not in model._geom_name2id:
        return None, None
    geom_id = model.geom_name2id(panel_geom)
    center_world = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3)
    rotation_world = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    return center_world, rotation_world


def _raw_panel_pose(raw_model, raw_data, panel_geom_name):
    """Read ``(center_world, rotation_world)`` from raw MuJoCo data."""
    import mujoco as _mj

    if panel_geom_name is None:
        return None, None
    gid = _mj.mj_name2id(raw_model, _mj.mjtObj.mjOBJ_GEOM, panel_geom_name)
    center = np.asarray(raw_data.geom_xpos[int(gid)], dtype=np.float64).reshape(3)
    rot = np.asarray(raw_data.geom_xmat[int(gid)], dtype=np.float64).reshape(3, 3)
    return center, rot


def _panel_geom_name(env):
    """Return the drawer door geom name used to track the panel frame."""
    model = env.sim.model
    drawer_name = env.drawer.name
    reg_name = f"{drawer_name}_door_reg_main"
    if reg_name in model._geom_name2id:
        return reg_name
    alt = f"{drawer_name}_door_g1"
    if alt in model._geom_name2id:
        return alt
    return None


def _set_arm_q_direct(raw_model, raw_data, arm_joint_names, q_arm, env):
    """Set arm joint positions directly into ``raw_data.qpos``."""
    for joint_name, q_val in zip(arm_joint_names, np.asarray(q_arm).reshape(-1)):
        try:
            addr = int(env.sim.model.get_joint_qpos_addr(joint_name))
        except Exception:
            continue
        raw_data.qpos[addr] = float(q_val)


def _visualize_mink_q_poses_popup(
    env,
    q_waypoints,
    robot_state,
    args,
    drawer_q=None,
    *,
    drawer_q_seq=None,
    ts_poses_ref=None,
    panel_pose_ref=None,
    drawer_qpos_addr=None,
    frame_seconds=None,
):
    """Popup viewer rendering ghost Panda hand at every mink q waypoint.

    When ``drawer_q_seq`` is supplied the popup turns into an *animation*:
    ``drawer_q_seq[i]`` is applied to the drawer joint (via
    ``raw_data.qpos[drawer_qpos_addr]``) on frame ``i``, the arm is set to
    ``q_waypoints[i]``, and any reference ``ts_poses_ref`` (a list of
    ``[drawer_q, PanelFrame, skeleton_poses]`` tuples, typically the output
    of ``solve_timeseries_skeleton_poses``) is projected to the current
    drawer panel pose so the skeleton follows the drawer slide.

    Without ``drawer_q_seq`` the behaviour matches the original: draw every
    q_waypoint pose with the (single, shared) ``drawer_q``.
    """
    if not bool(getattr(args, "autogen_visualize_mink_poses", True)):
        return
    q_waypoints = np.asarray(q_waypoints, dtype=np.float64)
    if q_waypoints.size == 0:
        return
    q_waypoints = q_waypoints.reshape(-1, 7)
    n_frames = q_waypoints.shape[0]
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    if frame_name not in env.sim.model._site_name2id:
        raise RuntimeError(
            f"Cannot visualize mink q poses: site {frame_name!r} not found"
        )

    try:
        import mujoco
        import mujoco.viewer
        from robocasa.demos import visualize_mujoco as viz_mj
    except Exception as exc:
        raise RuntimeError(f"Cannot visualize mink q poses: {exc}") from exc

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    site_id = int(env.sim.model.site_name2id(frame_name))
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    drawer_q = (
        float(_close_impl._drawer_joint_value(env))
        if drawer_q is None
        else float(drawer_q)
    )

    # --- Validate / normalise animation inputs.  An animation requires BOTH
    #     a drawer_q_seq AND a drawer_qpos_addr (so we can write the joint
    #     state directly into the viewer's raw_data each frame).  Anything
    #     else falls back to the original static behaviour.
    animate = (
        drawer_q_seq is not None
        and drawer_qpos_addr is not None
        and np.asarray(drawer_q_seq).size > 0
        and q_waypoints.shape[0] > 1
    )
    ref_ts = None  # list[tuple[drawer_q, PanelFrame, list[SkeletonPose]]]
    ref_panel = None
    panel_geom_name = None
    finger_radius = float(getattr(args, "autogen_skeleton_finger_radius", 0.004))
    frame_seconds_effective = float(
        frame_seconds
        if frame_seconds is not None
        else getattr(args, "autogen_mink_popup_frame_seconds", 0.1)
    )
    if animate:
        drawer_q_seq = np.asarray(drawer_q_seq, dtype=np.float64).reshape(-1)
        if drawer_q_seq.shape[0] < n_frames:
            pad = np.full(
                n_frames - drawer_q_seq.shape[0],
                float(drawer_q_seq[-1]) if drawer_q_seq.shape[0] else float(drawer_q),
                dtype=np.float64,
            )
            drawer_q_seq = np.concatenate([drawer_q_seq, pad])
        elif drawer_q_seq.shape[0] > n_frames:
            drawer_q_seq = drawer_q_seq[:n_frames]
        drawer_qpos_addr = int(drawer_qpos_addr)
        if ts_poses_ref:
            ref_ts = list(ts_poses_ref)
        if panel_pose_ref is None:
            try:
                center, rot = _drawer_panel_pose(env)
                if center is not None:
                    ref_panel = (
                        np.asarray(center, dtype=np.float64).reshape(3).copy(),
                        np.asarray(rot, dtype=np.float64).reshape(3, 3).copy(),
                    )
            except Exception:
                ref_panel = None
        else:
            ref_panel = (
                np.asarray(panel_pose_ref[0], dtype=np.float64).reshape(3).copy(),
                np.asarray(panel_pose_ref[1], dtype=np.float64).reshape(3, 3).copy(),
            )
        panel_geom_name = _panel_geom_name(env)
        del ts_poses_ref, panel_pose_ref

    poses = []
    try:
        _close_impl._set_drawer_joint_value(env, drawer_q)
        ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(env, frame_name)
        for q_arm in q_waypoints:
            _close_impl._set_env_arm_q(env, arm_joint_names, q_arm)
            _close_impl._set_drawer_joint_value(env, drawer_q)
            env.sim.forward()
            poses.append(
                (
                    np.asarray(
                        env.sim.data.site_xpos[site_id], dtype=np.float64
                    ).copy(),
                    np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64)
                    .reshape(3, 3)
                    .copy(),
                )
            )
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()

    if not poses:
        return
    lookat = np.mean(np.asarray([pose[0] for pose in poses], dtype=np.float64), axis=0)
    palette = np.asarray(
        [
            [0.05, 0.45, 1.0, float(getattr(args, "autogen_mink_ghost_alpha", 0.28))],
            [1.0, 0.58, 0.05, float(getattr(args, "autogen_mink_ghost_alpha", 0.28))],
            [0.05, 0.75, 0.35, float(getattr(args, "autogen_mink_ghost_alpha", 0.28))],
        ],
        dtype=np.float32,
    )
    green = np.array([0.1, 1.0, 0.2, 0.9], dtype=np.float32)
    try:
        skeleton = ee_skelton.build_panda_skeleton(env, frame_name)
    except Exception:
        skeleton = None
    with mujoco.viewer.launch_passive(
        raw_model,
        raw_data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        try:
            viewer.opt.geomgroup[:] = 0
            viewer.opt.geomgroup[1] = 1
            viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT)] = 0
            viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE)] = 0
        except Exception:
            pass
        viewer.cam.type = 0
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = float(
            getattr(args, "autogen_mink_popup_camera_distance", 0.85)
        )
        viewer.cam.azimuth = float(
            getattr(args, "autogen_mink_popup_camera_azimuth", 135.0)
        )
        viewer.cam.elevation = float(
            getattr(args, "autogen_mink_popup_camera_elevation", -25.0)
        )
        fps = max(float(getattr(args, "autogen_mink_popup_fps", 30.0)), 1.0)
        try:
            started_at = time.time()
            frame_idx = 0
            last_frame_idx = -1
            while viewer.is_running():
                # Advance the frame based on wall-clock time when animating, so
                # user sees the drawer slide + skeleton advance in (close to)
                # real-time regardless of render FPS.
                if animate:
                    elapsed = time.time() - started_at
                    frame_idx = int(elapsed / frame_seconds_effective) % n_frames
                if not animate or frame_idx != last_frame_idx:
                    last_frame_idx = frame_idx
                    if animate:
                        raw_data.qpos[drawer_qpos_addr] = float(drawer_q_seq[frame_idx])
                        q_arm = np.asarray(
                            q_waypoints[frame_idx], dtype=np.float64
                        ).reshape(7)
                        _set_arm_q_direct(
                            raw_model, raw_data, arm_joint_names, q_arm, env
                        )
                        mujoco.mj_forward(raw_model, raw_data)
                    if hasattr(viewer, "user_scn"):
                        viewer.user_scn.ngeom = 0
                        if animate:
                            # Current frame: full-opacity hand + skeleton.
                            for ghost in ghost_geoms:
                                viz_mj._add_ghost_geom(
                                    viewer.user_scn,
                                    ghost,
                                    np.asarray(
                                        raw_data.site_xpos[site_id], dtype=np.float64
                                    ),
                                    np.asarray(
                                        raw_data.site_xmat[site_id], dtype=np.float64
                                    ).reshape(3, 3),
                                    np.array([0.05, 0.45, 1.0, 0.95], dtype=np.float32),
                                )
                            # Skeletons: either project from ts_poses_ref for
                            # the nearest frame, or (if only panel_pose_ref was
                            # supplied) project a single reference bundle.
                            projected = None
                            if ref_ts:
                                # Pick the closest frame by drawer_q.
                                seq_qs = np.asarray(
                                    [float(t[0]) for t in ref_ts], dtype=np.float64
                                )
                                ref_frame_idx = int(
                                    np.argmin(np.abs(seq_qs - drawer_q_seq[frame_idx]))
                                )
                                ref_bundle = ref_ts[ref_frame_idx]
                                bundle_panel = (
                                    np.asarray(
                                        ref_bundle[1].center_world, dtype=np.float64
                                    ).reshape(3),
                                    np.asarray(
                                        ref_bundle[1].rotation_world, dtype=np.float64
                                    ).reshape(3, 3),
                                )
                                cur_center, cur_rot = (
                                    _raw_panel_pose(
                                        raw_model, raw_data, panel_geom_name
                                    )
                                    or bundle_panel
                                )
                                projected = _project_skeleton_poses_to_drawer_pose(
                                    list(ref_bundle[2]),
                                    bundle_panel,
                                    (cur_center, cur_rot),
                                )
                            if projected and skeleton is not None:
                                _skel_half_ext = None
                                for sp in projected:
                                    if viewer.user_scn.ngeom + 3 > int(
                                        viewer.user_scn.maxgeom
                                    ):
                                        break
                                    ee_pos = np.asarray(
                                        sp.ee_position, dtype=np.float64
                                    )
                                    ee_rot = np.asarray(
                                        sp.ee_rotation, dtype=np.float64
                                    )
                                    try:
                                        if _skel_half_ext is None:
                                            _skel_half_ext = np.asarray(
                                                ee_skelton._flat_hand_half_extents(
                                                    skeleton
                                                ),
                                                dtype=np.float64,
                                            )
                                        hand_center_w = ee_pos + ee_rot @ np.asarray(
                                            skeleton.hand_box_center_ee,
                                            dtype=np.float64,
                                        )
                                        hand_rot_w = ee_rot @ np.asarray(
                                            skeleton.hand_box_rotation_ee,
                                            dtype=np.float64,
                                        )
                                        ee_skelton._add_box(
                                            viewer.user_scn,
                                            hand_center_w,
                                            hand_rot_w,
                                            _skel_half_ext,
                                            np.array(
                                                [0.1, 1.0, 0.2, 0.7], dtype=np.float32
                                            ),
                                        )
                                        (
                                            left_seg,
                                            right_seg,
                                            _,
                                            _,
                                        ) = ee_skelton._finger_segments_with_opening(
                                            skeleton,
                                            float(
                                                getattr(
                                                    sp,
                                                    "gripper_opening",
                                                    ee_skelton.PANDA_DEFAULT_GRIPPER_OPENING,
                                                )
                                            ),
                                        )
                                        rgba_hand = np.array(
                                            [0.1, 1.0, 0.2, 0.7], dtype=np.float32
                                        )
                                        for seg in (left_seg, right_seg):
                                            sa = ee_pos + ee_rot @ np.asarray(
                                                seg[0], dtype=np.float64
                                            )
                                            sb = ee_pos + ee_rot @ np.asarray(
                                                seg[1], dtype=np.float64
                                            )
                                            ee_skelton._add_capsule_segment(
                                                viewer.user_scn,
                                                sa,
                                                sb,
                                                finger_radius,
                                                green,
                                            )
                                    except Exception:
                                        pass
                            # Marker on the drawer panel centre.
                            try:
                                if panel_geom_name is not None:
                                    c = _raw_panel_pose(
                                        raw_model, raw_data, panel_geom_name
                                    )
                                    if c is not None and c[0] is not None:
                                        viz_mj._add_scene_sphere(
                                            viewer.user_scn,
                                            np.asarray(c[0], dtype=np.float64),
                                            0.01,
                                            np.array(
                                                [0.9, 0.5, 0.1, 0.9], dtype=np.float32
                                            ),
                                        )
                            except Exception:
                                pass
                        else:
                            for pose_index, (target_pos, target_rot) in enumerate(
                                poses
                            ):
                                rgba = palette[min(pose_index, palette.shape[0] - 1)]
                                for ghost in ghost_geoms:
                                    viz_mj._add_ghost_geom(
                                        viewer.user_scn,
                                        ghost,
                                        target_pos,
                                        target_rot,
                                        rgba,
                                    )
                viewer.sync()
                time.sleep(1.0 / fps)
        finally:
            env.sim.data.qpos[:] = qpos_saved
            env.sim.data.qvel[:] = qvel_saved
            env.sim.forward()


def _visualize_floating_ee_poses_popup(env, refined_poses, args, drawer_q=None):
    """Pop a passive viewer rendering ghost Panda hand at every FloatingEEMPPI pose."""
    if not bool(getattr(args, "autogen_visualize_floating_ee", True)):
        return
    if not refined_poses:
        return
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    try:
        import mujoco
        import mujoco.viewer
        from robocasa.demos import visualize_mujoco as viz_mj
    except Exception as exc:
        print(f"Floating-EE viz skipped: {exc}")
        return

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    try:
        from robocasa.demos import ee_skelton as _ee_skelton
    except Exception:
        _ee_skelton = None
    skeleton = None
    hand_local_pos = None
    hand_local_rot = None
    hand_half_ext = None
    try:
        if drawer_q is not None:
            _close_impl._set_drawer_joint_value(env, float(drawer_q))
        env.sim.forward()
        ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(env, frame_name)
        if _ee_skelton is not None:
            try:
                skeleton = _ee_skelton.build_panda_skeleton(env, frame_name)
            except Exception as exc:
                print(f"[autogen] floating-EE skeleton build failed: {exc}", flush=True)
        # Pull panda_hand bbox center/local_rot in EE-site frame. Finger
        # capsules are segment endpoints, but mesh hand boxes need their bbox
        # center rather than geom_xpos because the mesh origin can be offset.
        try:
            import mujoco as _mj

            raw_model_ext, raw_data_ext = viz_mj._raw_model_data(env)
            _, _, body_id2name_ext, geom_id2name_ext = viz_mj._name_maps(env)
            site_pos_ext, site_rot_ext = viz_mj._site_pose(env, frame_name)
            # Only consider bodies belonging to the arm's EE subtree — this
            # excludes drawer_handle / cabinet_handle / etc. whose names contain
            # the substring "hand".
            ee_body_ids = viz_mj._ghost_source_body_ids(env)

            def _is_panda_hand(geom_name: str, body_name: str) -> bool:
                gl = geom_name.lower()
                bl = body_name.lower()
                if "finger" in gl or "finger" in bl:
                    return False
                for tok in ("panda_hand", "right_hand", "left_hand", "gripper_hand"):
                    if tok in gl or tok in bl:
                        return True
                # Fallback: body named exactly "hand" or ending in "_hand".
                return bl == "hand" or bl.endswith("_hand")

            def _geom_bbox_half_and_center_local(model, geom_id, fallback_size):
                if int(model.geom_type[geom_id]) == int(_mj.mjtGeom.mjGEOM_MESH):
                    try:
                        mesh_id = int(model.geom_dataid[geom_id])
                        vadr = int(model.mesh_vertadr[mesh_id])
                        vnum = int(model.mesh_vertnum[mesh_id])
                        verts = np.asarray(
                            model.mesh_vert[vadr : vadr + vnum], dtype=np.float64
                        )
                        if verts.size:
                            mesh_min = np.min(verts, axis=0)
                            mesh_max = np.max(verts, axis=0)
                            half = 0.5 * (mesh_max - mesh_min)
                            center = 0.5 * (mesh_max + mesh_min)
                            return (
                                np.maximum(
                                    half,
                                    np.array([0.003, 0.003, 0.003], dtype=np.float64),
                                ),
                                center,
                            )
                    except Exception:
                        pass
                half = np.maximum(
                    np.asarray(fallback_size, dtype=np.float64).reshape(3),
                    np.array([0.003, 0.003, 0.003], dtype=np.float64),
                )
                return half, np.zeros(3, dtype=np.float64)

            best_prod = -1.0
            for gid in range(int(raw_model_ext.ngeom)):
                body_id = int(raw_model_ext.geom_bodyid[gid])
                if body_id not in ee_body_ids:
                    continue
                geom_name = geom_id2name_ext.get(gid, "")
                body_name = body_id2name_ext.get(body_id, "")
                if not _is_panda_hand(geom_name, body_name):
                    continue
                geom_type = int(raw_model_ext.geom_type[gid])
                if geom_type == int(_mj.mjtGeom.mjGEOM_PLANE):
                    continue
                half, bbox_center_local = _geom_bbox_half_and_center_local(
                    raw_model_ext, gid, raw_model_ext.geom_size[gid]
                )
                prod = float(np.prod(np.maximum(half, 1e-6)))
                if prod <= best_prod:
                    continue
                gpos_w = np.asarray(
                    raw_data_ext.geom_xpos[gid], dtype=np.float64
                ).reshape(3)
                grot_w = np.asarray(
                    raw_data_ext.geom_xmat[gid], dtype=np.float64
                ).reshape(3, 3)
                bbox_center_w = gpos_w + grot_w @ bbox_center_local
                hand_local_pos = site_rot_ext.T @ (bbox_center_w - site_pos_ext)
                hand_local_rot = site_rot_ext.T @ grot_w
                hand_half_ext = half
                best_prod = prod
            print(
                f"[autogen] panda_hand geom lookup: found={hand_local_pos is not None}",
                flush=True,
            )
            ghost_hand_box = _hand_box_from_ghost_geoms(env, frame_name)
            if ghost_hand_box is not None:
                hand_local_pos, hand_local_rot, hand_half_ext = ghost_hand_box
                print(
                    "[autogen] panda_hand ghost override: "
                    f"local_pos={hand_local_pos} half={hand_half_ext}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[autogen] panda_hand geom lookup failed: {exc}", flush=True)
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()

    limit = max(int(getattr(args, "autogen_visualize_floating_ee_limit", 12)), 1)
    poses = refined_poses[:limit]
    lookat = np.mean(
        np.asarray([np.asarray(p[0], dtype=np.float64) for p in poses]), axis=0
    )
    palette = np.asarray(
        [
            [
                0.95,
                0.30,
                0.30,
                float(getattr(args, "autogen_floating_ee_ghost_alpha", 0.32)),
            ],
            [
                0.30,
                0.70,
                0.95,
                float(getattr(args, "autogen_floating_ee_ghost_alpha", 0.32)),
            ],
            [
                0.95,
                0.85,
                0.20,
                float(getattr(args, "autogen_floating_ee_ghost_alpha", 0.32)),
            ],
            [
                0.40,
                0.85,
                0.40,
                float(getattr(args, "autogen_floating_ee_ghost_alpha", 0.32)),
            ],
        ],
        dtype=np.float32,
    )
    with mujoco.viewer.launch_passive(
        raw_model,
        raw_data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        try:
            viewer.opt.geomgroup[:] = 0
            viewer.opt.geomgroup[1] = 1
        except Exception:
            pass
        viewer.cam.type = 0
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = float(
            getattr(args, "autogen_mink_popup_camera_distance", 0.85)
        )
        viewer.cam.azimuth = float(
            getattr(args, "autogen_mink_popup_camera_azimuth", 135.0)
        )
        viewer.cam.elevation = float(
            getattr(args, "autogen_mink_popup_camera_elevation", -25.0)
        )
        fps = max(float(getattr(args, "autogen_mink_popup_fps", 30.0)), 1.0)
        finger_radius = float(getattr(args, "autogen_skeleton_finger_radius", 0.004))
        gripper_opening_default = float(
            getattr(
                args,
                "autogen_skeleton_gripper_default",
                getattr(_ee_skelton, "PANDA_DEFAULT_GRIPPER_OPENING", 0.04)
                if _ee_skelton is not None
                else 0.04,
            )
        )
        while viewer.is_running():
            if hasattr(viewer, "user_scn"):
                viewer.user_scn.ngeom = 0
                for pose_index, (pos, rot, _g, _cid) in enumerate(poses):
                    rgba = palette[pose_index % palette.shape[0]]
                    for ghost in ghost_geoms:
                        viz_mj._add_ghost_geom(
                            viewer.user_scn,
                            ghost,
                            np.asarray(pos, dtype=np.float64),
                            np.asarray(rot, dtype=np.float64).reshape(3, 3),
                            rgba,
                        )
                    if skeleton is not None and _ee_skelton is not None:
                        try:
                            ee_pos = np.asarray(pos, dtype=np.float64).reshape(3)
                            ee_rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
                            if (
                                hand_local_pos is not None
                                and hand_local_rot is not None
                                and hand_half_ext is not None
                            ):
                                # Use the mesh bbox center, not geom_xpos. The
                                # latter is the geom origin; for mesh geoms it
                                # can be offset from the visual box center.
                                hand_center_w = ee_pos + ee_rot @ hand_local_pos
                                hand_rot_w = ee_rot @ hand_local_rot
                                half_ext_full = np.asarray(
                                    hand_half_ext, dtype=np.float64
                                ).reshape(3)
                            else:
                                # Fallback: skeleton's bbox-center parameterization.
                                half_ext_full = (
                                    np.asarray(
                                        skeleton.hand_box_half_extents_ee,
                                        dtype=np.float64,
                                    )
                                    .reshape(3)
                                    .copy()
                                )
                                hand_center_w = ee_pos + ee_rot @ np.asarray(
                                    skeleton.hand_box_center_ee, dtype=np.float64
                                )
                                hand_rot_w = ee_rot @ np.asarray(
                                    skeleton.hand_box_rotation_ee, dtype=np.float64
                                )
                            half_ext_full = (
                                np.asarray(half_ext_full, dtype=np.float64)
                                .reshape(3)
                                .copy()
                            )
                            half_ext_full[int(np.argmin(half_ext_full))] = 0.002
                            rgba_solid = np.asarray(rgba, dtype=np.float32).copy()
                            rgba_solid[3] = max(float(rgba_solid[3]), 0.6)
                            _ee_skelton._add_box(
                                viewer.user_scn,
                                hand_center_w,
                                hand_rot_w,
                                half_ext_full,
                                rgba_solid,
                            )
                            g_open = (
                                float(_g) if _g is not None else gripper_opening_default
                            )
                            (
                                left_seg,
                                right_seg,
                                _,
                                _,
                            ) = _ee_skelton._finger_segments_with_opening(
                                skeleton, g_open
                            )
                            for seg in (left_seg, right_seg):
                                sa = ee_pos + ee_rot @ seg[0]
                                sb = ee_pos + ee_rot @ seg[1]
                                _ee_skelton._add_capsule_segment(
                                    viewer.user_scn, sa, sb, finger_radius, rgba_solid
                                )
                        except Exception as exc:
                            print(
                                f"[autogen] floating-EE hand-box overlay skipped: {exc}",
                                flush=True,
                            )
            viewer.sync()
            time.sleep(1.0 / fps)


def _arm_pd_control_maps(env, arm_joint_names):
    model = env.sim.model
    joint_ids = [int(model.joint_name2id(name)) for name in arm_joint_names]
    qpos_addrs = np.asarray(
        [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids], dtype=np.int64
    )
    dof_addrs = np.asarray(
        [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids], dtype=np.int64
    )
    actuator_ids = []
    for joint_id, joint_name in zip(joint_ids, arm_joint_names):
        matches = [
            actuator_id
            for actuator_id in range(int(model.nu))
            if int(model.actuator_trnid[actuator_id, 0]) == joint_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one actuator for arm joint {joint_name!r}; found {len(matches)}"
            )
        actuator_ids.append(matches[0])
    return qpos_addrs, dof_addrs, np.asarray(actuator_ids, dtype=np.int64)


def _step_arm_pd(env, q_target, qpos_addrs, dof_addrs, actuator_ids, kp, kd):
    data = env.sim.data
    model = env.sim.model
    q_target = np.asarray(q_target, dtype=np.float64).reshape(7)
    q = np.asarray(data.qpos[qpos_addrs], dtype=np.float64)
    qd = np.asarray(data.qvel[dof_addrs], dtype=np.float64)
    bias = np.asarray(data.qfrc_bias[dof_addrs], dtype=np.float64)
    torque = (
        np.asarray(kp, dtype=np.float64) * (q_target - q)
        - np.asarray(kd, dtype=np.float64) * qd
        + bias
    )
    ctrlrange = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64)
    torque = np.clip(torque, ctrlrange[:, 0], ctrlrange[:, 1])
    data.ctrl[actuator_ids] = torque
    env.sim.step()


def _visualize_contact_marker_with_physical_pd(
    env,
    panel,
    selected,
    push_distance,
    robot_state,
    q_traj,
    mink_solution,
    all_mink_solutions,
    args,
    fallback_visualizer,
):
    if not bool(getattr(args, "autogen_physical_pd_visualization", True)):
        return fallback_visualizer(
            env,
            panel,
            selected,
            push_distance,
            robot_state,
            q_traj,
            mink_solution,
            all_mink_solutions,
            args,
        )
    if q_traj is None or np.asarray(q_traj).size == 0:
        return fallback_visualizer(
            env,
            panel,
            selected,
            push_distance,
            robot_state,
            q_traj,
            mink_solution,
            all_mink_solutions,
            args,
        )
    if env.viewer is None:
        print("Contact visualization skipped: renderer was not initialized.")
        return

    arm_trajectory = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    qpos_addrs, dof_addrs, actuator_ids = _arm_pd_control_maps(env, arm_joint_names)
    kp = np.full(7, float(getattr(args, "autogen_pd_kp", 350.0)), dtype=np.float64)
    kd = np.full(7, float(getattr(args, "autogen_pd_kd", 35.0)), dtype=np.float64)
    timestep = float(env.sim.model.opt.timestep)
    target_dt = float(
        getattr(
            args,
            "autogen_pd_target_dt",
            getattr(args, "curobo_interpolation_dt", 0.02),
        )
    )
    sim_steps_per_target = max(int(round(max(target_dt, timestep) / timestep)), 1)
    fps = max(float(getattr(args, "contact_marker_fps", 30.0)), 1.0)
    sync_interval = max(int(round((1.0 / fps) / timestep)), 1)
    marker_pos = selected.world_point + panel.outward_world * args.contact_marker_offset
    marker_rgba = np.array([1.0, 0.0, 0.0, args.contact_marker_alpha], dtype=np.float32)
    ee_marker_rgba = np.array([0.0, 1.0, 0.15, 1.0], dtype=np.float32)

    print(
        "Showing cuRobo trajectory with physical PD torque control "
        f"(frames={arm_trajectory.shape[0]}, sim_steps_per_target={sim_steps_per_target}, "
        f"kp={float(kp[0]):.1f}, kd={float(kd[0]):.1f}). "
        "Close the viewer window or press Ctrl+C to exit.",
        flush=True,
    )
    try:
        env.viewer.update()
    except Exception as exc:
        print(f"Contact visualization skipped: viewer launch failed: {exc}")
        return
    viewer = getattr(env.viewer, "viewer", None)
    if viewer is None:
        print("Contact visualization skipped: viewer launch failed.")
        return

    if hasattr(viewer, "cam"):
        viewer.cam.type = 0
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = marker_pos
        viewer.cam.distance = args.contact_camera_distance
        viewer.cam.azimuth = args.contact_camera_azimuth
        viewer.cam.elevation = args.contact_camera_elevation

    started_at = time.time()
    sim_step_index = 0
    try:
        for q_target in arm_trajectory:
            for _ in range(sim_steps_per_target):
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    return
                _step_arm_pd(env, q_target, qpos_addrs, dof_addrs, actuator_ids, kp, kd)
                sim_step_index += 1
                if hasattr(viewer, "user_scn") and sim_step_index % sync_interval == 0:
                    viewer.user_scn.ngeom = 0
                    _close_impl._draw_viewer_sphere(
                        viewer,
                        marker_pos,
                        args.contact_marker_size,
                        marker_rgba,
                    )
                    if mink_solution is not None:
                        ee_marker_pos = _close_impl._ee_sample_world(
                            env,
                            args.mink_contact_frame,
                            mink_solution.contact_offset_local,
                        )
                        _close_impl._draw_viewer_sphere(
                            viewer,
                            ee_marker_pos,
                            args.contact_marker_size * 0.75,
                            ee_marker_rgba,
                        )
                if hasattr(viewer, "sync") and sim_step_index % sync_interval == 0:
                    viewer.sync()
                if (
                    args.visualize_contact_seconds > 0
                    and time.time() - started_at >= args.visualize_contact_seconds
                ):
                    return
        hold_seconds = max(float(getattr(args, "autogen_pd_hold_seconds", 1.0)), 0.0)
        hold_steps = int(round(hold_seconds / timestep))
        q_target = arm_trajectory[-1]
        for _ in range(hold_steps):
            if hasattr(viewer, "is_running") and not viewer.is_running():
                return
            _step_arm_pd(env, q_target, qpos_addrs, dof_addrs, actuator_ids, kp, kd)
            sim_step_index += 1
            if hasattr(viewer, "sync") and sim_step_index % sync_interval == 0:
                viewer.sync()
        while (
            bool(getattr(args, "visualize_loop_trajectory", False))
            and hasattr(viewer, "is_running")
            and viewer.is_running()
        ):
            if hasattr(viewer, "sync"):
                viewer.sync()
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Physical PD contact visualization stopped: {exc}")


def _parse_close_autogen_args(original_parse_args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--autogen-visualize-mink-poses",
        dest="autogen_visualize_mink_poses",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-mink-poses",
        dest="autogen_visualize_mink_poses",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_mink_poses=False)
    parser.add_argument("--autogen-mink-ghost-alpha", type=float, default=0.28)
    parser.add_argument(
        "--autogen-mink-popup-camera-distance", type=float, default=0.85
    )
    parser.add_argument(
        "--autogen-mink-popup-camera-azimuth", type=float, default=135.0
    )
    parser.add_argument(
        "--autogen-mink-popup-camera-elevation", type=float, default=-25.0
    )
    parser.add_argument("--autogen-mink-popup-fps", type=float, default=30.0)
    parser.add_argument(
        "--autogen-visualize-skeleton",
        dest="autogen_visualize_skeleton",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-skeleton",
        dest="autogen_visualize_skeleton",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_skeleton=True)
    parser.add_argument(
        "--autogen-visualize-skeleton-poses",
        dest="autogen_visualize_skeleton_poses",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-skeleton-poses",
        dest="autogen_visualize_skeleton_poses",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_skeleton_poses=True)
    parser.add_argument(
        "--autogen-visualize-execution",
        dest="autogen_visualize_execution",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-execution",
        dest="autogen_visualize_execution",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_execution=True)
    parser.add_argument(
        "--autogen-physical-pd-visualization",
        dest="autogen_physical_pd_visualization",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-physical-pd-visualization",
        dest="autogen_physical_pd_visualization",
        action="store_false",
    )
    parser.set_defaults(autogen_physical_pd_visualization=True)
    parser.add_argument("--autogen-pd-kp", type=float, default=350.0)
    parser.add_argument("--autogen-pd-kd", type=float, default=35.0)
    parser.add_argument("--autogen-pd-target-dt", type=float, default=0.02)
    parser.add_argument("--autogen-pd-hold-seconds", type=float, default=1.0)
    parser.add_argument(
        "--autogen-visualize-floating-ee",
        dest="autogen_visualize_floating_ee",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-floating-ee",
        dest="autogen_visualize_floating_ee",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_floating_ee=True)
    parser.add_argument("--autogen-visualize-floating-ee-limit", type=int, default=12)
    parser.add_argument("--autogen-floating-ee-ghost-alpha", type=float, default=0.32)
    parser.add_argument("--autogen-mink-parallel-workers", type=int, default=None)
    parser.add_argument("--autogen-mink-q-checker-workers", type=int, default=None)
    parser.add_argument("--autogen-mink-q-worlds-per-worker", type=int, default=None)
    parser.add_argument(
        "--ts-skeleton",
        dest="ts_skeleton",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ts-viz-frame-seconds", type=float, default=None)
    parser.add_argument("--ts-viz-render-fps", type=float, default=None)
    parser.add_argument("--ts-viz-skeleton-pose-limit", type=int, default=None)
    parser.add_argument(
        "--solve_step2",
        "--solve-step2",
        dest="solve_step2",
        type=str,
        default=None,
        choices=("MPPI", "mink", "mppi", "MINK"),
    )
    popup_args, remaining = parser.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = original_parse_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(popup_args).items():
        if value is None:
            continue
        setattr(args, key, value)
    if not any(str(arg).startswith("--mink-solver") for arg in old_argv[1:]):
        args.mink_solver = "daqp"
    user_set_mink_workers = any(
        str(arg) == "--autogen-mink-parallel-workers"
        or str(arg).startswith("--autogen-mink-parallel-workers=")
        for arg in old_argv[1:]
    )
    # Show the skeleton-pose popup only after the DAQP batch finishes.  The
    # raw skeleton/EE geometry popup is intentionally disabled by default here
    # because it appears before DAQP and makes the progress-bar timing unclear.
    # Force the QP path with multi-theta sampling so contact rotations are not
    # collapsed to the demo EE orientation via the R0_override bypass in
    # ee_skelton.solve_skeleton_pose.
    # If the free-rotation QP yields zero poses (current symptom), let
    # _solve_contact_poses_with_skeleton fall back to R0_override automatically
    # rather than silently dropping into the geometric build_target_gripper_poses
    # path — see the skeleton_rotation_fallback=current_ee_rotation log line.
    args.autogen_visualize_skeleton = False
    args.autogen_visualize_skeleton_poses = True
    args.autogen_use_current_ee_rotation = False
    args.autogen_allow_current_rotation_fallback = True
    # Keep the drawer/handle convex penetration check enabled by default. The
    # contact tangent plane alone is not a volume collision constraint, and it
    # can accept skeleton poses whose finger capsules or hand box visibly pass
    # through the drawer.
    if not hasattr(args, "autogen_skeleton_disable_handle_convex"):
        args.autogen_skeleton_disable_handle_convex = False
    # Suppress the per-skeleton-pose step-2 solver status prints (yellow flood)
    # and show a tqdm progress bar instead; after solving, a single yellow line
    # reports total duration + feasible q count. Direct demo invocations keep the
    # detailed per-pose logs because the default (True) is unchanged.
    args.autogen_step2_verbose_poses = False
    args.autogen_skeleton_daqp_verbose = False
    if "--autogen-mink-q-debug" not in old_argv:
        args.autogen_mink_q_debug = False
    args.autogen_skeleton_object_penetration_tol = min(
        float(getattr(args, "autogen_skeleton_object_penetration_tol", 0.001)),
        0.001,
    )
    args.autogen_skeleton_clearance_tolerance = min(
        float(getattr(args, "autogen_skeleton_clearance_tolerance", 0.001)),
        0.001,
    )
    args.autogen_qmppi_penetration_threshold = min(
        float(getattr(args, "autogen_qmppi_penetration_threshold", 0.005)),
        float(getattr(args, "contact_standoff", 0.005)),
        float(getattr(args, "mink_collision_penetration_tolerance", 0.02)),
        0.001,
    )
    args.mink_collision_penetration_tolerance = min(
        float(getattr(args, "mink_collision_penetration_tolerance", 0.02)),
        0.001,
    )
    args.autogen_qmppi_accept_object_improvement_only = True
    # (Fix #1) pen_weight was left at 0.0 so MPPI's cost function was
    # `costs = 0.0 * penetration^2 + contact_weight * d^2 + tracking`. The
    # solver therefore only minimized tracking error and could never push an
    # EE pose out of penetration — every sampled pose was reported as
    # penetrating the drawer. A weight comparable to track_pos_weight (200)
    # scaled by the ratio of typical penetration depth (~0.01 m) to tracking
    # displacement makes the solver resolve penetration actively.
    args.autogen_qmppi_penetration_weight = 500.0
    args.autogen_qmppi_contact_weight = 0.0
    if str(getattr(args, "solve_step2", "MPPI")).strip().lower() == "mppi":
        args.autogen_skip_mink_q_after_mppi = True
    default_mink_workers = max(1, (os.cpu_count() or 1))
    if not user_set_mink_workers:
        args.autogen_mink_parallel_workers = default_mink_workers
    if (
        str(getattr(args, "solve_step2", "MPPI")).strip().lower() == "mink"
        and not bool(getattr(args, "autogen_mink_q_debug", False))
        and not user_set_mink_workers
    ):
        args.autogen_mink_parallel_workers = max(2, default_mink_workers)
    args.curobo_use_mujoco_world = True
    args.curobo_world_exclude_target_drawer = True
    # (Fix #2) Drive the world-padding for non-box obstacle geoms to zero so
    # the planner does not waste free space around the COACD-decomposed fixtures
    # (handle connectors / bolts that protrude from the panel). MuJoCo geoms are
    # already precise; AABB padding here was redundant and inflated them into
    # free air.
    args.curobo_world_padding = min(
        float(getattr(args, "curobo_world_padding", 0.005)), 0.0
    )
    if bool(getattr(args, "autogen_visualize_execution", True)):
        args.visualize_contact = True
    args.save_output = False
    args.save_trajectory_videos = False
    if float(getattr(args, "contact_cost_threshold", 0.10)) == 0.10:
        args.contact_cost_threshold = 0.35
    if not hasattr(args, "autogen_panel_edge_margin"):
        args.autogen_panel_edge_margin = 0.015
    if float(getattr(args, "autogen_panel_edge_margin", 0.015)) in (0.015, 0.03):
        args.autogen_panel_edge_margin = 0.05
    if not hasattr(args, "autogen_panel_edge_fraction"):
        args.autogen_panel_edge_fraction = 0.28
    if float(getattr(args, "autogen_panel_edge_fraction", 0.18)) == 0.18:
        args.autogen_panel_edge_fraction = 0.28
    if not hasattr(args, "autogen_panel_top_edge_fraction"):
        args.autogen_panel_top_edge_fraction = 0.38
    if not hasattr(args, "autogen_skeleton_pose_variants_per_contact"):
        args.autogen_skeleton_pose_variants_per_contact = 8
    elif int(getattr(args, "autogen_skeleton_pose_variants_per_contact", 4)) == 4:
        args.autogen_skeleton_pose_variants_per_contact = 8
    if not hasattr(args, "autogen_skeleton_pose_min_theta_separation"):
        args.autogen_skeleton_pose_min_theta_separation = float(np.pi / 6.0)
    if not hasattr(args, "autogen_visualize_skeleton_pose_limit"):
        args.autogen_visualize_skeleton_pose_limit = 120
    return args


def main():
    original_parse_args = _close_impl.parse_args
    original_solve_contact_poses = _close_impl.solve_contact_poses_with_mink
    original_solve_mink_for_drawer_candidate = (
        _close_impl._solve_mink_for_drawer_candidate
    )
    original_solve_skeleton_precontact_q = (
        _close_impl.mink_q.solve_skeleton_precontact_q
    )
    original_plan_with_curobo = _close_impl.plan_with_curobo
    original_visualize_contact_marker = _close_impl.visualize_contact_marker
    original_skeleton_solver = getattr(
        _close_impl, "_solve_contact_poses_with_skeleton", None
    )
    visualized = {"done": False}

    def parse_args_with_popup():
        return _parse_close_autogen_args(original_parse_args)

    def solve_contact_poses_with_popup(
        env, panel, candidates, push_distance, robot_state, args
    ):
        result = original_solve_contact_poses(
            env,
            panel,
            candidates,
            push_distance,
            robot_state,
            args,
        )
        mink_solution = result[0]
        if (
            mink_solution is not None
            and not visualized["done"]
            and bool(getattr(args, "autogen_visualize_mink_poses", False))
        ):
            q_waypoints = np.asarray(
                getattr(mink_solution, "q_waypoints", np.zeros((0, 7))),
                dtype=np.float64,
            ).reshape(-1, 7)
            drawer_qpos_addr = None
            try:
                drawer_qpos_addr = int(
                    env.sim.model.get_joint_qpos_addr(env.drawer.door_joint_names[0])
                )
            except Exception:
                drawer_qpos_addr = None
            drawer_q_seq = None
            ref_panel_pose = None
            ts_poses_ref = getattr(args, "_autogen_ts_skeleton_poses", None)
            if q_waypoints.size > 0 and drawer_qpos_addr is not None:
                # Close-drawer demo pushes the drawer IN (positive dq ->
                # toward 0).  Interpolate across the arm waypoints so the
                # first popup shows the drawer actually sliding instead of
                # freezing at its start pose.
                q_start = float(_close_impl._drawer_joint_value(env))
                n_frames = q_waypoints.shape[0]
                dj_id = int(env.sim.model.joint_name2id(env.drawer.door_joint_names[0]))
                jnt_range = np.asarray(env.sim.model.jnt_range[dj_id], dtype=np.float64)
                q_target = min(q_start + float(push_distance), float(jnt_range[1]))
                drawer_q_seq = np.linspace(
                    q_start, q_target, n_frames, dtype=np.float64
                )
                try:
                    ref_panel_pose = _drawer_panel_pose(env)
                except Exception:
                    ref_panel_pose = None
            _visualize_mink_q_poses_popup(
                env,
                q_waypoints,
                robot_state,
                args,
                drawer_q=float(_close_impl._drawer_joint_value(env))
                if drawer_q_seq is None
                else float(drawer_q_seq[0]),
                drawer_q_seq=drawer_q_seq,
                ts_poses_ref=ts_poses_ref,
                panel_pose_ref=ref_panel_pose,
                drawer_qpos_addr=drawer_qpos_addr,
            )
            visualized["done"] = True
        return result

    def skeleton_solver_with_marker(
        env, panel, candidates, push_distance, robot_state, args
    ):
        _autogen_print(
            "close_autogen_patch=skeleton "
            f"edge_margin={float(getattr(args, 'autogen_panel_edge_margin', 0.015)):.4f} "
            f"edge_fraction={float(getattr(args, 'autogen_panel_edge_fraction', 0.18)):.4f} "
            f"top_edge_fraction={float(getattr(args, 'autogen_panel_top_edge_fraction', 0.38)):.4f} "
            f"variants_per_contact={int(getattr(args, 'autogen_skeleton_pose_variants_per_contact', 4))} "
            f"theta_sep={float(getattr(args, 'autogen_skeleton_pose_min_theta_separation', np.pi / 6.0)):.4f}"
        )
        # Reset the shared refined-poses buffer so the popup reflects THIS run's
        # MPPI output only (the underlying module clears it on entry too, but the
        # autogen wrapper runs as a different call stack, so be explicit here).
        LAST_FLOATING_REFINED_POSES = getattr(
            _close_impl, "LAST_FLOATING_REFINED_POSES", None
        )
        if LAST_FLOATING_REFINED_POSES is not None and hasattr(
            LAST_FLOATING_REFINED_POSES, "clear"
        ):
            LAST_FLOATING_REFINED_POSES.clear()
        # Lazily build the per-worker cloned-MjData pool used by the DAQP inner
        # loop. Sharing env.sim.data across worker threads is only nominally
        # safe (mj_ray is read-only but the underlying MjData is not); the pool
        # gives each worker its own MjData 1:1 cloned from env, and re-syncs on
        # every DAQP batch.
        if getattr(args, "autogen_skeleton_scene_pool", None) is None:
            skeleton_workers = getattr(args, "autogen_skeleton_parallel_workers", None)
            if skeleton_workers is None:
                skeleton_workers = getattr(args, "autogen_mink_parallel_workers", 1)
            workers = max(1, int(skeleton_workers or 1))
            try:
                from robocasa.demos.skelton_scene import SkeletonScenePool

                args.autogen_skeleton_scene_pool = SkeletonScenePool.from_env(
                    env, num_workers=workers
                )
            except Exception as exc:
                print(f"[autogen] skeleton scene pool init failed: {exc}", flush=True)
                args.autogen_skeleton_scene_pool = None
        mppi_started = time.perf_counter()
        result = original_skeleton_solver(
            env,
            panel,
            candidates,
            push_distance,
            robot_state,
            args,
        )
        mppi_elapsed = time.perf_counter() - mppi_started
        solution = result[0]
        if solution is not None:
            _autogen_print(
                "close_autogen_patch=selected_skeleton "
                f"candidate={int(solution.drawer_candidate_index)} "
                f"theta={float(getattr(solution, 'roll_angle', 0.0)):.4f} "
                f"cost={float(solution.drawer_contact_cost):.6f}"
            )
        refined = list(getattr(_close_impl, "LAST_FLOATING_REFINED_POSES", []) or [])
        step2_label = str(getattr(args, "solve_step2", "MPPI")).strip().lower()
        print(
            "\033[93m"
            f"close_autogen_patch=step2_{step2_label} "
            f"refined_pose_count={len(refined)} "
            f"solve_s={mppi_elapsed:.3f}"
            "\033[0m",
            flush=True,
        )
        if refined and bool(getattr(args, "autogen_visualize_floating_ee", True)):
            _visualize_floating_ee_poses_popup(
                env,
                refined,
                args,
                drawer_q=float(_close_impl._drawer_joint_value(env)),
            )
        return result

    def plan_with_curobo_with_stats(
        robot_state, target_hand_poses_base, args, *extra_args, **extra_kwargs
    ):
        started = time.perf_counter()
        try:
            q_traj, segments = original_plan_with_curobo(
                robot_state,
                target_hand_poses_base,
                args,
                *extra_args,
                **extra_kwargs,
            )
        except Exception as exc:
            # Autogen's useful output is the parallel mink/contact solve.  Treat
            # cuRobo graph/trajopt failures as a non-fatal planning miss so the
            # run can still inspect and visualize the known mink q solution.
            print(
                "\033[93m"
                "curobo_status=failed_nonfatal "
                f"curobo_time={time.perf_counter() - started:.6f} "
                f"error={type(exc).__name__}: {exc}"
                "\033[0m",
                flush=True,
            )
            return None, [
                {
                    "name": "curobo_failed_nonfatal",
                    "planner": "curobo",
                    "status": "failed_nonfatal",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ]
        world_obstacle_count = 0
        world_model = "none"
        world_exclude_bodies = ()
        if segments:
            world_obstacle_count = int(segments[0].get("world_obstacle_count", 0))
            world_model = str(segments[0].get("world_collision_model", "none"))
            world_exclude_bodies = tuple(segments[0].get("world_exclude_bodies", ()))
        print(
            "\033[93m"
            "curobo_time="
            f"{time.perf_counter() - started:.6f} "
            f"successful_trajectories={int(len(segments))} "
            f"world_collision_model={world_model} "
            f"world_obstacle_count={world_obstacle_count} "
            f"world_exclude_bodies={world_exclude_bodies}"
            "\033[0m",
            flush=True,
        )
        return q_traj, segments

    try:
        from robocasa.demos import ee_skelton as _ee_skelton
    except Exception:
        _ee_skelton = None
    original_draw_skeleton = (
        getattr(_ee_skelton, "_draw_skeleton_into_scene", None)
        if _ee_skelton is not None
        else None
    )
    original_build_skeleton = (
        getattr(_ee_skelton, "build_panda_skeleton", None)
        if _ee_skelton is not None
        else None
    )

    def build_panda_skeleton_with_ghost_hand(env, ee_site_name):
        skeleton = original_build_skeleton(env, ee_site_name)
        try:
            ghost_hand_box = _hand_box_from_ghost_geoms(
                env, str(ee_site_name).split(":")[0]
            )
            if ghost_hand_box is None:
                return skeleton
            hand_local_pos, hand_local_rot, hand_half_ext = ghost_hand_box
            return _ee_skelton.EESkeleton(
                hand_box_half_extents_ee=np.asarray(hand_half_ext, dtype=np.float64)
                .reshape(3)
                .copy(),
                hand_box_center_ee=np.asarray(hand_local_pos, dtype=np.float64)
                .reshape(3)
                .copy(),
                hand_box_rotation_ee=np.asarray(hand_local_rot, dtype=np.float64)
                .reshape(3, 3)
                .copy(),
                left_finger_segment_ee=np.asarray(
                    skeleton.left_finger_segment_ee, dtype=np.float64
                ).copy(),
                right_finger_segment_ee=np.asarray(
                    skeleton.right_finger_segment_ee, dtype=np.float64
                ).copy(),
                finger_tip_offset_ee=np.asarray(
                    skeleton.finger_tip_offset_ee, dtype=np.float64
                ).copy(),
            )
        except Exception as exc:
            print(f"[autogen] ghost hand skeleton override skipped: {exc}", flush=True)
            return skeleton

    def draw_skeleton_with_flat_hand_box(
        scene,
        skeleton,
        ee_pos,
        ee_rot,
        gripper_opening,
        rgba_hand,
        rgba_finger,
        finger_radius,
    ):
        # Original draw without ee_skelton.py's verbose [skeleton_draw]
        # center_w diagnostic.
        try:
            half_ext_flat = _ee_skelton._flat_hand_half_extents(skeleton)
            flat_center_w = np.asarray(ee_pos, dtype=np.float64) + np.asarray(
                ee_rot, dtype=np.float64
            ) @ np.asarray(skeleton.hand_box_center_ee, dtype=np.float64)
            flat_rot_w = np.asarray(ee_rot, dtype=np.float64) @ np.asarray(
                skeleton.hand_box_rotation_ee, dtype=np.float64
            )
            _ee_skelton._add_box(
                scene,
                flat_center_w,
                flat_rot_w,
                half_ext_flat,
                rgba_hand,
            )
            left_seg, right_seg, _, _ = _ee_skelton._finger_segments_with_opening(
                skeleton, gripper_opening
            )
            for seg in (left_seg, right_seg):
                sa = np.asarray(ee_pos, dtype=np.float64) + np.asarray(
                    ee_rot, dtype=np.float64
                ) @ np.asarray(seg[0], dtype=np.float64)
                sb = np.asarray(ee_pos, dtype=np.float64) + np.asarray(
                    ee_rot, dtype=np.float64
                ) @ np.asarray(seg[1], dtype=np.float64)
                _ee_skelton._add_capsule_segment(
                    scene, sa, sb, finger_radius, rgba_finger
                )
        except Exception as exc:
            print(f"[autogen] skeleton draw skipped: {exc}", flush=True)

    if original_draw_skeleton is not None:
        _ee_skelton._draw_skeleton_into_scene = draw_skeleton_with_flat_hand_box
    if original_build_skeleton is not None:
        _ee_skelton.build_panda_skeleton = build_panda_skeleton_with_ghost_hand

    _close_impl.parse_args = parse_args_with_popup
    _close_impl._solve_mink_for_drawer_candidate = functools.partial(
        _close_impl.mink_q.solve_mink_for_drawer_candidate_parallel,
        close_ops=_close_impl,
        original_solver=original_solve_mink_for_drawer_candidate,
        status_printer=_autogen_print,
    )
    _close_impl.mink_q.solve_skeleton_precontact_q = (
        _close_impl.mink_q.solve_skeleton_precontact_q_parallel
    )
    _close_impl.solve_contact_poses_with_mink = solve_contact_poses_with_popup
    _close_impl.plan_with_curobo = plan_with_curobo_with_stats
    _close_impl.visualize_contact_marker = (
        lambda *call_args, **call_kwargs: _visualize_contact_marker_with_physical_pd(
            *call_args,
            **call_kwargs,
            fallback_visualizer=original_visualize_contact_marker,
        )
    )
    if original_skeleton_solver is not None:
        _close_impl._solve_contact_poses_with_skeleton = skeleton_solver_with_marker
    try:
        _close_contact_main()
    finally:
        _close_impl.parse_args = original_parse_args
        _close_impl._solve_mink_for_drawer_candidate = (
            original_solve_mink_for_drawer_candidate
        )
        _close_impl.mink_q.solve_skeleton_precontact_q = (
            original_solve_skeleton_precontact_q
        )
        _close_impl.solve_contact_poses_with_mink = original_solve_contact_poses
        _close_impl.plan_with_curobo = original_plan_with_curobo
        _close_impl.visualize_contact_marker = original_visualize_contact_marker
        if original_skeleton_solver is not None:
            _close_impl._solve_contact_poses_with_skeleton = original_skeleton_solver
        if original_draw_skeleton is not None and _ee_skelton is not None:
            _ee_skelton._draw_skeleton_into_scene = original_draw_skeleton
        if original_build_skeleton is not None and _ee_skelton is not None:
            _ee_skelton.build_panda_skeleton = original_build_skeleton


# ---------------------------------------------------------------------------
# Time-series skeleton poses from demonstration + DAQP + penetration filter
# ---------------------------------------------------------------------------


def _extract_drawer_trajectory_from_demo(args):
    """Load the origin demonstration and extract drawer + robot arm trajectories.

    Returns ``(times, drawer_qs, robot_arm_qs, arm_joint_names, duration,
    episode_name, initial_state, states)``.

    The LeRobot dataset state vector layout (nq/nv) may differ from the
    current env, which breaks ``reset_to``.  When that happens we fall back
    to a synthesized ramp from the current drawer position to fully closed
    and hold ``robot_arm_qs`` at the source env's current arm qpos.
    """
    dataset, episode_name, initial_state, states = _load_origin_demo_episode(args)
    source_env = _create_origin_demo_env(dataset, has_offscreen_renderer=False)
    try:
        control_freq = float(getattr(args, "control_freq", 20.0))
        drawer_joint_name = source_env.drawer.door_joint_names[0]
        addr = source_env.sim.model.get_joint_qpos_addr(drawer_joint_name)
        arm_joint_names = tuple(source_env.robots[0].robot_model.joints[:7])
        arm_addrs = [
            source_env.sim.model.get_joint_qpos_addr(n) for n in arm_joint_names
        ]
        n_frames = states.shape[0]
        drawer_qs = np.zeros(n_frames, dtype=np.float64)
        robot_arm_qs = np.zeros((n_frames, len(arm_joint_names)), dtype=np.float64)

        # Attempt 1: restore each state via reset_to.
        try:
            for i, state in enumerate(states):
                reset_to(source_env, {"states": state})
                drawer_qs[i] = float(source_env.sim.data.qpos[addr])
                robot_arm_qs[i, :] = [
                    float(source_env.sim.data.qpos[a]) for a in arm_addrs
                ]
        except Exception as exc:
            print(
                f"[ts-skel] reset_to extraction failed ({exc}); "
                "falling back to synthesized trajectory.",
                flush=True,
            )
            drawer_qs[:] = 0.0
            robot_arm_qs[:] = 0.0

        # Attempt 2 (fallback): synthesise from the drawer's current range.
        if np.std(drawer_qs) < 1e-6:
            dj_id = int(source_env.sim.model.joint_name2id(drawer_joint_name))
            jnt_range = np.asarray(
                source_env.sim.model.jnt_range[dj_id], dtype=np.float64
            )
            q_current = float(
                source_env.sim.data.qpos[
                    source_env.sim.model.get_joint_qpos_addr(drawer_joint_name)
                ]
            )
            q_closed = float(jnt_range[0])  # most negative = fully closed
            t = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
            drawer_qs = q_current + (q_closed - q_current) * t
            cur_arm = np.asarray(
                [float(source_env.sim.data.qpos[a]) for a in arm_addrs],
                dtype=np.float64,
            )
            robot_arm_qs[:] = cur_arm[None, :]
            print(
                f"[ts-skel] synthesized drawer trajectory "
                f"{q_current:.4f} -> {q_closed:.4f} over {n_frames} frames "
                f"(robot arm held constant)",
                flush=True,
            )

        times = np.arange(n_frames, dtype=np.float64) / control_freq
        duration = float(times[-1]) if n_frames > 0 else 0.0
        return (
            times,
            np.asarray(drawer_qs, dtype=np.float64),
            robot_arm_qs,
            arm_joint_names,
            duration,
            episode_name,
            initial_state,
            states,
        )
    finally:
        source_env.close()


def _interpolate_drawer_poses(times, drawer_qs, robot_arm_qs=None, hz=10.0):
    """Interpolate drawer + optional arm trajectory at a fixed ``hz`` sampling rate."""
    if len(times) == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            (None if robot_arm_qs is None else np.zeros((0, 0), dtype=np.float64)),
        )
    t_end = float(times[-1])
    n_frames = max(int(round(t_end * hz)) + 1, 2)
    new_times = np.linspace(0.0, t_end, n_frames, dtype=np.float64)
    new_drawer_qs = np.interp(new_times, times, drawer_qs)
    new_arm_qs = None
    if robot_arm_qs is not None and robot_arm_qs.size > 0:
        arm = np.asarray(robot_arm_qs, dtype=np.float64)
        if arm.ndim == 1:
            arm = arm.reshape(-1, 1)
        new_arm_qs = np.stack(
            [np.interp(new_times, times, arm[:, j]) for j in range(arm.shape[1])],
            axis=1,
        )
    return new_times, new_drawer_qs, new_arm_qs


# Keep the old name as a thin backward-compatible alias so any external caller
# expecting the 20Hz-labelled entry point still works.
def _interpolate_drawer_poses_20hz(times, drawer_qs, hz=20.0):
    nt, nq, _ = _interpolate_drawer_poses(times, drawer_qs, None, hz=hz)
    return nt, nq


def _project_skeleton_poses_to_drawer_pose(
    skeleton_poses,
    panel_pose_at_ref,
    panel_pose_at_new,
):
    """Project skeleton poses solved at ``panel_pose_at_ref`` to the new
    panel frame by applying the rigid drawer displacement in world."""
    if not skeleton_poses:
        return []
    c_ref, R_ref = panel_pose_at_ref
    c_new, R_new = panel_pose_at_new
    dR = R_new @ R_ref.T
    dp = c_new - dR @ c_ref
    projected = []
    for sp in skeleton_poses:
        projected.append(
            ee_skelton.SkeletonPose(
                ee_position=dR @ np.asarray(sp.ee_position, dtype=np.float64) + dp,
                ee_rotation=dR @ np.asarray(sp.ee_rotation, dtype=np.float64),
                contact_finger=sp.contact_finger,
                contact_point_world=dR
                @ np.asarray(sp.contact_point_world, dtype=np.float64)
                + dp,
                contact_normal_world=dR
                @ np.asarray(sp.contact_normal_world, dtype=np.float64),
                qp_cost=sp.qp_cost,
                lift=sp.lift,
                theta=sp.theta,
                gripper_opening=sp.gripper_opening,
                contact_primitive=sp.contact_primitive,
            )
        )
    return projected


def _solve_reference_skeleton_poses(env, panel, args):
    """Solve skeleton Poses at a given drawer configuration using DAQP.
    Returns ``(skeleton, ref_poses, feasible_cache, scene_geom_ids,
    object_eqs, initial_rot)``."""
    from robocasa.demos import ee_skelton as _skel

    frame_name = str(args.mink_contact_frame).split(":")[0]
    if frame_name not in env.sim.model._site_name2id:
        raise RuntimeError(
            f"mink contact frame '{frame_name}' is not present in the simulation model."
        )

    feasible_cache = getattr(args, "_autogen_feasible_cache", None)
    if feasible_cache is None:
        _, _, feasible_cache = _build_autogen_contact_candidates(env, panel, 0.0, args)

    skeleton = _skel.build_panda_skeleton(env, frame_name)
    select_count = int(getattr(args, "autogen_initial_pose_count", 200))
    (
        local_ids,
        points_world,
        normals_world,
    ) = _select_close_panel_interior_feasible_points(
        feasible_cache,
        panel,
        select_count,
        int(args.seed),
        args,
    )
    scene_geom_ids = _scene_geom_ids_for_skeleton(env, panel)
    object_eqs = getattr(feasible_cache, "handle_convex_equations", None)
    initial_rot = _current_site_pose(env, frame_name)[1]

    variants_per_contact = max(
        1, int(getattr(args, "autogen_skeleton_pose_variants_per_contact", 4))
    )
    theta_sep = float(
        getattr(args, "autogen_skeleton_pose_min_theta_separation", np.pi / 6.0)
    )
    primitive_specs = (
        ("left_finger", "left"),
        ("right_finger", "right"),
        ("hand", "left"),
    )

    # Build jobs: one (local_id, point, normal, primitive, finger) per candidate x primitive.
    jobs = []
    for local_id, point, normal in zip(local_ids, points_world, normals_world):
        for primitive, finger in primitive_specs:
            jobs.append(
                (
                    int(local_id),
                    np.asarray(point, dtype=np.float64),
                    np.asarray(normal, dtype=np.float64),
                    primitive,
                    finger,
                )
            )

    # Same parallel-DAQP pattern as _solve_contact_poses_with_skeleton (~L2195):
    # SkeletonScenePool + ThreadPoolExecutor + per-worker MjData clone.
    workers_arg = (
        getattr(args, "autogen_skeleton_parallel_workers", None)
        or getattr(args, "autogen_mink_parallel_workers", None)
        or max(1, (os.cpu_count() or 4) - 2)
    )
    workers = max(1, int(workers_arg))
    active_workers = max(1, min(workers, len(jobs) or 1))

    scene_pool = getattr(args, "_autogen_ts_scene_pool", None)
    if scene_pool is None:
        try:
            from robocasa.demos.skelton_scene import SkeletonScenePool

            scene_pool = SkeletonScenePool.from_env(env, num_workers=active_workers)
            args._autogen_ts_scene_pool = scene_pool
        except Exception as exc:
            print(f"[ts-skel] SkeletonScenePool init failed: {exc}", flush=True)
            scene_pool = None
    else:
        try:
            scene_pool.reset()
        except Exception:
            pass

    print(
        f"[ts-skel] REFERENCE_SKELETON_POSE_SOLVER backend=daqp "
        f"workers={active_workers} jobs={len(jobs)} "
        f"variants_per_contact={variants_per_contact}",
        flush=True,
    )

    def _solve_ref_job(job):
        local_id, point, normal, primitive, finger = job
        try:
            if scene_pool is not None:
                with scene_pool.borrow() as rmd:
                    poses = _skel.solve_skeleton_pose_candidates(
                        env,
                        skeleton,
                        point,
                        normal,
                        finger=finger,
                        contact_primitive=primitive,
                        object_convex_equations=object_eqs,
                        object_convex_equation_mask=None,
                        scene_geom_ids=scene_geom_ids,
                        initial_ee_rotation_world=None,
                        args=args,
                        max_candidates=variants_per_contact,
                        min_theta_separation=theta_sep,
                        raw_model_data=rmd,
                    )
            else:
                raw_model = getattr(env.sim.model, "_model", env.sim.model)
                raw_data = getattr(env.sim.data, "_data", env.sim.data)
                poses = _skel.solve_skeleton_pose_candidates(
                    env,
                    skeleton,
                    point,
                    normal,
                    finger=finger,
                    contact_primitive=primitive,
                    object_convex_equations=object_eqs,
                    object_convex_equation_mask=None,
                    scene_geom_ids=scene_geom_ids,
                    initial_ee_rotation_world=None,
                    args=args,
                    max_candidates=variants_per_contact,
                    min_theta_separation=theta_sep,
                    raw_model_data=(raw_model, raw_data),
                )
            return int(local_id), list(poses), None
        except Exception as exc:
            return int(local_id), [], f"{exc.__class__.__name__}:{exc}"

    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        _tqdm = None
    pbar = (
        _tqdm(
            total=len(jobs),
            desc=f"ref-daqp (workers={active_workers})",
            unit="job",
            file=sys.__stdout__,
            dynamic_ncols=True,
            mininterval=0.2,
            leave=True,
        )
        if _tqdm is not None
        else None
    )

    all_poses = []
    failures = {}
    if active_workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            _lid, poses, err = _solve_ref_job(job)
            if err is not None:
                failures[err] = failures.get(err, 0) + 1
            else:
                all_poses.extend(poses)
            if pbar is not None:
                pbar.update(1)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=active_workers) as ex:
            futures = [ex.submit(_solve_ref_job, job) for job in jobs]
            for fut in concurrent.futures.as_completed(futures):
                _lid, poses, err = fut.result()
                if err is not None:
                    failures[err] = failures.get(err, 0) + 1
                else:
                    all_poses.extend(poses)
                if pbar is not None:
                    pbar.update(1)
    if pbar is not None:
        pbar.close()

    if failures:
        top = sorted(failures.items(), key=lambda kv: -kv[1])[:3]
        print(
            f"[ts-skel] reference DAQP: {sum(failures.values())} / {len(jobs)} jobs failed; "
            f"top errors: {top}",
            flush=True,
        )
    return skeleton, all_poses, feasible_cache, scene_geom_ids, object_eqs, initial_rot


def solve_timeseries_skeleton_poses(args):
    """End-to-end: demo drawer traj -> 10Hz poses -> DAQP skeleton solve ->
    project -> mj_multiRay penetration filter.  Returns ``(ts_poses, env)``.

    ``ts_poses`` entries: ``(drawer_q, panel_cur, [(pose_id, pose), ...],
    frame_idx)``.  ``pose_id`` is assigned once at the reference frame and
    kept across projection/filtering so downstream code can tell which
    reference pose was dropped where."""
    from robocasa.demos import ee_skelton as _skel

    print("[ts-skel] extracting drawer + robot trajectory from demo...", flush=True)
    (
        times,
        drawer_qs_demo,
        robot_arm_qs_demo,
        arm_joint_names_demo,
        duration,
        episode_name,
        initial_state,
        states,
    ) = _extract_drawer_trajectory_from_demo(args)
    print(
        f"[ts-skel] demo: episode={episode_name} duration={duration:.2f}s "
        f"n_frames={len(times)} arm_qs_shape={robot_arm_qs_demo.shape}",
        flush=True,
    )

    ts_hz = float(getattr(args, "ts_sample_hz", 10.0))
    new_times, new_drawer_qs, new_arm_qs = _interpolate_drawer_poses(
        times,
        drawer_qs_demo,
        robot_arm_qs_demo,
        hz=ts_hz,
    )
    print(
        f"[ts-skel] interpolated to {ts_hz:.1f}Hz: {len(new_drawer_qs)} frames, "
        f"dt={1.0 / ts_hz:.3f}s duration={duration:.2f}s",
        flush=True,
    )
    # Record-only (per user: robot trajectory is loaded but does not drive the scene).
    args._autogen_ts_times = new_times
    args._autogen_ts_drawer_qs = new_drawer_qs
    args._autogen_ts_robot_qs = new_arm_qs
    args._autogen_ts_arm_joint_names = arm_joint_names_demo
    args._autogen_ts_duration = duration

    env = create_close_drawer_env(args)
    try:
        reset_to(env, initial_state)
        _set_drawer_joint_value(env, float(new_drawer_qs[0]))
        env.sim.forward()
        panel_ref = get_panel_frame(env)

        print(
            "[ts-skel] solving skeleton poses at reference drawer pose ...", flush=True
        )
        (
            skeleton,
            ref_poses,
            _fcache,
            _sgids,
            _oeqs,
            _irot,
        ) = _solve_reference_skeleton_poses(env, panel_ref, args)
        if not ref_poses:
            print(
                "[ts-skel] DAQP solve returned no skeleton poses at reference.",
                flush=True,
            )
            return [], env
        ref_poses = [p for p in ref_poses if isinstance(p, _skel.SkeletonPose)]
        max_ref_poses = int(getattr(args, "ts_skeleton_max_reference_poses", 512))
        if max_ref_poses > 0 and len(ref_poses) > max_ref_poses:
            rng = np.random.default_rng(int(getattr(args, "seed", 0)))
            keep_indices = np.sort(
                rng.choice(len(ref_poses), size=max_ref_poses, replace=False)
            )
            original_count = len(ref_poses)
            ref_poses = [ref_poses[int(i)] for i in keep_indices]
            args._autogen_ts_reference_sample_indices = keep_indices
            print(
                f"[ts-skel] downsampled reference skeleton poses: "
                f"{original_count} -> {len(ref_poses)} "
                f"(seed={int(getattr(args, 'seed', 0))})",
                flush=True,
            )
        else:
            args._autogen_ts_reference_sample_indices = np.arange(
                len(ref_poses), dtype=np.int64
            )
        # Assign a stable pose_id per reference pose; carried through project + filter.
        ref_id_poses = list(enumerate(ref_poses))  # [(pose_id, pose), ...]
        n_ref = len(ref_id_poses)
        print(
            f"[ts-skel] reference skeleton poses: {n_ref} (IDs 0..{n_ref - 1})",
            flush=True,
        )

        scene_pool = getattr(args, "_autogen_ts_scene_pool", None)
        pen_workers = int(
            getattr(args, "ts_penetration_workers", None)
            or getattr(args, "autogen_skeleton_parallel_workers", None)
            or getattr(args, "autogen_mink_parallel_workers", None)
            or max(1, (os.cpu_count() or 4) - 2)
        )
        if scene_pool is None or getattr(scene_pool, "num_workers", 1) < pen_workers:
            try:
                from robocasa.demos.skelton_scene import SkeletonScenePool

                scene_pool = SkeletonScenePool.from_env(env, num_workers=pen_workers)
                args._autogen_ts_scene_pool = scene_pool
            except Exception as exc:
                print(f"[ts-skel] SkeletonScenePool init failed: {exc}", flush=True)
                scene_pool = None

        finger_radius = float(getattr(args, "autogen_skeleton_finger_radius", 0.004))
        penetration_tol = float(
            getattr(args, "autogen_skeleton_object_penetration_tol", 0.001)
        )

        # --- GPU (mjwarp) penetration checker ---------------------------------
        # Batched narrowphase in a stripped floating-hand + static-kitchen scene
        # (drawer subtree excluded, since drawer is the intended contact target).
        # ~200x faster than the per-pose CPU mj_multiRay loop for ~3k poses.
        mjwarp_checker = None
        use_mjwarp = bool(getattr(args, "ts_penetration_mjwarp", True))
        if use_mjwarp:
            try:
                from robocasa.demos.ts_penetration_mjwarp import (
                    MjwarpSkelPenChecker,
                    _drawer_body_ids,
                    _robot_geom_ids,
                )

                # exclude = drawer subtree geoms + robot geoms + handle-inflating geoms.
                drawer_body_ids = _drawer_body_ids(env)
                raw_model = getattr(env.sim.model, "_model", env.sim.model)
                drawer_geom_ids = {
                    int(g)
                    for g in range(int(raw_model.ngeom))
                    if int(raw_model.geom_bodyid[g]) in drawer_body_ids
                }
                robot_ids = _robot_geom_ids(env)
                inflating_ids = set()
                try:
                    for name in _inflating_handle_geom_names(env):
                        if name in env.sim.model._geom_name2id:
                            inflating_ids.add(int(env.sim.model._geom_name2id[name]))
                except Exception:
                    pass
                exclude_ids = drawer_geom_ids | robot_ids | inflating_ids
                mjwarp_batch = int(getattr(args, "ts_penetration_mjwarp_batch", 512))
                mjwarp_batch = max(1, min(mjwarp_batch, max(1, n_ref)))
                frame_name_str = str(args.mink_contact_frame).split(":")[0]
                mjwarp_checker = MjwarpSkelPenChecker(
                    env,
                    ee_site_name=frame_name_str,
                    exclude_geom_ids=exclude_ids,
                    batch_size=mjwarp_batch,
                )
                print(
                    f"[ts-skel] mjwarp penetration checker: "
                    f"backend={mjwarp_checker.stats.backend_kind} "
                    f"batch={mjwarp_checker.stats.n_worlds} "
                    f"static_geoms={mjwarp_checker.stats.n_static_geoms} "
                    f"hand_geoms={mjwarp_checker.stats.n_hand_geoms} "
                    f"build_s={mjwarp_checker.stats.build_seconds:.2f}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[ts-skel] mjwarp checker init failed ({exc}); "
                    "falling back to CPU mj_multiRay checker.",
                    flush=True,
                )
                mjwarp_checker = None

        ts_poses = []
        panel_ref_tuple = (
            panel_ref.center_world.copy(),
            panel_ref.rotation_world.copy(),
        )

        try:
            from tqdm import tqdm as _tqdm
        except Exception:
            _tqdm = None
        frame_pbar = (
            _tqdm(
                total=len(new_drawer_qs),
                desc="ts-frames",
                unit="frame",
                file=sys.__stdout__,
                dynamic_ncols=True,
                mininterval=0.2,
                leave=True,
            )
            if _tqdm is not None
            else None
        )

        # Precompute reference ee positions/rotations once (they don't change).
        ee_positions_ref = np.stack(
            [np.asarray(p.ee_position, dtype=np.float64) for _pid, p in ref_id_poses],
            axis=0,
        )  # (n_ref, 3)
        ee_rotations_ref = np.stack(
            [np.asarray(p.ee_rotation, dtype=np.float64) for _pid, p in ref_id_poses],
            axis=0,
        )  # (n_ref, 3, 3)
        contact_points_ref = np.stack(
            [
                np.asarray(p.contact_point_world, dtype=np.float64)
                for _pid, p in ref_id_poses
            ],
            axis=0,
        )
        contact_normals_ref = np.stack(
            [
                np.asarray(p.contact_normal_world, dtype=np.float64)
                for _pid, p in ref_id_poses
            ],
            axis=0,
        )

        for frame_idx, drawer_q in enumerate(new_drawer_qs):
            _set_drawer_joint_value(env, float(drawer_q))
            env.sim.forward()
            panel_cur = get_panel_frame(env)
            panel_cur_tuple = (panel_cur.center_world, panel_cur.rotation_world)

            # Vectorized rigid projection: apply (dR, dp) to ee_pos, ee_rot in one shot.
            c_ref, R_ref = panel_ref_tuple
            c_new, R_new = panel_cur_tuple
            dR = R_new @ R_ref.T
            dp = c_new - dR @ c_ref
            new_positions = ee_positions_ref @ dR.T + dp[None, :]  # (n_ref, 3)
            new_rotations = dR[None, :, :] @ ee_rotations_ref  # (n_ref, 3, 3)
            new_contact_pts = contact_points_ref @ dR.T + dp[None, :]
            new_contact_normals = contact_normals_ref @ dR.T

            projected = [
                (
                    pid,
                    ee_skelton.SkeletonPose(
                        ee_position=new_positions[i],
                        ee_rotation=new_rotations[i],
                        contact_finger=ref_id_poses[i][1].contact_finger,
                        contact_point_world=new_contact_pts[i],
                        contact_normal_world=new_contact_normals[i],
                        qp_cost=ref_id_poses[i][1].qp_cost,
                        lift=ref_id_poses[i][1].lift,
                        theta=ref_id_poses[i][1].theta,
                        gripper_opening=ref_id_poses[i][1].gripper_opening,
                        contact_primitive=ref_id_poses[i][1].contact_primitive,
                    ),
                )
                for i, (pid, _p) in enumerate(ref_id_poses)
            ]

            if projected and mjwarp_checker is not None:
                try:
                    t_pen0 = time.perf_counter()
                    keep_mask = mjwarp_checker.check_batch(
                        new_positions,
                        new_rotations,
                        tol=penetration_tol,
                    )
                    kept = [
                        (pid, sp)
                        for i, (pid, sp) in enumerate(projected)
                        if bool(keep_mask[i])
                    ]
                    dropped_ids = [
                        pid
                        for i, (pid, _) in enumerate(projected)
                        if not bool(keep_mask[i])
                    ]
                    projected = kept
                    if frame_idx % 5 == 0:
                        print(
                            f"[ts-skel] frame {frame_idx} mjwarp filter "
                            f"kept={len(projected)}/{len(ref_id_poses)} "
                            f"dropped_ids={dropped_ids[:10]}"
                            f"{'...' if len(dropped_ids) > 10 else ''} "
                            f"t={time.perf_counter() - t_pen0:.3f}s",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"[ts-skel] frame {frame_idx} mjwarp check failed: {exc}",
                        flush=True,
                    )

            ts_poses.append((float(drawer_q), panel_cur, projected, frame_idx))
            if frame_pbar is not None:
                frame_pbar.update(1)

        if frame_pbar is not None:
            frame_pbar.close()

        print(
            f"[ts-skel] done. total frames={len(ts_poses)}, "
            f"avg poses/frame={np.mean([len(t[2]) for t in ts_poses]):.1f}, "
            f"n_ref_poses={n_ref}",
            flush=True,
        )
        # Stash n_ref so the visualizer can size palette by ID rather than max poses.
        args._autogen_ts_n_ref_poses = n_ref
        return ts_poses, env
    except Exception as exc:
        print(f"[ts-skel] error: {exc}", flush=True)
        import traceback

        traceback.print_exc()
        try:
            env.close()
        except Exception:
            pass
        return [], None


def _panel_frame_from_raw(raw_data, raw_model, drawer_name):
    """Read PanelFrame from raw MuJoCo data (thread-safe with viewer)."""
    panel_geom_name = f"{drawer_name}_door_reg_main"
    try:
        gid = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, panel_geom_name)
    except Exception:
        panel_geom_name = f"{drawer_name}_door_g1"
        gid = mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_GEOM, panel_geom_name)
    gid = int(gid)
    center = np.array(raw_data.geom_xpos[gid], dtype=np.float64).reshape(3)
    rot = np.array(raw_data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    half_size = np.asarray(raw_model.geom_size[gid], dtype=np.float64).copy()
    outward = -rot[:, 1]
    push_world = -outward
    if np.dot(push_world, rot[:, 1]) < 0.0:
        rot[:, 1] *= -1.0
    return PanelFrame(
        center_world=center,
        rotation_world=rot,
        half_size=half_size,
        outward_world=outward,
        push_world=push_world,
        geom_name=panel_geom_name,
    )


def visualize_timeseries_skeleton_poses(ts_poses, env, skeleton, args):
    """Animate drawer + skeleton poses in a passive MuJoCo popup."""
    if not ts_poses:
        print("[ts-viz] no time-series poses to visualize.", flush=True)
        return
    ts_poses = list(reversed(ts_poses))

    import contextlib
    import io
    import mujoco
    import mujoco.viewer
    from robocasa.demos import visualize_mujoco as viz_mj
    from robocasa.demos import ee_skelton as _skel

    drawer_centers = np.asarray(
        [np.asarray(t[1].center_world, dtype=np.float64) for t in ts_poses]
    )
    lookat = drawer_centers.mean(axis=0)
    finger_radius = float(getattr(args, "autogen_skeleton_finger_radius", 0.004))
    n_frames_total = len(ts_poses)
    # ts_poses[i][2] is [(pose_id, pose), ...] with stable IDs 0..n_ref-1.
    # Size the palette by n_ref so each pose_id has its own color across frames.
    n_ref_poses = int(getattr(args, "_autogen_ts_n_ref_poses", 0)) or max(
        (max((pid for pid, _ in t[2]), default=0) + 1 for t in ts_poses),
        default=1,
    )
    palette = _skel._hsv_palette(max(n_ref_poses, 1)).astype(np.float32)
    show_labels = bool(getattr(args, "ts_viz_show_labels", True))
    viz_pose_limit_raw = getattr(args, "ts_viz_skeleton_pose_limit", None)
    if viz_pose_limit_raw is None:
        viz_pose_limit_raw = getattr(args, "autogen_visualize_skeleton_pose_limit", 120)
    viz_pose_limit = int(viz_pose_limit_raw)

    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    drawer_joint_name = env.drawer.door_joint_names[0]
    drawer_qpos_addr = int(
        raw_model.jnt_qposadr[_mj_id(raw_model, "joint", drawer_joint_name)]
    )

    frame_duration = float(getattr(args, "ts_viz_frame_seconds", 0.1))
    print(
        f"[ts-viz] launching popup: {n_frames_total} frames, "
        f"1 update / {frame_duration:.1f}s. Close window or Ctrl+C to exit.",
        flush=True,
    )

    try:
        with mujoco.viewer.launch_passive(
            raw_model,
            raw_data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            try:
                viewer.opt.geomgroup[:] = 1
                viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT)] = 0
                viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE)] = 0
            except Exception:
                pass
            viewer.cam.type = 0
            viewer.cam.fixedcamid = -1
            viewer.cam.lookat[:] = lookat
            viewer.cam.distance = float(
                getattr(args, "autogen_mink_popup_camera_distance", 0.85)
            )
            viewer.cam.azimuth = float(
                getattr(args, "autogen_mink_popup_camera_azimuth", 135.0)
            )
            viewer.cam.elevation = float(
                getattr(args, "autogen_mink_popup_camera_elevation", -25.0)
            )

            render_fps = max(float(getattr(args, "ts_viz_render_fps", 30.0)), 1.0)
            render_interval = 1.0 / render_fps

            frame_idx = 0
            drawer_q_from = float(ts_poses[0][0])
            drawer_q_to = (
                float(ts_poses[1 % n_frames_total][0])
                if n_frames_total > 1
                else drawer_q_from
            )
            panel_ref_tuple = (
                ts_poses[0][1].center_world.copy(),
                ts_poses[0][1].rotation_world.copy(),
            )
            # Frame entries are [(pose_id, pose), ...] with stable IDs.
            frame_id_poses = list(ts_poses[0][2])
            started_at = time.time()

            while viewer.is_running():
                elapsed = time.time() - started_at
                alpha = min(elapsed / frame_duration, 1.0)

                cur_drawer_q = (1.0 - alpha) * drawer_q_from + alpha * drawer_q_to
                raw_data.qpos[drawer_qpos_addr] = float(cur_drawer_q)
                mujoco.mj_forward(raw_model, raw_data)

                panel_cur = _panel_frame_from_raw(raw_data, raw_model, env.drawer.name)
                panel_cur_tuple = (
                    panel_cur.center_world.copy(),
                    panel_cur.rotation_world.copy(),
                )
                if frame_id_poses:
                    proj_poses = _project_skeleton_poses_to_drawer_pose(
                        [p for _pid, p in frame_id_poses],
                        panel_ref_tuple,
                        panel_cur_tuple,
                    )
                    cur_id_poses = list(
                        zip(
                            [pid for pid, _p in frame_id_poses],
                            proj_poses,
                        )
                    )
                else:
                    cur_id_poses = []

                if hasattr(viewer, "user_scn"):
                    viewer.user_scn.ngeom = 0
                    _draw_fn = getattr(_skel, "_draw_skeleton_into_scene", None)
                    drawn_pose_count = 0
                    if _draw_fn is not None:
                        poses_to_draw = cur_id_poses
                        if viz_pose_limit > 0:
                            poses_to_draw = poses_to_draw[:viz_pose_limit]
                        for pid, sp in poses_to_draw:
                            # Skeleton: 1 hand box + 2 finger capsules. Markers add
                            # contact + label spheres, so reserve conservatively.
                            if viewer.user_scn.ngeom + 8 > int(viewer.user_scn.maxgeom):
                                break
                            rgba = palette[int(pid) % palette.shape[0]].copy()
                            rgba_hand = rgba.copy()
                            rgba_hand[3] = max(float(rgba_hand[3]), 0.75)
                            before = int(viewer.user_scn.ngeom)
                            try:
                                with contextlib.redirect_stdout(
                                    io.StringIO()
                                ), contextlib.redirect_stderr(io.StringIO()):
                                    _draw_fn(
                                        viewer.user_scn,
                                        skeleton,
                                        np.asarray(sp.ee_position, dtype=np.float64),
                                        np.asarray(sp.ee_rotation, dtype=np.float64),
                                        float(
                                            getattr(
                                                sp,
                                                "gripper_opening",
                                                _skel.PANDA_DEFAULT_GRIPPER_OPENING,
                                            )
                                        ),
                                        rgba_hand,
                                        rgba,
                                        finger_radius,
                                    )
                            except Exception:
                                try:
                                    half = np.asarray(
                                        _skel._flat_hand_half_extents(skeleton),
                                        dtype=np.float64,
                                    )
                                    _skel._add_box(
                                        viewer.user_scn,
                                        np.asarray(sp.ee_position, dtype=np.float64),
                                        np.asarray(sp.ee_rotation, dtype=np.float64),
                                        half,
                                        rgba_hand,
                                    )
                                except Exception:
                                    pass
                            if int(viewer.user_scn.ngeom) == before:
                                continue
                            drawn_pose_count += 1
                            # Contact marker: makes each timestep's retained pose set
                            # visible even when hand geometry overlaps the drawer.
                            if viewer.user_scn.ngeom < int(viewer.user_scn.maxgeom):
                                try:
                                    viz_mj._add_scene_sphere(
                                        viewer.user_scn,
                                        np.asarray(
                                            sp.contact_point_world, dtype=np.float64
                                        ),
                                        0.005,
                                        np.array(
                                            [rgba[0], rgba[1], rgba[2], 0.95],
                                            dtype=np.float32,
                                        ),
                                    )
                                except Exception:
                                    pass
                            # Attach a label sphere with the pose_id just above the
                            # EE position so the number follows the projected pose.
                            if show_labels and viewer.user_scn.ngeom < int(
                                viewer.user_scn.maxgeom
                            ):
                                try:
                                    label_pos = np.asarray(
                                        sp.ee_position, dtype=np.float64
                                    ) + np.array([0.0, 0.0, 0.03])
                                    gi = int(viewer.user_scn.ngeom)
                                    g = viewer.user_scn.geoms[gi]
                                    mujoco.mjv_initGeom(
                                        g,
                                        int(mujoco.mjtGeom.mjGEOM_SPHERE),
                                        np.array(
                                            [0.006, 0.006, 0.006], dtype=np.float64
                                        ),
                                        label_pos,
                                        np.eye(3, dtype=np.float64).reshape(9),
                                        rgba_hand,
                                    )
                                    try:
                                        g.label = f"{int(pid)}"
                                    except Exception:
                                        try:
                                            g.label = f"{int(pid)}".encode("ascii")
                                        except Exception:
                                            pass
                                    viewer.user_scn.ngeom = gi + 1
                                except Exception:
                                    pass
                    try:
                        viz_mj._add_scene_sphere(
                            viewer.user_scn,
                            np.asarray(panel_cur.center_world, dtype=np.float64),
                            0.01,
                            np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32),
                        )
                    except Exception:
                        pass

                viewer.sync()

                if elapsed >= frame_duration:
                    started_at = time.time()
                    frame_idx = (frame_idx + 1) % n_frames_total
                    drawer_q_from = float(ts_poses[frame_idx][0])
                    drawer_q_to = float(ts_poses[(frame_idx + 1) % n_frames_total][0])
                    panel_ref_tuple = (
                        ts_poses[frame_idx][1].center_world.copy(),
                        ts_poses[frame_idx][1].rotation_world.copy(),
                    )
                    frame_id_poses = list(ts_poses[frame_idx][2])
                    surviving_ids = [int(pid) for pid, _ in frame_id_poses]
                    if frame_idx % 10 == 0:
                        print(
                            f"[ts-viz] frame {frame_idx}/{n_frames_total} "
                            f"drawer_q={drawer_q_from:.4f} "
                            f"live_ids={surviving_ids[:15]}"
                            f"{'...' if len(surviving_ids) > 15 else ''} "
                            f"({len(surviving_ids)}/{n_ref_poses})",
                            flush=True,
                        )

                time.sleep(render_interval)

    except KeyboardInterrupt:
        print("[ts-viz] stopped by user.", flush=True)
    except Exception as exc:
        print(f"[ts-viz] stopped: {exc}", flush=True)
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()


def run_timeseries_pipeline(args):
    """Entry point for ``--ts-skeleton``."""
    print("[ts-pipeline] ====== time-series skeleton pipeline ======", flush=True)
    ts_poses, env = solve_timeseries_skeleton_poses(args)
    if not ts_poses:
        print("[ts-pipeline] no poses produced. Exiting.", flush=True)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        return
    if env is None:
        env = create_close_drawer_env(args)
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    import robocasa.demos.ee_skelton as _skel

    skeleton = _skel.build_panda_skeleton(env, frame_name)
    try:
        visualize_timeseries_skeleton_poses(ts_poses, env, skeleton, args)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
