import argparse
import contextlib
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
                "    from robocasa.demos.demo_open_drawer_autogen import main\n"
                "main()"
            ),
            *sys.argv[1:],
        ],
    )

import numpy as np
import tempfile
import trimesh
from scipy.spatial import Delaunay, cKDTree

from robocasa.demos.demo_open_drawer_contact_curobo import (
    _apply_config_overrides,
    _build_surface_contact_optimizer,
    _check_arm_q_collision_for_surface_base,
    _load_yaml_config,
    _solve_stage as _open_drawer_solve_stage,
    main as _open_drawer_main,
)
from robocasa.demos.object_cso import farthest_point_subset
from robocasa.demos import ee_skelton
from robocasa.demos.dream import solve_arm_q_mppi

_BASE_SOLVE_STAGE = getattr(
    _open_drawer_solve_stage, "__autogen_base__", _open_drawer_solve_stage
)


def _autogen_print(message):
    print(message, file=sys.__stdout__, flush=True)


_ANSI_YELLOW = "\033[33m"
_ANSI_RESET = "\033[0m"


def _autogen_print_yellow(message):
    print(f"{_ANSI_YELLOW}{message}{_ANSI_RESET}", file=sys.__stdout__, flush=True)


import threading as _threading

_DUAL_DAQP_STATS_LOCK = _threading.Lock()


def _record_dual_daqp_stats(args, record: dict) -> None:
    """Append one DAQP-call stats record to args._dual_daqp_stats (thread-safe)."""
    lst = getattr(args, "_dual_daqp_stats", None)
    if lst is None:
        lst = []
        setattr(args, "_dual_daqp_stats", lst)
    with _DUAL_DAQP_STATS_LOCK:
        lst.append(record)


with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
    open(os.devnull, "w")
):
    import robocasa.demos.demo_close_drawer_contact_curobo as close_demo
    from robocasa.demos import mink_q
    import robocasa.demos.mink_solver as mink_solver


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


def _normalize(vec, fallback=(1.0, 0.0, 0.0)):
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        return arr / norm
    return np.asarray(fallback, dtype=np.float64).reshape(3)


def _orthonormal_tangents(normal):
    normal = _normalize(normal)
    ref = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(normal[2])) < 0.9
        else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    t1 = _normalize(np.cross(ref, normal))
    t2 = _normalize(np.cross(normal, t1))
    return t1, t2


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
    """Subdivide a triangle mesh so no edge is longer than `max_edge`.

    Why: drawer handles in robocasa are typically MuJoCo box primitives (12
    triangles). Sampling one contact candidate per face leaves only a handful
    of candidates per COACD part, which produces extremely sparse and
    end-clustered feasible points after the friction-cone filter.
    """
    if max_edge is None or float(max_edge) <= 0.0:
        return mesh
    try:
        v, f = trimesh.remesh.subdivide_to_size(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            max_edge=float(max_edge),
        )
        if v.size == 0 or f.size == 0:
            return mesh
        return trimesh.Trimesh(vertices=v, faces=f, process=False)
    except Exception:
        return mesh


def _convex_part_equations(parts):
    """Return (P, H, 4) padded outward-normal hyperplane equations per COACD part.

    Each row encodes `n·x + d <= 0` for points inside the part. Padding rows
    are `(0, 0, 0, -1e6)` so they never bind in the max-over-planes signed
    distance used by `ee_skelton._signed_distance_to_convex`.
    """
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return np.zeros((0, 1, 4), dtype=np.float64)
    eq_list = []
    for part in parts:
        verts = np.asarray(part.vertices, dtype=np.float64)
        if verts.shape[0] < 4:
            continue
        try:
            hull = ConvexHull(verts)
            eq_list.append(np.asarray(hull.equations, dtype=np.float64))
        except Exception:
            continue
    if not eq_list:
        return np.zeros((0, 1, 4), dtype=np.float64)
    h_max = max(int(e.shape[0]) for e in eq_list)
    padded = np.zeros((len(eq_list), h_max, 4), dtype=np.float64)
    padded[..., 3] = -1e6
    for i, e in enumerate(eq_list):
        padded[i, : e.shape[0], :] = e
    return padded


def _coacd_parts(mesh, args, seed):
    try:
        import coacd
    except ImportError as exc:
        raise RuntimeError(
            "demo_open_drawer_autogen.py requires the `coacd` package"
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


def _polygon_records_from_mesh(mesh, geom_name, center_hint=None):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        return []
    center_hint = (
        np.asarray(center_hint, dtype=np.float64).reshape(3)
        if center_hint is not None
        else np.mean(vertices, axis=0)
    )
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
                "vertices": tri.copy(),
                "normal": normal,
                "tangent1": t1,
                "tangent2": t2,
                "geom_name": str(geom_name),
            }
        )
    return records


def _target_geom_ids(env, surface):
    ids = []
    for name in tuple(getattr(surface, "allowed_geom_names", ())) or (
        surface.geom_name,
    ):
        if name in env.sim.model._geom_name2id:
            ids.append(int(env.sim.model.geom_name2id(name)))
    if not ids and surface.geom_name in env.sim.model._geom_name2id:
        ids.append(int(env.sim.model.geom_name2id(surface.geom_name)))
    return tuple(sorted(set(ids)))


def _body_names_for_geom_ids(env, geom_ids):
    model = env.sim.model
    body_id_to_name = {
        int(body_id): str(name)
        for name, body_id in getattr(model, "_body_name2id", {}).items()
    }
    names = []
    for geom_id in geom_ids:
        body_id = int(model.geom_bodyid[int(geom_id)])
        names.append(body_id_to_name.get(body_id, str(body_id)))
    return tuple(dict.fromkeys(names))


def _build_autogen_contact_candidates(env, surface, pull_distance, args):
    cached = getattr(args, "_autogen_candidates_by_surface", None)
    cache_key = (surface.name, float(close_demo._drawer_joint_value(env)))
    if cached is not None and cache_key in cached:
        return cached[cache_key]

    rng = np.random.default_rng(int(args.seed) + 17011)
    records = []
    convex_parts_world = []
    max_edge = float(getattr(args, "autogen_handle_subdivide_max_edge", 0.005))
    target_geom_ids = _target_geom_ids(env, surface)
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
            _coacd_parts(mesh_world, args, int(args.seed) + geom_id)
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
            f"No COACD polygons were produced for surface {surface.name!r}"
        )

    centers = np.asarray([r["center"] for r in records], dtype=np.float64)
    limit = int(getattr(args, "autogen_object_point_count", 384))
    if centers.shape[0] > limit:
        subset = farthest_point_subset(
            centers, limit, initial_index=int(rng.integers(centers.shape[0]))
        )
        records = [records[int(i)] for i in subset]
        centers = centers[subset]

    local_points = (
        centers - np.asarray(surface.center_world, dtype=np.float64)
    ) @ np.asarray(surface.rotation_world, dtype=np.float64)
    normals_world = np.asarray(
        [_normalize(r["normal"]) for r in records], dtype=np.float64
    )
    normals_local = normals_world @ np.asarray(surface.rotation_world, dtype=np.float64)
    optimizer, mesh_path = _build_surface_contact_optimizer(surface, args)
    try:
        current_x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        x_d = np.array(
            [0.0, -float(pull_distance), 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        tau_o = np.zeros(6, dtype=np.float64)
        t1_batch = np.asarray(
            [r["tangent1"] @ surface.rotation_world for r in records], dtype=np.float64
        )
        t2_batch = np.asarray(
            [r["tangent2"] @ surface.rotation_world for r in records], dtype=np.float64
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

    pull_world = _normalize(surface.pull_world)
    mu = max(float(args.contact_friction), 1e-6)
    friction_threshold = 1.0 / float(np.sqrt(1.0 + mu * mu))
    candidates = []
    for idx, record in enumerate(records):
        normal_world = _normalize(normals_world[idx])
        pull_along_inward_normal = -float(np.dot(pull_world, normal_world))
        friction_ok = pull_along_inward_normal >= friction_threshold
        feasible = bool(
            np.isfinite(float(cost_batch[idx]))
            and float(cost_batch[idx]) <= float(args.contact_cost_threshold)
            and (friction_ok or not bool(args.require_friction_cone_pull))
        )
        candidate = close_demo.ContactCandidate(
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
        candidate.surface_projection_distance = 0.0
        candidate.friction_cone_ok = bool(friction_ok)
        candidate.friction_cone_pull_ratio = float(pull_along_inward_normal)
        candidate.scene_penetration = False
        candidate.scene_min_margin = float("nan")
        candidate.visible = True
        candidates.append(candidate)

    feasible_indices = np.asarray(
        [i for i, c in enumerate(candidates) if bool(c.feasible)], dtype=np.int64
    )
    if feasible_indices.size == 0 and bool(args.require_feasible_contact):
        best = min(candidates, key=lambda c: c.cost)
        raise RuntimeError(
            "Autogen COACD contact search found no feasible point. "
            f"best_cost={best.cost:.6f}, threshold={float(args.contact_cost_threshold):.6f}"
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
    for candidate in candidates:
        if not hasattr(candidate, "is_edge"):
            candidate.is_edge = False

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
    surface_rotation = np.asarray(surface.rotation_world, dtype=np.float64)
    if feasible_positions_world.shape[0] > 0:
        tree_points = (
            feasible_positions_world - surface.center_world
        ) @ surface_rotation
        tree = cKDTree(tree_points)
    else:
        tree_points = np.zeros((0, 3), dtype=np.float64)
        tree = cKDTree(np.zeros((1, 3), dtype=np.float64))
    feasible_cache = AutogenFeasibleContactCache(
        candidate_indices=feasible_indices,
        positions_world=feasible_positions_world,
        positions_object=tree_points,
        normals_world=feasible_normals_world,
        normals_object=feasible_normals_world @ surface_rotation,
        tangents1_world=np.asarray(tangents1_world, dtype=np.float64).reshape(-1, 3),
        tangents2_world=np.asarray(tangents2_world, dtype=np.float64).reshape(-1, 3),
        tangents1_object=np.asarray(tangents1_world, dtype=np.float64).reshape(-1, 3)
        @ surface_rotation,
        tangents2_object=np.asarray(tangents2_world, dtype=np.float64).reshape(-1, 3)
        @ surface_rotation,
        is_edge=is_edge,
        graph_edges=graph_edges,
        tree=tree,
    )
    selected = min(
        (c for c in candidates if bool(c.feasible)),
        key=lambda c: c.cost,
        default=min(candidates, key=lambda c: c.cost),
    )
    if cached is None:
        cached = {}
        args._autogen_candidates_by_surface = cached
    handle_convex_equations = _convex_part_equations(convex_parts_world)
    feasible_cache.handle_convex_equations = handle_convex_equations
    cached[cache_key] = (candidates, selected, feasible_cache)
    args._autogen_feasible_cache = feasible_cache
    args._autogen_handle_convex_equations = handle_convex_equations
    _autogen_print(
        "drawer_links="
        f"{','.join(_body_names_for_geom_ids(env, target_geom_ids))} "
        f"total_sample_points={len(candidates)}"
    )
    _autogen_print(
        f"feasible_points={int(feasible_indices.size)} "
        f"handle_convex_parts={int(handle_convex_equations.shape[0])} "
        f"coacd_parts={len(convex_parts_world)} "
        f"candidate_count={len(candidates)}"
    )
    # --- 2. COACD + candidate data flow: also export the handle mesh in the
    #     object frame so MIQP (step 3) and the rollout (step 8) share ONE
    #     geometry built in this same COACD run.
    try:
        _export_grasp_handle_mesh(
            env, surface, args, convex_parts_world, feasible_cache
        )
    except Exception as exc:
        _autogen_print(f"[grasp] handle mesh export failed: {exc!r}")
    return cached[cache_key]


def _export_grasp_handle_mesh(env, surface, args, convex_parts_world, feasible_cache):
    """Export the cached COACD parts as a single mesh in the *object* frame and
    record the object-frame pose on ``feasible_cache`` / ``args`` so MIQP and
    the grasp rollout evaluate force closure on identical geometry.

    Object frame convention (matches ``_build_autogen_contact_candidates``):
        local = (world - center) @ rotation_world
    so the local->world map is ``world = (rotation_world.T) @ local + center``.
    """
    center = np.asarray(surface.center_world, dtype=np.float64).reshape(3)
    rot_w2o = np.asarray(surface.rotation_world, dtype=np.float64).reshape(3, 3)
    local_parts = []
    for part in convex_parts_world:
        verts = np.asarray(part.vertices, dtype=np.float64)
        faces = np.asarray(part.faces, dtype=np.int64)
        if verts.size == 0 or faces.size == 0:
            continue
        local_verts = (verts - center) @ rot_w2o
        local_parts.append(
            trimesh.Trimesh(vertices=local_verts, faces=faces, process=False)
        )
    if not local_parts:
        return
    mesh_local = trimesh.util.concatenate(local_parts)
    mesh_path = tempfile.NamedTemporaryFile(
        prefix="robocasa_grasp_handle_", suffix=".stl", delete=False
    ).name
    mesh_local.export(mesh_path)

    rot_o2w = rot_w2o.T
    quat_xyzw = _quat_wxyz_from_matrix(rot_o2w)
    obj_quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    feasible_cache._grasp_handle_mesh_path = mesh_path
    feasible_cache._grasp_obj_pos = center
    feasible_cache._grasp_obj_quat_wxyz = obj_quat_wxyz
    feasible_cache._grasp_obj_scale = np.ones(3, dtype=np.float64)
    args._autogen_grasp_handle_mesh_path = mesh_path
    args._autogen_grasp_obj_pos = center
    args._autogen_grasp_obj_quat = obj_quat_wxyz
    _autogen_print(
        f"[grasp] handle_mesh verts={int(mesh_local.vertices.shape[0])} "
        f"parts={len(local_parts)} path={mesh_path}"
    )


def evaluate_open_contacts(env, surface, pull_distance, args):
    candidates, selected, feasible_cache = _build_autogen_contact_candidates(
        env, surface, pull_distance, args
    )
    setattr(
        args,
        "_last_contact_stage_stats",
        {
            "stage1_success_count": int(
                np.count_nonzero([c.feasible for c in candidates])
            ),
            "stage1_elapsed": 0.0,
        },
    )
    args._autogen_feasible_cache = feasible_cache
    return candidates, selected


def _visualize_mink_q_poses_popup(env, q_waypoints, robot_state, args, drawer_q=None):
    if not bool(getattr(args, "autogen_visualize_mink_poses", True)):
        return
    q_waypoints = np.asarray(q_waypoints, dtype=np.float64)
    if q_waypoints.size == 0:
        return
    q_waypoints = q_waypoints.reshape(-1, 7)
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
        float(close_demo._drawer_joint_value(env))
        if drawer_q is None
        else float(drawer_q)
    )
    poses = []
    try:
        close_demo._set_drawer_joint_value(env, drawer_q)
        ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(env, frame_name)
        for q_arm in q_waypoints:
            close_demo._set_env_arm_q(env, arm_joint_names, q_arm)
            close_demo._set_drawer_joint_value(env, drawer_q)
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
    with mujoco.viewer.launch_passive(
        raw_model,
        raw_data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Only show visual geoms (group 1) — by default launch_passive enables
        # all groups, which makes the red collision geoms overlap visuals and
        # paints the scene/arm in MuJoCo's debug colors.
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
        while viewer.is_running():
            if hasattr(viewer, "user_scn"):
                viewer.user_scn.ngeom = 0
                for pose_index, (target_pos, target_rot) in enumerate(poses):
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


def _visualize_grasp_precontact_popup(
    env,
    arm_joint_names,
    q_arm,
    gripper_opening,
    drawer_q,
    target_pos,
    target_rot,
    args,
):
    """Launch a passive MuJoCo viewer showing the accepted pre-grasp arm-q
    and gripper opening. Restores env state on close. Skipped when
    ``autogen_visualize_grasp_precontact`` is False."""
    if not bool(getattr(args, "autogen_visualize_grasp_precontact", True)):
        return
    try:
        import mujoco
        import mujoco.viewer
    except Exception as exc:
        _autogen_print(f"[grasp] precontact popup import failed: {exc!r}")
        return

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()

    def _set_finger_qpos(opening):
        for joint_name in (
            "gripper0_right_finger_joint1",
            "gripper0_right_finger_joint2",
        ):
            if joint_name in env.sim.model._joint_name2id:
                jid = env.sim.model.joint_name2id(joint_name)
                addr = int(env.sim.model.jnt_qposadr[jid])
                env.sim.data.qpos[addr] = 0.5 * float(opening)

    try:
        close_demo._set_env_arm_q(env, tuple(arm_joint_names), np.asarray(q_arm))
        close_demo._set_drawer_joint_value(env, float(drawer_q))
        _set_finger_qpos(gripper_opening)
        env.sim.forward()

        lookat = np.asarray(target_pos, dtype=np.float64).reshape(3)
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
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / fps)
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()


def _visualize_grasp_precontact_ghosts_popup(
    env,
    arm_joint_names,
    candidates,
    drawer_q,
    target_pos,
    args,
):
    """Ghost popup: renders hand+finger ghosts at every accepted pregrasp
    (q_arm, gripper_opening) candidate. Accepted candidate is highlighted
    (green, opaque); others colored by rollout score (yellow=high, red=low).

    ``candidates``: list of tuples (q_arm_np(7), gripper_opening, rollout_score,
    is_accepted).
    """
    if not bool(getattr(args, "autogen_visualize_grasp_precontact", True)):
        return
    if not candidates:
        return
    try:
        import mujoco
        import mujoco.viewer
        from robocasa.demos import visualize_mujoco as viz_mj
    except Exception as exc:
        _autogen_print(f"[grasp] ghost popup import failed: {exc!r}")
        return

    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    if frame_name not in env.sim.model._site_name2id:
        _autogen_print(f"[grasp] ghost popup: site {frame_name!r} not found")
        return

    max_ghosts = int(getattr(args, "autogen_visualize_grasp_ghost_max", 30))
    if len(candidates) > max_ghosts:
        # Keep the accepted one plus top (max-1) by score.
        accepted = [c for c in candidates if c[3]]
        others = sorted(
            (c for c in candidates if not c[3]),
            key=lambda c: -c[2],
        )[: max(0, max_ghosts - len(accepted))]
        candidates = accepted + others

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    site_id = int(env.sim.model.site_name2id(frame_name))

    def _set_finger_qpos(opening):
        for joint_name in (
            "gripper0_right_finger_joint1",
            "gripper0_right_finger_joint2",
        ):
            if joint_name in env.sim.model._joint_name2id:
                jid = env.sim.model.joint_name2id(joint_name)
                addr = int(env.sim.model.jnt_qposadr[jid])
                env.sim.data.qpos[addr] = 0.5 * float(opening)

    # Sample the EE pose (site) at each candidate's arm+gripper config.
    ghost_poses = []  # list of (pos, rot, opening, score, is_accepted)
    try:
        close_demo._set_drawer_joint_value(env, float(drawer_q))
        ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(env, frame_name)
        for q_arm, g_open, score, accepted in candidates:
            close_demo._set_env_arm_q(env, tuple(arm_joint_names), np.asarray(q_arm))
            close_demo._set_drawer_joint_value(env, float(drawer_q))
            _set_finger_qpos(g_open)
            env.sim.forward()
            ghost_poses.append(
                (
                    np.asarray(
                        env.sim.data.site_xpos[site_id], dtype=np.float64
                    ).copy(),
                    np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64)
                    .reshape(3, 3)
                    .copy(),
                    float(g_open),
                    float(score),
                    bool(accepted),
                )
            )
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()

    if not ghost_poses:
        return

    # Color palette: accepted = green opaque; others = red→yellow by score.
    scores = np.asarray([p[3] for p in ghost_poses], dtype=np.float64)
    smin, smax = float(scores.min()), float(scores.max())
    span = max(smax - smin, 1e-6)
    alpha = float(getattr(args, "autogen_mink_ghost_alpha", 0.32))

    def _color(score, accepted):
        if accepted:
            return np.array([0.1, 1.0, 0.2, 0.95], dtype=np.float32)
        t = float((score - smin) / span)  # 0=worst, 1=best
        # red (1,0.1,0.1) -> yellow (1,1,0.2)
        return np.array([1.0, 0.1 + 0.9 * t, 0.1 + 0.1 * t, alpha], dtype=np.float32)

    lookat = np.mean(np.asarray([p[0] for p in ghost_poses], dtype=np.float64), axis=0)
    if target_pos is not None:
        lookat = 0.5 * (lookat + np.asarray(target_pos, dtype=np.float64).reshape(3))

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
        _autogen_print(
            f"[grasp] ghost popup: {len(ghost_poses)} candidates "
            f"(accepted highlighted green, score range [{smin:+.3f}, {smax:+.3f}])"
        )
        while viewer.is_running():
            if hasattr(viewer, "user_scn"):
                viewer.user_scn.ngeom = 0
                for pos, rot, _open, score, accepted in ghost_poses:
                    rgba = _color(score, accepted)
                    for ghost in ghost_geoms:
                        viz_mj._add_ghost_geom(viewer.user_scn, ghost, pos, rot, rgba)
            viewer.sync()
            time.sleep(1.0 / fps)


def _solve_stage_autogen(env, surface, pull_distance, args, stage_name):
    # Grasp mode: disable the pull stage (we close the gripper instead).
    _pull_backup = getattr(args, "execute_pull_stage", True)
    args.execute_pull_stage = False
    try:
        stage, reports, robot_state = _BASE_SOLVE_STAGE(
            env, surface, pull_distance, args, stage_name
        )
    finally:
        args.execute_pull_stage = _pull_backup
    if not bool(getattr(args, "_autogen_visualized_mink_precontact_q", False)):
        _visualize_mink_q_poses_popup(
            env,
            getattr(stage.mink_solution, "q_waypoints", np.zeros((0, 7))),
            robot_state,
            args,
            drawer_q=float(stage.start_drawer_q),
        )
    feasible_cache = getattr(args, "_autogen_feasible_cache", None)
    if feasible_cache is not None:
        stage.feasible_graph_edges = np.asarray(
            feasible_cache.graph_edges, dtype=np.int64
        )
        stage.autogen_feasible_is_edge = np.asarray(feasible_cache.is_edge, dtype=bool)
        stage.autogen_feasible_tangents2_object = np.asarray(
            feasible_cache.tangents2_object, dtype=np.float64
        )
        stage.autogen_initial_poses = np.asarray(
            getattr(args, "_autogen_initial_poses", np.zeros((0, 7))),
            dtype=np.float64,
        )
        stage.autogen_initial_candidate_indices = np.asarray(
            getattr(
                args, "_autogen_initial_candidate_indices", np.zeros(0, dtype=np.int64)
            ),
            dtype=np.int64,
        )
    return stage, reports, robot_state


_solve_stage_autogen.__autogen_base__ = _BASE_SOLVE_STAGE


def _patch_skeleton_viewers_geomgroup(ee_skelton_module):
    """Replace ee_skelton's skeleton-popup viewers with versions that only
    enable the visual geom group (group 1).

    The originals set ``viewer.opt.geomgroup[:] = 1`` which turns on every
    group including collision geoms (group 0).  MuJoCo renders those in its
    debug red/green palette, so the whole scene and arm look wrong.  We make
    a minimal copy of each function with the corrected ``geomgroup`` lines so
    the popups render exactly like ``_visualize_mink_qposes_popup`` does.
    """
    import mujoco
    from robocasa.demos import visualize_mujoco as viz_mj

    def visualize_skeleton_poses_fixed(env, ee_site_name, skeleton, poses, args):
        if not bool(getattr(args, "autogen_visualize_skeleton_poses", True)):
            return
        if not poses:
            return
        geoms_per_pose = 4
        max_poses = max(
            int(
                getattr(
                    args,
                    "autogen_visualize_skeleton_pose_max",
                    getattr(args, "autogen_visualize_skeleton_pose_limit", 30),
                )
            ),
            1,
        )
        if len(poses) > max_poses:
            step = max(len(poses) // max_poses, 1)
            poses = poses[::step][:max_poses]
        raw_model, raw_data = viz_mj._raw_model_data(env)
        body_ids = viz_mj._ghost_source_body_ids(env)
        arm_geom_ids = [
            gid
            for gid in range(int(raw_model.ngeom))
            if int(raw_model.geom_bodyid[gid]) in body_ids
        ]
        saved_rgba = raw_model.geom_rgba.copy()
        try:
            for gid in arm_geom_ids:
                raw_model.geom_rgba[gid, 3] = 0.25
            lookat = np.mean(
                np.asarray([p.ee_position for p in poses], dtype=np.float64), axis=0
            )
            palette = ee_skelton_module._hsv_palette(len(poses))
            finger_radius = float(
                getattr(args, "autogen_skeleton_finger_radius", 0.004)
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
                green = np.array([0.1, 1.0, 0.2, 0.9], dtype=np.float32)
                while viewer.is_running():
                    if hasattr(viewer, "user_scn"):
                        viewer.user_scn.ngeom = 0
                        max_geom = int(viewer.user_scn.maxgeom)
                        for i, pose in enumerate(poses):
                            if viewer.user_scn.ngeom + geoms_per_pose > max_geom:
                                break
                            rgba = palette[i % palette.shape[0]]
                            rgba_capsule = rgba.copy()
                            rgba_capsule[3] = 0.9
                            rgba_hand = rgba_capsule.copy()
                            ee_skelton_module._draw_skeleton_into_scene(
                                viewer.user_scn,
                                skeleton,
                                np.asarray(pose.ee_position, dtype=np.float64),
                                np.asarray(pose.ee_rotation, dtype=np.float64),
                                float(
                                    getattr(
                                        pose,
                                        "gripper_opening",
                                        ee_skelton_module.PANDA_DEFAULT_GRIPPER_OPENING,
                                    )
                                ),
                                rgba_hand,
                                rgba_capsule,
                                finger_radius,
                            )
                            try:
                                viz_mj._add_scene_sphere(
                                    viewer.user_scn,
                                    np.asarray(
                                        pose.contact_point_world, dtype=np.float64
                                    ),
                                    0.005,
                                    green,
                                )
                            except Exception:
                                pass
                    viewer.sync()
                    time.sleep(1.0 / fps)
        finally:
            raw_model.geom_rgba[:] = saved_rgba

    def visualize_skeleton_and_ee_fixed(env, ee_site_name, skeleton, args):
        if not bool(getattr(args, "autogen_visualize_skeleton_preview", True)):
            return
        raw_model, raw_data = viz_mj._raw_model_data(env)
        body_ids = viz_mj._ghost_source_body_ids(env)
        arm_geom_ids = [
            gid
            for gid in range(int(raw_model.ngeom))
            if int(raw_model.geom_bodyid[gid]) in body_ids
        ]
        saved_rgba = raw_model.geom_rgba.copy()
        ee_pos, ee_rot = viz_mj._site_pose(env, ee_site_name)
        finger_radius = float(getattr(args, "autogen_skeleton_finger_radius", 0.004))
        gripper_opening = float(
            getattr(
                args,
                "autogen_skeleton_gripper_default",
                ee_skelton_module.PANDA_DEFAULT_GRIPPER_OPENING,
            )
        )
        try:
            for gid in arm_geom_ids:
                raw_model.geom_rgba[gid, 3] = 0.3
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
                viewer.cam.lookat[:] = ee_pos
                viewer.cam.distance = float(
                    getattr(args, "autogen_skeleton_preview_camera_distance", 0.35)
                )
                viewer.cam.azimuth = float(
                    getattr(args, "autogen_skeleton_preview_camera_azimuth", 135.0)
                )
                viewer.cam.elevation = float(
                    getattr(args, "autogen_skeleton_preview_camera_elevation", -20.0)
                )
                fps = max(float(getattr(args, "autogen_mink_popup_fps", 30.0)), 1.0)
                rgba_hand = np.array([0.1, 1.0, 0.2, 0.55], dtype=np.float32)
                rgba_finger = np.array([0.1, 1.0, 0.2, 0.95], dtype=np.float32)
                while viewer.is_running():
                    if hasattr(viewer, "user_scn"):
                        viewer.user_scn.ngeom = 0
                        ee_skelton_module._draw_skeleton_into_scene(
                            viewer.user_scn,
                            skeleton,
                            ee_pos,
                            ee_rot,
                            gripper_opening,
                            rgba_hand,
                            rgba_finger,
                            finger_radius,
                        )
                    viewer.sync()
                    time.sleep(1.0 / fps)
        finally:
            raw_model.geom_rgba[:] = saved_rgba

    if getattr(ee_skelton_module, "visualize_skeleton_poses", None) is not None:
        ee_skelton_module.visualize_skeleton_poses = visualize_skeleton_poses_fixed
    if getattr(ee_skelton_module, "visualize_skeleton_and_ee", None) is not None:
        ee_skelton_module.visualize_skeleton_and_ee = visualize_skeleton_and_ee_fixed


# ---------------------------------------------------------------------------
# Quiet skeleton draw: same as ee_skelton._draw_skeleton_into_scene but without
# the per-call `[skeleton_draw] hand half_ext=...` log line that floods stdout.
# ---------------------------------------------------------------------------
def _draw_skeleton_into_scene_quiet(
    scene,
    skeleton,
    ee_pos,
    ee_rot,
    gripper_opening,
    rgba_hand,
    rgba_finger,
    finger_radius,
):
    half_ext = ee_skelton._flat_hand_half_extents(skeleton)
    hand_center_w = np.asarray(ee_pos, dtype=np.float64) + np.asarray(
        ee_rot, dtype=np.float64
    ) @ np.asarray(skeleton.hand_box_center_ee, dtype=np.float64)
    hand_rot_w = np.asarray(ee_rot, dtype=np.float64) @ np.asarray(
        skeleton.hand_box_rotation_ee, dtype=np.float64
    )
    ee_skelton._add_box(scene, hand_center_w, hand_rot_w, half_ext, rgba_hand)
    left_seg, right_seg, _, _ = ee_skelton._finger_segments_with_opening(
        skeleton, gripper_opening
    )
    for seg in (left_seg, right_seg):
        sa = np.asarray(ee_pos, dtype=np.float64) + np.asarray(
            ee_rot, dtype=np.float64
        ) @ np.asarray(seg[0], dtype=np.float64)
        sb = np.asarray(ee_pos, dtype=np.float64) + np.asarray(
            ee_rot, dtype=np.float64
        ) @ np.asarray(seg[1], dtype=np.float64)
        ee_skelton._add_capsule_segment(scene, sa, sb, finger_radius, rgba_finger)


# ---------------------------------------------------------------------------
# DAQP-based skeleton pose solver with tqdm progress bar.
# Replaces the bare solve_skeleton_pose loop so the user sees a progress bar
# and the solver exploits the multi-theta DAQP batch path.
# ---------------------------------------------------------------------------
def _solve_skeleton_poses_daqp(
    env,
    skeleton,
    local_ids,
    points_world,
    normals_world,
    feasible_candidate_indices,
    handle_convex_equations,
    scene_geom_ids,
    demo_ee_rotation,
    args,
):
    jobs = []
    for local_id, point, normal in zip(local_ids, points_world, normals_world):
        candidate_index = int(feasible_candidate_indices[int(local_id)])
        for finger in ("left", "right"):
            jobs.append(
                (
                    int(candidate_index),
                    np.asarray(point, dtype=np.float64).reshape(3).copy(),
                    np.asarray(normal, dtype=np.float64).reshape(3).copy(),
                    str(finger),
                )
            )
    if not jobs:
        return []

    workers = int(getattr(args, "autogen_skeleton_parallel_workers", 1) or 1)
    active_workers = max(1, min(workers, len(jobs)))

    def _solve_one(job):
        candidate_index, point, normal, finger = job
        try:
            poses = ee_skelton.solve_skeleton_pose_candidates(
                env,
                skeleton,
                point,
                normal,
                finger=finger,
                object_convex_equations=handle_convex_equations,
                object_convex_equation_mask=None,
                scene_geom_ids=scene_geom_ids,
                initial_ee_rotation_world=demo_ee_rotation,
                args=args,
            )
            return int(candidate_index), list(poses), None
        except Exception as exc:
            return int(candidate_index), [], f"{exc.__class__.__name__}:{exc}"

    try:
        from tqdm import tqdm as _tqdm
    except Exception:
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

    skeleton_poses: list[tuple[int, ee_skelton.SkeletonPose]] = []
    if active_workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            candidate_index, poses, error = _solve_one(job)
            if error is None:
                for sp in poses:
                    skeleton_poses.append((candidate_index, sp))
            if pbar is not None:
                pbar.update(1)
    else:
        results_by_index = {}
        with ThreadPoolExecutor(max_workers=active_workers) as executor:
            future_by_index = {
                executor.submit(_solve_one, job): idx for idx, job in enumerate(jobs)
            }
            for future in as_completed(future_by_index):
                results_by_index[future_by_index[future]] = future.result()
                if pbar is not None:
                    pbar.update(1)
        for idx in range(len(jobs)):
            candidate_index, poses, error = results_by_index[idx]
            if error is None:
                for sp in poses:
                    skeleton_poses.append((candidate_index, sp))
    if pbar is not None:
        pbar.close()
    return skeleton_poses


# ---------------------------------------------------------------------------
# Grasp-mode helpers
# ---------------------------------------------------------------------------


def _extract_handle_mesh(env, surface):
    """Export the drawer handle mesh as a single ``trimesh.Trimesh`` (world frame)."""
    target_geom_ids = _target_geom_ids(env, surface)
    meshes = []
    max_edge = float(getattr(env.args, "autogen_handle_subdivide_max_edge", 0.005))
    for geom_id in target_geom_ids:
        mesh_world = _mesh_from_geom_world(env, geom_id)
        if mesh_world is None:
            continue
        for part_index, part in enumerate(
            _coacd_parts(mesh_world, env.args, int(env.args.seed) + geom_id)
        ):
            dense_part = _subdivide_mesh(part, max_edge)
            meshes.append(dense_part)
    if not meshes:
        return None
    combined = trimesh.util.concatenate(meshes)
    return combined


def _export_temp_mesh(mesh):
    """Export a trimesh to a temporary STL file and return its path."""
    path = tempfile.NamedTemporaryFile(
        prefix="robocasa_grasp_", suffix=".stl", delete=False
    ).name
    mesh.export(path)
    return path


def _quat_wxyz_from_matrix(rot_matrix):
    """3x3 rotation matrix -> xyzw quaternion (scipy convention)."""
    from scipy.spatial.transform import Rotation

    rot_matrix = np.asarray(rot_matrix, dtype=np.float64).reshape(3, 3)
    return Rotation.from_matrix(rot_matrix).as_quat().astype(np.float64)


def _quaternion_wxyz_to_mat(quat_wxyz):
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation

    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat_wxyz = quat_wxyz / max(float(np.linalg.norm(quat_wxyz)), 1e-9)
    return (
        Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        .as_matrix()
        .astype(np.float64)
    )


def _solve_dual_finger_skeleton_daqp(
    env,
    skeleton,
    left_contact_point,
    left_contact_normal,
    right_contact_point,
    right_contact_normal,
    *,
    demo_ee_rotation,
    handle_convex_equations,
    scene_geom_ids,
    args,
    return_candidates=False,
    max_candidates=None,
    min_theta_separation=0.0,
    raw_model_data=None,
):
    """DAQP-based dual-finger wrap-grasp skeleton pose solver.

    Finds a single EE pose + gripper opening that places the left/right
    finger *inner* surfaces at the two contact points simultaneously, while
    keeping all finger samples and hand box corners on the free-space side of
    the handle's COACD convex parts (the ``g`` grasp constraint).

    The decision variable is ``delta = [dx(3), omega(3), dg, sL, sR]`` (9
    vars): the SE(3) perturbation of the nominal EE pose, the gripper opening
    delta, and the per-finger segment parameters.  Left/right finger contact
    is enforced as affine equality rows in the DAQP constraint matrix (six
    rows total), not just as a soft cost.  The inequality constraints ``g`` are
    the linearized COACD clearance half-spaces for every finger sample and hand
    box corner, plus optional MuJoCo ``mj_ray`` scene-clearance half-spaces.

    Multiple ``theta`` samples (rotation about the contact-separation axis)
    are tried so the solver can return a diverse set of feasible poses.

    Returns a list of ``SkeletonPose`` (possibly empty).
    """
    import mujoco

    from robocasa.demos.mlqp_point_cabinet import _solve_qp_daqp

    finger_radius = float(getattr(args, "autogen_skeleton_finger_radius", 0.004))
    g_min = float(getattr(args, "autogen_skeleton_gripper_min", 0.005))
    g_max = float(
        getattr(
            args, "autogen_skeleton_gripper_max", ee_skelton.PANDA_MAX_GRIPPER_OPENING
        )
    )
    g_default = float(
        getattr(
            args,
            "autogen_skeleton_gripper_default",
            ee_skelton.PANDA_DEFAULT_GRIPPER_OPENING,
        )
    )
    contact_weight = float(getattr(args, "autogen_skeleton_contact_weight", 500.0))
    g_weight = float(getattr(args, "autogen_skeleton_gripper_weight", 10.0))
    reg_weight = float(getattr(args, "autogen_skeleton_reg_weight", 1.0))
    motion_bound = float(getattr(args, "autogen_skeleton_motion_bound", 0.05))
    rot_bound = float(getattr(args, "autogen_skeleton_rot_bound", 0.35))
    margin = float(getattr(args, "autogen_skeleton_margin", 0.002))
    # Finger vs. handle rows in the QP use a looser margin (default 0.0 =
    # "no positive clearance required") so the QP can produce wrap-around
    # solutions that graze the handle. Hand-box corners keep the strict
    # `margin` above. The post-validation clearance check below uses the
    # separate `penetration_tol_dual` which defaults to 5× the single-
    # contact penetration tolerance for the same reason.
    margin_finger_dual = float(getattr(args, "autogen_dual_object_margin", 0.0))
    clearance_tol = float(getattr(args, "autogen_skeleton_clearance_tolerance", 0.001))
    penetration_tol = float(
        getattr(args, "autogen_skeleton_object_penetration_tol", 0.001)
    )
    # Relaxed penetration tolerance used only in the DUAL-finger wrap-grasp
    # post-validation. When the QP places two fingers on the two selected
    # contact points, linearization near a curved handle often leaves finger
    # samples a few mm inside the COACD convex parts; the strict 0.001 m
    # tolerance rejects those poses even when they are physically plausible.
    penetration_tol_dual = float(
        getattr(
            args,
            "autogen_dual_object_penetration_tol",
            max(penetration_tol, 0.005),
        )
    )
    seg_samples = int(getattr(args, "autogen_skeleton_segment_samples", 5))
    theta_count = int(getattr(args, "autogen_dual_theta_count", 6))
    lift_min = float(getattr(args, "autogen_skeleton_lift_min", 0.002))
    lift_max = float(getattr(args, "autogen_skeleton_lift_max", 0.010))
    n_lift = max(1, int(getattr(args, "autogen_dual_n_lift", 1)))
    n_g = max(1, int(getattr(args, "autogen_dual_n_g", 1)))
    seed = int(getattr(args, "autogen_dual_seed", 7))
    rng = np.random.RandomState(seed)
    debug = bool(getattr(args, "autogen_dual_debug", False))
    dbg_qp_ok = 0
    dbg_err_fail = 0
    dbg_clear_fail = 0
    dbg_total = 0
    dbg_left_err_min = float("inf")
    dbg_right_err_min = float("inf")
    dbg_clear_min = float("inf")
    dbg_scene_rows = 0
    dbg_scene_fail = 0
    # Diagnostic accumulators: track WHY the QP is infeasible.
    dbg_status_hist: dict[int, int] = {}  # DAQP exitflag histogram
    dbg_eq_viol_min = float("inf")  # min over iters of max |A_eq x - b_eq| at x=0
    dbg_scene_rows_min = 0  # min scene rows per (theta, lift, g)
    dbg_scene_rows_max = 0
    # Per-iteration LS diagnostics (captured on first iteration).
    dbg_x_ls = np.zeros(9, dtype=np.float64)
    dbg_resid_ls = float("inf")
    dbg_rank_eq = 0
    dbg_ls_in_bounds = False
    dbg_ls_bound_viol = float("inf")
    dbg_eq_cond = float("inf")

    if raw_model_data is not None:
        raw_model, raw_data = raw_model_data
    else:
        raw_model = getattr(env.sim.model, "_model", env.sim.model)
        raw_data = getattr(env.sim.data, "_data", env.sim.data)
    scene_geom_ids_set = set(int(gid) for gid in (scene_geom_ids or ()))

    # NOTE: the scene pool is reset ONCE by the caller before dispatching
    # `_solve_dual_finger_skeleton_daqp` across a ThreadPoolExecutor. Do NOT
    # call `scene_pool_local.reset()` here — it invokes `mj_copyData` on the
    # shared per-worker `MjData` buffers, and concurrent calls from multiple
    # threads corrupt those buffers and segfault (observed as "skeleton_scene
    # _pool=on workers=8" followed by a bare segfault).

    left_contact_point = np.asarray(left_contact_point, dtype=np.float64).reshape(3)
    right_contact_point = np.asarray(right_contact_point, dtype=np.float64).reshape(3)
    left_contact_normal = _normalize(left_contact_normal)
    right_contact_normal = _normalize(right_contact_normal)

    # Finger segments + spread direction at the default opening.
    (
        left_seg0,
        right_seg0,
        y_hat_ee,
        left_sign,
    ) = ee_skelton._finger_segments_with_opening(skeleton, g_default)
    # _finger_segments_with_opening shifts each finger by
    # 0.5 * (opening - rest_opening), so the Jacobian wrt g is half the
    # signed spread direction.
    left_spread_ee = 0.5 * left_sign * y_hat_ee
    right_spread_ee = -0.5 * left_sign * y_hat_ee
    finger_dir_ee = _normalize(left_seg0[1] - left_seg0[0])

    # Contact-separation axis in world frame (left - right).
    sep_w = left_contact_point - right_contact_point
    dist_w = float(np.linalg.norm(sep_w))
    if dist_w < 1e-6:
        return [] if return_candidates else None
    y_w = sep_w / dist_w

    # EE basis: x = approach, y = finger-separation, z = finger (base->tip).
    x_ee = _normalize(np.cross(y_hat_ee, finger_dir_ee))
    finger_dir_ee = _normalize(np.cross(x_ee, y_hat_ee))
    E_ee = np.stack([x_ee, y_hat_ee, finger_dir_ee], axis=1)

    # Hand box corners in EE frame (conservative: use the flat half extents).
    hand_half = ee_skelton._flat_hand_half_extents(skeleton)
    hand_signs = np.array(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    hand_corners_local = hand_half[None, :] * hand_signs
    hand_corners_ee = (
        skeleton.hand_box_center_ee[None, :]
        + hand_corners_local @ skeleton.hand_box_rotation_ee.T
    )

    # COACD convex-part equations (world frame).
    has_object_eqs = (
        handle_convex_equations is not None
        and np.asarray(handle_convex_equations).size > 0
    )
    object_eqs = None
    if has_object_eqs:
        object_eqs = np.asarray(handle_convex_equations, dtype=np.float64)
        if object_eqs.ndim == 2:
            object_eqs = object_eqs[None, :, :]
        has_object_eqs = object_eqs.shape[0] > 0

    # Sample points on both fingers (in EE frame) at the default opening.
    seg_alphas = np.linspace(0.0, 1.0, max(int(seg_samples), 2), dtype=np.float64)
    left_samples_ee = (
        left_seg0[0][None, :] * (1.0 - seg_alphas[:, None])
        + left_seg0[1][None, :] * seg_alphas[:, None]
    )
    right_samples_ee = (
        right_seg0[0][None, :] * (1.0 - seg_alphas[:, None])
        + right_seg0[1][None, :] * seg_alphas[:, None]
    )
    all_samples_ee = np.concatenate(
        [left_samples_ee, right_samples_ee, hand_corners_ee], axis=0
    )
    all_sample_radii = np.concatenate(
        [
            np.full(left_samples_ee.shape[0], finger_radius, dtype=np.float64),
            np.full(right_samples_ee.shape[0], finger_radius, dtype=np.float64),
            np.zeros(hand_corners_ee.shape[0], dtype=np.float64),
        ],
        axis=0,
    )
    n_samples = all_samples_ee.shape[0]

    # Inner-surface anchor points (in EE frame) at the default opening.
    left_mid_ee = 0.5 * (left_seg0[0] + left_seg0[1])
    right_mid_ee = 0.5 * (right_seg0[0] + right_seg0[1])
    left_inner_ee = left_mid_ee - finger_radius * y_hat_ee
    right_inner_ee = right_mid_ee + finger_radius * y_hat_ee

    # Gripper opening that matches the contact separation (closed-form guess).
    g_match = float(dist_w + 2.0 * finger_radius)
    g_match = float(np.clip(g_match, g_min, g_max))

    # Theta samples: rotate the world basis about the contact-separation axis.
    thetas = np.linspace(0.0, 2.0 * np.pi, max(1, int(theta_count)), endpoint=False)

    candidates = []
    for theta in thetas:
        # World basis: y = contact-separation, x close to the demo approach.
        demo_x_w = demo_ee_rotation @ x_ee
        x_w = demo_x_w - float(np.dot(demo_x_w, y_w)) * y_w
        if float(np.linalg.norm(x_w)) < 1e-6:
            ref = (
                np.array([1.0, 0.0, 0.0], dtype=np.float64)
                if abs(float(y_w[0])) < 0.9
                else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            )
            x_w = ref - float(np.dot(ref, y_w)) * y_w
        x_w = _normalize(x_w)
        finger_w = _normalize(np.cross(x_w, y_w))
        x_w = _normalize(np.cross(y_w, finger_w))
        W_base = np.stack([x_w, y_w, finger_w], axis=1)

        # Rotate the world basis about y_w by theta.
        omega_y_w = theta * y_w
        R_theta = (
            ee_skelton._exp_so3(omega_y_w)
            if hasattr(ee_skelton, "_exp_so3")
            else _exp_so3(omega_y_w)
        )
        W = R_theta @ W_base
        R0 = W @ E_ee.T
        if debug and theta == thetas[0]:
            _autogen_print(
                f"[grasp] dual_daqp DEBUG axes: y_w(sep)={y_w} "
                f"x_w(approach)={x_w} lift_dir=-x_w"
            )

        # Nominal EE position: place the left inner surface at the left contact
        # point (lift along -y_w so the fingers approach from outside).
        p0 = left_contact_point - R0 @ left_inner_ee

        # Precompute per-sample world-frame Jacobians (linearized around x=0).
        # sample_w(x) = p0 + R0 @ sample_ee + dx - skew(R0 @ sample_ee) @ omega
        #               + dg * R0 @ spread_ee(sample) + dsL/R * R0 @ seg_dir_ee
        r_samples_w = (R0 @ all_samples_ee.T).T  # (n_samples, 3)
        cross_rb = np.cross(r_samples_w, y_w[None, :])  # for tangent-plane rows
        spread_per_sample = np.zeros((n_samples, 3), dtype=np.float64)
        segdir_per_sample = np.zeros((n_samples, 3), dtype=np.float64)
        n_left = left_samples_ee.shape[0]
        n_right = right_samples_ee.shape[0]
        for si in range(n_samples):
            if si < n_left:
                spread_per_sample[si] = R0 @ left_spread_ee
                segdir_per_sample[si] = R0 @ (left_seg0[1] - left_seg0[0])
            elif si < n_left + n_right:
                spread_per_sample[si] = R0 @ right_spread_ee
                segdir_per_sample[si] = R0 @ (right_seg0[1] - right_seg0[0])
            # hand box corners: no spread / segdir contribution

        # Contact Jacobians for the two inner-surface anchors.
        r_left_w = R0 @ left_inner_ee
        r_right_w = R0 @ right_inner_ee
        A_contact_left = np.zeros((3, 9), dtype=np.float64)
        A_contact_right = np.zeros((3, 9), dtype=np.float64)
        A_contact_left[:, 0:3] = np.eye(3)
        A_contact_left[:, 3:6] = -_skew(r_left_w)
        A_contact_left[:, 6] = R0 @ left_spread_ee
        A_contact_left[:, 7] = R0 @ (left_seg0[1] - left_seg0[0])
        # A_contact_left[:, 8] = 0 (sR does not move the left anchor)
        A_contact_right[:, 0:3] = np.eye(3)
        A_contact_right[:, 3:6] = -_skew(r_right_w)
        A_contact_right[:, 6] = R0 @ right_spread_ee
        # A_contact_right[:, 7] = 0 (sL does not move the right anchor)
        A_contact_right[:, 8] = R0 @ (right_seg0[1] - right_seg0[0])

        # Contact targets: inner surface AT the contact point (clearance = 0).
        b_left = left_contact_point - (p0 + r_left_w)
        b_right = right_contact_point - (p0 + r_right_w)

        # Lift + gripper seeds.
        if n_lift <= 1:
            lifts = np.array([0.0], dtype=np.float64)
        else:
            lifts = np.linspace(lift_min, lift_max, n_lift, dtype=np.float64)
        if n_g <= 1:
            g_seeds = np.array([g_match], dtype=np.float64)
        else:
            g_seeds = np.linspace(
                max(g_min, g_match - 0.02),
                min(g_max, g_match + 0.02),
                n_g,
                dtype=np.float64,
            )

        for lift in lifts:
            # Apply lift along -x_w (approach axis): retract the hand along the
            # approach direction so the fingers come in from outside the handle.
            # (Previously incorrectly along -y_w = contact-separation axis, which
            #  slid one finger off its contact instead of retracting.)
            lift_dir = -x_w
            p0_lift = p0 + lift * lift_dir
            b_left_lift = b_left - lift * lift_dir
            b_right_lift = b_right - lift * lift_dir

            for g_seed in g_seeds:
                dg_seed = float(g_seed - g_default)

                # --- Build QP cost ---
                H = np.zeros((9, 9), dtype=np.float64)
                f_cost = np.zeros(9, dtype=np.float64)
                # SE(3) regularization.
                H[0:6, 0:6] += 2.0 * reg_weight * np.eye(6)
                # Gripper-opening regularization around the seed.
                H[6, 6] += 2.0 * g_weight
                f_cost[6] += -2.0 * g_weight * dg_seed
                # Keep a small quadratic contact term for tie-breaking, but
                # the actual two-finger contact is enforced below as hard
                # affine DAQP equality constraints.
                soft_contact_weight = max(1e-6, 0.01 * contact_weight)
                H += (
                    2.0
                    * soft_contact_weight
                    * (
                        A_contact_left.T @ A_contact_left
                        + A_contact_right.T @ A_contact_right
                    )
                )
                f_cost += (
                    -2.0
                    * soft_contact_weight
                    * (
                        A_contact_left.T @ b_left_lift
                        + A_contact_right.T @ b_right_lift
                    )
                )
                # Lift penalty (penalize translation along y_w).
                bw_outer = np.outer(y_w, y_w)
                H[0:3, 0:3] += (
                    2.0 * float(getattr(args, "autogen_skeleton_lift_weight", 100.0))
                ) * bw_outer
                # SPD regularization.
                H += 1e-6 * np.eye(9)

                # --- Inequality constraints g: COACD clearance ---
                # Only constrain samples that start OUTSIDE the hull (sd0 > 0):
                # a linearized half-space keeps them from driving inward past
                # `margin + radius`.  Samples that start INSIDE (sd0 < 0) are
                # intentionally skipped — a wrap grasp legitimately has the
                # non-contact hand/finger samples *inside* the handle's convex
                # hull, and pushing the QP to push them all the way out to
                # `margin+radius` left the solver infeasible (see issue: 0
                # skeleton_poses across all thetas).  The post-QP validation
                # below (lines ~1650+) still rejects any solution whose final
                # clearance is worse than `margin - penetration_tol`, so bad
                # wrap poses are filtered without destroying feasibility.
                rows_A = []
                rows_b = []
                if has_object_eqs:
                    for si in range(n_samples):
                        sample_ee = all_samples_ee[si]
                        sample_w0 = p0_lift + R0 @ sample_ee
                        pts_h = np.concatenate([sample_w0, [1.0]])  # (4,)
                        plane_vals = np.einsum("phk,k->ph", object_eqs, pts_h)  # (P, H)
                        inside_depth = plane_vals.max(axis=1)  # (P,)
                        # Only the convex parts whose current signed distance
                        # is in (0, threshold): they are outside but not yet
                        # `margin + radius` clear, so the step must not push
                        # them deeper into the part.
                        threshold = margin + all_sample_radii[si] - 1e-9
                        J_s = None
                        for part_idx in range(object_eqs.shape[0]):
                            sd0 = float(inside_depth[part_idx])
                            # sd0 > 0 means sample is currently OUTSIDE this
                            # convex part.  sd0 < 0 means it is inside — skip
                            # the constraint (wrap-legitimate), defer to the
                            # post-validation clearance check.  sd0 >=
                            # threshold means already sufficiently clear.
                            if sd0 <= 0.0 or sd0 >= threshold:
                                continue
                            plane_idx = int(np.argmax(plane_vals[part_idx]))
                            n_plane = object_eqs[part_idx, plane_idx, :3]
                            if J_s is None:
                                J_s = np.zeros((3, 9), dtype=np.float64)
                                J_s[:, 0:3] = np.eye(3)
                                J_s[:, 3:6] = -_skew(R0 @ sample_ee)
                                J_s[:, 6] = spread_per_sample[si]
                                if si < n_left:
                                    J_s[:, 7] = segdir_per_sample[si]
                                elif si < n_left + n_right:
                                    J_s[:, 8] = segdir_per_sample[si]
                            lhs = n_plane @ J_s
                            # Finger sample rows use the relaxed margin;
                            # hand-box corner rows keep the strict `margin`.
                            row_margin = (
                                margin_finger_dual if si < n_left + n_right else margin
                            )
                            rhs = row_margin + all_sample_radii[si] - sd0
                            rows_A.append(lhs)
                            rows_b.append(rhs)

                if scene_geom_ids_set:
                    # MuJoCo ray scene clearance, matching the single-contact
                    # ee_skelton convention: cast from each *non-finger* skeleton
                    # sample toward nearby scene geometry; the half-space
                    # constrains motion along the opposite clearance direction.
                    #
                    # IMPORTANT: finger samples (indices < n_left + n_right)
                    # are intentionally SKIPPED here.  In a dual-finger wrap
                    # grasp the non-contacting finger legitimately sits near
                    # the handle body / the other finger; mj_ray from those
                    # sample origins almost always returns a short hit, and
                    # translating those hits into half-space rows gives DAQP
                    # 100+ inequality constraints that are mutually
                    # inconsistent with the 6 contact equalities — the solver
                    # returns exitflag -1 without ever reaching status 1.
                    # Post-validation (lines ~1726+) re-runs the same mj_ray
                    # test and rejects truly colliding solutions, so skipping
                    # fingers in-QP is safe.
                    ray_dirs = []
                    for d in (
                        left_contact_normal,
                        right_contact_normal,
                        -left_contact_normal,
                        -right_contact_normal,
                        x_w,
                        -x_w,
                        y_w,
                        -y_w,
                    ):
                        dn = _normalize(d)
                        if not any(abs(float(np.dot(dn, e))) > 0.985 for e in ray_dirs):
                            ray_dirs.append(dn)
                    geomgroup = np.ones(6, dtype=np.uint8)
                    geomid_scratch = np.zeros(1, dtype=np.int32)
                    n_finger_samples = n_left + n_right
                    for si in range(n_finger_samples, n_samples):
                        sample_ee = all_samples_ee[si]
                        sample_w0 = p0_lift + R0 @ sample_ee
                        J_s = np.zeros((3, 9), dtype=np.float64)
                        J_s[:, 0:3] = np.eye(3)
                        J_s[:, 3:6] = -_skew(R0 @ sample_ee)
                        J_s[:, 6] = spread_per_sample[si]
                        # hand box corners (si >= n_finger_samples) have no
                        # spread/segdir contribution, which is what we want.
                        for ray_dir in ray_dirs:
                            try:
                                geomid_scratch[0] = -1
                                dist_val = mujoco.mj_ray(
                                    raw_model,
                                    raw_data,
                                    np.ascontiguousarray(sample_w0, dtype=np.float64),
                                    np.ascontiguousarray(ray_dir, dtype=np.float64),
                                    geomgroup,
                                    1,
                                    -1,
                                    geomid_scratch,
                                )
                            except Exception:
                                dist_val = -1.0
                                geomid_scratch[0] = -1
                            if (
                                float(dist_val) > 0.0
                                and int(geomid_scratch[0]) in scene_geom_ids_set
                            ):
                                clearance_dir = -ray_dir
                                rows_A.append(clearance_dir @ J_s)
                                rows_b.append(
                                    margin + all_sample_radii[si] - float(dist_val)
                                )
                                dbg_scene_rows += 1

                # Tangent-plane half-space is intentionally not used here: a
                # valid wrap grasp straddles the handle, so a single contact
                # plane would reject the opposite finger.

                if rows_A:
                    A_ineq = np.stack(rows_A, axis=0)
                    b_lower = np.asarray(rows_b, dtype=np.float64)
                else:
                    A_ineq = np.zeros((0, 9), dtype=np.float64)
                    b_lower = np.zeros(0, dtype=np.float64)

                # --- Variable bounds ---
                I_n = np.eye(9, dtype=np.float64)
                bounds_lower = np.full(9, -1e30, dtype=np.float64)
                bounds_upper = np.full(9, 1e30, dtype=np.float64)
                bounds_lower[0:3] = -motion_bound
                bounds_upper[0:3] = motion_bound
                bounds_lower[3:6] = -rot_bound
                bounds_upper[3:6] = rot_bound
                bounds_lower[6] = g_min - g_default
                bounds_upper[6] = g_max - g_default
                bounds_lower[7] = -0.5  # sL perturbation around 0.5
                bounds_upper[7] = 0.5
                bounds_lower[8] = -0.5  # sR perturbation around 0.5
                bounds_upper[8] = 0.5

                A_contact_eq = np.concatenate([A_contact_left, A_contact_right], axis=0)
                b_contact_eq = np.concatenate([b_left_lift, b_right_lift], axis=0)
                A_full = np.concatenate([A_contact_eq, A_ineq, I_n], axis=0)
                n_ineq = A_ineq.shape[0]
                bupper = np.concatenate(
                    [b_contact_eq, np.full(n_ineq, 1e30), bounds_upper]
                )
                blower = np.concatenate([b_contact_eq, b_lower, bounds_lower])
                sense = np.concatenate(
                    [
                        np.full(A_contact_eq.shape[0], 5, dtype=np.int32),
                        np.zeros(n_ineq + 9, dtype=np.int32),
                    ]
                )

                # Diagnostic: max equality residual at the unperturbed seed
                # (x = 0).  If this is already huge, the contact targets are
                # mutually inconsistent and NO choice of dx/omega/dg/sL/sR
                # can satisfy them — DAQP has no hope.
                eq_resid_0 = float(
                    np.max(
                        np.concatenate(
                            [
                                np.abs(b_left_lift),
                                np.abs(b_right_lift),
                            ]
                        )
                    )
                )
                if eq_resid_0 < dbg_eq_viol_min:
                    dbg_eq_viol_min = eq_resid_0

                # Diagnostic: minimum-norm least-squares solution of the
                # equality system A_eq @ x = b_eq, and whether it satisfies
                # the variable bounds.  If the LS solution already violates
                # bounds, the QP is genuinely infeasible (the contact pair
                # cannot be realized within the allowed dx/omega/dg/sL/sR
                # range).  If the LS solution satisfies bounds but DAQP still
                # returns -1, the issue is numerical (ill-conditioning,
                # cycling) and we need a different solver or regularization.
                A_eq = np.concatenate([A_contact_left, A_contact_right], axis=0)
                b_eq = np.concatenate([b_left_lift, b_right_lift], axis=0)
                try:
                    x_ls, _, rank_eq, s_eq = np.linalg.lstsq(A_eq, b_eq, rcond=None)
                    resid_ls = float(np.max(np.abs(A_eq @ x_ls - b_eq)))
                    bounds_lower_x = np.array(
                        [
                            -motion_bound,
                            -motion_bound,
                            -motion_bound,
                            -rot_bound,
                            -rot_bound,
                            -rot_bound,
                            g_min - g_default,
                            -0.5,
                            -0.5,
                        ],
                        dtype=np.float64,
                    )
                    bounds_upper_x = np.array(
                        [
                            motion_bound,
                            motion_bound,
                            motion_bound,
                            rot_bound,
                            rot_bound,
                            rot_bound,
                            g_max - g_default,
                            0.5,
                            0.5,
                        ],
                        dtype=np.float64,
                    )
                    ls_in_bounds = bool(
                        np.all(x_ls >= bounds_lower_x - 1e-9)
                        and np.all(x_ls <= bounds_upper_x + 1e-9)
                    )
                    ls_bound_viol = float(
                        max(
                            np.max(bounds_lower_x - x_ls),
                            np.max(x_ls - bounds_upper_x),
                            0.0,
                        )
                    )
                except Exception:
                    x_ls = np.zeros(9)
                    resid_ls = float("inf")
                    rank_eq = 0
                    s_eq = np.zeros(0)
                    ls_in_bounds = False
                    ls_bound_viol = float("inf")
                if dbg_total == 1:
                    dbg_x_ls = x_ls
                    dbg_resid_ls = resid_ls
                    dbg_rank_eq = int(rank_eq)
                    dbg_ls_in_bounds = ls_in_bounds
                    dbg_ls_bound_viol = ls_bound_viol
                    dbg_eq_cond = (
                        float(s_eq[0] / s_eq[-1])
                        if s_eq.size > 0 and float(s_eq[-1]) > 0
                        else float("inf")
                    )

                x, status, err, iters = _solve_qp_daqp(
                    H,
                    f_cost,
                    A_full,
                    bupper,
                    blower=blower,
                    sense=sense,
                    max_iter=200,
                    tol=1e-6,
                )
                dbg_total += 1
                dbg_status_hist[int(status)] = dbg_status_hist.get(int(status), 0) + 1
                if dbg_total == 1 or int(status) != 1:
                    dbg_scene_rows_min = min(
                        dbg_scene_rows_min or dbg_scene_rows, dbg_scene_rows
                    )
                dbg_scene_rows_max = max(dbg_scene_rows_max, dbg_scene_rows)

                # DAQP FALLBACK: when DAQP returns non-1 (typically -1 =
                # unsolved/cycling, which the diagnostic LS check shows is a
                # solver-side numerical issue rather than genuine
                # infeasibility — resid ≈ 0, rank 6/6, cond ~60,
                # ls_in_bounds=True), fall back to the minimum-norm LS
                # solution.  The LS solution satisfies the 6 contact
                # equalities exactly and the variable bounds by construction;
                # post-validation below still rejects solutions that violate
                # the COACD / scene-clearance rows DAQP failed to respect.
                if int(status) != 1:
                    if ls_in_bounds and resid_ls < 1e-6:
                        x = x_ls.copy()
                        if debug:
                            _autogen_print(
                                f"[dual_daqp iter] theta={float(theta):+.3f} "
                                f"lift={float(lift):.4f} g_seed={float(g_seed):.4f} "
                                f"daqp_status={int(status)} ls_fallback=1 "
                                f"resid_ls={resid_ls:.3g} n_ineq={n_ineq} "
                                f"scene_rows={dbg_scene_rows} reject_reason=ls_fallback"
                            )
                    else:
                        if debug:
                            _autogen_print(
                                f"[dual_daqp iter] theta={float(theta):+.3f} "
                                f"lift={float(lift):.4f} g_seed={float(g_seed):.4f} "
                                f"daqp_status={int(status)} n_ineq={n_ineq} "
                                f"scene_rows={dbg_scene_rows} "
                                f"eq_resid@x0={eq_resid_0:.3g} "
                                f"ls_in_bounds={ls_in_bounds} "
                                f"ls_resid={resid_ls:.3g} "
                                f"reject_reason=qp_infeasible"
                            )
                        continue

                dbg_qp_ok += 1

                dx = x[0:3]
                omega = x[3:6]
                g_opt = g_default + float(x[6])
                sL = 0.5 + float(x[7])
                sR = 0.5 + float(x[8])
                sL = float(np.clip(sL, 0.0, 1.0))
                sR = float(np.clip(sR, 0.0, 1.0))

                p_final = p0_lift + dx
                R_final = _exp_so3(omega) @ R0

                # Reconstruct finger segments at the solved opening.
                (
                    left_seg,
                    right_seg,
                    y_hat_g,
                    _,
                ) = ee_skelton._finger_segments_with_opening(skeleton, g_opt)
                left_mid_g = 0.5 * (left_seg[0] + left_seg[1])
                right_mid_g = 0.5 * (right_seg[0] + right_seg[1])
                left_inner_g = (
                    left_seg[0] + sL * (left_seg[1] - left_seg[0])
                ) - finger_radius * y_hat_g
                right_inner_g = (
                    right_seg[0] + sR * (right_seg[1] - right_seg[0])
                ) + finger_radius * y_hat_g

                # --- Post-validation ---
                # 1) Both inner surfaces must land on their contact points.
                left_reach = p_final + R_final @ left_inner_g
                right_reach = p_final + R_final @ right_inner_g
                left_err = float(np.linalg.norm(left_reach - left_contact_point))
                right_err = float(np.linalg.norm(right_reach - right_contact_point))
                if left_err < dbg_left_err_min:
                    dbg_left_err_min = left_err
                if right_err < dbg_right_err_min:
                    dbg_right_err_min = right_err
                if left_err > 1e-2 or right_err > 1e-2:
                    dbg_err_fail += 1
                    if debug:
                        _autogen_print(
                            f"[dual_daqp iter] theta={float(theta):+.3f} "
                            f"lift={float(lift):.4f} g_seed={float(g_seed):.4f} "
                            f"daqp_status={int(status)} n_ineq={n_ineq} "
                            f"scene_rows={dbg_scene_rows} "
                            f"left_err={left_err:.4g} right_err={right_err:.4g} "
                            f"reject_reason=contact_err"
                        )
                    continue

                # 2) COACD clearance for finger samples AND hand box corners.
                alphas_v = np.linspace(
                    0.0, 1.0, max(int(seg_samples), 2), dtype=np.float64
                )
                left_v = (
                    left_seg[0][None, :] * (1.0 - alphas_v[:, None])
                    + left_seg[1][None, :] * alphas_v[:, None]
                )
                right_v = (
                    right_seg[0][None, :] * (1.0 - alphas_v[:, None])
                    + right_seg[1][None, :] * alphas_v[:, None]
                )
                all_v = np.concatenate([left_v, right_v, hand_corners_ee], axis=0)
                radii_v = np.concatenate(
                    [
                        np.full(left_v.shape[0], finger_radius, dtype=np.float64),
                        np.full(right_v.shape[0], finger_radius, dtype=np.float64),
                        np.zeros(hand_corners_ee.shape[0], dtype=np.float64),
                    ],
                    axis=0,
                )
                samples_w = (R_final @ all_v.T).T + p_final[None, :]
                if has_object_eqs:
                    signed = ee_skelton._signed_distance_to_convex(
                        samples_w, object_eqs
                    )
                    clearance = signed - radii_v
                    if (
                        clearance.size
                        and float(clearance.min()) < -penetration_tol_dual
                    ):
                        if float(clearance.min()) < dbg_clear_min:
                            dbg_clear_min = float(clearance.min())
                        dbg_clear_fail += 1
                        if debug:
                            _autogen_print(
                                f"[dual_daqp iter] theta={float(theta):+.3f} "
                                f"lift={float(lift):.4f} g_seed={float(g_seed):.4f} "
                                f"daqp_status={int(status)} n_ineq={n_ineq} "
                                f"scene_rows={dbg_scene_rows} "
                                f"min_clearance={float(clearance.min()):.4g} "
                                f"tol={penetration_tol_dual:.4g} "
                                f"reject_reason=coacd_pen"
                            )
                        continue
                    if clearance.size and float(clearance.min()) < dbg_clear_min:
                        dbg_clear_min = float(clearance.min())

                # 3) Post-check the same MuJoCo ray scene clearance used to
                # build linear rows. This catches large-rotation linearization
                # misses and verifies the scene pool/ray path is active.
                if scene_geom_ids_set:
                    # (a) Enclosure test via SkeletonScenePool.check_penetration
                    #     — catches samples that landed *inside* a scene geom
                    #     (drawer panel / table above the drawer), which the
                    #     single-ray mj_ray path below cannot detect because
                    #     mj_ray from inside a geom returns the *exit*
                    #     distance (positive), not a negative depth.
                    #
                    #     Why: mj_ray on hand-box corners that already sit
                    #     inside the drawer panel returned e.g. `dist_val ≈
                    #     panel_thickness > margin + radius`, so the single-
                    #     ray check silently accepted the pose. The multi-ray
                    #     enclosure test in SkeletonScenePool.check_penetration
                    #     (opposite-axis pair with matching geom id → signed
                    #     distance) is the only reliable path.
                    scene_pool_local = getattr(
                        args, "autogen_skeleton_scene_pool", None
                    )
                    if scene_pool_local is not None:
                        # Serialize the shared worker-0 MjData access: the
                        # DAQP fan-out calls this from multiple threads and
                        # `check_penetration` mutates MjData scratch inside
                        # mj_multiRay — concurrent calls otherwise segfault.
                        with scene_pool_local._lock:
                            (
                                signed_scene,
                                hit_geoms,
                            ) = scene_pool_local.check_penetration(
                                samples_w,
                                radii_v,
                                exclude_geom_ids=(),
                                margin=margin,
                            )
                        worst_idx = int(np.argmin(signed_scene))
                        worst = float(signed_scene[worst_idx])
                        if worst < -penetration_tol:
                            dbg_scene_fail += 1
                            if debug:
                                _autogen_print(
                                    f"[grasp] dual_daqp scene-pen sample={worst_idx} "
                                    f"depth={-worst:.4g} "
                                    f"geom={scene_pool_local.geom_name(int(hit_geoms[worst_idx]))}"
                                )
                            continue
                    ray_dirs = []
                    for d in (
                        left_contact_normal,
                        right_contact_normal,
                        -left_contact_normal,
                        -right_contact_normal,
                        x_w,
                        -x_w,
                        y_w,
                        -y_w,
                    ):
                        dn = _normalize(d)
                        if not any(abs(float(np.dot(dn, e))) > 0.985 for e in ray_dirs):
                            ray_dirs.append(dn)
                    geomgroup = np.ones(6, dtype=np.uint8)
                    geomid_scratch = np.zeros(1, dtype=np.int32)
                    scene_ok = True
                    for si, sample_w in enumerate(samples_w):
                        radius_i = float(radii_v[si])
                        for ray_dir in ray_dirs:
                            try:
                                geomid_scratch[0] = -1
                                dist_val = mujoco.mj_ray(
                                    raw_model,
                                    raw_data,
                                    np.ascontiguousarray(sample_w, dtype=np.float64),
                                    np.ascontiguousarray(ray_dir, dtype=np.float64),
                                    geomgroup,
                                    1,
                                    -1,
                                    geomid_scratch,
                                )
                            except Exception:
                                dist_val = -1.0
                                geomid_scratch[0] = -1
                            if (
                                float(dist_val) > 0.0
                                and int(geomid_scratch[0]) in scene_geom_ids_set
                                and float(dist_val) < margin + radius_i - clearance_tol
                            ):
                                scene_ok = False
                                break
                        if not scene_ok:
                            break
                    if not scene_ok:
                        dbg_scene_fail += 1
                        if debug:
                            _autogen_print(
                                f"[dual_daqp iter] theta={float(theta):+.3f} "
                                f"lift={float(lift):.4f} g_seed={float(g_seed):.4f} "
                                f"daqp_status={int(status)} n_ineq={n_ineq} "
                                f"scene_rows={dbg_scene_rows} "
                                f"reject_reason=scene_pen"
                            )
                        continue

                cost = float(0.5 * x @ (H @ x) + f_cost @ x)
                if debug:
                    _autogen_print(
                        f"[dual_daqp iter] theta={float(theta):+.3f} "
                        f"lift={float(lift):.4f} g_seed={float(g_seed):.4f} "
                        f"daqp_status={int(status)} cost={cost:.4g} "
                        f"g_opt={float(g_opt):.4f} left_err={left_err:.3g} "
                        f"right_err={right_err:.3g} reject_reason=ok"
                    )
                candidates.append(
                    (
                        cost,
                        float(theta),
                        R_final,
                        p_final,
                        float(g_opt),
                        float(lift),
                        float(left_err),
                        float(right_err),
                    )
                )

    if not candidates:
        _record_dual_daqp_stats(
            args,
            {
                "total": int(dbg_total),
                "qp_ok": int(dbg_qp_ok),
                "err_fail": int(dbg_err_fail),
                "clear_fail": int(dbg_clear_fail),
                "scene_fail": int(dbg_scene_fail),
                "scene_rows": int(dbg_scene_rows),
                "candidates": 0,
                "min_left_err": float(dbg_left_err_min),
                "min_right_err": float(dbg_right_err_min),
                "min_clearance": float(dbg_clear_min),
                "daqp_status_hist": dict(dbg_status_hist),
                "eq_resid_at_x0_min": float(dbg_eq_viol_min),
                "eq_ls_resid": float(dbg_resid_ls),
                "eq_ls_in_bounds": bool(dbg_ls_in_bounds),
                "eq_ls_bound_viol": float(dbg_ls_bound_viol),
                "eq_rank": int(dbg_rank_eq),
                "eq_cond": float(dbg_eq_cond),
            },
        )
        return [] if return_candidates else None
    _record_dual_daqp_stats(
        args,
        {
            "total": int(dbg_total),
            "qp_ok": int(dbg_qp_ok),
            "err_fail": int(dbg_err_fail),
            "clear_fail": int(dbg_clear_fail),
            "scene_fail": int(dbg_scene_fail),
            "scene_rows": int(dbg_scene_rows),
            "candidates": int(len(candidates)),
            "min_left_err": float(dbg_left_err_min),
            "min_right_err": float(dbg_right_err_min),
            "min_clearance": float(dbg_clear_min),
            "daqp_status_hist": dict(dbg_status_hist),
        },
    )

    if return_candidates:
        max_candidates = int(max_candidates or len(candidates))
        min_theta_separation = max(float(min_theta_separation), 0.0)
        selected = []
        selected_ids = set()
        sorted_candidates = sorted(candidates, key=lambda item: item[0])

        def _passes_theta(candidate):
            theta = float(candidate[1])
            if min_theta_separation > 0.0:
                for existing in selected:
                    dtheta = abs(
                        ((theta - float(existing[1]) + np.pi) % (2.0 * np.pi)) - np.pi
                    )
                    if dtheta < min_theta_separation:
                        return False
            return True

        for candidate in sorted_candidates:
            if len(selected) >= max_candidates:
                break
            if not _passes_theta(candidate):
                continue
            selected.append(candidate)
            selected_ids.add(id(candidate))
        return [
            ee_skelton.SkeletonPose(
                ee_position=np.asarray(p_final, dtype=np.float64),
                ee_rotation=np.asarray(R_final, dtype=np.float64),
                contact_finger="left",
                contact_point_world=np.asarray(left_contact_point, dtype=np.float64),
                contact_normal_world=np.asarray(left_contact_normal, dtype=np.float64),
                qp_cost=float(cost),
                lift=float(_lift),
                theta=float(theta),
                gripper_opening=float(g_opt),
                contact_primitive="dual_finger_inner",
            )
            for cost, theta, R_final, p_final, g_opt, _lift, _le, _re in selected
        ]

    cost, theta, R_final, p_final, g_opt, lift, _le, _re = min(
        candidates, key=lambda item: item[0]
    )
    return ee_skelton.SkeletonPose(
        ee_position=np.asarray(p_final, dtype=np.float64),
        ee_rotation=np.asarray(R_final, dtype=np.float64),
        contact_finger="left",
        contact_point_world=np.asarray(left_contact_point, dtype=np.float64),
        contact_normal_world=np.asarray(left_contact_normal, dtype=np.float64),
        qp_cost=float(cost),
        lift=float(lift),
        theta=float(theta),
        gripper_opening=float(g_opt),
        contact_primitive="dual_finger_inner",
    )


def _exp_so3(omega):
    """Exponential map from so(3) to SO(3)."""
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + _skew(omega)
    k = omega / theta
    K = _skew(k)
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * K
        + (1.0 - np.cos(theta)) * (K @ K)
    )


def _skew(v):
    """3-vector -> 3x3 skew-symmetric matrix."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def _solve_skeleton_grasp(
    env,
    skeleton,
    pair,
    demo_ee_rotation,
    handle_convex_equations,
    scene_geom_ids,
    args,
    use_mirror=False,
):
    """Solve a single dual-contact skeleton pose for one grasp contact pair.

    Uses the DAQP-based ``_solve_dual_finger_skeleton_daqp`` so a *single* EE
    pose matches both contact points simultaneously while respecting the
    ``g`` grasp constraint (COACD clearance of finger samples AND hand box
    corners).  The closed-form ``solve_dual_finger_skeleton_pose`` only checks
    finger segments, which is why the hand box used to penetrate the handle.

    If ``use_mirror`` is True the demo EE rotation is flipped 180° about z,
    giving the second grasp orientation candidate.  Returns a list of 0–N
    ``SkeletonPose`` (a list for API compatibility with the caller).
    """
    left_point = np.asarray(pair["contact_points_world"][0], dtype=np.float64)
    right_point = np.asarray(pair["contact_points_world"][1], dtype=np.float64)
    left_normal = np.asarray(pair["normals_world"][0], dtype=np.float64)
    right_normal = np.asarray(pair["normals_world"][1], dtype=np.float64)

    rot = demo_ee_rotation
    if use_mirror:
        from scipy.spatial.transform import Rotation as _R

        rot = demo_ee_rotation @ _R.from_rotvec([0.0, 0.0, np.pi]).as_matrix().astype(
            np.float64
        )

    sp = _solve_dual_finger_skeleton_daqp(
        env,
        skeleton,
        left_point,
        left_normal,
        right_point,
        right_normal,
        demo_ee_rotation=rot,
        handle_convex_equations=handle_convex_equations,
        scene_geom_ids=scene_geom_ids,
        args=args,
    )
    return [sp] if sp is not None else []


def _solve_skeleton_grasp_all(
    env,
    skeleton,
    grasp_pairs,
    demo_ee_rotation,
    handle_convex_equations,
    scene_geom_ids,
    args,
):
    """Solve dual-finger skeleton poses for all pairs × mirrors.

    Two orientation candidates are tried per pair:

    * ``none`` — the demo EE rotation as-is.
    * ``180deg`` — the demo EE rotation flipped 180° about z, the valid
      wrap-grasp mirror from symmetry.

    When ``grasp_skeleton_parallel`` is True the independent per-(pair, mirror)
    solves are farmed out to a thread pool.  Each solve only reads ``env``
    (never mutates it) and does its own numpy work, so the calls are
    thread-safe even though the MuJoCo model object is shared.

    Returns ``[(pair_index, SkeletonPose, mirror_tag), ...]``.
    """
    mirror_angles = [("none", 0.0), ("180deg", np.pi)]

    tasks = []
    for pair_index, pair in enumerate(grasp_pairs):
        for mirror_tag, mirror_angle in mirror_angles:
            tasks.append((pair_index, pair, mirror_tag, mirror_angle))

    skeleton_poses: list[tuple[int, ee_skelton.SkeletonPose, str]] = []
    # Reset the DAQP stats aggregator for this run — one record per
    # (pair, mirror) DAQP call gets appended.
    setattr(args, "_dual_daqp_stats", [])

    def _one(pair_index, pair, mirror_tag, mirror_angle):
        from scipy.spatial.transform import Rotation as _R

        rot = demo_ee_rotation @ _R.from_rotvec(
            [0.0, 0.0, float(mirror_angle)]
        ).as_matrix().astype(np.float64)
        poses = _solve_skeleton_grasp_with_rotation(
            env,
            skeleton,
            pair,
            rot,
            handle_convex_equations,
            scene_geom_ids,
            args,
        )
        return pair_index, poses, mirror_tag

    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        _tqdm = None

    _t0 = time.perf_counter()
    parallel = bool(getattr(args, "grasp_skeleton_parallel", True))
    if parallel and len(tasks) > 1:
        max_workers = min(
            int(getattr(args, "grasp_skeleton_max_workers", 8)),
            len(tasks),
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_one, pi, pair, mtag, mang): (pi, mtag)
                for pi, pair, mtag, mang in tasks
            }
            iterator = as_completed(futures)
            if _tqdm is not None:
                iterator = _tqdm(
                    iterator,
                    total=len(futures),
                    desc=f"dual_daqp (workers={max_workers})",
                    unit="task",
                    file=sys.__stdout__,
                    dynamic_ncols=True,
                    mininterval=0.2,
                    leave=True,
                )
            for fut in iterator:
                try:
                    pair_index, poses, mirror_tag = fut.result()
                except Exception as exc:
                    pi, mtag = futures[fut]
                    _autogen_print(
                        f"[grasp] skeleton solve failed pair={pi} mirror={mtag}: {exc!r}"
                    )
                    continue
                for sp in poses:
                    skeleton_poses.append((pair_index, sp, mirror_tag))
    else:
        iterator = tasks
        if _tqdm is not None:
            iterator = _tqdm(
                iterator,
                total=len(tasks),
                desc="dual_daqp (workers=1)",
                unit="task",
                file=sys.__stdout__,
                dynamic_ncols=True,
                mininterval=0.2,
                leave=True,
            )
        for pi, pair, mtag, mang in iterator:
            try:
                pair_index, poses, mirror_tag = _one(pi, pair, mtag, mang)
            except Exception as exc:
                _autogen_print(
                    f"[grasp] skeleton solve failed pair={pi} mirror={mtag}: {exc!r}"
                )
                continue
            for sp in poses:
                skeleton_poses.append((pair_index, sp, mirror_tag))

    elapsed = time.perf_counter() - _t0
    _autogen_print_yellow(
        f"[dual_daqp] done in {elapsed:.2f}s | feasible skeleton_poses={len(skeleton_poses)} "
        f"| pairs={len(grasp_pairs)} tasks={len(tasks)}"
    )

    if bool(getattr(args, "autogen_dual_debug", False)) or bool(
        getattr(args, "debug", False)
    ):
        _print_dual_daqp_debug_summary(args)

    return skeleton_poses


def _print_dual_daqp_debug_summary(args) -> None:
    """Aggregate all per-call DAQP stats and print a yellow one-liner summary.

    Focus: WHY DAQP rejects poses — QP infeasibility, contact-target error,
    penetration into the handle COACD parts, scene penetration.  Also lists
    the worst penetration depth observed and the DAQP status histogram
    (which tells us whether the QP itself is failing vs. later validation
    stages are).
    """
    stats = list(getattr(args, "_dual_daqp_stats", []) or [])
    if not stats:
        _autogen_print_yellow(
            "[dual_daqp DEBUG] no stats recorded — DAQP was never called."
        )
        return
    tot = sum(int(s.get("total", 0)) for s in stats)
    qp_ok = sum(int(s.get("qp_ok", 0)) for s in stats)
    err_fail = sum(int(s.get("err_fail", 0)) for s in stats)
    clear_fail = sum(int(s.get("clear_fail", 0)) for s in stats)
    scene_fail = sum(int(s.get("scene_fail", 0)) for s in stats)
    accepted = sum(int(s.get("candidates", 0)) for s in stats)
    qp_infeasible = max(0, tot - qp_ok)
    finite_clearances = [
        float(s.get("min_clearance", float("inf")))
        for s in stats
        if np.isfinite(float(s.get("min_clearance", float("inf"))))
    ]
    worst_pen = -min(finite_clearances) if finite_clearances else 0.0
    finite_left = [
        float(s.get("min_left_err", float("inf")))
        for s in stats
        if np.isfinite(float(s.get("min_left_err", float("inf"))))
    ]
    finite_right = [
        float(s.get("min_right_err", float("inf")))
        for s in stats
        if np.isfinite(float(s.get("min_right_err", float("inf"))))
    ]
    min_left = min(finite_left) if finite_left else float("nan")
    min_right = min(finite_right) if finite_right else float("nan")
    status_hist: dict[int, int] = {}
    for s in stats:
        for k, v in (s.get("daqp_status_hist") or {}).items():
            status_hist[int(k)] = status_hist.get(int(k), 0) + int(v)
    penetration_tol_dual = float(
        getattr(args, "autogen_dual_object_penetration_tol", 0.005)
    )
    has_penetration = worst_pen > penetration_tol_dual
    top_reason = max(
        [
            ("qp_infeasible", qp_infeasible),
            ("contact_err", err_fail),
            ("coacd_penetration", clear_fail),
            ("scene_penetration", scene_fail),
        ],
        key=lambda kv: kv[1],
    )[0]
    _autogen_print_yellow(
        f"[dual_daqp DEBUG] total_iters={tot} accepted={accepted} "
        f"qp_infeasible={qp_infeasible} contact_err={err_fail} "
        f"coacd_pen={clear_fail} scene_pen={scene_fail} top_reason={top_reason}"
    )
    _autogen_print_yellow(
        f"[dual_daqp DEBUG] has_penetration={has_penetration} "
        f"worst_penetration_depth={worst_pen:.4g}m "
        f"(tol={penetration_tol_dual:.4g}m) "
        f"min_left_err={min_left:.4g} min_right_err={min_right:.4g} "
        f"daqp_status_hist={status_hist}"
    )
    # Which link penetrated? The dual-DAQP scene-pool check
    # (SkeletonScenePool.check_penetration) already prints "geom=<name>" per
    # violation when args.autogen_dual_debug is True, so we surface that as
    # a hint.
    _autogen_print_yellow(
        "[dual_daqp DEBUG] penetrated links (if any) are logged inline as "
        "'[grasp] dual_daqp scene-pen ... geom=<name>' lines above."
    )


def _solve_skeleton_grasp_with_rotation(
    env,
    skeleton,
    pair,
    demo_ee_rotation,
    handle_convex_equations,
    scene_geom_ids,
    args,
):
    """Thin wrapper that calls the DAQP solver with a pre-rotated demo rotation."""
    left_point = np.asarray(pair["contact_points_world"][0], dtype=np.float64)
    right_point = np.asarray(pair["contact_points_world"][1], dtype=np.float64)
    left_normal = np.asarray(pair["normals_world"][0], dtype=np.float64)
    right_normal = np.asarray(pair["normals_world"][1], dtype=np.float64)
    max_candidates = int(getattr(args, "autogen_dual_max_candidates", 0)) or None
    min_theta_sep = float(getattr(args, "autogen_dual_min_theta_separation", 0.0))
    scene_pool = getattr(args, "autogen_skeleton_scene_pool", None)
    if scene_pool is not None:
        with scene_pool.borrow() as raw_model_data:
            poses = _solve_dual_finger_skeleton_daqp(
                env,
                skeleton,
                left_point,
                left_normal,
                right_point,
                right_normal,
                demo_ee_rotation=np.asarray(demo_ee_rotation, dtype=np.float64),
                handle_convex_equations=handle_convex_equations,
                scene_geom_ids=scene_geom_ids,
                args=args,
                return_candidates=True,
                max_candidates=max_candidates,
                min_theta_separation=min_theta_sep,
                raw_model_data=raw_model_data,
            )
    else:
        poses = _solve_dual_finger_skeleton_daqp(
            env,
            skeleton,
            left_point,
            left_normal,
            right_point,
            right_normal,
            demo_ee_rotation=np.asarray(demo_ee_rotation, dtype=np.float64),
            handle_convex_equations=handle_convex_equations,
            scene_geom_ids=scene_geom_ids,
            args=args,
            return_candidates=True,
            max_candidates=max_candidates,
            min_theta_separation=min_theta_sep,
        )
    return list(poses) if poses else []


def _robot_model_q_with_arm(env, robot_model, arm_joint_names, arm_q):
    q_robot = close_demo._current_robot_model_q(env, robot_model)
    for joint_name, value in zip(arm_joint_names, np.asarray(arm_q).reshape(-1)):
        if close_demo._mj_has_name(robot_model, "joint", joint_name):
            address = int(robot_model.joint(joint_name).qposadr[0])
            q_robot[address] = float(value)
    return q_robot


def _skeleton_pregrasping_gripper_opening(skeleton_pose, args):
    g_min = float(getattr(args, "autogen_skeleton_gripper_min", 0.005))
    g_max = float(
        getattr(
            args, "autogen_skeleton_gripper_max", ee_skelton.PANDA_MAX_GRIPPER_OPENING
        )
    )
    g_target = float(
        getattr(
            skeleton_pose,
            "gripper_opening",
            getattr(
                args,
                "autogen_skeleton_gripper_default",
                ee_skelton.PANDA_DEFAULT_GRIPPER_OPENING,
            ),
        )
    )
    return float(np.clip(g_target, g_min, g_max))


def _annotate_pregrasping_result(result, skeleton_pose, seed_arm_q, args):
    arm_q = np.asarray(result.arm_q, dtype=np.float64).reshape(7)
    seed_arm_q = np.asarray(seed_arm_q, dtype=np.float64).reshape(7)
    g_target = float(
        getattr(
            skeleton_pose,
            "gripper_opening",
            ee_skelton.PANDA_DEFAULT_GRIPPER_OPENING,
        )
    )
    g = _skeleton_pregrasping_gripper_opening(skeleton_pose, args)
    g_error = abs(float(g) - float(g_target))
    posture_error = float(np.linalg.norm(arm_q - seed_arm_q))
    cost = (
        float(result.position_error) * float(getattr(args, "mink_position_cost", 1.0))
        + float(result.rotation_error)
        * float(getattr(args, "mink_orientation_cost", 1.0))
        + max(float(result.max_penetration), 0.0)
        * float(getattr(args, "mink_pregrasping_penetration_cost", 1000.0))
        + g_error * float(getattr(args, "mink_pregrasping_gripper_cost", 10.0))
        + posture_error * float(getattr(args, "mink_pregrasping_posture_cost", 0.02))
    )
    return SimpleNamespace(
        arm_q=np.asarray(result.arm_q, dtype=np.float64).copy(),
        robot_q=np.asarray(result.robot_q, dtype=np.float64).copy(),
        target_position_world=np.asarray(
            result.target_position_world, dtype=np.float64
        ).copy(),
        target_rotation_world=np.asarray(
            result.target_rotation_world, dtype=np.float64
        ).copy(),
        actual_position_world=np.asarray(
            result.actual_position_world, dtype=np.float64
        ).copy(),
        actual_rotation_world=np.asarray(
            result.actual_rotation_world, dtype=np.float64
        ).copy(),
        position_error=float(result.position_error),
        rotation_error=float(result.rotation_error),
        max_penetration=float(result.max_penetration),
        collision_free=bool(result.collision_free),
        collision_reason=str(result.collision_reason),
        retreat_distance=float(result.retreat_distance),
        attempts=tuple(result.attempts),
        gripper_opening=float(g),
        target_gripper_opening=float(g_target),
        gripper_error=float(g_error),
        control_q8=np.concatenate([arm_q, np.array([g], dtype=np.float64)]),
        pregrasping_cost=float(cost),
    )


def solve_skeleton_pregrasping_q(
    env,
    *,
    surface,
    robot_state,
    robot_model,
    arm_joint_names,
    frame_name,
    skeleton_pose,
    seed_arm_q,
    args,
    max_workers=None,
):
    """Solve one 8D pre-grasping control: 7 arm joints + gripper distance."""
    arm_joint_names = tuple(arm_joint_names)
    q_start = _robot_model_q_with_arm(env, robot_model, arm_joint_names, seed_arm_q)
    q_posture = _robot_model_q_with_arm(
        env,
        robot_model,
        arm_joint_names,
        np.asarray(robot_state["q"], dtype=np.float64),
    )
    posture_cost = close_demo._make_mink_posture_cost(
        robot_model, arm_joint_names, args
    )

    def _collision_checker(q_arm):
        return _check_arm_q_collision_for_surface_base(
            env,
            surface,
            arm_joint_names,
            q_arm,
            close_demo._drawer_joint_value(env),
            set_arm_q=close_demo._set_env_arm_q,
            set_drawer_joint_value=close_demo._set_drawer_joint_value,
            allowed_ee_geom_name=None,
            penetration_tolerance=float(args.mink_collision_penetration_tolerance),
            collision_scope="arm",
        )

    result = mink_q.solve_skeleton_precontact_q_parallel(
        env,
        robot_model=robot_model,
        arm_joint_names=arm_joint_names,
        frame_name=frame_name,
        skeleton_pose=skeleton_pose,
        q_start=q_start,
        q_posture=q_posture,
        posture_cost=posture_cost,
        args=args,
        retreat_direction_world=np.asarray(
            getattr(skeleton_pose, "contact_normal_world", np.array([1.0, 0.0, 0.0])),
            dtype=np.float64,
        ),
        scene_collision_checker=_collision_checker,
        max_workers=max_workers,
    )
    return _annotate_pregrasping_result(result, skeleton_pose, seed_arm_q, args)


def solve_skeleton_pregrasping_q_batch(
    env,
    *,
    surface,
    robot_state,
    robot_model,
    arm_joint_names,
    frame_name,
    pose_entries,
    seed_arm_q,
    args,
    max_workers=None,
    progress_callback=None,
):
    arm_joint_names = tuple(arm_joint_names)
    q_start = _robot_model_q_with_arm(env, robot_model, arm_joint_names, seed_arm_q)
    q_posture = _robot_model_q_with_arm(
        env,
        robot_model,
        arm_joint_names,
        np.asarray(robot_state["q"], dtype=np.float64),
    )
    posture_cost = close_demo._make_mink_posture_cost(
        robot_model, arm_joint_names, args
    )

    def _collision_checker(q_arm):
        return _check_arm_q_collision_for_surface_base(
            env,
            surface,
            arm_joint_names,
            q_arm,
            close_demo._drawer_joint_value(env),
            set_arm_q=close_demo._set_env_arm_q,
            set_drawer_joint_value=close_demo._set_drawer_joint_value,
            allowed_ee_geom_name=None,
            penetration_tolerance=float(args.mink_collision_penetration_tolerance),
            collision_scope="arm",
        )

    raw_results = mink_q.solve_skeleton_precontact_q_batch(
        env,
        robot_model=robot_model,
        arm_joint_names=arm_joint_names,
        frame_name=frame_name,
        pose_entries=pose_entries,
        q_start=q_start,
        q_posture=q_posture,
        posture_cost=posture_cost,
        args=args,
        scene_collision_checker=_collision_checker,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )
    pose_by_index = {int(i): sp for i, sp, _normal in pose_entries}
    return {
        int(i): _annotate_pregrasping_result(
            res, pose_by_index[int(i)], seed_arm_q, args
        )
        for i, res in raw_results.items()
    }


def _solve_grasp_pregrasping_mink(
    env,
    surface,
    skeleton_poses,
    robot_state,
    demonstration_seed,
    frame_name,
    mesh_path,
    obj_pos,
    obj_quat,
    obj_scale,
    args,
    started_time,
):
    rng = np.random.default_rng(int(args.seed) + 29003)
    order = np.arange(len(skeleton_poses), dtype=np.int64)
    rng.shuffle(order)
    max_attempts = min(
        int(getattr(args, "autogen_mink_max_attempts", len(skeleton_poses))),
        len(skeleton_poses),
    )
    selected_order = order[:max_attempts]
    robot_model = env.robots[0].robot_model.mujoco_model
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    seed_arm_q = np.asarray(demonstration_seed.arm_q, dtype=np.float64).reshape(7)
    pose_entries = []
    for pose_index in selected_order:
        _candidate_index, sp, _mirror_tag = skeleton_poses[int(pose_index)]
        retreat_normal = np.asarray(
            getattr(sp, "contact_normal_world", np.array([1.0, 0.0, 0.0])),
            dtype=np.float64,
        ).reshape(3)
        pose_entries.append((int(pose_index), sp, retreat_normal))

    _autogen_print(
        f"[grasp] pregrasp mink batch poses={len(pose_entries)} "
        f"workers={int(getattr(args, 'autogen_mink_parallel_workers', 1))} "
        "control_dim=8"
    )
    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        _tqdm = None
    try:
        _mink_multipliers = mink_q._parse_multipliers(
            getattr(args, "mink_q_retreat_distance_multipliers", None)
        )
        _multiplier_count = max(len(_mink_multipliers), 1)
    except Exception:
        _multiplier_count = 1
    total_jobs = len(pose_entries) * _multiplier_count
    pbar = (
        _tqdm(
            total=total_jobs,
            desc=f"pregrasp mink (workers={int(getattr(args, 'autogen_mink_parallel_workers', 1))})",
            unit="job",
            file=sys.__stdout__,
            dynamic_ncols=True,
            mininterval=0.2,
            leave=True,
        )
        if _tqdm is not None
        else None
    )

    def _mink_progress(n):
        if pbar is not None:
            pbar.update(int(n))

    try:
        results = solve_skeleton_pregrasping_q_batch(
            env,
            surface=surface,
            robot_state=robot_state,
            robot_model=robot_model,
            arm_joint_names=arm_joint_names,
            frame_name=frame_name,
            pose_entries=pose_entries,
            seed_arm_q=seed_arm_q,
            args=args,
            max_workers=int(getattr(args, "autogen_mink_parallel_workers", 1)),
            progress_callback=_mink_progress,
        )
    finally:
        if pbar is not None:
            pbar.close()

    pos_tol = float(
        getattr(
            args,
            "mink_q_position_tolerance",
            getattr(args, "mink_position_tolerance", 0.01),
        )
    )
    pen_tol = float(getattr(args, "mink_collision_penetration_tolerance", 0.001))
    grasp_accept_score = float(getattr(args, "grasp_accept_score", 0.15))
    raw_data_local = getattr(env.sim.data, "_data", env.sim.data)
    site_id_local = int(env.sim.model.site_name2id(frame_name))
    qpos_outer_saved = env.sim.data.qpos.copy()
    qvel_outer_saved = env.sim.data.qvel.copy()
    drawer_q_now = float(close_demo._drawer_joint_value(env))
    best = None
    successful_precontact_q = 0
    ranked = sorted(
        ((pose_index, results[int(pose_index)]) for pose_index in results),
        key=lambda item: float(getattr(item[1], "pregrasping_cost", float("inf"))),
    )
    pull_world_dir = _normalize(getattr(surface, "pull_world", (1.0, 0.0, 0.0)))

    _grasp_rollout_cache = None
    pregrasp_candidates = []  # list of (q_arm, g_opening, rollout_score, is_accepted)
    for pose_index, result in ranked:
        candidate_index, sp, _mirror_tag = skeleton_poses[int(pose_index)]
        if not bool(result.collision_free):
            continue
        if float(result.position_error) > pos_tol:
            continue
        if float(result.max_penetration) > pen_tol:
            continue
        q_best = np.asarray(result.arm_q, dtype=np.float64).reshape(7)
        g_best = float(
            getattr(
                result,
                "gripper_opening",
                _skeleton_pregrasping_gripper_opening(sp, args),
            )
        )
        target_pos = np.asarray(result.target_position_world, dtype=np.float64).reshape(
            3
        )
        target_rot = np.asarray(result.target_rotation_world, dtype=np.float64).reshape(
            3, 3
        )
        rollout_score = float("-inf")
        rollout_slip = float("inf")
        try:
            from robocasa.demos.rollout_grasp import GraspRollout, GraspRolloutConfig

            if _grasp_rollout_cache is None:
                drawer_obj = getattr(env, "drawer", None)
                object_body_id = -1
                if drawer_obj is not None:
                    try:
                        object_body_id = int(
                            env.sim.model.body_name2id(drawer_obj.root_body_name)
                        )
                    except Exception:
                        object_body_id = -1
                _target_geom_ids_dbg = _target_geom_ids(env, surface)
                _handle_world_dbg = (
                    np.asarray(
                        env.sim.data.geom_xpos[_target_geom_ids_dbg[0]],
                        dtype=np.float64,
                    )
                    if _target_geom_ids_dbg
                    else np.zeros(3)
                )
                _autogen_print(
                    f"[grasp/build] obj_pos_arg={np.asarray(obj_pos)} "
                    f"surface.center_world={np.asarray(surface.center_world)} "
                    f"handle_geom_xpos={_handle_world_dbg} "
                    f"target_pos={target_pos} "
                    f"drawer_root_xpos={np.asarray(env.sim.data.body_xpos[object_body_id]) if object_body_id >= 0 else None}"
                )
                _grasp_rollout_cache = GraspRollout(
                    env,
                    object_body_id=object_body_id,
                    ee_site_name=frame_name,
                    config=GraspRolloutConfig(
                        horizon_steps=int(getattr(args, "grasp_rollout_steps", 15)),
                        gripper_closed_opening=float(
                            getattr(args, "grasp_closed_opening", 0.005)
                        ),
                        drag_steps=int(getattr(args, "grasp_rollout_drag_steps", 20)),
                        drag_distance=float(
                            getattr(args, "grasp_rollout_drag_distance", 0.02)
                        ),
                    ),
                    mesh_path=mesh_path,
                    obj_pos=np.asarray(obj_pos, dtype=np.float64),
                    obj_scale=(
                        np.asarray(obj_scale, dtype=np.float64)
                        if obj_scale is not None
                        else np.ones(3, dtype=np.float64)
                    ),
                    obj_quat=np.asarray(obj_quat, dtype=np.float64),
                )
            rollout = _grasp_rollout_cache
            r = rollout.run(
                target_pos, target_rot, g_best, drag_direction=pull_world_dir
            )
            rollout_score = float(r.score_normalized)
            rollout_slip = float(r.slip_distance)
            _autogen_print(
                f"[grasp] rollout pair={candidate_index} "
                f"score={rollout_score:+.4f} slip={rollout_slip:.4f} "
                f"progress={float(r.score_progress_sum):+.4g} "
                f"late={float(r.score_late_min_bonus):+.4g} "
                f"rebound={float(r.score_rebound):+.4g} "
                f"pen={float(r.max_penetration):.6f}"
            )
        except Exception as exc:
            _autogen_print(f"[grasp] rollout error: {exc!r}")

        successful_precontact_q += 1
        pregrasp_candidates.append(
            (q_best.copy(), float(g_best), float(rollout_score), False)
        )
        if best is None or rollout_score > best[0]:
            best = (rollout_score, result, int(candidate_index))
        if rollout_score < grasp_accept_score:
            continue

        try:
            env.sim.data.qpos[:] = qpos_outer_saved
            env.sim.data.qvel[:] = qvel_outer_saved
            close_demo._set_env_arm_q(env, arm_joint_names, q_best)
            close_demo._set_drawer_joint_value(env, drawer_q_now)
            env.sim.forward()
            actual_pos = np.asarray(
                raw_data_local.site_xpos[site_id_local], dtype=np.float64
            ).copy()
            actual_rot = (
                np.asarray(raw_data_local.site_xmat[site_id_local], dtype=np.float64)
                .reshape(3, 3)
                .copy()
            )
        finally:
            env.sim.data.qpos[:] = qpos_outer_saved
            env.sim.data.qvel[:] = qvel_outer_saved
            env.sim.forward()

        solution = mink_solver.PreContactMinkSolution(
            arm_q=q_best,
            target_position_world=target_pos,
            target_rotation_world=target_rot,
            actual_position_world=actual_pos,
            actual_rotation_world=actual_rot,
            position_error=float(result.position_error),
            rotation_error=float(result.rotation_error),
            collision_free=True,
            collision_reason="",
        )
        solution.gripper_opening = float(g_best)
        solution.control_q8 = np.asarray(result.control_q8, dtype=np.float64).copy()
        solution.force_closure_cost = float(-rollout_score)
        solution.rollout_score = float(rollout_score)
        solution.rollout_slip = float(rollout_slip)
        _autogen_print(
            f"[grasp] accept pregrasp_mink score={rollout_score:+.4f} "
            f"slip={rollout_slip:.4f} "
            f"pair={candidate_index} successful_precontact_q={successful_precontact_q} "
            f"control_dim=8 t={time.perf_counter() - started_time:.3f}s"
        )
        if pregrasp_candidates:
            _last = pregrasp_candidates[-1]
            pregrasp_candidates[-1] = (_last[0], _last[1], _last[2], True)
        try:
            _visualize_grasp_precontact_ghosts_popup(
                env,
                arm_joint_names,
                pregrasp_candidates,
                float(drawer_q_now),
                target_pos,
                args,
            )
        except Exception as exc:
            _autogen_print(f"[grasp] precontact ghost popup error: {exc!r}")
        return solution, int(candidate_index)

    _autogen_print(
        f"[grasp] pregrasp_mink done attempts={max_attempts} "
        f"successful_precontact_q={successful_precontact_q} "
        f"best_score={best[0] if best is not None else float('-inf'):+.4f} "
        f"t={time.perf_counter() - started_time:.3f}s"
    )
    if best is not None and not bool(getattr(args, "require_mink_precontact", True)):
        rollout_score, result, candidate_index = best
        solution = mink_solver.PreContactMinkSolution(
            arm_q=np.asarray(result.arm_q, dtype=np.float64).reshape(7),
            target_position_world=np.asarray(
                result.target_position_world, dtype=np.float64
            ),
            target_rotation_world=np.asarray(
                result.target_rotation_world, dtype=np.float64
            ),
            actual_position_world=np.asarray(
                result.actual_position_world, dtype=np.float64
            ),
            actual_rotation_world=np.asarray(
                result.actual_rotation_world, dtype=np.float64
            ),
            position_error=float(result.position_error),
            rotation_error=float(result.rotation_error),
            collision_free=True,
            collision_reason="grasp_best_effort",
        )
        solution.gripper_opening = float(getattr(result, "gripper_opening", 0.0))
        solution.control_q8 = np.asarray(result.control_q8, dtype=np.float64).copy()
        solution.force_closure_cost = float(-rollout_score)
        solution.rollout_score = float(rollout_score)
        return solution, int(candidate_index)

    raise RuntimeError(
        "Grasp pregrasp mink found no collision-free 8D control with "
        f"rollout score >= {grasp_accept_score}. attempts={max_attempts} "
        f"successful_q={successful_precontact_q}"
    )


def _solve_grasp_precontact_autogen(
    env, surface, candidates, demonstration_seed, robot_state, args
):
    """Grasp-mode pre-contact solver — MIQP + dual-finger skeleton + MPPI + rollout.

    Pipeline (mirrors ``_solve_mink_precontact_autogen`` where it makes sense):

      1. build / reuse the COACD handle mesh in the object frame (shared by
         MIQP and the rollout — single geometry, no double sampling)
      2. MIQP → up to ``grasp_num_pairs`` ranked contact pairs
      3. DAQP-based dual-finger skeleton solve per (pair × theta × mirror),
         with ``g`` = linearized COACD clearance for finger segments AND hand
         box corners, in parallel
      4. MPPI IK for pre-contact q, then a gripper-closing rollout that scores
         ``force_closure_cost`` (requires the same mesh + object pose)

    Returns ``(PreContactMinkSolution, pair_index)``.
    """
    mink_started = time.perf_counter()
    successful_precontact_q = 0
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    feasible_cache = getattr(args, "_autogen_feasible_cache", None)
    if feasible_cache is None:
        _, _, feasible_cache = _build_autogen_contact_candidates(
            env, surface, 0.01, args
        )
        args._autogen_feasible_cache = feasible_cache
    handle_convex_equations = getattr(feasible_cache, "handle_convex_equations", None)
    if handle_convex_equations is None:
        handle_convex_equations = getattr(
            args, "_autogen_handle_convex_equations", None
        )

    skeleton = ee_skelton.build_panda_skeleton(env, frame_name)

    # ---- 2/3. MIQP: compute feasible contact pairs on the SHARED mesh ----
    debug_mode = bool(getattr(args, "debug", False))
    num_pairs = int(getattr(args, "grasp_num_pairs", 512))
    min_pairs = int(getattr(args, "grasp_min_pairs", 16 if debug_mode else 256))
    sample_budget = int(
        getattr(args, "grasp_miqp_sample_budget", 32 if debug_mode else 120)
    )
    mesh_path = getattr(feasible_cache, "_grasp_handle_mesh_path", None)
    if mesh_path is None:
        mesh_path = getattr(args, "_autogen_grasp_handle_mesh_path", None)
    obj_pos = getattr(feasible_cache, "_grasp_obj_pos", None)
    obj_quat = getattr(feasible_cache, "_grasp_obj_quat_wxyz", None)
    obj_scale = getattr(feasible_cache, "_grasp_obj_scale", None)

    grasp_pairs = []
    if mesh_path is not None and obj_pos is not None and obj_quat is not None:
        try:
            from robocasa.demos.example_code.grasping.miqp_grasping import (
                solve_grasping_contact_pairs,
            )

            grasp_pairs = solve_grasping_contact_pairs(
                mesh_path=mesh_path,
                obj_pos=np.asarray(obj_pos, dtype=np.float64),
                obj_quat=np.asarray(obj_quat, dtype=np.float64),
                obj_scale=np.asarray(
                    obj_scale if obj_scale is not None else np.ones(3),
                    dtype=np.float64,
                ),
                num_pairs=num_pairs,
                min_pairs=min_pairs,
                sample_budget=sample_budget,
                verbose=True,
                debug=debug_mode,
            )
        except Exception as exc:
            _autogen_print(f"[grasp] MIQP failed: {exc!r}")
    _autogen_print(f"[grasp] miqp_pairs={len(grasp_pairs)}")

    if not grasp_pairs:
        raise RuntimeError("Grasp MIQP found no feasible contact pairs on the handle.")

    _autogen_print(
        f"[grasp] contact-pair popup: about to open (pairs={len(grasp_pairs)} "
        f"enabled={bool(getattr(args, 'autogen_visualize_contact_pairs', True))})"
    )
    try:
        _visualize_contact_pairs_popup(env, grasp_pairs, args)
        _autogen_print("[grasp] contact-pair popup: closed by user")
    except Exception as exc:
        _autogen_print(f"[grasp] contact-pair popup error: {exc!r}")

    # Diagnostic: report contact-pair separations vs gripper opening bounds so a
    # near-zero skeleton-pose count is interpretable (wide handle → g > g_max).
    g_max = float(
        getattr(
            args, "autogen_skeleton_gripper_max", ee_skelton.PANDA_MAX_GRIPPER_OPENING
        )
    )
    finger_radius = float(getattr(args, "autogen_skeleton_finger_radius", 0.004))
    seps = []
    for pair in grasp_pairs:
        p0 = np.asarray(pair["contact_points_world"][0], dtype=np.float64)
        p1 = np.asarray(pair["contact_points_world"][1], dtype=np.float64)
        seps.append(float(np.linalg.norm(p0 - p1)))
    if seps:
        seps_arr = np.asarray(seps, dtype=np.float64)
        _autogen_print(
            f"[grasp] pair_separation "
            f"min={seps_arr.min():.4f} med={np.median(seps_arr):.4f} "
            f"max={seps_arr.max():.4f} g_max={g_max:.4f} "
            f"fit_count={int(np.count_nonzero(seps_arr + 2 * finger_radius <= g_max))}"
            f"/{len(seps)}"
        )

    # ---- skeleton poses: dual-finger inner-face + 180° mirror (parallel) ----
    demo_ee_rotation = np.asarray(
        demonstration_seed.projected_ee_rotation_world, dtype=np.float64
    ).reshape(3, 3)
    if getattr(args, "autogen_skeleton_scene_pool", None) is None:
        try:
            from robocasa.demos.skelton_scene import SkeletonScenePool

            pool_workers = min(
                max(1, int(getattr(args, "grasp_skeleton_max_workers", 8))),
                max(1, len(grasp_pairs) * 3),
            )
            args.autogen_skeleton_scene_pool = SkeletonScenePool.from_env(
                env, num_workers=pool_workers
            )
        except Exception as exc:
            _autogen_print(f"[grasp] skeleton_scene_pool unavailable: {exc!r}")
            args.autogen_skeleton_scene_pool = None
    scene_pool = getattr(args, "autogen_skeleton_scene_pool", None)
    if scene_pool is not None:
        try:
            scene_pool.reset()
            _autogen_print(
                f"[grasp] skeleton_scene_pool=on workers={scene_pool.num_workers}"
            )
        except Exception as exc:
            _autogen_print(f"[grasp] skeleton_scene_pool reset failed: {exc!r}")
            args.autogen_skeleton_scene_pool = None
    skeleton_poses = _solve_skeleton_grasp_all(
        env,
        skeleton,
        grasp_pairs,
        demo_ee_rotation,
        handle_convex_equations,
        tuple(gid for gid in _iter_scene_geom_ids(env, surface)),
        args,
    )
    from collections import Counter

    # Downsample: if the DAQP fleet returned more feasible poses than
    # `autogen_grasp_max_skeleton_poses` (default 512), keep the best 512
    # by DAQP cost (SkeletonPose.qp_cost).  Silently keeping >1000 poses
    # made the downstream MPPI loop hang for minutes with no user feedback.
    max_poses = int(getattr(args, "autogen_grasp_max_skeleton_poses", 512))
    if max_poses > 0 and len(skeleton_poses) > max_poses:
        original_count = len(skeleton_poses)
        skeleton_poses = sorted(
            skeleton_poses, key=lambda item: float(item[1].qp_cost)
        )[:max_poses]
        _autogen_print_yellow(
            f"[dual_daqp] downsampled skeleton_poses {original_count} -> "
            f"{len(skeleton_poses)} (autogen_grasp_max_skeleton_poses={max_poses})"
        )
    mirror_counts = Counter(mtag for _, _, mtag in skeleton_poses)
    _autogen_print(
        f"[grasp] skeleton_poses={len(skeleton_poses)} "
        f"mirror_dist={dict(mirror_counts)}"
    )
    # Skeleton-pose popup for the grasp path. This runs only after the DAQP
    # tqdm has reached 100%, so the viewer shows the completed feasible set.
    if bool(getattr(args, "autogen_visualize_grasp_skeleton_poses", False)):
        _viz_skeleton_backup = getattr(args, "autogen_visualize_skeleton_poses", True)
        try:
            args.autogen_visualize_skeleton_poses = True
            ee_skelton.visualize_skeleton_poses(
                env,
                frame_name,
                skeleton,
                [sp for _, sp, _ in skeleton_poses],
                args,
            )
        except Exception as exc:
            _autogen_print(f"[grasp] skeleton_poses popup error: {exc!r}")
        finally:
            args.autogen_visualize_skeleton_poses = _viz_skeleton_backup
    if not skeleton_poses:
        raise RuntimeError(
            "Grasp dual-finger DAQP skeleton solver found no collision-free "
            "pose for any contact pair. Try increasing `autogen_dual_theta_count` "
            "`autogen_dual_n_lift`, or `autogen_dual_n_g`."
        )

    grasp_solver = str(getattr(args, "grasp_precontact_solver", "mink")).lower()
    if grasp_solver in {"mink", "auto", "pregrasp", "pregrasping"}:
        return _solve_grasp_pregrasping_mink(
            env,
            surface,
            skeleton_poses,
            robot_state,
            demonstration_seed,
            frame_name,
            mesh_path,
            obj_pos,
            obj_quat,
            obj_scale,
            args,
            mink_started,
        )

    # ---- MPPI for pre-contact q + grasp rollout ----
    rng = np.random.default_rng(int(args.seed) + 29003)
    order = np.arange(len(skeleton_poses), dtype=np.int64)
    rng.shuffle(order)
    max_attempts = min(
        int(getattr(args, "autogen_mink_max_attempts", len(skeleton_poses))),
        len(skeleton_poses),
    )
    pos_tol = float(getattr(args, "mink_position_tolerance", 0.01))
    pen_tol = float(getattr(args, "mink_collision_penetration_tolerance", 0.001))
    demo_arm_q_init = np.asarray(demonstration_seed.arm_q, dtype=np.float64).reshape(7)
    drawer_q_now = float(close_demo._drawer_joint_value(env))
    raw_data_local = getattr(env.sim.data, "_data", env.sim.data)
    site_id_local = int(env.sim.model.site_name2id(frame_name))
    qpos_outer_saved = env.sim.data.qpos.copy()
    qvel_outer_saved = env.sim.data.qvel.copy()

    best = None
    best_candidate_index = int(skeleton_poses[0][0]) if skeleton_poses else 0
    grasp_accept_threshold = float(getattr(args, "grasp_accept_threshold", 1.0))

    # Progress bar for the MPPI attempt loop. Without this the pipeline
    # appears to hang for minutes after the `skeleton_poses=...` log line
    # while `solve_arm_q_mppi` runs silently per attempt.
    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        _tqdm = None
    _mppi_iter = enumerate(order[:max_attempts])
    if _tqdm is not None:
        _mppi_iter = _tqdm(
            list(_mppi_iter),
            total=int(max_attempts),
            desc="grasp mppi",
            unit="pose",
            file=sys.__stdout__,
            dynamic_ncols=True,
            mininterval=0.2,
            leave=True,
        )
    _mppi_t0 = time.perf_counter()

    for attempt_id, pose_index in _mppi_iter:
        candidate_index, sp, mirror_tag = skeleton_poses[int(pose_index)]
        pair = grasp_pairs[int(candidate_index)]
        target_pos, target_quat = ee_skelton.skeleton_pose_to_ee_pose(skeleton, sp)
        target_rot = close_demo._matrix_from_quat_wxyz(target_quat)
        try:
            q_best, g_best, pos_err, rot_err, pen = solve_arm_q_mppi(
                env,
                robot_state=robot_state,
                q_mink=demo_arm_q_init,
                drawer_q=drawer_q_now,
                args=args,
                target_pos=np.asarray(target_pos, dtype=np.float64),
                target_rot=np.asarray(target_rot, dtype=np.float64),
                ee_site_name=frame_name,
                initial_gripper_opening=float(
                    getattr(
                        sp, "gripper_opening", ee_skelton.PANDA_DEFAULT_GRIPPER_OPENING
                    )
                ),
                optimize_gripper=True,
            )
        except Exception as exc:
            continue

        collision_free = bool(pen <= pen_tol)
        if not collision_free:
            continue
        if float(pos_err) > pos_tol:
            continue

        # ---- 8/9. grasp rollout: close the gripper, eval force_closure_cost ---
        # Pass the SAME mesh + object pose MIQP used so force-closure is scored
        # on the handle geometry rather than returning inf (the old code built
        # the rollout without a mesh, so force closure was never evaluated).
        fc_cost = float("inf")
        try:
            from robocasa.demos.rollout_grasp import GraspRollout, GraspRolloutConfig

            drawer_obj = getattr(env, "drawer", None)
            object_body_id = -1
            if drawer_obj is not None:
                try:
                    object_body_id = int(
                        env.sim.model.body_name2id(drawer_obj.root_body_name)
                    )
                except Exception:
                    object_body_id = -1
            rollout = GraspRollout(
                env,
                object_body_id=object_body_id,
                ee_site_name=frame_name,
                config=GraspRolloutConfig(
                    horizon_steps=int(getattr(args, "grasp_rollout_steps", 15)),
                    gripper_closed_opening=float(
                        getattr(args, "grasp_closed_opening", 0.005)
                    ),
                ),
                mesh_path=mesh_path,
                obj_pos=np.asarray(obj_pos, dtype=np.float64),
                obj_scale=(
                    np.asarray(obj_scale, dtype=np.float64)
                    if obj_scale is not None
                    else np.ones(3, dtype=np.float64)
                ),
                obj_quat=np.asarray(obj_quat, dtype=np.float64),
            )
            g_init = float(
                getattr(sp, "gripper_opening", ee_skelton.PANDA_DEFAULT_GRIPPER_OPENING)
            )
            r = rollout.run(
                np.asarray(target_pos, dtype=np.float64),
                target_rot,
                g_init,
            )
            fc_cost = float(r.force_closure_cost_min)
            _autogen_print(
                f"[grasp] rollout pair={candidate_index} "
                f"fc_min={float(r.force_closure_cost_min):.6f} "
                f"fc_final={float(r.force_closure_cost_final):.6f} "
                f"pen={float(r.max_penetration):.6f}"
            )
        except Exception as exc:
            _autogen_print(f"[grasp] rollout error: {exc!r}")
            fc_cost = float("inf")

        successful_precontact_q += 1
        if best is None or fc_cost < best[0]:
            best = (fc_cost, q_best, g_best, pos_err, rot_err, pen, candidate_index)

        if fc_cost < grasp_accept_threshold:
            # Build solution.
            try:
                env.sim.data.qpos[:] = qpos_outer_saved
                env.sim.data.qvel[:] = qvel_outer_saved
                close_demo._set_env_arm_q(
                    env, robot_state["robocasa_joint_names"], q_best
                )
                close_demo._set_drawer_joint_value(env, drawer_q_now)
                env.sim.forward()
                actual_pos = np.asarray(
                    raw_data_local.site_xpos[site_id_local], dtype=np.float64
                ).copy()
                actual_rot = (
                    np.asarray(
                        raw_data_local.site_xmat[site_id_local], dtype=np.float64
                    )
                    .reshape(3, 3)
                    .copy()
                )
            finally:
                env.sim.data.qpos[:] = qpos_outer_saved
                env.sim.data.qvel[:] = qvel_outer_saved
                env.sim.forward()
            solution = mink_solver.PreContactMinkSolution(
                arm_q=np.asarray(q_best, dtype=np.float64).reshape(7),
                target_position_world=np.asarray(target_pos, dtype=np.float64).reshape(
                    3
                ),
                target_rotation_world=np.asarray(target_rot, dtype=np.float64).reshape(
                    3, 3
                ),
                actual_position_world=actual_pos,
                actual_rotation_world=actual_rot,
                position_error=float(pos_err),
                rotation_error=float(rot_err),
                collision_free=True,
                collision_reason="",
            )
            if g_best is not None:
                solution.gripper_opening = float(g_best)
            solution.force_closure_cost = fc_cost
            _autogen_print(
                f"[grasp] accept fc_cost={fc_cost:.6f} pair={candidate_index} "
                f"successful_precontact_q={successful_precontact_q} "
                f"rollout_fc_cost={fc_cost:.6f} "
                f"t={time.perf_counter() - mink_started:.3f}s"
            )
            return solution, candidate_index

    _autogen_print(
        f"[grasp] done attempts={max_attempts} "
        f"successful_precontact_q={successful_precontact_q} "
        f"rollout_fc_cost={best[0] if best is not None else float('inf'):.6f} "
        f"t={time.perf_counter() - mink_started:.3f}s"
    )
    if best is not None and not bool(getattr(args, "require_mink_precontact", True)):
        fc_cost, q_best, g_best, pos_err, rot_err, pen, ci = best
        try:
            env.sim.data.qpos[:] = qpos_outer_saved
            env.sim.data.qvel[:] = qvel_outer_saved
            close_demo._set_env_arm_q(env, robot_state["robocasa_joint_names"], q_best)
            close_demo._set_drawer_joint_value(env, drawer_q_now)
            env.sim.forward()
            actual_pos = np.asarray(
                raw_data_local.site_xpos[site_id_local], dtype=np.float64
            ).copy()
            actual_rot = (
                np.asarray(raw_data_local.site_xmat[site_id_local], dtype=np.float64)
                .reshape(3, 3)
                .copy()
            )
        finally:
            env.sim.data.qpos[:] = qpos_outer_saved
            env.sim.data.qvel[:] = qvel_outer_saved
            env.sim.forward()
        solution = mink_solver.PreContactMinkSolution(
            arm_q=np.asarray(q_best, dtype=np.float64).reshape(7),
            target_position_world=np.zeros(3),
            target_rotation_world=np.eye(3),
            actual_position_world=actual_pos,
            actual_rotation_world=actual_rot,
            position_error=float(pos_err),
            rotation_error=float(rot_err),
            collision_free=True,
            collision_reason="grasp_best_effort",
        )
        if g_best is not None:
            solution.gripper_opening = float(g_best)
        solution.force_closure_cost = float(fc_cost)
        return solution, ci

    raise RuntimeError(
        "Grasp precontact search found no collision-free IK with force-closure "
        f"< {grasp_accept_threshold}. attempts={max_attempts} "
        f"successful_q={successful_precontact_q}"
    )


def _visualize_contact_pairs_popup(env, grasp_pairs, args):
    """Popup viewer showing MIQP contact pairs as colored spheres.

    One HSV color per pair; the *left* point is drawn light (low saturation +
    higher alpha) and the *right* point is drawn deep (high saturation, larger
    radius) so orientation is visually unambiguous. Blocks until the user
    closes the viewer, mirroring ``_visualize_mink_q_poses_popup``.
    """
    if not bool(getattr(args, "autogen_visualize_contact_pairs", True)):
        return
    if not grasp_pairs:
        return
    try:
        import mujoco
        import mujoco.viewer
        from robocasa.demos import visualize_mujoco as viz_mj
    except Exception as exc:
        _autogen_print(f"[grasp] contact-pair viewer unavailable: {exc!r}")
        return

    raw_model, raw_data = viz_mj._raw_model_data(env)
    palette = ee_skelton._hsv_palette(max(len(grasp_pairs), 1))
    pts = []
    for pair in grasp_pairs:
        pts.append(
            np.asarray(pair["contact_points_world"][0], dtype=np.float64).reshape(3)
        )
        pts.append(
            np.asarray(pair["contact_points_world"][1], dtype=np.float64).reshape(3)
        )
    if not pts:
        return
    lookat = np.mean(np.stack(pts, axis=0), axis=0)
    radius_left = float(getattr(args, "autogen_contact_pair_radius", 0.004))
    radius_right = radius_left * 1.6
    fps = max(float(getattr(args, "autogen_mink_popup_fps", 30.0)), 1.0)
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
        while viewer.is_running():
            if hasattr(viewer, "user_scn"):
                viewer.user_scn.ngeom = 0
                for pi, pair in enumerate(grasp_pairs):
                    base = palette[pi % palette.shape[0]]
                    # Light (left): desaturate toward white, higher alpha.
                    left_rgba = np.array(
                        [
                            0.5 + 0.5 * float(base[0]),
                            0.5 + 0.5 * float(base[1]),
                            0.5 + 0.5 * float(base[2]),
                            0.95,
                        ],
                        dtype=np.float32,
                    )
                    right_rgba = np.array(
                        [float(base[0]), float(base[1]), float(base[2]), 0.95],
                        dtype=np.float32,
                    )
                    left_pt = np.asarray(
                        pair["contact_points_world"][0], dtype=np.float64
                    ).reshape(3)
                    right_pt = np.asarray(
                        pair["contact_points_world"][1], dtype=np.float64
                    ).reshape(3)
                    try:
                        viz_mj._add_scene_sphere(
                            viewer.user_scn, left_pt, radius_left, left_rgba
                        )
                        viz_mj._add_scene_sphere(
                            viewer.user_scn, right_pt, radius_right, right_rgba
                        )
                    except Exception:
                        pass
            viewer.sync()
            time.sleep(1.0 / fps)


def _iter_scene_geom_ids(env, surface):
    """All geoms except the handle surface geoms and the drawer body.

    The drawer body panels sit right behind the handle; leaving them in the
    scene set makes ``mj_ray`` from finger samples hit those panels and inject
    false clearance rows into DAQP. Mirrors the exclusion set built in
    ``_solve_mink_precontact_autogen``.
    """
    target_geom_ids = set(_target_geom_ids(env, surface))
    model = env.sim.model
    drawer_body_prefix = str(surface.geom_name).split("_door")[0] + "_"
    drawer_body_geom_ids = {
        int(gid)
        for name, gid in model._geom_name2id.items()
        if name.startswith(drawer_body_prefix)
    }
    excluded = target_geom_ids | drawer_body_geom_ids
    raw_model = getattr(model, "_model", model)
    return tuple(gid for gid in range(int(raw_model.ngeom)) if int(gid) not in excluded)


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenDrawer autogen contact demo configured by YAML."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("open_drawer_contact_curobo.yaml")),
        help="YAML config file containing base demo arguments.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value. May be passed multiple times.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Fast debug loop: shrink DAQP / MIQP sample counts, disable all "
            "visualization popups, and enable per-iteration [dual_daqp iter] "
            "logs so DAQP failure modes are grep-able."
        ),
    )
    cli = parser.parse_args()
    config = _load_yaml_config(cli.config)
    config["scene_cache_dir"] = str(
        config.get("scene_cache_dir") or (REPO_ROOT / "outputs" / "scene_point_cache")
    )
    defaults = {
        "mink_arm_posture_cost": 0.02,
        "mink_locked_dof_cost": 200.0,
        "object_representative_point_count": 2048,
        "object_representative_min_per_geom": 16,
        "curobo_trajopt_tsteps": 32,
        "curobo_interpolation_dt": 0.02,
        "curobo_ik_seeds": 16,
        "curobo_graph_seeds": 2,
        "curobo_trajopt_seeds": 2,
        "curobo_max_attempts": 2,
        "curobo_enable_graph_attempt": 1,
        "disable_curobo_self_collision": False,
        "disable_curobo_cuda_graph": False,
        "curobo_world_padding": 0.005,
        "curobo_world_exclude_geoms": "",
        "curobo_world_exclude_bodies": "",
        "curobo_world_max_obstacles": None,
        "curobo_joint_enable_graph": False,
        "curobo_joint_enable_graph_attempt": None,
        "curobo_joint_disable_graph_attempt": None,
        "curobo_joint_max_attempts": 6,
        "curobo_joint_timeout": 5.0,
        "curobo_joint_retry_graph": True,
        "curobo_joint_graph_max_attempts": 2,
        "curobo_joint_graph_timeout": 8.0,
        "curobo_joint_enable_finetune_trajopt": False,
        "curobo_joint_check_start_validity": False,
        "autogen_object_point_count": 384,
        "autogen_handle_subdivide_max_edge": 0.005,
        "autogen_gripper_candidate_count": 50,
        "autogen_initial_pose_count": 200,
        "autogen_precontact_lift": config.get("precontact_distance", 0.04),
        "autogen_mink_max_attempts": 200,
        "autogen_mink_parallel_workers": max(1, (os.cpu_count() or 1)),
        "autogen_skeleton_parallel_workers": max(1, (os.cpu_count() or 1)),
        "autogen_coacd_threshold": 0.05,
        "autogen_coacd_max_convex_hull": 32,
        "autogen_coacd_preprocess_mode": "auto",
        "autogen_coacd_preprocess_resolution": 30,
        "autogen_coacd_resolution": 2000,
        "autogen_coacd_mcts_nodes": 20,
        "autogen_coacd_mcts_iterations": 100,
        "autogen_coacd_mcts_max_depth": 3,
        "autogen_coacd_max_ch_vertex": 256,
        "autogen_visualize_mink_poses": True,
        "autogen_mink_ghost_alpha": 0.28,
        "autogen_mink_popup_camera_distance": 0.85,
        "autogen_mink_popup_camera_azimuth": 135.0,
        "autogen_mink_popup_camera_elevation": -25.0,
        "autogen_mink_popup_fps": 30.0,
        "grasp_num_pairs": 200,
        "grasp_accept_threshold": 1.0,
        "grasp_rollout_steps": 15,
        "grasp_closed_opening": 0.005,
        "grasp_mppi_samples": 256,
        "grasp_mppi_iterations": 6,
        "grasp_precontact_solver": "mink",  # "mink" | "mppi" | "auto"
        "grasp_skeleton_parallel": True,
        "grasp_skeleton_max_workers": 8,
        # --- dual-finger DAQP skeleton solver ---
        "autogen_dual_theta_count": 6,  # number of contact-axis rotation samples
        "autogen_dual_n_lift": 1,  # lift seeds per theta (use >1 to scan approach)
        "autogen_dual_n_g": 1,  # gripper-opening seeds per theta
        "autogen_dual_seed": 7,
        "autogen_dual_max_candidates": 0,  # 0 -> unlimited per (pair, mirror)
        "autogen_dual_min_theta_separation": 0.0,  # radians
        "autogen_dual_debug": False,
        "autogen_visualize_contact_pairs": True,
        "autogen_contact_pair_radius": 0.004,
        # --- relaxed penetration for the dual-finger wrap-grasp path ---
        "autogen_dual_object_penetration_tol": 0.005,
        "autogen_dual_object_margin": 0.0,
        # --- grasp-path skeleton pose cap + optional popup ---
        "autogen_grasp_max_skeleton_poses": 512,
        "autogen_visualize_grasp_skeleton_poses": True,
        # --- debug fast-path (set by --debug) ---
        "debug": False,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    _apply_config_overrides(config, cli.overrides)
    if bool(getattr(cli, "debug", False)):
        config["debug"] = True
        # Only stomp keys the user did NOT already override; that keeps
        # `--set autogen_dual_theta_count=8 --debug` doing the sensible thing.
        _debug_overrides = {
            "autogen_dual_theta_count": 2,
            "autogen_dual_n_lift": 1,
            "autogen_dual_n_g": 1,
            "autogen_dual_debug": True,
            "autogen_skeleton_segment_samples": 2,
            "autogen_skeleton_n_random": 1,
            "autogen_skeleton_theta_count": 2,
            "autogen_visualize_mink_poses": False,
            "autogen_visualize_skeleton_poses": False,
            "autogen_visualize_skeleton_preview": False,
            "autogen_visualize_contact_pairs": False,
            "autogen_visualize_grasp_skeleton_poses": False,
            "grasp_num_pairs": 32,
        }
        _cli_override_keys = {
            key.split("=", 1)[0].replace("-", "_") for key in cli.overrides
        }
        for key, value in _debug_overrides.items():
            if key not in _cli_override_keys:
                config[key] = value
    return argparse.Namespace(**config)


def main():
    base_globals = _open_drawer_main.__globals__
    quiet_result = lambda *_args, **_kwargs: None

    original_solve_stages = base_globals.get("solve_stages")

    def curobo_count_result(name, elapsed_seconds, _count_name, count):
        if str(name) == "curobo":
            _autogen_print(
                "curobo_time="
                f"{float(elapsed_seconds):.6f} "
                f"successful_trajectories={int(count)}"
            )

    def solve_stages_with_stats(*args, **kwargs):
        started = time.perf_counter()
        result = original_solve_stages(*args, **kwargs)
        segments = result.get("segments", []) if isinstance(result, dict) else []
        _autogen_print(
            "curobo_time="
            f"{time.perf_counter() - started:.6f} "
            f"successful_trajectories={int(len(segments))}"
        )
        return result

    # --- Patch ee_skelton so the skeleton popup viewers only show visual
    #     geoms (group 1) instead of all groups.  The default
    #     `viewer.opt.geomgroup[:] = 1` enables every group including collision
    #     geoms (group 0), which MuJoCo renders in debug red/green colors and
    #     makes the whole scene/arm look wrong.  We also swap the verbose
    #     `_draw_skeleton_into_scene` for a quiet version that skips the
    #     per-call `[skeleton_draw] ...` log line.
    _original_draw_skeleton = getattr(ee_skelton, "_draw_skeleton_into_scene", None)
    if _original_draw_skeleton is not None:
        ee_skelton._draw_skeleton_into_scene = _draw_skeleton_into_scene_quiet

    _patch_skeleton_viewers_geomgroup(ee_skelton)

    base_globals["parse_args"] = parse_args
    base_globals["evaluate_open_contacts"] = evaluate_open_contacts
    base_globals["_solve_mink_precontact_seed"] = _solve_grasp_precontact_autogen
    base_globals["_solve_stage"] = _solve_stage_autogen
    base_globals["_stage_result"] = quiet_result
    base_globals["_count_result"] = curobo_count_result
    base_globals["_stage_banner"] = quiet_result
    if original_solve_stages is not None:
        base_globals["solve_stages"] = solve_stages_with_stats
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        _open_drawer_main()


if __name__ == "__main__":
    main()
