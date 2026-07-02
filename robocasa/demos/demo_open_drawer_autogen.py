import argparse
import contextlib
import os
import shutil
import sys
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
                "    from robocasa.demos.demo_open_drawer_autogen import main\n"
                "main()"
            ),
            *sys.argv[1:],
        ],
    )

import numpy as np
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


with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
    open(os.devnull, "w")
):
    import robocasa.demos.demo_close_drawer_contact_curobo as close_demo
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
        f"handle_convex_parts={int(handle_convex_equations.shape[0])}"
    )
    return cached[cache_key]


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


def _solve_stage_autogen(env, surface, pull_distance, args, stage_name):
    stage, reports, robot_state = _BASE_SOLVE_STAGE(
        env, surface, pull_distance, args, stage_name
    )
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


def _solve_mink_precontact_autogen(
    env,
    surface,
    candidates,
    demonstration_seed,
    robot_state,
    args,
):
    del candidates  # autogen path no longer needs the candidate list directly
    mink_started = time.perf_counter()
    successful_precontact_q = 0
    panel = close_demo.get_panel_frame(env)
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
    try:
        ee_skelton.visualize_skeleton_and_ee(env, frame_name, skeleton, args)
    except Exception:
        pass
    n_select = int(getattr(args, "autogen_initial_pose_count", 200))
    local_ids, points_world, normals_world = ee_skelton.select_interior_feasible_points(
        feasible_cache, n_select, int(args.seed)
    )
    feasible_candidate_indices = np.asarray(
        feasible_cache.candidate_indices, dtype=np.int64
    )
    target_geom_ids = set(_target_geom_ids(env, surface))
    model = env.sim.model
    drawer_body_prefix = str(surface.geom_name).split("_door")[0] + "_"
    drawer_body_geom_ids = {
        int(gid)
        for name, gid in model._geom_name2id.items()
        if name.startswith(drawer_body_prefix)
    }
    excluded_scene_geom_ids = target_geom_ids | drawer_body_geom_ids
    scene_geom_ids = tuple(
        gid
        for gid in range(int(getattr(env.sim.model, "_model", env.sim.model).ngeom))
        if gid not in excluded_scene_geom_ids
    )
    _autogen_print(
        f"skeleton_scene_exclude target={len(target_geom_ids)} "
        f"drawer_body={len(drawer_body_geom_ids)} "
        f"scene_geoms={len(scene_geom_ids)}"
    )

    skeleton_poses: list[tuple[int, ee_skelton.SkeletonPose]] = []
    demo_ee_rotation = np.asarray(
        demonstration_seed.projected_ee_rotation_world, dtype=np.float64
    ).reshape(3, 3)
    args._skeleton_debug_log = []
    for local_id, point, normal in zip(local_ids, points_world, normals_world):
        candidate_index = int(feasible_candidate_indices[int(local_id)])
        for finger in ("left", "right"):
            sp = ee_skelton.solve_skeleton_pose(
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
            if sp is not None:
                skeleton_poses.append((candidate_index, sp))
    _autogen_print(f"skeleton_poses={len(skeleton_poses)}")
    _dbg = list(getattr(args, "_skeleton_debug_log", []) or [])
    if _dbg:
        from collections import Counter as _C

        _status_hist = _C(int(d["status"]) for d in _dbg)
        _scene_rows = np.array([d["n_scene_rows"] for d in _dbg], dtype=np.int64)
        _viol = np.array([d["init_max_violation"] for d in _dbg], dtype=np.float64)
        _eqres = np.array([d["init_eq_residual"] for d in _dbg], dtype=np.float64)
        _autogen_print(
            "skeleton_qp_dbg "
            f"total={len(_dbg)} "
            f"status={dict(_status_hist)} "
            f"scene_rows[min/med/max]={int(_scene_rows.min())}/{int(np.median(_scene_rows))}/{int(_scene_rows.max())} "
            f"init_viol[med/max]={float(np.median(_viol)):.4f}/{float(_viol.max()):.4f} "
            f"init_eq_res[med/max]={float(np.median(_eqres)):.4f}/{float(_eqres.max()):.4f}"
        )

    try:
        ee_skelton.visualize_skeleton_poses(
            env, frame_name, skeleton, [sp for _, sp in skeleton_poses], args
        )
    except Exception:
        pass

    rng = np.random.default_rng(int(args.seed) + 29003)
    order = np.arange(len(skeleton_poses), dtype=np.int64)
    rng.shuffle(order)
    max_attempts = min(
        int(getattr(args, "autogen_mink_max_attempts", len(skeleton_poses))),
        len(skeleton_poses),
    )
    attempts = []
    best = None
    best_candidate_index = (
        int(skeleton_poses[int(order[0])][0]) if skeleton_poses else 0
    )
    drawer_q_now = float(close_demo._drawer_joint_value(env))
    poses_arr = []
    cand_arr = []
    pos_tol = float(getattr(args, "mink_position_tolerance", 0.01))
    pen_tol = float(getattr(args, "mink_collision_penetration_tolerance", 0.001))
    demo_arm_q_init = np.asarray(demonstration_seed.arm_q, dtype=np.float64).reshape(7)
    raw_data_local = getattr(env.sim.data, "_data", env.sim.data)
    site_id_local = int(env.sim.model.site_name2id(frame_name))
    qpos_outer_saved = env.sim.data.qpos.copy()
    qvel_outer_saved = env.sim.data.qvel.copy()

    def _build_solution(target_pos, target_rot, q_best, pos_err, rot_err, pen):
        try:
            env.sim.data.qpos[:] = qpos_outer_saved
            env.sim.data.qvel[:] = qvel_outer_saved
            close_demo._set_env_arm_q(env, robot_state["robocasa_joint_names"], q_best)
            close_demo._set_drawer_joint_value(env, drawer_q_now)
            env.sim.forward()
            actual_pos = (
                np.asarray(raw_data_local.site_xpos[site_id_local], dtype=np.float64)
                .reshape(3)
                .copy()
            )
            actual_rot = (
                np.asarray(raw_data_local.site_xmat[site_id_local], dtype=np.float64)
                .reshape(3, 3)
                .copy()
            )
        finally:
            env.sim.data.qpos[:] = qpos_outer_saved
            env.sim.data.qvel[:] = qvel_outer_saved
            env.sim.forward()
        collision_free = bool(pen <= pen_tol)
        reason = "" if collision_free else f"mppi_penetration:max_dist={-pen:.6f}"
        return mink_solver.PreContactMinkSolution(
            arm_q=np.asarray(q_best, dtype=np.float64).reshape(7),
            target_position_world=np.asarray(target_pos, dtype=np.float64).reshape(3),
            target_rotation_world=np.asarray(target_rot, dtype=np.float64).reshape(
                3, 3
            ),
            actual_position_world=actual_pos,
            actual_rotation_world=actual_rot,
            position_error=float(pos_err),
            rotation_error=float(rot_err),
            collision_free=collision_free,
            collision_reason=reason,
        )

    for attempt_id, pose_index in enumerate(order[:max_attempts]):
        candidate_index, sp = skeleton_poses[int(pose_index)]
        target_pos, target_quat = ee_skelton.skeleton_pose_to_ee_pose(skeleton, sp)
        target_rot = close_demo._matrix_from_quat_wxyz(target_quat)
        poses_arr.append(np.concatenate([target_pos, target_quat]))
        cand_arr.append(candidate_index)
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
            attempts.append((attempt_id, candidate_index, "exception", str(exc)))
            continue
        solution = _build_solution(
            target_pos, target_rot, q_best, pos_err, rot_err, pen
        )
        if g_best is not None:
            solution.gripper_opening = float(g_best)
        ok = bool(solution.position_error <= pos_tol and solution.collision_free)
        if ok:
            successful_precontact_q += 1
        attempts.append(
            (
                attempt_id,
                candidate_index,
                float(solution.position_error),
                bool(solution.collision_free),
                str(solution.collision_reason),
            )
        )
        score = float(solution.position_error) + (
            0.0 if solution.collision_free else 1000.0
        )
        if best is None or score < best[0]:
            best = (score, solution, candidate_index)
            best_candidate_index = candidate_index
        if ok:
            args._autogen_initial_poses = np.asarray(
                poses_arr, dtype=np.float64
            ).reshape(-1, 7)
            args._autogen_initial_candidate_indices = np.asarray(
                cand_arr, dtype=np.int64
            )
            args._autogen_mink_attempts = attempts
            _visualize_mink_q_poses_popup(
                env,
                np.asarray(solution.arm_q, dtype=np.float64).reshape(1, 7),
                robot_state,
                args,
                drawer_q=drawer_q_now,
            )
            args._autogen_visualized_mink_precontact_q = True
            _autogen_print(
                "mink_q_time="
                f"{time.perf_counter() - mink_started:.6f} "
                f"successful_pre_contact_q={successful_precontact_q}"
            )
            return solution, candidate_index
    if best is not None and not bool(getattr(args, "require_mink_precontact", True)):
        _autogen_print(
            "mink_q_time="
            f"{time.perf_counter() - mink_started:.6f} "
            f"successful_pre_contact_q={successful_precontact_q}"
        )
        return best[1], best_candidate_index
    _autogen_print(
        "mink_q_time="
        f"{time.perf_counter() - mink_started:.6f} "
        f"successful_pre_contact_q={successful_precontact_q}"
    )
    from collections import Counter

    _reason_counter: Counter = Counter()
    for rec in attempts:
        if len(rec) >= 5 and not bool(rec[3]):
            reason = str(rec[4]).split(":", 1)[0] or "unknown"
        elif len(rec) >= 4 and rec[2] == "exception":
            reason = "exception"
        else:
            continue
        _reason_counter[reason] += 1
    if _reason_counter:
        _autogen_print(
            "mink_fail_reasons="
            + ",".join(f"{k}={v}" for k, v in _reason_counter.most_common())
        )
    raise RuntimeError(
        "Autogen mink precontact search found no collision-free IK. "
        f"attempts={attempts[:8]} total_attempts={len(attempts)}"
    )


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
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    _apply_config_overrides(config, cli.overrides)
    _apply_drawer_action_overrides(config)
    return argparse.Namespace(**config)


def _apply_drawer_action_overrides(config):
    """Auto-tune contact/execution settings based on `drawer_action`.

    For `close`, the gripper pushes the drawer's outer face. Leaving
    `panda_hand` in `curobo_contact_sphere_links` lets the q-config MPC pick
    the back-of-hand sphere as the contact and tangent its centre against the
    surface, which forces the TCP ~10 cm into the drawer. Same reasoning makes
    the trailing pull/push waypoint drive the EE further inward — gate it off
    by default. Both can still be re-enabled via `--set`.
    """
    action = str(config.get("drawer_action", "open")).strip().lower()
    if action != "close":
        return
    links_field = "curobo_contact_sphere_links"
    raw_links = config.get(links_field, "")
    if isinstance(raw_links, str):
        link_iter = raw_links.split(",")
    else:
        link_iter = list(raw_links)
    filtered = [
        name
        for name in (str(item).strip() for item in link_iter)
        if name and name != "panda_hand"
    ]
    if filtered:
        config[links_field] = ",".join(filtered)
    config.setdefault("execute_pull_stage", False)


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

    base_globals["parse_args"] = parse_args
    base_globals["evaluate_open_contacts"] = evaluate_open_contacts
    base_globals["_solve_mink_precontact_seed"] = _solve_mink_precontact_autogen
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
