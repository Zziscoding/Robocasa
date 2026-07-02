"""Open3D visualizations for contact optimization and EE-pose stages.

Scene points follow PointWorld's camera convention: metric depth is
back-projected with camera intrinsics, transformed by the inverse world-to-
camera extrinsic, and paired with RGB from the same pixel.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import warnings
from typing import Iterable, Sequence

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)

from robocasa.demos.scene_process import _primitive_geom


def _o3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required for optimization-stage visualization."
        ) from exc
    return o3d


def _raw_model(model):
    return getattr(model, "_model", model)


def _raw_data(data):
    return getattr(data, "_data", data)


def _point_cloud(points, colors):
    o3d = _o3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64).reshape(-1, 3)
    )
    cloud.colors = o3d.utility.Vector3dVector(
        np.asarray(colors, dtype=np.float64).reshape(-1, 3).clip(0.0, 1.0)
    )
    return cloud


def _sample_image_colors(env, points_world, camera_name, rgb):
    """Project world points into RGB using the PointWorld camera convention."""

    height, width = rgb.shape[:2]
    intrinsic = get_camera_intrinsic_matrix(env.sim, camera_name, height, width)
    camera_to_world = get_camera_extrinsic_matrix(env.sim, camera_name)
    world_to_camera = np.linalg.inv(camera_to_world)
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    camera_points = points_h @ world_to_camera.T
    z = camera_points[:, 2]
    safe_z = np.where(z > 1e-8, z, 1.0)
    u = np.rint(
        intrinsic[0, 0] * camera_points[:, 0] / safe_z + intrinsic[0, 2]
    ).astype(np.int64)
    v = np.rint(
        intrinsic[1, 1] * camera_points[:, 1] / safe_z + intrinsic[1, 2]
    ).astype(np.int64)
    visible = (z > 1e-8) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    colors = np.full((points.shape[0], 3), 0.65, dtype=np.float64)
    colors[visible] = (
        np.asarray(rgb[v[visible], u[visible], :3], dtype=np.float64) / 255.0
    )
    return colors


def _project_rgbd_to_world(
    env, camera_name, rgb, depth, depth_min=0.0, depth_max=np.inf
):
    """PointWorld-style RGB-D back-projection with one RGB per depth pixel."""

    depth = np.asarray(depth, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    height, width = depth.shape
    if rgb.shape[:2] != (height, width):
        raise ValueError(
            f"RGB/depth shape mismatch: {rgb.shape[:2]} != {(height, width)}"
        )
    intrinsic = np.asarray(
        get_camera_intrinsic_matrix(env.sim, camera_name, height, width),
        dtype=np.float32,
    )
    camera_to_world = np.asarray(
        get_camera_extrinsic_matrix(env.sim, camera_name),
        dtype=np.float32,
    )
    valid = (
        np.isfinite(depth)
        & (depth > max(float(depth_min), 1e-6))
        & (depth < float(depth_max))
    )
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs]
    x = (xs.astype(np.float32) - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (ys.astype(np.float32) - intrinsic[1, 2]) / intrinsic[1, 1] * z
    camera_points = np.stack([x, y, z], axis=1)
    camera_points_h = np.concatenate(
        [
            camera_points,
            np.ones((camera_points.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    world_points = (camera_points_h @ camera_to_world.T)[:, :3]
    finite = np.isfinite(world_points).all(axis=1)
    return (
        world_points[finite].astype(np.float32, copy=False),
        rgb[ys, xs][finite].astype(np.uint8, copy=False),
    )


def _object_geom_ids(point_cloud) -> set[int]:
    if point_cloud is None:
        return set()
    result = set()
    for link in point_cloud.links:
        result.update(int(geom_id) for geom_id in link.runtime_geom_ids)
    return result


def _sample_model_scene(env, excluded_geom_ids, max_points):
    model = _raw_model(env.sim.model)
    data = _raw_data(env.sim.data)
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if geom_id not in excluded_geom_ids
        and float(model.geom_rgba[geom_id, 3]) > 1e-5
        and int(model.geom_type[geom_id])
        not in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)
    ]
    if not geom_ids:
        return np.zeros((0, 3)), np.zeros((0, 3))
    per_geom = max(int(max_points) // len(geom_ids), 16)
    rng = np.random.default_rng(0)
    point_chunks = []
    color_chunks = []
    for geom_id in geom_ids:
        mesh = _primitive_geom(model, geom_id)
        if mesh is None or not mesh.faces.size:
            continue
        points_local, _ = trimesh.sample.sample_surface(mesh, per_geom, seed=rng)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        point_chunks.append(points_local @ rotation.T + position)
        color_chunks.append(
            np.repeat(
                np.asarray(model.geom_rgba[geom_id, :3]).reshape(1, 3),
                per_geom,
                axis=0,
            )
        )
    if not point_chunks:
        return np.zeros((0, 3)), np.zeros((0, 3))
    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    if points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]
        colors = colors[indices]
    return points, colors


def _sampled_object_colors(point_cloud, model):
    chunks = []
    for link in point_cloud.links:
        geom_ids = [
            geom_id for geom_id in link.runtime_geom_ids if 0 <= geom_id < model.ngeom
        ]
        color = (
            np.mean(np.asarray(model.geom_rgba)[geom_ids, :3], axis=0)
            if geom_ids
            else np.array([0.7, 0.7, 0.7])
        )
        chunks.append(
            np.repeat(color.reshape(1, 3), link.points_local.shape[0], axis=0)
        )
    return np.concatenate(chunks, axis=0)


def _link_geom_color(link, model):
    geom_ids = [
        geom_id for geom_id in link.runtime_geom_ids if 0 <= geom_id < model.ngeom
    ]
    if not geom_ids:
        return np.array([0.65, 0.65, 0.65], dtype=np.float64)
    return np.mean(np.asarray(model.geom_rgba)[geom_ids, :3], axis=0)


def _visible_sampled_object_points(point_cloud, data):
    point_chunks = []
    link_indices = []
    for link_index, link in enumerate(point_cloud.links):
        visible = np.flatnonzero(np.asarray(link.mask, dtype=np.uint8) == 1)
        if visible.size == 0:
            continue
        point_chunks.append(link.world_points(data)[visible])
        link_indices.extend([link_index] * visible.size)
    if not point_chunks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return (
        np.concatenate(point_chunks, axis=0).astype(np.float32),
        np.asarray(link_indices, dtype=np.int64),
    )


def build_rgb_scene_geometries(env, args):
    """RGB-D scene cloud with the raw object removed and sampled links inserted."""

    camera_name = args.optimization_viz_camera
    width = int(args.optimization_viz_width)
    height = int(args.optimization_viz_height)
    capture_dir = tempfile.mkdtemp(prefix="robocasa_rgbd_")
    model_path = os.path.join(capture_dir, "model.mjb")
    state_path = os.path.join(capture_dir, "state.npz")
    output_path = os.path.join(capture_dir, "render.npz")
    raw_model = _raw_model(env.sim.model)
    raw_data = _raw_data(env.sim.data)
    mujoco.mj_saveModel(raw_model, model_path, None)
    np.savez(
        state_path,
        qpos=np.asarray(raw_data.qpos),
        qvel=np.asarray(raw_data.qvel),
        mocap_pos=np.asarray(raw_data.mocap_pos),
        mocap_quat=np.asarray(raw_data.mocap_quat),
        eq_active=np.asarray(raw_data.eq_active),
    )
    capture_script = (
        __import__("pathlib").Path(__file__).with_name("mujoco_rgbd_capture.py")
    )
    environment = os.environ.copy()
    environment.pop("MUJOCO_GL", None)
    rgbd_succeeded = False
    try:
        capture_result = subprocess.run(
            [
                sys.executable,
                str(capture_script),
                model_path,
                state_path,
                output_path,
                "--camera",
                camera_name,
                "--width",
                str(width),
                "--height",
                str(height),
            ],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
        rendered = np.load(output_path)
        rgb = np.asarray(rendered["rgb"], dtype=np.uint8)
        depth = np.asarray(rendered["depth"], dtype=np.float64)
        rgbd_succeeded = True
    except Exception as exc:
        stderr = getattr(exc, "stderr", "") or ""
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else str(exc)
        warnings.warn(
            "Isolated MuJoCo RGB-D rendering failed; using uniformly sampled "
            f"MuJoCo scene geometry with geom RGB colors instead: {detail}",
            RuntimeWarning,
        )
    finally:
        __import__("shutil").rmtree(capture_dir, ignore_errors=True)

    point_cloud = getattr(args, "_scene_point_cloud", None)
    object_ids = _object_geom_ids(point_cloud)
    max_points = max(int(args.optimization_viz_scene_points), 1)
    object_points = (
        point_cloud.world_points(env.sim.data)
        if point_cloud is not None and point_cloud.size
        else np.zeros((0, 3), dtype=np.float32)
    )
    visible_object_points, visible_link_indices = (
        _visible_sampled_object_points(point_cloud, env.sim.data)
        if point_cloud is not None and point_cloud.size
        else (np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int64))
    )
    if rgbd_succeeded:
        scene_points, scene_colors_uint8 = _project_rgbd_to_world(
            env,
            camera_name,
            rgb,
            depth,
            depth_min=args.optimization_viz_depth_min,
            depth_max=args.optimization_viz_depth_max,
        )
        if object_points.shape[0]:
            distances, _ = cKDTree(object_points).query(scene_points, k=1, workers=-1)
            keep = distances > float(args.optimization_viz_object_clearance)
            scene_points = scene_points[keep]
            scene_colors_uint8 = scene_colors_uint8[keep]
        scene_colors = scene_colors_uint8.astype(np.float64) / 255.0
    else:
        scene_points, scene_colors = _sample_model_scene(env, object_ids, max_points)
    if scene_points.shape[0] > max_points:
        indices = np.linspace(0, scene_points.shape[0] - 1, max_points, dtype=np.int64)
        scene_points = scene_points[indices]
        scene_colors = scene_colors[indices]

    geometries = [
        {
            "name": "rgb_scene",
            "geometry": _point_cloud(scene_points, scene_colors),
        }
    ]
    if visible_object_points.shape[0]:
        object_colors = (
            _sample_image_colors(env, visible_object_points, camera_name, rgb)
            if rgbd_succeeded
            else np.asarray(
                [
                    _link_geom_color(point_cloud.links[int(link_index)], raw_model)
                    for link_index in visible_link_indices
                ],
                dtype=np.float64,
            )
        )
        geometries.append(
            {
                "name": "sampled_object",
                "geometry": _point_cloud(visible_object_points, object_colors),
            }
        )
    return geometries


def _line_set(starts, ends, color):
    o3d = _o3d()
    starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
    ends = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
    count = starts.shape[0]
    points = np.concatenate([starts, ends], axis=0)
    lines = np.asarray([[index, index + count] for index in range(count)])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.repeat(np.asarray(color, dtype=np.float64).reshape(1, 3), count, axis=0)
    )
    return line_set


def show_feasible_contact_forces(env, candidates, args, title):
    if not args.visualize_optimization_stages:
        return
    feasible = [
        candidate
        for candidate in candidates
        if candidate.feasible
        and hasattr(candidate, "optimal_force_world")
        and np.linalg.norm(candidate.optimal_force_world) > 1e-10
    ]
    geometries = build_rgb_scene_geometries(env, args)
    if feasible:
        starts = np.asarray([candidate.world_point for candidate in feasible])
        directions = np.asarray(
            [
                candidate.optimal_force_world
                / max(np.linalg.norm(candidate.optimal_force_world), 1e-12)
                for candidate in feasible
            ]
        )
        ends = starts + directions * float(args.optimization_viz_force_length)
        geometries.append(
            {
                "name": "feasible_contacts",
                "geometry": _point_cloud(
                    starts,
                    np.repeat(np.array([[1.0, 0.15, 0.65]]), len(starts), axis=0),
                ),
            }
        )
        geometries.append(
            {
                "name": "optimal_force_flow",
                "geometry": _line_set(starts, ends, [1.0, 0.15, 0.65]),
            }
        )
    _draw(geometries, title, args)


def _body_descendants(model, root_body_id):
    result = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if int(model.body_parentid[body_id]) in result and body_id not in result:
                result.add(body_id)
                changed = True
    return result


def extract_ee_mesh_local(env, frame_name):
    """Return the actual Franka EE visual mesh expressed in the EE site frame."""

    model = _raw_model(env.sim.model)
    data = _raw_data(env.sim.data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
    if site_id < 0:
        raise ValueError(f"Unknown EE site {frame_name!r}.")
    body_ids = _body_descendants(model, int(model.site_bodyid[site_id]))
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
        and float(model.geom_rgba[geom_id, 3]) > 1e-5
        and int(model.geom_group[geom_id]) == 1
    ]
    if not geom_ids:
        geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in body_ids
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        ]
    site_position = np.asarray(data.site_xpos[site_id], dtype=np.float64)
    site_rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    meshes = []
    for geom_id in geom_ids:
        mesh = _primitive_geom(model, geom_id)
        if mesh is None or not mesh.faces.size:
            continue
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        vertices_world = np.asarray(mesh.vertices) @ rotation.T + position
        vertices_local = (vertices_world - site_position) @ site_rotation
        meshes.append(
            trimesh.Trimesh(
                vertices=vertices_local,
                faces=np.asarray(mesh.faces),
                process=False,
            )
        )
    if not meshes:
        raise RuntimeError("No Franka EE visual or collision meshes were found.")
    merged = trimesh.util.concatenate(meshes)
    merged.remove_unreferenced_vertices()
    return np.asarray(merged.vertices), np.asarray(merged.faces)


def transform_mesh(vertices_local, faces, position, rotation):
    o3d = _o3d()
    vertices_world = np.asarray(vertices_local, dtype=np.float64) @ np.asarray(
        rotation, dtype=np.float64
    ).reshape(3, 3).T + np.asarray(position, dtype=np.float64).reshape(1, 3)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices_world),
        triangles=o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    return mesh


def show_ghost_ee(
    env,
    vertices_local,
    faces,
    ee_position,
    ee_rotation,
    args,
    title,
    contact_point=None,
):
    if not args.visualize_optimization_stages:
        return
    geometries = build_rgb_scene_geometries(env, args)
    material = _o3d().visualization.rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [
        0.05,
        0.75,
        1.0,
        float(args.optimization_viz_ghost_alpha),
    ]
    geometries.append(
        {
            "name": "ghost_ee",
            "geometry": transform_mesh(
                vertices_local,
                faces,
                ee_position,
                ee_rotation,
            ),
            "material": material,
        }
    )
    if contact_point is not None:
        geometries.append(
            {
                "name": "contact",
                "geometry": _point_cloud(
                    np.asarray(contact_point).reshape(1, 3),
                    np.array([[1.0, 0.05, 0.05]]),
                ),
            }
        )
    _draw(geometries, title, args)


def _draw(geometries, title, args):
    o3d = _o3d()
    try:
        serialized = []
        for item in geometries:
            geometry = item["geometry"]
            spec = {"name": item["name"]}
            if isinstance(geometry, o3d.geometry.PointCloud):
                spec.update(
                    kind="point_cloud",
                    points=np.asarray(geometry.points),
                    colors=np.asarray(geometry.colors),
                )
            elif isinstance(geometry, o3d.geometry.LineSet):
                spec.update(
                    kind="line_set",
                    points=np.asarray(geometry.points),
                    lines=np.asarray(geometry.lines),
                    colors=np.asarray(geometry.colors),
                )
            elif isinstance(geometry, o3d.geometry.TriangleMesh):
                spec.update(
                    kind="triangle_mesh",
                    vertices=np.asarray(geometry.vertices),
                    triangles=np.asarray(geometry.triangles),
                )
            else:
                raise TypeError(
                    f"Unsupported Open3D geometry: {type(geometry).__name__}"
                )
            material = item.get("material")
            if material is not None:
                spec["material"] = {
                    "shader": material.shader,
                    "base_color": np.asarray(material.base_color),
                }
            serialized.append(spec)

        bundle = {
            "title": title,
            "width": int(args.optimization_viz_window_width),
            "height": int(args.optimization_viz_window_height),
            "point_size": float(args.optimization_viz_point_size),
            "line_width": float(args.optimization_viz_line_width),
            "geometries": serialized,
        }
        bundle_file = tempfile.NamedTemporaryFile(
            prefix="robocasa_o3d_",
            suffix=".pkl",
            delete=False,
        )
        bundle_path = bundle_file.name
        with bundle_file:
            pickle.dump(bundle, bundle_file, protocol=pickle.HIGHEST_PROTOCOL)
        viewer_script = (
            __import__("pathlib").Path(__file__).with_name("open3d_stage_viewer.py")
        )
        environment = os.environ.copy()
        # Do not inherit MuJoCo-specific GL backend selection in the GUI child.
        environment.pop("MUJOCO_GL", None)
        subprocess.run(
            [sys.executable, str(viewer_script), bundle_path],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        stderr = getattr(exc, "stderr", "") or ""
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else str(exc)
        warnings.warn(
            f"Open3D optimization visualization {title!r} failed: {detail}",
            RuntimeWarning,
        )
    finally:
        if "bundle_path" in locals():
            try:
                os.unlink(bundle_path)
            except OSError:
                pass


def show_viser_ghost_ee_pose(
    env,
    vertices_local,
    faces,
    ee_position,
    ee_rotation,
    args,
    title,
    *,
    contact_point=None,
    feasible_candidates=None,
):
    """Viser 3D popup: scene point cloud, object points, ghost EE mesh, and
    feasible-contact force lines (pink). Blocks until the user presses Enter.
    Gracefully falls back to a no-op when viser is not installed.
    """
    if not getattr(args, "visualize_optimization_stages", True):
        return

    try:
        import viser as _viser
    except ImportError:
        warnings.warn(
            "viser is not installed; skipping EE-pose 3D visualisation. "
            "Install with: pip install viser",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    port = int(getattr(args, "viser_port", 8090))

    # ── Scene + object RGB data ──────────────────────────────────────────────
    scene_pts = np.zeros((0, 3), dtype=np.float32)
    scene_cols = np.zeros((0, 3), dtype=np.uint8)
    obj_pts = np.zeros((0, 3), dtype=np.float32)
    obj_cols = np.zeros((0, 3), dtype=np.uint8)
    try:
        geometries = build_rgb_scene_geometries(env, args)
        o3d_lib = _o3d()
        for item in geometries:
            geom = item.get("geometry")
            if geom is None or not isinstance(geom, o3d_lib.geometry.PointCloud):
                continue
            pts = np.asarray(geom.points, dtype=np.float32)
            cols = (
                (np.asarray(geom.colors, dtype=np.float32) * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )
            if item["name"] == "sampled_object":
                obj_pts = (
                    np.concatenate([obj_pts, pts], axis=0) if obj_pts.size else pts
                )
                obj_cols = (
                    np.concatenate([obj_cols, cols], axis=0) if obj_cols.size else cols
                )
            else:
                scene_pts = (
                    np.concatenate([scene_pts, pts], axis=0) if scene_pts.size else pts
                )
                scene_cols = (
                    np.concatenate([scene_cols, cols], axis=0)
                    if scene_cols.size
                    else cols
                )
    except Exception as exc:
        warnings.warn(
            f"Failed to build scene for viser visualisation: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    # ── Ghost EE mesh in world coordinates ──────────────────────────────────
    ghost_verts = None
    ghost_faces_arr = None
    ghost_fallback_pts = None
    if vertices_local is not None and faces is not None:
        vl = np.asarray(vertices_local, dtype=np.float32)
        fc = np.asarray(faces, dtype=np.int32)
        ee_r = np.asarray(ee_rotation, dtype=np.float32).reshape(3, 3)
        ee_p = np.asarray(ee_position, dtype=np.float32).reshape(1, 3)
        ghost_verts = vl @ ee_r.T + ee_p
        ghost_faces_arr = fc
        try:
            _tmesh = trimesh.Trimesh(
                vertices=ghost_verts.astype(np.float64), faces=fc, process=False
            )
            ghost_fallback_pts, _ = trimesh.sample.sample_surface(_tmesh, 4000)
            ghost_fallback_pts = ghost_fallback_pts.astype(np.float32)
        except Exception:
            ghost_fallback_pts = ghost_verts

    # ── Feasible contact force segments (pink) ──────────────────────────────
    force_seg_pts = None
    force_seg_cols = None
    feas_cloud = None
    if feasible_candidates:
        force_length = float(getattr(args, "optimization_viz_force_length", 0.018))
        feas = [
            c
            for c in feasible_candidates
            if c.feasible
            and hasattr(c, "optimal_force_world")
            and np.linalg.norm(c.optimal_force_world) > 1e-10
        ]
        if feas:
            starts = np.asarray([c.world_point for c in feas], dtype=np.float32)
            dirs = np.asarray(
                [
                    c.optimal_force_world
                    / max(np.linalg.norm(c.optimal_force_world), 1e-12)
                    for c in feas
                ],
                dtype=np.float32,
            )
            ends = starts + dirs * force_length
            force_seg_pts = np.stack([starts, ends], axis=1)  # (N, 2, 3)
            force_seg_cols = np.full((len(feas), 2, 3), [255, 38, 166], dtype=np.uint8)
            feas_cloud = starts

    # ── Launch viser server ──────────────────────────────────────────────────
    server = None
    try:
        server = _viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
        try:
            server.scene.world_axes.visible = False
        except Exception:
            pass
        try:
            server.scene.set_up_direction((0.0, 0.0, 1.0))
        except Exception:
            pass

        pt_size = float(getattr(args, "optimization_viz_point_size", 3.0)) * 1e-3
        ghost_alpha = float(getattr(args, "optimization_viz_ghost_alpha", 0.32))

        if scene_pts.size:
            server.scene.add_point_cloud(
                "scene/background",
                points=scene_pts,
                colors=scene_cols,
                point_size=pt_size,
                point_shape="rounded",
                precision="float32",
            )
        if obj_pts.size:
            server.scene.add_point_cloud(
                "scene/object",
                points=obj_pts,
                colors=obj_cols,
                point_size=pt_size * 1.5,
                point_shape="rounded",
                precision="float32",
            )
        if ghost_verts is not None:
            ghost_added = False
            try:
                server.scene.add_mesh_simple(
                    "ghost_ee/mesh",
                    vertices=ghost_verts,
                    faces=ghost_faces_arr.astype(np.uint32),
                    color=(13, 191, 255),
                    opacity=ghost_alpha,
                    side="double",
                    flat_shading=False,
                )
                ghost_added = True
            except Exception:
                pass
            if not ghost_added and ghost_fallback_pts is not None:
                server.scene.add_point_cloud(
                    "ghost_ee/points",
                    points=ghost_fallback_pts,
                    colors=np.full(
                        (ghost_fallback_pts.shape[0], 3), [13, 191, 255], dtype=np.uint8
                    ),
                    point_size=pt_size * 1.2,
                    point_shape="rounded",
                    precision="float32",
                )
        if contact_point is not None:
            cp = np.asarray(contact_point, dtype=np.float32).reshape(1, 3)
            server.scene.add_point_cloud(
                "contact/selected",
                points=cp,
                colors=np.array([[255, 20, 20]], dtype=np.uint8),
                point_size=0.015,
                point_shape="rounded",
            )
        if force_seg_pts is not None:
            server.scene.add_line_segments(
                "contact/force_lines",
                points=force_seg_pts,
                colors=force_seg_cols,
                line_width=2.0,
            )
        if feas_cloud is not None:
            server.scene.add_point_cloud(
                "contact/feasible",
                points=feas_cloud,
                colors=np.full(
                    (feas_cloud.shape[0], 3), [255, 38, 166], dtype=np.uint8
                ),
                point_size=pt_size * 1.5,
                point_shape="rounded",
            )

        print(f"\n[viser] {title}")
        print(f"[viser] http://127.0.0.1:{port}")
        print("[viser] Press Enter to continue...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
    except Exception as exc:
        warnings.warn(
            f"Viser EE-pose visualisation failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    finally:
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass


def mujoco_contact_between(
    env,
    geom_names_a: Iterable[str],
    geom_names_b: Iterable[str],
):
    model = _raw_model(env.sim.model)
    data = _raw_data(env.sim.data)
    ids_a = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in geom_names_a
    }
    ids_b = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in geom_names_b
    }
    ids_a.discard(-1)
    ids_b.discard(-1)
    minimum_distance = float("inf")
    has_contact = False
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & ids_a and pair & ids_b:
            has_contact = True
            minimum_distance = min(minimum_distance, float(contact.dist))
    return has_contact, minimum_distance
