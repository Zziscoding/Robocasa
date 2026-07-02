"""COACD-indexed contact geometry for the OpenDrawer contact planner."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree

from robocasa.demos.scene_process import CoACDConfig, _decompose, _geom_mesh_in_body


@dataclass
class IndexedSurfacePoints:
    points: np.ndarray
    normals: np.ndarray
    part_ids: np.ndarray
    face_ids: np.ndarray
    global_face_ids: np.ndarray
    tree: cKDTree


@dataclass
class ConvexPart:
    vertices: np.ndarray
    faces: np.ndarray
    equations: np.ndarray
    part_id: int
    face_offset: int


@dataclass
class FeasibleSurfaceGraph:
    points: IndexedSurfacePoints
    edges: np.ndarray
    adjacency_indptr: np.ndarray
    adjacency_indices: np.ndarray

    def neighbors(self, index: int) -> np.ndarray:
        start = int(self.adjacency_indptr[int(index)])
        end = int(self.adjacency_indptr[int(index) + 1])
        return self.adjacency_indices[start:end]


@dataclass
class GripperModeGeometry:
    mode: str
    ee_pose_wxyz: np.ndarray
    finger_joint_qpos_ids: np.ndarray
    finger_joint_qpos: np.ndarray
    sampled_finger_points_ee: IndexedSurfacePoints
    finger_ids: np.ndarray
    contact_group_ids: np.ndarray
    convex_parts: list[ConvexPart]


@dataclass
class ContactGeometryCache:
    cache_dir: Path
    object_body_id: int
    object_parts: list[ConvexPart]
    feasible_graph: FeasibleSurfaceGraph
    gripper_modes: dict[str, GripperModeGeometry]


def _raw_model(model: Any) -> mujoco.MjModel:
    return getattr(model, "_model", model)


def _raw_data(data: Any) -> mujoco.MjData:
    return getattr(data, "_data", data)


def _transform_mesh(
    mesh: trimesh.Trimesh,
    position: np.ndarray,
    rotation: np.ndarray,
) -> trimesh.Trimesh:
    result = mesh.copy()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    result.apply_transform(transform)
    return result


def _mesh_in_reference_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    reference_position: np.ndarray,
    reference_rotation: np.ndarray,
) -> trimesh.Trimesh | None:
    mesh = _geom_mesh_in_body(model, int(geom_id))
    if mesh is None:
        return None
    body_id = int(model.geom_bodyid[int(geom_id)])
    mesh_world = _transform_mesh(
        mesh,
        np.asarray(data.xpos[body_id], dtype=np.float64),
        np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3),
    )
    world_to_reference = np.eye(4, dtype=np.float64)
    world_to_reference[:3, :3] = reference_rotation.T
    world_to_reference[:3, 3] = -reference_rotation.T @ reference_position
    mesh_world.apply_transform(world_to_reference)
    return mesh_world


def _merge_geoms_in_reference_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: Iterable[int],
    reference_position: np.ndarray,
    reference_rotation: np.ndarray,
) -> trimesh.Trimesh:
    meshes = []
    for geom_id in geom_ids:
        mesh = _mesh_in_reference_frame(
            model,
            data,
            int(geom_id),
            reference_position,
            reference_rotation,
        )
        if mesh is not None and mesh.faces.size:
            meshes.append(mesh)
    if not meshes:
        raise RuntimeError("No mesh geometry was available for contact preprocessing")
    merged = trimesh.util.concatenate(meshes)
    merged.remove_unreferenced_vertices()
    return merged


def _deduplicate_planes(equations: np.ndarray, decimals: int = 6) -> np.ndarray:
    equations = np.asarray(equations, dtype=np.float64).reshape(-1, 4)
    normalized = []
    for equation in equations:
        scale = max(float(np.linalg.norm(equation[:3])), 1e-12)
        normalized.append(equation / scale)
    normalized = np.asarray(normalized, dtype=np.float64)
    _, unique = np.unique(
        np.round(normalized, decimals=decimals),
        axis=0,
        return_index=True,
    )
    return normalized[np.sort(unique)]


def decompose_mesh(
    mesh: trimesh.Trimesh,
    *,
    seed: int,
    config: CoACDConfig | None = None,
) -> list[ConvexPart]:
    parts = _decompose(mesh, config or CoACDConfig(), int(seed))
    result = []
    face_offset = 0
    for part_id, part in enumerate(parts):
        vertices = np.asarray(part.vertices, dtype=np.float64)
        faces = np.asarray(part.faces, dtype=np.int64)
        if vertices.shape[0] < 4 or faces.shape[0] < 4:
            continue
        hull = ConvexHull(vertices)
        equations = _deduplicate_planes(hull.equations)
        result.append(
            ConvexPart(
                vertices=vertices,
                faces=faces,
                equations=equations,
                part_id=int(part_id),
                face_offset=int(face_offset),
            )
        )
        face_offset += int(faces.shape[0])
    if not result:
        raise RuntimeError("COACD did not produce a usable convex decomposition")
    return result


def _part_face_centers(parts: Sequence[ConvexPart]):
    centers = []
    normals = []
    part_ids = []
    face_ids = []
    global_face_ids = []
    for part in parts:
        triangles = part.vertices[part.faces]
        centers.append(triangles.mean(axis=1))
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        cross /= np.linalg.norm(cross, axis=1, keepdims=True).clip(min=1e-12)
        normals.append(cross)
        count = triangles.shape[0]
        part_ids.append(np.full(count, part.part_id, dtype=np.int32))
        face_ids.append(np.arange(count, dtype=np.int32))
        global_face_ids.append(part.face_offset + np.arange(count, dtype=np.int32))
    return (
        np.concatenate(centers, axis=0),
        np.concatenate(normals, axis=0),
        np.concatenate(part_ids, axis=0),
        np.concatenate(face_ids, axis=0),
        np.concatenate(global_face_ids, axis=0),
    )


def index_surface_points(
    points: np.ndarray,
    parts: Sequence[ConvexPart],
    *,
    project: bool = False,
) -> IndexedSurfacePoints:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centers, normals, part_ids, face_ids, global_face_ids = _part_face_centers(parts)
    center_tree = cKDTree(centers)
    neighbor_count = min(8, centers.shape[0])
    _, nearest_candidates = center_tree.query(points, k=neighbor_count)
    nearest_candidates = np.asarray(nearest_candidates, dtype=np.int64).reshape(
        points.shape[0], -1
    )
    all_triangles = np.concatenate(
        [part.vertices[part.faces] for part in parts],
        axis=0,
    )
    selected = np.empty(points.shape[0], dtype=np.int64)
    projected = np.empty_like(points)
    for point_index, candidate_faces in enumerate(nearest_candidates):
        triangles = all_triangles[candidate_faces]
        repeated = np.repeat(
            points[point_index : point_index + 1], triangles.shape[0], axis=0
        )
        closest = trimesh.triangles.closest_point(triangles, repeated)
        distance = np.linalg.norm(closest - points[point_index], axis=1)
        local_best = int(np.argmin(distance))
        selected[point_index] = int(candidate_faces[local_best])
        projected[point_index] = closest[local_best]
    nearest = selected
    indexed_points = projected if project else points
    return IndexedSurfacePoints(
        points=indexed_points,
        normals=normals[nearest],
        part_ids=part_ids[nearest],
        face_ids=face_ids[nearest],
        global_face_ids=global_face_ids[nearest],
        tree=cKDTree(indexed_points),
    )


def farthest_point_sample(
    points: np.ndarray,
    count: int,
    *,
    seed_index: int | None = None,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    count = min(max(int(count), 0), points.shape[0])
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    if seed_index is None:
        center = points.mean(axis=0)
        seed_index = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(seed_index)
    min_distance = np.linalg.norm(points - points[selected[0]], axis=1)
    for sample_index in range(1, count):
        selected[sample_index] = int(np.argmax(min_distance))
        min_distance = np.minimum(
            min_distance,
            np.linalg.norm(points - points[selected[sample_index]], axis=1),
        )
    return selected


def _sample_mesh_dense(
    mesh: trimesh.Trimesh,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    area = np.linalg.norm(cross, axis=1)
    valid = area > 1e-12
    triangles = triangles[valid]
    cross = cross[valid]
    area = area[valid]
    face_ids = rng.choice(
        triangles.shape[0],
        size=max(int(count), 1),
        replace=True,
        p=area / area.sum(),
    )
    chosen = triangles[face_ids]
    u = np.sqrt(rng.random((face_ids.size, 1)))
    v = rng.random((face_ids.size, 1))
    points = (
        (1.0 - u) * chosen[:, 0] + u * (1.0 - v) * chosen[:, 1] + u * v * chosen[:, 2]
    )
    normals = cross[face_ids]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    return points, normals


def _sample_finger_points(
    finger_meshes: Sequence[trimesh.Trimesh],
    parts: Sequence[ConvexPart],
    *,
    mode: str,
    count_per_finger: int,
    seed: int,
) -> tuple[IndexedSurfacePoints, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    centroids = np.asarray([mesh.centroid for mesh in finger_meshes], dtype=np.float64)
    selected_points = []
    selected_finger_ids = []
    for finger_id, mesh in enumerate(finger_meshes):
        dense_points, dense_normals = _sample_mesh_dense(
            mesh,
            max(1000, int(count_per_finger) * 80),
            rng,
        )
        if mode == "gripper_open" and len(finger_meshes) == 2:
            other = centroids[1 - finger_id]
            inward = other - centroids[finger_id]
            inward /= max(float(np.linalg.norm(inward)), 1e-12)
            inward_projection = (dense_points - centroids[finger_id]) @ inward
            inner_threshold = np.quantile(inward_projection, 0.8)
            inner = np.flatnonzero(
                (dense_normals @ inward > 0.45) | (inward_projection >= inner_threshold)
            )
            inner_count = min(max(int(count_per_finger) // 2, 1), inner.size)
            inner_choice = (
                inner[farthest_point_sample(dense_points[inner], inner_count)]
                if inner_count
                else np.zeros(0, dtype=np.int64)
            )
            remaining = max(int(count_per_finger) - inner_choice.size, 0)
            available = np.setdiff1d(
                np.arange(dense_points.shape[0], dtype=np.int64),
                inner_choice,
                assume_unique=False,
            )
            outer_choice = (
                available[farthest_point_sample(dense_points[available], remaining)]
                if remaining
                else np.zeros(0, dtype=np.int64)
            )
            choice = np.concatenate([inner_choice, outer_choice])
        else:
            choice = farthest_point_sample(dense_points, int(count_per_finger))
        selected_points.append(dense_points[choice])
        selected_finger_ids.append(np.full(choice.size, finger_id, dtype=np.int32))
    points = np.concatenate(selected_points, axis=0)
    finger_ids = np.concatenate(selected_finger_ids, axis=0)
    indexed = index_surface_points(points, parts)
    contact_group_ids = (
        finger_ids.copy()
        if mode == "gripper_open"
        else np.zeros(finger_ids.shape[0], dtype=np.int32)
    )
    return indexed, finger_ids, contact_group_ids


def _strict_segment_intersects_convex(
    start: np.ndarray,
    end: np.ndarray,
    equations: np.ndarray,
    *,
    margin: float,
) -> bool:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    equations = np.asarray(equations, dtype=np.float64).reshape(-1, 4)
    value0 = equations[:, :3] @ start + equations[:, 3] + float(margin)
    delta = equations[:, :3] @ (end - start)
    lower = 0.0
    upper = 1.0
    for offset, slope in zip(value0, delta):
        if abs(float(slope)) < 1e-12:
            if offset > 0.0:
                return False
            continue
        crossing = -float(offset) / float(slope)
        if slope > 0.0:
            upper = min(upper, crossing)
        else:
            lower = max(lower, crossing)
        if lower >= upper:
            return False
    epsilon = 1e-7
    return max(lower, epsilon) < min(upper, 1.0 - epsilon)


def edge_penetrates_parts(
    start: np.ndarray,
    end: np.ndarray,
    parts: Sequence[ConvexPart],
    *,
    margin: float = 1e-5,
) -> bool:
    return any(
        _strict_segment_intersects_convex(
            start,
            end,
            part.equations,
            margin=margin,
        )
        for part in parts
    )


def _candidate_edges(points: np.ndarray) -> set[tuple[int, int]]:
    count = int(points.shape[0])
    edges: set[tuple[int, int]] = set()
    if count < 2:
        return edges
    if count >= 4:
        try:
            simplices = Delaunay(points, qhull_options="QJ Qbb Qc").simplices
            for simplex in np.asarray(simplices, dtype=np.int64):
                for row in range(simplex.size):
                    for col in range(row + 1, simplex.size):
                        a, b = sorted((int(simplex[row]), int(simplex[col])))
                        edges.add((a, b))
        except QhullError:
            pass
    if not edges:
        tree = cKDTree(points)
        neighbor_count = min(5, count)
        _, neighbors = tree.query(points, k=neighbor_count)
        neighbors = np.asarray(neighbors, dtype=np.int64).reshape(count, -1)
        for index, row in enumerate(neighbors):
            for neighbor in row[1:]:
                a, b = sorted((int(index), int(neighbor)))
                if a != b:
                    edges.add((a, b))
    return edges


def build_feasible_surface_graph(
    feasible_points_object: np.ndarray,
    object_parts: Sequence[ConvexPart],
    *,
    edge_margin: float = 1e-5,
) -> FeasibleSurfaceGraph:
    indexed = index_surface_points(
        feasible_points_object,
        object_parts,
        project=True,
    )
    edges = [
        edge
        for edge in sorted(_candidate_edges(indexed.points))
        if not edge_penetrates_parts(
            indexed.points[edge[0]],
            indexed.points[edge[1]],
            object_parts,
            margin=edge_margin,
        )
    ]
    if indexed.points.shape[0] > 1 and not edges:
        raise RuntimeError(
            "No non-penetrating edge could be built between feasible contact points"
        )
    adjacency = [[] for _ in range(indexed.points.shape[0])]
    for first, second in edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    indptr = [0]
    flat = []
    for neighbors in adjacency:
        flat.extend(sorted(set(neighbors)))
        indptr.append(len(flat))
    return FeasibleSurfaceGraph(
        points=indexed,
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        adjacency_indptr=np.asarray(indptr, dtype=np.int64),
        adjacency_indices=np.asarray(flat, dtype=np.int64),
    )


def sample_connected_contact_sets(
    graph: FeasibleSurfaceGraph,
    *,
    count: int,
    min_size: int,
    max_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if graph.edges.shape[0] == 0:
        raise RuntimeError("Contact-set sampling requires at least one graph edge")
    rng = np.random.default_rng(int(seed))
    count = max(int(count), 1)
    min_size = max(int(min_size), 2)
    max_size = max(int(max_size), min_size)
    point_sets = np.zeros((count, max_size, 3), dtype=np.float64)
    masks = np.zeros((count, max_size), dtype=bool)
    indices = np.full((count, max_size), -1, dtype=np.int64)
    for sample_index in range(count):
        edge = graph.edges[int(rng.integers(0, graph.edges.shape[0]))]
        selected = [int(edge[0]), int(edge[1])]
        target_size = int(rng.integers(min_size, max_size + 1))
        while len(selected) < target_size:
            frontier = sorted(
                {
                    int(neighbor)
                    for node in selected
                    for neighbor in graph.neighbors(node)
                    if int(neighbor) not in selected
                }
            )
            if not frontier:
                break
            selected.append(frontier[int(rng.integers(0, len(frontier)))])
        size = len(selected)
        indices[sample_index, :size] = selected
        point_sets[sample_index, :size] = graph.points.points[selected]
        masks[sample_index, :size] = True
    return point_sets, masks, indices


def convex_equation_tensor(
    parts: Sequence[ConvexPart],
) -> tuple[np.ndarray, np.ndarray]:
    max_planes = max(part.equations.shape[0] for part in parts)
    equations = np.zeros((len(parts), max_planes, 4), dtype=np.float32)
    mask = np.zeros((len(parts), max_planes), dtype=bool)
    for index, part in enumerate(parts):
        count = part.equations.shape[0]
        equations[index, :count] = part.equations.astype(np.float32)
        mask[index, :count] = True
    return equations, mask


def _mode_joint_values(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    qpos_ids = []
    values = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Cannot find gripper joint {name!r}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
        if mode == "gripper_open":
            value = joint_range[np.argmax(np.abs(joint_range))]
        elif mode == "gripper_closed":
            value = joint_range[np.argmin(np.abs(joint_range))]
        else:
            raise ValueError(f"Unknown gripper mode {mode!r}")
        values.append(float(value))
    return np.asarray(qpos_ids, dtype=np.int64), np.asarray(values, dtype=np.float64)


def _geom_ids_for_body_names(
    model: mujoco.MjModel,
    body_names: Sequence[str],
) -> list[list[int]]:
    result = []
    for body_name in body_names:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        if body_id < 0:
            raise RuntimeError(f"Cannot find finger body {body_name!r}")
        geom_ids = np.flatnonzero(np.asarray(model.geom_bodyid) == body_id)
        collision = [
            int(geom_id)
            for geom_id in geom_ids
            if int(model.geom_group[int(geom_id)]) != 1
            and int(model.geom_type[int(geom_id)])
            not in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)
        ]
        result.append(collision or [int(geom_id) for geom_id in geom_ids])
    return result


def _save_indexed_points(path: Path, points: IndexedSurfacePoints) -> None:
    np.savez_compressed(
        path,
        points=points.points,
        normals=points.normals,
        part_ids=points.part_ids,
        face_ids=points.face_ids,
        global_face_ids=points.global_face_ids,
    )
    with path.with_suffix(".kdtree.pkl").open("wb") as stream:
        pickle.dump(points.tree, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _geometry_digest(
    model: mujoco.MjModel,
    object_geom_ids: Sequence[int],
    feasible_points_object: np.ndarray,
    ee_pose_wxyz: np.ndarray,
    settings: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(object_geom_ids, dtype=np.int32).tobytes())
    digest.update(np.asarray(feasible_points_object, dtype=np.float64).tobytes())
    digest.update(np.asarray(ee_pose_wxyz, dtype=np.float64).tobytes())
    digest.update(np.asarray(model.geom_size, dtype=np.float32).tobytes())
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:20]


def build_contact_geometry_cache(
    env: Any,
    *,
    object_body_id: int,
    object_geom_ids: Sequence[int],
    feasible_points_world: np.ndarray,
    demonstration_ee_pose_wxyz: np.ndarray,
    ee_site_name: str,
    cache_root: str | Path,
    seed: int,
    finger_points_per_finger: int = 20,
    edge_margin: float = 1e-5,
    force: bool = False,
) -> ContactGeometryCache:
    model = _raw_model(env.sim.model)
    data = _raw_data(env.sim.data)
    ee_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        ee_site_name,
    )
    if ee_site_id < 0:
        raise RuntimeError(f"Cannot find EE site {ee_site_name!r}")
    object_position = np.asarray(data.xpos[object_body_id], dtype=np.float64)
    object_rotation = np.asarray(data.xmat[object_body_id], dtype=np.float64).reshape(
        3, 3
    )
    feasible_points_object = (
        np.asarray(feasible_points_world, dtype=np.float64) - object_position
    ) @ object_rotation
    settings = {
        "version": 2,
        "seed": int(seed),
        "finger_points_per_finger": int(finger_points_per_finger),
        "edge_margin": float(edge_margin),
        "ee_site_name": str(ee_site_name),
    }
    digest = _geometry_digest(
        model,
        object_geom_ids,
        feasible_points_object,
        demonstration_ee_pose_wxyz,
        settings,
    )
    cache_dir = Path(cache_root) / digest
    bundle_path = cache_dir / "geometry.pkl"
    if bundle_path.exists() and not force:
        with bundle_path.open("rb") as stream:
            cached = pickle.load(stream)
        for indexed in [
            cached.feasible_graph.points,
            *(mode.sampled_finger_points_ee for mode in cached.gripper_modes.values()),
        ]:
            indexed.tree = cKDTree(indexed.points)
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    object_mesh = _merge_geoms_in_reference_frame(
        model,
        data,
        object_geom_ids,
        object_position,
        object_rotation,
    )
    object_parts = decompose_mesh(object_mesh, seed=int(seed))
    feasible_graph = build_feasible_surface_graph(
        feasible_points_object,
        object_parts,
        edge_margin=float(edge_margin),
    )

    joint_names = (
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
    )
    finger_body_names = (
        "gripper0_right_leftfinger",
        "gripper0_right_rightfinger",
    )
    finger_geom_ids = _geom_ids_for_body_names(model, finger_body_names)
    saved_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    gripper_modes = {}
    try:
        for mode_index, mode in enumerate(("gripper_open", "gripper_closed")):
            qpos_ids, joint_values = _mode_joint_values(model, joint_names, mode)
            data.qpos[qpos_ids] = joint_values
            mujoco.mj_forward(model, data)
            ee_position = np.asarray(data.site_xpos[ee_site_id], dtype=np.float64)
            ee_rotation = np.asarray(
                data.site_xmat[ee_site_id], dtype=np.float64
            ).reshape(3, 3)
            finger_meshes = [
                _merge_geoms_in_reference_frame(
                    model,
                    data,
                    geom_ids,
                    ee_position,
                    ee_rotation,
                )
                for geom_ids in finger_geom_ids
            ]
            decomposition_mesh = trimesh.util.concatenate(finger_meshes)
            decomposition_mesh.remove_unreferenced_vertices()
            if mode == "gripper_closed":
                # Closed fingers overlap at the pads. Their geometric union is
                # represented as one watertight whole before COACD.
                decomposition_mesh = decomposition_mesh.convex_hull
            parts = decompose_mesh(
                decomposition_mesh,
                seed=int(seed) + 100 + mode_index,
            )
            indexed, finger_ids, contact_group_ids = _sample_finger_points(
                finger_meshes,
                parts,
                mode=mode,
                count_per_finger=int(finger_points_per_finger),
                seed=int(seed) + 1000 + mode_index,
            )
            gripper_modes[mode] = GripperModeGeometry(
                mode=mode,
                ee_pose_wxyz=np.asarray(
                    demonstration_ee_pose_wxyz,
                    dtype=np.float64,
                ).reshape(7),
                finger_joint_qpos_ids=qpos_ids,
                finger_joint_qpos=joint_values,
                sampled_finger_points_ee=indexed,
                finger_ids=finger_ids,
                contact_group_ids=contact_group_ids,
                convex_parts=parts,
            )
    finally:
        data.qpos[:] = saved_qpos
        mujoco.mj_forward(model, data)

    cache = ContactGeometryCache(
        cache_dir=cache_dir,
        object_body_id=int(object_body_id),
        object_parts=object_parts,
        feasible_graph=feasible_graph,
        gripper_modes=gripper_modes,
    )
    _save_indexed_points(cache_dir / "feasible_points.npz", feasible_graph.points)
    for mode, geometry in gripper_modes.items():
        _save_indexed_points(
            cache_dir / f"{mode}_finger_points.npz",
            geometry.sampled_finger_points_ee,
        )
    manifest = {
        **settings,
        "object_body_id": int(object_body_id),
        "object_part_count": len(object_parts),
        "feasible_point_count": int(feasible_graph.points.points.shape[0]),
        "edge_count": int(feasible_graph.edges.shape[0]),
        "modes": {
            mode: {
                "finger_point_count": int(
                    geometry.sampled_finger_points_ee.points.shape[0]
                ),
                "finger_part_count": len(geometry.convex_parts),
                "finger_joint_qpos": geometry.finger_joint_qpos.tolist(),
            }
            for mode, geometry in gripper_modes.items()
        },
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with bundle_path.open("wb") as stream:
        pickle.dump(cache, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


__all__ = [
    "ContactGeometryCache",
    "ConvexPart",
    "FeasibleSurfaceGraph",
    "GripperModeGeometry",
    "IndexedSurfacePoints",
    "build_contact_geometry_cache",
    "build_feasible_surface_graph",
    "convex_equation_tensor",
    "edge_penetrates_parts",
    "farthest_point_sample",
    "sample_connected_contact_sets",
]
