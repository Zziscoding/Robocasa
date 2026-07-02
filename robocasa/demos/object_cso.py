"""Object contact-set utilities for drawer contact optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class FeasibleContactCache:
    candidate_indices: np.ndarray
    positions_object: np.ndarray
    positions_world: np.ndarray
    normals_object: np.ndarray
    normals_world: np.ndarray
    tangents_object: np.ndarray
    tangents_world: np.ndarray
    tree: cKDTree


@dataclass
class ObjectRepresentativePoints:
    points_world: np.ndarray
    normals_world: np.ndarray
    geom_ids: np.ndarray


def _normalize(vec, fallback=None):
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        return arr / norm
    if fallback is None:
        fallback = np.zeros_like(arr)
        fallback[0] = 1.0
    return np.asarray(fallback, dtype=np.float64).reshape(arr.shape)


def farthest_point_subset(points, count, initial_index=0):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    count = min(max(int(count), 1), points.shape[0])
    selected = [int(np.clip(initial_index, 0, points.shape[0] - 1))]
    min_distance = np.linalg.norm(points - points[selected[0]], axis=1)
    while len(selected) < count:
        min_distance[selected] = -np.inf
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        min_distance = np.minimum(
            min_distance,
            np.linalg.norm(points - points[next_index], axis=1),
        )
    return np.asarray(selected, dtype=np.int64)


def allocate_surface_samples(areas, total_surface_points, min_points_per_geom):
    areas = np.asarray(areas, dtype=np.float64)
    geom_count = int(areas.size)
    if geom_count == 0:
        return np.zeros(0, dtype=np.int64)
    total_surface_points = max(int(total_surface_points), geom_count)
    min_points_per_geom = max(int(min_points_per_geom), 1)
    if min_points_per_geom * geom_count > total_surface_points:
        min_points_per_geom = max(total_surface_points // geom_count, 1)
    counts = np.full(geom_count, min_points_per_geom, dtype=np.int64)
    remaining = max(total_surface_points - int(counts.sum()), 0)
    if remaining == 0:
        return counts
    weights = np.maximum(areas, 1e-12)
    weights /= weights.sum()
    exact = weights * remaining
    extra = np.floor(exact).astype(np.int64)
    counts += extra
    leftover = remaining - int(extra.sum())
    if leftover > 0:
        order = np.argsort(-(exact - extra))
        counts[order[:leftover]] += 1
    return counts


def sample_object_representative_points(env, surface, candidates, args):
    """Sample deterministic oriented surface points from the drawer object."""
    from robocasa.demos.scene_process import (
        _geom_mesh_in_body,
        _uniform_surface_samples,
    )

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    drawer_prefix = f"{env.drawer.name}_"
    geom_name_by_id = {
        int(geom_id): str(name) for name, geom_id in env.sim.model._geom_name2id.items()
    }
    geom_ids = [
        geom_id
        for geom_id in range(int(raw_model.ngeom))
        if geom_name_by_id.get(geom_id, "").startswith(drawer_prefix)
        and float(raw_model.geom_rgba[geom_id, 3]) > 1e-5
    ]
    collision_geom_ids = [
        geom_id for geom_id in geom_ids if int(raw_model.geom_group[geom_id]) == 0
    ]
    if collision_geom_ids:
        geom_ids = collision_geom_ids
    meshes = []
    mesh_geom_ids = []
    for geom_id in geom_ids:
        mesh_body = _geom_mesh_in_body(raw_model, int(geom_id))
        if mesh_body is None or not mesh_body.faces.size:
            continue
        body_id = int(raw_model.geom_bodyid[int(geom_id)])
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
        mesh_geom_ids.append(int(geom_id))
    if not meshes:
        raise RuntimeError(
            "No drawer geometry was available for representative-point sampling"
        )

    sample_counts = allocate_surface_samples(
        [float(mesh.area) for mesh in meshes],
        total_surface_points=int(args.object_representative_point_count),
        min_points_per_geom=int(args.object_representative_min_per_geom),
    )
    rng = np.random.default_rng(int(args.seed))
    points = []
    normals = []
    sampled_geom_ids = []
    for mesh, geom_id, count in zip(meshes, mesh_geom_ids, sample_counts):
        sampled_points, sampled_normals = _uniform_surface_samples(
            mesh, int(count), rng
        )
        geom_center = np.asarray(
            raw_data.geom_xpos[int(geom_id)], dtype=np.float64
        ).reshape(3)
        radial = sampled_points.astype(np.float64) - geom_center
        flip = np.sum(radial * sampled_normals, axis=1) < 0.0
        sampled_normals[flip] *= -1.0
        points.append(sampled_points.astype(np.float64))
        normals.append(sampled_normals.astype(np.float64))
        sampled_geom_ids.append(
            np.full(sampled_points.shape[0], int(geom_id), dtype=np.int64)
        )

    for candidate in [
        candidate for candidate in candidates if bool(candidate.feasible)
    ]:
        points.append(np.asarray(candidate.world_point, dtype=np.float64).reshape(1, 3))
        normals.append(
            _normalize(np.asarray(candidate.approach_world, dtype=np.float64)).reshape(
                1, 3
            )
        )
        sampled_geom_ids.append(np.asarray([-1], dtype=np.int64))

    return ObjectRepresentativePoints(
        points_world=np.concatenate(points, axis=0),
        normals_world=np.concatenate(normals, axis=0),
        geom_ids=np.concatenate(sampled_geom_ids, axis=0),
    )


def _matrix_from_quat_wxyz(quat):
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def sphere_centers_world_from_pose(ee_pose_wxyz, sphere_model):
    pose = np.asarray(ee_pose_wxyz, dtype=np.float64).reshape(7)
    rotation = _matrix_from_quat_wxyz(pose[3:])
    return np.asarray(sphere_model.centers_ee, dtype=np.float64) @ rotation.T + pose[:3]


def point_sdf_for_spheres_numpy(
    sphere_centers_world, sphere_radii, representative_points
):
    centers = np.asarray(sphere_centers_world, dtype=np.float64).reshape(-1, 3)
    points = np.asarray(representative_points.points_world, dtype=np.float64).reshape(
        -1, 3
    )
    normals = np.asarray(representative_points.normals_world, dtype=np.float64).reshape(
        -1, 3
    )
    distances, nearest = cKDTree(points).query(centers, k=1)
    deltas = centers - points[nearest]
    normal_projection = np.sum(deltas * normals[nearest], axis=1)
    signed_center_distance = np.where(normal_projection >= 0.0, distances, -distances)
    return signed_center_distance - np.asarray(sphere_radii, dtype=np.float64).reshape(
        -1
    )


def build_feasible_contact_cache(surface, candidates) -> FeasibleContactCache:
    candidate_indices = np.asarray(
        [i for i, candidate in enumerate(candidates) if bool(candidate.feasible)],
        dtype=np.int64,
    )
    if candidate_indices.size == 0:
        raise RuntimeError("No feasible contact candidates available")
    positions_world = np.asarray(
        [candidates[i].world_point for i in candidate_indices],
        dtype=np.float64,
    )
    normals_world = np.asarray(
        [_normalize(candidates[i].approach_world) for i in candidate_indices],
        dtype=np.float64,
    )
    surface_rotation = np.asarray(surface.rotation_world, dtype=np.float64)
    positions_object = (
        positions_world - np.asarray(surface.center_world, dtype=np.float64)
    ) @ surface_rotation
    normals_object = normals_world @ surface_rotation
    tangents_world = []
    tangents_object = []
    for normal_world, normal_object in zip(normals_world, normals_object):
        ref_world = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(normal_world[2])) < 0.95
            else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        )
        tangents_world.append(
            _normalize(
                np.cross(ref_world, normal_world), fallback=np.array([1.0, 0.0, 0.0])
            )
        )
        ref_object = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(normal_object[2])) < 0.95
            else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        )
        tangents_object.append(
            _normalize(
                np.cross(ref_object, normal_object), fallback=np.array([1.0, 0.0, 0.0])
            )
        )
    return FeasibleContactCache(
        candidate_indices=candidate_indices,
        positions_object=positions_object,
        positions_world=positions_world,
        normals_object=normals_object,
        normals_world=normals_world,
        tangents_object=np.asarray(tangents_object, dtype=np.float64),
        tangents_world=np.asarray(tangents_world, dtype=np.float64),
        tree=cKDTree(positions_object),
    )


def select_contact_set_from_cache(cache: FeasibleContactCache, args) -> np.ndarray:
    radius = float(getattr(args, "q_config_mpc_contact_set_neighbor_radius", 0.025))
    min_neighbors = max(
        int(getattr(args, "q_config_mpc_contact_set_min_neighbors", 2)),
        1,
    )
    neighbor_counts = np.asarray(
        [
            len(cache.tree.query_ball_point(point, r=radius))
            for point in cache.positions_object
        ],
        dtype=np.int64,
    )
    interior = np.flatnonzero(neighbor_counts >= min_neighbors)
    if interior.size == 0:
        interior = np.arange(cache.positions_object.shape[0], dtype=np.int64)
    limit = int(getattr(args, "q_config_mpc_contact_set_size", 0) or 0)
    if limit > 0 and interior.size > limit:
        seed_index = int(np.argmax(neighbor_counts[interior]))
        subset = farthest_point_subset(
            cache.positions_object[interior],
            count=limit,
            initial_index=seed_index,
        )
        interior = interior[subset]
    return np.asarray(interior, dtype=np.int64)


__all__ = [
    "FeasibleContactCache",
    "ObjectRepresentativePoints",
    "allocate_surface_samples",
    "build_feasible_contact_cache",
    "farthest_point_subset",
    "point_sdf_for_spheres_numpy",
    "sample_object_representative_points",
    "select_contact_set_from_cache",
    "sphere_centers_world_from_pose",
]
