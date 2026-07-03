import numpy as np
import trimesh


def surface_contact_sample_count(surface, args):
    if args.contact_sample_count is not None:
        return max(int(args.contact_sample_count), 1)
    if surface.name == "handle":
        return max(int(args.handle_contact_sample_count), 1)
    return max(int(args.panel_contact_sample_count), 1)


def project_contact_samples_to_actual_geoms(
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


_surface_contact_sample_count = surface_contact_sample_count
_project_contact_samples_to_actual_geoms = project_contact_samples_to_actual_geoms


__all__ = [
    "project_contact_samples_to_actual_geoms",
    "surface_contact_sample_count",
    "_project_contact_samples_to_actual_geoms",
    "_surface_contact_sample_count",
]
