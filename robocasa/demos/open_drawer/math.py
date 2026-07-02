import numpy as np


def _rotation_matrix_to_quat_wxyz(rotation):
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(
        np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    ).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _matrix_from_quat_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _ee_pose_contact_points_world(ee_poses_wxyz, contact_offset_ee):
    poses = np.asarray(ee_poses_wxyz, dtype=np.float64).reshape(-1, 7)
    contact_offset_ee = np.asarray(contact_offset_ee, dtype=np.float64).reshape(3)
    contact_points = []
    for pose in poses:
        rotation = _matrix_from_quat_wxyz(pose[3:])
        contact_points.append(pose[:3] + rotation @ contact_offset_ee)
    return np.asarray(contact_points, dtype=np.float64)


def _nearest_point_distances(points, targets):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    if targets.shape[0] == 0:
        return (
            np.full(points.shape[0], np.inf, dtype=np.float64),
            np.full(points.shape[0], -1, dtype=np.int64),
        )
    distances = np.linalg.norm(points[:, None, :] - targets[None, :, :], axis=-1)
    nearest_indices = np.argmin(distances, axis=1)
    return distances[np.arange(points.shape[0]), nearest_indices], nearest_indices


def _score_ee_pose_contacts(
    ee_poses_wxyz,
    contact_offset_ee,
    feasible_points_world,
    max_distance,
):
    contact_points = _ee_pose_contact_points_world(
        ee_poses_wxyz,
        contact_offset_ee,
    )
    distances, indices = _nearest_point_distances(
        contact_points,
        feasible_points_world,
    )
    mask = distances <= float(max_distance)
    return {
        "contact_points_world": contact_points,
        "nearest_feasible_distances": distances,
        "nearest_feasible_indices": indices,
        "feasible_mask": mask,
        "feasible_fraction": float(mask.mean()) if mask.size else 0.0,
    }


def _score_ee_pose_contact_targets(
    ee_poses_wxyz,
    contact_offset_ee,
    target_points_world,
    max_distance,
):
    contact_points = _ee_pose_contact_points_world(
        ee_poses_wxyz,
        contact_offset_ee,
    )
    target_points = np.asarray(target_points_world, dtype=np.float64).reshape(-1, 3)
    if target_points.shape[0] == 1 and contact_points.shape[0] != 1:
        target_points = np.repeat(target_points, contact_points.shape[0], axis=0)
    if target_points.shape[0] != contact_points.shape[0]:
        raise ValueError(
            "target_points_world must contain one target or one target per EE pose"
        )
    distances = np.linalg.norm(contact_points - target_points, axis=1)
    mask = distances <= float(max_distance)
    return {
        "contact_points_world": contact_points,
        "target_points_world": target_points,
        "target_distances": distances,
        "target_mask": mask,
        "target_fraction": float(mask.mean()) if mask.size else 0.0,
    }


def _distance_summary(distances):
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return {"min": float("inf"), "median": float("inf"), "max": float("inf")}
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def _finite_quantiles(values, quantiles=(0.0, 0.25, 0.5, 0.75, 1.0)):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"q{int(q * 100):02d}": float("inf") for q in quantiles}
    return {f"q{int(q * 100):02d}": float(np.quantile(finite, q)) for q in quantiles}


def _array_value_or_nan(values, index):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float("nan")
    return float(values[int(index)])


def _snap_ee_pose_to_contact_target(
    ee_pose_wxyz, contact_target_world, contact_offset_ee
):
    pose = np.asarray(ee_pose_wxyz, dtype=np.float64).reshape(7).copy()
    rotation = _matrix_from_quat_wxyz(pose[3:])
    pose[:3] = np.asarray(contact_target_world, dtype=np.float64).reshape(
        3
    ) - rotation @ np.asarray(contact_offset_ee, dtype=np.float64).reshape(3)
    return pose


def _torch_quat_wxyz_to_matrix(quaternion):
    import torch

    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(-1)
    one = torch.ones_like(w)
    two = 2.0
    return torch.stack(
        (
            one - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            one - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            one - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _rotation_angle_error(rot_a, rot_b):
    delta = np.asarray(rot_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        rot_b, dtype=np.float64
    ).reshape(3, 3)
    cos_angle = (float(np.trace(delta)) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
