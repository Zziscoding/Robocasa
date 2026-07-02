"""Franka EE skeleton extraction, per-candidate pose solver, and visualization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from robocasa.demos.object_cso import farthest_point_subset


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EESkeleton:
    hand_box_half_extents_ee: np.ndarray  # (3,)
    hand_box_center_ee: np.ndarray  # (3,)
    hand_box_rotation_ee: np.ndarray  # (3, 3)
    left_finger_segment_ee: np.ndarray  # (2, 3)
    right_finger_segment_ee: np.ndarray  # (2, 3)
    finger_tip_offset_ee: np.ndarray  # (3,)


@dataclass
class SkeletonPose:
    ee_position: np.ndarray  # (3,)
    ee_rotation: np.ndarray  # (3, 3)
    contact_finger: str  # "left" | "right" | "hand"
    contact_point_world: np.ndarray
    contact_normal_world: np.ndarray
    qp_cost: float
    lift: float
    theta: float
    gripper_opening: float = 0.04
    contact_primitive: str = "left"


PANDA_DEFAULT_GRIPPER_OPENING = 0.04
PANDA_MAX_GRIPPER_OPENING = 0.08


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    t1 = ref - float(np.dot(ref, normal)) * normal
    t1 = _normalize(t1, [0.0, 1.0, 0.0])
    t2 = _normalize(np.cross(normal, t1), [0.0, 0.0, 1.0])
    return t1, t2


def _rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(1.0 + trace)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] >= m[1, 1]) and (m[0, 0] >= m[2, 2]):
        s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12))
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12))
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12))
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def _bbox_half_extents_from_mesh(raw_model, geom_id, fallback):
    """Mirror visualize_mujoco._mesh_bbox_size logic."""
    try:
        mesh_id = int(raw_model.geom_dataid[geom_id])
        vadr = int(raw_model.mesh_vertadr[mesh_id])
        vnum = int(raw_model.mesh_vertnum[mesh_id])
        verts = np.asarray(raw_model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)
        if verts.size:
            return verts
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Skeleton extraction
# ---------------------------------------------------------------------------


def _classify_panda_geom(raw_model, geom_id, body_id2name, geom_name):
    """Classify a panda hand/finger geom as 'left', 'right', 'hand', or None.

    Walks up the body tree starting at the geom's own body and returns the
    first match. Recognizes the franka_emika_panda naming
    (left_finger / right_finger / panda_leftfinger / panda_rightfinger /
    panda_hand) as well as the robosuite panda_gripper naming
    (leftfinger / rightfinger / right_gripper, with pad geoms living under
    finger_joint{1,2}_tip and geoms named finger1_* / finger2_* / hand_*).
    """

    def _match(name: str):
        n = name.lower()
        if "leftfinger" in n or "left_finger" in n or "finger1" in n or "joint1" in n:
            return "left"
        if "rightfinger" in n or "right_finger" in n or "finger2" in n or "joint2" in n:
            return "right"
        if "hand" in n or "gripper" in n:
            return "hand"
        return None

    geom_match = _match(geom_name)
    if geom_match is not None:
        return geom_match
    body_id = int(raw_model.geom_bodyid[int(geom_id)])
    seen = set()
    while body_id > 0 and body_id not in seen:
        seen.add(body_id)
        body_name = body_id2name.get(int(body_id), "")
        result = _match(body_name)
        if result is not None:
            return result
        body_id = int(raw_model.body_parentid[int(body_id)])
    return None


def build_panda_skeleton(env, ee_site_name: str) -> EESkeleton:
    """Build a Franka hand+fingers skeleton in the EE-site frame."""
    import mujoco

    from robocasa.demos import visualize_mujoco as viz_mj

    raw_model, raw_data = viz_mj._raw_model_data(env)
    _, _, body_id2name, geom_id2name = viz_mj._name_maps(env)
    body_ids = viz_mj._ghost_source_body_ids(env)
    site_pos, site_rot = viz_mj._site_pose(env, ee_site_name)

    hand_extents = None
    hand_center_ee = np.zeros(3, dtype=np.float64)
    hand_rot_ee = np.eye(3, dtype=np.float64)
    left_seg = None
    right_seg = None

    for geom_id in range(int(raw_model.ngeom)):
        body_id = int(raw_model.geom_bodyid[geom_id])
        geom_name = geom_id2name.get(geom_id, "")
        kind = _classify_panda_geom(raw_model, geom_id, body_id2name, geom_name)
        if body_id not in body_ids and kind is None:
            continue
        geom_type = int(raw_model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        geom_pos_w = np.asarray(raw_data.geom_xpos[geom_id], dtype=np.float64).reshape(
            3
        )
        geom_rot_w = np.asarray(raw_data.geom_xmat[geom_id], dtype=np.float64).reshape(
            3, 3
        )
        local_pos = site_rot.T @ (geom_pos_w - site_pos)
        local_rot = site_rot.T @ geom_rot_w

        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            verts = _bbox_half_extents_from_mesh(raw_model, geom_id, None)
            if verts is None:
                size = np.asarray(raw_model.geom_size[geom_id], dtype=np.float64).copy()
                half = np.maximum(
                    size, np.array([0.003, 0.003, 0.003], dtype=np.float64)
                )
                mesh_min = -half
                mesh_max = half
            else:
                mesh_min = verts.min(axis=0)
                mesh_max = verts.max(axis=0)
        else:
            size = np.asarray(raw_model.geom_size[geom_id], dtype=np.float64).copy()
            half = np.maximum(size, np.array([0.003, 0.003, 0.003], dtype=np.float64))
            mesh_min = -half
            mesh_max = half

        half_ext = 0.5 * (mesh_max - mesh_min)
        bbox_center_local = 0.5 * (mesh_max + mesh_min)
        # convert bbox center / extents to EE-site frame
        center_ee = local_pos + local_rot @ bbox_center_local

        # endpoints along the longest axis (in geom-local), expressed in EE frame.
        long_axis = int(np.argmax(half_ext))
        ep_local_a = bbox_center_local.copy()
        ep_local_b = bbox_center_local.copy()
        ep_local_a[long_axis] -= half_ext[long_axis]
        ep_local_b[long_axis] += half_ext[long_axis]
        ep_ee_a = local_pos + local_rot @ ep_local_a
        ep_ee_b = local_pos + local_rot @ ep_local_b

        if kind == "left":
            if left_seg is None or float(np.linalg.norm(ep_ee_b - ep_ee_a)) > float(
                np.linalg.norm(left_seg[1] - left_seg[0])
            ):
                left_seg = np.stack([ep_ee_a, ep_ee_b], axis=0)
        elif kind == "right":
            if right_seg is None or float(np.linalg.norm(ep_ee_b - ep_ee_a)) > float(
                np.linalg.norm(right_seg[1] - right_seg[0])
            ):
                right_seg = np.stack([ep_ee_a, ep_ee_b], axis=0)
        elif kind == "hand":
            if hand_extents is None or float(np.prod(half_ext)) > float(
                np.prod(hand_extents)
            ):
                hand_extents = half_ext.copy()
                hand_center_ee = center_ee.copy()
                hand_rot_ee = local_rot.copy()

    if hand_extents is None:
        hand_extents = np.array([0.03, 0.07, 0.02], dtype=np.float64)
    if left_seg is None:
        left_seg = np.array([[0.0, -0.04, 0.04], [0.0, -0.04, 0.1]], dtype=np.float64)
    if right_seg is None:
        right_seg = np.array([[0.0, 0.04, 0.04], [0.0, 0.04, 0.1]], dtype=np.float64)
    finger_tip_offset_ee = 0.5 * (left_seg[-1] + right_seg[-1])
    return EESkeleton(
        hand_box_half_extents_ee=hand_extents,
        hand_box_center_ee=hand_center_ee,
        hand_box_rotation_ee=hand_rot_ee,
        left_finger_segment_ee=left_seg,
        right_finger_segment_ee=right_seg,
        finger_tip_offset_ee=finger_tip_offset_ee,
    )


# ---------------------------------------------------------------------------
# Feasible point selection
# ---------------------------------------------------------------------------


def select_interior_feasible_points(feasible_cache, n_select: int, seed: int):
    positions = np.asarray(feasible_cache.positions_world, dtype=np.float64).reshape(
        -1, 3
    )
    normals = np.asarray(feasible_cache.normals_world, dtype=np.float64).reshape(-1, 3)
    is_edge = np.asarray(feasible_cache.is_edge, dtype=bool).reshape(-1)
    if positions.shape[0] == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )
    interior_mask = ~is_edge
    interior_ids = np.flatnonzero(interior_mask)
    # On a thin/handle-shaped feasible set (near 1-D strip), the Delaunay
    # convex hull tags nearly every point as an edge, collapsing the interior
    # to ~0 and biasing FPS toward the two ends. Fall back to the full
    # feasible set whenever the interior cannot supply the requested count.
    if interior_ids.size < int(n_select):
        interior_ids = np.arange(positions.shape[0], dtype=np.int64)
    n_select = int(max(1, min(n_select, interior_ids.size)))
    interior_pts = positions[interior_ids]
    rng = np.random.default_rng(int(seed))
    init = int(rng.integers(interior_pts.shape[0]))
    fps_local = farthest_point_subset(interior_pts, n_select, initial_index=init)
    local_ids = interior_ids[fps_local]
    return (
        local_ids.astype(np.int64),
        positions[local_ids],
        normals[local_ids],
    )


# ---------------------------------------------------------------------------
# Skeleton pose solver
# ---------------------------------------------------------------------------


def _segment_samples(p0: np.ndarray, p1: np.ndarray, count: int) -> np.ndarray:
    count = max(int(count), 2)
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
    return (1.0 - alpha) * p0[None, :] + alpha * p1[None, :]


def _box_corners(
    center: np.ndarray, rot: np.ndarray, half_ext: np.ndarray
) -> np.ndarray:
    signs = np.array(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    return center[None, :] + (signs * half_ext[None, :]) @ rot.T


def _signed_distance_to_convex(points: np.ndarray, equations: np.ndarray) -> np.ndarray:
    """For each convex part (H, 4), signed distance = max over planes of (n.p + d).
    Returns array of shape (P, points.shape[0]); negative means inside that convex part.
    """
    # equations: (P, H, 4) where each row is (nx, ny, nz, d) outward-normal form
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # (N,4)
    # plane_vals[p, h, n] = eq[p,h,:] @ pts_h[n]
    plane_vals = np.einsum("phk,nk->phn", equations, pts_h)
    inside_depth = plane_vals.max(axis=1)  # (P, N): negative = inside
    return inside_depth.min(axis=0)  # min over parts: most-penetrating part


def _finger_segments_with_opening(skeleton: EESkeleton, gripper_opening: float):
    """Apply symmetric finger spread to the skeleton's nominal finger segments.

    The spread direction is inferred directly from the two extracted finger
    segments (left_mid → right_mid), so this works for any gripper orientation
    in the EE-site frame regardless of the panda hand's own axis convention.
    `gripper_opening` is the *full* distance between finger midpoints; the
    rest-pose opening is the current separation in the EE frame.
    """
    left_mid = 0.5 * (
        skeleton.left_finger_segment_ee[0] + skeleton.left_finger_segment_ee[1]
    )
    right_mid = 0.5 * (
        skeleton.right_finger_segment_ee[0] + skeleton.right_finger_segment_ee[1]
    )
    sep_vec = left_mid - right_mid
    sep_norm = float(np.linalg.norm(sep_vec))
    if sep_norm > 1e-9:
        spread_dir_ee = sep_vec / sep_norm
        rest_opening = sep_norm
    else:
        # Fall back to the hand's y axis (legacy behavior) when the two
        # extracted segments coincide (e.g. fingers missing).
        spread_dir_ee = skeleton.hand_box_rotation_ee[:, 1]
        rest_opening = 0.0
    delta = 0.5 * (float(gripper_opening) - rest_opening)
    left_sign = +1.0  # by construction: left is on +spread_dir_ee side of right
    left_offset = left_sign * delta * spread_dir_ee
    right_offset = -left_sign * delta * spread_dir_ee
    left_seg = skeleton.left_finger_segment_ee + left_offset[None, :]
    right_seg = skeleton.right_finger_segment_ee + right_offset[None, :]
    return left_seg, right_seg, spread_dir_ee, left_sign


def _skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def _exp_so3(omega: np.ndarray) -> np.ndarray:
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


def solve_skeleton_pose(
    env,
    skeleton: EESkeleton,
    contact_point: np.ndarray,
    contact_normal: np.ndarray,
    *,
    finger: str,
    contact_primitive: Optional[str] = None,
    object_convex_equations: Optional[np.ndarray] = None,
    object_convex_equation_mask: Optional[np.ndarray] = None,
    scene_geom_ids: Optional[Sequence[int]] = None,
    initial_ee_rotation_world: Optional[np.ndarray] = None,
    args=None,
    return_candidates: bool = False,
    max_candidates: Optional[int] = None,
    min_theta_separation: float = 0.0,
    raw_model_data: Optional[tuple] = None,
) -> Optional[SkeletonPose]:
    """QP pose solver for an EE contact primitive.

    The QP linearizes the rigid hand+fingers around an initial pose where the
    chosen finger's midpoint already touches the contact point, with the finger
    direction aligned to a tangent of the surface. The 6 DoF small-angle SE(3)
    perturbation and the gripper opening `g` are optimized via DAQP so that:
      - the chosen primitive still passes through the contact point
        (linearized soft equality on its midpoint / hand-face point),
      - all hand-rectangle corners and sampled segment points stay on the
        free-space side of the contact tangent plane and the scene clearance,
      - the solution is close to the initial pose with `g` near
        PANDA_DEFAULT_GRIPPER_OPENING.
    """
    import mujoco

    from robocasa.demos.mlqp_point_cabinet import _solve_qp_daqp

    contact_point = np.asarray(contact_point, dtype=np.float64).reshape(3)
    contact_normal = _normalize(contact_normal)
    t1, t2 = _orthonormal_tangents(contact_normal)
    margin = float(getattr(args, "autogen_skeleton_margin", 0.002))
    theta_count = int(getattr(args, "autogen_skeleton_theta_count", 12))
    seg_samples = int(getattr(args, "autogen_skeleton_segment_samples", 5))
    initial_lift = float(getattr(args, "autogen_skeleton_initial_lift", 0.005))
    lift_weight = float(getattr(args, "autogen_skeleton_lift_weight", 100.0))
    g_default = float(
        getattr(args, "autogen_skeleton_gripper_default", PANDA_DEFAULT_GRIPPER_OPENING)
    )
    g_weight = float(getattr(args, "autogen_skeleton_gripper_weight", 10.0))
    g_min = float(getattr(args, "autogen_skeleton_gripper_min", 0.005))
    g_max = float(
        getattr(args, "autogen_skeleton_gripper_max", PANDA_MAX_GRIPPER_OPENING)
    )
    reg_weight = float(getattr(args, "autogen_skeleton_reg_weight", 1.0))
    motion_bound = float(getattr(args, "autogen_skeleton_motion_bound", 0.05))
    rot_bound = float(getattr(args, "autogen_skeleton_rot_bound", 0.35))
    clearance_tol = float(getattr(args, "autogen_skeleton_clearance_tolerance", 0.001))
    finger_radius = float(getattr(args, "autogen_skeleton_finger_radius", 0.004))

    if finger == "left":
        finger_sign = +1.0
    elif finger == "right":
        finger_sign = -1.0
    elif finger == "hand":
        finger_sign = +1.0
    else:
        raise ValueError(f"finger must be 'left', 'right', or 'hand', got {finger!r}")
    if contact_primitive is not None:
        contact_primitive = str(contact_primitive)
        valid_primitives = {"left_finger", "right_finger", "hand"}
        if contact_primitive not in valid_primitives:
            raise ValueError(
                f"contact_primitive must be one of {sorted(valid_primitives)}, "
                f"got {contact_primitive!r}"
            )

    if raw_model_data is not None:
        raw_model, raw_data = raw_model_data
    else:
        raw_model = getattr(env.sim.model, "_model", env.sim.model)
        raw_data = getattr(env.sim.data, "_data", env.sim.data)
    scene_geom_ids_set = set(int(g) for g in (scene_geom_ids or ()))

    # Build EE-frame finger geometry at default opening.
    left_seg0, right_seg0, y_hat_ee, left_sign = _finger_segments_with_opening(
        skeleton, g_default
    )
    contact_seg_ee = left_seg0 if finger == "left" else right_seg0
    other_seg_ee = right_seg0 if finger == "left" else left_seg0
    finger_dir_ee = contact_seg_ee[1] - contact_seg_ee[0]
    finger_len = float(np.linalg.norm(finger_dir_ee))
    if finger_len < 1e-8:
        return None
    finger_dir_ee = finger_dir_ee / finger_len
    finger_mid_ee = 0.5 * (contact_seg_ee[0] + contact_seg_ee[1])
    # spread direction sign for the chosen finger:
    # left_seg uses (left_sign) * y_hat_ee; right uses -left_sign.
    finger_spread_dir_ee = (left_sign if finger == "left" else -left_sign) * y_hat_ee

    # Build initial EE frame for each theta candidate.
    has_object_eqs = (
        object_convex_equations is not None
        and np.asarray(object_convex_equations).size > 0
    )
    if has_object_eqs:
        object_eqs = np.asarray(object_convex_equations, dtype=np.float64)
        if object_eqs.ndim == 2:
            object_eqs = object_eqs[None, :, :]
        if object_convex_equation_mask is not None:
            mask = np.asarray(object_convex_equation_mask, dtype=bool).reshape(-1)
            if mask.size == object_eqs.shape[0]:
                object_eqs = object_eqs[mask]
                has_object_eqs = object_eqs.shape[0] > 0

    # Sample check-points (in EE-frame) once: segment samples on both fingers +
    # 4 corners of the flat hand rectangle. The hand is a *flat* rectangle
    # with one thin axis (we still treat all 8 box corners conservatively).
    hand_signs = np.array(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    hand_corners_local = hand_signs * skeleton.hand_box_half_extents_ee[None, :]
    hand_corners_ee = (
        skeleton.hand_box_center_ee[None, :]
        + hand_corners_local @ skeleton.hand_box_rotation_ee.T
    )

    seg_alphas = np.linspace(0.0, 1.0, max(int(seg_samples), 2), dtype=np.float64)
    # Build base finger sample arrays (at default opening).
    contact_seg_samples_ee = (
        contact_seg_ee[0][None, :] * (1.0 - seg_alphas[:, None])
        + contact_seg_ee[1][None, :] * seg_alphas[:, None]
    )
    other_seg_samples_ee = (
        other_seg_ee[0][None, :] * (1.0 - seg_alphas[:, None])
        + other_seg_ee[1][None, :] * seg_alphas[:, None]
    )
    # Track each sample's spread direction (so g changes shift it by a known
    # amount along EE-frame y_hat).
    contact_spread_dir = np.tile(
        finger_spread_dir_ee[None, :], (contact_seg_samples_ee.shape[0], 1)
    )
    other_spread_dir = np.tile(
        -finger_spread_dir_ee[None, :], (other_seg_samples_ee.shape[0], 1)
    )
    hand_spread_dir = np.zeros_like(hand_corners_ee)
    all_samples_ee = np.concatenate(
        [contact_seg_samples_ee, other_seg_samples_ee, hand_corners_ee], axis=0
    )
    all_spread_dir_ee = np.concatenate(
        [contact_spread_dir, other_spread_dir, hand_spread_dir], axis=0
    )
    all_sample_radii = np.concatenate(
        [
            np.full(contact_seg_samples_ee.shape[0], finger_radius, dtype=np.float64),
            np.full(other_seg_samples_ee.shape[0], finger_radius, dtype=np.float64),
            np.zeros(hand_corners_ee.shape[0], dtype=np.float64),
        ],
        axis=0,
    )
    n_samples = all_samples_ee.shape[0]

    thetas = np.linspace(0.0, 2.0 * np.pi, int(max(1, theta_count)), endpoint=False)
    r0_anchor_ee = finger_mid_ee
    r0_contact_finger = str(finger)
    r0_contact_primitive = str(contact_primitive or f"{finger}_finger")
    if contact_primitive == "left_finger":
        r0_anchor_ee = 0.5 * (left_seg0[0] + left_seg0[1])
        r0_contact_finger = "left"
    elif contact_primitive == "right_finger":
        r0_anchor_ee = 0.5 * (right_seg0[0] + right_seg0[1])
        r0_contact_finger = "right"
    elif contact_primitive == "hand":
        hand_half_r0 = skeleton.hand_box_half_extents_ee.copy()
        thin_axis_r0 = int(np.argmin(hand_half_r0))
        hand_to_tip_local_r0 = skeleton.hand_box_rotation_ee.T @ (
            skeleton.finger_tip_offset_ee - skeleton.hand_box_center_ee
        )
        front_sign_r0 = (
            1.0 if float(hand_to_tip_local_r0[thin_axis_r0]) >= 0.0 else -1.0
        )
        hand_face_signs_r0 = np.array(
            [
                [sx, sy, sz]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        front_mask_r0 = hand_face_signs_r0[:, thin_axis_r0] == front_sign_r0
        hand_face_corners_local_r0 = (
            hand_face_signs_r0[front_mask_r0] * hand_half_r0[None, :]
        )
        hand_face_verts_ee_r0 = (
            skeleton.hand_box_center_ee[None, :]
            + hand_face_corners_local_r0 @ skeleton.hand_box_rotation_ee.T
        )
        r0_anchor_ee = np.mean(hand_face_verts_ee_r0, axis=0)
        r0_contact_finger = "hand"
    R0_override = None
    if initial_ee_rotation_world is not None:
        R0_override = np.asarray(initial_ee_rotation_world, dtype=np.float64).reshape(
            3, 3
        )
        # Demo wrap-around grasp: bypass the QP entirely. The QP enforces "all
        # finger samples and hand corners on +normal side", which is correct
        # for one-finger push contact but not for two-finger wrap-around grasp
        # of a cylindrical handle (the other finger is on the -normal side by
        # construction). Use the demo orientation directly and translate so the
        # chosen finger midpoint lies at contact_point + lift*normal. Downstream
        # mink IK + collision check filters out unreachable / colliding ones.
        p_final = (
            contact_point + initial_lift * contact_normal - R0_override @ r0_anchor_ee
        )
        # Reject poses where the hand box or finger samples penetrate the
        # target object's COACD convex parts. Without this, scene_geom_ids is
        # built with the handle EXCLUDED (the handle is the contact target,
        # not "scene"), and the only handle-side constraint in this bypass
        # branch is the tangent-plane half-space — leaving the hand free to
        # sweep through the handle body.
        if has_object_eqs:
            samples_w = (R0_override @ all_samples_ee.T).T + p_final[None, :]
            signed = _signed_distance_to_convex(samples_w, object_eqs)
            clearance = signed - all_sample_radii
            penetration_tol = float(
                getattr(args, "autogen_skeleton_object_penetration_tol", 0.001)
            )
            debug_log = (
                getattr(args, "_skeleton_debug_log", None) if args is not None else None
            )
            if debug_log is not None:
                debug_log.append(
                    {
                        "status": -99,  # synthetic code for "R0 path penetration check"
                        "err": 0.0,
                        "n_ineq": 0,
                        "n_scene_rows": 0,
                        "init_max_violation": float(-clearance.min())
                        if clearance.size
                        else 0.0,
                        "init_eq_residual": 0.0,
                        "samples_extent_max": float(
                            np.max(
                                np.linalg.norm(
                                    samples_w - contact_point[None, :], axis=1
                                )
                            )
                        )
                        if samples_w.size
                        else 0.0,
                        "r0_signed_min": float(signed.min()) if signed.size else 0.0,
                        "r0_clearance_min": float(clearance.min())
                        if clearance.size
                        else 0.0,
                        "r0_rejected": bool(
                            clearance.size and float(clearance.min()) < -penetration_tol
                        ),
                    }
                )
            if clearance.size and float(clearance.min()) < -penetration_tol:
                return [] if return_candidates else None
        return (
            SkeletonPose(
                ee_position=np.asarray(p_final, dtype=np.float64),
                ee_rotation=np.asarray(R0_override, dtype=np.float64),
                contact_finger=r0_contact_finger,
                contact_point_world=np.asarray(contact_point, dtype=np.float64),
                contact_normal_world=np.asarray(contact_normal, dtype=np.float64),
                qp_cost=0.0,
                lift=float(initial_lift),
                theta=0.0,
                gripper_opening=float(g_default),
                contact_primitive=r0_contact_primitive,
            )
            if not return_candidates
            else [
                SkeletonPose(
                    ee_position=np.asarray(p_final, dtype=np.float64),
                    ee_rotation=np.asarray(R0_override, dtype=np.float64),
                    contact_finger=r0_contact_finger,
                    contact_point_world=np.asarray(contact_point, dtype=np.float64),
                    contact_normal_world=np.asarray(contact_normal, dtype=np.float64),
                    qp_cost=0.0,
                    lift=float(initial_lift),
                    theta=0.0,
                    gripper_opening=float(g_default),
                    contact_primitive=r0_contact_primitive,
                )
            ]
        )
    best = None
    candidates = []

    contact_weight = float(getattr(args, "autogen_skeleton_contact_weight", 500.0))
    n_random = max(int(getattr(args, "autogen_skeleton_n_random", 3)), 1)
    lift_range = (
        float(getattr(args, "autogen_skeleton_lift_min", 0.002)),
        float(getattr(args, "autogen_skeleton_lift_max", 0.01)),
    )
    g_range = (
        float(getattr(args, "autogen_skeleton_g_sample_min", 0.01)),
        float(getattr(args, "autogen_skeleton_g_sample_max", 0.06)),
    )
    rng = np.random.RandomState(seed=int(getattr(args, "autogen_skeleton_seed", 42)))

    # Build primitives: each is (anchor_ee, spread_dir_ee, primitive_name, extra_vars_count)
    # For fingers: 1 extra var (s ∈ [0,1] along segment)
    # For hand box face: 4 extra vars (α₁..α₄ convex combination of face vertices)
    left_seg0, right_seg0, y_hat_ee, left_sign = _finger_segments_with_opening(
        skeleton, g_default
    )
    left_spread = left_sign * y_hat_ee
    right_spread = -left_sign * y_hat_ee

    # Hand faces: use every face of the hand box as a possible contact
    # primitive. Earlier versions only used the front face near the fingers,
    # which made side-of-hand drawer contacts impossible to generate.
    hand_half = skeleton.hand_box_half_extents_ee.copy()
    thin_axis = int(np.argmin(hand_half))
    hand_to_tip_local = skeleton.hand_box_rotation_ee.T @ (
        skeleton.finger_tip_offset_ee - skeleton.hand_box_center_ee
    )
    front_sign = 1.0 if float(hand_to_tip_local[thin_axis]) >= 0.0 else -1.0
    hand_face_signs = np.array(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    hand_face_defs = []
    for face_axis in range(3):
        remaining_axes = [axis for axis in range(3) if axis != face_axis]
        tangent_axis = max(remaining_axes, key=lambda axis: float(hand_half[axis]))
        for face_sign in (-1.0, 1.0):
            face_mask = hand_face_signs[:, face_axis] == face_sign
            face_corners_local = hand_face_signs[face_mask] * hand_half[None, :]
            face_verts_ee = (
                skeleton.hand_box_center_ee[None, :]
                + face_corners_local @ skeleton.hand_box_rotation_ee.T
            )
            if face_axis == thin_axis and face_sign == front_sign:
                face_name = "hand_front"
            elif face_axis == thin_axis:
                face_name = "hand_back"
            else:
                suffix = "pos" if face_sign > 0.0 else "neg"
                face_name = f"hand_side_axis{face_axis}_{suffix}"
            face_normal_ee = _normalize(
                face_sign * skeleton.hand_box_rotation_ee[:, face_axis],
                [1.0, 0.0, 0.0],
            )
            face_tangent_ee = _normalize(
                skeleton.hand_box_rotation_ee[:, tangent_axis],
                [0.0, 0.0, 1.0],
            )
            hand_face_defs.append(
                (face_name, face_verts_ee, face_normal_ee, face_tangent_ee)
            )

    def _finger_side_defs(prefix: str, seg: np.ndarray, outward_ee: np.ndarray):
        finger_axis = _normalize(seg[1] - seg[0], [1.0, 0.0, 0.0])
        outward = np.asarray(outward_ee, dtype=np.float64).reshape(3)
        outward = outward - float(np.dot(outward, finger_axis)) * finger_axis
        outward = _normalize(outward, [0.0, 1.0, 0.0])
        cross_side = _normalize(np.cross(finger_axis, outward), [0.0, 0.0, 1.0])
        return [
            (f"{prefix}_outer", seg, outward_ee, outward),
            (f"{prefix}_inner", seg, outward_ee, -outward),
            (f"{prefix}_side_pos", seg, outward_ee, cross_side),
            (f"{prefix}_side_neg", seg, outward_ee, -cross_side),
        ]

    primitives = []
    primitives.extend(_finger_side_defs("left_finger", left_seg0, left_spread))
    primitives.extend(_finger_side_defs("right_finger", right_seg0, right_spread))
    primitives.extend(
        (name, verts, np.zeros(3, dtype=np.float64), normal)
        for name, verts, normal, _ in hand_face_defs
    )
    if contact_primitive is not None:
        if contact_primitive == "hand":
            primitives = [prim for prim in primitives if prim[0].startswith("hand")]
        elif contact_primitive == "left_finger":
            primitives = [
                prim for prim in primitives if prim[0].startswith("left_finger")
            ]
        elif contact_primitive == "right_finger":
            primitives = [
                prim for prim in primitives if prim[0].startswith("right_finger")
            ]
        else:
            primitives = [prim for prim in primitives if prim[0] == contact_primitive]

    def _initial_rotation_for_primitive(
        prim_name: str,
        tangent_dir_world: np.ndarray,
        contact_normal_ee: np.ndarray,
    ):
        b_w = contact_normal
        a_w = _normalize(tangent_dir_world)
        # Object contact_normal points from drawer into free space. The
        # contacting surface normal of the hand/finger points back into the
        # drawer, so the primitive-local contact normal must align with -b_w.
        contact_surface_normal_w = -b_w
        c_w = _normalize(np.cross(a_w, contact_surface_normal_w))
        R_world_target = np.stack([a_w, contact_surface_normal_w, c_w], axis=1)
        if prim_name.startswith("hand"):
            face_lookup = {
                name: (normal, tangent) for name, _, normal, tangent in hand_face_defs
            }
            face_normal_ee, face_tangent_ee = face_lookup[prim_name]
            hand_tangent = (
                face_tangent_ee
                - float(np.dot(face_tangent_ee, face_normal_ee)) * face_normal_ee
            )
            hand_tangent = _normalize(hand_tangent, [0.0, 0.0, 1.0])
            hand_binormal = _normalize(
                np.cross(hand_tangent, face_normal_ee), [0.0, 1.0, 0.0]
            )
            hand_normal = _normalize(
                np.cross(hand_binormal, hand_tangent), [1.0, 0.0, 0.0]
            )
            R_ee_basis = np.stack([hand_tangent, hand_normal, hand_binormal], axis=1)
            return R_world_target @ R_ee_basis.T

        prim_seg = left_seg0 if prim_name.startswith("left_finger") else right_seg0
        prim_finger_dir_ee = _normalize(prim_seg[1] - prim_seg[0])
        norm_axis_ee = np.asarray(contact_normal_ee, dtype=np.float64).reshape(3)
        norm_axis_ee = (
            norm_axis_ee
            - float(np.dot(norm_axis_ee, prim_finger_dir_ee)) * prim_finger_dir_ee
        )
        norm_axis_ee = _normalize(norm_axis_ee, [0.0, 1.0, 0.0])
        third_ee = _normalize(
            np.cross(prim_finger_dir_ee, norm_axis_ee), [0.0, 0.0, 1.0]
        )
        norm_axis_ee = _normalize(
            np.cross(third_ee, prim_finger_dir_ee), [0.0, 1.0, 0.0]
        )
        R_ee_basis = np.stack([prim_finger_dir_ee, norm_axis_ee, third_ee], axis=1)
        return R_world_target @ R_ee_basis.T

    for theta in thetas:
        tangent_dir_world = _normalize(
            np.cos(float(theta)) * t1 + np.sin(float(theta)) * t2
        )
        b_w = contact_normal

        for prim_name, prim_geom, prim_spread_ee, prim_normal_ee in primitives:
            R0 = _initial_rotation_for_primitive(
                prim_name, tangent_dir_world, prim_normal_ee
            )
            # Sample random (initial_lift_i, g_default_i)
            lifts = rng.uniform(lift_range[0], lift_range[1], size=n_random)
            g_defaults = rng.uniform(g_range[0], g_range[1], size=n_random)

            for rand_i in range(n_random):
                lift_i = float(lifts[rand_i])
                g_def_i = float(g_defaults[rand_i])

                if prim_name.startswith("left_finger") or prim_name.startswith(
                    "right_finger"
                ):
                    # Finger primitive: anchor = midpoint of segment, extra var s ∈ [0,1]
                    seg = prim_geom  # (2, 3)
                    anchor_ee = 0.5 * (seg[0] + seg[1])  # nominal s=0.5
                    spread_ee = prim_spread_ee
                    anchor_clearance = margin + finger_radius
                    nvar = 8  # [dx,dy,dz,rx,ry,rz,g,s]
                else:
                    # Hand box: anchor = face center (mean of 4 verts), extra vars α₁..α₄
                    verts = prim_geom  # (4, 3)
                    anchor_ee = np.mean(verts, axis=0)
                    spread_ee = np.zeros(3, dtype=np.float64)
                    anchor_clearance = margin
                    nvar = 11  # [dx,dy,dz,rx,ry,rz,g,α₁,α₂,α₃,α₄]

                p0 = contact_point + lift_i * b_w - R0 @ anchor_ee

                # --- Inequality constraints (same structure, padded for extra vars) ---
                samples_w = (R0 @ all_samples_ee.T).T + p0[None, :]
                spread_w = (R0 @ all_spread_dir_ee.T).T
                r_vec = samples_w - p0[None, :]
                cross_rb = np.cross(r_vec, b_w[None, :])
                A_obj_7 = np.zeros((n_samples, 7), dtype=np.float64)
                A_obj_7[:, 0:3] = b_w[None, :]
                A_obj_7[:, 3:6] = cross_rb
                A_obj_7[:, 6] = spread_w @ b_w
                b0_obj = (samples_w - contact_point[None, :]) @ b_w

                # Pad to nvar columns
                A_obj = np.zeros((n_samples, nvar), dtype=np.float64)
                A_obj[:, :7] = A_obj_7

                rows_A = [A_obj]
                rows_b = [margin + all_sample_radii - b0_obj]

                if scene_geom_ids_set:
                    geomgroup = np.ones(6, dtype=np.uint8)
                    scene_distances = np.full(n_samples, np.inf, dtype=np.float64)
                    for si in range(n_samples):
                        start = samples_w[si]
                        direction = -b_w
                        try:
                            geomid_out = np.zeros(1, dtype=np.int32)
                            dist = mujoco.mj_ray(
                                raw_model,
                                raw_data,
                                np.asarray(start, dtype=np.float64),
                                np.asarray(direction, dtype=np.float64),
                                geomgroup,
                                1,
                                -1,
                                geomid_out,
                            )
                        except Exception:
                            dist = -1.0
                            geomid_out = np.array([-1], dtype=np.int32)
                        if dist > 0 and int(geomid_out[0]) in scene_geom_ids_set:
                            scene_distances[si] = float(dist)
                    mask = np.isfinite(scene_distances)
                    if bool(np.any(mask)):
                        A_scene = np.zeros((int(np.sum(mask)), nvar), dtype=np.float64)
                        A_scene[:, :7] = A_obj_7[mask]
                        rows_A.append(A_scene)
                        rows_b.append(
                            margin + all_sample_radii[mask] - scene_distances[mask]
                        )

                # --- Soft contact constraint as quadratic penalty ---
                # Anchor world position (linearized):
                #   p_anchor(x) = p0 + R0@anchor_ee + dx + ω×(R0@anchor_ee) + dg*R0@spread_ee
                #                 + param_contribution(extra_vars)
                # Residual: p_anchor(x) - contact_point = lift_i*b_w + A_contact @ x.
                # The primitive surface, not the finger capsule centerline, is
                # what should reach the contact point. Keep finger centerlines
                # one radius outside the drawer and hand faces at the margin.
                r_anchor = R0 @ anchor_ee
                spread_anchor_w = R0 @ spread_ee
                A_contact = np.zeros((3, nvar), dtype=np.float64)
                A_contact[:, 0:3] = np.eye(3)
                A_contact[:, 3:6] = -_skew(r_anchor)
                A_contact[:, 6] = spread_anchor_w

                if prim_name.startswith("left_finger") or prim_name.startswith(
                    "right_finger"
                ):
                    # ∂p/∂s = R0 @ (seg[1] - seg[0]), linearized around s=0.5
                    d_finger_w = R0 @ (seg[1] - seg[0])
                    A_contact[:, 7] = d_finger_w
                else:
                    # ∂p/∂αᵢ = R0 @ (hᵢ - anchor_ee), deviation from nominal center
                    for vi in range(4):
                        A_contact[:, 7 + vi] = R0 @ (verts[vi] - anchor_ee)

                b_contact = (float(anchor_clearance) - lift_i) * b_w

                # --- Build QP cost ---
                H = np.zeros((nvar, nvar), dtype=np.float64)
                f_cost = np.zeros(nvar, dtype=np.float64)
                # Regularization on SE(3) perturbation
                H[:6, :6] += 2.0 * reg_weight * np.eye(6)
                # x[6] is dg = g - g_default. Earlier versions mixed an
                # absolute g cost/bounds with delta-g kinematics, which biased
                # finger locations toward a different opening than the QP
                # geometry was linearized around.
                H[6, 6] += 2.0 * g_weight
                f_cost[6] += -2.0 * g_weight * (g_def_i - g_default)
                # Lift penalty
                bw_outer = np.outer(b_w, b_w)
                H[:3, :3] += 2.0 * lift_weight * bw_outer
                # Soft contact penalty: contact_weight * ||A_contact x - b_contact||²
                H += 2.0 * contact_weight * (A_contact.T @ A_contact)
                f_cost += -2.0 * contact_weight * (A_contact.T @ b_contact)
                # SPD regularization
                H += 1e-6 * np.eye(nvar)

                # --- Assemble constraints (inequalities only, no equality) ---
                A_ineq = np.concatenate(rows_A, axis=0)
                b_ineq_lower = np.concatenate(rows_b, axis=0)

                # Box bounds on base 7 vars + extra vars
                I_n = np.eye(nvar, dtype=np.float64)
                bounds_lower = np.full(nvar, -1e30, dtype=np.float64)
                bounds_upper = np.full(nvar, 1e30, dtype=np.float64)
                bounds_lower[:3] = -motion_bound
                bounds_upper[:3] = motion_bound
                bounds_lower[3:6] = -rot_bound
                bounds_upper[3:6] = rot_bound
                bounds_lower[6] = g_min - g_default
                bounds_upper[6] = g_max - g_default
                if prim_name.startswith("left_finger") or prim_name.startswith(
                    "right_finger"
                ):
                    # s ∈ [0, 1], but centered at 0.5 so perturbation ∈ [-0.5, 0.5]
                    bounds_lower[7] = -0.5
                    bounds_upper[7] = 0.5
                else:
                    # αᵢ perturbations: nominal is 0.25 each, so αᵢ ∈ [0,1] → Δαᵢ ∈ [-0.25, 0.75]
                    for vi in range(4):
                        bounds_lower[7 + vi] = -0.25
                        bounds_upper[7 + vi] = 0.75

                A_full = np.concatenate([A_ineq, I_n], axis=0)
                n_ineq = A_ineq.shape[0]
                bupper = np.concatenate([np.full(n_ineq, 1e30), bounds_upper])
                blower = np.concatenate([b_ineq_lower, bounds_lower])
                sense = np.zeros(A_full.shape[0], dtype=np.int32)

                if prim_name.startswith("hand"):
                    # Simplex equality: Σ Δαᵢ = 0 (since nominal sums to 1)
                    A_simplex = np.zeros((1, nvar), dtype=np.float64)
                    A_simplex[0, 7:11] = 1.0
                    A_full = np.concatenate([A_simplex, A_full], axis=0)
                    bupper = np.concatenate([np.array([0.0]), bupper])
                    blower = np.concatenate([np.array([0.0]), blower])
                    sense = np.concatenate([np.array([5], dtype=np.int32), sense])

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
                debug_log = (
                    getattr(args, "_skeleton_debug_log", None)
                    if args is not None
                    else None
                )
                if debug_log is not None:
                    debug_log.append(
                        {
                            "status": int(status),
                            "err": float(err),
                            "n_ineq": int(n_ineq),
                            "n_scene_rows": 0,
                            "init_max_violation": 0.0,
                            "init_eq_residual": 0.0,
                            "samples_extent_max": 0.0,
                            "primitive": prim_name,
                        }
                    )
                if int(status) != 1:
                    continue

                dx = x[0:3]
                omega = x[3:6]
                g_opt = float(g_default + x[6])
                p_final = p0 + dx
                R_final = _exp_so3(omega) @ R0
                lift_value = float(np.dot(b_w, dx) + lift_i)

                final_samples_ee = (
                    all_samples_ee + (g_opt - g_default) * all_spread_dir_ee
                )
                final_samples_w = (R_final @ final_samples_ee.T).T + p_final[None, :]
                plane_clearance = (
                    final_samples_w - contact_point[None, :]
                ) @ b_w - all_sample_radii
                if (
                    plane_clearance.size
                    and float(np.min(plane_clearance)) < margin - clearance_tol
                ):
                    debug_log = (
                        getattr(args, "_skeleton_debug_log", None)
                        if args is not None
                        else None
                    )
                    if debug_log is not None:
                        debug_log.append(
                            {
                                "status": -98,
                                "err": 0.0,
                                "n_ineq": int(n_ineq),
                                "n_scene_rows": 0,
                                "init_max_violation": float(
                                    margin - np.min(plane_clearance)
                                ),
                                "init_eq_residual": 0.0,
                                "samples_extent_max": float(
                                    np.max(
                                        np.linalg.norm(
                                            final_samples_w - contact_point[None, :],
                                            axis=1,
                                        )
                                    )
                                ),
                                "primitive": prim_name,
                                "rejected": "post_clearance",
                            }
                        )
                    continue
                if has_object_eqs:
                    signed = _signed_distance_to_convex(final_samples_w, object_eqs)
                    clearance = signed - all_sample_radii
                    penetration_tol = float(
                        getattr(args, "autogen_skeleton_object_penetration_tol", 0.001)
                    )
                    if clearance.size and float(clearance.min()) < -penetration_tol:
                        debug_log = (
                            getattr(args, "_skeleton_debug_log", None)
                            if args is not None
                            else None
                        )
                        if debug_log is not None:
                            debug_log.append(
                                {
                                    "status": -97,
                                    "err": 0.0,
                                    "n_ineq": int(n_ineq),
                                    "n_scene_rows": 0,
                                    "init_max_violation": float(-clearance.min()),
                                    "init_eq_residual": 0.0,
                                    "samples_extent_max": float(
                                        np.max(
                                            np.linalg.norm(
                                                final_samples_w
                                                - contact_point[None, :],
                                                axis=1,
                                            )
                                        )
                                    ),
                                    "primitive": prim_name,
                                    "rejected": "post_object_convex",
                                }
                            )
                        continue

                cost = float(0.5 * x @ (H @ x) + f_cost @ x)
                candidates.append(
                    (cost, lift_value, float(theta), R_final, p_final, g_opt, prim_name)
                )
                if best is None or cost < best[0]:
                    best = (
                        cost,
                        lift_value,
                        float(theta),
                        R_final,
                        p_final,
                        g_opt,
                        prim_name,
                    )

    if best is None:
        return [] if return_candidates else None
    if return_candidates:
        max_candidates = int(max_candidates or len(candidates))
        min_theta_separation = max(float(min_theta_separation), 0.0)
        selected = []
        selected_ids = set()
        sorted_candidates = sorted(candidates, key=lambda item: item[0])

        def _passes_theta(candidate):
            theta = float(candidate[2])
            primitive_name = str(candidate[6])
            if min_theta_separation > 0.0:
                for existing in selected:
                    if str(existing[6]) != primitive_name:
                        continue
                    dtheta = abs(
                        ((theta - float(existing[2]) + np.pi) % (2.0 * np.pi)) - np.pi
                    )
                    if dtheta < min_theta_separation:
                        return False
            return True

        # Pick by exact primitive first, then by cost. This prevents a cheap
        # front-face or outer-finger solution from crowding out side-face /
        # inner-finger candidates before visualization and IK/MPPI get a chance
        # to evaluate them.
        primitive_order = []
        by_primitive = {}
        for candidate in sorted_candidates:
            primitive_name = str(candidate[6])
            if primitive_name not in by_primitive:
                by_primitive[primitive_name] = []
                primitive_order.append(primitive_name)
            by_primitive[primitive_name].append(candidate)
        while len(selected) < max_candidates and any(by_primitive.values()):
            made_progress = False
            for prim_name in primitive_order:
                bucket = by_primitive[prim_name]
                while bucket:
                    candidate = bucket.pop(0)
                    if not _passes_theta(candidate):
                        continue
                    selected.append(candidate)
                    selected_ids.add(id(candidate))
                    made_progress = True
                    break
                if len(selected) >= max_candidates:
                    break
            if not made_progress:
                break

        for candidate in sorted_candidates:
            if len(selected) >= max_candidates:
                break
            if id(candidate) in selected_ids or not _passes_theta(candidate):
                continue
            selected.append(candidate)
            selected_ids.add(id(candidate))
        return [
            SkeletonPose(
                ee_position=np.asarray(p_final, dtype=np.float64),
                ee_rotation=np.asarray(R_final, dtype=np.float64),
                contact_finger=(
                    "hand"
                    if str(prim).startswith("hand")
                    else ("left" if str(prim).startswith("left_finger") else "right")
                ),
                contact_point_world=np.asarray(contact_point, dtype=np.float64),
                contact_normal_world=np.asarray(contact_normal, dtype=np.float64),
                qp_cost=float(cost),
                lift=float(lift_value),
                theta=float(theta_opt),
                gripper_opening=float(g_opt),
                contact_primitive=str(prim),
            )
            for cost, lift_value, theta_opt, R_final, p_final, g_opt, prim in selected
        ]
    cost, lift_value, theta_opt, R_final, p_final, g_opt, prim = best
    return SkeletonPose(
        ee_position=np.asarray(p_final, dtype=np.float64),
        ee_rotation=np.asarray(R_final, dtype=np.float64),
        contact_finger=(
            "hand"
            if str(prim).startswith("hand")
            else ("left" if str(prim).startswith("left_finger") else "right")
        ),
        contact_point_world=np.asarray(contact_point, dtype=np.float64),
        contact_normal_world=np.asarray(contact_normal, dtype=np.float64),
        qp_cost=float(cost),
        lift=float(lift_value),
        theta=float(theta_opt),
        gripper_opening=float(g_opt),
        contact_primitive=str(prim),
    )


def solve_skeleton_pose_candidates(
    env,
    skeleton: EESkeleton,
    contact_point: np.ndarray,
    contact_normal: np.ndarray,
    *,
    finger: str,
    contact_primitive: Optional[str] = None,
    object_convex_equations: Optional[np.ndarray] = None,
    object_convex_equation_mask: Optional[np.ndarray] = None,
    scene_geom_ids: Optional[Sequence[int]] = None,
    initial_ee_rotation_world: Optional[np.ndarray] = None,
    args=None,
    max_candidates: Optional[int] = None,
    min_theta_separation: float = 0.0,
    raw_model_data: Optional[tuple] = None,
) -> list[SkeletonPose]:
    poses = solve_skeleton_pose(
        env,
        skeleton,
        contact_point,
        contact_normal,
        finger=finger,
        contact_primitive=contact_primitive,
        object_convex_equations=object_convex_equations,
        object_convex_equation_mask=object_convex_equation_mask,
        scene_geom_ids=scene_geom_ids,
        initial_ee_rotation_world=initial_ee_rotation_world,
        args=args,
        return_candidates=True,
        max_candidates=max_candidates,
        min_theta_separation=min_theta_separation,
        raw_model_data=raw_model_data,
    )
    return list(poses or [])


def skeleton_pose_to_ee_pose(skeleton: EESkeleton, skeleton_pose: SkeletonPose):
    """Skeleton is parameterized in the EE-site frame, so this is identity."""
    return (
        np.asarray(skeleton_pose.ee_position, dtype=np.float64),
        _rotation_matrix_to_quat_wxyz(
            np.asarray(skeleton_pose.ee_rotation, dtype=np.float64)
        ),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _hsv_palette(n: int) -> np.ndarray:
    n = max(int(n), 1)
    import colorsys

    out = []
    for i in range(n):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
        out.append((r, g, b, 1.0))
    return np.asarray(out, dtype=np.float32)


def _add_capsule_segment(scene, start, end, radius, rgba):
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        np.array([float(radius), float(radius), float(radius)], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    try:
        mujoco.mjv_connector(
            geom,
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            float(radius),
            np.asarray(start, dtype=np.float64).reshape(3),
            np.asarray(end, dtype=np.float64).reshape(3),
        )
    except Exception:
        pass
    scene.ngeom += 1


def _add_box(scene, center, rotation, half_extents, rgba):
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        int(mujoco.mjtGeom.mjGEOM_BOX),
        np.asarray(half_extents, dtype=np.float64),
        np.asarray(center, dtype=np.float64).reshape(3),
        np.asarray(rotation, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _flat_hand_half_extents(skeleton: EESkeleton) -> np.ndarray:
    """Return half-extents for a thin flat rectangle approximating the panda hand.

    Picks the smallest axis of the XML hand bounding box and clamps it to
    ~0.002 m (2 mm thick); the other two extents are preserved.
    """
    half = (
        np.asarray(skeleton.hand_box_half_extents_ee, dtype=np.float64)
        .reshape(3)
        .copy()
    )
    thin_axis = int(np.argmin(half))
    half[thin_axis] = 0.002
    return half


def _draw_skeleton_into_scene(
    scene,
    skeleton: EESkeleton,
    ee_pos: np.ndarray,
    ee_rot: np.ndarray,
    gripper_opening: float,
    rgba_hand,
    rgba_finger,
    finger_radius: float,
):
    """Draw solid semi-transparent hand box and two finger capsules."""
    half_ext = _flat_hand_half_extents(skeleton)
    hand_center_w = ee_pos + ee_rot @ skeleton.hand_box_center_ee
    hand_rot_w = ee_rot @ skeleton.hand_box_rotation_ee
    import sys as _sys

    print(
        f"[skeleton_draw] hand half_ext={half_ext} center_w={hand_center_w} "
        f"rgba_hand={rgba_hand} ngeom_before={scene.ngeom} maxgeom={scene.maxgeom}",
        file=_sys.__stdout__,
        flush=True,
    )
    _add_box(scene, hand_center_w, hand_rot_w, half_ext, rgba_hand)
    left_seg, right_seg, _, _ = _finger_segments_with_opening(skeleton, gripper_opening)
    for seg in (left_seg, right_seg):
        sa = ee_pos + ee_rot @ seg[0]
        sb = ee_pos + ee_rot @ seg[1]
        _add_capsule_segment(scene, sa, sb, finger_radius, rgba_finger)


def visualize_skeleton_poses(env, ee_site_name: str, skeleton: EESkeleton, poses, args):
    if not bool(getattr(args, "autogen_visualize_skeleton_poses", True)):
        return
    if not poses:
        return
    geoms_per_pose = 4  # 1 box + 2 finger capsules + 1 contact sphere
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
    import mujoco
    import mujoco.viewer

    from robocasa.demos import visualize_mujoco as viz_mj

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
        palette = _hsv_palette(len(poses))
        finger_radius = float(getattr(args, "autogen_skeleton_finger_radius", 0.004))
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
                        _draw_skeleton_into_scene(
                            viewer.user_scn,
                            skeleton,
                            np.asarray(pose.ee_position, dtype=np.float64),
                            np.asarray(pose.ee_rotation, dtype=np.float64),
                            float(
                                getattr(
                                    pose,
                                    "gripper_opening",
                                    PANDA_DEFAULT_GRIPPER_OPENING,
                                )
                            ),
                            rgba_hand,
                            rgba_capsule,
                            finger_radius,
                        )
                        try:
                            viz_mj._add_scene_sphere(
                                viewer.user_scn,
                                np.asarray(pose.contact_point_world, dtype=np.float64),
                                0.005,
                                green,
                            )
                        except Exception:
                            pass
                viewer.sync()
                time.sleep(1.0 / fps)
    finally:
        raw_model.geom_rgba[:] = saved_rgba


def visualize_skeleton_and_ee(env, ee_site_name: str, skeleton: EESkeleton, args):
    """Preview popup: real panda EE (semi-transparent) + skeleton overlaid at live pose."""
    if not bool(getattr(args, "autogen_visualize_skeleton_preview", True)):
        return
    import mujoco
    import mujoco.viewer

    from robocasa.demos import visualize_mujoco as viz_mj

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
        getattr(args, "autogen_skeleton_gripper_default", PANDA_DEFAULT_GRIPPER_OPENING)
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
                    _draw_skeleton_into_scene(
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
