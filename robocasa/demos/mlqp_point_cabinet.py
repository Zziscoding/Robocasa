"""GPU-batched contact-force QP for point contacts.

Each candidate contact has three decision variables in its local contact
frame: normal force and two tangential forces.  The friction pyramid and
normal-force bounds are linear, while the local pose tracking objective is a
weighted least-squares objective.  JAXopt OSQP solves all candidate QPs in one
vmapped, JIT-compiled call.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _normalize(vector: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return vector / norm


def _load_mesh(
    mesh_path: str | Path, scale_factors: Sequence[float]
) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"Mesh scene has no triangle geometry: {mesh_path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.vertices.shape[0] == 0:
        raise ValueError(f"Unable to load a non-empty triangle mesh: {mesh_path}")
    mesh = loaded.copy()
    mesh.apply_scale(np.asarray(scale_factors, dtype=np.float64).reshape(3))
    return mesh


def _farthest_vertex_indices(vertices: np.ndarray, count: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    count = min(max(int(count), 1), vertices.shape[0])
    if count == vertices.shape[0]:
        return np.arange(count, dtype=np.int64)
    indices = np.empty(count, dtype=np.int64)
    indices[0] = int(np.argmin(np.linalg.norm(vertices, axis=1)))
    distances = np.full(vertices.shape[0], np.inf, dtype=np.float64)
    for sample_index in range(1, count):
        distances = np.minimum(
            distances,
            np.linalg.norm(vertices - vertices[indices[sample_index - 1]], axis=1),
        )
        indices[sample_index] = int(np.argmax(distances))
    return indices


def _contact_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    n_out = np.empty_like(normals)
    tangent1 = np.empty_like(normals)
    tangent2 = np.empty_like(normals)
    for index, normal in enumerate(normals):
        normal = _normalize(normal, [1.0, 0.0, 0.0])
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(reference, normal))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t1 = _normalize(reference - np.dot(reference, normal) * normal, [0.0, 1.0, 0.0])
        t2 = _normalize(np.cross(normal, t1), [0.0, 0.0, 1.0])
        t1 = _normalize(np.cross(t2, normal), t1)
        n_out[index] = normal
        tangent1[index] = t1
        tangent2[index] = t2
    return n_out, tangent1, tangent2


class _MeshProjection:
    """Small mesh projection helper without the previous Open3D dependency."""

    def __init__(
        self,
        mesh_path: str | Path,
        scale_factors: Sequence[float],
        sample_num: int,
    ):
        self.scaled_mesh = _load_mesh(mesh_path, scale_factors)
        self.vertices = np.asarray(self.scaled_mesh.vertices, dtype=np.float64)
        self.vertex_normals = np.asarray(
            self.scaled_mesh.vertex_normals, dtype=np.float64
        )
        self.kdtree = cKDTree(self.vertices)
        sample_indices = _farthest_vertex_indices(self.vertices, sample_num)
        self.sample_points = self.vertices[sample_indices]
        # ProjectionPoint historically returned inward-facing normals.
        self.sample_normals, self.sample_t1, self.sample_t2 = _contact_frames(
            -self.vertex_normals[sample_indices]
        )

    def project_point_to_mesh(
        self, point: Sequence[float]
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        _, vertex_index = self.kdtree.query(
            np.asarray(point, dtype=np.float64).reshape(3)
        )
        normal, tangent1, tangent2 = _contact_frames(
            -self.vertex_normals[int(vertex_index)].reshape(1, 3)
        )
        return int(vertex_index), normal[0], tangent1[0], tangent2[0]


def _quat_normalize(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(
        min=1e-12
    )


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[..., 1:] *= -1.0
    return result


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _relative_rotvec(current_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    current_quat = _quat_normalize(current_quat)
    target_quat = _quat_normalize(target_quat)
    relative = _quat_normalize(
        _quat_multiply(_quat_conjugate(current_quat), target_quat)
    )
    relative = np.where(relative[..., :1] < 0.0, -relative, relative)
    vector = relative[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, relative[..., :1].clip(min=1e-12))
    scale = np.full_like(angle, 2.0)
    np.divide(angle, vector_norm, out=scale, where=vector_norm > 1e-10)
    return vector * scale


def _integrate_pose(current_x: np.ndarray, delta: np.ndarray) -> np.ndarray:
    current_x = np.asarray(current_x, dtype=np.float64).reshape(-1, 7)
    delta = np.asarray(delta, dtype=np.float64).reshape(-1, 6)
    quaternion = _quat_normalize(current_x[:, 3:7])
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    qmat = np.stack(
        [
            np.stack([-x, -y, -z], axis=-1),
            np.stack([w, -z, y], axis=-1),
            np.stack([z, w, -x], axis=-1),
            np.stack([-y, x, w], axis=-1),
        ],
        axis=1,
    )
    next_quaternion = quaternion + 0.5 * np.einsum("bij,bj->bi", qmat, delta[:, 3:6])
    next_quaternion = _quat_normalize(next_quaternion)
    return np.concatenate([current_x[:, :3] + delta[:, :3], next_quaternion], axis=1)


def _solve_qp_daqp(
    H,
    f,
    A,
    bupper,
    blower=None,
    sense=None,
    max_iter=1000,
    tol=1e-5,
):
    """Solve `min 0.5 x'Hx + f'x s.t. blower <= A x <= bupper` via DAQP.

    Returns (x, status_code, error, iterations) where status_code follows the
    original convention: 1 = solved, anything else = failure.
    """
    try:
        import daqp
    except ImportError as exc:
        raise RuntimeError(
            "DAQP is required for LambdaContactControlOptimizer. "
            "Install it with: pip install daqp"
        ) from exc

    H = np.ascontiguousarray(np.asarray(H, dtype=np.float64))
    f = np.ascontiguousarray(np.asarray(f, dtype=np.float64).reshape(-1))
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
    bupper = np.ascontiguousarray(np.asarray(bupper, dtype=np.float64).reshape(-1))
    nvar = H.shape[0]
    m = A.shape[0]
    if blower is None:
        blower = np.full(m, -1e30, dtype=np.float64)
    else:
        blower = np.ascontiguousarray(np.asarray(blower, dtype=np.float64).reshape(-1))
    if sense is None:
        sense = np.zeros(m, dtype=np.int32)
    else:
        sense = np.ascontiguousarray(np.asarray(sense, dtype=np.int32).reshape(-1))
    # daqp also requires variable bounds; we have none → use large limits.
    bupper_var = np.full(nvar, 1e30, dtype=np.float64)
    blower_var = np.full(nvar, -1e30, dtype=np.float64)
    full_bupper = np.concatenate([bupper_var, bupper])
    full_blower = np.concatenate([blower_var, blower])
    full_sense = np.concatenate([np.zeros(nvar, dtype=np.int32), sense])
    try:
        x, fval, exitflag, info = daqp.solve(
            H, f, A, full_bupper, full_blower, full_sense
        )
    except Exception as exc:
        return (
            np.zeros(nvar, dtype=np.float64),
            -99,
            float("inf"),
            0,
        )
    # DAQP exitflag: 1 = optimal, others = failure modes
    iters = 0
    if isinstance(info, dict):
        iters = int(info.get("iter", 0) or info.get("iterations", 0) or 0)
    status_code = 1 if int(exitflag) == 1 else int(exitflag)
    err = 0.0 if status_code == 1 else float("inf")
    return np.asarray(x, dtype=np.float64).reshape(nvar), status_code, err, iters


class LambdaContactControlOptimizer:
    def __init__(
        self,
        mesh_path,
        obj_mass=0.01,
        arm_friction=0.9,
        contact_stiffness=12.5,
        time_step=0.01,
        max_contacts=10,
        sample_num=70,
        pos_coef=1.0,
        ori_coef=0.0005,
        task_position_weights=(1.0, 1.0, 1.0),
        scale_factors=(1.0, 1.0, 1.0),
        sampling_stl_specs=None,
        nlp_solver=None,
        qp_device="auto",
        qp_maxiter=1000,
        qp_tol=1e-5,
        qp_regularization=1e-6,
        qp_objective_scale=100.0,
        qp_batch_size=8192,
    ):
        del nlp_solver  # Kept only so older callers fail gracefully without API churn.
        self.m = float(obj_mass)
        self.mu_arm_obj = float(arm_friction)
        self.K_contact = float(contact_stiffness)
        self.h = float(time_step)
        self.max_contacts = int(max_contacts)
        self.pos_coef = float(pos_coef)
        self.ori_coef = float(ori_coef)
        self.task_position_weights = np.asarray(
            task_position_weights, dtype=np.float64
        ).reshape(3)
        self.qp_regularization = max(float(qp_regularization), 1e-10)
        self.qp_objective_scale = max(float(qp_objective_scale), 1e-6)
        self.qp_batch_size = max(int(qp_batch_size), 1)
        self._default_lam_upper_bound = 2.0

        self.pp = _MeshProjection(mesh_path, scale_factors, sample_num)
        self.sample_point = self.pp.sample_points.copy()
        self.normal = self.pp.sample_normals.copy()
        self.t1 = self.pp.sample_t1.copy()
        self.t2 = self.pp.sample_t2.copy()
        self.sample_num = self.sample_point.shape[0]
        self.point_idx = np.arange(self.sample_num)

        self.sampling_stl_specs = sampling_stl_specs or {}
        self.sampling_part_masks = self._build_sampling_part_masks()
        self.J_tilde = np.zeros((4 * self.max_contacts, 6), dtype=np.float64)

        self.obj_inertia = np.eye(6, dtype=np.float64)
        self.obj_inertia[:3, :3] = 50.0 * np.eye(3)
        self.obj_inertia[3:, 3:] = 0.05 * np.eye(3)
        self.Q_inv = np.linalg.inv(self.obj_inertia + 1e-8 * np.eye(6))

        self._jax = None
        self._jnp = None
        self._device = "cpu"
        self.qp_device = "cpu"
        self.nlp_solver = None
        self._solver_backend = "daqp"
        self._qp_maxiter = int(qp_maxiter)
        self._qp_tol = float(qp_tol)
        self.optimization_fn = self._solve_batch

    def _select_device(self, requested):
        requested = str(requested or "auto").strip().lower()
        if requested in {"auto", "gpu", "cuda"} or requested.startswith("cuda:"):
            try:
                gpu_devices = self._jax.devices("gpu")
            except RuntimeError:
                gpu_devices = []
            if gpu_devices:
                index = 0
                if ":" in requested:
                    index = int(requested.split(":", 1)[1])
                if index >= len(gpu_devices):
                    raise ValueError(
                        f"Requested GPU index {index}, but JAX exposes {len(gpu_devices)} GPU(s)."
                    )
                return gpu_devices[index]
            if requested not in {"auto"}:
                raise RuntimeError(
                    f"QP device {requested!r} was requested, but JAX has no GPU device."
                )
        return self._jax.devices("cpu")[0]

    @staticmethod
    def _apply_sampling_spec(vertices, spec):
        transformed = np.asarray(vertices, dtype=np.float64).copy()
        scale = spec.get("scale_factors")
        if scale is not None:
            transformed *= np.asarray(scale, dtype=np.float64)
        rotation = spec.get("rotation")
        if rotation is not None:
            transformed = (
                transformed @ np.asarray(rotation, dtype=np.float64).reshape(3, 3).T
            )
        translation = spec.get("translation")
        if translation is not None:
            transformed += np.asarray(translation, dtype=np.float64).reshape(1, 3)
        transform = spec.get("transform")
        if transform is not None:
            homogeneous = np.concatenate(
                [transformed, np.ones((transformed.shape[0], 1))], axis=1
            )
            transformed = (
                homogeneous @ np.asarray(transform, dtype=np.float64).reshape(4, 4).T
            )[:, :3]
        return transformed

    def _build_sampling_part_masks(self):
        if not self.sampling_stl_specs:
            return {}
        closest_distance = np.full(self.sample_num, np.inf, dtype=np.float64)
        closest_name = np.full(self.sample_num, None, dtype=object)
        for name, spec in self.sampling_stl_specs.items():
            mesh = _load_mesh(spec["mesh_path"], (1.0, 1.0, 1.0))
            vertices = self._apply_sampling_spec(mesh.vertices, spec)
            distances, _ = cKDTree(vertices).query(self.sample_point, k=1)
            update = distances < closest_distance
            closest_distance[update] = distances[update]
            closest_name[update] = name
        masks = {}
        for name, spec in self.sampling_stl_specs.items():
            mask = closest_name == name
            if spec.get("point_bounds_min") is not None:
                mask &= np.all(
                    self.sample_point
                    >= np.asarray(spec["point_bounds_min"], dtype=np.float64).reshape(
                        1, 3
                    ),
                    axis=1,
                )
            if spec.get("point_bounds_max") is not None:
                mask &= np.all(
                    self.sample_point
                    <= np.asarray(spec["point_bounds_max"], dtype=np.float64).reshape(
                        1, 3
                    ),
                    axis=1,
                )
            masks[name] = np.flatnonzero(mask)
        return masks

    def update_Jacobian(self, J_tilde=None):
        if J_tilde is None:
            return
        required_rows = 4 * self.max_contacts
        jacobian = np.asarray(J_tilde, dtype=np.float64).reshape(-1, 6)
        self.J_tilde = np.zeros((required_rows, 6), dtype=np.float64)
        rows = min(required_rows, jacobian.shape[0])
        self.J_tilde[:rows] = jacobian[:rows]

    @staticmethod
    def compute_contact_jacobian(p):
        p = np.asarray(p, dtype=np.float64).reshape(3)
        jacobian = np.zeros((3, 6), dtype=np.float64)
        jacobian[:, :3] = np.eye(3)
        jacobian[0, 4], jacobian[0, 5] = p[2], -p[1]
        jacobian[1, 3], jacobian[1, 5] = -p[2], p[0]
        jacobian[2, 3], jacobian[2, 4] = p[1], -p[0]
        return jacobian

    @staticmethod
    def _broadcast_rows(value, count, width, name):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = np.broadcast_to(array.reshape(1, width), (count, width))
        elif array.shape != (count, width):
            raise ValueError(f"{name} must have shape ({width},) or ({count}, {width})")
        return np.asarray(array, dtype=np.float64)

    def _effective_inverse(self, tau_o):
        """Affine active-set model of the environment contact response."""
        if not np.any(self.J_tilde):
            return self.Q_inv
        free_velocity_impulse = self.Q_inv @ np.asarray(tau_o, dtype=np.float64)
        active = (-self.K_contact * self.h * self.J_tilde @ free_velocity_impulse) > 0.0
        contact_map = np.diag(active.astype(np.float64)) @ (
            -self.K_contact * self.h * self.J_tilde @ self.Q_inv
        )
        return self.Q_inv + self.Q_inv @ self.J_tilde.T @ contact_map

    def _assemble_qp_batch(
        self,
        x_d,
        current_x,
        tau_o,
        n_arm,
        t1,
        t2,
        p_arm,
        curr_ori_coef,
        lam_upper_bound,
    ):
        points = np.asarray(p_arm, dtype=np.float64).reshape(-1, 3)
        count = points.shape[0]
        normals = self._broadcast_rows(n_arm, count, 3, "n_arm")
        tangent1 = self._broadcast_rows(t1, count, 3, "t1")
        tangent2 = self._broadcast_rows(t2, count, 3, "t2")
        normals, generated_t1, generated_t2 = _contact_frames(normals)
        tangent1 = (
            tangent1 - np.sum(tangent1 * normals, axis=1, keepdims=True) * normals
        )
        invalid_t1 = np.linalg.norm(tangent1, axis=1) <= 1e-10
        tangent1[invalid_t1] = generated_t1[invalid_t1]
        tangent1 /= np.linalg.norm(tangent1, axis=1, keepdims=True).clip(min=1e-12)
        tangent2 = np.cross(normals, tangent1)
        invalid_t2 = np.linalg.norm(tangent2, axis=1) <= 1e-10
        tangent2[invalid_t2] = generated_t2[invalid_t2]
        tangent2 /= np.linalg.norm(tangent2, axis=1, keepdims=True).clip(min=1e-12)

        current = self._broadcast_rows(current_x, count, 7, "current_x")
        target = self._broadcast_rows(x_d, count, 7, "x_d")
        tau = self._broadcast_rows(tau_o, count, 6, "tau_o")
        if np.asarray(curr_ori_coef).ndim == 0:
            ori_scale = np.full(count, float(curr_ori_coef), dtype=np.float64)
        else:
            ori_scale = np.asarray(curr_ori_coef, dtype=np.float64).reshape(count)
        if np.asarray(lam_upper_bound).ndim == 0:
            upper = np.full(count, float(lam_upper_bound), dtype=np.float64)
        else:
            upper = np.asarray(lam_upper_bound, dtype=np.float64).reshape(count)
        upper = np.maximum(upper, 1e-3)

        jacobians = np.zeros((count, 3, 6), dtype=np.float64)
        jacobians[:, :, :3] = np.eye(3)[None]
        jacobians[:, 0, 4] = points[:, 2]
        jacobians[:, 0, 5] = -points[:, 1]
        jacobians[:, 1, 3] = -points[:, 2]
        jacobians[:, 1, 5] = points[:, 0]
        jacobians[:, 2, 3] = points[:, 1]
        jacobians[:, 2, 4] = -points[:, 0]
        contact_rotation = np.stack([normals, tangent1, tangent2], axis=-1)
        wrench_map = np.einsum("bji,bjk->bik", jacobians, contact_rotation)

        state_maps = np.empty((count, 6, 3), dtype=np.float64)
        base_delta = np.empty((count, 6), dtype=np.float64)
        for index in range(count):
            effective_inverse = self._effective_inverse(tau[index])
            state_maps[index] = effective_inverse @ wrench_map[index]
            base_delta[index] = effective_inverse @ tau[index]

        target_delta = np.concatenate(
            [
                target[:, :3] - current[:, :3],
                _relative_rotvec(current[:, 3:7], target[:, 3:7]),
            ],
            axis=1,
        )
        weights = np.empty((count, 6), dtype=np.float64)
        weights[:, :3] = self.pos_coef * self.task_position_weights.reshape(1, 3)
        weights[:, 3:] = self.ori_coef * ori_scale[:, None]
        residual_zero = base_delta - target_delta
        weighted_maps = weights[:, :, None] * state_maps
        q_matrices = 2.0 * np.einsum("bki,bkj->bij", state_maps, weighted_maps)
        q_matrices += 2.0 * self.qp_regularization * np.eye(3)[None]
        linear = 2.0 * np.einsum("bki,bk->bi", state_maps, weights * residual_zero)

        base_g = np.asarray(
            [
                [-self.mu_arm_obj, 1.0, 0.0],
                [-self.mu_arm_obj, -1.0, 0.0],
                [-self.mu_arm_obj, 0.0, 1.0],
                [-self.mu_arm_obj, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        inequality_matrix = np.broadcast_to(base_g, (count, 6, 3)).copy()
        inequality_bound = np.zeros((count, 6), dtype=np.float64)
        inequality_bound[:, 4] = -1e-3
        inequality_bound[:, 5] = upper
        return (
            q_matrices,
            linear,
            inequality_matrix,
            inequality_bound,
            state_maps,
            base_delta,
            target_delta,
            weights,
            current,
        )

    def _solve_batch(
        self,
        x_d,
        current_x,
        tau_o,
        n_arm,
        t1,
        t2,
        p_arm,
        curr_ori_coef=1.0,
        lam_upper_bound=None,
        batch_size=None,
        task_target_position=None,
        task_current_position=None,
    ):
        points = np.asarray(p_arm, dtype=np.float64).reshape(-1, 3)
        count = points.shape[0]
        if count == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 7), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
                np.asarray([], dtype=str),
            )
        if lam_upper_bound is None:
            lam_upper_bound = self._default_lam_upper_bound
        if task_target_position is not None:
            x_d = np.asarray(x_d, dtype=np.float64).copy()
            if x_d.ndim == 1:
                x_d[:3] = np.asarray(task_target_position, dtype=np.float64).reshape(3)
            else:
                x_d[:, :3] = self._broadcast_rows(
                    task_target_position, count, 3, "task_target_position"
                )
        if task_current_position is not None:
            current_x = np.asarray(current_x, dtype=np.float64).copy()
            if current_x.ndim == 1:
                current_x[:3] = np.asarray(
                    task_current_position, dtype=np.float64
                ).reshape(3)
            else:
                current_x[:, :3] = self._broadcast_rows(
                    task_current_position, count, 3, "task_current_position"
                )
        assembled = self._assemble_qp_batch(
            x_d,
            current_x,
            tau_o,
            n_arm,
            t1,
            t2,
            points,
            curr_ori_coef,
            lam_upper_bound,
        )
        q_matrices, linear, matrix_g, bound_h = assembled[:4]
        state_maps, base_delta, target_delta, weights, current = assembled[4:]
        chunk_size = max(int(batch_size or self.qp_batch_size), 1)

        solutions = np.empty((count, 3), dtype=np.float32)
        status_codes = np.empty(count, dtype=np.int32)
        errors = np.empty(count, dtype=np.float32)
        iterations = np.empty(count, dtype=np.int32)
        del chunk_size
        Q_scaled = (q_matrices * self.qp_objective_scale).astype(np.float64)
        f_scaled = (linear * self.qp_objective_scale).astype(np.float64)
        A_all = matrix_g.astype(np.float64)
        b_all = bound_h.astype(np.float64)
        # bound_h column 4 is a *lower* bound on the normal force: ineq row 4
        # in matrix_g is [+1, 0, 0] meaning lam_n >= 1e-3 (bound_h[4]=-1e-3
        # is a >= using -row form). Convert all ineq to bupper form (A x <= b)
        # consistent with the original convention (matrix_g x <= bound_h).
        for index in range(count):
            sol, status, err, iters = _solve_qp_daqp(
                Q_scaled[index],
                f_scaled[index],
                A_all[index],
                b_all[index],
                blower=None,
                sense=None,
                max_iter=self._qp_maxiter,
                tol=self._qp_tol,
            )
            solutions[index] = np.asarray(sol, dtype=np.float32)
            status_codes[index] = int(status)
            errors[index] = float(err)
            iterations[index] = int(iters)

        delta = base_delta + np.einsum("bij,bj->bi", state_maps, solutions)
        resulting_pose = _integrate_pose(current, delta).astype(np.float32)
        residual = delta - target_delta
        costs = np.sum(weights * residual * residual, axis=1)
        solved = status_codes == 1
        costs[~solved] = np.inf
        statuses = np.asarray(
            [
                (
                    f"DAQP_SOLVED(iter={iterations[index]},err={errors[index]:.3e})"
                    if solved[index]
                    else (
                        f"DAQP_STATUS_{status_codes[index]}"
                        f"(iter={iterations[index]},err={errors[index]:.3e})"
                    )
                )
                for index in range(count)
            ],
            dtype=str,
        )
        return solutions, resulting_pose, costs, statuses

    def _solve_once(
        self,
        x_d,
        current_x,
        tau_o,
        n_arm,
        t1,
        t2,
        p_arm,
        curr_ori_coef=1.0,
        lam_upper_bound=None,
        task_target_position=None,
        task_current_position=None,
    ):
        if np.asarray(p_arm).ndim == 2:
            return self._solve_batch(
                x_d=x_d,
                current_x=current_x,
                tau_o=tau_o,
                n_arm=n_arm,
                t1=t1,
                t2=t2,
                p_arm=p_arm,
                curr_ori_coef=curr_ori_coef,
                lam_upper_bound=lam_upper_bound,
                task_target_position=task_target_position,
                task_current_position=task_current_position,
            )
        solution, resulting_pose, cost, status = self._solve_batch(
            x_d=x_d,
            current_x=current_x,
            tau_o=tau_o,
            n_arm=np.asarray(n_arm).reshape(1, 3),
            t1=np.asarray(t1).reshape(1, 3),
            t2=np.asarray(t2).reshape(1, 3),
            p_arm=np.asarray(p_arm).reshape(1, 3),
            curr_ori_coef=curr_ori_coef,
            lam_upper_bound=lam_upper_bound,
            batch_size=1,
            task_target_position=task_target_position,
            task_current_position=task_current_position,
        )
        return solution[0], resulting_pose[0], float(cost[0]), str(status[0])

    def optimize_control_input(
        self,
        x_d,
        current_x,
        tau_o,
        p_arm=None,
        task_target_position=None,
        task_current_position=None,
    ):
        if p_arm is None:
            p_arm = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        target_quat = _quat_normalize(np.asarray(x_d, dtype=np.float64)[3:7])
        current_quat = _quat_normalize(np.asarray(current_x, dtype=np.float64)[3:7])
        alignment_sq = float(np.dot(current_quat, target_quat) ** 2)
        curr_ori_coef = 1.0 + np.tanh(10.0 * (alignment_sq - 0.85))
        closest_idx, normal, tangent1, tangent2 = self.pp.project_point_to_mesh(p_arm)
        point = self.pp.vertices[closest_idx]
        started_at = time.perf_counter()
        lam, x_plus, objective, status = self._solve_once(
            x_d,
            current_x,
            tau_o,
            normal,
            tangent1,
            tangent2,
            point,
            curr_ori_coef,
            task_target_position=task_target_position,
            task_current_position=task_current_position,
        )
        return (
            point,
            self.pp.vertex_normals[closest_idx],
            x_plus,
            objective,
            {
                "solve_time": time.perf_counter() - started_at,
                "control_input": lam,
                "resulting_pose": x_plus,
                "solver_status": status,
                "solver_backend": self._solver_backend,
                "qp_device": self.qp_device,
            },
        )

    def choose_contact_points(
        self,
        x_d,
        current_x,
        tau_o,
        visible_face_idx,
        task_target_position=None,
        task_current_position=None,
    ):
        visible_face_idx = np.asarray(visible_face_idx, dtype=np.int64).reshape(-1)
        if visible_face_idx.size == 0:
            return self.sample_point[0], self.normal[0], 1.5, 1.0, 0.0
        target_quat = _quat_normalize(np.asarray(x_d, dtype=np.float64)[3:7])
        current_quat = _quat_normalize(np.asarray(current_x, dtype=np.float64)[3:7])
        alignment_sq = float(np.dot(current_quat, target_quat) ** 2)
        curr_ori_coef = 1.0 + np.tanh(10.0 * (alignment_sq - 0.85))
        _, _, costs, _ = self._solve_batch(
            x_d=x_d,
            current_x=current_x,
            tau_o=tau_o,
            n_arm=self.normal[visible_face_idx],
            t1=self.t1[visible_face_idx],
            t2=self.t2[visible_face_idx],
            p_arm=self.sample_point[visible_face_idx],
            curr_ori_coef=curr_ori_coef,
            task_target_position=task_target_position,
            task_current_position=task_current_position,
        )
        best_local = int(np.argmin(costs))
        best_index = int(visible_face_idx[best_local])
        return (
            self.sample_point[best_index],
            self.normal[best_index],
            float(np.min(costs)),
            float(np.max(costs)),
            float(curr_ori_coef),
        )

    def get_availble_point_idx(
        self, pos, R, target_pos, threshold=0.025, stl_names=None
    ):
        del target_pos
        centers_world = (
            self.sample_point @ np.asarray(R, dtype=np.float64).reshape(3, 3).T
        )
        centers_world += np.asarray(pos, dtype=np.float64).reshape(1, 3)
        mask = centers_world[:, 2] > float(threshold)
        if stl_names is not None:
            if isinstance(stl_names, str):
                stl_names = [stl_names]
            allowed = np.zeros(self.sample_num, dtype=bool)
            unknown = []
            for name in stl_names:
                if name not in self.sampling_part_masks:
                    unknown.append(name)
                else:
                    allowed[self.sampling_part_masks[name]] = True
            if unknown:
                raise ValueError(
                    f"Unknown sampling STL names {unknown}. "
                    f"Known names: {sorted(self.sampling_part_masks)}"
                )
            mask &= allowed
        return np.flatnonzero(mask)
