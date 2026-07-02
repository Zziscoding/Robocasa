"""Sampling-based Franka end-effector MPC with Comfree Warp rollouts.

The generic entry point is :func:`solve_ee_pose_mpc`.  Candidate actions are
absolute end-effector poses in world coordinates:

    [x, y, z, qw, qx, qy, qz]

The solver keeps the sampling dimensions used by
``do-as-i-do/retargeting/config/override/do_as_i_do.yaml`` by default:
1024 candidates, 4 disturbance replicas, a 3 second horizon, 5 ms simulation
steps, and 200 ms sampling knots.

Task semantics intentionally live outside this module.  A task supplies a
``cost_fn(RolloutStep) -> Tensor[nworld]`` callback and may inspect any batched
MuJoCo/Comfree state exposed by ``RolloutStep``.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from robocasa.demos.open_drawer.utils import _empty_cuda_caches


Tensor = torch.Tensor


def _backend_tensor(value: Any) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value
    warp = importlib.import_module("warp")
    return warp.to_torch(value)


def _raw_model(model):
    return getattr(model, "_model", model)


def _raw_data(data):
    return getattr(data, "_data", data)


def _object_body_id_for_surface(env, surface):
    model = env.sim.model
    geom_name = str(getattr(surface, "geom_name", ""))
    if geom_name in model._geom_name2id:
        return int(model.geom_bodyid[model.geom_name2id(geom_name)])
    drawer_body = getattr(getattr(env, "drawer", None), "root_body", "")
    if drawer_body and drawer_body in model._body_name2id:
        return int(model.body_name2id(drawer_body))
    return 0


def _linear_pose_nominal(start_pose, target_pose, steps):
    start_pose = np.asarray(start_pose, dtype=np.float64).reshape(7)
    target_pose = np.asarray(target_pose, dtype=np.float64).reshape(7)
    steps = max(int(steps), 2)
    alpha = np.linspace(0.0, 1.0, steps, dtype=np.float64)[:, None]
    poses = (1.0 - alpha) * start_pose + alpha * target_pose
    quat = poses[:, 3:]
    if np.dot(start_pose[3:], target_pose[3:]) < 0.0:
        target_quat = -target_pose[3:]
        poses[:, 3:] = (1.0 - alpha) * start_pose[3:] + alpha * target_quat
        quat = poses[:, 3:]
    poses[:, 3:] = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
    return poses


class TaskCost(Protocol):
    """Per-step task cost interface.

    Return one non-negative cost per simulated world.  A scalar is broadcast,
    and one value per candidate is repeated over disturbance replicas.
    """

    def __call__(self, step: "RolloutStep") -> Tensor:
        ...


@dataclass
class DreamConfig:
    """MPC, rollout, and task-space controller configuration.

    Defaults mirror the active ``do_as_i_do`` retargeting override.  Set
    ``num_perturb_samples=1`` when the task does not install a stochastic
    ``pre_step_fn``.
    """

    device: str = "cuda:0"
    seed: int = 0

    # Retargeting rollout dimensions.
    num_samples: int = 1024
    num_perturb_samples: int = 4
    sim_dt: float = 0.005
    horizon: float = 3.0
    knot_dt: float = 0.2
    nconmax_per_env: int = 120
    njmax_per_env: int = 500

    # DIAL/MPPI-style sampling schedule.
    max_num_iterations: int = 32
    temperature: float = 1.0
    elite_ratio: float = 0.1
    final_noise_scale: float = 0.01
    first_ctrl_noise_scale: float = 1.0
    last_ctrl_noise_scale: float = 4.0
    pos_noise_scale: float = 0.01
    rot_noise_scale: float = 0.01
    exploit_ratio: float = 0.01
    exploit_noise_scale: float = 0.01
    zero_first_knot_noise: bool = True
    optimize_initial_pose: bool = False
    improvement_threshold: float = 0.01
    improvement_check_steps: int = 2
    initial_ik_iterations: int = 24
    initial_ik_damping: float = 1e-2
    initial_ik_step_scale: float = 0.7
    initial_ik_max_step_norm: float = 0.25
    initial_pose_tolerance: float = 0.01
    initial_pose_error_weight: float = 100.0

    # Comfree contact model.
    contact_stiffness: float = 0.2
    contact_damping: float = 0.001

    # Franka operational-space PD controller.
    position_kp: float = 350.0
    position_kd: float = 35.0
    rotation_kp: float = 80.0
    rotation_kd: float = 8.0
    joint_damping: float = 1.0
    max_joint_torque: tuple[float, ...] = (
        87.0,
        87.0,
        87.0,
        87.0,
        12.0,
        12.0,
        12.0,
    )
    neutralize_arm_actuators: bool = True
    compile_cuda_graph: bool = True
    cost_aggregation: str = "sum"

    # Backward-compatible fields used by OpenDrawerDreamModel.
    dt: float | None = None
    max_open_distance: float = 0.5
    contact_radius: float = 0.04
    max_contact_force: float = 150.0
    ee_velocity_limit: float = 0.25
    control_mode: str = "velocity"

    horizon_steps: int = field(init=False)
    knot_steps: int = field(init=False)
    num_worlds: int = field(init=False)

    def __post_init__(self) -> None:
        if self.dt is not None:
            self.sim_dt = float(self.dt)
        self.horizon_steps = int(round(self.horizon / self.sim_dt))
        self.knot_steps = int(round(self.knot_dt / self.sim_dt))
        self.num_worlds = int(self.num_samples) * int(self.num_perturb_samples)
        if self.num_samples < 1 or self.num_perturb_samples < 1:
            raise ValueError("num_samples and num_perturb_samples must be positive")
        if self.max_num_iterations < 1:
            raise ValueError("max_num_iterations must be positive")
        if not 0.0 < self.elite_ratio <= 1.0:
            raise ValueError("elite_ratio must be in (0, 1]")
        if self.horizon_steps < 1 or self.knot_steps < 1:
            raise ValueError("horizon and knot_dt must be at least one sim_dt")
        if not math.isclose(
            self.horizon_steps * self.sim_dt, self.horizon, abs_tol=1e-7
        ):
            raise ValueError("horizon must be divisible by sim_dt")
        if not math.isclose(self.knot_steps * self.sim_dt, self.knot_dt, abs_tol=1e-7):
            raise ValueError("knot_dt must be divisible by sim_dt")
        if len(self.max_joint_torque) != 7:
            raise ValueError("Franka max_joint_torque must contain 7 values")
        if self.cost_aggregation not in {"sum", "mean"}:
            raise ValueError("cost_aggregation must be 'sum' or 'mean'")


@dataclass(frozen=True)
class RolloutStep:
    """Batched state passed to task costs and pre-step hooks."""

    step_index: int
    is_terminal: bool
    target_ee_pose: Tensor
    ee_position: Tensor
    ee_rotation: Tensor
    qpos: Tensor
    qvel: Tensor
    data: Any
    model: Any
    candidate_ids: Tensor
    replica_ids: Tensor
    ee_jacobian: Tensor | None = None
    arm_qpos: Tensor | None = None
    arm_joint_lower: Tensor | None = None
    arm_joint_upper: Tensor | None = None


@dataclass
class ObjectFramePositionTaskCost:
    """Squared moving-body position error in a parent/object frame."""

    object_body_id: int
    frame_body_id: int
    target_position_local: np.ndarray
    weight: float = 1.0
    _target_tensor: Tensor | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_mujoco(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        object_body_id: int,
        frame_body_id: int,
        target_position_local: Sequence[float],
        weight: float = 1.0,
    ) -> "ObjectFramePositionTaskCost":
        del model, data
        return cls(
            object_body_id=int(object_body_id),
            frame_body_id=int(frame_body_id),
            target_position_local=np.asarray(
                target_position_local, dtype=np.float64
            ).reshape(3),
            weight=float(weight),
        )

    @staticmethod
    def local_position_numpy(
        data: mujoco.MjData,
        object_body_id: int,
        frame_body_id: int,
    ) -> np.ndarray:
        frame_position = np.asarray(data.xpos[frame_body_id], dtype=np.float64)
        frame_rotation = np.asarray(data.xmat[frame_body_id], dtype=np.float64).reshape(
            3, 3
        )
        object_position = np.asarray(data.xpos[object_body_id], dtype=np.float64)
        return frame_rotation.T @ (object_position - frame_position)

    def evaluate_numpy(self, current_position_local: Sequence[float]) -> float:
        error = (
            np.asarray(current_position_local, dtype=np.float64).reshape(3)
            - self.target_position_local
        )
        return float(self.weight * np.dot(error, error))

    def __call__(self, step: RolloutStep) -> Tensor:
        positions = _backend_tensor(step.data.xpos)
        rotations = _backend_tensor(step.data.xmat)
        frame_position = positions[:, self.frame_body_id]
        frame_rotation = rotations[:, self.frame_body_id]
        object_position = positions[:, self.object_body_id]
        position_local = torch.bmm(
            frame_rotation.transpose(1, 2),
            (object_position - frame_position).unsqueeze(-1),
        ).squeeze(-1)
        if (
            self._target_tensor is None
            or self._target_tensor.device != position_local.device
            or self._target_tensor.dtype != position_local.dtype
        ):
            self._target_tensor = torch.as_tensor(
                self.target_position_local,
                device=position_local.device,
                dtype=position_local.dtype,
            )
        return float(self.weight) * (position_local - self._target_tensor).square().sum(
            dim=-1
        )


@dataclass
class ObjectFramePointPositionTaskCost:
    """Squared position error for a body-fixed point in a reference frame.

    Tracking a point away from a hinge gives rotational articulated objects a
    non-degenerate position objective even when the moving body's origin lies
    on the joint axis.
    """

    object_body_id: int
    frame_body_id: int
    point_position_object: np.ndarray
    target_position_local: np.ndarray
    weight: float = 1.0
    _point_tensor: Tensor | None = field(default=None, init=False, repr=False)
    _target_tensor: Tensor | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_mujoco(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        object_body_id: int,
        frame_body_id: int,
        point_position_object: Sequence[float],
        target_position_local: Sequence[float],
        weight: float = 1.0,
    ) -> "ObjectFramePointPositionTaskCost":
        del model, data
        return cls(
            object_body_id=int(object_body_id),
            frame_body_id=int(frame_body_id),
            point_position_object=np.asarray(
                point_position_object, dtype=np.float64
            ).reshape(3),
            target_position_local=np.asarray(
                target_position_local, dtype=np.float64
            ).reshape(3),
            weight=float(weight),
        )

    @staticmethod
    def local_position_numpy(
        data: mujoco.MjData,
        object_body_id: int,
        frame_body_id: int,
        point_position_object: Sequence[float],
    ) -> np.ndarray:
        object_position = np.asarray(data.xpos[object_body_id], dtype=np.float64)
        object_rotation = np.asarray(
            data.xmat[object_body_id], dtype=np.float64
        ).reshape(3, 3)
        point_world = object_position + object_rotation @ np.asarray(
            point_position_object, dtype=np.float64
        ).reshape(3)
        frame_position = np.asarray(data.xpos[frame_body_id], dtype=np.float64)
        frame_rotation = np.asarray(data.xmat[frame_body_id], dtype=np.float64).reshape(
            3, 3
        )
        return frame_rotation.T @ (point_world - frame_position)

    def evaluate_numpy(self, current_position_local: Sequence[float]) -> float:
        error = (
            np.asarray(current_position_local, dtype=np.float64).reshape(3)
            - self.target_position_local
        )
        return float(self.weight * np.dot(error, error))

    def __call__(self, step: RolloutStep) -> Tensor:
        positions = _backend_tensor(step.data.xpos)
        rotations = _backend_tensor(step.data.xmat)
        object_position = positions[:, self.object_body_id]
        object_rotation = rotations[:, self.object_body_id]
        frame_position = positions[:, self.frame_body_id]
        frame_rotation = rotations[:, self.frame_body_id]
        if (
            self._point_tensor is None
            or self._point_tensor.device != object_position.device
            or self._point_tensor.dtype != object_position.dtype
        ):
            self._point_tensor = torch.as_tensor(
                self.point_position_object,
                device=object_position.device,
                dtype=object_position.dtype,
            )
        if (
            self._target_tensor is None
            or self._target_tensor.device != object_position.device
            or self._target_tensor.dtype != object_position.dtype
        ):
            self._target_tensor = torch.as_tensor(
                self.target_position_local,
                device=object_position.device,
                dtype=object_position.dtype,
            )
        point_world = object_position + torch.bmm(
            object_rotation,
            self._point_tensor.expand(object_position.shape[0], -1).unsqueeze(-1),
        ).squeeze(-1)
        position_local = torch.bmm(
            frame_rotation.transpose(1, 2),
            (point_world - frame_position).unsqueeze(-1),
        ).squeeze(-1)
        return float(self.weight) * (position_local - self._target_tensor).square().sum(
            dim=-1
        )


@dataclass
class JointPositionTaskCost:
    """Squared qpos target for one scalar joint."""

    qpos_index: int
    target_position: float
    weight: float = 1.0

    def __call__(self, step: RolloutStep) -> Tensor:
        q = step.qpos[:, int(self.qpos_index)]
        return float(self.weight) * (q - float(self.target_position)).square()


@dataclass
class FeasibleContactRewardCost:
    """Reward EE-object contacts whose contact positions are near feasible points."""

    geom_set_a: Sequence[int]
    geom_set_b: Sequence[int]
    feasible_points_world: np.ndarray
    max_distance: float = 0.025
    reward: float = 1.0
    start_step: int = 0
    _mask_a: Tensor | None = field(default=None, init=False, repr=False)
    _mask_b: Tensor | None = field(default=None, init=False, repr=False)
    _feasible_tensor: Tensor | None = field(default=None, init=False, repr=False)
    _contact_ids: Tensor | None = field(default=None, init=False, repr=False)

    def _prepare(self, step: RolloutStep) -> None:
        device = step.qpos.device
        dtype = step.qpos.dtype
        if self._mask_a is None or self._mask_a.device != device:
            ngeom = int(step.model.geom_bodyid.shape[0])
            self._mask_a = torch.zeros(ngeom, dtype=torch.bool, device=device)
            self._mask_b = torch.zeros(ngeom, dtype=torch.bool, device=device)
            self._mask_a[
                torch.as_tensor(self.geom_set_a, device=device, dtype=torch.long)
            ] = True
            self._mask_b[
                torch.as_tensor(self.geom_set_b, device=device, dtype=torch.long)
            ] = True
        if (
            self._feasible_tensor is None
            or self._feasible_tensor.device != device
            or self._feasible_tensor.dtype != dtype
        ):
            self._feasible_tensor = torch.as_tensor(
                self.feasible_points_world, device=device, dtype=dtype
            ).reshape(-1, 3)
        contact_capacity = int(step.data.contact.dist.shape[0])
        if (
            self._contact_ids is None
            or self._contact_ids.device != device
            or self._contact_ids.numel() != contact_capacity
        ):
            self._contact_ids = torch.arange(
                contact_capacity, device=device, dtype=torch.long
            )

    def __call__(self, step: RolloutStep) -> Tensor:
        if int(step.step_index) < int(self.start_step):
            return torch.zeros(
                step.qpos.shape[0],
                device=step.qpos.device,
                dtype=step.qpos.dtype,
            )
        self._prepare(step)
        assert self._mask_a is not None
        assert self._mask_b is not None
        assert self._feasible_tensor is not None
        assert self._contact_ids is not None
        if self._feasible_tensor.numel() == 0:
            return torch.zeros(
                step.qpos.shape[0],
                device=step.qpos.device,
                dtype=step.qpos.dtype,
            )
        geom = _backend_tensor(step.data.contact.geom).long().reshape(-1, 2)
        world_ids = _backend_tensor(step.data.contact.worldid).long().reshape(-1)
        contact_pos = _backend_tensor(step.data.contact.pos).reshape(-1, 3)
        nacon = _backend_tensor(step.data.nacon).long().reshape(-1)[0]
        active = self._contact_ids < nacon
        valid_geom = (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
        safe_g0 = geom[:, 0].clamp(0, self._mask_a.numel() - 1)
        safe_g1 = geom[:, 1].clamp(0, self._mask_a.numel() - 1)
        pair = (self._mask_a[safe_g0] & self._mask_b[safe_g1]) | (
            self._mask_b[safe_g0] & self._mask_a[safe_g1]
        )
        nearest = torch.cdist(contact_pos, self._feasible_tensor).amin(dim=1)
        feasible = nearest <= float(self.max_distance)
        hit = active & valid_geom & pair & feasible
        per_world = torch.zeros(
            step.qpos.shape[0], device=step.qpos.device, dtype=step.qpos.dtype
        )
        per_world.scatter_add_(
            0,
            world_ids.clamp(0, per_world.numel() - 1),
            hit.to(step.qpos.dtype),
        )
        # Return a negative cost so more feasible contacts are better.
        return -float(self.reward) * per_world


@dataclass
class NonPenetrationCost:
    """Per-world max penetration squared, matching retargeting's penalty."""

    geom_set_a: Sequence[int]
    geom_set_b: Sequence[int]
    margin: float = 0.003
    weight: float = 1.0
    start_step: int = 0
    _mask_a: Tensor | None = field(default=None, init=False, repr=False)
    _mask_b: Tensor | None = field(default=None, init=False, repr=False)
    _contact_ids: Tensor | None = field(default=None, init=False, repr=False)

    def _prepare(self, step: RolloutStep) -> None:
        device = step.qpos.device
        if self._mask_a is None or self._mask_a.device != device:
            ngeom = int(step.model.geom_bodyid.shape[0])
            self._mask_a = torch.zeros(ngeom, dtype=torch.bool, device=device)
            self._mask_b = torch.zeros(ngeom, dtype=torch.bool, device=device)
            self._mask_a[
                torch.as_tensor(self.geom_set_a, device=device, dtype=torch.long)
            ] = True
            self._mask_b[
                torch.as_tensor(self.geom_set_b, device=device, dtype=torch.long)
            ] = True
        contact_capacity = int(step.data.contact.dist.shape[0])
        if (
            self._contact_ids is None
            or self._contact_ids.device != device
            or self._contact_ids.numel() != contact_capacity
        ):
            self._contact_ids = torch.arange(
                contact_capacity, device=device, dtype=torch.long
            )

    def __call__(self, step: RolloutStep) -> Tensor:
        if int(step.step_index) < int(self.start_step):
            return torch.zeros(
                step.qpos.shape[0],
                device=step.qpos.device,
                dtype=step.qpos.dtype,
            )
        self._prepare(step)
        assert self._mask_a is not None
        assert self._mask_b is not None
        assert self._contact_ids is not None
        dist = _backend_tensor(step.data.contact.dist).reshape(-1)
        geom = _backend_tensor(step.data.contact.geom).long().reshape(-1, 2)
        world_ids = _backend_tensor(step.data.contact.worldid).long().reshape(-1)
        nacon = _backend_tensor(step.data.nacon).long().reshape(-1)[0]
        active = self._contact_ids < nacon
        valid_geom = (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
        safe_g0 = geom[:, 0].clamp(0, self._mask_a.numel() - 1)
        safe_g1 = geom[:, 1].clamp(0, self._mask_a.numel() - 1)
        pair = (self._mask_a[safe_g0] & self._mask_b[safe_g1]) | (
            self._mask_b[safe_g0] & self._mask_a[safe_g1]
        )
        penetration = torch.clamp(-dist - float(self.margin), min=0.0)
        penetration = torch.where(
            active & valid_geom & pair,
            torch.nan_to_num(penetration, nan=0.0),
            torch.zeros_like(penetration),
        )
        per_world = torch.zeros(
            step.qpos.shape[0], device=step.qpos.device, dtype=step.qpos.dtype
        )
        per_world.scatter_reduce_(
            0,
            world_ids.clamp(0, per_world.numel() - 1),
            penetration.square(),
            reduce="amax",
            include_self=True,
        )
        return float(self.weight) * per_world


@dataclass
class FingerContactSetLineOfSightCost:
    """Match contact sets to sampled gripper points without line penetration.

    Each MPC sample owns one contact set in an object body frame.  At a rollout
    step, contact points are transformed by the live object pose, sampled finger
    points are transformed by the live EE pose, and each contact point is scored
    against its best visible finger point. COACD convex-part half-spaces provide
    an exact inside test for every sampled segment point in the object frame.
    """

    object_body_id: int
    contact_points_object: np.ndarray
    finger_points_ee: np.ndarray
    object_convex_equations: np.ndarray
    object_convex_equation_mask: np.ndarray
    proximity_weight: float = 500.0
    line_penetration_weight: float = 200.0
    line_margin: float = 0.001
    line_samples: int = 7
    start_step: int = 0
    contact_mask: np.ndarray | None = None
    _contact_tensor: Tensor | None = field(default=None, init=False, repr=False)
    _finger_tensor: Tensor | None = field(default=None, init=False, repr=False)
    _equation_tensor: Tensor | None = field(default=None, init=False, repr=False)
    _equation_mask_tensor: Tensor | None = field(default=None, init=False, repr=False)
    _alpha_tensor: Tensor | None = field(default=None, init=False, repr=False)
    _mask_tensor: Tensor | None = field(default=None, init=False, repr=False)

    def _prepare(self, step: RolloutStep) -> None:
        device = step.qpos.device
        dtype = step.qpos.dtype
        if (
            self._contact_tensor is None
            or self._contact_tensor.device != device
            or self._contact_tensor.dtype != dtype
        ):
            contacts_np = np.asarray(self.contact_points_object, dtype=np.float64)
            if contacts_np.ndim != 3 or contacts_np.shape[-1] != 3:
                raise ValueError(
                    "contact_points_object must have shape (num_samples, K, 3)"
                )
            fingers_np = np.asarray(self.finger_points_ee, dtype=np.float64)
            if fingers_np.ndim != 2 or fingers_np.shape[-1] != 3:
                raise ValueError("finger_points_ee must have shape (F, 3)")
            self._contact_tensor = torch.as_tensor(
                contacts_np, device=device, dtype=dtype
            )
            self._finger_tensor = torch.as_tensor(
                fingers_np, device=device, dtype=dtype
            )
            equations_np = np.asarray(self.object_convex_equations, dtype=np.float64)
            equation_mask_np = np.asarray(self.object_convex_equation_mask, dtype=bool)
            if equations_np.ndim != 3 or equations_np.shape[-1] != 4:
                raise ValueError("object_convex_equations must have shape (P, H, 4)")
            if equation_mask_np.shape != equations_np.shape[:2]:
                raise ValueError(
                    "object_convex_equation_mask must have shape "
                    "object_convex_equations[:2]"
                )
            self._equation_tensor = torch.as_tensor(
                equations_np, device=device, dtype=dtype
            )
            self._equation_mask_tensor = torch.as_tensor(
                equation_mask_np, device=device, dtype=torch.bool
            )
            count = max(int(self.line_samples), 2)
            # Endpoints are allowed to lie on the surface; interior samples
            # carry the penetration check.
            self._alpha_tensor = torch.linspace(
                0.0,
                1.0,
                count,
                device=device,
                dtype=dtype,
            )[1:-1]
            if self.contact_mask is None:
                self._mask_tensor = torch.ones(
                    contacts_np.shape[:2], device=device, dtype=torch.bool
                )
            else:
                mask_np = np.asarray(self.contact_mask, dtype=bool)
                if mask_np.shape != contacts_np.shape[:2]:
                    raise ValueError(
                        "contact_mask must have shape contact_points_object[:2]"
                    )
                self._mask_tensor = torch.as_tensor(
                    mask_np, device=device, dtype=torch.bool
                )

    def __call__(self, step: RolloutStep) -> Tensor:
        if int(step.step_index) < int(self.start_step):
            return torch.zeros(
                step.qpos.shape[0],
                device=step.qpos.device,
                dtype=step.qpos.dtype,
            )
        self._prepare(step)
        assert self._contact_tensor is not None
        assert self._finger_tensor is not None
        assert self._equation_tensor is not None
        assert self._equation_mask_tensor is not None
        assert self._alpha_tensor is not None
        assert self._mask_tensor is not None

        positions = _backend_tensor(step.data.xpos)
        rotations = _backend_tensor(step.data.xmat)
        object_position = positions[:, int(self.object_body_id)]
        object_rotation = rotations[:, int(self.object_body_id)]
        contacts_local = self._contact_tensor[step.candidate_ids]
        contacts_world = object_position[:, None, :] + torch.bmm(
            object_rotation,
            contacts_local.transpose(1, 2),
        ).transpose(1, 2)

        fingers_world = step.ee_position[:, None, :] + torch.bmm(
            step.ee_rotation,
            self._finger_tensor.t()
            .unsqueeze(0)
            .expand(step.ee_position.shape[0], -1, -1),
        ).transpose(1, 2)

        delta = contacts_world[:, :, None, :] - fingers_world[:, None, :, :]
        distance_cost = delta.square().sum(dim=-1)

        if self._alpha_tensor.numel() == 0:
            line_cost = torch.zeros_like(distance_cost)
        else:
            line_points = (
                fingers_world[:, None, :, None, :]
                + self._alpha_tensor.view(1, 1, 1, -1, 1) * delta[:, :, :, None, :]
            )
            line_local = torch.einsum(
                "wkfld,wdj->wkflj",
                line_points - object_position[:, None, None, None, :],
                object_rotation,
            )
            max_penetration = torch.zeros(
                line_local.shape[:-1],
                device=step.qpos.device,
                dtype=step.qpos.dtype,
            )
            for part_index in range(self._equation_tensor.shape[0]):
                equations = self._equation_tensor[part_index]
                plane_mask = self._equation_mask_tensor[part_index]
                signed = (
                    torch.einsum(
                        "...d,hd->...h",
                        line_local,
                        equations[:, :3],
                    )
                    + equations[:, 3]
                )
                signed = torch.where(
                    plane_mask.view(*([1] * (signed.ndim - 1)), -1),
                    signed,
                    torch.full_like(signed, -torch.inf),
                )
                deepest_plane = signed.amax(dim=-1)
                penetration = torch.clamp(
                    -deepest_plane - float(self.line_margin),
                    min=0.0,
                )
                max_penetration = torch.maximum(
                    max_penetration,
                    penetration,
                )
            line_cost = max_penetration.square().amax(dim=-1)

        pair_cost = (
            float(self.proximity_weight) * distance_cost
            + float(self.line_penetration_weight) * line_cost
        )
        per_contact = pair_cost.amin(dim=-1)
        active = self._mask_tensor[step.candidate_ids]
        per_contact = torch.where(
            active,
            per_contact,
            torch.zeros_like(per_contact),
        )
        denom = active.to(step.qpos.dtype).sum(dim=-1).clamp_min(1.0)
        return per_contact.sum(dim=-1) / denom


@dataclass
class BatchedLMIKCost:
    """GPU-batched one-step LM reachability estimate for every rollout world."""

    damping: float = 1e-2
    residual_weight: float = 1.0
    step_weight: float = 1e-3
    joint_limit_weight: float = 10.0
    _identity: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, step: RolloutStep) -> Tensor:
        if (
            step.ee_jacobian is None
            or step.arm_qpos is None
            or step.arm_joint_lower is None
            or step.arm_joint_upper is None
        ):
            return torch.zeros(
                step.qpos.shape[0],
                device=step.qpos.device,
                dtype=step.qpos.dtype,
            )
        target_rotation = _quat_to_matrix(step.target_ee_pose[:, 3:])
        pose_error = torch.cat(
            (
                step.target_ee_pose[:, :3] - step.ee_position,
                _orientation_error(target_rotation, step.ee_rotation),
            ),
            dim=-1,
        )
        jacobian = step.ee_jacobian
        jt = jacobian.transpose(1, 2)
        if (
            self._identity is None
            or self._identity.device != jacobian.device
            or self._identity.dtype != jacobian.dtype
            or self._identity.shape[-1] != jacobian.shape[-1]
        ):
            self._identity = torch.eye(
                jacobian.shape[-1],
                device=jacobian.device,
                dtype=jacobian.dtype,
            ).unsqueeze(0)
        lhs = torch.bmm(jt, jacobian) + float(self.damping) * self._identity
        rhs = torch.bmm(jt, pose_error.unsqueeze(-1))
        delta_q = torch.linalg.solve(lhs, rhs).squeeze(-1)
        predicted_error = pose_error - torch.bmm(
            jacobian, delta_q.unsqueeze(-1)
        ).squeeze(-1)
        q_next = step.arm_qpos + delta_q
        lower_violation = torch.clamp(step.arm_joint_lower - q_next, min=0.0)
        upper_violation = torch.clamp(q_next - step.arm_joint_upper, min=0.0)
        return (
            float(self.residual_weight) * predicted_error.square().sum(dim=-1)
            + float(self.step_weight) * delta_q.square().sum(dim=-1)
            + float(self.joint_limit_weight)
            * (
                lower_violation.square().sum(dim=-1)
                + upper_violation.square().sum(dim=-1)
            )
        )


from robocasa.demos.cost import CompositeRolloutCost


@dataclass(frozen=True)
class EEMPCResult:
    """Result of one sampling-based MPC solve."""

    ee_pose_sequence: Tensor
    first_ee_pose: Tensor
    best_cost: float
    best_index: int
    best_iteration: int
    candidate_costs: Tensor
    iteration_best_costs: tuple[float, ...]
    iterations: int
    initial_pose_errors: Tensor | None = None
    initial_position_errors: Tensor | None = None
    initial_rotation_errors: Tensor | None = None
    candidate_ee_pose_sequences: Tensor | None = None
    arm_qpos_sequence: Tensor | None = None


@dataclass(frozen=True)
class EEMPCJob:
    """One independently sampled MPC environment assigned to one GPU."""

    name: str
    device: str
    config: DreamConfig
    cost_fn: TaskCost
    nominal_ee_poses: Tensor | np.ndarray | None = None
    initial_candidate_ee_poses: Tensor | np.ndarray | None = None
    terminal_cost_fn: TaskCost | None = None
    qpos_overrides: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiGPUEEMPCResult:
    job_names: tuple[str, ...]
    results: tuple[EEMPCResult, ...]
    best_job_index: int

    @property
    def best_result(self) -> EEMPCResult:
        return self.results[self.best_job_index]


def _normalize_quaternion(q: Tensor) -> Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    lw, lx, ly, lz = lhs.unbind(-1)
    rw, rx, ry, rz = rhs.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_inverse(q: Tensor) -> Tensor:
    q = _normalize_quaternion(q)
    result = q.clone()
    result[..., 1:].neg_()
    return result


def _rotvec_to_quat(rotvec: Tensor) -> Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1e-7,
        torch.sin(half) / angle,
        0.5 - angle.square() / 48.0,
    )
    return _normalize_quaternion(torch.cat((torch.cos(half), rotvec * scale), dim=-1))


def _quat_to_matrix(q: Tensor) -> Tensor:
    q = _normalize_quaternion(q)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _matrix_to_quat(matrix: Tensor) -> Tensor:
    """Branch-free rotation matrix to wxyz quaternion conversion."""

    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)),
        matrix[..., 2, 1] - matrix[..., 1, 2],
    )
    qy = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)),
        matrix[..., 0, 2] - matrix[..., 2, 0],
    )
    qz = 0.5 * torch.copysign(
        torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)),
        matrix[..., 1, 0] - matrix[..., 0, 1],
    )
    return _normalize_quaternion(torch.stack((qw, qx, qy, qz), dim=-1))


def _orientation_error(target: Tensor, current: Tensor) -> Tensor:
    """World-frame small-angle orientation error."""

    return 0.5 * (
        torch.linalg.cross(current[..., :, 0], target[..., :, 0])
        + torch.linalg.cross(current[..., :, 1], target[..., :, 1])
        + torch.linalg.cross(current[..., :, 2], target[..., :, 2])
    )


def _import_comfree_backend() -> tuple[Any, Any, Callable, Callable]:
    """Import the in-repository Comfree package without requiring installation."""

    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "comfree_warp"
    module = sys.modules.get("comfree_warp")
    if module is not None and not hasattr(module, "put_model"):
        # Running from the repository root can create a namespace package that
        # shadows comfree_warp/comfree_warp.
        del sys.modules["comfree_warp"]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    try:
        wp = importlib.import_module("warp")
        cfwarp = importlib.import_module("comfree_warp")
        core_forward = importlib.import_module("comfree_warp.comfree_core._src.forward")
        support = importlib.import_module("comfree_warp.mujoco_warp._src.support")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import the Comfree backend. This checkout declares "
            "warp-lang>=1.12 and mujoco==3.6.0 in comfree_warp/pyproject.toml."
        ) from exc
    return wp, cfwarp, core_forward.step_comfree, support.jac


def _resolve_site_id(model: mujoco.MjModel, site: str | int) -> int:
    if isinstance(site, int):
        site_id = site
    else:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    if site_id < 0 or site_id >= model.nsite:
        raise ValueError(f"Unknown end-effector site: {site!r}")
    return int(site_id)


def _infer_franka_joint_ids(
    model: mujoco.MjModel,
    ee_site_id: int,
    joint_names: Sequence[str] | None,
) -> list[int]:
    if joint_names is not None:
        ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in joint_names
        ]
        if any(joint_id < 0 for joint_id in ids):
            missing = [name for name, joint_id in zip(joint_names, ids) if joint_id < 0]
            raise ValueError(f"Unknown Franka joints: {missing}")
    else:
        ids = []
        body_id = int(model.site_bodyid[ee_site_id])
        while body_id > 0:
            joint_adr = int(model.body_jntadr[body_id])
            joint_num = int(model.body_jntnum[body_id])
            if joint_adr >= 0:
                ids.extend(range(joint_adr, joint_adr + joint_num))
            body_id = int(model.body_parentid[body_id])
        ids.reverse()
        ids = [
            joint_id
            for joint_id in ids
            if model.jnt_type[joint_id]
            in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ]
    if len(ids) != 7:
        raise ValueError(
            "Expected exactly 7 one-DoF joints on the Franka EE chain; "
            f"found {len(ids)}. Pass arm_joint_names explicitly."
        )
    return [int(joint_id) for joint_id in ids]


class ComfreeEEMPPI:
    """Parallel Comfree rollout and sampling-based EE-pose optimizer."""

    _STATE_FIELDS = (
        "qpos",
        "qvel",
        "qacc",
        "time",
        "ctrl",
        "act",
        "act_dot",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "eq_active",
        "userdata",
    )

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        ee_site: str | int,
        cost_fn: TaskCost,
        config: DreamConfig | None = None,
        terminal_cost_fn: TaskCost | None = None,
        pre_step_fn: Callable[[RolloutStep], None] | None = None,
        arm_joint_names: Sequence[str] | None = None,
    ) -> None:
        self.config = config or DreamConfig()
        self.model_cpu = model
        self.data_cpu = data
        self.cost_fn = cost_fn
        self.terminal_cost_fn = terminal_cost_fn
        self.pre_step_fn = pre_step_fn
        self.ee_site_id = _resolve_site_id(model, ee_site)
        self.ee_body_id = int(model.site_bodyid[self.ee_site_id])
        self.arm_joint_ids = _infer_franka_joint_ids(
            model, self.ee_site_id, arm_joint_names
        )
        self.arm_dof_ids = [int(model.jnt_dofadr[j]) for j in self.arm_joint_ids]
        self.arm_qpos_ids = [int(model.jnt_qposadr[j]) for j in self.arm_joint_ids]

        self.wp, self.cfwarp, self._step_comfree, self._jac = _import_comfree_backend()
        self.wp.init()
        self.wp.set_device(self.config.device)
        self.torch_device = torch.device(self.config.device)
        self.dtype = torch.float32
        self.generator = torch.Generator(device=self.torch_device)
        self.generator.manual_seed(self.config.seed)

        original_timestep = float(self.model_cpu.opt.timestep)
        original_cone = int(self.model_cpu.opt.cone)
        try:
            self.model_cpu.opt.timestep = float(self.config.sim_dt)
            # comfree_core currently implements contact constraints only for
            # pyramidal friction cones. The RoboCasa model may request an
            # elliptic cone, so convert only the model snapshot uploaded to
            # Comfree and restore the live MuJoCo model afterwards.
            self.model_cpu.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            mujoco.mj_forward(self.model_cpu, self.data_cpu)
            with self.wp.ScopedDevice(self.config.device):
                self.model = self.cfwarp.put_model(
                    self.model_cpu,
                    comfree_stiffness=self.config.contact_stiffness,
                    comfree_damping=self.config.contact_damping,
                )
                self.data = self.cfwarp.put_data(
                    self.model_cpu,
                    self.data_cpu,
                    nworld=self.config.num_worlds,
                    nconmax=self.config.nconmax_per_env,
                    njmax=self.config.njmax_per_env,
                )
                self._jacp = self.wp.zeros(
                    (self.config.num_worlds, 3, model.nv),
                    dtype=float,
                    device=self.config.device,
                )
                self._jacr = self.wp.zeros(
                    (self.config.num_worlds, 3, model.nv),
                    dtype=float,
                    device=self.config.device,
                )
                self._jac_points = self.wp.zeros(
                    self.config.num_worlds,
                    dtype=self.wp.vec3,
                    device=self.config.device,
                )
                self._jac_bodies = self.wp.array(
                    np.full(
                        self.config.num_worlds,
                        self.ee_body_id,
                        dtype=np.int32,
                    ),
                    dtype=int,
                    device=self.config.device,
                )
        finally:
            self.model_cpu.opt.timestep = original_timestep
            self.model_cpu.opt.cone = original_cone
            mujoco.mj_forward(self.model_cpu, self.data_cpu)

        self._candidate_ids = torch.arange(
            self.config.num_samples, device=self.torch_device, dtype=torch.long
        ).repeat_interleave(self.config.num_perturb_samples)
        self._replica_ids = torch.arange(
            self.config.num_perturb_samples,
            device=self.torch_device,
            dtype=torch.long,
        ).repeat(self.config.num_samples)
        self._arm_dof_tensor = torch.as_tensor(
            self.arm_dof_ids, device=self.torch_device, dtype=torch.long
        )
        self._arm_qpos_tensor = torch.as_tensor(
            self.arm_qpos_ids, device=self.torch_device, dtype=torch.long
        )
        joint_lower = []
        joint_upper = []
        for joint_id in self.arm_joint_ids:
            if bool(model.jnt_limited[joint_id]):
                joint_lower.append(float(model.jnt_range[joint_id, 0]))
                joint_upper.append(float(model.jnt_range[joint_id, 1]))
            else:
                joint_lower.append(-1.0e6)
                joint_upper.append(1.0e6)
        self._arm_joint_lower = torch.as_tensor(
            joint_lower, device=self.torch_device, dtype=self.dtype
        )
        self._arm_joint_upper = torch.as_tensor(
            joint_upper, device=self.torch_device, dtype=self.dtype
        )
        self._torque_limit = torch.as_tensor(
            self.config.max_joint_torque,
            device=self.torch_device,
            dtype=self.dtype,
        )
        self._arm_actuator_modes = self._find_arm_actuators()
        self._snapshot = self._save_state()
        self._last_ee_jacobian: Tensor | None = None
        self._last_initial_pose_errors: Tensor | None = None
        self._last_initial_position_errors: Tensor | None = None
        self._last_initial_rotation_errors: Tensor | None = None
        self._step_graph = self._compile_step_graph()
        self._restore_state()

    @property
    def num_worlds(self) -> int:
        return self.config.num_worlds

    def _find_arm_actuators(self) -> list[tuple[int, int, bool]]:
        result: list[tuple[int, int, bool]] = []
        for qpos_id, joint_id in zip(self.arm_qpos_ids, self.arm_joint_ids):
            matches = np.flatnonzero(
                (self.model_cpu.actuator_trnid[:, 0] == joint_id)
                & (self.model_cpu.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            )
            for actuator_id in matches:
                bias = self.model_cpu.actuator_biasprm[actuator_id]
                is_position = (
                    self.model_cpu.actuator_biastype[actuator_id]
                    == mujoco.mjtBias.mjBIAS_AFFINE
                    and bias[1] < 0.0
                )
                result.append((int(actuator_id), qpos_id, bool(is_position)))
        return result

    def _save_state(self) -> dict[str, Tensor]:
        return {
            name: self.wp.to_torch(getattr(self.data, name)).clone()
            for name in self._STATE_FIELDS
            if hasattr(self.data, name)
        }

    def _restore_state(self) -> None:
        with self.wp.ScopedDevice(self.config.device):
            for name, value in self._snapshot.items():
                self.wp.copy(getattr(self.data, name), self.wp.from_torch(value))
            self.cfwarp.forward(self.model, self.data)

    def sync_from_mujoco(self, data: mujoco.MjData | None = None) -> None:
        """Replace the rollout initial state with the supplied MuJoCo state."""

        source = data if data is not None else self.data_cpu
        n = self.config.num_worlds
        updates: Mapping[str, np.ndarray] = {
            "qpos": np.asarray(source.qpos, dtype=np.float32),
            "qvel": np.asarray(source.qvel, dtype=np.float32),
            "qacc": np.asarray(source.qacc, dtype=np.float32),
            "ctrl": np.asarray(source.ctrl, dtype=np.float32),
            "act": np.asarray(source.act, dtype=np.float32),
            "qacc_warmstart": np.asarray(source.qacc_warmstart, dtype=np.float32),
            "qfrc_applied": np.asarray(source.qfrc_applied, dtype=np.float32),
            "xfrc_applied": np.asarray(source.xfrc_applied, dtype=np.float32),
            "mocap_pos": np.asarray(source.mocap_pos, dtype=np.float32),
            "mocap_quat": np.asarray(source.mocap_quat, dtype=np.float32),
            "eq_active": np.asarray(source.eq_active, dtype=np.uint8),
            "userdata": np.asarray(source.userdata, dtype=np.float32),
        }
        for name, value in updates.items():
            if name not in self._snapshot:
                continue
            tensor = torch.as_tensor(
                value,
                device=self.torch_device,
                dtype=self._snapshot[name].dtype,
            )
            self._snapshot[name] = tensor.unsqueeze(0).repeat((n,) + (1,) * tensor.ndim)
        if "time" in self._snapshot:
            self._snapshot["time"] = torch.full_like(
                self._snapshot["time"], float(source.time)
            )
        if "act_dot" in self._snapshot:
            self._snapshot["act_dot"].zero_()
        self._restore_state()

    def _compile_step_graph(self) -> Any | None:
        if not (
            self.config.compile_cuda_graph
            and self.wp.get_device(self.config.device).is_cuda
        ):
            return None
        try:
            self._step_comfree(self.model, self.data)
            self._step_comfree(self.model, self.data)
            self.wp.synchronize()
            with self.wp.ScopedDevice(self.config.device):
                with self.wp.ScopedCapture() as capture:
                    self._step_comfree(self.model, self.data)
            self.wp.synchronize()
            return capture.graph
        except Exception as exc:
            # Graph capture is an optimization only. Restore the rollout state
            # and continue through the ordinary Comfree step path.
            sys.stderr.write(
                f"[dream] CUDA graph capture failed; falling back to per-step "
                f"launch (this will be 5-20x slower). Reason: {exc!r}\n"
            )
            sys.stderr.flush()
            self._restore_state()
            return None

    def _step_backend(self) -> None:
        if self._step_graph is None:
            self._step_comfree(self.model, self.data)
        else:
            self.wp.capture_launch(self._step_graph)

    def _set_rollout_initial_ee_poses(self, candidate_initial_poses: Tensor) -> None:
        """IK-project every parallel world onto its candidate initial EE pose."""

        targets = candidate_initial_poses.repeat_interleave(
            self.config.num_perturb_samples,
            dim=0,
        ).clone()
        targets[:, 3:] = _normalize_quaternion(targets[:, 3:])
        qpos = self.wp.to_torch(self.data.qpos)
        qvel = self.wp.to_torch(self.data.qvel)
        task_identity = torch.eye(
            6,
            device=self.torch_device,
            dtype=self.dtype,
        ).unsqueeze(0)
        errors = torch.full(
            (self.config.num_worlds,),
            torch.inf,
            device=self.torch_device,
            dtype=self.dtype,
        )
        for _ in range(max(int(self.config.initial_ik_iterations), 1)):
            ee_position = self.wp.to_torch(self.data.site_xpos)[:, self.ee_site_id]
            ee_rotation = self.wp.to_torch(self.data.site_xmat)[:, self.ee_site_id]
            target_rotation = _quat_to_matrix(targets[:, 3:])
            pose_error = torch.cat(
                (
                    targets[:, :3] - ee_position,
                    _orientation_error(target_rotation, ee_rotation),
                ),
                dim=-1,
            )
            errors = torch.linalg.vector_norm(pose_error, dim=-1)
            if bool(torch.all(errors <= float(self.config.initial_pose_tolerance))):
                break
            self.wp.copy(
                self._jac_points,
                self.wp.from_torch(ee_position.contiguous(), dtype=self.wp.vec3),
            )
            self._jac(
                self.model,
                self.data,
                self._jacp,
                self._jacr,
                self._jac_points,
                self._jac_bodies,
            )
            jacp = self.wp.to_torch(self._jacp)[:, :, self._arm_dof_tensor]
            jacr = self.wp.to_torch(self._jacr)[:, :, self._arm_dof_tensor]
            jacobian = torch.cat((jacp, jacr), dim=1)
            # Solve the damped task-space system J^T (J J^T + λI)^-1 e.
            # This avoids squaring the condition number of the redundant
            # 6x7 arm Jacobian, which caused most parallel IK projections to
            # stall even for nearby demonstration-derived poses.
            jt = jacobian.transpose(1, 2)
            lhs = torch.bmm(jacobian, jt) + (
                float(self.config.initial_ik_damping) * task_identity
            )
            task_step = torch.linalg.solve(lhs, pose_error.unsqueeze(-1))
            delta_q = torch.bmm(jt, task_step).squeeze(-1)
            max_step_norm = max(float(self.config.initial_ik_max_step_norm), 1e-6)
            step_norm = torch.linalg.vector_norm(
                delta_q, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            delta_q *= torch.clamp(max_step_norm / step_norm, max=1.0)
            active = errors > float(self.config.initial_pose_tolerance)
            delta_q = torch.where(active[:, None], delta_q, 0.0)
            arm_q = qpos[:, self._arm_qpos_tensor]
            arm_q = arm_q + float(self.config.initial_ik_step_scale) * delta_q
            arm_q = torch.maximum(
                torch.minimum(arm_q, self._arm_joint_upper),
                self._arm_joint_lower,
            )
            qpos[:, self._arm_qpos_tensor] = arm_q
            qvel[:, self._arm_dof_tensor] = 0.0
            self.cfwarp.forward(self.model, self.data)
        ee_position = self.wp.to_torch(self.data.site_xpos)[:, self.ee_site_id]
        ee_rotation = self.wp.to_torch(self.data.site_xmat)[:, self.ee_site_id]
        final_position_error = targets[:, :3] - ee_position
        final_rotation_error = _orientation_error(
            _quat_to_matrix(targets[:, 3:]), ee_rotation
        )
        final_pose_error = torch.cat(
            (
                final_position_error,
                final_rotation_error,
            ),
            dim=-1,
        )
        errors = torch.linalg.vector_norm(final_pose_error, dim=-1)
        position_errors = torch.linalg.vector_norm(final_position_error, dim=-1)
        rotation_errors = torch.linalg.vector_norm(final_rotation_error, dim=-1)
        self._last_initial_pose_errors = errors.view(
            self.config.num_samples,
            self.config.num_perturb_samples,
        ).amax(dim=1)
        self._last_initial_position_errors = position_errors.view(
            self.config.num_samples,
            self.config.num_perturb_samples,
        ).amax(dim=1)
        self._last_initial_rotation_errors = rotation_errors.view(
            self.config.num_samples,
            self.config.num_perturb_samples,
        ).amax(dim=1)

    def current_ee_pose(self) -> Tensor:
        """Current first-world EE pose as ``[x,y,z,qw,qx,qy,qz]``."""

        position = self.wp.to_torch(self.data.site_xpos)[0, self.ee_site_id]
        rotation = self.wp.to_torch(self.data.site_xmat)[0, self.ee_site_id]
        return torch.cat((position, _matrix_to_quat(rotation)), dim=-1)

    def _prepare_nominal(self, nominal: Tensor | np.ndarray | None) -> Tensor:
        h = self.config.horizon_steps
        if nominal is None:
            nominal_t = self.current_ee_pose().repeat(h, 1)
        else:
            nominal_t = torch.as_tensor(
                nominal, device=self.torch_device, dtype=self.dtype
            )
            if nominal_t.shape == (7,):
                nominal_t = nominal_t.repeat(h, 1)
            if nominal_t.ndim != 2 or nominal_t.shape[1] != 7:
                raise ValueError("nominal_ee_poses must have shape (H, 7) or (7,)")
            if nominal_t.shape[0] < h:
                nominal_t = torch.cat(
                    (
                        nominal_t,
                        nominal_t[-1:].repeat(h - nominal_t.shape[0], 1),
                    ),
                    dim=0,
                )
            elif nominal_t.shape[0] > h:
                nominal_t = nominal_t[:h]
        nominal_t = nominal_t.clone()
        nominal_t[:, 3:] = _normalize_quaternion(nominal_t[:, 3:])
        for index in range(1, nominal_t.shape[0]):
            if torch.dot(nominal_t[index - 1, 3:], nominal_t[index, 3:]) < 0:
                nominal_t[index, 3:].neg_()
        return nominal_t

    def sample_ee_poses(self, nominal: Tensor, noise_scale: float = 1.0) -> Tensor:
        """Sample ``(num_samples, horizon_steps, 7)`` absolute EE poses."""

        c = self.config
        n_knots = max(2, int(round(c.horizon / c.knot_dt)))
        ramp = torch.logspace(
            math.log10(c.first_ctrl_noise_scale),
            math.log10(c.last_ctrl_noise_scale),
            n_knots,
            device=self.torch_device,
            dtype=self.dtype,
        ).view(1, n_knots, 1)
        component_scale = torch.tensor(
            [c.pos_noise_scale] * 3 + [c.rot_noise_scale] * 3,
            device=self.torch_device,
            dtype=self.dtype,
        ).view(1, 1, 6)
        knot_noise = torch.randn(
            (c.num_samples, n_knots, 6),
            device=self.torch_device,
            dtype=self.dtype,
            generator=self.generator,
        )
        knot_noise *= ramp * component_scale * float(noise_scale)
        if c.zero_first_knot_noise and not c.optimize_initial_pose:
            knot_noise[:, 0].zero_()
        knot_noise[0].zero_()
        exploit = int(c.num_samples * c.exploit_ratio)
        if exploit > 0:
            knot_noise[-exploit:] *= c.exploit_noise_scale
        noise = F.interpolate(
            knot_noise.transpose(1, 2),
            size=c.horizon_steps,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        sampled = nominal.unsqueeze(0).repeat(c.num_samples, 1, 1)
        sampled[..., :3] += noise[..., :3]
        sampled[..., 3:] = _quat_mul(_rotvec_to_quat(noise[..., 3:]), sampled[..., 3:])
        sampled[..., 3:] = _normalize_quaternion(sampled[..., 3:])
        return sampled

    def _neutralize_actuators(self) -> None:
        if not self.config.neutralize_arm_actuators:
            return
        ctrl = self.wp.to_torch(self.data.ctrl)
        qpos = self.wp.to_torch(self.data.qpos)
        for actuator_id, qpos_id, is_position in self._arm_actuator_modes:
            ctrl[:, actuator_id] = qpos[:, qpos_id] if is_position else 0.0

    def _apply_task_space_pd(self, target_pose: Tensor) -> RolloutStep:
        ee_position = self.wp.to_torch(self.data.site_xpos)[:, self.ee_site_id]
        ee_rotation = self.wp.to_torch(self.data.site_xmat)[:, self.ee_site_id]
        self.wp.copy(
            self._jac_points,
            self.wp.from_torch(ee_position.contiguous(), dtype=self.wp.vec3),
        )
        self._jac(
            self.model,
            self.data,
            self._jacp,
            self._jacr,
            self._jac_points,
            self._jac_bodies,
        )
        jacp = self.wp.to_torch(self._jacp)[:, :, self._arm_dof_tensor]
        jacr = self.wp.to_torch(self._jacr)[:, :, self._arm_dof_tensor]
        jacobian = torch.cat((jacp, jacr), dim=1)
        self._last_ee_jacobian = jacobian
        qvel = self.wp.to_torch(self.data.qvel)
        arm_qvel = qvel[:, self._arm_dof_tensor]
        ee_velocity = torch.bmm(jacobian, arm_qvel.unsqueeze(-1)).squeeze(-1)

        target_rotation = _quat_to_matrix(target_pose[:, 3:])
        position_error = target_pose[:, :3] - ee_position
        rotation_error = _orientation_error(target_rotation, ee_rotation)
        wrench = torch.cat(
            (
                self.config.position_kp * position_error
                - self.config.position_kd * ee_velocity[:, :3],
                self.config.rotation_kp * rotation_error
                - self.config.rotation_kd * ee_velocity[:, 3:],
            ),
            dim=-1,
        )
        torque = torch.bmm(jacobian.transpose(1, 2), wrench.unsqueeze(-1)).squeeze(-1)
        torque -= self.config.joint_damping * arm_qvel
        torque = torch.clamp(torque, -self._torque_limit, self._torque_limit)

        qfrc = self.wp.to_torch(self.data.qfrc_applied)
        qfrc[:] = self._snapshot["qfrc_applied"]
        qfrc[:, self._arm_dof_tensor] += torque
        if "xfrc_applied" in self._snapshot:
            self.wp.to_torch(self.data.xfrc_applied)[:] = self._snapshot["xfrc_applied"]
        self._neutralize_actuators()
        return RolloutStep(
            step_index=-1,
            is_terminal=False,
            target_ee_pose=target_pose,
            ee_position=ee_position,
            ee_rotation=ee_rotation,
            qpos=self.wp.to_torch(self.data.qpos),
            qvel=qvel,
            data=self.data,
            model=self.model,
            candidate_ids=self._candidate_ids,
            replica_ids=self._replica_ids,
            ee_jacobian=jacobian,
            arm_qpos=self.wp.to_torch(self.data.qpos)[:, self._arm_qpos_tensor],
            arm_joint_lower=self._arm_joint_lower,
            arm_joint_upper=self._arm_joint_upper,
        )

    def _make_step_context(
        self, step_index: int, target_pose: Tensor, is_terminal: bool
    ) -> RolloutStep:
        return RolloutStep(
            step_index=step_index,
            is_terminal=is_terminal,
            target_ee_pose=target_pose,
            ee_position=self.wp.to_torch(self.data.site_xpos)[:, self.ee_site_id],
            ee_rotation=self.wp.to_torch(self.data.site_xmat)[:, self.ee_site_id],
            qpos=self.wp.to_torch(self.data.qpos),
            qvel=self.wp.to_torch(self.data.qvel),
            data=self.data,
            model=self.model,
            candidate_ids=self._candidate_ids,
            replica_ids=self._replica_ids,
            ee_jacobian=self._last_ee_jacobian,
            arm_qpos=self.wp.to_torch(self.data.qpos)[:, self._arm_qpos_tensor],
            arm_joint_lower=self._arm_joint_lower,
            arm_joint_upper=self._arm_joint_upper,
        )

    def _coerce_cost(self, value: Tensor | float) -> Tensor:
        cost = torch.as_tensor(
            value, device=self.torch_device, dtype=self.dtype
        ).reshape(-1)
        if cost.numel() == 1:
            cost = cost.expand(self.config.num_worlds)
        elif cost.numel() == self.config.num_samples:
            cost = cost.repeat_interleave(self.config.num_perturb_samples)
        elif cost.numel() != self.config.num_worlds:
            raise ValueError(
                "cost_fn must return a scalar, num_samples values, or "
                "num_samples*num_perturb_samples values"
            )
        return torch.nan_to_num(cost, nan=torch.inf, posinf=torch.inf, neginf=torch.inf)

    @torch.no_grad()
    def rollout(self, candidate_ee_poses: Tensor) -> Tensor:
        """Roll out candidates and return summed (or configured mean) cost."""

        candidate_ee_poses = torch.as_tensor(
            candidate_ee_poses, device=self.torch_device, dtype=self.dtype
        )
        expected = (
            self.config.num_samples,
            self.config.horizon_steps,
            7,
        )
        if tuple(candidate_ee_poses.shape) != expected:
            raise ValueError(
                f"candidate_ee_poses must have shape {expected}, got "
                f"{tuple(candidate_ee_poses.shape)}"
            )
        candidate_ee_poses = candidate_ee_poses.clone()
        candidate_ee_poses[..., 3:] = _normalize_quaternion(candidate_ee_poses[..., 3:])
        actions = candidate_ee_poses.repeat_interleave(
            self.config.num_perturb_samples, dim=0
        )
        self._restore_state()
        self._set_rollout_initial_ee_poses(candidate_ee_poses[:, 0])
        # The previous rollout's terminal Jacobian is invalid after restoring
        # and IK-projecting a new candidate batch.
        self._last_ee_jacobian = None
        cumulative = torch.zeros(
            self.config.num_worlds,
            device=self.torch_device,
            dtype=self.dtype,
        )
        if self._last_initial_pose_errors is not None:
            initial_errors = self._last_initial_pose_errors.repeat_interleave(
                self.config.num_perturb_samples
            )
            cumulative += (
                float(self.config.initial_pose_error_weight) * initial_errors.square()
            )
        initial_context = self._make_step_context(
            0,
            actions[:, 0],
            self.config.horizon_steps == 1,
        )
        cumulative += self._coerce_cost(self.cost_fn(initial_context))
        if self.config.horizon_steps == 1 and self.terminal_cost_fn is not None:
            cumulative += self._coerce_cost(self.terminal_cost_fn(initial_context))
        for step_index in range(1, self.config.horizon_steps):
            target_pose = actions[:, step_index]
            pre_context = self._apply_task_space_pd(target_pose)
            if self.pre_step_fn is not None:
                self.pre_step_fn(replace(pre_context, step_index=step_index))
            self._step_backend()
            terminal = step_index == self.config.horizon_steps - 1
            context = self._make_step_context(step_index, target_pose, terminal)
            cumulative += self._coerce_cost(self.cost_fn(context))
            if terminal and self.terminal_cost_fn is not None:
                cumulative += self._coerce_cost(self.terminal_cost_fn(context))
        if self.config.cost_aggregation == "mean":
            cumulative /= float(self.config.horizon_steps)
        return cumulative.view(
            self.config.num_samples, self.config.num_perturb_samples
        ).mean(dim=1)

    @torch.no_grad()
    def rollout_and_select(self, candidate_ee_poses: Tensor) -> EEMPCResult:
        """Roll out a supplied candidate batch and return its exact argmin."""

        candidates = torch.as_tensor(
            candidate_ee_poses, device=self.torch_device, dtype=self.dtype
        ).clone()
        candidates[..., 3:] = _normalize_quaternion(candidates[..., 3:])
        costs = self.rollout(candidates)
        best_cost, best_index = torch.min(costs, dim=0)
        best_index_int = int(best_index.detach().cpu())
        best = candidates[best_index_int].clone()
        best_arm_qpos = self.trace_arm_qpos_sequence(best)
        return EEMPCResult(
            ee_pose_sequence=best,
            first_ee_pose=best[0],
            best_cost=float(best_cost.detach().cpu()),
            best_index=best_index_int,
            best_iteration=0,
            candidate_costs=costs,
            iteration_best_costs=(float(best_cost.detach().cpu()),),
            iterations=1,
            initial_pose_errors=(
                None
                if self._last_initial_pose_errors is None
                else self._last_initial_pose_errors.clone()
            ),
            initial_position_errors=(
                None
                if self._last_initial_position_errors is None
                else self._last_initial_position_errors.clone()
            ),
            initial_rotation_errors=(
                None
                if self._last_initial_rotation_errors is None
                else self._last_initial_rotation_errors.clone()
            ),
            candidate_ee_pose_sequences=candidates.clone(),
            arm_qpos_sequence=best_arm_qpos,
        )

    @torch.no_grad()
    def trace_arm_qpos_sequence(self, ee_pose_sequence: Tensor | np.ndarray) -> Tensor:
        """Replay one EE-pose sequence and return the resulting arm qpos trace.

        The trace is produced by the same Comfree task-space PD controller used
        during MPPI rollouts, so consecutive rows are the integrated joint
        targets that the MuJoCo viewer can track with its joint-space PD loop.
        """

        sequence = torch.as_tensor(
            ee_pose_sequence, device=self.torch_device, dtype=self.dtype
        ).clone()
        expected = (self.config.horizon_steps, 7)
        if tuple(sequence.shape) != expected:
            raise ValueError(
                f"ee_pose_sequence must have shape {expected}, got {tuple(sequence.shape)}"
            )
        sequence[..., 3:] = _normalize_quaternion(sequence[..., 3:])
        candidates = (
            sequence.unsqueeze(0).expand(self.config.num_samples, -1, -1).clone()
        )
        actions = candidates.repeat_interleave(self.config.num_perturb_samples, dim=0)
        self._restore_state()
        self._set_rollout_initial_ee_poses(candidates[:, 0])
        self._last_ee_jacobian = None
        q_trace = [self.wp.to_torch(self.data.qpos)[0, self._arm_qpos_tensor].clone()]
        for step_index in range(1, self.config.horizon_steps):
            target_pose = actions[:, step_index]
            pre_context = self._apply_task_space_pd(target_pose)
            if self.pre_step_fn is not None:
                self.pre_step_fn(replace(pre_context, step_index=step_index))
            self._step_backend()
            q_trace.append(
                self.wp.to_torch(self.data.qpos)[0, self._arm_qpos_tensor].clone()
            )
        return torch.stack(q_trace, dim=0)

    def _weighted_pose_mean(self, candidates: Tensor, costs: Tensor) -> Tensor:
        count = max(1, int(math.ceil(self.config.elite_ratio * costs.numel())))
        elite_cost, elite_id = torch.topk(costs, k=count, largest=False)
        safe = torch.isfinite(elite_cost)
        if not bool(safe.any()):
            return candidates[0]
        elite_cost = elite_cost[safe]
        elite = candidates[elite_id[safe]]
        normalized = (elite_cost - elite_cost.mean()) / (
            elite_cost.std(unbiased=False) + 1e-2
        )
        weight = torch.softmax(-normalized / max(self.config.temperature, 1e-6), dim=0)
        position = (weight[:, None, None] * elite[..., :3]).sum(dim=0)
        reference = elite[0, :, 3:]
        quaternion = elite[..., 3:]
        sign = torch.where(
            (quaternion * reference.unsqueeze(0)).sum(dim=-1, keepdim=True) < 0,
            -1.0,
            1.0,
        )
        quaternion = _normalize_quaternion(
            (weight[:, None, None] * quaternion * sign).sum(dim=0)
        )
        return torch.cat((position, quaternion), dim=-1)

    @torch.no_grad()
    def solve(
        self,
        nominal_ee_poses: Tensor | np.ndarray | None = None,
        initial_candidate_ee_poses: Tensor | np.ndarray | None = None,
    ) -> EEMPCResult:
        """Optimize an EE-pose sequence and return the lowest-cost rollout."""

        nominal = self._prepare_nominal(nominal_ee_poses)
        initial_candidates = None
        if initial_candidate_ee_poses is not None:
            initial_candidates = torch.as_tensor(
                initial_candidate_ee_poses,
                device=self.torch_device,
                dtype=self.dtype,
            )
            if initial_candidates.shape == (self.config.num_samples, 7):
                initial_candidates = initial_candidates.clone()
                initial_candidates[:, 3:] = _normalize_quaternion(
                    initial_candidates[:, 3:]
                )
            elif initial_candidates.shape == (
                self.config.num_samples,
                self.config.horizon_steps,
                7,
            ):
                initial_candidates = initial_candidates.clone()
                initial_candidates[..., 3:] = _normalize_quaternion(
                    initial_candidates[..., 3:]
                )
            else:
                raise ValueError(
                    "initial_candidate_ee_poses must have shape "
                    f"({self.config.num_samples}, 7) or "
                    f"({self.config.num_samples}, {self.config.horizon_steps}, 7)"
                )
        global_best_cost = float("inf")
        global_best_index = 0
        global_best_iteration = 0
        global_best = nominal.clone()
        global_best_costs = torch.full(
            (self.config.num_samples,),
            torch.inf,
            device=self.torch_device,
            dtype=self.dtype,
        )
        global_best_initial_pose_errors = None
        global_best_initial_position_errors = None
        global_best_initial_rotation_errors = None
        global_best_candidates = None
        final_costs = torch.full(
            (self.config.num_samples,),
            torch.inf,
            device=self.torch_device,
            dtype=self.dtype,
        )
        history: list[float] = []
        improvements: list[float] = []
        beta = self.config.final_noise_scale ** (
            1.0 / max(self.config.max_num_iterations, 1)
        )

        for iteration in range(self.config.max_num_iterations):
            candidates = self.sample_ee_poses(nominal, beta**iteration)
            if iteration == 0 and initial_candidates is not None:
                if initial_candidates.ndim == 3:
                    candidates = initial_candidates.clone()
                else:
                    position_delta = initial_candidates[:, :3] - candidates[:, 0, :3]
                    candidates[..., :3] += position_delta[:, None, :]
                    rotation_delta = _quat_mul(
                        initial_candidates[:, 3:],
                        _quat_inverse(candidates[:, 0, 3:]),
                    )
                    candidates[..., 3:] = _normalize_quaternion(
                        _quat_mul(
                            rotation_delta[:, None, :].expand(
                                -1,
                                candidates.shape[1],
                                -1,
                            ),
                            candidates[..., 3:],
                        )
                    )
                    candidates[:, 0] = initial_candidates
            costs = self.rollout(candidates)
            final_costs = costs
            best_cost_tensor, best_index_tensor = torch.min(costs, dim=0)
            best_cost = float(best_cost_tensor.detach().cpu())
            best_index = int(best_index_tensor.detach().cpu())
            previous = global_best_cost
            if best_cost < global_best_cost:
                global_best_cost = best_cost
                global_best_index = best_index
                global_best_iteration = iteration
                global_best = candidates[best_index].clone()
                global_best_costs = costs.clone()
                global_best_initial_pose_errors = (
                    None
                    if self._last_initial_pose_errors is None
                    else self._last_initial_pose_errors.clone()
                )
                global_best_initial_position_errors = (
                    None
                    if self._last_initial_position_errors is None
                    else self._last_initial_position_errors.clone()
                )
                global_best_initial_rotation_errors = (
                    None
                    if self._last_initial_rotation_errors is None
                    else self._last_initial_rotation_errors.clone()
                )
                global_best_candidates = candidates.clone()
            history.append(best_cost)
            improvements.append(
                float("inf")
                if not math.isfinite(previous)
                else previous - global_best_cost
            )
            nominal = self._weighted_pose_mean(candidates, costs)

            check = self.config.improvement_check_steps
            if check > 0 and len(improvements) >= check:
                recent = improvements[-check:]
                if all(
                    math.isfinite(value) and value < self.config.improvement_threshold
                    for value in recent
                ):
                    break

        return EEMPCResult(
            ee_pose_sequence=global_best,
            first_ee_pose=global_best[0],
            best_cost=global_best_cost,
            best_index=global_best_index,
            best_iteration=global_best_iteration,
            candidate_costs=(
                global_best_costs if math.isfinite(global_best_cost) else final_costs
            ),
            iteration_best_costs=tuple(history),
            iterations=len(history),
            initial_pose_errors=global_best_initial_pose_errors,
            initial_position_errors=global_best_initial_position_errors,
            initial_rotation_errors=global_best_initial_rotation_errors,
            candidate_ee_pose_sequences=global_best_candidates,
            arm_qpos_sequence=(
                self.trace_arm_qpos_sequence(global_best)
                if math.isfinite(global_best_cost)
                else None
            ),
        )

    def shift_nominal(self, sequence: Tensor, steps: int = 1) -> Tensor:
        """Receding-horizon shift, padding the tail with the final pose."""

        steps = min(max(int(steps), 0), sequence.shape[0])
        if steps == 0:
            return sequence.clone()
        return torch.cat((sequence[steps:], sequence[-1:].repeat(steps, 1)), dim=0)


def solve_ee_pose_mpc(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ee_site: str | int,
    cost_fn: TaskCost,
    nominal_ee_poses: Tensor | np.ndarray | None = None,
    initial_candidate_ee_poses: Tensor | np.ndarray | None = None,
    config: DreamConfig | None = None,
    terminal_cost_fn: TaskCost | None = None,
    pre_step_fn: Callable[[RolloutStep], None] | None = None,
    arm_joint_names: Sequence[str] | None = None,
) -> EEMPCResult:
    """Build a Comfree EE-pose solver, roll out candidates, and take the best."""

    solver = ComfreeEEMPPI(
        model,
        data,
        ee_site=ee_site,
        cost_fn=cost_fn,
        config=config,
        terminal_cost_fn=terminal_cost_fn,
        pre_step_fn=pre_step_fn,
        arm_joint_names=arm_joint_names,
    )
    return solver.solve(
        nominal_ee_poses,
        initial_candidate_ee_poses=initial_candidate_ee_poses,
    )


def _copy_mujoco_data(source: mujoco.MjData, target: mujoco.MjData) -> None:
    for name in (
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "eq_active",
        "userdata",
    ):
        if not hasattr(source, name) or not hasattr(target, name):
            continue
        source_value = np.asarray(getattr(source, name))
        target_value = getattr(target, name)
        if np.asarray(target_value).shape == source_value.shape:
            target_value[...] = source_value
    target.time = float(source.time)


def solve_ee_pose_mpc_multi_gpu(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ee_site: str | int,
    jobs: Sequence[EEMPCJob],
    arm_joint_names: Sequence[str] | None = None,
) -> MultiGPUEEMPCResult:
    """Run independent MPC environments concurrently, one worker per GPU.

    Jobs assigned to the same device run sequentially so two gripper modes do
    not compete for the same GPU memory. Different devices execute in parallel.
    Each worker loads its own binary MuJoCo model and applies its gripper-state
    qpos overrides before uploading the environment to Comfree Warp.
    """

    jobs = tuple(jobs)
    if not jobs:
        raise ValueError("At least one EEMPCJob is required")
    handle = tempfile.NamedTemporaryFile(suffix=".mjb", delete=False)
    model_path = handle.name
    handle.close()
    mujoco.mj_saveModel(model, model_path, None)
    grouped: dict[str, list[tuple[int, EEMPCJob]]] = {}
    for index, job in enumerate(jobs):
        grouped.setdefault(str(job.device), []).append((index, job))
    results: list[EEMPCResult | None] = [None] * len(jobs)
    warp = importlib.import_module("warp")
    warp.init()
    for device in grouped:
        warp.get_device(device)

    def run_device(device_jobs: list[tuple[int, EEMPCJob]]):
        device_results = []
        for index, job in device_jobs:
            local_model = mujoco.MjModel.from_binary_path(model_path)
            local_data = mujoco.MjData(local_model)
            _copy_mujoco_data(data, local_data)
            for qpos_id, value in job.qpos_overrides.items():
                local_data.qpos[int(qpos_id)] = float(value)
            mujoco.mj_forward(local_model, local_data)
            config = replace(job.config, device=str(job.device))
            solver = ComfreeEEMPPI(
                local_model,
                local_data,
                ee_site=ee_site,
                cost_fn=job.cost_fn,
                terminal_cost_fn=job.terminal_cost_fn,
                config=config,
                arm_joint_names=arm_joint_names,
            )
            result = solver.solve(
                job.nominal_ee_poses,
                initial_candidate_ee_poses=job.initial_candidate_ee_poses,
            )
            device_results.append((index, result))
        return device_results

    try:
        if len(grouped) == 1:
            completed_groups = [run_device(next(iter(grouped.values())))]
        else:
            with ThreadPoolExecutor(max_workers=len(grouped)) as executor:
                futures = [
                    executor.submit(run_device, device_jobs)
                    for device_jobs in grouped.values()
                ]
                completed_groups = [future.result() for future in futures]
        for completed_group in completed_groups:
            for index, result in completed_group:
                results[index] = result
    finally:
        try:
            os.unlink(model_path)
        except OSError:
            pass
    if any(result is None for result in results):
        raise RuntimeError("One or more multi-GPU MPC jobs did not return a result")
    completed = tuple(result for result in results if result is not None)
    best_job_index = min(
        range(len(completed)),
        key=lambda index: float(completed[index].best_cost),
    )
    return MultiGPUEEMPCResult(
        job_names=tuple(job.name for job in jobs),
        results=completed,
        best_job_index=int(best_job_index),
    )


@torch.no_grad()
def score_feasible_contact_initial_poses(
    initial_ee_poses: Tensor | np.ndarray,
    *,
    ee_contact_point_local: Tensor | np.ndarray,
    feasible_points_world: Tensor | np.ndarray,
    max_distance: float,
    device: str = "cuda:0",
) -> dict[str, Tensor]:
    """Fast batched check that demo-derived EE contact points hit feasible points.

    This is intentionally geometry-only: penetration is allowed at this stage.
    The returned tensors live on CPU so callers can store them directly.
    """

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    poses = torch.as_tensor(
        initial_ee_poses, device=torch_device, dtype=torch.float32
    ).reshape(-1, 7)
    point_local = torch.as_tensor(
        ee_contact_point_local, device=torch_device, dtype=torch.float32
    ).reshape(1, 3)
    feasible = torch.as_tensor(
        feasible_points_world, device=torch_device, dtype=torch.float32
    ).reshape(-1, 3)
    if feasible.shape[0] == 0:
        nearest = torch.full((poses.shape[0],), torch.inf, device=torch_device)
        contact = torch.full((poses.shape[0], 3), torch.nan, device=torch_device)
        passed = torch.zeros((poses.shape[0],), dtype=torch.bool, device=torch_device)
    else:
        rotation = _quat_to_matrix(poses[:, 3:])
        contact = poses[:, :3] + torch.bmm(
            rotation, point_local.expand(poses.shape[0], -1).unsqueeze(-1)
        ).squeeze(-1)
        distances = torch.cdist(contact, feasible)
        nearest, _ = torch.min(distances, dim=1)
        passed = nearest <= float(max_distance)
    return {
        "contact_points_world": contact.detach().cpu(),
        "nearest_feasible_distances": nearest.detach().cpu(),
        "feasible_mask": passed.detach().cpu(),
        "feasible_fraction": passed.float().mean().detach().cpu(),
    }


# ---------------------------------------------------------------------------
# Lightweight compatibility model used by demo_open_drawer_contact_curobo.py.
# The real MuJoCo/Comfree path is ComfreeEEMPPI above.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenDrawerDreamState:
    ee_pos: Tensor
    ee_vel: Tensor
    drawer_open_distance: Tensor
    drawer_velocity: Tensor


@dataclass(frozen=True)
class OpenDrawerDreamRollout:
    ee_pos: Tensor
    ee_vel: Tensor
    drawer_open_distance: Tensor
    drawer_velocity: Tensor
    contact_force: Tensor


class OpenDrawerDreamModel:
    """Small batched drawer surrogate retained for existing contact scoring."""

    def __init__(
        self,
        *,
        start_drawer_q: float,
        contact_world: np.ndarray,
        pull_world: np.ndarray,
        approach_world: np.ndarray,
        contact_rest_offset: float,
        config: DreamConfig,
        device: str = "cuda:0",
    ) -> None:
        self.config = config
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.start_drawer_q = float(start_drawer_q)
        self.contact_world = torch.as_tensor(
            contact_world, device=self.device, dtype=self.dtype
        )
        self.pull_world = F.normalize(
            torch.as_tensor(pull_world, device=self.device, dtype=self.dtype),
            dim=0,
        )
        self.approach_world = F.normalize(
            torch.as_tensor(approach_world, device=self.device, dtype=self.dtype),
            dim=0,
        )
        self.contact_rest_offset = float(contact_rest_offset)

    def initial_state(
        self, *, num_envs: int, ee_pos: np.ndarray | Tensor
    ) -> OpenDrawerDreamState:
        position = (
            torch.as_tensor(ee_pos, device=self.device, dtype=self.dtype)
            .reshape(1, 3)
            .repeat(num_envs, 1)
        )
        zeros = torch.zeros(num_envs, device=self.device, dtype=self.dtype)
        return OpenDrawerDreamState(
            ee_pos=position,
            ee_vel=torch.zeros_like(position),
            drawer_open_distance=zeros.clone(),
            drawer_velocity=zeros.clone(),
        )

    @torch.no_grad()
    def rollout(
        self,
        actions: Tensor,
        *,
        state: OpenDrawerDreamState,
    ) -> OpenDrawerDreamRollout:
        if actions.ndim != 3 or actions.shape[-1] not in (3, 7):
            raise ValueError("actions must have shape (N, H, 3) or (N, H, 7)")
        dt = self.config.sim_dt
        ee_pos = state.ee_pos.clone()
        ee_vel = state.ee_vel.clone()
        drawer = state.drawer_open_distance.clone()
        drawer_vel = state.drawer_velocity.clone()
        pos_trace, vel_trace, drawer_trace, drawer_vel_trace, force_trace = (
            [],
            [],
            [],
            [],
            [],
        )
        for index in range(actions.shape[1]):
            action = actions[:, index]
            if self.config.control_mode == "velocity" or action.shape[-1] == 3:
                ee_vel = torch.clamp(
                    action[:, :3],
                    -self.config.ee_velocity_limit,
                    self.config.ee_velocity_limit,
                )
                ee_pos = ee_pos + dt * ee_vel
            else:
                target = action[:, :3]
                ee_vel = torch.clamp(
                    (target - ee_pos) / dt,
                    -self.config.ee_velocity_limit,
                    self.config.ee_velocity_limit,
                )
                ee_pos = target

            relative = ee_pos - self.contact_world
            normal_error = (relative * self.approach_world).sum(
                dim=-1
            ) - self.contact_rest_offset
            lateral = (
                relative
                - (relative * self.pull_world).sum(dim=-1, keepdim=True)
                * self.pull_world
            )
            contact = (normal_error.abs() <= self.config.contact_radius) & (
                torch.linalg.vector_norm(lateral, dim=-1)
                <= 2.0 * self.config.contact_radius
            )
            ee_pull = (relative * self.pull_world).sum(dim=-1)
            extension = torch.clamp(ee_pull - drawer, min=0.0)
            force = (
                self.config.contact_stiffness * extension
                - self.config.contact_damping * drawer_vel
            )
            force = torch.where(contact, force, torch.zeros_like(force))
            force = torch.clamp(force, min=0.0, max=self.config.max_contact_force)
            drawer_acc = force - 2.0 * drawer_vel
            drawer_vel = drawer_vel + dt * drawer_acc
            drawer = torch.clamp(
                drawer + dt * drawer_vel,
                min=0.0,
                max=self.config.max_open_distance,
            )
            pos_trace.append(ee_pos.clone())
            vel_trace.append(ee_vel.clone())
            drawer_trace.append(drawer.clone())
            drawer_vel_trace.append(drawer_vel.clone())
            force_trace.append(force)

        return OpenDrawerDreamRollout(
            ee_pos=torch.stack(pos_trace, dim=1),
            ee_vel=torch.stack(vel_trace, dim=1),
            drawer_open_distance=torch.stack(drawer_trace, dim=1),
            drawer_velocity=torch.stack(drawer_vel_trace, dim=1),
            contact_force=torch.stack(force_trace, dim=1),
        )

    def terminal_opening_cost(
        self,
        rollout: OpenDrawerDreamRollout,
        *,
        target_open_distance: float,
    ) -> Tensor:
        error = rollout.drawer_open_distance[:, -1] - float(target_open_distance)
        force_penalty = 1e-5 * rollout.contact_force.square().mean(dim=1)
        return error.square() + force_penalty

    def trajectory_task_cost(
        self,
        rollout: OpenDrawerDreamRollout,
        *,
        target_open_distance: float,
    ) -> Tensor:
        """Sum squared object-frame drawer-position error over rollout steps."""

        error = rollout.drawer_open_distance - float(target_open_distance)
        force_penalty = 1e-5 * rollout.contact_force.square()
        return (error.square() + force_penalty).sum(dim=1)


# ---------------------------------------------------------------------------
# q-space sampling optimizer for collision-sphere contact placement.
# ---------------------------------------------------------------------------


@dataclass
class QConfigOptimizerConfig:
    """Sampling-based static q-config optimizer settings."""

    device: str = "cuda:0"
    seed: int = 0
    num_samples: int = 1024
    max_num_iterations: int = 16
    elite_ratio: float = 0.1
    temperature: float = 1.0
    arm_noise_scale: float = 0.15
    gripper_noise_scale: float = 0.02
    final_noise_scale: float = 0.02
    improvement_threshold: float = 1e-4
    improvement_check_steps: int = 2
    contact_weight: float = 200.0
    penetration_weight: float = 400.0
    penetration_margin: float = 0.002
    regularization_weight: float = 1.0
    joint_limit_weight: float = 100.0
    nconmax_per_env: int = 120
    njmax_per_env: int = 500


@dataclass(frozen=True)
class QConfigResult:
    """Result of a static q-config sampling optimization."""

    best_q: Tensor
    best_cost: float
    best_index: int
    best_iteration: int
    candidate_q: Tensor
    candidate_costs: Tensor
    iteration_best_costs: tuple[float, ...]
    iterations: int
    best_sphere_centers_world: Tensor
    best_contact_distances: Tensor
    best_min_penetration_distance: float
    best_feasible_contact_count: int = 0
    successful_config_count: int = 0


class ComfreeQConfigOptimizer:
    """Sample arm+gripper q around a seed and score collision-sphere contacts.

    Single-step (no rollout): for each sampled q, run Comfree's parallel forward
    kinematics to obtain the EE site pose, transform the EE-frame collision
    spheres to world coordinates, and score:

    - Contact: sum of squared distances between each (sphere_index, target_point)
      pair specified in ``sphere_target_pairs``.
    - Non-penetration: hinge-penalty over every sphere vs every supplied object
      surface point ``object_points_world`` (radius - distance, with margin).
    - Joint-limit hinge penalty and quadratic regularization around the seed q.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        ee_site: str | int,
        arm_joint_names: Sequence[str],
        gripper_joint_names: Sequence[str] = (),
        sphere_centers_ee: np.ndarray | Tensor,
        sphere_radii: np.ndarray | Tensor,
        config: QConfigOptimizerConfig | None = None,
    ) -> None:
        self.config = config or QConfigOptimizerConfig()
        self.model_cpu = model
        self.data_cpu = data
        self.ee_site_id = _resolve_site_id(model, ee_site)

        def _resolve_joints(names: Sequence[str]) -> tuple[list[int], list[int]]:
            joint_ids: list[int] = []
            qpos_ids: list[int] = []
            for name in names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"Unknown joint: {name!r}")
                joint_ids.append(int(jid))
                qpos_ids.append(int(model.jnt_qposadr[jid]))
            return joint_ids, qpos_ids

        self.arm_joint_ids, self.arm_qpos_ids = _resolve_joints(arm_joint_names)
        self.gripper_joint_ids, self.gripper_qpos_ids = _resolve_joints(
            gripper_joint_names
        )
        self.num_arm = len(self.arm_qpos_ids)
        self.num_gripper = len(self.gripper_qpos_ids)
        self.num_q = self.num_arm + self.num_gripper
        if self.num_q == 0:
            raise ValueError("ComfreeQConfigOptimizer requires at least one joint")

        self.wp, self.cfwarp, _, _ = _import_comfree_backend()
        self.wp.init()
        self.wp.set_device(self.config.device)
        self.torch_device = torch.device(self.config.device)
        self.dtype = torch.float32
        self.generator = torch.Generator(device=self.torch_device)
        self.generator.manual_seed(int(self.config.seed))

        original_cone = int(self.model_cpu.opt.cone)
        try:
            self.model_cpu.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            mujoco.mj_forward(self.model_cpu, self.data_cpu)
            with self.wp.ScopedDevice(self.config.device):
                self.model = self.cfwarp.put_model(self.model_cpu)
                self.data = self.cfwarp.put_data(
                    self.model_cpu,
                    self.data_cpu,
                    nworld=int(self.config.num_samples),
                    nconmax=max(
                        int(self.config.nconmax_per_env),
                        int(getattr(self.data_cpu, "ncon", 0)),
                        int(getattr(self.model_cpu, "nconmax", 0) or 0),
                    ),
                    njmax=max(
                        int(self.config.njmax_per_env),
                        int(getattr(self.data_cpu, "nefc", 0)),
                        int(getattr(self.model_cpu, "njmax", 0) or 0),
                    ),
                )
        finally:
            self.model_cpu.opt.cone = original_cone
            mujoco.mj_forward(self.model_cpu, self.data_cpu)

        self._arm_qpos_tensor = torch.as_tensor(
            self.arm_qpos_ids, device=self.torch_device, dtype=torch.long
        )
        self._gripper_qpos_tensor = torch.as_tensor(
            self.gripper_qpos_ids, device=self.torch_device, dtype=torch.long
        )
        lower: list[float] = []
        upper: list[float] = []
        for jid in self.arm_joint_ids + self.gripper_joint_ids:
            if bool(model.jnt_limited[jid]):
                lower.append(float(model.jnt_range[jid, 0]))
                upper.append(float(model.jnt_range[jid, 1]))
            else:
                lower.append(-1.0e6)
                upper.append(1.0e6)
        self._joint_lower = torch.as_tensor(
            lower, device=self.torch_device, dtype=self.dtype
        )
        self._joint_upper = torch.as_tensor(
            upper, device=self.torch_device, dtype=self.dtype
        )
        self._sphere_centers_ee = torch.as_tensor(
            np.asarray(sphere_centers_ee, dtype=np.float64).reshape(-1, 3),
            device=self.torch_device,
            dtype=self.dtype,
        )
        self._sphere_radii = torch.as_tensor(
            np.asarray(sphere_radii, dtype=np.float64).reshape(-1),
            device=self.torch_device,
            dtype=self.dtype,
        )
        if self._sphere_radii.numel() != self._sphere_centers_ee.shape[0]:
            raise ValueError("sphere_radii length must match sphere_centers_ee")
        self._base_qpos = self.wp.to_torch(self.data.qpos).clone()

    @property
    def num_samples(self) -> int:
        return int(self.config.num_samples)

    def _coerce_seed(self, seed_q: Sequence[float] | Tensor | np.ndarray) -> Tensor:
        seed = torch.as_tensor(
            np.asarray(seed_q, dtype=np.float64).reshape(-1),
            device=self.torch_device,
            dtype=self.dtype,
        )
        if seed.numel() != self.num_q:
            raise ValueError(
                f"seed_q must contain {self.num_q} values "
                f"(arm={self.num_arm}, gripper={self.num_gripper}); got {seed.numel()}"
            )
        return seed

    def _sample_q(self, mean: Tensor, noise_scale: float) -> Tensor:
        scale = torch.empty(self.num_q, device=self.torch_device, dtype=self.dtype)
        scale[: self.num_arm] = float(self.config.arm_noise_scale)
        if self.num_gripper:
            scale[self.num_arm :] = float(self.config.gripper_noise_scale)
        noise = (
            torch.randn(
                (self.config.num_samples, self.num_q),
                device=self.torch_device,
                dtype=self.dtype,
                generator=self.generator,
            )
            * scale
            * float(noise_scale)
        )
        # Keep the first sample at the mean so the elite always contains the
        # incumbent and the algorithm cannot regress when a noisy iteration
        # finds no improvement.
        noise[0].zero_()
        sampled = mean.unsqueeze(0) + noise
        return torch.maximum(
            torch.minimum(sampled, self._joint_upper),
            self._joint_lower,
        )

    def _write_qpos(self, q_samples: Tensor) -> None:
        qpos = self.wp.to_torch(self.data.qpos)
        qpos[:] = self._base_qpos
        if self.num_arm:
            qpos[:, self._arm_qpos_tensor] = q_samples[:, : self.num_arm]
        if self.num_gripper:
            qpos[:, self._gripper_qpos_tensor] = q_samples[:, self.num_arm :]
        qvel = self.wp.to_torch(self.data.qvel)
        qvel.zero_()

    def _sphere_centers_world(self) -> Tensor:
        site_pos = self.wp.to_torch(self.data.site_xpos)[:, self.ee_site_id]
        site_mat = self.wp.to_torch(self.data.site_xmat)[:, self.ee_site_id]
        centers_local = (
            self._sphere_centers_ee.t().unsqueeze(0).expand(site_pos.shape[0], -1, -1)
        )
        return site_pos[:, None, :] + torch.bmm(site_mat, centers_local).transpose(1, 2)

    def _score(
        self,
        q_samples: Tensor,
        seed_q: Tensor,
        target_sphere_indices: Tensor,
        target_points_world: Tensor,
        object_points_world: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._write_qpos(q_samples)
        with self.wp.ScopedDevice(self.config.device):
            self.cfwarp.forward(self.model, self.data)
        sphere_world = self._sphere_centers_world()
        selected = sphere_world[:, target_sphere_indices, :]
        contact_error = (
            (selected - target_points_world[None, :, :]).square().sum(dim=-1)
        )
        contact_cost = float(self.config.contact_weight) * contact_error.sum(dim=-1)

        if object_points_world is not None and object_points_world.shape[0] > 0:
            # Per (world, sphere) min distance to object points.
            ws = sphere_world.shape[0]
            distances = (
                torch.cdist(
                    sphere_world.reshape(ws * sphere_world.shape[1], 3),
                    object_points_world,
                )
                .reshape(ws, sphere_world.shape[1], -1)
                .amin(dim=-1)
            )
            penetration = torch.clamp(
                self._sphere_radii[None, :]
                + float(self.config.penetration_margin)
                - distances,
                min=0.0,
            )
            penetration_cost = float(
                self.config.penetration_weight
            ) * penetration.square().sum(dim=-1)
            min_clearance = distances - self._sphere_radii[None, :]
        else:
            penetration_cost = torch.zeros_like(contact_cost)
            min_clearance = torch.full(
                (sphere_world.shape[0], sphere_world.shape[1]),
                float("inf"),
                device=self.torch_device,
                dtype=self.dtype,
            )

        delta = q_samples - seed_q[None, :]
        regularization_cost = float(
            self.config.regularization_weight
        ) * delta.square().sum(dim=-1)
        lower_violation = torch.clamp(self._joint_lower - q_samples, min=0.0)
        upper_violation = torch.clamp(q_samples - self._joint_upper, min=0.0)
        joint_limit_cost = float(self.config.joint_limit_weight) * (
            lower_violation.square().sum(dim=-1) + upper_violation.square().sum(dim=-1)
        )
        total = contact_cost + penetration_cost + regularization_cost + joint_limit_cost
        return total, sphere_world, min_clearance

    def _contact_metrics(
        self,
        geom_set_a: Sequence[int],
        geom_set_b: Sequence[int],
        feasible_points_world: Tensor,
        contact_tolerance: float,
        penetration_tolerance: float,
    ) -> tuple[Tensor, Tensor]:
        device = self.torch_device
        dtype = self.dtype
        nworld = int(self.config.num_samples)
        ngeom = int(self.model_cpu.ngeom)
        mask_a = torch.zeros(ngeom, dtype=torch.bool, device=device)
        mask_b = torch.zeros(ngeom, dtype=torch.bool, device=device)
        if geom_set_a:
            mask_a[
                torch.as_tensor(
                    sorted(int(geom_id) for geom_id in geom_set_a),
                    dtype=torch.long,
                    device=device,
                )
            ] = True
        if geom_set_b:
            mask_b[
                torch.as_tensor(
                    sorted(int(geom_id) for geom_id in geom_set_b),
                    dtype=torch.long,
                    device=device,
                )
            ] = True

        dist = _backend_tensor(self.data.contact.dist).reshape(-1)
        geom = _backend_tensor(self.data.contact.geom).long().reshape(-1, 2)
        world_ids = _backend_tensor(self.data.contact.worldid).long().reshape(-1)
        contact_pos = _backend_tensor(self.data.contact.pos).reshape(-1, 3)
        nacon = _backend_tensor(self.data.nacon).long().reshape(-1)[0]
        contact_ids = torch.arange(dist.numel(), dtype=torch.long, device=device)
        active = contact_ids < nacon
        valid = active & (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
        safe_g0 = geom[:, 0].clamp(0, ngeom - 1)
        safe_g1 = geom[:, 1].clamp(0, ngeom - 1)
        pair = (mask_a[safe_g0] & mask_b[safe_g1]) | (mask_b[safe_g0] & mask_a[safe_g1])

        penetration = torch.where(
            valid & pair,
            torch.clamp(-dist - float(penetration_tolerance), min=0.0),
            torch.zeros_like(dist),
        )
        max_penetration = torch.zeros(nworld, device=device, dtype=dtype)
        max_penetration.scatter_reduce_(
            0,
            world_ids.clamp(0, nworld - 1),
            penetration,
            reduce="amax",
            include_self=True,
        )

        if feasible_points_world.numel() == 0:
            feasible_hit = torch.zeros_like(active)
        else:
            nearest = torch.cdist(contact_pos, feasible_points_world).amin(dim=1)
            feasible_hit = nearest <= float(contact_tolerance)
        contact_counts = torch.zeros(nworld, device=device, dtype=dtype)
        contact_counts.scatter_add_(
            0,
            world_ids.clamp(0, nworld - 1),
            (valid & pair & feasible_hit).to(dtype),
        )
        return max_penetration, contact_counts

    @torch.no_grad()
    def solve_contact_set(
        self,
        seed_q: Sequence[float] | Tensor | np.ndarray,
        *,
        contact_sphere_indices: Sequence[int],
        target_points_world: np.ndarray | Tensor,
        feasible_points_world: np.ndarray | Tensor,
        geom_set_a: Sequence[int],
        geom_set_b: Sequence[int],
        object_points_world: np.ndarray | Tensor | None = None,
        contact_tolerance: float = 0.025,
        penetration_tolerance: float = 0.0,
        success_contact_fraction: float = 0.5,
    ) -> QConfigResult:
        seed_tensor = self._coerce_seed(seed_q)
        target_points = torch.as_tensor(
            np.asarray(target_points_world, dtype=np.float64).reshape(-1, 3),
            device=self.torch_device,
            dtype=self.dtype,
        )
        if target_points.shape[0] != int(self.config.num_samples):
            raise ValueError("target_points_world must contain one point per sample")
        sphere_indices = torch.as_tensor(
            np.asarray(contact_sphere_indices, dtype=np.int64).reshape(-1),
            device=self.torch_device,
            dtype=torch.long,
        )
        if sphere_indices.numel() != int(self.config.num_samples):
            raise ValueError("contact_sphere_indices must contain one index per sample")
        feasible_tensor = torch.as_tensor(
            np.asarray(feasible_points_world, dtype=np.float64).reshape(-1, 3),
            device=self.torch_device,
            dtype=self.dtype,
        )
        object_tensor: Tensor | None = None
        if object_points_world is not None:
            object_tensor = torch.as_tensor(
                np.asarray(object_points_world, dtype=np.float64).reshape(-1, 3),
                device=self.torch_device,
                dtype=self.dtype,
            )

        mean_q = seed_tensor.clone()
        beta = self.config.final_noise_scale ** (
            1.0 / max(self.config.max_num_iterations, 1)
        )
        history: list[float] = []
        best_cost = float("inf")
        best_q = seed_tensor.clone()
        best_index = 0
        best_iteration = 0
        best_candidates = mean_q.unsqueeze(0).repeat(self.config.num_samples, 1)
        best_costs = torch.full(
            (self.config.num_samples,),
            float("inf"),
            device=self.torch_device,
            dtype=self.dtype,
        )
        best_sphere_centers = torch.zeros(
            (self._sphere_centers_ee.shape[0], 3),
            device=self.torch_device,
            dtype=self.dtype,
        )
        best_clearance = torch.full(
            (self._sphere_centers_ee.shape[0],),
            float("inf"),
            device=self.torch_device,
            dtype=self.dtype,
        )
        best_contact_count = 0
        successful_config_count = 0
        required_contacts = max(
            1,
            int(math.floor(float(success_contact_fraction) * feasible_tensor.shape[0]))
            + 1,
        )

        for iteration in range(self.config.max_num_iterations):
            samples = self._sample_q(mean_q, beta**iteration)
            self._write_qpos(samples)
            with self.wp.ScopedDevice(self.config.device):
                self.cfwarp.forward(self.model, self.data)
            sphere_world = self._sphere_centers_world()
            selected = sphere_world[
                torch.arange(self.config.num_samples, device=self.torch_device),
                sphere_indices,
            ]
            contact_error = (selected - target_points).square().sum(dim=-1)
            contact_cost = float(self.config.contact_weight) * contact_error

            if object_tensor is not None and object_tensor.shape[0] > 0:
                ws = sphere_world.shape[0]
                distances = (
                    torch.cdist(
                        sphere_world.reshape(ws * sphere_world.shape[1], 3),
                        object_tensor,
                    )
                    .reshape(ws, sphere_world.shape[1], -1)
                    .amin(dim=-1)
                )
                penetration = torch.clamp(
                    self._sphere_radii[None, :]
                    + float(self.config.penetration_margin)
                    - distances,
                    min=0.0,
                )
                sphere_penalty = float(
                    self.config.penetration_weight
                ) * penetration.square().sum(dim=-1)
                min_clearance = distances - self._sphere_radii[None, :]
            else:
                sphere_penalty = torch.zeros_like(contact_cost)
                min_clearance = torch.full(
                    (sphere_world.shape[0], sphere_world.shape[1]),
                    float("inf"),
                    device=self.torch_device,
                    dtype=self.dtype,
                )

            max_penetration, feasible_contact_counts = self._contact_metrics(
                geom_set_a,
                geom_set_b,
                feasible_tensor,
                contact_tolerance,
                penetration_tolerance,
            )
            contact_bonus = -float(self.config.contact_weight) * feasible_contact_counts
            penetration_cost = (
                float(self.config.penetration_weight) * max_penetration.square()
            )
            delta = samples - seed_tensor[None, :]
            regularization_cost = float(
                self.config.regularization_weight
            ) * delta.square().sum(dim=-1)
            lower_violation = torch.clamp(self._joint_lower - samples, min=0.0)
            upper_violation = torch.clamp(samples - self._joint_upper, min=0.0)
            joint_limit_cost = float(self.config.joint_limit_weight) * (
                lower_violation.square().sum(dim=-1)
                + upper_violation.square().sum(dim=-1)
            )
            costs = torch.nan_to_num(
                contact_cost
                + sphere_penalty
                + penetration_cost
                + regularization_cost
                + joint_limit_cost
                + contact_bonus,
                nan=float("inf"),
                posinf=float("inf"),
            )
            success = (max_penetration <= 0.0) & (
                feasible_contact_counts >= float(required_contacts)
            )
            successful_config_count = int(success.sum().detach().cpu())
            ranking_costs = torch.where(success, costs - 1.0e6, costs)
            iteration_best_cost_tensor, iteration_best_index_tensor = torch.min(
                ranking_costs, dim=0
            )
            iteration_best_index = int(iteration_best_index_tensor.detach().cpu())
            iteration_best_cost = float(costs[iteration_best_index].detach().cpu())
            if iteration_best_cost < best_cost or bool(success[iteration_best_index]):
                best_cost = iteration_best_cost
                best_q = samples[iteration_best_index].clone()
                best_index = iteration_best_index
                best_iteration = iteration
                best_candidates = samples.clone()
                best_costs = costs.clone()
                best_sphere_centers = sphere_world[iteration_best_index].clone()
                best_clearance = min_clearance[iteration_best_index].clone()
                best_contact_count = int(
                    feasible_contact_counts[iteration_best_index].detach().cpu()
                )
            history.append(iteration_best_cost)
            if bool(success.any()):
                break
            mean_q = self._weighted_q_mean(samples, costs)

        contact_distances = torch.linalg.vector_norm(
            best_sphere_centers[sphere_indices[best_index].reshape(1)]
            - target_points[best_index].reshape(1, 3),
            dim=-1,
        )
        return QConfigResult(
            best_q=best_q,
            best_cost=best_cost,
            best_index=best_index,
            best_iteration=best_iteration,
            candidate_q=best_candidates,
            candidate_costs=best_costs,
            iteration_best_costs=tuple(history),
            iterations=len(history),
            best_sphere_centers_world=best_sphere_centers,
            best_contact_distances=contact_distances,
            best_min_penetration_distance=float(
                torch.min(best_clearance).detach().cpu()
            )
            if best_clearance.numel()
            else float("inf"),
            best_feasible_contact_count=best_contact_count,
            successful_config_count=successful_config_count,
        )

    def _weighted_q_mean(self, q_samples: Tensor, costs: Tensor) -> Tensor:
        count = max(1, int(math.ceil(self.config.elite_ratio * costs.numel())))
        elite_cost, elite_id = torch.topk(costs, k=count, largest=False)
        safe = torch.isfinite(elite_cost)
        if not bool(safe.any()):
            return q_samples[0]
        elite_cost = elite_cost[safe]
        elite_q = q_samples[elite_id[safe]]
        normalized = (elite_cost - elite_cost.mean()) / (
            elite_cost.std(unbiased=False) + 1e-3
        )
        weight = torch.softmax(-normalized / max(self.config.temperature, 1e-6), dim=0)
        return (weight[:, None] * elite_q).sum(dim=0)

    @torch.no_grad()
    def solve(
        self,
        seed_q: Sequence[float] | Tensor | np.ndarray,
        sphere_target_pairs: Sequence[tuple[int, Sequence[float]]],
        *,
        object_points_world: np.ndarray | Tensor | None = None,
    ) -> QConfigResult:
        seed_tensor = self._coerce_seed(seed_q)
        if not sphere_target_pairs:
            raise ValueError("sphere_target_pairs must be non-empty")
        sphere_indices = torch.as_tensor(
            [int(idx) for idx, _ in sphere_target_pairs],
            device=self.torch_device,
            dtype=torch.long,
        )
        target_points = torch.as_tensor(
            np.asarray(
                [
                    np.asarray(pt, dtype=np.float64).reshape(3)
                    for _, pt in sphere_target_pairs
                ]
            ),
            device=self.torch_device,
            dtype=self.dtype,
        )
        object_tensor: Tensor | None = None
        if object_points_world is not None:
            object_tensor = torch.as_tensor(
                np.asarray(object_points_world, dtype=np.float64).reshape(-1, 3),
                device=self.torch_device,
                dtype=self.dtype,
            )

        mean_q = seed_tensor.clone()
        beta = self.config.final_noise_scale ** (
            1.0 / max(self.config.max_num_iterations, 1)
        )
        history: list[float] = []
        improvements: list[float] = []
        best_cost = float("inf")
        best_q = seed_tensor.clone()
        best_index = 0
        best_iteration = 0
        best_candidates = mean_q.unsqueeze(0).repeat(self.config.num_samples, 1)
        best_costs = torch.full(
            (self.config.num_samples,),
            float("inf"),
            device=self.torch_device,
            dtype=self.dtype,
        )
        best_sphere_centers = torch.zeros(
            (self._sphere_centers_ee.shape[0], 3),
            device=self.torch_device,
            dtype=self.dtype,
        )
        best_clearance = torch.full(
            (self._sphere_centers_ee.shape[0],),
            float("inf"),
            device=self.torch_device,
            dtype=self.dtype,
        )

        for iteration in range(self.config.max_num_iterations):
            samples = self._sample_q(mean_q, beta**iteration)
            costs, sphere_world, min_clearance = self._score(
                samples,
                seed_tensor,
                sphere_indices,
                target_points,
                object_tensor,
            )
            costs = torch.nan_to_num(costs, nan=float("inf"), posinf=float("inf"))
            iteration_best_cost_tensor, iteration_best_index_tensor = torch.min(
                costs, dim=0
            )
            iteration_best_cost = float(iteration_best_cost_tensor.detach().cpu())
            iteration_best_index = int(iteration_best_index_tensor.detach().cpu())
            previous = best_cost
            if iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_q = samples[iteration_best_index].clone()
                best_index = iteration_best_index
                best_iteration = iteration
                best_candidates = samples.clone()
                best_costs = costs.clone()
                best_sphere_centers = sphere_world[iteration_best_index].clone()
                best_clearance = min_clearance[iteration_best_index].clone()
            history.append(iteration_best_cost)
            improvements.append(
                float("inf") if not math.isfinite(previous) else previous - best_cost
            )
            mean_q = self._weighted_q_mean(samples, costs)
            check = self.config.improvement_check_steps
            if check > 0 and len(improvements) >= check:
                recent = improvements[-check:]
                if all(
                    math.isfinite(v) and v < float(self.config.improvement_threshold)
                    for v in recent
                ):
                    break

        # Per-pair contact distances of the winning configuration.
        if best_sphere_centers.shape[0]:
            contact_distances = torch.linalg.vector_norm(
                best_sphere_centers[sphere_indices] - target_points,
                dim=-1,
            )
        else:
            contact_distances = torch.zeros(
                len(sphere_target_pairs),
                device=self.torch_device,
                dtype=self.dtype,
            )
        return QConfigResult(
            best_q=best_q,
            best_cost=best_cost,
            best_index=best_index,
            best_iteration=best_iteration,
            candidate_q=best_candidates,
            candidate_costs=best_costs,
            iteration_best_costs=tuple(history),
            iterations=len(history),
            best_sphere_centers_world=best_sphere_centers,
            best_contact_distances=contact_distances,
            best_min_penetration_distance=float(
                torch.min(best_clearance).detach().cpu()
            )
            if best_clearance.numel()
            else float("inf"),
        )


def solve_q_config_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ee_site: str | int,
    arm_joint_names: Sequence[str],
    gripper_joint_names: Sequence[str] = (),
    sphere_centers_ee: np.ndarray | Tensor,
    sphere_radii: np.ndarray | Tensor,
    seed_q: Sequence[float] | Tensor | np.ndarray,
    sphere_target_pairs: Sequence[tuple[int, Sequence[float]]],
    object_points_world: np.ndarray | Tensor | None = None,
    config: QConfigOptimizerConfig | None = None,
) -> QConfigResult:
    """One-shot helper that builds the optimizer and returns its best q."""

    optimizer = ComfreeQConfigOptimizer(
        model,
        data,
        ee_site=ee_site,
        arm_joint_names=arm_joint_names,
        gripper_joint_names=gripper_joint_names,
        sphere_centers_ee=sphere_centers_ee,
        sphere_radii=sphere_radii,
        config=config,
    )
    return optimizer.solve(
        seed_q,
        sphere_target_pairs,
        object_points_world=object_points_world,
    )


def solve_q_config_contact_set(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    ee_site: str | int,
    arm_joint_names: Sequence[str],
    gripper_joint_names: Sequence[str] = (),
    sphere_centers_ee: np.ndarray | Tensor,
    sphere_radii: np.ndarray | Tensor,
    seed_q: Sequence[float] | Tensor | np.ndarray,
    contact_sphere_indices: Sequence[int],
    target_points_world: np.ndarray | Tensor,
    feasible_points_world: np.ndarray | Tensor,
    geom_set_a: Sequence[int],
    geom_set_b: Sequence[int],
    object_points_world: np.ndarray | Tensor | None = None,
    contact_tolerance: float = 0.025,
    penetration_tolerance: float = 0.0,
    success_contact_fraction: float = 0.5,
    config: QConfigOptimizerConfig | None = None,
) -> QConfigResult:
    optimizer = ComfreeQConfigOptimizer(
        model,
        data,
        ee_site=ee_site,
        arm_joint_names=arm_joint_names,
        gripper_joint_names=gripper_joint_names,
        sphere_centers_ee=sphere_centers_ee,
        sphere_radii=sphere_radii,
        config=config,
    )
    return optimizer.solve_contact_set(
        seed_q,
        contact_sphere_indices=contact_sphere_indices,
        target_points_world=target_points_world,
        feasible_points_world=feasible_points_world,
        geom_set_a=geom_set_a,
        geom_set_b=geom_set_b,
        object_points_world=object_points_world,
        contact_tolerance=contact_tolerance,
        penetration_tolerance=penetration_tolerance,
        success_contact_fraction=success_contact_fraction,
    )


def _candidate_q_from_q_config_solution(q_config_solution):
    candidate_q = np.asarray(
        q_config_solution.get("successful_hypothesis_q", []), dtype=np.float64
    ).reshape(
        -1,
        len(q_config_solution["arm_joint_names"])
        + len(q_config_solution["gripper_joint_names"]),
    )
    if candidate_q.shape[0] == 0:
        candidate_q = np.asarray(q_config_solution["candidate_q"], dtype=np.float64)
    candidate_costs = np.asarray(
        q_config_solution.get(
            "successful_hypothesis_costs",
            np.full(candidate_q.shape[0], np.inf, dtype=np.float64),
        ),
        dtype=np.float64,
    ).reshape(-1)
    if candidate_costs.shape[0] != candidate_q.shape[0]:
        candidate_costs = np.full(candidate_q.shape[0], np.inf, dtype=np.float64)
    return candidate_q, candidate_costs


def solve_plan_once_mppi_for_EE_pose(
    env,
    surface,
    q_config_solution,
    contact_offset_ee,
    args,
    *,
    robot_state=None,
    frame_name: str | None = None,
):
    """Run one Dream/Comfree EE MPPI solve for q-config hypotheses."""

    from robocasa.demos import demo_open_drawer_contact_curobo as open_demo
    from robocasa.demos.cost import OpenDrawerPlanOnceCost

    candidate_q, candidate_costs = _candidate_q_from_q_config_solution(
        q_config_solution
    )
    finite = np.isfinite(candidate_costs)
    order = np.argsort(np.where(finite, candidate_costs, np.inf))
    limit = int(
        getattr(args, "q_config_mpc_hypothesis_count", candidate_q.shape[0])
        or candidate_q.shape[0]
    )
    order = order[: max(limit, 1)]

    feasible_cache = q_config_solution["feasible_contact_cache"]
    representative_points = q_config_solution.get("representative_points")
    object_points_world = np.asarray(
        getattr(representative_points, "points_world", representative_points),
        dtype=np.float64,
    ).reshape(-1, 3)
    feasible_points_world = np.asarray(feasible_cache.positions_world, dtype=np.float64)
    object_body_id = _object_body_id_for_surface(env, surface)
    action_distance = float(
        q_config_solution.get(
            "action_distance",
            min(float(getattr(args, "dream_max_action_distance", 0.08)), 0.08),
        )
    )
    object_target = (
        np.asarray(env.sim.data.body_xpos[object_body_id], dtype=np.float64).copy()
        + np.asarray(surface.pull_world, dtype=np.float64).reshape(3) * action_distance
    )
    (
        robot_geoms,
        _ee_geoms,
        target_geoms,
    ) = open_demo._robot_contact_geom_sets_for_surface(env, surface)
    arm_joint_names = tuple(q_config_solution["arm_joint_names"])
    arm_dof = len(arm_joint_names)
    frame_name = frame_name or str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    )
    device = str(getattr(args, "dream_device", "cuda:0"))

    horizon_steps = max(int(getattr(args, "dream_mppi_horizon_steps", 25)), 2)
    sim_dt = max(float(getattr(args, "dream_mppi_dt", 0.005)), 1e-4)
    knot_steps = min(
        max(int(getattr(args, "dream_mppi_knot_steps", 5)), 1), horizon_steps
    )
    config = DreamConfig(
        device=device,
        seed=int(args.seed),
        num_samples=max(int(getattr(args, "dream_mppi_num_samples", 128)), 1),
        num_perturb_samples=max(
            int(getattr(args, "dream_mppi_num_perturb_samples", 1)), 1
        ),
        sim_dt=sim_dt,
        horizon=horizon_steps * sim_dt,
        knot_dt=knot_steps * sim_dt,
        max_num_iterations=max(int(getattr(args, "dream_mppi_iterations", 12)), 1),
        pos_noise_scale=float(getattr(args, "dream_mppi_position_noise", 0.01)),
        rot_noise_scale=float(getattr(args, "dream_mppi_rotation_noise", 0.01)),
        zero_first_knot_noise=False,
        contact_stiffness=float(getattr(args, "dream_contact_stiffness", 0.2)),
        contact_damping=float(getattr(args, "dream_contact_damping", 0.001)),
        nconmax_per_env=max(int(getattr(args, "dream_mppi_nconmax_per_env", 120)), 1),
        njmax_per_env=max(int(getattr(args, "dream_mppi_njmax_per_env", 500)), 1),
        compile_cuda_graph=not bool(getattr(args, "disable_dream_cuda_graph", False)),
        initial_pose_tolerance=float(
            getattr(args, "dream_initial_pose_tolerance", 0.01)
        ),
        cost_aggregation="sum",
    )

    raw_model = _raw_model(env.sim.model)
    raw_data = _raw_data(env.sim.data)
    data_qpos = raw_data.qpos.copy()
    data_qvel = raw_data.qvel.copy()
    start_site_id = env.sim.model.site_name2id(frame_name)
    start_pose = np.concatenate(
        [
            np.asarray(env.sim.data.site_xpos[start_site_id], dtype=np.float64),
            open_demo._rotation_matrix_to_quat_wxyz(
                np.asarray(
                    env.sim.data.site_xmat[start_site_id], dtype=np.float64
                ).reshape(3, 3)
            ),
        ]
    )

    if robot_state is None:
        robot_state = {"robocasa_joint_names": arm_joint_names}

    cache_key = "_comfree_ee_mppi_solver"
    solver = getattr(args, cache_key, None)
    cache_signature = (
        int(config.num_samples),
        int(config.num_perturb_samples),
        int(config.horizon_steps),
        float(config.sim_dt),
        int(config.nconmax_per_env),
        int(config.njmax_per_env),
        str(frame_name),
        tuple(arm_joint_names),
    )
    if (
        solver is not None
        and getattr(solver, "_cache_signature", None) != cache_signature
    ):
        try:
            del solver
        except Exception:
            pass
        solver = None
        setattr(args, cache_key, None)
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    results = []
    try:
        for rank, q_index in enumerate(order):
            q_seed = np.asarray(candidate_q[int(q_index)], dtype=np.float64).reshape(-1)
            target_pos, target_rot = open_demo._site_pose_for_arm_q(
                env,
                robot_state,
                q_seed[:arm_dof],
                frame_name,
            )
            target_pose = np.concatenate(
                [target_pos, open_demo._rotation_matrix_to_quat_wxyz(target_rot)]
            )
            cost = CompositeRolloutCost(
                [
                    OpenDrawerPlanOnceCost(
                        mink_q=q_seed[:arm_dof],
                        contact_offset_ee=contact_offset_ee,
                        feasible_points_world=feasible_points_world,
                        object_points_world=object_points_world,
                        object_body_id=object_body_id,
                        object_target_position=object_target,
                        geom_set_a=tuple(robot_geoms),
                        geom_set_b=tuple(target_geoms),
                        q_weight=float(getattr(args, "dream_mppi_mink_q_weight", 5.0)),
                        orientation_weight=float(
                            getattr(args, "dream_mppi_orientation_weight", 10.0)
                        ),
                        contact_weight=float(
                            getattr(args, "dream_mppi_contact_weight", 500.0)
                        ),
                        object_weight=float(
                            getattr(args, "dream_mppi_object_pose_weight", 100.0)
                        ),
                        collision_weight=float(
                            getattr(
                                args,
                                "dream_mppi_penetration_weight",
                                args.q_config_mpc_penetration_weight,
                            )
                        ),
                        collision_margin=float(
                            getattr(
                                args,
                                "dream_mppi_penetration_margin",
                                args.q_config_mpc_penetration_margin,
                            )
                        ),
                        feasible_distance=float(
                            args.dream_initial_contact_feasible_distance
                        ),
                    ),
                    BatchedLMIKCost(
                        damping=float(getattr(args, "dream_mppi_lm_damping", 1e-2)),
                        residual_weight=float(
                            getattr(args, "dream_mppi_ik_weight", 1.0)
                        ),
                        step_weight=float(
                            getattr(args, "dream_mppi_lm_step_weight", 1e-3)
                        ),
                        joint_limit_weight=float(
                            getattr(args, "dream_mppi_joint_limit_weight", 10.0)
                        ),
                    ),
                ]
            )
            if solver is None:
                solver = ComfreeEEMPPI(
                    raw_model,
                    raw_data,
                    ee_site=frame_name,
                    cost_fn=cost,
                    config=replace(config, seed=int(args.seed) + 7100 + rank),
                    arm_joint_names=arm_joint_names,
                )
                solver._cache_signature = cache_signature
                setattr(args, cache_key, solver)
            else:
                solver.cost_fn = cost
                try:
                    solver.generator.manual_seed(int(args.seed) + 7100 + rank)
                except Exception:
                    pass
            solver.sync_from_mujoco(raw_data)
            result = solver.solve(
                _linear_pose_nominal(start_pose, target_pose, horizon_steps)
            )
            results.append((float(result.best_cost), int(q_index), q_seed, result))
            _empty_cuda_caches(device)
    finally:
        raw_data.qpos[:] = data_qpos
        raw_data.qvel[:] = data_qvel
        env.sim.forward()

    if not results:
        raise RuntimeError("Dream MPPI plan_once produced no candidate result")
    results.sort(key=lambda item: item[0])
    best_cost, best_index, best_q, best_result = results[0]
    return {
        "best_cost": float(best_cost),
        "best_q_index": int(best_index),
        "best_q": np.asarray(best_q, dtype=np.float64),
        "result": best_result,
        "sequence": best_result.ee_pose_sequence.detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "candidate_count": int(len(results)),
    }


class ComfreeJointMPPI(ComfreeEEMPPI):
    """MPPI over per-step arm-joint position targets (joint displacements).

    Actions live in R^7: each row is an absolute target qpos for the 7 arm
    joints; the rollout applies joint-space PD `tau = kp*(q_des - q) - kd*qd`
    on the arm DoFs while Comfree steps the simulator.
    """

    def _sample_q_noise(self, num_samples: int, noise_scale: float) -> Tensor:
        c = self.config
        n_knots = max(2, int(round(c.horizon / c.knot_dt)))
        ramp = torch.logspace(
            math.log10(c.first_ctrl_noise_scale),
            math.log10(c.last_ctrl_noise_scale),
            n_knots,
            device=self.torch_device,
            dtype=self.dtype,
        ).view(1, n_knots, 1)
        scale_value = float(getattr(c, "q_noise_scale", c.pos_noise_scale))
        knot_noise = (
            torch.randn(
                (num_samples, n_knots, 7),
                device=self.torch_device,
                dtype=self.dtype,
                generator=self.generator,
            )
            * ramp
            * scale_value
            * float(noise_scale)
        )
        if c.zero_first_knot_noise:
            knot_noise[:, 0].zero_()
        knot_noise[0].zero_()
        noise = F.interpolate(
            knot_noise.transpose(1, 2),
            size=c.horizon_steps,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        return noise

    def sample_q_sequences(self, q_nominal: Tensor, noise_scale: float = 1.0) -> Tensor:
        sampled = q_nominal.unsqueeze(0).repeat(self.config.num_samples, 1, 1)
        sampled = sampled + self._sample_q_noise(self.config.num_samples, noise_scale)
        sampled = torch.maximum(
            torch.minimum(sampled, self._arm_joint_upper),
            self._arm_joint_lower,
        )
        return sampled

    def _apply_joint_pd(self, target_q: Tensor) -> RolloutStep:
        # Read EE pose and Jacobian (needed by BatchedLMIKCost / cost_fn).
        ee_position = self.wp.to_torch(self.data.site_xpos)[:, self.ee_site_id]
        ee_rotation = self.wp.to_torch(self.data.site_xmat)[:, self.ee_site_id]
        self.wp.copy(
            self._jac_points,
            self.wp.from_torch(ee_position.contiguous(), dtype=self.wp.vec3),
        )
        self._jac(
            self.model,
            self.data,
            self._jacp,
            self._jacr,
            self._jac_points,
            self._jac_bodies,
        )
        jacp = self.wp.to_torch(self._jacp)[:, :, self._arm_dof_tensor]
        jacr = self.wp.to_torch(self._jacr)[:, :, self._arm_dof_tensor]
        jacobian = torch.cat((jacp, jacr), dim=1)
        self._last_ee_jacobian = jacobian

        qpos = self.wp.to_torch(self.data.qpos)
        qvel = self.wp.to_torch(self.data.qvel)
        arm_q = qpos[:, self._arm_qpos_tensor]
        arm_qd = qvel[:, self._arm_dof_tensor]

        kp = float(self.config.position_kp)
        kd = float(self.config.position_kd)
        torque = kp * (target_q - arm_q) - kd * arm_qd
        torque = torch.clamp(torque, -self._torque_limit, self._torque_limit)

        qfrc = self.wp.to_torch(self.data.qfrc_applied)
        qfrc[:] = self._snapshot["qfrc_applied"]
        qfrc[:, self._arm_dof_tensor] += torque
        if "xfrc_applied" in self._snapshot:
            self.wp.to_torch(self.data.xfrc_applied)[:] = self._snapshot["xfrc_applied"]
        self._neutralize_actuators()

        # Synthesize a target_ee_pose by forward-kinematics-free placeholder:
        # the cost expects target_ee_pose for orientation/IK regularizers; use
        # the current EE pose so those terms vanish when not relevant.
        target_pose = torch.cat((ee_position, _matrix_to_quat(ee_rotation)), dim=-1)
        return RolloutStep(
            step_index=-1,
            is_terminal=False,
            target_ee_pose=target_pose,
            ee_position=ee_position,
            ee_rotation=ee_rotation,
            qpos=qpos,
            qvel=qvel,
            data=self.data,
            model=self.model,
            candidate_ids=self._candidate_ids,
            replica_ids=self._replica_ids,
            ee_jacobian=jacobian,
            arm_qpos=arm_q,
            arm_joint_lower=self._arm_joint_lower,
            arm_joint_upper=self._arm_joint_upper,
        )

    def _prepare_q_nominal(self, q_nominal: Tensor | np.ndarray | None) -> Tensor:
        h = self.config.horizon_steps
        if q_nominal is None:
            current = self.wp.to_torch(self.data.qpos)[0, self._arm_qpos_tensor]
            return current.unsqueeze(0).repeat(h, 1).clone()
        q_t = torch.as_tensor(q_nominal, device=self.torch_device, dtype=self.dtype)
        if q_t.shape == (7,):
            q_t = q_t.unsqueeze(0).repeat(h, 1)
        if q_t.ndim != 2 or q_t.shape[1] != 7:
            raise ValueError("nominal q sequence must have shape (H, 7) or (7,)")
        if q_t.shape[0] < h:
            q_t = torch.cat((q_t, q_t[-1:].repeat(h - q_t.shape[0], 1)), dim=0)
        elif q_t.shape[0] > h:
            q_t = q_t[:h]
        return q_t.clone()

    @torch.no_grad()
    def rollout_q(self, candidate_q_sequences: Tensor) -> Tensor:
        candidates = torch.as_tensor(
            candidate_q_sequences, device=self.torch_device, dtype=self.dtype
        )
        expected = (self.config.num_samples, self.config.horizon_steps, 7)
        if tuple(candidates.shape) != expected:
            raise ValueError(
                f"candidate_q_sequences must have shape {expected}, got {tuple(candidates.shape)}"
            )
        actions = candidates.repeat_interleave(self.config.num_perturb_samples, dim=0)
        self._restore_state()
        # Snap initial qpos to the first nominal target (per-world).
        qpos = self.wp.to_torch(self.data.qpos)
        qpos[:, self._arm_qpos_tensor] = actions[:, 0]
        qvel = self.wp.to_torch(self.data.qvel)
        qvel[:, self._arm_dof_tensor] = 0.0
        self.cfwarp.forward(self.model, self.data)
        self._last_ee_jacobian = None

        cumulative = torch.zeros(
            self.config.num_worlds, device=self.torch_device, dtype=self.dtype
        )
        initial_ctx = self._make_step_context(
            0,
            torch.zeros(
                (self.config.num_worlds, 7), device=self.torch_device, dtype=self.dtype
            ),
            self.config.horizon_steps == 1,
        )
        cumulative += self._coerce_cost(self.cost_fn(initial_ctx))
        for step_index in range(1, self.config.horizon_steps):
            target_q = actions[:, step_index]
            pre_ctx = self._apply_joint_pd(target_q)
            if self.pre_step_fn is not None:
                self.pre_step_fn(replace(pre_ctx, step_index=step_index))
            self._step_backend()
            terminal = step_index == self.config.horizon_steps - 1
            ctx = self._make_step_context(
                step_index,
                pre_ctx.target_ee_pose,
                terminal,
            )
            cumulative += self._coerce_cost(self.cost_fn(ctx))
            if terminal and self.terminal_cost_fn is not None:
                cumulative += self._coerce_cost(self.terminal_cost_fn(ctx))
        if self.config.cost_aggregation == "mean":
            cumulative /= float(self.config.horizon_steps)
        return cumulative.view(
            self.config.num_samples, self.config.num_perturb_samples
        ).mean(dim=1)

    @torch.no_grad()
    def trace_arm_qpos_from_q(self, q_sequence: Tensor | np.ndarray) -> Tensor:
        seq = torch.as_tensor(
            q_sequence, device=self.torch_device, dtype=self.dtype
        ).clone()
        expected = (self.config.horizon_steps, 7)
        if tuple(seq.shape) != expected:
            raise ValueError(
                f"q_sequence must have shape {expected}, got {tuple(seq.shape)}"
            )
        candidates = seq.unsqueeze(0).expand(self.config.num_samples, -1, -1).clone()
        actions = candidates.repeat_interleave(self.config.num_perturb_samples, dim=0)
        self._restore_state()
        qpos = self.wp.to_torch(self.data.qpos)
        qpos[:, self._arm_qpos_tensor] = actions[:, 0]
        qvel = self.wp.to_torch(self.data.qvel)
        qvel[:, self._arm_dof_tensor] = 0.0
        self.cfwarp.forward(self.model, self.data)
        self._last_ee_jacobian = None
        trace = [self.wp.to_torch(self.data.qpos)[0, self._arm_qpos_tensor].clone()]
        for step_index in range(1, self.config.horizon_steps):
            self._apply_joint_pd(actions[:, step_index])
            self._step_backend()
            trace.append(
                self.wp.to_torch(self.data.qpos)[0, self._arm_qpos_tensor].clone()
            )
        return torch.stack(trace, dim=0)

    @torch.no_grad()
    def solve_q(self, q_nominal: Tensor | np.ndarray | None = None) -> EEMPCResult:
        nominal = self._prepare_q_nominal(q_nominal)
        global_best_cost = float("inf")
        global_best = nominal.clone()
        global_best_costs = torch.full(
            (self.config.num_samples,),
            torch.inf,
            device=self.torch_device,
            dtype=self.dtype,
        )
        history: list[float] = []
        beta = self.config.final_noise_scale ** (
            1.0 / max(self.config.max_num_iterations, 1)
        )
        best_index_int = 0
        best_iteration = 0
        for iteration in range(self.config.max_num_iterations):
            candidates = self.sample_q_sequences(nominal, beta**iteration)
            costs = self.rollout_q(candidates)
            best_cost_t, best_index_t = torch.min(costs, dim=0)
            best_cost = float(best_cost_t.detach().cpu())
            if best_cost < global_best_cost:
                global_best_cost = best_cost
                best_index_int = int(best_index_t.detach().cpu())
                best_iteration = iteration
                global_best = candidates[best_index_int].clone()
                global_best_costs = costs.clone()
            history.append(best_cost)
            # Update nominal as elite-weighted mean over q-space.
            count = max(1, int(math.ceil(self.config.elite_ratio * costs.numel())))
            elite_cost, elite_id = torch.topk(costs, k=count, largest=False)
            safe = torch.isfinite(elite_cost)
            if bool(safe.any()):
                ec = elite_cost[safe]
                elite = candidates[elite_id[safe]]
                normalized = (ec - ec.mean()) / (ec.std(unbiased=False) + 1e-2)
                weight = torch.softmax(
                    -normalized / max(self.config.temperature, 1e-6), dim=0
                )
                nominal = (weight[:, None, None] * elite).sum(dim=0)
        # Trace the resulting joint trajectory through the same PD rollout
        # so the viewer's joint-PD playback sees a feasible q sequence.
        arm_trace = (
            self.trace_arm_qpos_from_q(global_best)
            if math.isfinite(global_best_cost)
            else None
        )
        # Pack into EEMPCResult; ee_pose_sequence is filled with the EE poses
        # observed along the trace (best-effort), so downstream code that reads
        # `result.ee_pose_sequence` still works.
        ee_pose_trace = torch.zeros(
            (self.config.horizon_steps, 7),
            device=self.torch_device,
            dtype=self.dtype,
        )
        if arm_trace is not None:
            # We already executed the trace; sample the final per-step EE pose
            # from the FIRST world only. Cheap and adequate for downstream use.
            # (We re-run forward for each row to keep this minimal-coupling.)
            self._restore_state()
            qpos = self.wp.to_torch(self.data.qpos)
            for i in range(self.config.horizon_steps):
                qpos[0, self._arm_qpos_tensor] = arm_trace[i]
                self.cfwarp.forward(self.model, self.data)
                p = self.wp.to_torch(self.data.site_xpos)[0, self.ee_site_id]
                r = self.wp.to_torch(self.data.site_xmat)[0, self.ee_site_id]
                ee_pose_trace[i, :3] = p
                ee_pose_trace[i, 3:] = _matrix_to_quat(r)
        return EEMPCResult(
            ee_pose_sequence=ee_pose_trace,
            first_ee_pose=ee_pose_trace[0],
            best_cost=global_best_cost,
            best_index=best_index_int,
            best_iteration=best_iteration,
            candidate_costs=global_best_costs,
            iteration_best_costs=tuple(history),
            iterations=len(history),
            arm_qpos_sequence=arm_trace,
        )


def solve_plan_once_mppi_for_q_configs(
    env,
    surface,
    q_config_solution,
    contact_offset_ee,
    args,
    *,
    robot_state=None,
    frame_name: str | None = None,
):
    """Plan a per-step JOINT-position (Δq) trajectory via Comfree MPPI.

    Same interface as :func:`solve_plan_once_mppi_for_EE_pose`, but the MPPI
    decision variables are absolute arm-q targets per step (equivalently
    joint displacements from the nominal). Returns the same dict shape.
    """

    from robocasa.demos import demo_open_drawer_contact_curobo as open_demo
    from robocasa.demos.cost import OpenDrawerPlanOnceCost

    candidate_q, candidate_costs = _candidate_q_from_q_config_solution(
        q_config_solution
    )
    finite = np.isfinite(candidate_costs)
    order = np.argsort(np.where(finite, candidate_costs, np.inf))
    limit = int(
        getattr(args, "q_config_mpc_hypothesis_count", candidate_q.shape[0])
        or candidate_q.shape[0]
    )
    order = order[: max(limit, 1)]

    feasible_cache = q_config_solution["feasible_contact_cache"]
    representative_points = q_config_solution.get("representative_points")
    object_points_world = np.asarray(
        getattr(representative_points, "points_world", representative_points),
        dtype=np.float64,
    ).reshape(-1, 3)
    feasible_points_world = np.asarray(feasible_cache.positions_world, dtype=np.float64)
    object_body_id = _object_body_id_for_surface(env, surface)
    action_distance = float(
        q_config_solution.get(
            "action_distance",
            min(float(getattr(args, "dream_max_action_distance", 0.08)), 0.08),
        )
    )
    object_target = (
        np.asarray(env.sim.data.body_xpos[object_body_id], dtype=np.float64).copy()
        + np.asarray(surface.pull_world, dtype=np.float64).reshape(3) * action_distance
    )
    (
        robot_geoms,
        _ee_geoms,
        target_geoms,
    ) = open_demo._robot_contact_geom_sets_for_surface(env, surface)
    arm_joint_names = tuple(q_config_solution["arm_joint_names"])
    arm_dof = len(arm_joint_names)
    frame_name = frame_name or str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    )
    device = str(getattr(args, "dream_device", "cuda:0"))

    horizon_steps = max(int(getattr(args, "dream_mppi_horizon_steps", 25)), 2)
    sim_dt = max(float(getattr(args, "dream_mppi_dt", 0.005)), 1e-4)
    knot_steps = min(
        max(int(getattr(args, "dream_mppi_knot_steps", 5)), 1), horizon_steps
    )
    config = DreamConfig(
        device=device,
        seed=int(args.seed),
        num_samples=max(int(getattr(args, "dream_mppi_num_samples", 128)), 1),
        num_perturb_samples=max(
            int(getattr(args, "dream_mppi_num_perturb_samples", 1)), 1
        ),
        sim_dt=sim_dt,
        horizon=horizon_steps * sim_dt,
        knot_dt=knot_steps * sim_dt,
        max_num_iterations=max(int(getattr(args, "dream_mppi_iterations", 12)), 1),
        pos_noise_scale=float(getattr(args, "dream_mppi_q_noise", 0.05)),
        rot_noise_scale=float(getattr(args, "dream_mppi_q_noise", 0.05)),
        zero_first_knot_noise=False,
        contact_stiffness=float(getattr(args, "dream_contact_stiffness", 0.2)),
        contact_damping=float(getattr(args, "dream_contact_damping", 0.001)),
        nconmax_per_env=max(int(getattr(args, "dream_mppi_nconmax_per_env", 120)), 1),
        njmax_per_env=max(int(getattr(args, "dream_mppi_njmax_per_env", 500)), 1),
        compile_cuda_graph=not bool(getattr(args, "disable_dream_cuda_graph", False)),
        initial_pose_tolerance=float(
            getattr(args, "dream_initial_pose_tolerance", 0.01)
        ),
        cost_aggregation="sum",
    )
    # ComfreeJointMPPI uses pos_noise_scale as joint-space noise magnitude.

    raw_model = _raw_model(env.sim.model)
    raw_data = _raw_data(env.sim.data)
    data_qpos = raw_data.qpos.copy()
    data_qvel = raw_data.qvel.copy()

    if robot_state is None:
        robot_state = {"robocasa_joint_names": arm_joint_names}

    # Current arm q is the nominal starting point.
    qpos_addrs = [env.sim.model.get_joint_qpos_addr(n) for n in arm_joint_names]
    start_q = np.asarray(env.sim.data.qpos[qpos_addrs], dtype=np.float64)

    # Reuse one Comfree solver across ALL q hypotheses AND across replans.
    # Each ComfreeJointMPPI allocates num_worlds parallel envs + Jacobian
    # buffers + a CUDA graph; recreating per-hypothesis blows GPU memory.
    cache_key = "_comfree_joint_mppi_solver"
    solver = getattr(args, cache_key, None)
    cache_signature = (
        int(config.num_samples),
        int(config.num_perturb_samples),
        int(config.horizon_steps),
        float(config.sim_dt),
        int(config.nconmax_per_env),
        int(config.njmax_per_env),
        str(frame_name),
        tuple(arm_joint_names),
    )
    if (
        solver is not None
        and getattr(solver, "_cache_signature", None) != cache_signature
    ):
        # Config changed (e.g. horizon); drop the old solver.
        try:
            del solver
        except Exception:
            pass
        solver = None
        setattr(args, cache_key, None)
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    results = []
    try:
        for rank, q_index in enumerate(order):
            q_seed = np.asarray(candidate_q[int(q_index)], dtype=np.float64).reshape(
                -1
            )[:arm_dof]
            # Nominal q-sequence: linear interpolation from current arm q to
            # the q-config hypothesis q_seed.
            alpha = np.linspace(0.0, 1.0, horizon_steps, dtype=np.float64)[:, None]
            q_nominal = (1.0 - alpha) * start_q[None, :] + alpha * q_seed[None, :]

            cost = CompositeRolloutCost(
                [
                    OpenDrawerPlanOnceCost(
                        mink_q=q_seed,
                        contact_offset_ee=contact_offset_ee,
                        feasible_points_world=feasible_points_world,
                        object_points_world=object_points_world,
                        object_body_id=object_body_id,
                        object_target_position=object_target,
                        geom_set_a=tuple(robot_geoms),
                        geom_set_b=tuple(target_geoms),
                        q_weight=float(getattr(args, "dream_mppi_mink_q_weight", 5.0)),
                        orientation_weight=float(
                            getattr(args, "dream_mppi_orientation_weight", 10.0)
                        ),
                        contact_weight=float(
                            getattr(args, "dream_mppi_contact_weight", 500.0)
                        ),
                        object_weight=float(
                            getattr(args, "dream_mppi_object_pose_weight", 100.0)
                        ),
                        collision_weight=float(
                            getattr(
                                args,
                                "dream_mppi_penetration_weight",
                                args.q_config_mpc_penetration_weight,
                            )
                        ),
                        collision_margin=float(
                            getattr(
                                args,
                                "dream_mppi_penetration_margin",
                                args.q_config_mpc_penetration_margin,
                            )
                        ),
                        feasible_distance=float(
                            args.dream_initial_contact_feasible_distance
                        ),
                    ),
                    BatchedLMIKCost(
                        damping=float(getattr(args, "dream_mppi_lm_damping", 1e-2)),
                        residual_weight=float(
                            getattr(args, "dream_mppi_ik_weight", 1.0)
                        ),
                        step_weight=float(
                            getattr(args, "dream_mppi_lm_step_weight", 1e-3)
                        ),
                        joint_limit_weight=float(
                            getattr(args, "dream_mppi_joint_limit_weight", 10.0)
                        ),
                    ),
                ]
            )
            if solver is None:
                solver = ComfreeJointMPPI(
                    raw_model,
                    raw_data,
                    ee_site=frame_name,
                    cost_fn=cost,
                    config=replace(config, seed=int(args.seed) + 7100 + rank),
                    arm_joint_names=arm_joint_names,
                )
                solver._cache_signature = cache_signature
                setattr(args, cache_key, solver)
            else:
                # Swap cost_fn and reseed; reuse all CUDA buffers.
                solver.cost_fn = cost
                try:
                    solver.generator.manual_seed(int(args.seed) + 7100 + rank)
                except Exception:
                    pass
            # Always re-sync from live MuJoCo state (drawer / base may have moved).
            solver.sync_from_mujoco(raw_data)
            result = solver.solve_q(q_nominal)
            results.append((float(result.best_cost), int(q_index), q_seed, result))
            _empty_cuda_caches(device)
    finally:
        raw_data.qpos[:] = data_qpos
        raw_data.qvel[:] = data_qvel
        env.sim.forward()

    if not results:
        raise RuntimeError("Joint-space MPPI plan_once produced no candidate result")
    results.sort(key=lambda item: item[0])
    best_cost, best_index, best_q, best_result = results[0]
    return {
        "best_cost": float(best_cost),
        "best_q_index": int(best_index),
        "best_q": np.asarray(best_q, dtype=np.float64),
        "result": best_result,
        "sequence": best_result.ee_pose_sequence.detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "candidate_count": int(len(results)),
    }


def _mppi_segments_for_stage(
    stage_name: str, q_sequence: np.ndarray
) -> list[dict[str, Any]]:
    if q_sequence.shape[0] <= 0:
        return []
    contact_steps = max(1, int(np.ceil(q_sequence.shape[0] * 0.5)))
    pull_steps = max(0, int(q_sequence.shape[0]) - contact_steps)
    segments = [
        {
            "name": f"{stage_name}:contact",
            "steps": int(contact_steps),
            "planner": "mppi_composite_rollout_cost",
        }
    ]
    if pull_steps > 0:
        segments.append(
            {
                "name": f"{stage_name}:pull",
                "steps": int(pull_steps),
                "planner": "mppi_composite_rollout_cost",
            }
        )
    return segments


def _cached_curobo_joint_planner(args):
    from robocasa.demos import demo_close_drawer_contact_curobo as close_demo

    close_demo._ensure_curobo_importable()
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    signature = (
        str(getattr(args, "curobo_robot_cfg", "franka.yml")),
        int(getattr(args, "curobo_trajopt_tsteps", 32)),
        float(getattr(args, "curobo_interpolation_dt", 0.02)),
        int(getattr(args, "curobo_ik_seeds", 16)),
        int(getattr(args, "curobo_graph_seeds", 2)),
        int(getattr(args, "curobo_trajopt_seeds", 2)),
        bool(getattr(args, "disable_curobo_self_collision", False)),
        bool(getattr(args, "disable_curobo_cuda_graph", False)),
    )
    cached = getattr(args, "_curobo_joint_goal_planner", None)
    if cached is not None and cached.get("signature") == signature:
        return cached

    tensor_args = TensorDeviceType()
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        signature[0],
        None,
        tensor_args,
        trajopt_tsteps=signature[1],
        interpolation_dt=signature[2],
        use_cuda_graph=not signature[7],
        self_collision_check=not signature[6],
        self_collision_opt=not signature[6],
        num_ik_seeds=signature[3],
        num_graph_seeds=signature[4],
        num_trajopt_seeds=signature[5],
        collision_checker_type=None,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(enable_graph=not signature[7])
    cached = {
        "signature": signature,
        "tensor_args": tensor_args,
        "motion_gen": motion_gen,
        "curobo_joint_names": tuple(close_demo._extract_curobo_joint_names(motion_gen)),
    }
    setattr(args, "_curobo_joint_goal_planner", cached)
    return cached


def _plan_curobo_joint_goal(
    env,
    robot_state: Mapping[str, Any],
    goal_q_robosuite: np.ndarray,
    args,
    *,
    name: str,
    extra_exclude_body_names=None,
):
    from robocasa.demos.curobo_planning import plan_joint_goal

    return plan_joint_goal(
        env,
        robot_state,
        goal_q_robosuite,
        args,
        name=name,
        extra_exclude_body_names=extra_exclude_body_names,
    )


def _validate_curobo_approach_collision_free(
    env,
    stage,
    q_approach: np.ndarray,
    arm_joint_names: tuple[str, ...],
    drawer_q: float,
    args,
):
    from robocasa.demos import demo_close_drawer_contact_curobo as close_demo
    from robocasa.demos.open_drawer.collision import (
        check_arm_q_collision_for_surface as _check_open_drawer_arm_collision,
    )

    ctx = getattr(stage, "_replan_context", None)
    surface = None if ctx is None else ctx.get("surface", stage)
    if surface is None:
        return
    failures = []
    for step_index, q_arm in enumerate(
        np.asarray(q_approach, dtype=np.float64).reshape(-1, 7)
    ):
        ok, reason = _check_open_drawer_arm_collision(
            env,
            surface,
            arm_joint_names,
            q_arm,
            float(drawer_q),
            set_arm_q=close_demo._set_env_arm_q,
            set_drawer_joint_value=close_demo._set_drawer_joint_value,
            allowed_ee_geom_name=None,
            penetration_tolerance=float(
                getattr(args, "mink_collision_penetration_tolerance", 0.0)
            ),
            collision_scope="arm",
        )
        if not ok:
            failures.append((int(step_index), str(reason)))
            if len(failures) >= int(getattr(args, "joint_validation_max_failures", 8)):
                break
    if failures:
        detail = "; ".join(f"step={step}: {reason}" for step, reason in failures)
        raise RuntimeError(
            f"cuRobo approach for stage '{stage.name}' is not collision-free "
            f"in MuJoCo validation: {detail}"
        )


def _approach_exclude_body_names(surface) -> tuple[str, ...]:
    """Bodies to remove from the cuRobo world for an approach plan.

    Approach goals sit at precontact poses that brush the target drawer / handle.
    cuRobo's collision world (even with convex-hull meshes) flags this as
    in-collision because the gripper spheres are within `activation_distance` of
    the handle's hull. Since terminal contact is intended and the full approach
    trajectory is re-checked in MuJoCo by `_validate_curobo_approach_collision_free`
    using exact-mesh contacts, it is safe to drop the target body from the cuRobo
    world for the approach plan only.
    """

    if surface is None:
        return ()
    names: list[str] = []
    object_body_names = getattr(surface, "object_body_names_", None)
    if object_body_names:
        names.extend(str(n) for n in object_body_names)
    drawer_body = getattr(surface, "drawer_body_name_", None)
    if drawer_body:
        names.append(str(drawer_body))
    seen = set()
    result = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        result.append(n)
    return tuple(result)


def solve_stages(env, stages, args, *, robot_state=None, frame_name: str | None = None):
    """Solve staged open-drawer execution.

    For each stage, approach is planned by cuRobo to the mink precontact
    waypoint. The contact/pull phase then switches to the existing joint-space
    MPPI planner from the live end of the approach trajectory.
    """

    from robocasa.demos import demo_close_drawer_contact_curobo as close_demo

    if not stages:
        raise RuntimeError("solve_stages requires at least one stage")
    if robot_state is None:
        robot_state = close_demo.get_robot_arm_state(env)

    frame_name = frame_name or str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    )
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    current_robot_state = dict(robot_state)
    current_q = np.asarray(robot_state["q"], dtype=np.float64).reshape(7).copy()
    chunks: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []
    stage_solutions = []

    try:
        for stage_index, stage in enumerate(stages):
            q_waypoints = np.asarray(stage.mink_solution.q_waypoints, dtype=np.float64)
            if q_waypoints.size == 0:
                raise RuntimeError(
                    f"Stage '{stage.name}' has no mink q_waypoints for approach"
                )
            q_waypoints = q_waypoints.reshape(-1, 7)
            stage_start_drawer_q = float(
                getattr(stage, "start_drawer_q", close_demo._drawer_joint_value(env))
            )
            ctx = getattr(stage, "_replan_context", None)
            precontact_solution = (
                None if ctx is None else ctx.get("precontact_solution")
            )
            if precontact_solution is not None and hasattr(
                precontact_solution, "arm_q"
            ):
                approach_goal_q = np.asarray(
                    precontact_solution.arm_q, dtype=np.float64
                ).reshape(7)
            else:
                approach_goal_q = q_waypoints[0]
            surface = None if ctx is None else ctx.get("surface", stage)
            if surface is not None:
                from robocasa.demos.open_drawer.collision import (
                    check_arm_q_collision_for_surface as _check_open_drawer_arm_collision,
                )

                ok, reason = _check_open_drawer_arm_collision(
                    env,
                    surface,
                    arm_joint_names,
                    approach_goal_q,
                    stage_start_drawer_q,
                    set_arm_q=close_demo._set_env_arm_q,
                    set_drawer_joint_value=close_demo._set_drawer_joint_value,
                    allowed_ee_geom_name=None,
                    penetration_tolerance=float(
                        getattr(args, "mink_collision_penetration_tolerance", 0.0)
                    ),
                    collision_scope="arm",
                )
                if not ok:
                    raise RuntimeError(
                        f"Stage '{stage.name}' approach goal is not full-arm "
                        f"collision-free before cuRobo planning: {reason}"
                    )

            close_demo._set_drawer_joint_value(env, stage_start_drawer_q)
            close_demo._set_env_arm_q(env, arm_joint_names, current_q)
            env.sim.forward()
            current_robot_state = close_demo.get_robot_arm_state(env)

            q_approach, approach_segments = _plan_curobo_joint_goal(
                env,
                current_robot_state,
                approach_goal_q,
                args,
                name=f"{stage.name}:approach",
                extra_exclude_body_names=_approach_exclude_body_names(surface),
            )
            q_approach = np.asarray(q_approach, dtype=np.float64).reshape(-1, 7)
            if q_approach.shape[0] == 0:
                raise RuntimeError(f"cuRobo produced empty approach for {stage.name}")
            if chunks and np.allclose(q_approach[0], chunks[-1][-1]):
                q_approach = q_approach[1:]
            precontact_q = np.asarray(approach_goal_q, dtype=np.float64).reshape(1, 7)
            if q_approach.shape[0] == 0 or not np.allclose(
                q_approach[-1],
                precontact_q[0],
                atol=float(getattr(args, "solve_stages_mink_q_atol", 1e-3)),
                rtol=0.0,
            ):
                q_approach = np.concatenate([q_approach, precontact_q], axis=0)
            _validate_curobo_approach_collision_free(
                env,
                stage,
                q_approach,
                arm_joint_names,
                stage_start_drawer_q,
                args,
            )
            if q_approach.shape[0] > 0:
                chunks.append(q_approach)
            for segment in approach_segments:
                segment = dict(segment)
                segment["name"] = f"{stage.name}:approach"
                segment["steps"] = int(q_approach.shape[0])
                segment["planner"] = "curobo_joint_goal_to_mink_q"
                segment["terminal_collision_allowed"] = False
                segments.append(segment)

            current_q = np.asarray(chunks[-1][-1], dtype=np.float64).reshape(7)
            close_demo._set_env_arm_q(env, arm_joint_names, current_q)
            env.sim.forward()
            current_robot_state = close_demo.get_robot_arm_state(env)

            if ctx is None:
                raise RuntimeError(f"Stage '{stage.name}' has no MPPI replan context")
            surface = ctx.get("surface", stage)
            mppi_solution = solve_plan_once_mppi_for_q_configs(
                env,
                surface,
                ctx["q_config_solution"],
                ctx["contact_offset_ee"],
                args,
                robot_state=current_robot_state,
                frame_name=frame_name,
            )
            stage_solutions.append(mppi_solution)
            stage.dream_result = mppi_solution["result"]
            stage.dream_sequence = np.asarray(
                mppi_solution["sequence"], dtype=np.float64
            )
            diagnostics = dict(getattr(stage, "dream_diagnostics", {}) or {})
            diagnostics.update(
                {
                    "solve_stages_stage_index": int(stage_index),
                    "solve_stages_approach_steps": int(q_approach.shape[0]),
                    "solve_stages_mppi_best_cost": float(mppi_solution["best_cost"]),
                    "solve_stages_mppi_best_q_index": int(
                        mppi_solution["best_q_index"]
                    ),
                    "solve_stages_mppi_candidate_count": int(
                        mppi_solution["candidate_count"]
                    ),
                }
            )
            stage.dream_diagnostics = diagnostics

            arm_seq = getattr(mppi_solution["result"], "arm_qpos_sequence", None)
            if arm_seq is None:
                raise RuntimeError(
                    f"Stage '{stage.name}' MPPI result has no arm_qpos_sequence"
                )
            if hasattr(arm_seq, "detach"):
                arm_seq = arm_seq.detach().cpu().numpy()
            arm_seq = np.asarray(arm_seq, dtype=np.float64).reshape(-1, 7)
            if arm_seq.shape[0] == 0:
                raise RuntimeError(f"Stage '{stage.name}' MPPI arm sequence is empty")
            if chunks and np.allclose(arm_seq[0], chunks[-1][-1]):
                arm_seq_for_traj = arm_seq[1:]
            else:
                arm_seq_for_traj = arm_seq
            if arm_seq_for_traj.shape[0] > 0:
                chunks.append(arm_seq_for_traj)
            segments.extend(_mppi_segments_for_stage(stage.name, arm_seq_for_traj))
            current_q = np.asarray(arm_seq[-1], dtype=np.float64).reshape(7)
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()

    if not chunks:
        raise RuntimeError("solve_stages produced no executable arm trajectory")
    q_traj = np.concatenate(chunks, axis=0)
    q_delta = np.linalg.norm(q_traj - q_traj[:1], axis=1)
    print(
        "[solve_stages] "
        f"frames={q_traj.shape[0]} "
        f"segments={len(segments)} "
        f"max_joint_delta={float(np.max(q_delta)):.6f} "
        f"final_joint_delta={float(q_delta[-1]):.6f}",
        flush=True,
    )
    return {
        "q_traj": q_traj,
        "segments": segments,
        "stage_solutions": stage_solutions,
    }


__all__ = [
    "ComfreeEEMPPI",
    "ComfreeQConfigOptimizer",
    "QConfigOptimizerConfig",
    "QConfigResult",
    "BatchedLMIKCost",
    "CompositeRolloutCost",
    "DreamConfig",
    "EEMPCJob",
    "EEMPCResult",
    "FeasibleContactRewardCost",
    "FingerContactSetLineOfSightCost",
    "JointPositionTaskCost",
    "MultiGPUEEMPCResult",
    "NonPenetrationCost",
    "ObjectFramePointPositionTaskCost",
    "ObjectFramePositionTaskCost",
    "OpenDrawerDreamModel",
    "OpenDrawerDreamRollout",
    "OpenDrawerDreamState",
    "RolloutStep",
    "TaskCost",
    "score_feasible_contact_initial_poses",
    "solve_ee_pose_mpc",
    "solve_ee_pose_mpc_multi_gpu",
    "solve_plan_once_mppi_for_q_configs",
    "solve_plan_once_mppi_for_EE_pose",
    "solve_stages",
    "ComfreeJointMPPI",
    "solve_q_config_contact",
    "solve_q_config_contact_set",
]


@torch.no_grad()
def solve_arm_q_mppi(
    env,
    *,
    robot_state,
    q_mink,
    drawer_q=None,
    args=None,
    target_pos=None,
    target_rot=None,
    ee_site_name=None,
    initial_gripper_opening=None,
    optimize_gripper=False,
) -> np.ndarray:
    """MPPI refinement of a single 7-D arm-q against non-penetration cost.

    Two tracking modes:
      * joint-space (default): tracks toward ``q_mink`` in joint space.
      * cartesian: when ``target_pos`` is provided, tracks the EE site
        ``ee_site_name`` toward ``(target_pos, target_rot)``. In this mode
        ``q_mink`` is purely the initial guess / sampling nominal — typically
        the demonstration arm-q — and joint-space tracking is dropped.

    Optional gripper DoF (``optimize_gripper=True``): MPPI samples an extra
    1-D opening ``g`` (full fingertip distance, half applied to each finger
    joint), initialised from ``initial_gripper_opening``. The skeleton EE
    pose ignores the gripper's true volume, so even when the skeleton pose
    is feasible the MuJoCo hand may collide with the handle; jointly tuning
    ``g`` lets MPPI open/close the gripper to clear those penetrations.
    """
    q_mink = np.asarray(q_mink, dtype=np.float64).reshape(7)
    num_samples = (
        int(getattr(args, "autogen_qmppi_num_samples", 256))
        if args is not None
        else 256
    )
    num_iterations = (
        int(getattr(args, "autogen_qmppi_num_iterations", 6)) if args is not None else 6
    )
    elite_ratio = (
        float(getattr(args, "autogen_qmppi_elite_ratio", 0.1))
        if args is not None
        else 0.1
    )
    temperature = (
        float(getattr(args, "autogen_qmppi_temperature", 1.0))
        if args is not None
        else 1.0
    )
    q_noise_scale = (
        float(getattr(args, "autogen_qmppi_q_noise_scale", 0.05))
        if args is not None
        else 0.05
    )
    tracking_weight = (
        float(getattr(args, "autogen_qmppi_tracking_weight", 1.0))
        if args is not None
        else 1.0
    )
    penetration_weight = (
        float(getattr(args, "autogen_qmppi_penetration_weight", 1000.0))
        if args is not None
        else 1000.0
    )
    pos_weight = (
        float(getattr(args, "autogen_qmppi_pos_weight", 200.0))
        if args is not None
        else 200.0
    )
    rot_weight = (
        float(getattr(args, "autogen_qmppi_rot_weight", 20.0))
        if args is not None
        else 20.0
    )
    g_noise_scale = (
        float(getattr(args, "autogen_qmppi_gripper_noise_scale", 0.015))
        if args is not None
        else 0.015
    )
    g_tracking_weight = (
        float(getattr(args, "autogen_qmppi_gripper_tracking_weight", 50.0))
        if args is not None
        else 50.0
    )
    g_min = (
        float(getattr(args, "autogen_qmppi_gripper_min", 0.0))
        if args is not None
        else 0.0
    )
    g_max = (
        float(getattr(args, "autogen_qmppi_gripper_max", 0.08))
        if args is not None
        else 0.08
    )
    seed = (
        int(getattr(args, "autogen_qmppi_seed", getattr(args, "seed", 0)))
        if args is not None
        else 0
    )

    cartesian_mode = target_pos is not None
    if cartesian_mode:
        target_pos_arr = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_rot_arr = (
            np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
            if target_rot is not None
            else None
        )
        if ee_site_name is None:
            raise ValueError("solve_arm_q_mppi cartesian mode requires ee_site_name")

    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    model = env.sim.model
    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)

    arm_qpos_addr = np.array(
        [int(model.get_joint_qpos_addr(name)) for name in arm_joint_names],
        dtype=np.int64,
    )
    drawer_addr = None
    if drawer_q is not None:
        drawer_obj = getattr(env, "drawer", None)
        if drawer_obj is not None and len(drawer_obj.door_joint_names) > 0:
            drawer_addr = int(model.get_joint_qpos_addr(drawer_obj.door_joint_names[0]))

    site_id = None
    if cartesian_mode:
        site_id = int(model.site_name2id(str(ee_site_name)))

    # Gripper DoF lookup (full opening g → g/2 per finger joint).
    gripper_qpos_addr = None
    g_init = None
    use_gripper = bool(optimize_gripper)
    if use_gripper:
        try:
            from robocasa.demos.franka_collision_model import (
                gripper_joint_names_for_q_mpc,
            )

            g_joint_names = gripper_joint_names_for_q_mpc(env)
        except Exception:
            g_joint_names = ()
        addrs = []
        for name in g_joint_names:
            try:
                addrs.append(int(model.get_joint_qpos_addr(name)))
            except Exception:
                pass
        if len(addrs) == 0:
            use_gripper = False
        else:
            gripper_qpos_addr = np.asarray(addrs, dtype=np.int64)
            g_init = (
                float(initial_gripper_opening)
                if initial_gripper_opening is not None
                else 2.0 * float(env.sim.data.qpos[gripper_qpos_addr[0]])
            )
            g_init = float(np.clip(g_init, g_min, g_max))

    # arm body ids — descendants of panda links
    try:
        from robocasa.demos import visualize_mujoco as viz_mj

        arm_body_ids = viz_mj._ghost_source_body_ids(env)
    except Exception:
        arm_body_ids = set()
    arm_geom_ids = set(
        int(gid)
        for gid in range(int(raw_model.ngeom))
        if int(raw_model.geom_bodyid[gid]) in arm_body_ids
    )

    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()

    rng = np.random.default_rng(int(seed))
    state_dim = 8 if use_gripper else 7
    nominal = np.zeros(state_dim, dtype=np.float64)
    nominal[:7] = q_mink
    if use_gripper:
        nominal[7] = g_init
    best_state = nominal.copy()
    best_cost = float("inf")
    best_pos_err = float("nan")
    best_rot_err = float("nan")
    best_pen = float("nan")

    def _eval(state_vec):
        env.sim.data.qpos[arm_qpos_addr] = state_vec[:7]
        if use_gripper:
            env.sim.data.qpos[gripper_qpos_addr] = 0.5 * float(state_vec[7])
        if drawer_addr is not None:
            env.sim.data.qpos[drawer_addr] = float(drawer_q)
        env.sim.forward()
        ncon = int(raw_data.ncon)
        max_pen = 0.0
        for ci in range(ncon):
            c = raw_data.contact[ci]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if g1 in arm_geom_ids or g2 in arm_geom_ids:
                d = float(c.dist)
                if d < 0.0 and (-d) > max_pen:
                    max_pen = -d
        pos_err = 0.0
        rot_err = 0.0
        if cartesian_mode:
            site_pos = np.asarray(
                raw_data.site_xpos[site_id], dtype=np.float64
            ).reshape(3)
            pos_err = float(np.linalg.norm(site_pos - target_pos_arr))
            if target_rot_arr is not None:
                site_rot = np.asarray(
                    raw_data.site_xmat[site_id], dtype=np.float64
                ).reshape(3, 3)
                trace = float(np.trace(target_rot_arr.T @ site_rot))
                rot_err = float(np.arccos(np.clip(0.5 * (trace - 1.0), -1.0, 1.0)))
        return max_pen, pos_err, rot_err

    try:
        for iter_idx in range(num_iterations):
            anneal = (1.0 - iter_idx / max(1, num_iterations - 1)) * 0.99 + 0.01
            q_scale = q_noise_scale * anneal
            perturb = np.zeros((num_samples, state_dim), dtype=np.float64)
            perturb[:, :7] = rng.normal(0.0, q_scale, size=(num_samples, 7))
            if use_gripper:
                perturb[:, 7] = rng.normal(
                    0.0, g_noise_scale * anneal, size=num_samples
                )
            candidates = nominal[None, :] + perturb
            if use_gripper:
                candidates[:, 7] = np.clip(candidates[:, 7], g_min, g_max)
            # Always include nominal itself.
            candidates[0] = nominal
            costs = np.zeros(num_samples, dtype=np.float64)
            for i in range(num_samples):
                state_c = candidates[i]
                pen, pos_err, rot_err = _eval(state_c)
                if cartesian_mode:
                    tracking = pos_weight * (pos_err * pos_err) + rot_weight * (
                        rot_err * rot_err
                    )
                else:
                    tracking = float(np.sum((state_c[:7] - q_mink) ** 2))
                if use_gripper:
                    tracking += g_tracking_weight * float((state_c[7] - g_init) ** 2)
                costs[i] = tracking_weight * tracking + penetration_weight * (pen * pen)
                if costs[i] < best_cost:
                    best_cost = float(costs[i])
                    best_state = state_c.copy()
                    best_pos_err = float(pos_err)
                    best_rot_err = float(rot_err)
                    best_pen = float(pen)
            # Elite softmax update
            elite_count = max(1, int(np.ceil(elite_ratio * num_samples)))
            elite_idx = np.argsort(costs)[:elite_count]
            elite_costs = costs[elite_idx]
            elite_cands = candidates[elite_idx]
            mu = float(np.mean(elite_costs))
            sigma = float(np.std(elite_costs)) + 1e-2
            normalized = (elite_costs - mu) / sigma
            weights = np.exp(-normalized / max(temperature, 1e-6))
            weights /= max(float(weights.sum()), 1e-12)
            nominal = np.sum(weights[:, None] * elite_cands, axis=0)
            if use_gripper:
                nominal[7] = float(np.clip(nominal[7], g_min, g_max))
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()

    best_q_arr = np.asarray(best_state[:7], dtype=np.float64).reshape(7)
    best_g = float(best_state[7]) if use_gripper else None
    if cartesian_mode:
        return (
            best_q_arr,
            best_g,
            float(best_pos_err),
            float(best_rot_err),
            float(best_pen),
        )
    return best_q_arr


__all__.append("solve_arm_q_mppi")
