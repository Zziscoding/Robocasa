"""Reusable rollout costs for Dream/Comfree MPC demos."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import torch


def _backend_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    warp = importlib.import_module("warp")
    return warp.to_torch(value)


def _matrix_to_quat_wxyz_torch(matrix):
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
    quat = torch.stack((qw, qx, qy, qz), dim=-1)
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)


class TaskCost(Protocol):
    def __call__(self, step: Any) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class CompositeRolloutCost:
    costs: Sequence[TaskCost]

    def __call__(self, step: Any) -> torch.Tensor:
        total = torch.zeros(
            step.qpos.shape[0], device=step.qpos.device, dtype=step.qpos.dtype
        )
        for cost in self.costs:
            total = total + cost(step)
        return total


@dataclass
class OpenDrawerPlanOnceCost:
    """Cartesian MPPI cost for pre-contact/contact open-drawer rollouts.

    Contact and object-pose terms are evaluated directly on Comfree rollout
    state, so callers can reuse the same contact model configured in Dream.
    """

    mink_q: np.ndarray
    contact_offset_ee: np.ndarray
    feasible_points_world: np.ndarray
    object_points_world: np.ndarray
    object_body_id: int
    object_target_position: np.ndarray
    geom_set_a: tuple
    geom_set_b: tuple
    q_weight: float
    orientation_weight: float
    contact_weight: float
    object_weight: float
    collision_weight: float
    collision_margin: float
    feasible_distance: float
    _mink_q_tensor: object = None
    _contact_offset_tensor: object = None
    _feasible_tensor: object = None
    _object_tensor: object = None
    _object_target_tensor: object = None
    _mask_a: object = None
    _mask_b: object = None
    _contact_ids: object = None

    def _prepare(self, step):
        device = step.qpos.device
        dtype = step.qpos.dtype
        if (
            self._mink_q_tensor is None
            or self._mink_q_tensor.device != device
            or self._mink_q_tensor.dtype != dtype
        ):
            self._mink_q_tensor = torch.as_tensor(
                np.asarray(self.mink_q, dtype=np.float64).reshape(-1),
                device=device,
                dtype=dtype,
            )
            self._contact_offset_tensor = torch.as_tensor(
                np.asarray(self.contact_offset_ee, dtype=np.float64).reshape(3),
                device=device,
                dtype=dtype,
            )
            self._feasible_tensor = torch.as_tensor(
                np.asarray(self.feasible_points_world, dtype=np.float64).reshape(-1, 3),
                device=device,
                dtype=dtype,
            )
            self._object_tensor = torch.as_tensor(
                np.asarray(self.object_points_world, dtype=np.float64).reshape(-1, 3),
                device=device,
                dtype=dtype,
            )
            self._object_target_tensor = torch.as_tensor(
                np.asarray(self.object_target_position, dtype=np.float64).reshape(3),
                device=device,
                dtype=dtype,
            )
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

    def __call__(self, step):
        self._prepare(step)
        contact_world = step.ee_position + torch.bmm(
            step.ee_rotation,
            self._contact_offset_tensor.view(1, 3, 1).expand(
                step.ee_position.shape[0], -1, -1
            ),
        ).squeeze(-1)

        if self._object_tensor.numel() > 0:
            object_distances = torch.cdist(contact_world, self._object_tensor)
            nearest_object_index = torch.argmin(object_distances, dim=1)
            nearest_object_point = self._object_tensor[nearest_object_index]
        else:
            nearest_object_point = contact_world

        if self._feasible_tensor.numel() > 0:
            feasible_distance = torch.cdist(
                nearest_object_point, self._feasible_tensor
            ).amin(dim=1)
            nearest_feasible = torch.cdist(contact_world, self._feasible_tensor).amin(
                dim=1
            )
            feasible = feasible_distance <= float(self.feasible_distance)
        else:
            nearest_feasible = torch.full(
                (step.qpos.shape[0],),
                torch.inf,
                device=step.qpos.device,
                dtype=step.qpos.dtype,
            )
            feasible = torch.zeros(
                step.qpos.shape[0], device=step.qpos.device, dtype=torch.bool
            )

        q_cost = torch.zeros(
            step.qpos.shape[0], device=step.qpos.device, dtype=step.qpos.dtype
        )
        if (
            step.arm_qpos is not None
            and self._mink_q_tensor.numel() >= step.arm_qpos.shape[1]
        ):
            q_error = step.arm_qpos - self._mink_q_tensor[: step.arm_qpos.shape[1]]
            q_cost = float(self.q_weight) * q_error.square().sum(dim=-1)

        target_rotation = step.target_ee_pose[:, 3:]
        target_rotation = target_rotation / torch.linalg.vector_norm(
            target_rotation, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        ee_quat = _matrix_to_quat_wxyz_torch(step.ee_rotation)
        quat_dot = torch.abs((ee_quat * target_rotation).sum(dim=-1)).clamp(max=1.0)
        orientation_cost = float(self.orientation_weight) * (1.0 - quat_dot).square()

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
        penetration = torch.clamp(-dist - float(self.collision_margin), min=0.0)
        penetration = torch.where(
            active & valid_geom & pair,
            torch.nan_to_num(penetration, nan=0.0),
            torch.zeros_like(penetration),
        )
        collision_cost = torch.zeros(
            step.qpos.shape[0], device=step.qpos.device, dtype=step.qpos.dtype
        )
        collision_cost.scatter_reduce_(
            0,
            world_ids.clamp(0, collision_cost.numel() - 1),
            penetration.square(),
            reduce="amax",
            include_self=True,
        )
        approach_cost = (
            q_cost + orientation_cost + float(self.collision_weight) * collision_cost
        )

        positions = _backend_tensor(step.data.xpos)
        object_position = positions[:, int(self.object_body_id)]
        object_cost = float(self.object_weight) * (
            object_position - self._object_target_tensor
        ).square().sum(dim=-1)
        contact_cost = float(self.contact_weight) * nearest_feasible.square()
        return torch.where(feasible, contact_cost, approach_cost) + object_cost


__all__ = ["CompositeRolloutCost", "OpenDrawerPlanOnceCost"]
