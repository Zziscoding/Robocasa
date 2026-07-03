"""Parallel Comfree rollouts for push / grasp feasibility scoring.

Two branches:

``SimPush``
    Drives a stripped floating-hand + sliding-object model along a candidate
    contact point.  Two entry points:

    * ``rollout_point`` — for each feasible contact point apply an optimal 3D
      force at that point on the object body, step the parallel comfree env,
      and score the resulting object motion.
    * ``rollout_ee`` — mimic ``FloatingEERollout``: drive the floating EE along
      the approach direction and score the object-cost profile.

``SimGrasp``
    Placeholder for the prehensile branch (TODO).

The stripped model construction and backend helpers are reused from
``ee_floating_mppi.py`` (already imported by ``rollout.py``).
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import mujoco
import numpy as np
import torch

# Reuse the stripped floating-hand model construction and backend helpers from
# ee_floating_mppi. They are private (underscore-prefixed) but stable.
from robocasa.demos.ee_floating_mppi import (  # noqa: E402
    _backend_tensor,
    _build_floating_ee_mjcf,
    _import_comfree_backend,
    _import_mjwarp_backend,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Shared config for SimPush / SimGrasp."""

    device: str = "cuda:0"
    # Parallel simulation dimensions.
    n_sim: int = 64
    n_horizon: int = 30
    sim_dt: float = 0.05
    # Contact model.
    contact_stiffness: float = 0.2
    contact_damping: float = 0.001
    nconmax_per_env: int = 120
    njmax_per_env: int = 500
    compile_cuda_graph: bool = False
    prefer_comfree: bool = True
    # --- score formula weights (mirror RolloutConfig) -----------------------
    # score = Σ γ^t · progress[t]  +  λ · late_min_bonus  −  μ · rebound
    score_gamma: float = 0.95
    score_late_weight: float = 5.0
    score_rebound_weight: float = 10.0
    score_accept_threshold: float = 0.15

    def post_init(self) -> None:
        assert self.n_sim >= 1
        assert self.n_horizon >= 1
        assert self.sim_dt > 0.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointRolloutResult:
    """Per-point result from :meth:`SimPush.rollout_point`.

    Arrays are shaped ``(n_sim, ...)`` — one entry per parallel world.
    """

    # Object-cost profile: (n_sim, n_horizon)
    object_costs: np.ndarray
    # Object positions: (n_sim, n_horizon, 3)
    object_positions: np.ndarray
    # Max penetration per step: (n_sim, n_horizon)
    penetrations: np.ndarray
    # Per-world summary.
    object_cost_min: np.ndarray  # (n_sim,)
    object_cost_step0: np.ndarray  # (n_sim,)
    object_cost_final: np.ndarray  # (n_sim,)
    object_cost_delta: np.ndarray  # (n_sim,)
    max_penetration: np.ndarray  # (n_sim,)
    # Composite score per world: (n_sim,)
    score: np.ndarray
    score_normalized: np.ndarray  # (n_sim,)
    # Diagnostics: (n_sim,)
    score_progress_sum: np.ndarray
    score_late_min_bonus: np.ndarray
    score_rebound: np.ndarray


@dataclass(frozen=True)
class EERolloutResult:
    """Per-world result from :meth:`SimPush.rollout_ee`.

    Identical layout to :class:`PointRolloutResult`.
    """

    object_costs: np.ndarray
    object_positions: np.ndarray
    penetrations: np.ndarray
    object_cost_min: np.ndarray
    object_cost_step0: np.ndarray
    object_cost_final: np.ndarray
    object_cost_delta: np.ndarray
    max_penetration: np.ndarray
    score: np.ndarray
    score_normalized: np.ndarray
    score_progress_sum: np.ndarray
    score_late_min_bonus: np.ndarray
    score_rebound: np.ndarray


# ---------------------------------------------------------------------------
# SimPush
# ---------------------------------------------------------------------------


class SimPush:
    """Parallel comfree rollouts for the *push* branch.

    Built once per scene (hand subtree + object subtree stripped from the env),
    then the two rollout entry points can be called repeatedly without
    rebuilding the backend.

    ``rollout_point`` — apply an optimal 3D force at each feasible contact
    point and score the resulting object motion.

    ``rollout_ee`` — drive the floating EE along the approach direction
    (mimicking ``FloatingEERollout``) in parallel and score.
    """

    def __init__(
        self,
        env,
        *,
        hand_xml_path: str | None,
        finger_geom_names: Sequence[str],
        object_body_id: int,
        ee_site_name: str,
        config: SimConfig | None = None,
        approach_world: np.ndarray | None = None,
        target_object_position: np.ndarray | None = None,
    ) -> None:
        self.config = config or SimConfig()
        self.config.post_init()
        self.env = env
        self.ee_site_name = str(ee_site_name)
        self.object_body_id = int(object_body_id)

        if approach_world is None:
            self.approach_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.approach_world = (
                np.asarray(approach_world, dtype=np.float64).reshape(3).copy()
            )
            n = float(np.linalg.norm(self.approach_world))
            if n > 1e-9:
                self.approach_world /= n

        if target_object_position is None:
            # Default: use the object's current position as target (cost = 0
            # at start).  Callers should override with the actual goal.
            self.target_object_position = None  # resolved in _build
        else:
            self.target_object_position = np.asarray(
                target_object_position, dtype=np.float64
            ).reshape(3)

        self._build_stripped_model(env, hand_xml_path, finger_geom_names)
        self._init_backend()

    # ----- model construction -----------------------------------------------

    def _build_stripped_model(
        self,
        env,
        hand_xml_path: str | None,
        finger_geom_names: Sequence[str],
    ) -> None:
        """Build a minimal floating-hand + sliding-object contact model."""
        del hand_xml_path, finger_geom_names
        raw_model = getattr(env.sim.model, "_model", env.sim.model)
        raw_data = getattr(env.sim.data, "_data", env.sim.data)

        # Resolve target_object_position from current object pos if not given.
        if self.target_object_position is None:
            self.target_object_position = np.asarray(
                raw_data.xpos[self.object_body_id], dtype=np.float64
            ).reshape(3)

        self.model_cpu = mujoco.MjModel.from_xml_string(
            _build_floating_ee_mjcf(
                env,
                raw_model,
                raw_data,
                self.ee_site_name,
                self.object_body_id,
                self.approach_world,
                float(self.config.sim_dt),
            )
        )
        self.object_body_id = int(
            mujoco.mj_name2id(
                self.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "floating_object"
            )
        )
        hand_body_id = int(
            mujoco.mj_name2id(self.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "floating_hand")
        )
        free_jnt = int(
            mujoco.mj_name2id(
                self.model_cpu, mujoco.mjtObj.mjOBJ_JOINT, "floating_hand_freejoint"
            )
        )
        if self.object_body_id < 0 or hand_body_id < 0 or free_jnt < 0:
            raise ValueError("SimPush minimal model is missing required bodies/joints")
        self._hand_free_qaddr = int(self.model_cpu.jnt_qposadr[free_jnt])
        self._hand_free_daddr = int(self.model_cpu.jnt_dofadr[free_jnt])
        self._finger_qaddrs = np.zeros(0, dtype=np.int64)
        self._object_slide_qaddr = int(
            self.model_cpu.jnt_qposadr[
                mujoco.mj_name2id(
                    self.model_cpu, mujoco.mjtObj.mjOBJ_JOINT, "floating_object_slide"
                )
            ]
        )
        self._active_hand_geoms = np.array(
            [
                gid
                for gid in range(int(self.model_cpu.ngeom))
                if int(self.model_cpu.geom_bodyid[gid]) == hand_body_id
            ],
            dtype=np.int64,
        )
        self._active_object_geoms = np.array(
            [
                gid
                for gid in range(int(self.model_cpu.ngeom))
                if int(self.model_cpu.geom_bodyid[gid]) == self.object_body_id
            ],
            dtype=np.int64,
        )

        self.model_cpu.opt.timestep = float(self.config.sim_dt)
        self.data_cpu = mujoco.MjData(self.model_cpu)

    # ----- backend init -----------------------------------------------------

    def _init_backend(self) -> None:
        cfg = self.config
        nworld = int(cfg.n_sim)
        self.nworld = nworld
        backend_kind = None

        if cfg.prefer_comfree:
            try:
                self.wp, self.cfwarp, step_fn = _import_comfree_backend()
                self.model = self.cfwarp.put_model(
                    self.model_cpu,
                    comfree_stiffness=float(cfg.contact_stiffness),
                    comfree_damping=float(cfg.contact_damping),
                )
                self.data = self.cfwarp.put_data(
                    self.model_cpu,
                    self.data_cpu,
                    nworld=nworld,
                    nconmax=int(cfg.nconmax_per_env),
                    njmax=int(cfg.njmax_per_env),
                )
                self._step_fn = lambda: step_fn(self.model, self.data)
                self._forward_fn = lambda: self.cfwarp.forward(self.model, self.data)
                backend_kind = "comfree"
            except Exception as exc:
                sys.stderr.write(
                    f"[SimPush] Comfree backend unavailable, "
                    f"falling back to native mjwarp: {exc!r}\n"
                )
                sys.stderr.flush()

        if backend_kind is None:
            self.wp, mjwarp, step_fn = _import_mjwarp_backend()
            self.cfwarp = mjwarp
            self.model = mjwarp.put_model(self.model_cpu)
            self.data = mjwarp.put_data(
                self.model_cpu,
                self.data_cpu,
                nworld=nworld,
                nconmax=int(cfg.nconmax_per_env),
                njmax=int(cfg.njmax_per_env),
            )
            self._step_fn = lambda: step_fn(self.model, self.data)
            self._forward_fn = lambda: mjwarp.forward(self.model, self.data)
            backend_kind = "mjwarp"

        self.backend_kind = backend_kind
        self.torch_device = torch.device(cfg.device)
        self._step_graph = None

    # ----- state IO helpers -------------------------------------------------

    def _write_mocap_pose(
        self,
        positions: torch.Tensor,  # (nworld, 3)
        quats: torch.Tensor,  # (nworld, 4) wxyz
        gripper: torch.Tensor,  # (nworld,)
        *,
        reset_object: bool = False,
    ) -> None:
        """Overwrite the freejoint qpos for all worlds."""
        qpos = _backend_tensor(self.data.qpos)
        a = self._hand_free_qaddr
        qpos[:, a : a + 3] = positions
        qpos[:, a + 3 : a + 7] = quats
        if reset_object and hasattr(self, "_object_slide_qaddr"):
            qpos[:, int(self._object_slide_qaddr)] = 0.0
        qvel = _backend_tensor(self.data.qvel)
        if reset_object:
            qvel.zero_()
        elif hasattr(self, "_hand_free_daddr"):
            d = int(self._hand_free_daddr)
            qvel[:, d : d + 6] = 0.0

    def _read_object_position(self) -> torch.Tensor:
        """Return (nworld, 3) object positions."""
        xpos = _backend_tensor(self.data.xpos)
        return xpos[:, self.object_body_id, :].clone()

    def _read_penetration(self) -> torch.Tensor:
        """Max penetration (m) per world across hand/object contacts.

        Returns a tensor of shape ``(nworld,)``.
        """
        contact = self.data.contact
        ncon = int(getattr(self.data, "ncon", 0)) if hasattr(self.data, "ncon") else 0
        if ncon <= 0:
            return torch.zeros(self.nworld, device=self.torch_device)
        try:
            dist = _backend_tensor(contact.dist)
            if hasattr(contact, "geom1") and hasattr(contact, "geom2"):
                g1 = _backend_tensor(contact.geom1).long()
                g2 = _backend_tensor(contact.geom2).long()
            else:
                geom = _backend_tensor(contact.geom).long().reshape(-1, 2)
                g1 = geom[:, 0]
                g2 = geom[:, 1]
            worldid = _backend_tensor(
                getattr(contact, "worldid", getattr(contact, "world_id", None))
            ).long()
        except Exception:
            return torch.zeros(self.nworld, device=self.torch_device)

        hand_geoms = torch.as_tensor(
            self._active_hand_geoms, device=dist.device, dtype=g1.dtype
        )
        object_geoms = torch.as_tensor(
            self._active_object_geoms, device=dist.device, dtype=g1.dtype
        )
        g1_hand = torch.isin(g1, hand_geoms)
        g2_hand = torch.isin(g2, hand_geoms)
        g1_object = torch.isin(g1, object_geoms)
        g2_object = torch.isin(g2, object_geoms)
        hand_object_pair = (g1_hand & g2_object) | (g2_hand & g1_object)
        pen = torch.clamp(-dist - 1e-3, min=0.0) * hand_object_pair.float()
        out = torch.zeros(self.nworld, device=pen.device, dtype=pen.dtype)
        out.scatter_reduce_(0, worldid, pen, reduce="amax", include_self=True)
        return out

    # ----- score computation ------------------------------------------------

    @staticmethod
    def _compute_scores(
        object_costs: np.ndarray,  # (n_sim, n_horizon)
        cfg: SimConfig,
    ) -> dict[str, np.ndarray]:
        """Vectorised score computation over all worlds.

        Returns a dict of ``(n_sim,)`` arrays.
        """
        n_sim, n_horizon = object_costs.shape
        step0 = object_costs[:, 0]  # (n_sim,)
        obj_min = object_costs[:, 1:].min(axis=1) if n_horizon > 1 else step0.copy()
        obj_final = object_costs[:, -1]
        obj_delta = obj_min - step0

        # progress[t] = cost[0] - cost[t]
        progress = step0[:, None] - object_costs  # (n_sim, n_horizon)
        discount = cfg.score_gamma ** np.arange(n_horizon, dtype=np.float64)
        progress_sum = (progress * discount[None, :]).sum(axis=1)  # (n_sim,)

        half = max(n_horizon // 2, 1)
        late_min = object_costs[:, half:].min(axis=1) if n_horizon > 1 else step0.copy()
        late_min_bonus = step0 - late_min
        rebound = np.maximum(0.0, obj_final - late_min)

        score = (
            progress_sum
            + float(cfg.score_late_weight) * late_min_bonus
            - float(cfg.score_rebound_weight) * rebound
        )
        denom = np.maximum(step0 * float(n_horizon), 1e-12)
        score_normalized = score / denom

        return {
            "object_cost_min": obj_min,
            "object_cost_step0": step0,
            "object_cost_final": obj_final,
            "object_cost_delta": obj_delta,
            "score": score,
            "score_normalized": score_normalized,
            "score_progress_sum": progress_sum,
            "score_late_min_bonus": late_min_bonus,
            "score_rebound": rebound,
        }

    # ----- rollout_point ----------------------------------------------------

    @torch.no_grad()
    def rollout_point(
        self,
        feasible_points: np.ndarray,
        optimal_forces: np.ndarray,
        normals: np.ndarray | None = None,
        tangentials: np.ndarray | None = None,
    ) -> PointRolloutResult:
        """Apply optimal 3D forces at feasible contact points and score.

        Each of the ``n_sim`` parallel worlds is assigned one feasible contact
        point + optimal force pair.  The force is applied as a body wrench on
        the object body (world-frame force + ``r × F`` torque) at every sim
        step, and the resulting object-cost profile is scored.

        Args:
            feasible_points: ``(n_sim, 3)`` contact points in world frame.
            optimal_forces: ``(n_sim, 3)`` optimal 3D forces in world frame.
            normals: ``(n_sim, 3)`` contact normals (unused in the rollout
                itself, but accepted for API symmetry with the planner).
            tangentials: ``(n_sim, 3)`` contact tangents (same).

        Returns:
            :class:`PointRolloutResult` with per-world scores.
        """
        del normals, tangentials  # accepted for API symmetry
        cfg = self.config
        n_sim = int(cfg.n_sim)
        n_horizon = int(cfg.n_horizon)

        points = np.asarray(feasible_points, dtype=np.float64).reshape(n_sim, 3)
        forces = np.asarray(optimal_forces, dtype=np.float64).reshape(n_sim, 3)

        # Build the body wrench: [force(3), torque(3)] per world.
        # Torque = (point - body_pos) × force.
        body_pos_0 = np.asarray(
            self.data_cpu.xpos[self.object_body_id], dtype=np.float64
        ).reshape(3)
        arms = points - body_pos_0  # (n_sim, 3)
        torques = np.cross(arms, forces)  # (n_sim, 3)
        wrench = np.concatenate([forces, torques], axis=-1).astype(
            np.float32
        )  # (n_sim, 6)

        # Reset object slide qpos and qvel to zero for all worlds.
        qpos = _backend_tensor(self.data.qpos)
        qvel = _backend_tensor(self.data.qvel)
        qpos[:, int(self._object_slide_qaddr)] = 0.0
        qvel.zero_()

        # Upload the wrench to all worlds.
        xfrc = _backend_tensor(self.data.xfrc_applied)
        xfrc.zero_()
        xfrc[:, self.object_body_id, :] = torch.as_tensor(
            wrench, device=xfrc.device, dtype=xfrc.dtype
        )

        target = self.target_object_position  # (3,)

        def _cost_torch(pos: torch.Tensor) -> torch.Tensor:
            """pos: (nworld, 3) -> (nworld,) squared distance to target."""
            d = pos - torch.as_tensor(target, device=pos.device, dtype=pos.dtype)
            return (d * d).sum(dim=-1)

        object_costs = np.zeros((n_sim, n_horizon), dtype=np.float64)
        object_positions = np.zeros((n_sim, n_horizon, 3), dtype=np.float64)
        penetrations = np.zeros((n_sim, n_horizon), dtype=np.float64)

        for j in range(n_horizon):
            if j == 0:
                self._forward_fn()
            else:
                self._step_fn()
            obj_pos = self._object_pos_to_numpy()
            object_positions[:, j, :] = obj_pos
            penetrations[:, j] = (
                self._read_penetration().detach().cpu().numpy().astype(np.float64)
            )
            obj_pos_t = torch.as_tensor(
                obj_pos, device=self.torch_device, dtype=torch.float32
            )
            cost_t = _cost_torch(obj_pos_t)
            object_costs[:, j] = cost_t.detach().cpu().numpy().astype(np.float64)

        scores = self._compute_scores(object_costs, cfg)
        max_pen = penetrations.max(axis=1)  # (n_sim,)

        return PointRolloutResult(
            object_costs=object_costs,
            object_positions=object_positions,
            penetrations=penetrations,
            max_penetration=max_pen,
            **scores,
        )

    # ----- rollout_ee -------------------------------------------------------

    @torch.no_grad()
    def rollout_ee(
        self,
        refined_xyz: np.ndarray,
        refined_rot: np.ndarray,
        refined_gripper: float,
        approach_distance: float | None = None,
    ) -> EERolloutResult:
        """Drive the floating EE along the approach direction and score.

        Mimics ``FloatingEERollout.run`` but runs ``n_sim`` worlds in parallel.
        All worlds share the same refined pose; the per-step mocap pose is
        ``refined_xyz + j * step_dist * approach_world``.

        Args:
            refined_xyz: EE-site world position ``(3,)``.
            refined_rot: EE-site world rotation matrix ``(3, 3)``.
            refined_gripper: gripper opening (m).
            approach_distance: total push distance.  Defaults to
                ``0.15`` (matching ``RolloutConfig.approach_total_distance``).

        Returns:
            :class:`EERolloutResult` with per-world scores.
        """
        cfg = self.config
        n_sim = int(cfg.n_sim)
        n_horizon = int(cfg.n_horizon)

        refined_xyz = np.asarray(refined_xyz, dtype=np.float64).reshape(3)
        refined_rot = np.asarray(refined_rot, dtype=np.float64).reshape(3, 3)
        refined_gripper = float(refined_gripper)

        quat_wxyz = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_wxyz, refined_rot.reshape(9))

        if approach_distance is None:
            approach_distance = 0.15
        step_dist = approach_distance / max(1, n_horizon)

        # Broadcast the same pose to all worlds.
        positions_base = (
            torch.as_tensor(
                refined_xyz.astype(np.float32),
                device=self.torch_device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(n_sim, -1)
        )
        quats_base = (
            torch.as_tensor(
                quat_wxyz.astype(np.float32),
                device=self.torch_device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(n_sim, -1)
        )
        gripper_base = torch.full(
            (n_sim,), refined_gripper, device=self.torch_device, dtype=torch.float32
        )

        approach_t = torch.as_tensor(
            self.approach_world.astype(np.float32),
            device=self.torch_device,
            dtype=torch.float32,
        )

        target = self.target_object_position  # (3,)

        def _cost_torch(pos: torch.Tensor) -> torch.Tensor:
            d = pos - torch.as_tensor(target, device=pos.device, dtype=pos.dtype)
            return (d * d).sum(dim=-1)

        object_costs = np.zeros((n_sim, n_horizon), dtype=np.float64)
        object_positions = np.zeros((n_sim, n_horizon, 3), dtype=np.float64)
        penetrations = np.zeros((n_sim, n_horizon), dtype=np.float64)

        for j in range(n_horizon):
            offset = (float(j) * step_dist) * approach_t  # (3,)
            pos_j = positions_base + offset[None, :]
            self._write_mocap_pose(
                pos_j,
                quats_base,
                gripper_base,
                reset_object=(j == 0),
            )
            if j == 0:
                self._forward_fn()
            else:
                self._step_fn()
            obj_pos = self._object_pos_to_numpy()
            object_positions[:, j, :] = obj_pos
            penetrations[:, j] = (
                self._read_penetration().detach().cpu().numpy().astype(np.float64)
            )
            obj_pos_t = torch.as_tensor(
                obj_pos, device=self.torch_device, dtype=torch.float32
            )
            cost_t = _cost_torch(obj_pos_t)
            object_costs[:, j] = cost_t.detach().cpu().numpy().astype(np.float64)

        scores = self._compute_scores(object_costs, cfg)
        max_pen = penetrations.max(axis=1)  # (n_sim,)

        return EERolloutResult(
            object_costs=object_costs,
            object_positions=object_positions,
            penetrations=penetrations,
            max_penetration=max_pen,
            **scores,
        )

    # ----- convenience helpers ----------------------------------------------

    def _object_pos_to_numpy(self) -> np.ndarray:
        """Read object position as ``(nworld, 3)`` float64 numpy array."""
        xpos = _backend_tensor(self.data.xpos)
        return xpos[:, self.object_body_id, :].detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# SimGrasp (placeholder)
# ---------------------------------------------------------------------------


class SimGrasp:
    """Placeholder for the prehensile (grasp) branch.

    TODO: implement parallel grasp feasibility scoring.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "SimGrasp is a placeholder and has not been implemented yet."
        )


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

__all__ = [
    "SimConfig",
    "PointRolloutResult",
    "EERolloutResult",
    "SimPush",
    "SimGrasp",
]
