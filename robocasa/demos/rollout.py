"""Approach rollout for a refined floating EE pose.

After :class:`FloatingEEMPPI` (``ee_floating_mppi.py``) accepts a single-step
refined EE pose, this module evaluates whether actually driving that pose along
the approach direction improves the object configuration. It reimplements the
"translate along the optimal force direction and report object-cost" loop that
previously lived inside ``FloatingEEMPPI.solve``.

The hand+fingers are treated as a floating mocap-driven body in a stripped scene
that contains only the target object subtree and the hand/finger ghost geoms.
We advance ``horizon_steps - 1`` approach steps of uniform size along
``approach_world`` from the refined pose and track the object-cost
(||object_pos - target||^2) and penetration profile across the rollout.

Typical usage from a caller such as ``_solve_contact_poses_with_skeleton`` in
``demos/demo_close_drawer_contact_curobo.py``::

    rollout = FloatingEERollout(
        env,
        hand_xml_path=None,
        finger_geom_names=(),
        object_body_id=drawer_body_id,
        ee_site_name=frame_name,
        config=RolloutConfig(horizon_steps=5, approach_total_distance=0.01),
        approach_world=force_direction,
        target_object_position=target_drawer_pos,
    )
    r = rollout.run(refined_xyz, refined_rot_matrix, refined_gripper)
    # r.object_cost_delta < -eps   -> push helps, accept the candidate
    # r.object_cost_delta >= -eps  -> push does not help, reject the candidate
"""

from __future__ import annotations

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
# Config / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RolloutConfig:
    device: str = "cuda:0"
    sim_dt: float = 0.005
    horizon_steps: int = 5  # 1 hold (the refined pose) + 4 approach steps
    approach_total_distance: float = 0.01
    object_improvement_eps: float = 1e-5
    contact_stiffness: float = 0.2
    contact_damping: float = 0.001
    nconmax_per_env: int = 120
    njmax_per_env: int = 500
    compile_cuda_graph: bool = False
    prefer_comfree: bool = True

    def post_init(self) -> None:
        assert self.horizon_steps >= 1


@dataclass(frozen=True)
class RolloutResult:
    # Object-cost profile over the (single-world) rollout, indexed by step.
    # object_costs[0] is the hold step at the refined pose; object_costs[j] for
    # j > 0 is after the j-th approach micro-step.
    object_costs: np.ndarray  # (horizon_steps,) float64
    object_positions: np.ndarray  # (horizon_steps, 3) float64, world frame
    penetrations: np.ndarray  # (horizon_steps,) float64
    # Summary: dynamic-min over j>=1 of object_costs[j] minus the hold cost.
    # Negative means the push drove the drawer closer to the target.
    object_cost_min: float
    object_cost_step0: float
    object_cost_final: float
    object_cost_delta: float  # object_cost_min - object_cost_step0
    max_penetration: float
    steps: int


# ---------------------------------------------------------------------------
# FloatingEERollout
# ---------------------------------------------------------------------------


class FloatingEERollout:
    """Rollout evaluates a single refined EE pose along the approach direction.

    Built once per scene (hand subtree + object subtree stripped from the env),
    then :meth:`run` can be called repeatedly for different refined poses without
    rebuilding the backend. The rollout uses a single simulation world.
    """

    def __init__(
        self,
        env,
        *,
        hand_xml_path: str | None,
        finger_geom_names: Sequence[str],
        object_body_id: int,
        ee_site_name: str,
        config: RolloutConfig | None = None,
        approach_world: np.ndarray,
        target_object_position: np.ndarray,
        selected_contact_point_world: np.ndarray | None = None,
        drawer_qpos_addr: int | None = None,
        drawer_qpos_value: float | None = None,
    ) -> None:
        self.config = config or RolloutConfig()
        self.config.post_init()
        self.env = env
        self.ee_site_name = str(ee_site_name)
        self.object_body_id = int(object_body_id)
        self.approach_world = (
            np.asarray(approach_world, dtype=np.float64).reshape(3).copy()
        )
        n = float(np.linalg.norm(self.approach_world))
        if n > 1e-9:
            self.approach_world /= n
        self.target_object_position = np.asarray(
            target_object_position, dtype=np.float64
        ).reshape(3)
        if selected_contact_point_world is None:
            self.selected_contact_point_world = None
        else:
            self.selected_contact_point_world = (
                np.asarray(selected_contact_point_world, dtype=np.float64)
                .reshape(3)
                .copy()
            )
        self.drawer_qpos_addr = drawer_qpos_addr
        self.drawer_qpos_value = drawer_qpos_value

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
            raise ValueError(
                "FloatingEERollout minimal model is missing required bodies/joints"
            )
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
        self.drawer_qpos_addr = None
        self.drawer_qpos_value = None
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
        nworld = 1
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
                    nconmax=nworld * cfg.nconmax_per_env,
                    njmax=nworld * cfg.njmax_per_env,
                )
                self._step_fn = lambda: step_fn(self.model, self.data)
                self._forward_fn = lambda: self.cfwarp.forward(self.model, self.data)
                backend_kind = "comfree"
            except Exception as exc:
                sys.stderr.write(
                    f"[rollout] Comfree backend unavailable, "
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
                nconmax=nworld * cfg.nconmax_per_env,
                njmax=nworld * cfg.njmax_per_env,
            )
            self._step_fn = lambda: step_fn(self.model, self.data)
            self._forward_fn = lambda: mjwarp.forward(self.model, self.data)
            backend_kind = "mjwarp"
        self.backend_kind = backend_kind
        self.torch_device = torch.device(cfg.device)
        self._step_graph = None

    # ----- single-world state IO -------------------------------------------

    def _write_mocap_pose(
        self,
        position: np.ndarray,
        quat_wxyz: np.ndarray,
        gripper: float,
        *,
        reset_object: bool = False,
    ) -> None:
        """Overwrite the one freejoint + finger qpos for the single world."""
        qpos = _backend_tensor(self.data.qpos)
        a = self._hand_free_qaddr
        qpos[0, a : a + 3] = torch.as_tensor(
            np.asarray(position, dtype=np.float32).reshape(3),
            device=qpos.device,
            dtype=qpos.dtype,
        )
        qpos[0, a + 3 : a + 7] = torch.as_tensor(
            np.asarray(quat_wxyz, dtype=np.float32).reshape(4),
            device=qpos.device,
            dtype=qpos.dtype,
        )
        if self.drawer_qpos_addr is not None and self.drawer_qpos_value is not None:
            qpos[0, int(self.drawer_qpos_addr)] = float(self.drawer_qpos_value)
        if reset_object and hasattr(self, "_object_slide_qaddr"):
            qpos[0, int(self._object_slide_qaddr)] = 0.0
            qvel = _backend_tensor(self.data.qvel)
            qvel.zero_()

    def _read_object_position(self) -> np.ndarray:
        xpos = _backend_tensor(self.data.xpos)
        return (
            xpos[0, self.object_body_id, :]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .reshape(3)
        )

    def _read_penetration(self) -> float:
        """Max penetration (m) over the single world's hand/object contacts."""
        contact = self.data.contact
        ncon = int(getattr(self.data, "ncon", 0)) if hasattr(self.data, "ncon") else 0
        if ncon <= 0:
            return 0.0
        try:
            dist = _backend_tensor(contact.dist)
            if hasattr(contact, "geom1") and hasattr(contact, "geom2"):
                g1 = _backend_tensor(contact.geom1).long()
                g2 = _backend_tensor(contact.geom2).long()
            else:
                geom = _backend_tensor(contact.geom).long().reshape(-1, 2)
                g1 = geom[:, 0]
                g2 = geom[:, 1]
        except Exception:
            return 0.0
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
        return float(pen.max().item())

    # ----- main rollout -----------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        refined_xyz: np.ndarray,
        refined_rot: np.ndarray,
        refined_gripper: float,
    ) -> RolloutResult:
        """Drive ``refined_xyz`` (world) along ``approach_world`` and report cost.

        Args:
            refined_xyz: EE-site world position (3,) accepted by MPPI.
            refined_rot: EE-site world rotation matrix (3, 3).
            refined_gripper: gripper opening (m).

        Returns:
            :class:`RolloutResult` with the per-step object-cost / penetration
            profile and the summary deltas used to decide acceptance.
        """
        cfg = self.config
        refined_xyz = np.asarray(refined_xyz, dtype=np.float64).reshape(3)
        refined_rot = np.asarray(refined_rot, dtype=np.float64).reshape(3, 3)
        refined_gripper = float(refined_gripper)

        quat_wxyz = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_wxyz, refined_rot.reshape(9))

        step_dist = cfg.approach_total_distance / max(1, cfg.horizon_steps - 1)
        target = self.target_object_position

        object_costs = np.zeros(int(cfg.horizon_steps), dtype=np.float64)
        object_positions = np.zeros((int(cfg.horizon_steps), 3), dtype=np.float64)
        penetrations = np.zeros(int(cfg.horizon_steps), dtype=np.float64)

        def _cost(pos: np.ndarray) -> float:
            d = np.asarray(pos, dtype=np.float64).reshape(3) - target
            return float(d.dot(d))

        for j in range(int(cfg.horizon_steps)):
            pos_j = refined_xyz + j * step_dist * self.approach_world
            self._write_mocap_pose(
                pos_j,
                quat_wxyz,
                refined_gripper,
                reset_object=(j == 0),
            )
            if j == 0:
                self._forward_fn()
            else:
                self._step_fn()
            object_positions[j] = self._read_object_position()
            penetrations[j] = self._read_penetration()
            object_costs[j] = _cost(object_positions[j])

        step0 = float(object_costs[0])
        if object_costs.shape[0] > 1:
            obj_min = float(np.min(object_costs[1:]))
        else:
            obj_min = step0
        obj_final = float(object_costs[-1])
        obj_delta = obj_min - step0
        max_pen = float(np.max(penetrations))

        return RolloutResult(
            object_costs=object_costs.copy(),
            object_positions=object_positions.copy(),
            penetrations=penetrations.copy(),
            object_cost_min=obj_min,
            object_cost_step0=step0,
            object_cost_final=obj_final,
            object_cost_delta=obj_delta,
            max_penetration=max_pen,
            steps=int(cfg.horizon_steps),
        )


__all__ = ["RolloutConfig", "RolloutResult", "FloatingEERollout"]
