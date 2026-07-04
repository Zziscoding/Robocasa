"""Comfree-based prehensile-need probe.

Given a set of feasible contact points on an articulated surface (drawer face /
handle), spin up a parallel Comfree world per point, apply a constant pull/push
force at that point on the surface body, roll out for a short horizon, and
measure how much the articulated joint actually moves. If the best achievable
joint motion is below threshold, the surface is judged "hard to manipulate
without grasping" and prehensile manipulation is required.

Public API:
    evaluate_prehensile_need(env, surface, feasible_points_world,
                             pull_direction_world, args, *, task_name) -> dict

Returns a dict with at least:
    use_prehens: int   (0 or 1)
    best_displacement: float
    expected_displacement: float
    ratio: float
    per_point_displacement: np.ndarray
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import mujoco  # noqa: F401
except Exception:
    mujoco = None  # type: ignore

from robocasa.demos.dream import _import_comfree_backend


def _drawer_joint_qpos_addr(env) -> int:
    joint_name = env.drawer.door_joint_names[0]
    return int(env.sim.model.get_joint_qpos_addr(joint_name))


def _surface_body_id(env, surface) -> int:
    model = env.sim.model
    geom_name = str(getattr(surface, "geom_name", ""))
    if geom_name in model._geom_name2id:
        return int(model.geom_bodyid[model.geom_name2id(geom_name)])
    return int(model.body_name2id(env.drawer.root_body))


@dataclass
class PrehensProbeConfig:
    horizon_steps: int = 80
    sim_dt: float = 0.01
    applied_force: float = 30.0
    success_ratio: float = 0.3
    contact_stiffness: float = 0.2
    contact_damping: float = 0.001
    device: str = "cuda:0"


def _probe_config_from_args(args) -> PrehensProbeConfig:
    return PrehensProbeConfig(
        horizon_steps=int(getattr(args, "world_model_horizon_steps", 80)),
        sim_dt=float(getattr(args, "world_model_sim_dt", 0.01)),
        applied_force=float(getattr(args, "world_model_applied_force", 30.0)),
        success_ratio=float(getattr(args, "world_model_success_ratio", 0.3)),
        contact_stiffness=float(getattr(args, "world_model_contact_stiffness", 0.2)),
        contact_damping=float(getattr(args, "world_model_contact_damping", 0.001)),
        device=str(getattr(args, "world_model_device", "cuda:0")),
    )


def _print(msg: str) -> None:
    print(f"[world_model] {msg}", file=sys.__stdout__, flush=True)


def evaluate_prehensile_need(
    env,
    surface,
    feasible_points_world: np.ndarray,
    pull_direction_world: Sequence[float],
    args,
    *,
    task_name: str,
    expected_displacement: float | None = None,
) -> dict:
    """Probe whether the task is feasible without grasping.

    Each parallel Comfree world owns one feasible point. A constant body wrench
    is applied at that point along ``pull_direction_world`` (for "open") or its
    opposite (for "close"), and the resulting articulated joint displacement is
    measured at the end of the horizon. The best-achieving world's ratio of
    realized-to-expected displacement determines ``use_prehens``.
    """

    cfg = _probe_config_from_args(args)
    points = np.asarray(feasible_points_world, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        _print("no feasible points; defaulting to prehensile=1")
        return {
            "use_prehens": 1,
            "best_displacement": 0.0,
            "expected_displacement": float(expected_displacement or 0.0),
            "ratio": 0.0,
            "per_point_displacement": np.zeros(0, dtype=np.float64),
        }

    direction = np.asarray(pull_direction_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm
    if str(task_name).strip().lower().startswith("close"):
        direction = -direction
    force_world = direction * float(cfg.applied_force)

    if expected_displacement is None:
        expected_displacement = float(
            getattr(args, "pull_distance", None)
            or getattr(args, "precontact_distance", 0.05)
        )
    expected_displacement = max(float(expected_displacement), 1e-4)

    body_id = _surface_body_id(env, surface)
    qpos_addr = _drawer_joint_qpos_addr(env)
    q0 = float(env.sim.data.qpos[qpos_addr])

    model_cpu = getattr(env.sim.model, "_model", env.sim.model)
    data_cpu = getattr(env.sim.data, "_data", env.sim.data)

    try:
        wp, cfwarp, step_comfree, _ = _import_comfree_backend()
    except Exception as exc:
        _print(f"comfree unavailable ({exc!r}); falling back to serial mujoco probe")
        return _serial_mujoco_probe(
            env, body_id, qpos_addr, q0, points, force_world, expected_displacement, cfg
        )

    nworld = int(points.shape[0])
    original_timestep = float(model_cpu.opt.timestep)
    original_cone = int(model_cpu.opt.cone)
    try:
        model_cpu.opt.timestep = float(cfg.sim_dt)
        if mujoco is not None:
            model_cpu.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
            mujoco.mj_forward(model_cpu, data_cpu)
        wp.init()
        wp.set_device(cfg.device)
        with wp.ScopedDevice(cfg.device):
            model = cfwarp.put_model(
                model_cpu,
                comfree_stiffness=cfg.contact_stiffness,
                comfree_damping=cfg.contact_damping,
            )
            data = cfwarp.put_data(
                model_cpu,
                data_cpu,
                nworld=nworld,
                nconmax=120,
                njmax=500,
            )

            import torch  # local import; dream.py already pulls torch

            xfrc = wp.to_torch(data.xfrc_applied)
            xfrc.zero_()
            # body frame torque from world-frame force at world-frame point:
            #   tau_body = (p - body_xpos) x F
            body_pos = wp.to_torch(data.xpos)[:, body_id].cpu().numpy()
            forces = np.tile(force_world.reshape(1, 3), (nworld, 1))
            arms = points - body_pos
            torques = np.cross(arms, forces)
            wrench = np.concatenate([forces, torques], axis=-1).astype(np.float32)
            xfrc[:, body_id, :] = torch.as_tensor(
                wrench, device=xfrc.device, dtype=xfrc.dtype
            )

            for _ in range(int(cfg.horizon_steps)):
                step_comfree(model, data)
            wp.synchronize()
            qpos = wp.to_torch(data.qpos).cpu().numpy()
            q_end = qpos[:, qpos_addr]
    finally:
        model_cpu.opt.timestep = original_timestep
        model_cpu.opt.cone = original_cone
        if mujoco is not None:
            mujoco.mj_forward(model_cpu, data_cpu)

    displacement = q_end - q0
    # Sign convention: for "open" we want q to increase; for "close" to decrease.
    if str(task_name).strip().lower().startswith("close"):
        signed = -displacement
    else:
        signed = displacement
    best = float(np.max(signed)) if signed.size else 0.0
    ratio = best / expected_displacement
    use_prehens = int(ratio < cfg.success_ratio)
    _print(
        f"task={task_name} points={nworld} best_disp={best:.4f} "
        f"expected={expected_displacement:.4f} ratio={ratio:.3f} "
        f"threshold={cfg.success_ratio:.3f} use_prehens={use_prehens}"
    )
    return {
        "use_prehens": use_prehens,
        "best_displacement": best,
        "expected_displacement": float(expected_displacement),
        "ratio": float(ratio),
        "per_point_displacement": np.asarray(signed, dtype=np.float64),
    }


def _serial_mujoco_probe(
    env, body_id, qpos_addr, q0, points, force_world, expected_displacement, cfg
) -> dict:
    """Sequential fallback when Comfree backend cannot be imported."""

    if mujoco is None:
        raise RuntimeError("Neither comfree nor mujoco are available for probing")
    model_cpu = getattr(env.sim.model, "_model", env.sim.model)
    data_cpu = getattr(env.sim.data, "_data", env.sim.data)

    qpos_saved = np.asarray(data_cpu.qpos, dtype=np.float64).copy()
    qvel_saved = np.asarray(data_cpu.qvel, dtype=np.float64).copy()
    xfrc_saved = np.asarray(data_cpu.xfrc_applied, dtype=np.float64).copy()
    timestep_saved = float(model_cpu.opt.timestep)
    try:
        model_cpu.opt.timestep = float(cfg.sim_dt)
        disp = np.zeros(points.shape[0], dtype=np.float64)
        for i, point in enumerate(points):
            data_cpu.qpos[:] = qpos_saved
            data_cpu.qvel[:] = qvel_saved
            data_cpu.xfrc_applied[:] = 0.0
            mujoco.mj_forward(model_cpu, data_cpu)
            arm = np.asarray(point, dtype=np.float64) - np.asarray(
                data_cpu.xpos[body_id], dtype=np.float64
            )
            torque = np.cross(arm, force_world)
            data_cpu.xfrc_applied[body_id, :3] = force_world
            data_cpu.xfrc_applied[body_id, 3:] = torque
            for _ in range(int(cfg.horizon_steps)):
                mujoco.mj_step(model_cpu, data_cpu)
            disp[i] = float(data_cpu.qpos[qpos_addr]) - q0
    finally:
        data_cpu.qpos[:] = qpos_saved
        data_cpu.qvel[:] = qvel_saved
        data_cpu.xfrc_applied[:] = xfrc_saved
        model_cpu.opt.timestep = timestep_saved
        mujoco.mj_forward(model_cpu, data_cpu)

    signed = disp  # caller decides sign in the parallel path; keep raw here
    best = float(np.max(np.abs(signed))) if signed.size else 0.0
    ratio = best / expected_displacement
    use_prehens = int(ratio < cfg.success_ratio)
    _print(
        f"[serial-fallback] best_disp={best:.4f} expected={expected_displacement:.4f} "
        f"ratio={ratio:.3f} use_prehens={use_prehens}"
    )
    return {
        "use_prehens": use_prehens,
        "best_displacement": best,
        "expected_displacement": float(expected_displacement),
        "ratio": float(ratio),
        "per_point_displacement": np.asarray(signed, dtype=np.float64),
    }
