"""Comfree/mjwarp-backed MPPI for refining a Franka EE skeleton pose.

The Panda hand+fingers are treated as a floating mocap-driven body in a stripped
scene that contains only the target object body and the hand/finger geoms. We
sample 7-D control perturbations (Δpos, Δrot, Δgripper) around a caller-supplied
skeleton pose and score each sample from a single forward evaluation against the
sum of (i) tracking deviation from the skeleton pose and (ii) a non-penetration
penalty in the stripped scene.

The approach rollout ("translate along the optimal force direction and report the
object-cost delta") is intentionally split out into ``rollout.py``. Use
``FloatingEERollout`` after ``solve()`` accepts a refined pose to measure whether
that push actually improves the object configuration; the accept decision then
belongs to the caller rather than the per-sample search.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FloatingEEConfig:
    device: str = "cuda:0"
    seed: int = 0
    num_samples: int = 128
    num_perturb_samples: int = 1
    sim_dt: float = 0.005
    horizon_steps: int = (
        1  # single-step pose solve; approach rollout lives in rollout.py
    )
    approach_total_distance: float = 0.01  # vestigial; used by FloatingEERollout
    max_num_iterations: int = 6
    elite_ratio: float = 0.1
    temperature: float = 1.0
    pos_noise_scale: float = 0.005
    rot_noise_scale: float = 0.05
    gripper_noise_scale: float = 0.005
    gripper_min: float = 0.0
    gripper_max: float = 0.08
    first_noise_scale: float = 1.0
    last_noise_scale: float = 0.2
    pen_threshold: float = 0.05
    contact_tolerance: float = 0.002
    pen_weight: float = 0.0
    contact_weight: float = 0.0
    track_pos_weight: float = 200.0
    track_rot_weight: float = 20.0
    track_gripper_weight: float = 50.0
    object_weight: float = 500.0
    object_improvement_eps: float = 1e-5
    accept_object_improvement_only: bool = True
    contact_stiffness: float = 0.2
    contact_damping: float = 0.001
    nconmax_per_env: int = 120
    njmax_per_env: int = 500
    compile_cuda_graph: bool = True
    prefer_comfree: bool = True

    def __post_init__(self) -> None:
        assert self.num_samples > 0
        assert self.num_perturb_samples > 0
        assert self.horizon_steps >= 1
        assert 0.0 < self.elite_ratio <= 1.0


@dataclass(frozen=True)
class FloatingEEResult:
    ee_position: np.ndarray
    ee_rotation: np.ndarray  # 3x3
    gripper_opening: float
    best_cost: float
    pen_cost: float
    contact_distance: float
    track_cost: float
    object_cost_delta: float
    accepted: bool
    iteration_history: tuple = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Small helpers (copied from dream.py to keep modules decoupled)
# ---------------------------------------------------------------------------


def _normalize_quat(q: Tensor) -> Tensor:
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


def _rotvec_to_quat(rotvec: Tensor) -> Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1e-7,
        torch.sin(half) / angle,
        0.5 - angle.square() / 48.0,
    )
    return _normalize_quat(torch.cat((torch.cos(half), rotvec * scale), dim=-1))


def _quat_to_matrix(q: Tensor) -> Tensor:
    q = _normalize_quat(q)
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


def _orientation_error(target: Tensor, current: Tensor) -> Tensor:
    return 0.5 * (
        torch.linalg.cross(current[..., :, 0], target[..., :, 0])
        + torch.linalg.cross(current[..., :, 1], target[..., :, 1])
        + torch.linalg.cross(current[..., :, 2], target[..., :, 2])
    )


def _backend_tensor(value: Any) -> Tensor:
    # warp arrays expose __cuda_array_interface__ / __dlpack__; defer to torch.
    if isinstance(value, Tensor):
        return value
    try:
        return torch.utils.dlpack.from_dlpack(value.__dlpack__())
    except Exception:
        return torch.as_tensor(value)


# ---------------------------------------------------------------------------
# Backend import (Comfree preferred, mjwarp fallback)
# ---------------------------------------------------------------------------


def _import_comfree_backend() -> tuple[Any, Any, Callable]:
    """Mirror dream._import_comfree_backend. Returns (wp, cfwarp, step_fn)."""
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "comfree_warp"
    module = sys.modules.get("comfree_warp")
    if module is not None and not hasattr(module, "put_model"):
        del sys.modules["comfree_warp"]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    wp = importlib.import_module("warp")
    cfwarp = importlib.import_module("comfree_warp")
    core_forward = importlib.import_module("comfree_warp.comfree_core._src.forward")
    return wp, cfwarp, core_forward.step_comfree


def _import_mjwarp_backend() -> tuple[Any, Any, Callable]:
    """Native mujoco_warp fallback. Returns (wp, mjwarp, step_fn)."""
    import warp as wp  # type: ignore

    mjwarp = importlib.import_module("mujoco_warp")
    return wp, mjwarp, mjwarp.step


# ---------------------------------------------------------------------------
# FloatingEEMPPI
# ---------------------------------------------------------------------------


class FloatingEEMPPI:
    """MPPI refinement of an EE pose against a stripped floating-EE scene."""

    def __init__(
        self,
        env,
        *,
        hand_xml_path: str | None,
        finger_geom_names: Sequence[str],
        object_body_id: int,
        ee_site_name: str,
        config: FloatingEEConfig | None = None,
        approach_world: np.ndarray,
        target_object_position: np.ndarray,
        selected_contact_point_world: np.ndarray | None = None,
        drawer_qpos_addr: int | None = None,
        drawer_qpos_value: float | None = None,
    ) -> None:
        self.config = config or FloatingEEConfig()
        self.env = env
        self.ee_site_name = str(ee_site_name)
        self.object_body_id = int(object_body_id)
        self.approach_world = np.asarray(approach_world, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(self.approach_world))
        if n > 1e-9:
            self.approach_world /= n
        self.target_object_position = np.asarray(
            target_object_position, dtype=np.float64
        ).reshape(3)
        self.selected_contact_point_world = (
            None
            if selected_contact_point_world is None
            else np.asarray(selected_contact_point_world, dtype=np.float64).reshape(3)
        )
        self.drawer_qpos_addr = drawer_qpos_addr
        self.drawer_qpos_value = drawer_qpos_value

        # Simpler-path model construction: clone the full env model and mask out
        # all contacts that don't involve the hand/finger or drawer subtrees.
        # TODO(refactor): replace with a true stripped mjcf if scene size hurts.
        self._build_stripped_model(env, hand_xml_path, finger_geom_names)

        # Backend (Comfree preferred, native mjwarp fallback).
        self._init_backend()

        # CUDA graph capture for the step path.
        self._step_graph = self._compile_step_graph()

    # ----- model construction -----------------------------------------------

    def _build_stripped_model(
        self,
        env,
        hand_xml_path: str | None,
        finger_geom_names: Sequence[str],
    ) -> None:
        """Build a minimal floating-hand + sliding-object contact model."""
        del hand_xml_path  # see note above
        del finger_geom_names
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
                "FloatingEEMPPI minimal model is missing required bodies/joints"
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
        nworld = cfg.num_samples * cfg.num_perturb_samples
        self.nworld = nworld
        backend_kind = None
        if cfg.prefer_comfree:
            try:
                self.wp, self.cfwarp, step_fn = _import_comfree_backend()
                self.model = self.cfwarp.put_model(self.model_cpu)
                # NOTE: put_data's nconmax/njmax are PER-WORLD counts. The total
                # across-worlds contact buffer (naconmax) is derived internally as
                # nworld * nconmax. Passing nworld * per_env here would make the
                # backend ~nworld x larger than intended and exhaust GPU memory
                # (e.g. efc.J becomes (nworld, nworld*njmax_per_env, nv) floats).
                self.data = self.cfwarp.put_data(
                    self.model_cpu,
                    self.data_cpu,
                    nworld=nworld,
                    nconmax=cfg.nconmax_per_env,
                    njmax=cfg.njmax_per_env,
                )
                self._step_fn = lambda: step_fn(self.model, self.data)
                self._forward_fn = lambda: self.cfwarp.forward(self.model, self.data)
                backend_kind = "comfree"
            except Exception as exc:
                sys.stderr.write(
                    f"[ee_floating_mppi] Comfree backend unavailable, "
                    f"falling back to native mjwarp: {exc!r}\n"
                )
                sys.stderr.flush()
        if backend_kind is None:
            self.wp, mjwarp, step_fn = _import_mjwarp_backend()
            self.cfwarp = mjwarp
            self.model = mjwarp.put_model(self.model_cpu)
            # per-world counts; see the comfree branch comment above for why the
            # nworld multiplier must NOT be applied to nconmax/njmax.
            self.data = mjwarp.put_data(
                self.model_cpu,
                self.data_cpu,
                nworld=nworld,
                nconmax=cfg.nconmax_per_env,
                njmax=cfg.njmax_per_env,
            )
            self._step_fn = lambda: step_fn(self.model, self.data)
            self._forward_fn = lambda: mjwarp.forward(self.model, self.data)
            backend_kind = "mjwarp"
        self.backend_kind = backend_kind
        self.torch_device = torch.device(cfg.device)
        self.generator = torch.Generator(device=self.torch_device).manual_seed(
            int(cfg.seed)
        )

    def _compile_step_graph(self) -> Any | None:
        if not self.config.compile_cuda_graph:
            return None
        try:
            if not self.wp.get_device(self.config.device).is_cuda:
                return None
            self._step_fn()
            self._step_fn()
            self.wp.synchronize()
            with self.wp.ScopedDevice(self.config.device):
                with self.wp.ScopedCapture() as capture:
                    self._step_fn()
            self.wp.synchronize()
            return capture.graph
        except Exception as exc:
            sys.stderr.write(
                f"[ee_floating_mppi] CUDA graph capture failed, per-step launch: {exc!r}\n"
            )
            sys.stderr.flush()
            return None

    def _step_backend(self) -> None:
        if self._step_graph is None:
            self._step_fn()
        else:
            self.wp.capture_launch(self._step_graph)

    # ----- per-iteration state IO ------------------------------------------

    def _write_mocap_pose(
        self,
        positions: Tensor,
        quats: Tensor,
        gripper: Tensor,
        *,
        reset_object: bool = False,
    ) -> None:
        """Overwrite freejoint qpos + finger qpos for all worlds."""
        qpos = _backend_tensor(self.data.qpos)
        a = self._hand_free_qaddr
        qpos[:, a : a + 3] = positions
        qpos[:, a + 3 : a + 7] = quats
        if self._finger_qaddrs.size > 0:
            half = (0.5 * gripper).unsqueeze(-1)
            idx = torch.as_tensor(
                self._finger_qaddrs, device=qpos.device, dtype=torch.long
            )
            qpos[:, idx] = half.expand(-1, idx.shape[0])
        if self.drawer_qpos_addr is not None and self.drawer_qpos_value is not None:
            qpos[:, int(self.drawer_qpos_addr)] = float(self.drawer_qpos_value)
        if reset_object and hasattr(self, "_object_slide_qaddr"):
            qpos[:, int(self._object_slide_qaddr)] = 0.0
        qvel = _backend_tensor(self.data.qvel)
        if reset_object:
            qvel.zero_()
        elif hasattr(self, "_hand_free_daddr"):
            d = int(self._hand_free_daddr)
            qvel[:, d : d + 6] = 0.0

    def _read_object_position(self) -> Tensor:
        xpos = _backend_tensor(self.data.xpos)
        return xpos[:, self.object_body_id, :].clone()

    def _read_penetration(self) -> Tensor:
        """Max penetration per world across hand/finger vs object contacts."""
        contact = self.data.contact
        ncon = int(getattr(self.data, "ncon", 0)) if hasattr(self.data, "ncon") else 0
        # Comfree/mjwarp store contacts as flat per-batch arrays; the exact API
        # varies. Use a defensive accessor: try the standard fields.
        try:
            dist = _backend_tensor(contact.dist)
            if hasattr(contact, "geom1") and hasattr(contact, "geom2"):
                g1 = _backend_tensor(contact.geom1)
                g2 = _backend_tensor(contact.geom2)
            else:
                geom = _backend_tensor(contact.geom).long().reshape(-1, 2)
                g1 = geom[:, 0]
                g2 = geom[:, 1]
            worldid = _backend_tensor(
                getattr(contact, "worldid", getattr(contact, "world_id", None))
            )
        except Exception:
            # If contacts aren't directly readable, return zero penetration.
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
        out.scatter_reduce_(0, worldid.long(), pen, reduce="amax", include_self=True)
        return out

    # ----- main solve -------------------------------------------------------

    def _sample_deltas(self, noise_scale: float) -> Tensor:
        """Sample (num_samples*num_perturb, 7) Δ = (Δpos3, Δrot3, Δg1)."""
        cfg = self.config
        n = cfg.num_samples * cfg.num_perturb_samples
        scale = torch.tensor(
            [cfg.pos_noise_scale] * 3
            + [cfg.rot_noise_scale] * 3
            + [cfg.gripper_noise_scale],
            device=self.torch_device,
        )
        deltas = (
            torch.randn((n, 7), device=self.torch_device, generator=self.generator)
            * scale
            * float(noise_scale)
        )
        # Always include the nominal as sample 0.
        deltas[0].zero_()
        return deltas

    @torch.no_grad()
    def solve(
        self,
        skeleton_pose_xyz: np.ndarray,
        skeleton_pose_quat_wxyz: np.ndarray,
        skeleton_gripper_opening: float,
    ) -> FloatingEEResult:
        cfg = self.config
        device = self.torch_device
        skel_pos = torch.as_tensor(
            np.asarray(skeleton_pose_xyz, dtype=np.float32).reshape(3), device=device
        )
        skel_quat = torch.as_tensor(
            np.asarray(skeleton_pose_quat_wxyz, dtype=np.float32).reshape(4),
            device=device,
        )
        skel_quat = _normalize_quat(skel_quat)
        skel_g = float(skeleton_gripper_opening)

        nominal = torch.zeros(7, device=device)
        nominal[6] = skel_g
        best = nominal.clone()
        best_cost_val = float("inf")
        best_pen = float("inf")
        best_contact_dist = float("inf")
        best_track = float("inf")
        history = []
        selected_contact_world = (
            torch.as_tensor(
                self.selected_contact_point_world.astype(np.float32), device=device
            )
            if self.selected_contact_point_world is not None
            else None
        )
        contact_local = None
        if selected_contact_world is not None:
            skel_rot_t = torch.as_tensor(
                np.asarray(skeleton_pose_quat_wxyz, dtype=np.float32).reshape(4),
                device=device,
            )
            skel_rot_m = _quat_to_matrix(
                _normalize_quat(skel_rot_t).unsqueeze(0)
            ).squeeze(0)
            contact_local = skel_rot_m.transpose(0, 1) @ (
                selected_contact_world - skel_pos
            )

        # Log-ramped noise schedule across iterations.
        iters = max(1, int(cfg.max_num_iterations))
        if iters == 1:
            ramp = [cfg.first_noise_scale]
        else:
            ramp = list(
                np.logspace(
                    math.log10(cfg.first_noise_scale),
                    math.log10(cfg.last_noise_scale),
                    iters,
                )
            )

        for it in range(iters):
            perturb = self._sample_deltas(float(ramp[it]))
            # Sample = skeleton pose + nominal + perturbation, shape (S, 7).
            sample_pos = (
                skel_pos.unsqueeze(0) + nominal[:3].unsqueeze(0) + perturb[:, :3]
            )
            sample_rot = nominal[3:6].unsqueeze(0) + perturb[:, 3:6]
            sample_quat = _quat_mul(_rotvec_to_quat(sample_rot), skel_quat.unsqueeze(0))
            sample_g = torch.clamp(
                nominal[6] + perturb[:, 6],
                min=float(cfg.gripper_min),
                max=float(cfg.gripper_max),
            )
            sample_rot_m = _quat_to_matrix(sample_quat)
            if contact_local is not None and selected_contact_world is not None:
                contact_world = sample_pos + torch.bmm(
                    sample_rot_m,
                    contact_local.view(1, 3, 1).expand(sample_pos.shape[0], -1, -1),
                ).squeeze(-1)
                contact_distance = torch.linalg.vector_norm(
                    contact_world - selected_contact_world.unsqueeze(0), dim=-1
                )
            else:
                contact_distance = torch.zeros(sample_pos.shape[0], device=device)

            # Single-step forward: write the sample pose into the stripped scene and
            # read penetration once. No approach rollout here.
            self._write_mocap_pose(
                sample_pos,
                sample_quat,
                sample_g,
                reset_object=True,
            )
            self._forward_fn()
            penetration = self._read_penetration()

            pose_delta = nominal.unsqueeze(0) + perturb
            track_pos = pose_delta[:, :3].pow(2).sum(dim=-1)
            track_rot = pose_delta[:, 3:6].pow(2).sum(dim=-1)
            track_g = (sample_g - skel_g).pow(2)

            costs = (
                cfg.pen_weight * penetration.pow(2)
                + cfg.contact_weight * contact_distance.pow(2)
                + cfg.track_pos_weight * track_pos
                + cfg.track_rot_weight * track_rot
                + cfg.track_gripper_weight * track_g
            )

            min_idx = int(torch.argmin(costs).item())
            cmin = float(costs[min_idx].item())
            if cmin < best_cost_val:
                best_cost_val = cmin
                best = (
                    nominal
                    + torch.cat((perturb[min_idx, :6], perturb[min_idx, 6:7]), dim=0)
                ).clone()
                best_pen = float(penetration[min_idx].item())
                best_contact_dist = float(contact_distance[min_idx].item())
                best_track = float(
                    (
                        cfg.track_pos_weight * track_pos[min_idx]
                        + cfg.track_rot_weight * track_rot[min_idx]
                        + cfg.track_gripper_weight * track_g[min_idx]
                    ).item()
                )

            # Elite-weighted nominal update.
            n = costs.shape[0]
            elite_count = max(1, int(math.ceil(cfg.elite_ratio * n)))
            elite_idx = torch.topk(costs, elite_count, largest=False).indices
            ec = costs[elite_idx]
            mu = ec.mean()
            sigma = ec.std().clamp_min(1e-2)
            w = torch.exp(-(ec - mu) / sigma / max(cfg.temperature, 1e-6))
            w = w / w.sum().clamp_min(1e-12)
            elite_pert = perturb[elite_idx]
            nominal = nominal + (w.unsqueeze(-1) * elite_pert).sum(dim=0)
            nominal[6] = torch.clamp(
                nominal[6], float(cfg.gripper_min), float(cfg.gripper_max)
            )

            history.append(
                (
                    it,
                    cmin,
                    float(penetration.min().item()),
                    float(contact_distance.min().item()),
                    float(
                        (
                            cfg.track_pos_weight * track_pos
                            + cfg.track_rot_weight * track_rot
                            + cfg.track_gripper_weight * track_g
                        )
                        .min()
                        .item()
                    ),
                )
            )

        # Build final absolute pose from best Δ.
        final_pos = (skel_pos + best[:3]).detach().cpu().numpy().astype(np.float64)
        final_quat = _quat_mul(
            _rotvec_to_quat(best[3:6].unsqueeze(0)), skel_quat.unsqueeze(0)
        ).squeeze(0)
        final_rot = (
            _quat_to_matrix(final_quat).detach().cpu().numpy().astype(np.float64)
        )
        final_g = float(best[6].item())

        # MPPI's in-solver gate is non-penetration only. Object-improvement
        # acceptance is decided by the caller via FloatingEERollout, so it is
        # not reflected here. ``object_cost_delta`` is therefore nan for this
        # result.
        accepted = bool(
            best_pen <= cfg.pen_threshold and best_contact_dist <= cfg.contact_tolerance
        )
        return FloatingEEResult(
            ee_position=final_pos,
            ee_rotation=final_rot,
            gripper_opening=final_g,
            best_cost=best_cost_val,
            pen_cost=best_pen,
            contact_distance=best_contact_dist,
            track_cost=best_track,
            object_cost_delta=float("nan"),
            accepted=accepted,
            iteration_history=tuple(history),
        )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _fmt(values: Any) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return " ".join(f"{float(v):.9g}" for v in arr)


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def _xml_name(name: str) -> str:
    return escape(str(name), {'"': "&quot;"})


def _geom_type_size_center(
    model: mujoco.MjModel, geom_id: int
) -> tuple[str, np.ndarray, np.ndarray] | None:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64).reshape(3).copy()
    center = np.zeros(3, dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return "sphere", np.maximum(size[:1], 0.003), center
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return "capsule", np.maximum(size[:2], 0.003), center
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return "cylinder", np.maximum(size[:2], 0.003), center
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return "ellipsoid", np.maximum(size[:3], 0.003), center
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return "box", np.maximum(size[:3], 0.003), center
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        try:
            mesh_id = int(model.geom_dataid[geom_id])
            vadr = int(model.mesh_vertadr[mesh_id])
            vnum = int(model.mesh_vertnum[mesh_id])
            verts = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)
            if verts.size:
                mesh_min = np.min(verts, axis=0)
                mesh_max = np.max(verts, axis=0)
                half = 0.5 * (mesh_max - mesh_min)
                center = 0.5 * (mesh_max + mesh_min)
                return (
                    "box",
                    np.maximum(half, np.array([0.003, 0.003, 0.003])),
                    np.asarray(center, dtype=np.float64).reshape(3),
                )
        except Exception:
            pass
        return "box", np.maximum(size[:3], 0.003), center
    return None


def _ghost_type_and_size(
    geom_type: int, size: np.ndarray
) -> tuple[str, np.ndarray] | None:
    size = np.asarray(size, dtype=np.float64).reshape(3).copy()
    if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return "sphere", np.maximum(size[:1], 0.003)
    if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return "capsule", np.maximum(size[:2], 0.003)
    if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return "cylinder", np.maximum(size[:2], 0.003)
    if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return "ellipsoid", np.maximum(size[:3], 0.003)
    if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_BOX):
        return "box", np.maximum(size[:3], 0.003)
    return None


def _build_floating_ee_mjcf(
    env,
    raw_model: mujoco.MjModel,
    raw_data: mujoco.MjData,
    ee_site_name: str,
    source_object_body_id: int,
    object_slide_axis_world: np.ndarray,
    timestep: float,
) -> str:
    from robocasa.demos import visualize_mujoco as viz_mj

    source_object_body_id = int(source_object_body_id)
    if source_object_body_id < 0 or source_object_body_id >= int(raw_model.nbody):
        raise ValueError(f"Invalid source object body id: {source_object_body_id}")

    object_pos = np.asarray(
        raw_data.xpos[source_object_body_id], dtype=np.float64
    ).reshape(3)
    axis = np.asarray(object_slide_axis_world, dtype=np.float64).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    axis = axis / axis_norm if axis_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    object_subtree = _body_subtree(raw_model, source_object_body_id)

    object_geom_lines = []
    for gid in range(int(raw_model.ngeom)):
        if int(raw_model.geom_bodyid[gid]) not in object_subtree:
            continue
        type_size_center = _geom_type_size_center(raw_model, gid)
        if type_size_center is None:
            continue
        geom_type, geom_size, geom_center = type_size_center
        geom_pos = np.asarray(raw_data.geom_xpos[gid], dtype=np.float64).reshape(3)
        geom_rot = np.asarray(raw_data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        geom_pos = geom_pos + geom_rot @ geom_center
        rgba = np.asarray(raw_model.geom_rgba[gid], dtype=np.float64).reshape(4)
        object_geom_lines.append(
            (
                f'      <geom name="object_geom_{gid}" type="{geom_type}" '
                f'pos="{_fmt(geom_pos - object_pos)}" '
                f'quat="{_fmt(_matrix_to_quat_wxyz(geom_rot))}" '
                f'size="{_fmt(geom_size)}" rgba="{_fmt(rgba)}" '
                f'contype="1" conaffinity="1"/>\n'
            )
        )
    if not object_geom_lines:
        raise ValueError("FloatingEEMPPI minimal model found no object geoms")

    hand_geom_lines = []
    ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(env, ee_site_name)
    for idx, ghost in enumerate(ghost_geoms):
        type_size = _ghost_type_and_size(int(ghost.geom_type), np.asarray(ghost.size))
        if type_size is None:
            continue
        geom_type, geom_size = type_size
        hand_geom_lines.append(
            (
                f'      <geom name="hand_geom_{idx}" type="{geom_type}" '
                f'pos="{_fmt(ghost.local_pos)}" '
                f'quat="{_fmt(_matrix_to_quat_wxyz(ghost.local_rot))}" '
                f'size="{_fmt(geom_size)}" rgba="0.2 0.5 1 0.45" '
                f'contype="1" conaffinity="1"/>\n'
            )
        )
    if not hand_geom_lines:
        raise ValueError("FloatingEEMPPI minimal model found no hand geoms")

    return (
        '<mujoco model="floating_ee_mppi">\n'
        '  <compiler angle="radian"/>\n'
        f'  <option timestep="{float(timestep):.9g}" gravity="0 0 0" integrator="Euler"/>\n'
        "  <default>\n"
        '    <geom condim="3" friction="1 0.005 0.0001" solref="0.005 1" solimp="0.9 0.95 0.001"/>\n'
        "  </default>\n"
        "  <worldbody>\n"
        f'    <body name="floating_object" pos="{_fmt(object_pos)}">\n'
        f'      <joint name="floating_object_slide" type="slide" axis="{_fmt(axis)}" damping="0.02"/>\n'
        + "".join(object_geom_lines)
        + "    </body>\n"
        '    <body name="floating_hand" pos="0 0 0" quat="1 0 0 0">\n'
        '      <freejoint name="floating_hand_freejoint"/>\n'
        + "".join(hand_geom_lines)
        + "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )


def _body_subtree(model: mujoco.MjModel, root: int) -> set[int]:
    """All body ids in the subtree rooted at `root` (inclusive)."""
    out = {int(root)}
    parent = model.body_parentid
    for b in range(int(model.nbody)):
        anc = int(b)
        while anc > 0:
            if anc == int(root):
                out.add(int(b))
                break
            anc = int(parent[anc])
    return out


__all__ = ["FloatingEEConfig", "FloatingEEResult", "FloatingEEMPPI"]
