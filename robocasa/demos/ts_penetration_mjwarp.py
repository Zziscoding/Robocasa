"""Batched GPU (mjwarp) penetration checker for skeleton-pose sequences.

The CPU ``mj_ray`` / ``mj_multiRay`` checker in :mod:`skelton_scene` casts
6 axis rays per hand-sample and iterates per-pose in Python. For the
``--ts-skeleton`` pipeline that runs 3k+ poses per frame across ~50
frames, this is the wall-clock bottleneck (~10 s/frame, mostly CPU
narrowphase).

This module replaces that with a batched mjwarp narrowphase:

* Build a stripped floating-hand + static-kitchen MJCF *once*, with the
  drawer subtree excluded (drawer contact is the *intended* target so we
  never want to test against it).
* ``put_data(nworld = batch)`` mirrors the scene across worlds; each pose
  becomes one world.
* Per frame: write per-world mocap pose (freejoint qpos), call
  ``forward`` once, read per-world hand↔scene contact ``dist``.
* A pose is dropped if the min contact dist for its world is below
  ``-tol``.

Only primitive-approximated geoms are included (mesh → bounding-box); the
existing skeleton solver uses the same approximation via
``ee_floating_mppi._geom_type_size_center``, so parity is intentional.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np
import torch

from robocasa.demos.ee_floating_mppi import (
    _backend_tensor,
    _body_subtree,
    _geom_type_size_center,
    _ghost_type_and_size,
    _import_comfree_backend,
    _import_mjwarp_backend,
    _matrix_to_quat_wxyz,
    _fmt,
)


def _drawer_body_ids(env) -> set[int]:
    """All body ids belonging to the target drawer subtree."""
    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    try:
        drawer_root_name = env.drawer.name
    except Exception:
        return set()
    try:
        root_id = int(
            mujoco.mj_name2id(
                raw_model,
                mujoco.mjtObj.mjOBJ_BODY,
                drawer_root_name,
            )
        )
    except Exception:
        return set()
    if root_id < 0:
        return set()
    return _body_subtree(raw_model, root_id)


def _robot_geom_ids(env) -> set[int]:
    """Every geom that belongs to a robot (name-prefix ``robot`` or ``gripper``)."""
    model = env.sim.model
    out = set()
    for name, gid in getattr(model, "_geom_name2id", {}).items():
        if name.startswith("robot") or name.startswith("gripper"):
            out.add(int(gid))
    return out


def _build_static_scene_mjcf(
    env,
    ee_site_name: str,
    exclude_geom_ids: Sequence[int],
    timestep: float,
) -> str:
    """MJCF with (a) baked static kitchen geoms and (b) a floating hand body.

    ``exclude_geom_ids`` are dropped entirely — normally the drawer subtree +
    handle inflating geoms + the robot geoms.
    """
    from robocasa.demos import visualize_mujoco as viz_mj

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    exclude = {int(g) for g in exclude_geom_ids}

    # Static kitchen: bake world pos/quat into each geom, attached to worldbody.
    static_lines = []
    for gid in range(int(raw_model.ngeom)):
        if int(gid) in exclude:
            continue
        info = _geom_type_size_center(raw_model, gid)
        if info is None:
            continue
        geom_type, geom_size, geom_center = info
        geom_pos_world = np.asarray(raw_data.geom_xpos[gid], dtype=np.float64).reshape(
            3
        )
        geom_rot_world = np.asarray(raw_data.geom_xmat[gid], dtype=np.float64).reshape(
            3, 3
        )
        geom_pos_world = geom_pos_world + geom_rot_world @ geom_center
        rgba = np.asarray(raw_model.geom_rgba[gid], dtype=np.float64).reshape(4)
        static_lines.append(
            f'      <geom name="static_geom_{gid}" type="{geom_type}" '
            f'pos="{_fmt(geom_pos_world)}" '
            f'quat="{_fmt(_matrix_to_quat_wxyz(geom_rot_world))}" '
            f'size="{_fmt(geom_size)}" rgba="{_fmt(rgba)}" '
            f'contype="2" conaffinity="1"/>\n'
        )
    if not static_lines:
        raise ValueError("Stripped scene contains zero static geoms after exclusions")

    # Floating hand: ghost hand geoms attached to a body with a freejoint.
    hand_lines = []
    ghosts = viz_mj._extract_hand_finger_ghost_geoms(env, ee_site_name)
    for idx, ghost in enumerate(ghosts):
        ts = _ghost_type_and_size(int(ghost.geom_type), np.asarray(ghost.size))
        if ts is None:
            continue
        gtype, gsize = ts
        hand_lines.append(
            f'      <geom name="hand_geom_{idx}" type="{gtype}" '
            f'pos="{_fmt(ghost.local_pos)}" '
            f'quat="{_fmt(_matrix_to_quat_wxyz(ghost.local_rot))}" '
            f'size="{_fmt(gsize)}" rgba="0.2 0.5 1 0.5" '
            f'contype="1" conaffinity="2"/>\n'
        )
    if not hand_lines:
        raise ValueError("Stripped scene has no floating-hand ghost geoms")

    return (
        '<mujoco model="skel_pen_check">\n'
        '  <compiler angle="radian"/>\n'
        f'  <option timestep="{float(timestep):.9g}" gravity="0 0 0" integrator="Euler">\n'
        '    <flag warmstart="disable" energy="disable"/>\n'
        "  </option>\n"
        "  <default>\n"
        '    <geom condim="3" solref="0.005 1" solimp="0.9 0.95 0.001"/>\n'
        "  </default>\n"
        "  <worldbody>\n"
        + '    <body name="static_kitchen">\n'
        + "".join(static_lines)
        + "    </body>\n"
        '    <body name="floating_hand" pos="0 0 0" quat="1 0 0 0">\n'
        '      <freejoint name="floating_hand_freejoint"/>\n'
        + "".join(hand_lines)
        + "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )


@dataclass
class MjwarpCheckerStats:
    build_seconds: float
    n_worlds: int
    n_static_geoms: int
    n_hand_geoms: int
    backend_kind: str


class MjwarpSkelPenChecker:
    """Batched narrowphase penetration test for floating-hand skeleton poses.

    Build once for a given ``env`` + exclude-set (drawer + robot geoms), then
    call :meth:`check_batch` per frame with the projected pose array.
    """

    def __init__(
        self,
        env,
        *,
        ee_site_name: str,
        exclude_geom_ids: Sequence[int],
        batch_size: int = 512,
        sim_dt: float = 0.01,
        nconmax_per_env: int = 32,
        njmax_per_env: int = 32,
        prefer_comfree: bool = True,
        device: str = "cuda:0",
    ) -> None:
        t0 = time.perf_counter()
        self.env = env
        self.ee_site_name = str(ee_site_name)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        mjcf = _build_static_scene_mjcf(
            env,
            self.ee_site_name,
            exclude_geom_ids,
            sim_dt,
        )
        self.model_cpu = mujoco.MjModel.from_xml_string(mjcf)
        self.model_cpu.opt.timestep = float(sim_dt)
        self.data_cpu = mujoco.MjData(self.model_cpu)

        self._hand_body_id = int(
            mujoco.mj_name2id(
                self.model_cpu,
                mujoco.mjtObj.mjOBJ_BODY,
                "floating_hand",
            )
        )
        free_jnt = int(
            mujoco.mj_name2id(
                self.model_cpu,
                mujoco.mjtObj.mjOBJ_JOINT,
                "floating_hand_freejoint",
            )
        )
        self._hand_qaddr = int(self.model_cpu.jnt_qposadr[free_jnt])
        self._hand_geom_ids = np.array(
            [
                g
                for g in range(int(self.model_cpu.ngeom))
                if int(self.model_cpu.geom_bodyid[g]) == self._hand_body_id
            ],
            dtype=np.int64,
        )
        static_body = int(
            mujoco.mj_name2id(
                self.model_cpu,
                mujoco.mjtObj.mjOBJ_BODY,
                "static_kitchen",
            )
        )
        self._static_geom_ids = np.array(
            [
                g
                for g in range(int(self.model_cpu.ngeom))
                if int(self.model_cpu.geom_bodyid[g]) == static_body
            ],
            dtype=np.int64,
        )

        nworld = int(self.batch_size)
        self.nworld = nworld
        backend_kind = None
        if prefer_comfree:
            try:
                self.wp, self.cfwarp, _step = _import_comfree_backend()
                self.model = self.cfwarp.put_model(
                    self.model_cpu,
                    comfree_stiffness=0.2,
                    comfree_damping=0.001,
                )
                self.data = self.cfwarp.put_data(
                    self.model_cpu,
                    self.data_cpu,
                    nworld=nworld,
                    nconmax=nworld * int(nconmax_per_env),
                    njmax=nworld * int(njmax_per_env),
                )
                self._forward_fn = lambda: self.cfwarp.forward(self.model, self.data)
                backend_kind = "comfree"
            except Exception as exc:
                print(
                    f"[ts-mjwarp] comfree unavailable, falling back to mjwarp: {exc!r}",
                    flush=True,
                )
        if backend_kind is None:
            self.wp, mjwarp, _step = _import_mjwarp_backend()
            self.cfwarp = mjwarp
            self.model = mjwarp.put_model(self.model_cpu)
            self.data = mjwarp.put_data(
                self.model_cpu,
                self.data_cpu,
                nworld=nworld,
                nconmax=nworld * int(nconmax_per_env),
                njmax=nworld * int(njmax_per_env),
            )
            self._forward_fn = lambda: mjwarp.forward(self.model, self.data)
            backend_kind = "mjwarp"

        self.stats = MjwarpCheckerStats(
            build_seconds=time.perf_counter() - t0,
            n_worlds=nworld,
            n_static_geoms=int(self._static_geom_ids.size),
            n_hand_geoms=int(self._hand_geom_ids.size),
            backend_kind=backend_kind,
        )

    # ---------------------------------------------------------------- api ---

    @torch.no_grad()
    def check_batch(
        self,
        ee_positions: np.ndarray,  # (N, 3)
        ee_rotations: np.ndarray,  # (N, 3, 3)
        *,
        tol: float = 1e-3,
        park_pos: Sequence[float] = (0.0, 0.0, 5.0),
    ) -> np.ndarray:
        """Return a (N,) bool array — True means the pose has NO penetration.

        Poses are placed one-per-world in successive ``batch_size`` chunks;
        remaining worlds in the final chunk are parked far away (5 m up) so
        they cannot generate false contacts.
        """
        ee_positions = np.asarray(ee_positions, dtype=np.float64).reshape(-1, 3)
        ee_rotations = np.asarray(ee_rotations, dtype=np.float64).reshape(-1, 3, 3)
        n = ee_positions.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool)
        if ee_rotations.shape[0] != n:
            raise ValueError("ee_rotations and ee_positions must have same N")

        # Convert rotations → wxyz quats (CPU).
        quats = np.zeros((n, 4), dtype=np.float64)
        for i in range(n):
            mujoco.mju_mat2Quat(quats[i], ee_rotations[i].reshape(9))

        keep = np.ones(n, dtype=bool)
        park = np.asarray(park_pos, dtype=np.float64).reshape(3)
        park_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Precompute mask tensors once per call.
        hand_ids_t = torch.as_tensor(
            self._hand_geom_ids, device=self.device, dtype=torch.long
        )
        static_ids_t = torch.as_tensor(
            self._static_geom_ids, device=self.device, dtype=torch.long
        )

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            k = end - start

            qpos = _backend_tensor(self.data.qpos)  # (nworld, nq)
            a = self._hand_qaddr
            # Batched write for active worlds.
            pos_t = torch.as_tensor(
                ee_positions[start:end].astype(np.float32),
                device=qpos.device,
                dtype=qpos.dtype,
            )
            quat_t = torch.as_tensor(
                quats[start:end].astype(np.float32),
                device=qpos.device,
                dtype=qpos.dtype,
            )
            qpos[:k, a : a + 3] = pos_t
            qpos[:k, a + 3 : a + 7] = quat_t
            # Park remaining worlds.
            if k < self.nworld:
                park_pos_t = torch.as_tensor(
                    park.astype(np.float32),
                    device=qpos.device,
                    dtype=qpos.dtype,
                )
                park_quat_t = torch.as_tensor(
                    park_quat.astype(np.float32),
                    device=qpos.device,
                    dtype=qpos.dtype,
                )
                qpos[k:, a : a + 3] = park_pos_t.unsqueeze(0)
                qpos[k:, a + 3 : a + 7] = park_quat_t.unsqueeze(0)

            qvel = _backend_tensor(self.data.qvel)
            qvel.zero_()

            self._forward_fn()

            # Read contacts: mjwarp exposes flattened contact.dist / geom / worldid.
            contact = self.data.contact
            try:
                dist = _backend_tensor(contact.dist)
            except Exception:
                dist = None
            if dist is None or int(dist.numel()) == 0:
                continue

            # Contact geom pairs.
            try:
                if hasattr(contact, "geom1") and hasattr(contact, "geom2"):
                    g1 = _backend_tensor(contact.geom1).long()
                    g2 = _backend_tensor(contact.geom2).long()
                else:
                    geom = _backend_tensor(contact.geom).long().reshape(-1, 2)
                    g1, g2 = geom[:, 0], geom[:, 1]
                wid = (
                    _backend_tensor(contact.worldid).long()
                    if hasattr(contact, "worldid")
                    else None
                )
            except Exception:
                continue
            if wid is None:
                # Single-world backend: shouldn't happen but treat all contacts as world 0.
                wid = torch.zeros_like(g1)

            hand_pair = torch.isin(g1, hand_ids_t) | torch.isin(g2, hand_ids_t)
            static_pair = torch.isin(g1, static_ids_t) | torch.isin(g2, static_ids_t)
            valid = hand_pair & static_pair
            penetrating = valid & (dist < -float(tol))
            if not bool(penetrating.any().item()):
                continue

            bad_worlds = torch.unique(wid[penetrating]).detach().cpu().numpy()
            for w in bad_worlds:
                w = int(w)
                if 0 <= w < k:
                    keep[start + w] = False

        return keep


__all__ = ["MjwarpSkelPenChecker", "MjwarpCheckerStats"]
