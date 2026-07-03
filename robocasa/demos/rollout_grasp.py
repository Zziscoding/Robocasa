"""Grasp rollout: close the gripper and evaluate force-closure cost.

After the skeleton-pose solver places the EE at a pre-contact pose, this
module strips the scene down to the hand + object subtree and drives the
two fingers toward each other (gripper closing) instead of translating
the EE.  The quality metric is the force-closure cost of the contact
pair, evaluated at the final closed configuration by
:func:`miqp_grasping.evaluate_force_closure_at_points`.

Reference: :class:`rollout.FloatingEERollout` (translation rollout).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np
import torch

from robocasa.demos.ee_floating_mppi import (
    _backend_tensor,
    _import_comfree_backend,
    _import_mjwarp_backend,
)


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GraspRolloutConfig:
    device: str = "cuda:0"
    sim_dt: float = 0.01
    horizon_steps: int = 15
    # Gripper travel: closing from full open to this amount (full finger
    # fingertip distance).
    gripper_closed_opening: float = 0.005
    contact_stiffness: float = 0.2
    contact_damping: float = 0.001
    nconmax_per_env: int = 120
    njmax_per_env: int = 500
    compile_cuda_graph: bool = False
    prefer_comfree: bool = True
    # Evaluate force_closure_cost at this step index (-1 = last step).
    force_closure_eval_at_step: int = -1


@dataclass(frozen=True)
class GraspRolloutResult:
    force_closure_costs: np.ndarray  # (horizon_steps,); non-evaluated steps = NaN
    gripper_openings: np.ndarray  # (horizon_steps,) full fingertip distance
    penetrations: np.ndarray  # (horizon_steps,) max penetration depth (m)
    force_closure_cost_final: float
    force_closure_cost_min: float
    gripper_opening_at_min: float
    max_penetration: float
    steps: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(vec):
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _quat_wxyz_to_mat(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _extract_hand_finger_geoms(env, ee_site_name: str):
    """Extract hand/finger geoms classified as palm / left / right.

    Returns
    -------
    palm_geoms : list[(size_3, pos_3, rot_3x3)]
    left_geoms  : list[(size_3, pos_3, rot_3x3)]
    right_geoms : list[(size_3, pos_3, rot_3x3)]
    All positions/rotations are in the EE-site frame.
    """
    from robocasa.demos import visualize_mujoco as viz_mj

    raw_model, raw_data = viz_mj._raw_model_data(env)
    site_pos, site_rot = viz_mj._site_pose(env, ee_site_name)
    body_ids = viz_mj._ghost_source_body_ids(env)

    palm, left, right = [], [], []

    for geom_id in range(int(raw_model.ngeom)):
        body_id = int(raw_model.geom_bodyid[geom_id])
        geom_name = ""
        for name, gid in getattr(raw_model, "_geom_name2id", {}).items():
            if int(gid) == geom_id:
                geom_name = name
                break
        if body_id not in body_ids and not any(
            tok in geom_name.lower()
            for tok in (
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
                "hand",
                "finger",
                "gripper",
                "pad",
            )
        ):
            continue
        geom_type = int(raw_model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        size = np.asarray(raw_model.geom_size[geom_id], dtype=np.float64).copy()
        draw_type = geom_type
        local_center = np.zeros(3, dtype=np.float64)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            draw_type = int(mujoco.mjtGeom.mjGEOM_BOX)
            try:
                mesh_id = int(raw_model.geom_dataid[geom_id])
                vadr = int(raw_model.mesh_vertadr[mesh_id])
                vnum = int(raw_model.mesh_vertnum[mesh_id])
                verts = np.asarray(
                    raw_model.mesh_vert[vadr : vadr + vnum], dtype=np.float64
                )
                local_center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
                size = 0.5 * (verts.max(axis=0) - verts.min(axis=0))
            except Exception:
                pass
        geom_pos = np.asarray(raw_data.geom_xpos[geom_id], dtype=np.float64).reshape(3)
        geom_rot = np.asarray(raw_data.geom_xmat[geom_id], dtype=np.float64).reshape(
            3, 3
        )
        geom_pos = geom_pos + geom_rot @ local_center
        local_pos = site_rot.T @ (geom_pos - site_pos)
        local_rot = site_rot.T @ geom_rot

        entry = (size, local_pos, local_rot)

        nl = geom_name.lower()
        if "leftfinger" in nl or "left_finger" in nl or "finger1" in nl:
            left.append(entry)
        elif "rightfinger" in nl or "right_finger" in nl or "finger2" in nl:
            right.append(entry)
        else:
            palm.append(entry)
    return palm, left, right


def _extract_object_geoms(
    env, raw_model: mujoco.MjModel, raw_data: mujoco.MjData, object_body_id: int
) -> list[tuple]:
    """Extract all geoms in the object subtree.

    Returns list of (type_str, size_3, pos_3_in_body_frame, quat_wxyz).
    """
    import collections

    children = collections.defaultdict(list)
    for b in range(int(raw_model.nbody)):
        p = int(raw_model.body_parentid[b])
        children[p].append(b)

    subtree = set()
    queue = [int(object_body_id)]
    while queue:
        node = queue.pop()
        subtree.add(node)
        queue.extend(children.get(node, []))

    geoms_out = []
    for gid in range(int(raw_model.ngeom)):
        if int(raw_model.geom_bodyid[gid]) not in subtree:
            continue
        geom_type = int(raw_model.geom_type[gid])
        type_map = {
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
        }
        type_str = type_map.get(geom_type, None)
        if type_str is None:
            continue
        body_id = int(raw_model.geom_bodyid[gid])
        body_pos = np.asarray(raw_data.xpos[body_id], dtype=np.float64).reshape(3)
        body_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            body_quat,
            np.asarray(raw_data.xmat[body_id], dtype=np.float64).reshape(9),
        )
        body_rot = _quat_wxyz_to_mat(body_quat)
        geom_pos = np.asarray(raw_data.geom_xpos[gid], dtype=np.float64).reshape(3)
        geom_rot = np.asarray(raw_data.xmat[gid], dtype=np.float64).reshape(3, 3)
        size = np.asarray(raw_model.geom_size[gid], dtype=np.float64).copy()
        # Position relative to object body frame.
        rel_pos = body_rot.T @ (geom_pos - body_pos)
        rel_rot = body_rot.T @ geom_rot
        rel_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(rel_quat, rel_rot.reshape(9))
        geoms_out.append((type_str, size, rel_pos, rel_quat))
    return geoms_out


def _build_grasp_rollout_mjcf(
    env,
    raw_model: mujoco.MjModel,
    raw_data: mujoco.MjData,
    ee_site_name: str,
    object_body_id: int,
    timestep: float,
    zero_gravity: bool = True,
) -> str:
    """Build an MJCF with palm + 2 finger bodies + object subtree.

    Finger bodies each have a slide joint along the local y-axis; this
    lets the rollout close the gripper by driving both finger joints to
    decrease their y offset.
    """
    palm_geoms, left_geoms, right_geoms = _extract_hand_finger_geoms(env, ee_site_name)
    object_geoms = _extract_object_geoms(env, raw_model, raw_data, object_body_id)

    obj = raw_model
    obj_body_pos = np.asarray(raw_data.xpos[object_body_id], dtype=np.float64).reshape(
        3
    )
    obj_body_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(
        obj_body_quat,
        np.asarray(raw_data.xmat[object_body_id], dtype=np.float64).reshape(9),
    )

    def _geom_xml(size_3, pos_3, quat_wxyz, name, rgba="0.5 0.5 0.5 1"):
        return (
            f'      <geom name="{name}" type="box" size="{_fmt(size_3)}" '
            f'pos="{_fmt(pos_3)}" quat="{_fmt(quat_wxyz)}" rgba="{rgba}" '
            f'contype="1" conaffinity="1"/>\n'
        )

    palm_xml = "".join(
        _geom_xml(s, p, q, f"palm_{i}", "0.3 0.5 1 0.6")
        for i, (s, p, r) in enumerate(palm_geoms)
        for q in [_rot_to_quat_wxyz(r)]
    )
    left_xml = "".join(
        _geom_xml(s, p, q, f"left_finger_{i}", "0.2 0.8 0.4 0.7")
        for i, (s, p, r) in enumerate(left_geoms)
        for q in [_rot_to_quat_wxyz(r)]
    )
    right_xml = "".join(
        _geom_xml(s, p, q, f"right_finger_{i}", "0.9 0.3 0.3 0.7")
        for i, (s, p, r) in enumerate(right_geoms)
        for q in [_rot_to_quat_wxyz(r)]
    )

    # Per-finger initial y position (half spread).
    left_y = _finger_spread_y(left_geoms)
    right_y = _finger_spread_y(right_geoms)

    obj_xml = "".join(
        _geom_xml(size, pos, quat, f"obj_{i}", "0.88 0.52 0.22 1")
        for i, (_type, size, pos, quat) in enumerate(object_geoms)
    )

    gravity = "0 0 0" if zero_gravity else "0 0 -9.81"

    return (
        '<mujoco model="grasp_rollout">\n'
        '  <compiler angle="radian"/>\n'
        f'  <option timestep="{float(timestep):.9g}" gravity="{gravity}" integrator="Euler"/>\n'
        "  <default>\n"
        '    <geom condim="3" friction="1 0.05 0.005" solref="0.01 1" solimp="0.9 0.95 0.001"/>\n'
        "  </default>\n"
        "  <worldbody>\n"
        f'    <body name="grasp_object" pos="{_fmt(obj_body_pos)}" '
        f'quat="{_fmt(obj_body_quat)}">\n'
        f'      <freejoint name="grasp_object_freejoint"/>\n'
        + obj_xml
        + "    </body>\n"
        '    <body name="palm" pos="0 0 0" quat="1 0 0 0">\n'
        '      <freejoint name="palm_freejoint"/>\n'
        + palm_xml
        + f'      <body name="left_finger" pos="0 {float(left_y):.6f} 0" quat="1 0 0 0">\n'
        f'        <joint name="left_finger_slide" type="slide" axis="0 1 0" '
        f'range="-0.1 0.1" damping="0.5"/>\n' + left_xml + "      </body>\n"
        f'      <body name="right_finger" pos="0 {float(right_y):.6f} 0" quat="1 0 0 0">\n'
        f'        <joint name="right_finger_slide" type="slide" axis="0 1 0" '
        f'range="-0.1 0.1" damping="0.5"/>\n' + right_xml + "      </body>\n"
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n"
    )


def _finger_spread_y(finger_geoms):
    if not finger_geoms:
        return 0.04
    ys = [float(p[1]) for (_s, p, _r) in finger_geoms]
    return np.mean(ys) if ys else 0.04


def _rot_to_quat_wxyz(rot):
    q = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, np.asarray(rot, dtype=np.float64).reshape(9))
    return q


def _fmt(val):
    return " ".join(f"{float(v):.8f}" for v in np.asarray(val).reshape(-1))


# ---------------------------------------------------------------------------
# GraspRollout
# ---------------------------------------------------------------------------


class GraspRollout:
    """Close the gripper from an initial opening and evaluate force-closure cost.

    Built once per scene (hand subtree + object subtree stripped from the
    env).  :meth:`run` can be called repeatedly for different EE poses.
    """

    def __init__(
        self,
        env,
        *,
        object_body_id: int,
        ee_site_name: str,
        config: Optional[GraspRolloutConfig] = None,
        mesh_path: Optional[str] = None,
        obj_pos: Optional[np.ndarray] = None,
        obj_scale: np.ndarray = np.ones(3),
        obj_quat: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config or GraspRolloutConfig()
        self.env = env
        self.ee_site_name = str(ee_site_name)
        self.object_body_id = int(object_body_id)
        self.mesh_path = mesh_path
        self.obj_pos = (
            None
            if obj_pos is None
            else np.asarray(obj_pos, dtype=np.float64).reshape(3)
        )
        self.obj_scale = np.asarray(obj_scale, dtype=np.float64)
        self.obj_quat = obj_quat
        self._build_stripped_model(env, ee_site_name, object_body_id)
        self._init_backend()

    # ----- model construction -----------------------------------------------

    def _build_stripped_model(
        self, env, ee_site_name: str, object_body_id: int
    ) -> None:
        raw_model = getattr(env.sim.model, "_model", env.sim.model)
        raw_data = getattr(env.sim.data, "_data", env.sim.data)
        self.model_cpu = mujoco.MjModel.from_xml_string(
            _build_grasp_rollout_mjcf(
                env,
                raw_model,
                raw_data,
                ee_site_name,
                object_body_id,
                float(self.config.sim_dt),
                zero_gravity=True,
            )
        )
        self.object_body_id = int(
            mujoco.mj_name2id(self.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "grasp_object")
        )
        palm_id = int(
            mujoco.mj_name2id(self.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "palm")
        )
        left_finger_id = int(
            mujoco.mj_name2id(self.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        )
        right_finger_id = int(
            mujoco.mj_name2id(self.model_cpu, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        )
        palm_free = int(
            mujoco.mj_name2id(
                self.model_cpu, mujoco.mjtObj.mjOBJ_JOINT, "palm_freejoint"
            )
        )
        left_slide = int(
            mujoco.mj_name2id(
                self.model_cpu, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_slide"
            )
        )
        right_slide = int(
            mujoco.mj_name2id(
                self.model_cpu, mujoco.mjtObj.mjOBJ_JOINT, "right_finger_slide"
            )
        )
        if any(
            x < 0
            for x in (
                self.object_body_id,
                palm_id,
                left_finger_id,
                right_finger_id,
                palm_free,
            )
        ):
            raise ValueError("GraspRollout model missing required bodies/joints")
        self._palm_free_qaddr = int(self.model_cpu.jnt_qposadr[palm_free])
        self._palm_free_daddr = int(self.model_cpu.jnt_dofadr[palm_free])
        self._left_slide_qaddr = int(self.model_cpu.jnt_qposadr[left_slide])
        self._right_slide_qaddr = int(self.model_cpu.jnt_qposadr[right_slide])
        # Initial finger offsets (half spread).
        q0 = np.zeros(self.model_cpu.nq, dtype=np.float64)
        mujoco.mj_fwdPosition(self.model_cpu, mujoco.MjData(self.model_cpu))
        self._left_slide_init = float(q0[self._left_slide_qaddr])
        self._right_slide_init = float(q0[self._right_slide_qaddr])

        self._active_hand_geoms = np.array(
            [
                gid
                for gid in range(int(self.model_cpu.ngeom))
                if int(self.model_cpu.geom_bodyid[gid])
                in (palm_id, left_finger_id, right_finger_id)
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
        self._left_finger_body_id = left_finger_id
        self._right_finger_body_id = right_finger_id

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
                    f"[grasp_rollout] Comfree backend unavailable: {exc!r}\n"
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

    # ----- single-world state IO -------------------------------------------

    def _write_mocap_and_fingers(
        self,
        position: np.ndarray,
        quat_wxyz: np.ndarray,
        left_finger_y: float,
        right_finger_y: float,
    ) -> None:
        qpos = _backend_tensor(self.data.qpos)
        a = self._palm_free_qaddr
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
        if self._left_slide_qaddr >= 0:
            qpos[0, self._left_slide_qaddr] = float(left_finger_y)
        if self._right_slide_qaddr >= 0:
            qpos[0, self._right_slide_qaddr] = float(right_finger_y)

    def _read_penetration(self) -> float:
        contact = self.data.contact
        ncon = int(getattr(self.data, "ncon", 0))
        if ncon <= 0:
            return 0.0
        try:
            dist = _backend_tensor(contact.dist)
            g1 = _backend_tensor(contact.geom1).long()
            g2 = _backend_tensor(contact.geom2).long()
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
        g1_obj = torch.isin(g1, object_geoms)
        g2_obj = torch.isin(g2, object_geoms)
        hand_object_pair = (g1_hand & g2_obj) | (g2_hand & g1_obj)
        pen = torch.clamp(-dist - 1e-3, min=0.0) * hand_object_pair.float()
        return float(pen.max().item()) if pen.numel() else 0.0

    def _sync_cpu_from_backend(self) -> None:
        """Copy the single rollout world's generalized state into MjData."""
        qpos = _backend_tensor(self.data.qpos).detach().cpu().numpy()[0]
        qvel = _backend_tensor(self.data.qvel).detach().cpu().numpy()[0]
        self.data_cpu.qpos[:] = qpos
        self.data_cpu.qvel[:] = qvel
        mujoco.mj_forward(self.model_cpu, self.data_cpu)

    # ----- main rollout -----------------------------------------------------

    def run(
        self,
        ee_position: np.ndarray,
        ee_rotation: np.ndarray,
        initial_gripper_opening: float,
    ) -> GraspRolloutResult:
        """Close the gripper and track penetration / force-closure cost.

        Parameters
        ----------
        ee_position : (3,) palm world position.
        ee_rotation : (3, 3) palm world rotation.
        initial_gripper_opening : full fingertip distance at the start of
            the rollout (e.g. 0.04 m).

        Returns
        -------
        :class:`GraspRolloutResult`
        """
        cfg = self.config
        ee_position = np.asarray(ee_position, dtype=np.float64).reshape(3)
        ee_rotation = np.asarray(ee_rotation, dtype=np.float64).reshape(3, 3)
        quat_wxyz = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_wxyz, ee_rotation.reshape(9))

        target_open = float(cfg.gripper_closed_opening)
        steps = int(cfg.horizon_steps)
        eval_step = int(cfg.force_closure_eval_at_step)
        if eval_step < 0:
            eval_step = steps - 1

        fc_costs = np.full(steps, np.nan, dtype=np.float64)
        openings_arr = np.zeros(steps, dtype=np.float64)
        pens = np.zeros(steps, dtype=np.float64)

        half_init = 0.5 * float(initial_gripper_opening)
        half_target = 0.5 * target_open
        # Finger initial sign relative to palm centre.
        left_sign = (
            np.sign(self._left_slide_init) if self._left_slide_init != 0.0 else +1.0
        )
        right_sign = (
            np.sign(self._right_slide_init) if self._right_slide_init != 0.0 else -1.0
        )

        best_fc_cost = float("inf")
        best_fc_open = float(initial_gripper_opening)
        worst_pen = 0.0

        for j in range(steps):
            alpha = j / max(1, steps - 1)
            half_cur = half_init + (half_target - half_init) * alpha
            left_y = left_sign * half_cur
            right_y = right_sign * half_cur
            openings_arr[j] = 2.0 * half_cur

            self._write_mocap_and_fingers(ee_position, quat_wxyz, left_y, right_y)
            if j == 0:
                self._forward_fn()
            else:
                self._step_fn()
            pens[j] = self._read_penetration()
            worst_pen = max(worst_pen, pens[j])

            if j == eval_step and self.mesh_path is not None:
                fc_costs[j] = self._eval_force_closure() or float("inf")
            if np.isfinite(fc_costs[j]) and fc_costs[j] < best_fc_cost:
                best_fc_cost = fc_costs[j]
                best_fc_open = openings_arr[j]

        return GraspRolloutResult(
            force_closure_costs=fc_costs,
            gripper_openings=openings_arr,
            penetrations=pens,
            force_closure_cost_final=fc_costs[eval_step]
            if fc_costs[eval_step] == fc_costs[eval_step]
            else float("inf"),
            force_closure_cost_min=best_fc_cost,
            gripper_opening_at_min=best_fc_open,
            max_penetration=worst_pen,
            steps=steps,
        )

    def _eval_force_closure(self) -> Optional[float]:
        """Evaluate force_closure_cost for current finger contact points."""
        try:
            from robocasa.demos.example_code.grasping.miqp_grasping import (
                evaluate_force_closure_at_points,
            )
        except ImportError:
            return None
        if self.mesh_path is None or self.obj_quat is None:
            return None
        if self._left_finger_body_id < 0 or self._right_finger_body_id < 0:
            return None
        self._sync_cpu_from_backend()
        # Find closest surface vertex to each finger body centre.
        left_pos = np.asarray(
            self.data_cpu.xpos[self._left_finger_body_id], dtype=np.float64
        ).reshape(3)
        right_pos = np.asarray(
            self.data_cpu.xpos[self._right_finger_body_id], dtype=np.float64
        ).reshape(3)
        from scipy.spatial.transform import Rotation

        obj_rot = Rotation.from_quat(
            [self.obj_quat[1], self.obj_quat[2], self.obj_quat[3], self.obj_quat[0]]
        ).as_matrix()
        from robocasa.demos.example_code.grasping.mlqp_point_v2 import (
            LambdaContactControlOptimizer,
        )

        opt = LambdaContactControlOptimizer(
            self.mesh_path,
            scale_factors=tuple(float(s) for s in self.obj_scale),
            num_grasp_contacts=2,
            sample_num=80,
            nlp_solver="ipopt",
        )
        object_pos = (
            np.asarray(self.obj_pos, dtype=np.float64).reshape(3)
            if self.obj_pos is not None
            else np.asarray(
                self.data_cpu.xpos[self.object_body_id], dtype=np.float64
            ).reshape(3)
        )
        left_local = obj_rot.T @ (left_pos - object_pos)
        right_local = obj_rot.T @ (right_pos - object_pos)
        try:
            li, _, _, _ = opt.pp.project_point_to_mesh(left_local)
            ri, _, _, _ = opt.pp.project_point_to_mesh(right_local)
        except Exception:
            return None
        res = evaluate_force_closure_at_points(
            self.mesh_path,
            self.obj_scale,
            np.array([li, ri]),
            obj_quat=self.obj_quat,
            nlp_solver="ipopt",
            sample_budget=80,
        )
        return float(res["force_closure_cost"]) if res.get("valid", False) else None


__all__ = [
    "GraspRolloutConfig",
    "GraspRolloutResult",
    "GraspRollout",
    "solve_grasp_rollout",
]


def solve_grasp_rollout(
    env,
    object_body_id: int,
    ee_site_name: str,
    ee_position: np.ndarray,
    ee_rotation: np.ndarray,
    initial_gripper_opening: float,
    config: Optional[GraspRolloutConfig] = None,
    mesh_path: Optional[str] = None,
    obj_scale: np.ndarray = np.ones(3),
    obj_quat: Optional[np.ndarray] = None,
) -> GraspRolloutResult:
    """One-shot convenience: build + run + return."""
    rollout = GraspRollout(
        env,
        object_body_id=object_body_id,
        ee_site_name=ee_site_name,
        config=config,
        mesh_path=mesh_path,
        obj_scale=obj_scale,
        obj_quat=obj_quat,
    )
    return rollout.run(ee_position, ee_rotation, initial_gripper_opening)
