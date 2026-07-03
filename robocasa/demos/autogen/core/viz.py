"""Visualization: 3 popup viewers + physical-PD contact marker playback.

Port of the three popup visualizers from
``demo_close_drawer_autogen.py``:

1. ``visualize_mink_q_poses_popup`` — ghost hand at each mink q waypoint.
2. ``visualize_floating_ee_poses_popup`` — ghost hand + skeleton at each
   refined floating-EE pose.
3. ``visualize_contact_marker_with_physical_pd`` — PD torque control
   playback with contact marker spheres.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

from robocasa.demos import visualize_mujoco as viz_mj  # noqa: E402
from robocasa.demos.demo_close_drawer_contact_curobo import (  # noqa: E402
    _draw_viewer_sphere,
    _ee_sample_world,
    _set_drawer_joint_value,
    _set_env_arm_q,
    _drawer_joint_value,
)

from .context import PipelineContext, autogen_print


def _hand_box_from_ghost_geoms(env: Any, frame_name: str) -> tuple | None:
    """Extract the palm/hand bbox (local_pos, local_rot, half_ext) from ghost geoms."""
    geoms = viz_mj._extract_hand_finger_ghost_geoms(env, frame_name)
    best = None
    best_volume = -1.0
    for ghost in geoms:
        half = np.asarray(ghost.size, dtype=np.float64).reshape(3)
        if float(np.max(half)) > 0.20:
            continue
        volume = float(np.prod(np.maximum(half, 1e-6)))
        if volume > best_volume:
            best = ghost
            best_volume = volume
    if best is None:
        return None
    return (
        np.asarray(best.local_pos, dtype=np.float64).reshape(3).copy(),
        np.asarray(best.local_rot, dtype=np.float64).reshape(3, 3).copy(),
        np.asarray(best.size, dtype=np.float64).reshape(3).copy(),
    )


def visualize_mink_q_poses_popup(ctx: PipelineContext, args: Any) -> None:
    """Popup rendering ghost Panda hand at every mink q waypoint."""
    if not bool(getattr(args, "autogen_visualize_mink_poses", True)):
        return
    if ctx.mink_solution is None:
        return
    q_waypoints = np.asarray(
        getattr(ctx.mink_solution, "q_waypoints", np.zeros((0, 7))),
        dtype=np.float64,
    )
    if q_waypoints.size == 0:
        return
    q_waypoints = q_waypoints.reshape(-1, 7)
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    if frame_name not in ctx.env.sim.model._site_name2id:
        return

    import mujoco
    import mujoco.viewer

    raw_model = getattr(ctx.env.sim.model, "_model", ctx.env.sim.model)
    raw_data = getattr(ctx.env.sim.data, "_data", ctx.env.sim.data)
    qpos_saved = ctx.env.sim.data.qpos.copy()
    qvel_saved = ctx.env.sim.data.qvel.copy()
    site_id = int(ctx.env.sim.model.site_name2id(frame_name))
    arm_joint_names = tuple(ctx.robot_state["robocasa_joint_names"])
    drawer_q = float(_drawer_joint_value(ctx.env))
    poses = []
    try:
        _set_drawer_joint_value(ctx.env, drawer_q)
        ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(ctx.env, frame_name)
        for q_arm in q_waypoints:
            _set_env_arm_q(ctx.env, arm_joint_names, q_arm)
            _set_drawer_joint_value(ctx.env, drawer_q)
            ctx.env.sim.forward()
            poses.append(
                (
                    np.asarray(
                        ctx.env.sim.data.site_xpos[site_id], dtype=np.float64
                    ).copy(),
                    np.asarray(ctx.env.sim.data.site_xmat[site_id], dtype=np.float64)
                    .reshape(3, 3)
                    .copy(),
                )
            )
    finally:
        ctx.env.sim.data.qpos[:] = qpos_saved
        ctx.env.sim.data.qvel[:] = qvel_saved
        ctx.env.sim.forward()

    if not poses:
        return
    lookat = np.mean(np.asarray([p[0] for p in poses], dtype=np.float64), axis=0)
    palette = np.asarray(
        [
            [0.05, 0.45, 1.0, float(getattr(args, "autogen_mink_ghost_alpha", 0.28))],
            [1.0, 0.58, 0.05, float(getattr(args, "autogen_mink_ghost_alpha", 0.28))],
            [0.05, 0.75, 0.35, float(getattr(args, "autogen_mink_ghost_alpha", 0.28))],
        ],
        dtype=np.float32,
    )
    with mujoco.viewer.launch_passive(
        raw_model, raw_data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        try:
            viewer.opt.geomgroup[:] = 0
            viewer.opt.geomgroup[1] = 1
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
        while viewer.is_running():
            if hasattr(viewer, "user_scn"):
                viewer.user_scn.ngeom = 0
                for pose_index, (target_pos, target_rot) in enumerate(poses):
                    rgba = palette[min(pose_index, palette.shape[0] - 1)]
                    for ghost in ghost_geoms:
                        viz_mj._add_ghost_geom(
                            viewer.user_scn, ghost, target_pos, target_rot, rgba
                        )
            viewer.sync()
            time.sleep(1.0 / fps)


def visualize_floating_ee_poses_popup(ctx: PipelineContext, args: Any) -> None:
    """Popup rendering ghost Panda hand at every refined floating-EE pose."""
    if not bool(getattr(args, "autogen_visualize_floating_ee", True)):
        return
    if not ctx.refined_poses:
        return
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]

    import mujoco
    import mujoco.viewer

    raw_model = getattr(ctx.env.sim.model, "_model", ctx.env.sim.model)
    raw_data = getattr(ctx.env.sim.data, "_data", ctx.env.sim.data)
    qpos_saved = ctx.env.sim.data.qpos.copy()
    qvel_saved = ctx.env.sim.data.qvel.copy()
    try:
        from robocasa.demos import ee_skelton as _ee_skelton
    except Exception:
        _ee_skelton = None
    skeleton = None
    hand_local_pos = None
    hand_local_rot = None
    hand_half_ext = None
    try:
        _set_drawer_joint_value(ctx.env, float(_drawer_joint_value(ctx.env)))
        ctx.env.sim.forward()
        ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(ctx.env, frame_name)
        if _ee_skelton is not None:
            try:
                skeleton = _ee_skelton.build_panda_skeleton(ctx.env, frame_name)
            except Exception:
                pass
        try:
            ghost_hand_box = _hand_box_from_ghost_geoms(ctx.env, frame_name)
            if ghost_hand_box is not None:
                hand_local_pos, hand_local_rot, hand_half_ext = ghost_hand_box
        except Exception:
            pass
    finally:
        ctx.env.sim.data.qpos[:] = qpos_saved
        ctx.env.sim.data.qvel[:] = qvel_saved
        ctx.env.sim.forward()

    limit = max(int(getattr(args, "autogen_visualize_floating_ee_limit", 12)), 1)
    poses = ctx.refined_poses[:limit]
    lookat = np.mean(
        np.asarray([np.asarray(p[0], dtype=np.float64) for p in poses]), axis=0
    )
    palette = np.asarray(
        [
            [
                0.95,
                0.30,
                0.30,
                float(getattr(args, "autogen_floating_ee_ghost_alpha", 0.32)),
            ],
            [
                0.30,
                0.70,
                0.95,
                float(getattr(args, "autogen_floating_ee_ghost_alpha", 0.32)),
            ],
            [
                0.95,
                0.85,
                0.20,
                float(getattr(args, "autogen_floating_ee_ghost_alpha", 0.32)),
            ],
            [
                0.40,
                0.85,
                0.40,
                float(getattr(args, "autogen_floating_ee_ghost_alpha", 0.32)),
            ],
        ],
        dtype=np.float32,
    )
    with mujoco.viewer.launch_passive(
        raw_model, raw_data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        try:
            viewer.opt.geomgroup[:] = 0
            viewer.opt.geomgroup[1] = 1
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
        finger_radius = float(getattr(args, "autogen_skeleton_finger_radius", 0.004))
        while viewer.is_running():
            if hasattr(viewer, "user_scn"):
                viewer.user_scn.ngeom = 0
                for pose_index, (pos, rot, _g, _cid) in enumerate(poses):
                    rgba = palette[pose_index % palette.shape[0]]
                    for ghost in ghost_geoms:
                        viz_mj._add_ghost_geom(
                            viewer.user_scn,
                            ghost,
                            np.asarray(pos, dtype=np.float64),
                            np.asarray(rot, dtype=np.float64).reshape(3, 3),
                            rgba,
                        )
                    if skeleton is not None and _ee_skelton is not None:
                        try:
                            ee_pos = np.asarray(pos, dtype=np.float64).reshape(3)
                            ee_rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
                            if hand_local_pos is not None:
                                hand_center_w = ee_pos + ee_rot @ hand_local_pos
                                hand_rot_w = ee_rot @ hand_local_rot
                                half_ext_full = np.asarray(
                                    hand_half_ext, dtype=np.float64
                                ).reshape(3)
                            else:
                                half_ext_full = np.asarray(
                                    skeleton.hand_box_half_extents_ee, dtype=np.float64
                                ).reshape(3)
                                hand_center_w = ee_pos + ee_rot @ np.asarray(
                                    skeleton.hand_box_center_ee, dtype=np.float64
                                )
                                hand_rot_w = ee_rot @ np.asarray(
                                    skeleton.hand_box_rotation_ee, dtype=np.float64
                                )
                            half_ext_full[int(np.argmin(half_ext_full))] = 0.002
                            rgba_solid = np.asarray(rgba, dtype=np.float32).copy()
                            rgba_solid[3] = max(float(rgba_solid[3]), 0.6)
                            _ee_skelton._add_box(
                                viewer.user_scn,
                                hand_center_w,
                                hand_rot_w,
                                half_ext_full,
                                rgba_solid,
                            )
                            g_open = float(_g) if _g is not None else 0.04
                            (
                                left_seg,
                                right_seg,
                                _,
                                _,
                            ) = _ee_skelton._finger_segments_with_opening(
                                skeleton, g_open
                            )
                            for seg in (left_seg, right_seg):
                                sa = ee_pos + ee_rot @ seg[0]
                                sb = ee_pos + ee_rot @ seg[1]
                                _ee_skelton._add_capsule_segment(
                                    viewer.user_scn, sa, sb, finger_radius, rgba_solid
                                )
                        except Exception:
                            pass
            viewer.sync()
            time.sleep(1.0 / fps)


def _arm_pd_control_maps(env: Any, arm_joint_names: tuple) -> tuple:
    """Build qpos/dof/actuator address arrays for PD control."""
    model = env.sim.model
    joint_ids = [int(model.joint_name2id(name)) for name in arm_joint_names]
    qpos_addrs = np.asarray(
        [int(model.jnt_qposadr[jid]) for jid in joint_ids], dtype=np.int64
    )
    dof_addrs = np.asarray(
        [int(model.jnt_dofadr[jid]) for jid in joint_ids], dtype=np.int64
    )
    actuator_ids = []
    for jid, jname in zip(joint_ids, arm_joint_names):
        matches = [
            aid
            for aid in range(int(model.nu))
            if int(model.actuator_trnid[aid, 0]) == jid
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one actuator for arm joint {jname!r}; found {len(matches)}"
            )
        actuator_ids.append(matches[0])
    return qpos_addrs, dof_addrs, np.asarray(actuator_ids, dtype=np.int64)


def _step_arm_pd(
    env: Any,
    q_target: np.ndarray,
    qpos_addrs: np.ndarray,
    dof_addrs: np.ndarray,
    actuator_ids: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> None:
    """One PD torque step."""
    data = env.sim.data
    model = env.sim.model
    q_target = np.asarray(q_target, dtype=np.float64).reshape(7)
    q = np.asarray(data.qpos[qpos_addrs], dtype=np.float64)
    qd = np.asarray(data.qvel[dof_addrs], dtype=np.float64)
    bias = np.asarray(data.qfrc_bias[dof_addrs], dtype=np.float64)
    torque = kp * (q_target - q) - kd * qd + bias
    ctrlrange = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64)
    torque = np.clip(torque, ctrlrange[:, 0], ctrlrange[:, 1])
    data.ctrl[actuator_ids] = torque
    env.sim.step()


def visualize_contact_marker_with_physical_pd(ctx: PipelineContext, args: Any) -> None:
    """Play back the cuRobo trajectory with physical PD torque control.

    Renders the contact marker sphere and EE marker sphere in the viewer.
    """
    if not bool(getattr(args, "autogen_physical_pd_visualization", True)):
        return
    if ctx.q_traj is None or np.asarray(ctx.q_traj).size == 0:
        return
    if ctx.env.viewer is None:
        autogen_print("Contact visualization skipped: renderer was not initialized.")
        return

    arm_trajectory = np.asarray(ctx.q_traj, dtype=np.float64).reshape(-1, 7)
    arm_joint_names = tuple(ctx.robot_state["robocasa_joint_names"])
    qpos_addrs, dof_addrs, actuator_ids = _arm_pd_control_maps(ctx.env, arm_joint_names)
    kp = np.full(7, float(getattr(args, "autogen_pd_kp", 350.0)), dtype=np.float64)
    kd = np.full(7, float(getattr(args, "autogen_pd_kd", 35.0)), dtype=np.float64)
    timestep = float(ctx.env.sim.model.opt.timestep)
    target_dt = float(
        getattr(
            args,
            "autogen_pd_target_dt",
            getattr(args, "curobo_interpolation_dt", 0.02),
        )
    )
    sim_steps_per_target = max(int(round(max(target_dt, timestep) / timestep)), 1)
    fps = max(float(getattr(args, "contact_marker_fps", 30.0)), 1.0)
    sync_interval = max(int(round((1.0 / fps) / timestep)), 1)
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    marker_pos = (
        ctx.selected.world_point + ctx.panel.outward_world * args.contact_marker_offset
    )
    marker_rgba = np.array([1.0, 0.0, 0.0, args.contact_marker_alpha], dtype=np.float32)
    ee_marker_rgba = np.array([0.0, 1.0, 0.15, 1.0], dtype=np.float32)

    autogen_print(
        "Showing cuRobo trajectory with physical PD torque control "
        f"(frames={arm_trajectory.shape[0]}, sim_steps_per_target={sim_steps_per_target}, "
        f"kp={float(kp[0]):.1f}, kd={float(kd[0]):.1f}). "
        "Close the viewer window or press Ctrl+C to exit."
    )
    try:
        ctx.env.viewer.update()
    except Exception:
        return
    viewer = getattr(ctx.env.viewer, "viewer", None)
    if viewer is None:
        return
    if hasattr(viewer, "cam"):
        viewer.cam.type = 0
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = marker_pos
        viewer.cam.distance = args.contact_camera_distance
        viewer.cam.azimuth = args.contact_camera_azimuth
        viewer.cam.elevation = args.contact_camera_elevation

    started_at = time.time()
    sim_step_index = 0
    try:
        for q_target in arm_trajectory:
            for _ in range(sim_steps_per_target):
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    return
                _step_arm_pd(
                    ctx.env, q_target, qpos_addrs, dof_addrs, actuator_ids, kp, kd
                )
                sim_step_index += 1
                if hasattr(viewer, "user_scn") and sim_step_index % sync_interval == 0:
                    viewer.user_scn.ngeom = 0
                    _draw_viewer_sphere(
                        viewer, marker_pos, args.contact_marker_size, marker_rgba
                    )
                    if ctx.mink_solution is not None:
                        ee_marker_pos = _ee_sample_world(
                            ctx.env,
                            frame_name,
                            ctx.mink_solution.contact_offset_local,
                        )
                        _draw_viewer_sphere(
                            viewer,
                            ee_marker_pos,
                            args.contact_marker_size * 0.75,
                            ee_marker_rgba,
                        )
                if hasattr(viewer, "sync") and sim_step_index % sync_interval == 0:
                    viewer.sync()
                if (
                    args.visualize_contact_seconds > 0
                    and time.time() - started_at >= args.visualize_contact_seconds
                ):
                    return
        hold_steps = int(
            round(
                max(float(getattr(args, "autogen_pd_hold_seconds", 1.0)), 0.0)
                / timestep
            )
        )
        q_target = arm_trajectory[-1]
        for _ in range(hold_steps):
            if hasattr(viewer, "is_running") and not viewer.is_running():
                return
            _step_arm_pd(ctx.env, q_target, qpos_addrs, dof_addrs, actuator_ids, kp, kd)
            sim_step_index += 1
            if hasattr(viewer, "sync") and sim_step_index % sync_interval == 0:
                viewer.sync()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        autogen_print(f"Physical PD contact visualization stopped: {exc}")
