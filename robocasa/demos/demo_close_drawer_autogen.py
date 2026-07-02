import argparse
import functools
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions_gcc11")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("ROBOCASA_ALLOW_VERSION_MISMATCH", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
python_bin_dir = str(Path(sys.executable).resolve().parent)
path_entries = os.environ.get("PATH", "").split(os.pathsep)
if python_bin_dir not in path_entries:
    os.environ["PATH"] = os.pathsep.join([python_bin_dir] + path_entries)
if Path("/usr/bin/g++-11").exists():
    os.environ.setdefault("CXX", "/usr/bin/g++-11")
elif shutil.which("c++") is None and Path("/usr/bin/g++").exists():
    os.environ.setdefault("CXX", "/usr/bin/g++")
if Path("/usr/bin/gcc-11").exists():
    os.environ.setdefault("CC", "/usr/bin/gcc-11")
elif shutil.which("cc") is None and Path("/usr/bin/gcc").exists():
    os.environ.setdefault("CC", "/usr/bin/gcc")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import robocasa.demos.demo_close_drawer_contact_curobo as close_demo


def _autogen_print(message):
    print(message, file=sys.__stdout__, flush=True)


def _hand_box_from_ghost_geoms(env, frame_name):
    from robocasa.demos import visualize_mujoco as viz_mj

    geoms = viz_mj._extract_hand_finger_ghost_geoms(env, frame_name)
    best = None
    best_volume = -1.0
    for ghost in geoms:
        half = np.asarray(ghost.size, dtype=np.float64).reshape(3)
        # Fingers are long and thin; the palm/hand mesh is the largest compact
        # box among the EE ghost geoms. Reject implausibly large scene links if
        # the source-body filter ever becomes too broad.
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


def _visualize_mink_q_poses_popup(env, q_waypoints, robot_state, args, drawer_q=None):
    if not bool(getattr(args, "autogen_visualize_mink_poses", True)):
        return
    q_waypoints = np.asarray(q_waypoints, dtype=np.float64)
    if q_waypoints.size == 0:
        return
    q_waypoints = q_waypoints.reshape(-1, 7)
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    if frame_name not in env.sim.model._site_name2id:
        raise RuntimeError(
            f"Cannot visualize mink q poses: site {frame_name!r} not found"
        )

    try:
        import mujoco
        import mujoco.viewer
        from robocasa.demos import visualize_mujoco as viz_mj
    except Exception as exc:
        raise RuntimeError(f"Cannot visualize mink q poses: {exc}") from exc

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    site_id = int(env.sim.model.site_name2id(frame_name))
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    drawer_q = (
        float(close_demo._drawer_joint_value(env))
        if drawer_q is None
        else float(drawer_q)
    )
    poses = []
    try:
        close_demo._set_drawer_joint_value(env, drawer_q)
        ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(env, frame_name)
        for q_arm in q_waypoints:
            close_demo._set_env_arm_q(env, arm_joint_names, q_arm)
            close_demo._set_drawer_joint_value(env, drawer_q)
            env.sim.forward()
            poses.append(
                (
                    np.asarray(
                        env.sim.data.site_xpos[site_id], dtype=np.float64
                    ).copy(),
                    np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64)
                    .reshape(3, 3)
                    .copy(),
                )
            )
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()

    if not poses:
        return
    lookat = np.mean(np.asarray([pose[0] for pose in poses], dtype=np.float64), axis=0)
    palette = np.asarray(
        [
            [0.05, 0.45, 1.0, float(getattr(args, "autogen_mink_ghost_alpha", 0.28))],
            [1.0, 0.58, 0.05, float(getattr(args, "autogen_mink_ghost_alpha", 0.28))],
            [0.05, 0.75, 0.35, float(getattr(args, "autogen_mink_ghost_alpha", 0.28))],
        ],
        dtype=np.float32,
    )
    with mujoco.viewer.launch_passive(
        raw_model,
        raw_data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        try:
            viewer.opt.geomgroup[:] = 0
            viewer.opt.geomgroup[1] = 1
            viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT)] = 0
            viewer.opt.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE)] = 0
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
                            viewer.user_scn,
                            ghost,
                            target_pos,
                            target_rot,
                            rgba,
                        )
            viewer.sync()
            time.sleep(1.0 / fps)


def _visualize_floating_ee_poses_popup(env, refined_poses, args, drawer_q=None):
    """Pop a passive viewer rendering ghost Panda hand at every FloatingEEMPPI pose."""
    if not bool(getattr(args, "autogen_visualize_floating_ee", True)):
        return
    if not refined_poses:
        return
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    try:
        import mujoco
        import mujoco.viewer
        from robocasa.demos import visualize_mujoco as viz_mj
    except Exception as exc:
        print(f"Floating-EE viz skipped: {exc}")
        return

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    try:
        from robocasa.demos import ee_skelton as _ee_skelton
    except Exception:
        _ee_skelton = None
    skeleton = None
    hand_local_pos = None
    hand_local_rot = None
    hand_half_ext = None
    try:
        if drawer_q is not None:
            close_demo._set_drawer_joint_value(env, float(drawer_q))
        env.sim.forward()
        ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(env, frame_name)
        if _ee_skelton is not None:
            try:
                skeleton = _ee_skelton.build_panda_skeleton(env, frame_name)
            except Exception as exc:
                print(f"[autogen] floating-EE skeleton build failed: {exc}", flush=True)
        # Pull panda_hand bbox center/local_rot in EE-site frame. Finger
        # capsules are segment endpoints, but mesh hand boxes need their bbox
        # center rather than geom_xpos because the mesh origin can be offset.
        try:
            import mujoco as _mj

            raw_model_ext, raw_data_ext = viz_mj._raw_model_data(env)
            _, _, body_id2name_ext, geom_id2name_ext = viz_mj._name_maps(env)
            site_pos_ext, site_rot_ext = viz_mj._site_pose(env, frame_name)
            # Only consider bodies belonging to the arm's EE subtree — this
            # excludes drawer_handle / cabinet_handle / etc. whose names contain
            # the substring "hand".
            ee_body_ids = viz_mj._ghost_source_body_ids(env)

            def _is_panda_hand(geom_name: str, body_name: str) -> bool:
                gl = geom_name.lower()
                bl = body_name.lower()
                if "finger" in gl or "finger" in bl:
                    return False
                for tok in ("panda_hand", "right_hand", "left_hand", "gripper_hand"):
                    if tok in gl or tok in bl:
                        return True
                # Fallback: body named exactly "hand" or ending in "_hand".
                return bl == "hand" or bl.endswith("_hand")

            def _geom_bbox_half_and_center_local(model, geom_id, fallback_size):
                if int(model.geom_type[geom_id]) == int(_mj.mjtGeom.mjGEOM_MESH):
                    try:
                        mesh_id = int(model.geom_dataid[geom_id])
                        vadr = int(model.mesh_vertadr[mesh_id])
                        vnum = int(model.mesh_vertnum[mesh_id])
                        verts = np.asarray(
                            model.mesh_vert[vadr : vadr + vnum], dtype=np.float64
                        )
                        if verts.size:
                            mesh_min = np.min(verts, axis=0)
                            mesh_max = np.max(verts, axis=0)
                            half = 0.5 * (mesh_max - mesh_min)
                            center = 0.5 * (mesh_max + mesh_min)
                            return (
                                np.maximum(
                                    half,
                                    np.array([0.003, 0.003, 0.003], dtype=np.float64),
                                ),
                                center,
                            )
                    except Exception:
                        pass
                half = np.maximum(
                    np.asarray(fallback_size, dtype=np.float64).reshape(3),
                    np.array([0.003, 0.003, 0.003], dtype=np.float64),
                )
                return half, np.zeros(3, dtype=np.float64)

            best_prod = -1.0
            for gid in range(int(raw_model_ext.ngeom)):
                body_id = int(raw_model_ext.geom_bodyid[gid])
                if body_id not in ee_body_ids:
                    continue
                geom_name = geom_id2name_ext.get(gid, "")
                body_name = body_id2name_ext.get(body_id, "")
                if not _is_panda_hand(geom_name, body_name):
                    continue
                geom_type = int(raw_model_ext.geom_type[gid])
                if geom_type == int(_mj.mjtGeom.mjGEOM_PLANE):
                    continue
                half, bbox_center_local = _geom_bbox_half_and_center_local(
                    raw_model_ext, gid, raw_model_ext.geom_size[gid]
                )
                prod = float(np.prod(np.maximum(half, 1e-6)))
                if prod <= best_prod:
                    continue
                gpos_w = np.asarray(
                    raw_data_ext.geom_xpos[gid], dtype=np.float64
                ).reshape(3)
                grot_w = np.asarray(
                    raw_data_ext.geom_xmat[gid], dtype=np.float64
                ).reshape(3, 3)
                bbox_center_w = gpos_w + grot_w @ bbox_center_local
                hand_local_pos = site_rot_ext.T @ (bbox_center_w - site_pos_ext)
                hand_local_rot = site_rot_ext.T @ grot_w
                hand_half_ext = half
                best_prod = prod
            print(
                f"[autogen] panda_hand geom lookup: found={hand_local_pos is not None}",
                flush=True,
            )
            ghost_hand_box = _hand_box_from_ghost_geoms(env, frame_name)
            if ghost_hand_box is not None:
                hand_local_pos, hand_local_rot, hand_half_ext = ghost_hand_box
                print(
                    "[autogen] panda_hand ghost override: "
                    f"local_pos={hand_local_pos} half={hand_half_ext}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[autogen] panda_hand geom lookup failed: {exc}", flush=True)
    finally:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()

    limit = max(int(getattr(args, "autogen_visualize_floating_ee_limit", 12)), 1)
    poses = refined_poses[:limit]
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
        raw_model,
        raw_data,
        show_left_ui=False,
        show_right_ui=False,
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
        gripper_opening_default = float(
            getattr(
                args,
                "autogen_skeleton_gripper_default",
                getattr(_ee_skelton, "PANDA_DEFAULT_GRIPPER_OPENING", 0.04)
                if _ee_skelton is not None
                else 0.04,
            )
        )
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
                            if (
                                hand_local_pos is not None
                                and hand_local_rot is not None
                                and hand_half_ext is not None
                            ):
                                # Use the mesh bbox center, not geom_xpos. The
                                # latter is the geom origin; for mesh geoms it
                                # can be offset from the visual box center.
                                hand_center_w = ee_pos + ee_rot @ hand_local_pos
                                hand_rot_w = ee_rot @ hand_local_rot
                                half_ext_full = np.asarray(
                                    hand_half_ext, dtype=np.float64
                                ).reshape(3)
                            else:
                                # Fallback: skeleton's bbox-center parameterization.
                                half_ext_full = (
                                    np.asarray(
                                        skeleton.hand_box_half_extents_ee,
                                        dtype=np.float64,
                                    )
                                    .reshape(3)
                                    .copy()
                                )
                                hand_center_w = ee_pos + ee_rot @ np.asarray(
                                    skeleton.hand_box_center_ee, dtype=np.float64
                                )
                                hand_rot_w = ee_rot @ np.asarray(
                                    skeleton.hand_box_rotation_ee, dtype=np.float64
                                )
                            half_ext_full = (
                                np.asarray(half_ext_full, dtype=np.float64)
                                .reshape(3)
                                .copy()
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
                            g_open = (
                                float(_g) if _g is not None else gripper_opening_default
                            )
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
                        except Exception as exc:
                            print(
                                f"[autogen] floating-EE hand-box overlay skipped: {exc}",
                                flush=True,
                            )
            viewer.sync()
            time.sleep(1.0 / fps)


def _arm_pd_control_maps(env, arm_joint_names):
    model = env.sim.model
    joint_ids = [int(model.joint_name2id(name)) for name in arm_joint_names]
    qpos_addrs = np.asarray(
        [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids], dtype=np.int64
    )
    dof_addrs = np.asarray(
        [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids], dtype=np.int64
    )
    actuator_ids = []
    for joint_id, joint_name in zip(joint_ids, arm_joint_names):
        matches = [
            actuator_id
            for actuator_id in range(int(model.nu))
            if int(model.actuator_trnid[actuator_id, 0]) == joint_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one actuator for arm joint {joint_name!r}; found {len(matches)}"
            )
        actuator_ids.append(matches[0])
    return qpos_addrs, dof_addrs, np.asarray(actuator_ids, dtype=np.int64)


def _step_arm_pd(env, q_target, qpos_addrs, dof_addrs, actuator_ids, kp, kd):
    data = env.sim.data
    model = env.sim.model
    q_target = np.asarray(q_target, dtype=np.float64).reshape(7)
    q = np.asarray(data.qpos[qpos_addrs], dtype=np.float64)
    qd = np.asarray(data.qvel[dof_addrs], dtype=np.float64)
    bias = np.asarray(data.qfrc_bias[dof_addrs], dtype=np.float64)
    torque = (
        np.asarray(kp, dtype=np.float64) * (q_target - q)
        - np.asarray(kd, dtype=np.float64) * qd
        + bias
    )
    ctrlrange = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64)
    torque = np.clip(torque, ctrlrange[:, 0], ctrlrange[:, 1])
    data.ctrl[actuator_ids] = torque
    env.sim.step()


def _visualize_contact_marker_with_physical_pd(
    env,
    panel,
    selected,
    push_distance,
    robot_state,
    q_traj,
    mink_solution,
    all_mink_solutions,
    args,
    fallback_visualizer,
):
    if not bool(getattr(args, "autogen_physical_pd_visualization", True)):
        return fallback_visualizer(
            env,
            panel,
            selected,
            push_distance,
            robot_state,
            q_traj,
            mink_solution,
            all_mink_solutions,
            args,
        )
    if q_traj is None or np.asarray(q_traj).size == 0:
        return fallback_visualizer(
            env,
            panel,
            selected,
            push_distance,
            robot_state,
            q_traj,
            mink_solution,
            all_mink_solutions,
            args,
        )
    if env.viewer is None:
        print("Contact visualization skipped: renderer was not initialized.")
        return

    arm_trajectory = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    qpos_addrs, dof_addrs, actuator_ids = _arm_pd_control_maps(env, arm_joint_names)
    kp = np.full(7, float(getattr(args, "autogen_pd_kp", 350.0)), dtype=np.float64)
    kd = np.full(7, float(getattr(args, "autogen_pd_kd", 35.0)), dtype=np.float64)
    timestep = float(env.sim.model.opt.timestep)
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
    marker_pos = selected.world_point + panel.outward_world * args.contact_marker_offset
    marker_rgba = np.array([1.0, 0.0, 0.0, args.contact_marker_alpha], dtype=np.float32)
    ee_marker_rgba = np.array([0.0, 1.0, 0.15, 1.0], dtype=np.float32)

    print(
        "Showing cuRobo trajectory with physical PD torque control "
        f"(frames={arm_trajectory.shape[0]}, sim_steps_per_target={sim_steps_per_target}, "
        f"kp={float(kp[0]):.1f}, kd={float(kd[0]):.1f}). "
        "Close the viewer window or press Ctrl+C to exit.",
        flush=True,
    )
    try:
        env.viewer.update()
    except Exception as exc:
        print(f"Contact visualization skipped: viewer launch failed: {exc}")
        return
    viewer = getattr(env.viewer, "viewer", None)
    if viewer is None:
        print("Contact visualization skipped: viewer launch failed.")
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
                _step_arm_pd(env, q_target, qpos_addrs, dof_addrs, actuator_ids, kp, kd)
                sim_step_index += 1
                if hasattr(viewer, "user_scn") and sim_step_index % sync_interval == 0:
                    viewer.user_scn.ngeom = 0
                    close_demo._draw_viewer_sphere(
                        viewer,
                        marker_pos,
                        args.contact_marker_size,
                        marker_rgba,
                    )
                    if mink_solution is not None:
                        ee_marker_pos = close_demo._ee_sample_world(
                            env,
                            args.mink_contact_frame,
                            mink_solution.contact_offset_local,
                        )
                        close_demo._draw_viewer_sphere(
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
        hold_seconds = max(float(getattr(args, "autogen_pd_hold_seconds", 1.0)), 0.0)
        hold_steps = int(round(hold_seconds / timestep))
        q_target = arm_trajectory[-1]
        for _ in range(hold_steps):
            if hasattr(viewer, "is_running") and not viewer.is_running():
                return
            _step_arm_pd(env, q_target, qpos_addrs, dof_addrs, actuator_ids, kp, kd)
            sim_step_index += 1
            if hasattr(viewer, "sync") and sim_step_index % sync_interval == 0:
                viewer.sync()
        while (
            bool(getattr(args, "visualize_loop_trajectory", False))
            and hasattr(viewer, "is_running")
            and viewer.is_running()
        ):
            if hasattr(viewer, "sync"):
                viewer.sync()
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Physical PD contact visualization stopped: {exc}")


def _parse_close_autogen_args(original_parse_args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--autogen-visualize-mink-poses",
        dest="autogen_visualize_mink_poses",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-mink-poses",
        dest="autogen_visualize_mink_poses",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_mink_poses=False)
    parser.add_argument("--autogen-mink-ghost-alpha", type=float, default=0.28)
    parser.add_argument(
        "--autogen-mink-popup-camera-distance", type=float, default=0.85
    )
    parser.add_argument(
        "--autogen-mink-popup-camera-azimuth", type=float, default=135.0
    )
    parser.add_argument(
        "--autogen-mink-popup-camera-elevation", type=float, default=-25.0
    )
    parser.add_argument("--autogen-mink-popup-fps", type=float, default=30.0)
    parser.add_argument(
        "--autogen-visualize-skeleton",
        dest="autogen_visualize_skeleton",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-skeleton",
        dest="autogen_visualize_skeleton",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_skeleton=True)
    parser.add_argument(
        "--autogen-visualize-skeleton-poses",
        dest="autogen_visualize_skeleton_poses",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-skeleton-poses",
        dest="autogen_visualize_skeleton_poses",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_skeleton_poses=True)
    parser.add_argument(
        "--autogen-visualize-execution",
        dest="autogen_visualize_execution",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-execution",
        dest="autogen_visualize_execution",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_execution=True)
    parser.add_argument(
        "--autogen-physical-pd-visualization",
        dest="autogen_physical_pd_visualization",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-physical-pd-visualization",
        dest="autogen_physical_pd_visualization",
        action="store_false",
    )
    parser.set_defaults(autogen_physical_pd_visualization=True)
    parser.add_argument("--autogen-pd-kp", type=float, default=350.0)
    parser.add_argument("--autogen-pd-kd", type=float, default=35.0)
    parser.add_argument("--autogen-pd-target-dt", type=float, default=0.02)
    parser.add_argument("--autogen-pd-hold-seconds", type=float, default=1.0)
    parser.add_argument(
        "--autogen-visualize-floating-ee",
        dest="autogen_visualize_floating_ee",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-floating-ee",
        dest="autogen_visualize_floating_ee",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_floating_ee=True)
    parser.add_argument("--autogen-visualize-floating-ee-limit", type=int, default=12)
    parser.add_argument("--autogen-floating-ee-ghost-alpha", type=float, default=0.32)
    parser.add_argument("--autogen-mink-parallel-workers", type=int, default=None)
    parser.add_argument("--autogen-mink-q-checker-workers", type=int, default=None)
    parser.add_argument("--autogen-mink-q-worlds-per-worker", type=int, default=None)
    parser.add_argument(
        "--solve_step2",
        "--solve-step2",
        dest="solve_step2",
        type=str,
        default=None,
        choices=("MPPI", "mink", "mppi", "MINK"),
    )
    popup_args, remaining = parser.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = original_parse_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(popup_args).items():
        if value is None:
            continue
        setattr(args, key, value)
    if not any(str(arg).startswith("--mink-solver") for arg in old_argv[1:]):
        args.mink_solver = "daqp"
    user_set_mink_workers = any(
        str(arg) == "--autogen-mink-parallel-workers"
        or str(arg).startswith("--autogen-mink-parallel-workers=")
        for arg in old_argv[1:]
    )
    # Show the skeleton-pose popup only after the DAQP batch finishes.  The
    # raw skeleton/EE geometry popup is intentionally disabled by default here
    # because it appears before DAQP and makes the progress-bar timing unclear.
    # Force the QP path with multi-theta sampling so contact rotations are not
    # collapsed to the demo EE orientation via the R0_override bypass in
    # ee_skelton.solve_skeleton_pose.
    # If the free-rotation QP yields zero poses (current symptom), let
    # _solve_contact_poses_with_skeleton fall back to R0_override automatically
    # rather than silently dropping into the geometric build_target_gripper_poses
    # path — see the skeleton_rotation_fallback=current_ee_rotation log line.
    args.autogen_visualize_skeleton = False
    args.autogen_visualize_skeleton_poses = True
    args.autogen_use_current_ee_rotation = False
    args.autogen_allow_current_rotation_fallback = True
    # Keep the drawer/handle convex penetration check enabled by default. The
    # contact tangent plane alone is not a volume collision constraint, and it
    # can accept skeleton poses whose finger capsules or hand box visibly pass
    # through the drawer.
    if not hasattr(args, "autogen_skeleton_disable_handle_convex"):
        args.autogen_skeleton_disable_handle_convex = False
    # Suppress the per-skeleton-pose step-2 solver status prints (yellow flood)
    # and show a tqdm progress bar instead; after solving, a single yellow line
    # reports total duration + feasible q count. Direct demo invocations keep the
    # detailed per-pose logs because the default (True) is unchanged.
    args.autogen_step2_verbose_poses = False
    args.autogen_skeleton_daqp_verbose = False
    if "--autogen-mink-q-debug" not in old_argv:
        args.autogen_mink_q_debug = False
    args.autogen_skeleton_object_penetration_tol = min(
        float(getattr(args, "autogen_skeleton_object_penetration_tol", 0.001)),
        0.001,
    )
    args.autogen_skeleton_clearance_tolerance = min(
        float(getattr(args, "autogen_skeleton_clearance_tolerance", 0.001)),
        0.001,
    )
    args.autogen_qmppi_penetration_threshold = min(
        float(getattr(args, "autogen_qmppi_penetration_threshold", 0.005)),
        float(getattr(args, "contact_standoff", 0.005)),
        float(getattr(args, "mink_collision_penetration_tolerance", 0.02)),
        0.001,
    )
    args.mink_collision_penetration_tolerance = min(
        float(getattr(args, "mink_collision_penetration_tolerance", 0.02)),
        0.001,
    )
    args.autogen_qmppi_accept_object_improvement_only = True
    # (Fix #1) pen_weight was left at 0.0 so MPPI's cost function was
    # `costs = 0.0 * penetration^2 + contact_weight * d^2 + tracking`. The
    # solver therefore only minimized tracking error and could never push an
    # EE pose out of penetration — every sampled pose was reported as
    # penetrating the drawer. A weight comparable to track_pos_weight (200)
    # scaled by the ratio of typical penetration depth (~0.01 m) to tracking
    # displacement makes the solver resolve penetration actively.
    args.autogen_qmppi_penetration_weight = 500.0
    args.autogen_qmppi_contact_weight = 0.0
    if str(getattr(args, "solve_step2", "MPPI")).strip().lower() == "mppi":
        args.autogen_skip_mink_q_after_mppi = True
    default_mink_workers = max(1, (os.cpu_count() or 1))
    if not user_set_mink_workers:
        args.autogen_mink_parallel_workers = default_mink_workers
    if (
        str(getattr(args, "solve_step2", "MPPI")).strip().lower() == "mink"
        and not bool(getattr(args, "autogen_mink_q_debug", False))
        and not user_set_mink_workers
    ):
        args.autogen_mink_parallel_workers = max(2, default_mink_workers)
    args.curobo_use_mujoco_world = True
    args.curobo_world_exclude_target_drawer = True
    # (Fix #2) Drive the world-padding for non-box obstacle geoms to zero so
    # the planner does not waste free space around the COACD-decomposed fixtures
    # (handle connectors / bolts that protrude from the panel). MuJoCo geoms are
    # already precise; AABB padding here was redundant and inflated them into
    # free air.
    args.curobo_world_padding = min(
        float(getattr(args, "curobo_world_padding", 0.005)), 0.0
    )
    if bool(getattr(args, "autogen_visualize_execution", True)):
        args.visualize_contact = True
    args.save_output = False
    args.save_trajectory_videos = False
    if float(getattr(args, "contact_cost_threshold", 0.10)) == 0.10:
        args.contact_cost_threshold = 0.35
    if not hasattr(args, "autogen_panel_edge_margin"):
        args.autogen_panel_edge_margin = 0.015
    if float(getattr(args, "autogen_panel_edge_margin", 0.015)) in (0.015, 0.03):
        args.autogen_panel_edge_margin = 0.05
    if not hasattr(args, "autogen_panel_edge_fraction"):
        args.autogen_panel_edge_fraction = 0.28
    if float(getattr(args, "autogen_panel_edge_fraction", 0.18)) == 0.18:
        args.autogen_panel_edge_fraction = 0.28
    if not hasattr(args, "autogen_panel_top_edge_fraction"):
        args.autogen_panel_top_edge_fraction = 0.38
    if not hasattr(args, "autogen_skeleton_pose_variants_per_contact"):
        args.autogen_skeleton_pose_variants_per_contact = 8
    elif int(getattr(args, "autogen_skeleton_pose_variants_per_contact", 4)) == 4:
        args.autogen_skeleton_pose_variants_per_contact = 8
    if not hasattr(args, "autogen_skeleton_pose_min_theta_separation"):
        args.autogen_skeleton_pose_min_theta_separation = float(np.pi / 6.0)
    if not hasattr(args, "autogen_visualize_skeleton_pose_limit"):
        args.autogen_visualize_skeleton_pose_limit = 120
    return args


def main():
    original_parse_args = close_demo.parse_args
    original_solve_contact_poses = close_demo.solve_contact_poses_with_mink
    original_solve_mink_for_drawer_candidate = (
        close_demo._solve_mink_for_drawer_candidate
    )
    original_solve_skeleton_precontact_q = close_demo.mink_q.solve_skeleton_precontact_q
    original_plan_with_curobo = close_demo.plan_with_curobo
    original_visualize_contact_marker = close_demo.visualize_contact_marker
    original_skeleton_solver = getattr(
        close_demo, "_solve_contact_poses_with_skeleton", None
    )
    visualized = {"done": False}

    def parse_args_with_popup():
        return _parse_close_autogen_args(original_parse_args)

    def solve_contact_poses_with_popup(
        env, panel, candidates, push_distance, robot_state, args
    ):
        result = original_solve_contact_poses(
            env,
            panel,
            candidates,
            push_distance,
            robot_state,
            args,
        )
        mink_solution = result[0]
        if (
            mink_solution is not None
            and not visualized["done"]
            and bool(getattr(args, "autogen_visualize_mink_poses", False))
        ):
            _visualize_mink_q_poses_popup(
                env,
                getattr(mink_solution, "q_waypoints", np.zeros((0, 7))),
                robot_state,
                args,
                drawer_q=float(close_demo._drawer_joint_value(env)),
            )
            visualized["done"] = True
        return result

    def skeleton_solver_with_marker(
        env, panel, candidates, push_distance, robot_state, args
    ):
        _autogen_print(
            "close_autogen_patch=skeleton "
            f"edge_margin={float(getattr(args, 'autogen_panel_edge_margin', 0.015)):.4f} "
            f"edge_fraction={float(getattr(args, 'autogen_panel_edge_fraction', 0.18)):.4f} "
            f"top_edge_fraction={float(getattr(args, 'autogen_panel_top_edge_fraction', 0.38)):.4f} "
            f"variants_per_contact={int(getattr(args, 'autogen_skeleton_pose_variants_per_contact', 4))} "
            f"theta_sep={float(getattr(args, 'autogen_skeleton_pose_min_theta_separation', np.pi / 6.0)):.4f}"
        )
        # Reset the shared refined-poses buffer so the popup reflects THIS run's
        # MPPI output only (the underlying module clears it on entry too, but the
        # autogen wrapper runs as a different call stack, so be explicit here).
        LAST_FLOATING_REFINED_POSES = getattr(
            close_demo, "LAST_FLOATING_REFINED_POSES", None
        )
        if LAST_FLOATING_REFINED_POSES is not None and hasattr(
            LAST_FLOATING_REFINED_POSES, "clear"
        ):
            LAST_FLOATING_REFINED_POSES.clear()
        # Lazily build the per-worker cloned-MjData pool used by the DAQP inner
        # loop. Sharing env.sim.data across worker threads is only nominally
        # safe (mj_ray is read-only but the underlying MjData is not); the pool
        # gives each worker its own MjData 1:1 cloned from env, and re-syncs on
        # every DAQP batch.
        if getattr(args, "autogen_skeleton_scene_pool", None) is None:
            skeleton_workers = getattr(args, "autogen_skeleton_parallel_workers", None)
            if skeleton_workers is None:
                skeleton_workers = getattr(args, "autogen_mink_parallel_workers", 1)
            workers = max(1, int(skeleton_workers or 1))
            try:
                from robocasa.demos.skelton_scene_mjwarp import SkeletonScenePool

                args.autogen_skeleton_scene_pool = SkeletonScenePool.from_env(
                    env, num_workers=workers
                )
            except Exception as exc:
                print(f"[autogen] skeleton scene pool init failed: {exc}", flush=True)
                args.autogen_skeleton_scene_pool = None
        mppi_started = time.perf_counter()
        result = original_skeleton_solver(
            env,
            panel,
            candidates,
            push_distance,
            robot_state,
            args,
        )
        mppi_elapsed = time.perf_counter() - mppi_started
        solution = result[0]
        if solution is not None:
            _autogen_print(
                "close_autogen_patch=selected_skeleton "
                f"candidate={int(solution.drawer_candidate_index)} "
                f"theta={float(getattr(solution, 'roll_angle', 0.0)):.4f} "
                f"cost={float(solution.drawer_contact_cost):.6f}"
            )
        refined = list(getattr(close_demo, "LAST_FLOATING_REFINED_POSES", []) or [])
        step2_label = str(getattr(args, "solve_step2", "MPPI")).strip().lower()
        print(
            "\033[93m"
            f"close_autogen_patch=step2_{step2_label} "
            f"refined_pose_count={len(refined)} "
            f"solve_s={mppi_elapsed:.3f}"
            "\033[0m",
            flush=True,
        )
        if refined and bool(getattr(args, "autogen_visualize_floating_ee", True)):
            _visualize_floating_ee_poses_popup(
                env,
                refined,
                args,
                drawer_q=float(close_demo._drawer_joint_value(env)),
            )
        return result

    def plan_with_curobo_with_stats(
        robot_state, target_hand_poses_base, args, *extra_args, **extra_kwargs
    ):
        started = time.perf_counter()
        try:
            q_traj, segments = original_plan_with_curobo(
                robot_state,
                target_hand_poses_base,
                args,
                *extra_args,
                **extra_kwargs,
            )
        except Exception as exc:
            # Autogen's useful output is the parallel mink/contact solve.  Treat
            # cuRobo graph/trajopt failures as a non-fatal planning miss so the
            # run can still inspect and visualize the known mink q solution.
            print(
                "\033[93m"
                "curobo_status=failed_nonfatal "
                f"curobo_time={time.perf_counter() - started:.6f} "
                f"error={type(exc).__name__}: {exc}"
                "\033[0m",
                flush=True,
            )
            return None, [
                {
                    "name": "curobo_failed_nonfatal",
                    "planner": "curobo",
                    "status": "failed_nonfatal",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ]
        world_obstacle_count = 0
        world_model = "none"
        world_exclude_bodies = ()
        if segments:
            world_obstacle_count = int(segments[0].get("world_obstacle_count", 0))
            world_model = str(segments[0].get("world_collision_model", "none"))
            world_exclude_bodies = tuple(segments[0].get("world_exclude_bodies", ()))
        print(
            "\033[93m"
            "curobo_time="
            f"{time.perf_counter() - started:.6f} "
            f"successful_trajectories={int(len(segments))} "
            f"world_collision_model={world_model} "
            f"world_obstacle_count={world_obstacle_count} "
            f"world_exclude_bodies={world_exclude_bodies}"
            "\033[0m",
            flush=True,
        )
        return q_traj, segments

    try:
        from robocasa.demos import ee_skelton as _ee_skelton
    except Exception:
        _ee_skelton = None
    original_draw_skeleton = (
        getattr(_ee_skelton, "_draw_skeleton_into_scene", None)
        if _ee_skelton is not None
        else None
    )
    original_build_skeleton = (
        getattr(_ee_skelton, "build_panda_skeleton", None)
        if _ee_skelton is not None
        else None
    )

    def build_panda_skeleton_with_ghost_hand(env, ee_site_name):
        skeleton = original_build_skeleton(env, ee_site_name)
        try:
            ghost_hand_box = _hand_box_from_ghost_geoms(
                env, str(ee_site_name).split(":")[0]
            )
            if ghost_hand_box is None:
                return skeleton
            hand_local_pos, hand_local_rot, hand_half_ext = ghost_hand_box
            return _ee_skelton.EESkeleton(
                hand_box_half_extents_ee=np.asarray(hand_half_ext, dtype=np.float64)
                .reshape(3)
                .copy(),
                hand_box_center_ee=np.asarray(hand_local_pos, dtype=np.float64)
                .reshape(3)
                .copy(),
                hand_box_rotation_ee=np.asarray(hand_local_rot, dtype=np.float64)
                .reshape(3, 3)
                .copy(),
                left_finger_segment_ee=np.asarray(
                    skeleton.left_finger_segment_ee, dtype=np.float64
                ).copy(),
                right_finger_segment_ee=np.asarray(
                    skeleton.right_finger_segment_ee, dtype=np.float64
                ).copy(),
                finger_tip_offset_ee=np.asarray(
                    skeleton.finger_tip_offset_ee, dtype=np.float64
                ).copy(),
            )
        except Exception as exc:
            print(f"[autogen] ghost hand skeleton override skipped: {exc}", flush=True)
            return skeleton

    def draw_skeleton_with_flat_hand_box(
        scene,
        skeleton,
        ee_pos,
        ee_rot,
        gripper_opening,
        rgba_hand,
        rgba_finger,
        finger_radius,
    ):
        # Original draw without ee_skelton.py's verbose [skeleton_draw]
        # center_w diagnostic.
        try:
            half_ext_flat = _ee_skelton._flat_hand_half_extents(skeleton)
            flat_center_w = np.asarray(ee_pos, dtype=np.float64) + np.asarray(
                ee_rot, dtype=np.float64
            ) @ np.asarray(skeleton.hand_box_center_ee, dtype=np.float64)
            flat_rot_w = np.asarray(ee_rot, dtype=np.float64) @ np.asarray(
                skeleton.hand_box_rotation_ee, dtype=np.float64
            )
            _ee_skelton._add_box(
                scene,
                flat_center_w,
                flat_rot_w,
                half_ext_flat,
                rgba_hand,
            )
            left_seg, right_seg, _, _ = _ee_skelton._finger_segments_with_opening(
                skeleton, gripper_opening
            )
            for seg in (left_seg, right_seg):
                sa = np.asarray(ee_pos, dtype=np.float64) + np.asarray(
                    ee_rot, dtype=np.float64
                ) @ np.asarray(seg[0], dtype=np.float64)
                sb = np.asarray(ee_pos, dtype=np.float64) + np.asarray(
                    ee_rot, dtype=np.float64
                ) @ np.asarray(seg[1], dtype=np.float64)
                _ee_skelton._add_capsule_segment(
                    scene, sa, sb, finger_radius, rgba_finger
                )
        except Exception as exc:
            print(f"[autogen] skeleton draw skipped: {exc}", flush=True)

    if original_draw_skeleton is not None:
        _ee_skelton._draw_skeleton_into_scene = draw_skeleton_with_flat_hand_box
    if original_build_skeleton is not None:
        _ee_skelton.build_panda_skeleton = build_panda_skeleton_with_ghost_hand

    close_demo.parse_args = parse_args_with_popup
    close_demo._solve_mink_for_drawer_candidate = functools.partial(
        close_demo.mink_q.solve_mink_for_drawer_candidate_parallel,
        close_ops=close_demo,
        original_solver=original_solve_mink_for_drawer_candidate,
        status_printer=_autogen_print,
    )
    close_demo.mink_q.solve_skeleton_precontact_q = (
        close_demo.mink_q.solve_skeleton_precontact_q_parallel
    )
    close_demo.solve_contact_poses_with_mink = solve_contact_poses_with_popup
    close_demo.plan_with_curobo = plan_with_curobo_with_stats
    close_demo.visualize_contact_marker = (
        lambda *call_args, **call_kwargs: _visualize_contact_marker_with_physical_pd(
            *call_args,
            **call_kwargs,
            fallback_visualizer=original_visualize_contact_marker,
        )
    )
    if original_skeleton_solver is not None:
        close_demo._solve_contact_poses_with_skeleton = skeleton_solver_with_marker
    try:
        close_demo.main()
    finally:
        close_demo.parse_args = original_parse_args
        close_demo._solve_mink_for_drawer_candidate = (
            original_solve_mink_for_drawer_candidate
        )
        close_demo.mink_q.solve_skeleton_precontact_q = (
            original_solve_skeleton_precontact_q
        )
        close_demo.solve_contact_poses_with_mink = original_solve_contact_poses
        close_demo.plan_with_curobo = original_plan_with_curobo
        close_demo.visualize_contact_marker = original_visualize_contact_marker
        if original_skeleton_solver is not None:
            close_demo._solve_contact_poses_with_skeleton = original_skeleton_solver
        if original_draw_skeleton is not None and _ee_skelton is not None:
            _ee_skelton._draw_skeleton_into_scene = original_draw_skeleton
        if original_build_skeleton is not None and _ee_skelton is not None:
            _ee_skelton.build_panda_skeleton = original_build_skeleton


if __name__ == "__main__":
    main()
