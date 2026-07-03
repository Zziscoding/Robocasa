import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np

_QUIET_IMPORT_SINK = open(os.devnull, "w")
with contextlib.redirect_stdout(_QUIET_IMPORT_SINK), contextlib.redirect_stderr(
    _QUIET_IMPORT_SINK
):
    from robosuite.controllers import load_composite_controller_config
    import robosuite
    import robocasa.demos.demo_close_drawer_contact_curobo as close_demo
from robocasa.demos.open_drawer.collision import (
    check_arm_q_collision_for_surface as _check_arm_q_collision_for_surface_base,
    current_surface_for_stage as _current_surface_for_stage,
    diagnose_curobo_trajectory_contacts as _diagnose_curobo_trajectory_contacts_base,
)
from robocasa.demos.open_drawer.math import (
    _array_value_or_nan,
    _distance_summary,
    _rotation_angle_error,
)
from robocasa.demos.open_drawer.utils import (
    report_reason_counts as _report_reason_counts,
)


def _diagnose_curobo_trajectory_contacts(env, stages, q_traj, segments, args):
    return _diagnose_curobo_trajectory_contacts_base(
        env,
        stages,
        q_traj,
        segments,
        args,
        drawer_trajectory_from_curobo_segments=_drawer_trajectory_from_curobo_segments,
    )


def _configure_native_viewer_camera(viewer, lookat):
    try:
        cam = viewer.cam
        cam.lookat[:] = np.asarray(lookat, dtype=np.float64).reshape(3)
        cam.distance = 1.25
        cam.azimuth = 135.0
        cam.elevation = -25.0
    except Exception:
        pass


def create_open_drawer_env(args):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=args.robot,
    )
    has_offscreen_renderer = False
    try:
        env = robosuite.make(
            env_name="OpenDrawer",
            robots=args.robot,
            controller_configs=controller_config,
            has_renderer=bool(args.render),
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=None,
            renderer="mjviewer",
            use_camera_obs=False,
            ignore_done=True,
            reward_shaping=True,
            control_freq=args.control_freq,
            layout_ids=args.layout,
            style_ids=args.style,
            seed=args.seed,
        )
        env.reset()
        return env
    except ImportError:
        if not has_offscreen_renderer:
            raise
        pass
    env = robosuite.make(
        env_name="OpenDrawer",
        robots=args.robot,
        controller_configs=controller_config,
        has_renderer=bool(args.render),
        has_offscreen_renderer=False,
        render_camera=None,
        renderer="mjviewer",
        use_camera_obs=False,
        ignore_done=True,
        reward_shaping=True,
        control_freq=args.control_freq,
        layout_ids=args.layout,
        style_ids=args.style,
        seed=args.seed,
    )
    env.reset()
    return env


def _validate_joint_space_trajectory_collisions(env, stages, q_traj, segments, args):
    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    drawer_q, stage_indices = _drawer_trajectory_from_curobo_segments(
        stages,
        segments,
        arm_traj,
    )
    arm_joint_names = tuple(env.robots[0].robot_model.joints[:7])
    failures = []
    for step_index, (q_arm, drawer_value, stage_index) in enumerate(
        zip(arm_traj, drawer_q, stage_indices)
    ):
        stage = stages[int(np.clip(stage_index, 0, len(stages) - 1))]
        surface = _current_surface_for_stage(env, stage)
        ok, reason = _check_arm_q_collision_for_surface_base(
            env,
            surface,
            arm_joint_names,
            q_arm,
            float(drawer_value),
            set_arm_q=close_demo._set_env_arm_q,
            set_drawer_joint_value=close_demo._set_drawer_joint_value,
            allowed_ee_geom_name=str(stage.mink_solution.ee_contact_geom_name),
            penetration_tolerance=float(args.mink_collision_penetration_tolerance),
            collision_scope=str(args.mink_collision_scope),
        )
        if not ok:
            failures.append(
                {
                    "step": int(step_index),
                    "stage": str(stage.name),
                    "drawer_q": float(drawer_value),
                    "reason": str(reason),
                }
            )
            if len(failures) >= int(getattr(args, "joint_validation_max_failures", 8)):
                break
    return {
        "valid": not failures,
        "evaluated_steps": int(arm_traj.shape[0]),
        "failures": failures,
    }


def _validate_curobo_reaches_gripper_targets(
    env,
    robot_state,
    target_gripper_poses,
    q_traj,
    segments,
    args,
):
    """Check cuRobo joint output against robosuite grip-site targets in MuJoCo FK."""
    if q_traj is None or not np.asarray(q_traj).size:
        return []
    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    if not segments:
        return []

    model = env.sim.model
    data = env.sim.data
    frame_name = str(getattr(args, "mink_contact_frame", "gripper0_right_grip_site"))
    if frame_name not in model._site_name2id:
        raise RuntimeError(
            f"Cannot validate cuRobo targets: site '{frame_name}' not found."
        )
    site_id = model.site_name2id(frame_name)
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    diagnostics = []
    cumulative_steps = 0
    try:
        for (target_name, target_pos, target_rot), segment in zip(
            target_gripper_poses,
            segments,
        ):
            steps = max(int(segment.get("steps", 0)), 0)
            if steps <= 0:
                continue
            cumulative_steps += steps
            q_index = min(cumulative_steps - 1, arm_traj.shape[0] - 1)
            close_demo._set_env_arm_q(
                env,
                robot_state["robocasa_joint_names"],
                arm_traj[q_index],
            )
            actual_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
            actual_rot = (
                np.asarray(data.site_xmat[site_id], dtype=np.float64)
                .reshape(3, 3)
                .copy()
            )
            position_error = float(
                np.linalg.norm(
                    actual_pos - np.asarray(target_pos, dtype=np.float64).reshape(3)
                )
            )
            rotation_error = _rotation_angle_error(actual_rot, target_rot)
            diagnostics.append(
                {
                    "name": str(target_name),
                    "segment_name": str(segment.get("name", "")),
                    "q_index": int(q_index),
                    "position_error": position_error,
                    "rotation_error": rotation_error,
                    "target_pos": np.asarray(target_pos, dtype=np.float64).tolist(),
                    "actual_pos": actual_pos.tolist(),
                }
            )
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        env.sim.forward()

    max_pos_error = max((entry["position_error"] for entry in diagnostics), default=0.0)
    max_rot_error = max((entry["rotation_error"] for entry in diagnostics), default=0.0)
    args._curobo_target_diagnostics = diagnostics
    if max_pos_error > float(
        args.curobo_target_position_tolerance
    ) or max_rot_error > float(args.curobo_target_rotation_tolerance):
        worst = max(
            diagnostics,
            key=lambda entry: (
                float(entry["position_error"]),
                float(entry["rotation_error"]),
            ),
        )
        raise RuntimeError(
            "cuRobo reported success, but the MuJoCo grip site does not reach "
            "the requested MPC target. "
            f"worst_segment='{worst['name']}', "
            f"position_error={worst['position_error']:.6f} m, "
            f"rotation_error={worst['rotation_error']:.6f} rad. "
            "This indicates a frame or joint-order mismatch between cuRobo and robosuite."
        )
    return diagnostics


def _stage_index_by_name(stages):
    return {stage.name: index for index, stage in enumerate(stages)}


def _drawer_trajectory_from_curobo_segments(stages, segments, q_traj):
    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    drawer_q = float(stages[0].start_drawer_q)
    if not segments:
        return np.full(arm_traj.shape[0], drawer_q, dtype=np.float64), np.zeros(
            arm_traj.shape[0], dtype=np.int64
        )
    stage_by_name = _stage_index_by_name(stages)
    drawer_values = []
    stage_indices = []
    for segment in segments:
        steps = max(int(segment.get("steps", 0)), 0)
        if steps <= 0:
            continue
        name = str(segment.get("name", ""))
        stage_name, _, phase = name.partition(":")
        stage_index = stage_by_name.get(
            stage_name, min(len(stages) - 1, len(stage_indices))
        )
        segment_drawer = np.full(steps, drawer_q, dtype=np.float64)
        drawer_values.extend(float(value) for value in segment_drawer)
        stage_indices.extend([stage_index] * steps)
    if len(drawer_values) < arm_traj.shape[0]:
        drawer_values.extend([drawer_q] * (arm_traj.shape[0] - len(drawer_values)))
        stage_indices.extend(
            [len(stages) - 1] * (arm_traj.shape[0] - len(stage_indices))
        )
    drawer_values = np.asarray(drawer_values[: arm_traj.shape[0]], dtype=np.float64)
    stage_indices = np.asarray(stage_indices[: arm_traj.shape[0]], dtype=np.int64)
    return drawer_values, stage_indices


def _play_open_trajectory_viewer(
    env, stages, q_traj, segments, robot_state, args, *, replan_fn=None
):
    if not bool(getattr(args, "execution_viewer", True)):
        return
    if q_traj is None or not np.asarray(q_traj).size:
        print("[execution_viewer] skipped: no arm trajectory to play", flush=True)
        return
    try:
        import mujoco
        import mujoco.viewer
    except Exception as exc:
        print(
            f"[execution_viewer] skipped: failed to import mujoco.viewer: {exc}",
            flush=True,
        )
        return

    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    try:
        from robocasa.demos import visualize_mujoco as viz_mj

        ghost_geoms = viz_mj._extract_hand_finger_ghost_geoms(
            env, str(args.mink_contact_frame)
        )
    except Exception as exc:
        ghost_geoms = ()
        print(f"[execution_viewer] ghost EE overlay disabled: {exc}", flush=True)
    ghost_poses = []
    if ghost_geoms:
        try:
            site_id = env.sim.model.site_name2id(str(args.mink_contact_frame))
            for stage_index, stage in enumerate(stages):
                q_waypoints = np.asarray(
                    stage.mink_solution.q_waypoints, dtype=np.float64
                ).reshape(-1, 7)
                for waypoint_index, q_arm in enumerate(q_waypoints):
                    close_demo._set_env_arm_q(env, arm_joint_names, q_arm)
                    close_demo._set_drawer_joint_value(env, float(stage.start_drawer_q))
                    env.sim.forward()
                    ghost_poses.append(
                        (
                            stage_index,
                            waypoint_index,
                            np.asarray(
                                env.sim.data.site_xpos[site_id], dtype=np.float64
                            ).copy(),
                            np.asarray(
                                env.sim.data.site_xmat[site_id], dtype=np.float64
                            )
                            .reshape(3, 3)
                            .copy(),
                        )
                    )
        finally:
            env.sim.data.qpos[:] = qpos_saved
            env.sim.data.qvel[:] = qvel_saved
            env.sim.forward()
    frame_dt = 1.0 / max(float(getattr(args, "execution_viewer_fps", 30.0)), 1.0)
    sim_dt = float(getattr(raw_model, "opt", raw_model).timestep)
    substeps = max(
        int(getattr(args, "execution_viewer_pd_substeps", 0) or 0),
        int(np.ceil(frame_dt / max(sim_dt, 1e-6))),
        1,
    )
    kp = float(getattr(args, "execution_viewer_pd_kp", 350.0))
    kd = float(getattr(args, "execution_viewer_pd_kd", 35.0))
    loop = bool(getattr(args, "execution_viewer_loop", False))
    qpos_addrs = np.asarray(
        [env.sim.model.get_joint_qpos_addr(name) for name in arm_joint_names],
        dtype=np.int64,
    )
    dof_addrs = np.asarray(
        [int(raw_model.joint(name).dofadr[0]) for name in arm_joint_names],
        dtype=np.int64,
    )

    # robosuite installs joint-position actuators that hold the arm at the
    # reset setpoint with up to ~17 Nm — larger than our PD output (kp*Δq).
    # Zero gain/bias on every actuator driving an arm joint so qfrc_applied
    # is the only force on the arm; restore on exit.
    arm_joint_ids = np.asarray(
        [int(raw_model.joint(name).id) for name in arm_joint_names],
        dtype=np.int64,
    )
    arm_actuator_ids = []
    for aid in range(int(raw_model.nu)):
        if int(raw_model.actuator_trntype[aid]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        if int(raw_model.actuator_trnid[aid, 0]) in arm_joint_ids:
            arm_actuator_ids.append(int(aid))
    arm_actuator_ids = np.asarray(arm_actuator_ids, dtype=np.int64)
    saved_gainprm = raw_model.actuator_gainprm[arm_actuator_ids].copy()
    saved_biasprm = raw_model.actuator_biasprm[arm_actuator_ids].copy()
    raw_model.actuator_gainprm[arm_actuator_ids] = 0.0
    raw_model.actuator_biasprm[arm_actuator_ids] = 0.0

    print(
        f"[execution_viewer] playing {arm_traj.shape[0]} frames with PD control; "
        f"planner={str(getattr(args, '_execution_planner_source', 'unknown'))}; "
        f"neutralized {arm_actuator_ids.size} arm actuators; "
        "close the MuJoCo window to finish",
        flush=True,
    )
    try:
        env.sim.data.qpos[qpos_addrs] = arm_traj[0]
        env.sim.data.qvel[dof_addrs] = 0.0
        close_demo._set_drawer_joint_value(env, float(stages[0].start_drawer_q))
        env.sim.forward()
        lookat = np.asarray(stages[0].selected_contact_world, dtype=np.float64)
        with mujoco.viewer.launch_passive(raw_model, raw_data) as viewer:
            _configure_native_viewer_camera(viewer, lookat)
            ghost_rgba = np.asarray(
                [
                    float(getattr(args, "execution_viewer_ghost_r", 0.05)),
                    float(getattr(args, "execution_viewer_ghost_g", 0.45)),
                    float(getattr(args, "execution_viewer_ghost_b", 1.0)),
                    float(getattr(args, "execution_viewer_ghost_alpha", 0.26)),
                ],
                dtype=np.float32,
            )
            while viewer.is_running():
                for q_des in arm_traj:
                    if not viewer.is_running():
                        break
                    for _ in range(substeps):
                        q = np.asarray(env.sim.data.qpos[qpos_addrs], dtype=np.float64)
                        qd = np.asarray(env.sim.data.qvel[dof_addrs], dtype=np.float64)
                        raw_data.qfrc_applied[:] = 0.0
                        # Gravity/Coriolis feedforward so PD only fights position error.
                        gravity_comp = np.asarray(
                            raw_data.qfrc_bias[dof_addrs], dtype=np.float64
                        )
                        raw_data.qfrc_applied[dof_addrs] = (
                            gravity_comp + kp * (q_des - q) - kd * qd
                        )
                        mujoco.mj_step(raw_model, raw_data)
                    if hasattr(viewer, "user_scn") and ghost_geoms:
                        viewer.user_scn.ngeom = 0
                        for (
                            _stage_index,
                            waypoint_index,
                            target_pos,
                            target_rot,
                        ) in ghost_poses:
                            rgba = ghost_rgba.copy()
                            if waypoint_index == 1:
                                rgba[:3] = np.asarray(
                                    [1.0, 0.62, 0.05], dtype=np.float32
                                )
                            elif waypoint_index >= 2:
                                rgba[:3] = np.asarray(
                                    [0.05, 0.75, 0.35], dtype=np.float32
                                )
                            for ghost in ghost_geoms:
                                viz_mj._add_ghost_geom(
                                    viewer.user_scn,
                                    ghost,
                                    target_pos,
                                    target_rot,
                                    rgba,
                                )
                    viewer.sync()
                # End of one trajectory. Receding-horizon: replan from the
                # CURRENT joint state and continue. Stops when replan_fn
                # returns None (e.g. success criterion met or budget exhausted).
                if replan_fn is not None:
                    try:
                        current_q = np.asarray(
                            env.sim.data.qpos[qpos_addrs], dtype=np.float64
                        )
                        new_traj = replan_fn(current_q)
                    except Exception as exc:
                        print(
                            f"[execution_viewer] replan failed: {exc}; stopping.",
                            flush=True,
                        )
                        new_traj = None
                    if new_traj is not None and np.asarray(new_traj).size:
                        arm_traj = np.asarray(new_traj, dtype=np.float64).reshape(-1, 7)
                        print(
                            f"[execution_viewer] replanned: next {arm_traj.shape[0]} frames "
                            f"max_step={float(np.max(np.abs(arm_traj[-1] - arm_traj[0]))):.4f}",
                            flush=True,
                        )
                        continue
                if not loop:
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(frame_dt)
                    break
    finally:
        raw_model.actuator_gainprm[arm_actuator_ids] = saved_gainprm
        raw_model.actuator_biasprm[arm_actuator_ids] = saved_biasprm
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()


def _save_open_outputs(path, env, stages, target_hand_poses, q_traj, segments, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene_point_cloud = getattr(env, "_scene_point_cloud", None)
    scene_summary = (
        scene_point_cloud.summary() if scene_point_cloud is not None else None
    )
    curobo_contact = _diagnose_curobo_trajectory_contacts(
        env,
        stages,
        q_traj,
        segments,
        args,
    )
    metadata = {
        "task": "OpenDrawer",
        "drawer_name": env.drawer.name,
        "layout_id": int(env.layout_id)
        if hasattr(env, "layout_id") and not isinstance(env.layout_id, dict)
        else str(getattr(env, "layout_id", "")),
        "style_id": int(env.style_id)
        if hasattr(env, "style_id") and not isinstance(env.style_id, dict)
        else str(getattr(env, "style_id", "")),
        "drawer_size": np.asarray(env.drawer.size, dtype=np.float64).tolist(),
        "drawer_state": {
            k: float(v) for k, v in env.drawer.get_door_state(env).items()
        },
        "planned_total_pull_distance": float(
            sum(stage.pull_distance for stage in stages)
        ),
        "execution_planner_source": str(
            getattr(args, "_execution_planner_source", "none")
        ),
        "segments": segments,
        "curobo_target_diagnostics": getattr(args, "_curobo_target_diagnostics", []),
        "joint_space_validation": getattr(args, "_joint_space_validation", None),
        "scene_processing": scene_summary,
        "curobo_contact_diagnostics": curobo_contact["summary"],
        "stages": [
            {
                "name": stage.name,
                "surface": stage.surface_name,
                "start_drawer_q": float(stage.start_drawer_q),
                "pull_distance": float(stage.pull_distance),
                "has_executable_arm_trajectory": bool(
                    np.asarray(stage.mink_solution.q_waypoints).size
                ),
                "selected_contact_index": int(stage.selected_contact_index),
                "selected_contact_face": str(
                    getattr(
                        stage.candidates[stage.selected_contact_index],
                        "contact_face",
                        "",
                    )
                ),
                "selected_contact_world": stage.selected_contact_world.tolist(),
                "selected_contact_local": stage.selected_contact_local.tolist(),
                "selected_contact_cost": float(stage.selected_contact_cost),
                "gripper_mode": str(getattr(stage, "gripper_mode", "")),
                "dream_device": str(getattr(stage, "dream_device", "")),
                "contact_geometry_cache": str(
                    getattr(stage, "contact_geometry_cache", "")
                ),
                "feasible_graph_edge_count": int(
                    np.asarray(
                        getattr(
                            stage,
                            "feasible_graph_edges",
                            np.zeros((0, 2), dtype=np.int64),
                        )
                    ).shape[0]
                ),
                "representative_point_count": int(
                    np.asarray(
                        getattr(
                            stage,
                            "object_representative_points_world",
                            np.zeros((0, 3), dtype=np.float64),
                        )
                    ).shape[0]
                ),
                "curobo_collision_sphere_count": int(
                    np.asarray(
                        getattr(
                            stage,
                            "curobo_collision_sphere_radii",
                            np.zeros(0, dtype=np.float64),
                        )
                    ).size
                ),
                "selected_curobo_sphere_index": int(
                    getattr(stage, "selected_curobo_sphere_index", -1)
                ),
                "selected_curobo_sphere_link": (
                    str(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "selected_sphere_link", ""
                        )
                    )
                ),
                "minimum_curobo_sphere_sdf": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "minimum_sphere_sdf", float("nan")
                    )
                ),
                "selected_contact_set_candidate_indices": np.asarray(
                    getattr(
                        stage,
                        "selected_contact_set_candidate_indices",
                        np.zeros(0, dtype=np.int64),
                    ),
                    dtype=np.int64,
                ).tolist(),
                "candidate_count": len(stage.candidates),
                "feasible_candidate_count": int(
                    sum(candidate.feasible for candidate in stage.candidates)
                ),
                "visible_candidate_count": int(
                    sum(
                        bool(getattr(candidate, "visible", True))
                        for candidate in stage.candidates
                    )
                ),
                "selected_contact_visible": bool(
                    getattr(
                        stage.candidates[stage.selected_contact_index], "visible", True
                    )
                ),
                "selected_scene_link": str(
                    getattr(
                        stage.candidates[stage.selected_contact_index],
                        "scene_link_name",
                        "",
                    )
                ),
                "selected_scene_point_index": int(
                    getattr(
                        stage.candidates[stage.selected_contact_index],
                        "scene_point_index",
                        -1,
                    )
                ),
                "selected_scene_point_distance": float(
                    getattr(
                        stage.candidates[stage.selected_contact_index],
                        "scene_point_distance",
                        float("nan"),
                    )
                ),
                "selected_surface_projection_distance": float(
                    getattr(
                        stage.candidates[stage.selected_contact_index],
                        "surface_projection_distance",
                        float("nan"),
                    )
                ),
                "projected_seed_contact_world": getattr(
                    stage,
                    "projected_seed_contact_world",
                    np.full(3, np.nan, dtype=np.float64),
                ).tolist(),
                "projected_seed_contact_offset_error": float(
                    getattr(stage, "projected_seed_contact_offset_error", float("nan"))
                ),
                "projected_seed_contact_feasible_distance": float(
                    getattr(
                        stage, "projected_seed_contact_feasible_distance", float("nan")
                    )
                ),
                "projected_seed_contact_feasible_index": int(
                    getattr(stage, "projected_seed_contact_feasible_index", -1)
                ),
                "initial_pose_source": "demonstration_base_aligned_mink_precontact",
                "base_alignment": {
                    "applied": bool(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_applied", False
                        )
                    ),
                    "reason": str(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_reason", ""
                        )
                    ),
                    "yaw_delta": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_yaw_delta", float("nan")
                        )
                    ),
                    "translation_delta": list(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_translation_delta", []
                        )
                    ),
                    "initial_position_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_initial_position_error", float("nan")
                        )
                    ),
                    "final_position_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_final_position_error", float("nan")
                        )
                    ),
                    "initial_rotation_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_initial_rotation_error", float("nan")
                        )
                    ),
                    "final_rotation_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "base_alignment_final_rotation_error", float("nan")
                        )
                    ),
                },
                "mink_precontact": {
                    "candidate_index": int(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "precontact_mink_candidate_index", -1
                        )
                    ),
                    "position_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "precontact_mink_position_error", float("nan")
                        )
                    ),
                    "rotation_error": float(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "precontact_mink_rotation_error", float("nan")
                        )
                    ),
                    "collision_free": bool(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "precontact_mink_collision_free", False
                        )
                    ),
                    "collision_reason": str(
                        getattr(stage, "dream_diagnostics", {}).get(
                            "precontact_mink_collision_reason", ""
                        )
                    ),
                },
                "mink_contact_pose": {
                    "contact_frame": stage.mink_solution.contact_frame,
                    "ee_sample_index": int(stage.mink_solution.ee_sample_index),
                    "ee_sample_name": stage.mink_solution.ee_sample_name,
                    "ee_contact_geom_name": stage.mink_solution.ee_contact_geom_name,
                    "contact_offset_local": np.asarray(
                        stage.mink_solution.contact_offset_local,
                        dtype=np.float64,
                    ).tolist(),
                    "contact_position_error": float(
                        stage.mink_solution.contact_position_error
                    ),
                    "collision_free": bool(stage.mink_solution.collision_free),
                    "ee_sample_point_error": float(
                        getattr(
                            stage.mink_solution, "ee_sample_point_error", float("nan")
                        )
                    ),
                    "actual_contact": bool(
                        getattr(stage.mink_solution, "actual_contact", False)
                    ),
                    "actual_contact_near_candidate": bool(
                        getattr(
                            stage.mink_solution, "actual_contact_near_candidate", False
                        )
                    ),
                    "actual_contact_distance": float(
                        getattr(
                            stage.mink_solution, "actual_contact_distance", float("nan")
                        )
                    ),
                    "actual_contact_point_error": float(
                        getattr(
                            stage.mink_solution,
                            "actual_contact_point_error",
                            float("nan"),
                        )
                    ),
                    "actual_contact_geom_name": str(
                        getattr(stage.mink_solution, "actual_contact_geom_name", "")
                    ),
                    "selected_ee_geom_contact": bool(
                        getattr(stage.mink_solution, "selected_ee_geom_contact", False)
                    ),
                    "contact_standoff": float(
                        getattr(stage.mink_solution, "contact_standoff", float("nan"))
                    ),
                    "roll_angle": float(stage.mink_solution.roll_angle),
                },
                "demonstration_seed": {
                    "dataset": str(stage.demonstration_seed.dataset),
                    "episode_name": str(stage.demonstration_seed.episode_name),
                    "frame_index": int(stage.demonstration_seed.frame_index),
                    "arm_q": stage.demonstration_seed.arm_q.tolist(),
                    "ee_position_object": stage.demonstration_seed.ee_position_object.tolist(),
                    "ee_rotation_object": stage.demonstration_seed.ee_rotation_object.tolist(),
                    "contact_position_object": stage.demonstration_seed.contact_position_object.tolist(),
                    "contact_offset_ee": stage.demonstration_seed.contact_offset_ee.tolist(),
                    "source_position_roundtrip_error": float(
                        stage.demonstration_seed.source_position_roundtrip_error
                    ),
                    "source_rotation_roundtrip_error": float(
                        stage.demonstration_seed.source_rotation_roundtrip_error
                    ),
                    "projected_position_roundtrip_error": float(
                        stage.demonstration_seed.projected_position_roundtrip_error
                    ),
                    "projected_rotation_roundtrip_error": float(
                        stage.demonstration_seed.projected_rotation_roundtrip_error
                    ),
                },
                "dream_initial_pose_count": int(stage.dream_initial_poses.shape[0]),
                "dream_initial_candidate_indices": getattr(
                    stage,
                    "dream_initial_candidate_indices",
                    np.zeros(0, dtype=np.int64),
                ).tolist(),
                "dream_initial_feasible_contact_count": int(
                    np.asarray(
                        getattr(
                            stage, "dream_initial_contact_mask", np.zeros(0, dtype=bool)
                        ),
                        dtype=bool,
                    ).sum()
                ),
                "dream_initial_feasible_contact_fraction": float(
                    getattr(stage, "dream_initial_contact_fraction", float("nan"))
                ),
                "dream_initial_contact_distance_min": float(
                    np.nanmin(
                        np.asarray(
                            getattr(stage, "dream_initial_contact_distances", [np.nan]),
                            dtype=np.float64,
                        )
                    )
                ),
                "dream_initial_contact_distance_median": float(
                    np.nanmedian(
                        np.asarray(
                            getattr(stage, "dream_initial_contact_distances", [np.nan]),
                            dtype=np.float64,
                        )
                    )
                ),
                "dream_contact_target_fraction": float(
                    getattr(stage, "dream_contact_target_fraction", float("nan"))
                ),
                "dream_contact_target_distance_first": float(
                    _array_value_or_nan(
                        getattr(stage, "dream_contact_target_distances", np.zeros(0)),
                        0,
                    )
                ),
                "dream_contact_target_distance_terminal": float(
                    _array_value_or_nan(
                        getattr(stage, "dream_contact_target_distances", np.zeros(0)),
                        -1,
                    )
                ),
                "dream_contact_target_distance_max": float(
                    _distance_summary(
                        getattr(
                            stage,
                            "dream_contact_target_distances",
                            np.asarray([np.nan]),
                        )
                    )["max"]
                ),
                "dream_first_feasible_contact_distance": float(
                    getattr(
                        stage, "dream_first_feasible_contact_distance", float("nan")
                    )
                ),
                "dream_first_feasible_contact_index": int(
                    getattr(stage, "dream_first_feasible_contact_index", -1)
                ),
                "dream_target_snap_applied": bool(
                    getattr(stage, "dream_target_snap_applied", False)
                ),
                "target_contact_distances": np.asarray(
                    getattr(
                        stage, "target_contact_distances", np.zeros(0, dtype=np.float64)
                    ),
                    dtype=np.float64,
                ).tolist(),
                "target_contact_mask": np.asarray(
                    getattr(stage, "target_contact_mask", np.zeros(0, dtype=bool)),
                    dtype=bool,
                ).tolist(),
                "dream_success_count": (
                    int(
                        bool(
                            getattr(stage, "dream_diagnostics", {}).get(
                                "drawer_open_success", False
                            )
                        )
                    )
                ),
                "dream_drawer_open_success": bool(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "drawer_open_success", False
                    )
                ),
                "dream_terminal_drawer_q": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "terminal_drawer_q", float("nan")
                    )
                ),
                "dream_terminal_open_distance": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "terminal_open_distance", float("nan")
                    )
                ),
                "dream_candidate_open_success_count": int(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "candidate_open_success_count", 0
                    )
                ),
                "dream_candidate_open_success_fraction": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "candidate_open_success_fraction", float("nan")
                    )
                ),
                "dream_candidate_open_distance_median": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "candidate_open_distance_median", float("nan")
                    )
                ),
                "dream_candidate_open_distance_max": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "candidate_open_distance_max", float("nan")
                    )
                ),
                "dream_candidate_contact_target_fraction": float(
                    getattr(stage, "dream_diagnostics", {}).get(
                        "candidate_contact_target_fraction", float("nan")
                    )
                ),
                "dream_best_cost": (
                    float(stage.dream_result.best_cost)
                    if stage.dream_result is not None
                    else float("nan")
                ),
                "dream_best_index": (
                    int(stage.dream_result.best_index)
                    if stage.dream_result is not None
                    else -1
                ),
                "dream_initial_pose_error_min": (
                    float(
                        np.nanmin(
                            stage.dream_result.initial_pose_errors.detach()
                            .cpu()
                            .numpy()
                        )
                    )
                    if stage.dream_result is not None
                    and stage.dream_result.initial_pose_errors is not None
                    else float("nan")
                ),
                "dream_initial_position_error_min": (
                    float(
                        np.nanmin(
                            stage.dream_result.initial_position_errors.detach()
                            .cpu()
                            .numpy()
                        )
                    )
                    if stage.dream_result is not None
                    and stage.dream_result.initial_position_errors is not None
                    else float("nan")
                ),
                "dream_initial_rotation_error_min": (
                    float(
                        np.nanmin(
                            stage.dream_result.initial_rotation_errors.detach()
                            .cpu()
                            .numpy()
                        )
                    )
                    if stage.dream_result is not None
                    and stage.dream_result.initial_rotation_errors is not None
                    else float("nan")
                ),
                "dream_failure_reason": str(
                    getattr(stage, "dream_failure_detail", {}).get("reason", "unknown")
                ),
                "dream_failure_detail": getattr(stage, "dream_failure_detail", {}),
                "mink_attempt_reason_counts": _report_reason_counts(stage.mink_reports),
            }
            for stage in stages
        ],
    }

    np.savez(
        path,
        q_traj=np.asarray(q_traj, dtype=np.float64)
        if q_traj is not None
        else np.zeros((0, 7)),
        target_hand_pos_base=np.asarray(
            [pose[1] for pose in target_hand_poses], dtype=np.float64
        ),
        target_hand_quat_wxyz_base=np.asarray(
            [close_demo._quat_wxyz_from_matrix(pose[2]) for pose in target_hand_poses],
            dtype=np.float64,
        ),
        stage_names=np.asarray([stage.name for stage in stages]),
        stage_surfaces=np.asarray([stage.surface_name for stage in stages]),
        stage_start_drawer_q=np.asarray(
            [stage.start_drawer_q for stage in stages], dtype=np.float64
        ),
        stage_pull_distance=np.asarray(
            [stage.pull_distance for stage in stages], dtype=np.float64
        ),
        selected_contact_world=np.asarray(
            [stage.selected_contact_world for stage in stages], dtype=np.float64
        ),
        selected_contact_local=np.asarray(
            [stage.selected_contact_local for stage in stages], dtype=np.float64
        ),
        selected_contact_cost=np.asarray(
            [stage.selected_contact_cost for stage in stages], dtype=np.float64
        ),
        selected_contact_offset_ee=np.asarray(
            [
                np.asarray(
                    stage.mink_solution.contact_offset_local,
                    dtype=np.float64,
                )
                for stage in stages
            ],
            dtype=np.float64,
        ),
        selected_curobo_sphere_index=np.asarray(
            [getattr(stage, "selected_curobo_sphere_index", -1) for stage in stages],
            dtype=np.int64,
        ),
        curobo_collision_sphere_centers_ee=np.asarray(
            [
                getattr(
                    stage,
                    "curobo_collision_sphere_centers_ee",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        curobo_collision_sphere_radii=np.asarray(
            [
                getattr(
                    stage,
                    "curobo_collision_sphere_radii",
                    np.zeros(0, dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        curobo_collision_sphere_links=np.asarray(
            [
                np.asarray(
                    getattr(
                        stage,
                        "curobo_collision_sphere_links",
                        tuple(),
                    ),
                    dtype=str,
                )
                for stage in stages
            ],
            dtype=object,
        ),
        curobo_sphere_sdf=np.asarray(
            [
                getattr(
                    stage,
                    "curobo_sphere_sdf",
                    np.zeros(0, dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        object_representative_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "object_representative_points_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        object_representative_normals_world=np.asarray(
            [
                getattr(
                    stage,
                    "object_representative_normals_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        mink_q_waypoints=np.asarray(
            [
                stage.mink_solution.q_waypoints
                if np.asarray(stage.mink_solution.q_waypoints).size
                else np.zeros((0, 7), dtype=np.float64)
                for stage in stages
            ],
            dtype=object,
        ),
        demonstration_arm_q=np.asarray(
            [stage.demonstration_seed.arm_q for stage in stages],
            dtype=np.float64,
        ),
        demonstration_ee_position_object=np.asarray(
            [stage.demonstration_seed.ee_position_object for stage in stages],
            dtype=np.float64,
        ),
        demonstration_ee_rotation_object=np.asarray(
            [stage.demonstration_seed.ee_rotation_object for stage in stages],
            dtype=np.float64,
        ),
        demonstration_contact_position_object=np.asarray(
            [stage.demonstration_seed.contact_position_object for stage in stages],
            dtype=np.float64,
        ),
        projected_seed_contact_world=np.asarray(
            [
                getattr(
                    stage,
                    "projected_seed_contact_world",
                    np.full(3, np.nan, dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=np.float64,
        ),
        projected_seed_contact_offset_error=np.asarray(
            [
                getattr(stage, "projected_seed_contact_offset_error", np.nan)
                for stage in stages
            ],
            dtype=np.float64,
        ),
        projected_seed_contact_feasible_distance=np.asarray(
            [
                getattr(stage, "projected_seed_contact_feasible_distance", np.nan)
                for stage in stages
            ],
            dtype=np.float64,
        ),
        dream_initial_ee_poses=np.asarray(
            [stage.dream_initial_poses for stage in stages],
            dtype=object,
        ),
        dream_initial_candidate_indices=np.asarray(
            [
                getattr(
                    stage,
                    "dream_initial_candidate_indices",
                    np.zeros(0, dtype=np.int64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_initial_contact_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "dream_initial_contact_points",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_initial_contact_distances=np.asarray(
            [
                getattr(
                    stage,
                    "dream_initial_contact_distances",
                    np.zeros(0, dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_initial_contact_mask=np.asarray(
            [
                getattr(stage, "dream_initial_contact_mask", np.zeros(0, dtype=bool))
                for stage in stages
            ],
            dtype=object,
        ),
        dream_ee_pose_sequences=np.asarray(
            [
                getattr(stage, "dream_ee_sequence", np.zeros((0, 7), dtype=np.float64))
                for stage in stages
            ],
            dtype=object,
        ),
        dream_contact_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "dream_contact_points_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_contact_target_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "dream_contact_target_points_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_contact_target_distances=np.asarray(
            [
                getattr(
                    stage,
                    "dream_contact_target_distances",
                    np.zeros(0, dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_contact_target_mask=np.asarray(
            [
                getattr(stage, "dream_contact_target_mask", np.zeros(0, dtype=bool))
                for stage in stages
            ],
            dtype=object,
        ),
        target_contact_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "target_contact_points_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        target_contact_target_points_world=np.asarray(
            [
                getattr(
                    stage,
                    "target_contact_target_points_world",
                    np.zeros((0, 3), dtype=np.float64),
                )
                for stage in stages
            ],
            dtype=object,
        ),
        target_contact_distances=np.asarray(
            [
                getattr(
                    stage, "target_contact_distances", np.zeros(0, dtype=np.float64)
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_initial_position_errors=np.asarray(
            [
                (
                    stage.dream_result.initial_position_errors.detach().cpu().numpy()
                    if stage.dream_result is not None
                    and stage.dream_result.initial_position_errors is not None
                    else np.zeros(0, dtype=np.float64)
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_initial_rotation_errors=np.asarray(
            [
                (
                    stage.dream_result.initial_rotation_errors.detach().cpu().numpy()
                    if stage.dream_result is not None
                    and stage.dream_result.initial_rotation_errors is not None
                    else np.zeros(0, dtype=np.float64)
                )
                for stage in stages
            ],
            dtype=object,
        ),
        dream_failure_reasons=np.asarray(
            [
                str(getattr(stage, "dream_failure_detail", {}).get("reason", "unknown"))
                for stage in stages
            ]
        ),
        dream_failure_details=np.asarray(
            [getattr(stage, "dream_failure_detail", {}) for stage in stages],
            dtype=object,
        ),
        curobo_contact_stage_indices=np.asarray(
            curobo_contact["stage_indices"],
            dtype=np.int64,
        ),
        curobo_contact_drawer_q=np.asarray(
            curobo_contact["drawer_q"],
            dtype=np.float64,
        ),
        curobo_ee_contact_point_error=np.asarray(
            curobo_contact["ee_contact_point_error"],
            dtype=np.float64,
        ),
        curobo_actual_contact=np.asarray(
            curobo_contact["actual_contact"],
            dtype=bool,
        ),
        curobo_actual_contact_near_selected=np.asarray(
            curobo_contact["actual_contact_near_selected"],
            dtype=bool,
        ),
        curobo_actual_contact_distance=np.asarray(
            curobo_contact["actual_contact_distance"],
            dtype=np.float64,
        ),
        curobo_actual_contact_point_error=np.asarray(
            curobo_contact["actual_contact_point_error"],
            dtype=np.float64,
        ),
        scene_points_world=(
            scene_point_cloud.world_points(env.sim.data)
            if scene_point_cloud is not None
            else np.zeros((0, 3), dtype=np.float32)
        ),
        scene_visibility_mask=(
            scene_point_cloud.mask
            if scene_point_cloud is not None
            else np.zeros(0, dtype=np.uint8)
        ),
        scene_link_names=(
            np.asarray([link.name for link in scene_point_cloud.links])
            if scene_point_cloud is not None
            else np.asarray([], dtype=str)
        ),
        scene_link_point_counts=(
            np.asarray(
                [link.points_local.shape[0] for link in scene_point_cloud.links],
                dtype=np.int64,
            )
            if scene_point_cloud is not None
            else np.zeros(0, dtype=np.int64)
        ),
        metadata_json=json.dumps(metadata, indent=2),
    )
    return metadata
