import numpy as np

from robocasa.demos.open_drawer.math import _distance_summary
from robocasa.demos.open_drawer.scene import (
    make_handle_inner_surface as _make_handle_inner_surface,
    make_panel_inner_surface as _make_panel_inner_surface,
)


def robot_contact_geom_sets_for_surface(env, surface):
    model = env.sim.model
    robot_geoms = {
        geom_id
        for name, geom_id in model._geom_name2id.items()
        if name.startswith("robot0_") and "collision" in name
    }
    ee_geoms = {
        geom_id
        for name, geom_id in model._geom_name2id.items()
        if name.startswith("gripper0_right_") and "collision" in name
    }
    robot_geoms.update(ee_geoms)
    drawer_name = surface.geom_name.split("_door")[0]
    allowed_target_geoms = {
        geom_id
        for name, geom_id in model._geom_name2id.items()
        if name.startswith(f"{drawer_name}_")
    }
    return robot_geoms, ee_geoms, allowed_target_geoms


def check_arm_q_collision_for_surface(
    env,
    surface,
    arm_joint_names,
    q_arm,
    drawer_q,
    *,
    set_arm_q,
    set_drawer_joint_value,
    allowed_ee_geom_name=None,
    penetration_tolerance=0.0,
    collision_scope="ee",
):
    model = env.sim.model
    data = env.sim.data
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    geom_id_to_name = {geom_id: name for name, geom_id in model._geom_name2id.items()}
    try:
        set_arm_q(env, arm_joint_names, q_arm)
        set_drawer_joint_value(env, drawer_q)
        env.sim.forward()
        (
            robot_geoms,
            ee_geoms,
            allowed_target_geoms,
        ) = robot_contact_geom_sets_for_surface(
            env,
            surface,
        )
        checked_robot_geoms = ee_geoms if collision_scope == "ee" else robot_geoms
        allowed_ee_geoms = set(ee_geoms)
        if (
            allowed_ee_geom_name is not None
            and allowed_ee_geom_name in model._geom_name2id
        ):
            allowed_ee_geoms = {model.geom_name2id(allowed_ee_geom_name)}
        penetration_limit = -max(float(penetration_tolerance), 0.0)
        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in checked_robot_geoms and geom2 not in checked_robot_geoms:
                continue
            if geom1 in robot_geoms and geom2 in robot_geoms:
                continue
            robot_geom = geom1 if geom1 in checked_robot_geoms else geom2
            other_geom = geom2 if robot_geom == geom1 else geom1
            contact_dist = float(contact.dist)
            if other_geom not in allowed_target_geoms:
                if contact_dist < penetration_limit:
                    robot_name = geom_id_to_name.get(robot_geom, str(robot_geom))
                    other_name = geom_id_to_name.get(other_geom, str(other_geom))
                    scope = "ee" if robot_geom in ee_geoms else "arm"
                    return (
                        False,
                        f"{scope}_env_penetration:{robot_name}--{other_name}:dist={contact_dist:.6f}",
                    )
                continue
            if robot_geom in allowed_ee_geoms:
                if contact_dist >= penetration_limit:
                    continue
                robot_name = geom_id_to_name.get(robot_geom, str(robot_geom))
                other_name = geom_id_to_name.get(other_geom, str(other_geom))
                return (
                    False,
                    f"target_penetration:{robot_name}--{other_name}:dist={contact_dist:.6f}",
                )
            if contact_dist >= penetration_limit:
                continue
            robot_name = geom_id_to_name.get(robot_geom, str(robot_geom))
            other_name = geom_id_to_name.get(other_geom, str(other_geom))
            return (
                False,
                f"collision:{robot_name}--{other_name}:dist={contact_dist:.6f}",
            )
        return True, "collision_free"
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        env.sim.forward()


def current_surface_for_stage(env, stage):
    import robocasa.demos.demo_close_drawer_contact_curobo as close_demo

    panel = close_demo.get_panel_frame(env)
    if stage.surface_name == "handle":
        return _make_handle_inner_surface(env, panel)
    return _make_panel_inner_surface(env, panel)


def diagnose_current_stage_contact(env, stage, args):
    model = env.sim.model
    data = env.sim.data
    surface = current_surface_for_stage(env, stage)
    expected_contact_world = surface.center_world + surface.rotation_world @ np.asarray(
        stage.selected_contact_local, dtype=np.float64
    )
    frame_name = str(args.mink_contact_frame)
    ee_contact_point = np.full(3, np.nan, dtype=np.float64)
    ee_contact_point_error = float("inf")
    if frame_name in model._site_name2id:
        site_id = model.site_name2id(frame_name)
        site_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        site_rot = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        ee_contact_point = site_pos + site_rot @ np.asarray(
            stage.mink_solution.contact_offset_local,
            dtype=np.float64,
        )
        ee_contact_point_error = float(
            np.linalg.norm(ee_contact_point - expected_contact_world)
        )

    _, ee_geom_ids, target_geom_ids = robot_contact_geom_sets_for_surface(env, surface)
    best_contact_distance = float("inf")
    best_contact_point_error = float("inf")
    best_contact_world = np.full(3, np.nan, dtype=np.float64)
    geom_id_to_name = {
        int(geom_id): name for name, geom_id in model._geom_name2id.items()
    }
    best_contact_geom_name = ""
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        pair_matches = (geom1 in ee_geom_ids and geom2 in target_geom_ids) or (
            geom2 in ee_geom_ids and geom1 in target_geom_ids
        )
        if not pair_matches:
            continue
        contact_world = np.asarray(contact.pos, dtype=np.float64).copy()
        point_error = float(np.linalg.norm(contact_world - expected_contact_world))
        contact_distance = float(contact.dist)
        rank = (point_error, abs(contact_distance))
        best_rank = (best_contact_point_error, abs(best_contact_distance))
        if rank < best_rank:
            contact_ee_geom_id = geom1 if geom1 in ee_geom_ids else geom2
            best_contact_distance = contact_distance
            best_contact_point_error = point_error
            best_contact_world = contact_world
            best_contact_geom_name = geom_id_to_name.get(
                int(contact_ee_geom_id),
                str(contact_ee_geom_id),
            )
    actual_contact = bool(
        np.isfinite(best_contact_distance)
        and best_contact_distance <= float(args.mink_actual_contact_max_distance)
    )
    actual_contact_near_selected = bool(
        actual_contact
        and best_contact_point_error <= float(args.mink_actual_contact_point_tolerance)
    )
    return {
        "expected_contact_world": expected_contact_world,
        "ee_contact_point_world": ee_contact_point,
        "ee_contact_point_error": ee_contact_point_error,
        "actual_contact": actual_contact,
        "actual_contact_near_selected": actual_contact_near_selected,
        "actual_contact_distance": best_contact_distance,
        "actual_contact_point_error": best_contact_point_error,
        "actual_contact_world": best_contact_world,
        "actual_contact_geom_name": best_contact_geom_name,
    }


def diagnose_curobo_trajectory_contacts(
    env,
    stages,
    q_traj,
    segments,
    args,
    *,
    drawer_trajectory_from_curobo_segments,
):
    if q_traj is None or not np.asarray(q_traj).size:
        return {
            "summary": {
                "evaluated_steps": 0,
                "actual_contact_step_count": 0,
                "actual_contact_near_selected_step_count": 0,
                "actual_contact_fraction": 0.0,
                "actual_contact_near_selected_fraction": 0.0,
                "ee_contact_point_error_min": float("inf"),
                "ee_contact_point_error_median": float("inf"),
                "ee_contact_point_error_max": float("inf"),
            },
            "stage_indices": np.zeros(0, dtype=np.int64),
            "drawer_q": np.zeros(0, dtype=np.float64),
            "ee_contact_point_error": np.zeros(0, dtype=np.float64),
            "actual_contact": np.zeros(0, dtype=bool),
            "actual_contact_near_selected": np.zeros(0, dtype=bool),
            "actual_contact_distance": np.zeros(0, dtype=np.float64),
            "actual_contact_point_error": np.zeros(0, dtype=np.float64),
        }

    import robocasa.demos.demo_close_drawer_contact_curobo as close_demo

    arm_traj = np.asarray(q_traj, dtype=np.float64).reshape(-1, 7)
    drawer_q, stage_indices = drawer_trajectory_from_curobo_segments(
        stages,
        segments,
        arm_traj,
    )
    data = env.sim.data
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    arm_joint_names = tuple(env.robots[0].robot_model.joints[:7])
    ee_errors = []
    actual_contacts = []
    actual_near = []
    actual_distances = []
    actual_point_errors = []
    try:
        for q_arm, drawer_value, stage_index in zip(arm_traj, drawer_q, stage_indices):
            close_demo._set_env_arm_q(env, arm_joint_names, q_arm)
            close_demo._set_drawer_joint_value(env, float(drawer_value))
            env.sim.forward()
            stage = stages[int(np.clip(stage_index, 0, len(stages) - 1))]
            diag = diagnose_current_stage_contact(env, stage, args)
            ee_errors.append(float(diag["ee_contact_point_error"]))
            actual_contacts.append(bool(diag["actual_contact"]))
            actual_near.append(bool(diag["actual_contact_near_selected"]))
            actual_distances.append(float(diag["actual_contact_distance"]))
            actual_point_errors.append(float(diag["actual_contact_point_error"]))
    finally:
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        env.sim.forward()

    ee_errors = np.asarray(ee_errors, dtype=np.float64)
    actual_contacts = np.asarray(actual_contacts, dtype=bool)
    actual_near = np.asarray(actual_near, dtype=bool)
    actual_distances = np.asarray(actual_distances, dtype=np.float64)
    actual_point_errors = np.asarray(actual_point_errors, dtype=np.float64)
    error_summary = _distance_summary(ee_errors)
    evaluated_steps = int(arm_traj.shape[0])
    summary = {
        "evaluated_steps": evaluated_steps,
        "actual_contact_step_count": int(actual_contacts.sum()),
        "actual_contact_near_selected_step_count": int(actual_near.sum()),
        "actual_contact_fraction": float(actual_contacts.mean())
        if evaluated_steps
        else 0.0,
        "actual_contact_near_selected_fraction": float(actual_near.mean())
        if evaluated_steps
        else 0.0,
        "ee_contact_point_error_min": float(error_summary["min"]),
        "ee_contact_point_error_median": float(error_summary["median"]),
        "ee_contact_point_error_max": float(error_summary["max"]),
        "actual_contact_distance_min": float(
            _distance_summary(actual_distances)["min"]
        ),
        "actual_contact_point_error_min": float(
            _distance_summary(actual_point_errors)["min"]
        ),
    }
    return {
        "summary": summary,
        "stage_indices": stage_indices,
        "drawer_q": drawer_q,
        "ee_contact_point_error": ee_errors,
        "actual_contact": actual_contacts,
        "actual_contact_near_selected": actual_near,
        "actual_contact_distance": actual_distances,
        "actual_contact_point_error": actual_point_errors,
    }


_current_surface_for_stage = current_surface_for_stage
_diagnose_current_stage_contact = diagnose_current_stage_contact
_diagnose_curobo_trajectory_contacts = diagnose_curobo_trajectory_contacts


__all__ = [
    "check_arm_q_collision_for_surface",
    "current_surface_for_stage",
    "diagnose_current_stage_contact",
    "diagnose_curobo_trajectory_contacts",
    "robot_contact_geom_sets_for_surface",
    "_current_surface_for_stage",
    "_diagnose_current_stage_contact",
    "_diagnose_curobo_trajectory_contacts",
]
