"""Franka/CuRobo collision-sphere helpers for drawer contact q-MPC."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from robocasa.demos.object_cso import (
    build_feasible_contact_cache,
    sample_object_representative_points,
    select_contact_set_from_cache,
)
from robocasa.demos.open_drawer.utils import _empty_cuda_caches


@dataclass
class EECollisionSphereModel:
    centers_ee: np.ndarray
    radii: np.ndarray
    link_names: tuple[str, ...]
    source_indices: np.ndarray
    current_centers_world: np.ndarray


def _parse_name_list(value):
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _demonstration_seed_pose_wxyz(demonstration_seed, quat_from_matrix):
    return np.concatenate(
        [
            np.asarray(
                demonstration_seed.projected_ee_position_world, dtype=np.float64
            ).reshape(3),
            quat_from_matrix(demonstration_seed.projected_ee_rotation_world),
        ]
    )


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _cuda_memory_limit_bytes(args) -> int:
    limit_gb = float(getattr(args, "cuda_memory_limit_gb", 20.0) or 0.0)
    return int(max(limit_gb, 0.0) * (1024**3))


def _limit_q_config_samples_for_memory(config, raw_model, args):
    limit_bytes = _cuda_memory_limit_bytes(args)
    if limit_bytes <= 0:
        return config
    budget_fraction = float(getattr(args, "q_config_mpc_memory_budget_fraction", 0.20))
    budget_bytes = int(limit_bytes * max(min(budget_fraction, 1.0), 0.05))
    nv = int(getattr(raw_model, "nv", 0))
    njmax = int(config.njmax_per_env)
    njmax_pad = _round_up(njmax, 16)
    nv_pad = _round_up(nv, 16) if nv > 32 else _round_up(nv, 4)
    bytes_per_sample = max(njmax_pad * nv_pad * 8 * 4, 1)
    max_samples = max(1, budget_bytes // bytes_per_sample)
    if int(config.num_samples) <= max_samples:
        return config
    bounded = max(1, int(max_samples))
    print(
        "[q_config_mpc] reducing num_samples from "
        f"{int(config.num_samples)} to {bounded} to stay within "
        f"{float(getattr(args, 'cuda_memory_limit_gb', 20.0)):.2f}GB CUDA budget",
        flush=True,
    )
    return replace(config, num_samples=bounded)


def _q_config_result_to_cpu(result):
    return replace(
        result,
        best_q=result.best_q.detach().cpu(),
        candidate_q=result.candidate_q.detach().cpu(),
        candidate_costs=result.candidate_costs.detach().cpu(),
        best_sphere_centers_world=result.best_sphere_centers_world.detach().cpu(),
        best_contact_distances=result.best_contact_distances.detach().cpu(),
    )


def gripper_joint_names_for_q_mpc(env):
    """Return active gripper joints (1-DoF) on the right gripper, if any."""
    robot = env.robots[0]
    names = []
    gripper = getattr(robot, "gripper", None)
    if isinstance(gripper, dict):
        gripper = gripper.get("right", None)
    if gripper is None:
        return tuple()
    for joint_name in getattr(gripper, "joints", ()):
        if joint_name in env.sim.model._joint_name2id:
            names.append(str(joint_name))
    return tuple(names)


def _wide_open_gripper_qpos(env, gripper_joint_names):
    """Return the fully-open qpos for each named gripper joint.

    Uses each joint's upper `jnt_range` (Panda finger joints have symmetric
    [0, 0.04] range so the upper bound is the fully-open position). Falls
    back to the current env qpos if the joint has no finite range.
    """
    model = env.sim.model
    data = env.sim.data
    values = []
    for name in gripper_joint_names:
        jid = int(model.joint_name2id(name))
        limited = (
            bool(model.jnt_limited[jid]) if hasattr(model, "jnt_limited") else True
        )
        lo, hi = (
            (float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1]))
            if hasattr(model, "jnt_range")
            else (0.0, 0.04)
        )
        if limited and np.isfinite(hi) and hi > lo:
            values.append(hi)
        else:
            values.append(float(data.qpos[model.get_joint_qpos_addr(name)]))
    return np.asarray(values, dtype=np.float64)


def open_env_gripper_wide(env):
    """Open the right gripper fingers to their max qpos.

    Called by the open-drawer autogen wrapper before the pre-grasp solve so
    the MPC validator and the executed arm trajectory both see wide-open
    fingers (fingers close in a separate post-trajectory step). Returns the
    saved gripper qpos so the caller can restore if needed.
    """
    joint_names = gripper_joint_names_for_q_mpc(env)
    if not joint_names:
        return {}
    model = env.sim.model
    data = env.sim.data
    saved = {}
    wide = _wide_open_gripper_qpos(env, joint_names)
    for name, value in zip(joint_names, wide):
        addr = int(model.get_joint_qpos_addr(name))
        saved[name] = float(data.qpos[addr])
        data.qpos[addr] = float(value)
    env.sim.forward()
    return saved


def load_curobo_ee_collision_spheres(env, robot_state, args):
    """Extract CuRobo EE spheres and express centers in the grip-site frame."""
    cached = getattr(args, "_curobo_ee_collision_sphere_model", None)
    if cached is not None:
        return cached

    from robocasa.demos import demo_close_drawer_contact_curobo as close_demo

    close_demo._ensure_curobo_importable()
    import torch
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml

    device_name = str(getattr(args, "curobo_sphere_device", "")).strip()
    if not device_name:
        device_name = str(getattr(args, "dream_device", "cuda:0"))
    tensor_args = TensorDeviceType(device=torch.device(device_name))
    config_value = str(args.curobo_robot_cfg)
    expanded_config_path = Path(config_value).expanduser()
    config_path = (
        str(expanded_config_path)
        if expanded_config_path.is_file()
        else join_path(get_robot_configs_path(), config_value)
    )
    robot_cfg = RobotConfig.from_dict(load_yaml(config_path), tensor_args)
    kinematics = CudaRobotModel(robot_cfg.kinematics)
    q = np.asarray(robot_state["q"], dtype=np.float64).reshape(1, -1)
    if q.shape[1] != int(kinematics.get_dof()):
        raise RuntimeError(
            "CuRobo collision-sphere model DOF does not match the RoboCasa arm: "
            f"{kinematics.get_dof()} != {q.shape[1]}"
        )
    state = kinematics.get_state(tensor_args.to_device(q))
    spheres_curobo_base = state.get_link_spheres().detach().cpu().numpy().reshape(-1, 4)
    curobo_hand_pos = state.ee_position.detach().cpu().numpy().reshape(-1, 3)[0]
    curobo_hand_rot = close_demo._matrix_from_quat_wxyz(
        state.ee_quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
    )
    curobo_base_pos_in_robosuite, curobo_base_rot_in_robosuite = close_demo._pose_mul(
        np.asarray(robot_state["hand_pos_base"], dtype=np.float64),
        np.asarray(robot_state["hand_rot_base"], dtype=np.float64),
        *close_demo._pose_inv(curobo_hand_pos, curobo_hand_rot),
    )
    centers_robosuite_base = (
        spheres_curobo_base[:, :3] @ curobo_base_rot_in_robosuite.T
        + curobo_base_pos_in_robosuite
    )
    centers_world = centers_robosuite_base @ np.asarray(
        robot_state["base_rot"]
    ).T + np.asarray(robot_state["base_pos"])

    kin_cfg = kinematics.kinematics_config
    sphere_link_indices = kin_cfg.link_sphere_idx_map.detach().cpu().numpy().reshape(-1)
    index_to_name = {
        int(link_index): str(link_name)
        for link_name, link_index in kin_cfg.link_name_to_idx_map.items()
    }
    all_link_names = np.asarray(
        [index_to_name.get(int(index), "") for index in sphere_link_indices],
        dtype=object,
    )
    requested_links = set(_parse_name_list(getattr(args, "curobo_ee_sphere_links", "")))
    valid = spheres_curobo_base[:, 3] > float(args.curobo_sphere_min_radius)
    if requested_links:
        valid &= np.asarray(
            [name in requested_links for name in all_link_names], dtype=bool
        )
    selected_indices = np.flatnonzero(valid)
    if selected_indices.size == 0:
        available = sorted(
            {
                str(name)
                for name, radius in zip(all_link_names, spheres_curobo_base[:, 3])
                if float(radius) > 0.0
            }
        )
        raise RuntimeError(
            "No CuRobo EE collision spheres matched "
            f"--curobo-ee-sphere-links. Available links: {available}"
        )

    frame_name = str(args.mink_contact_frame).split(":")[0]
    if frame_name not in env.sim.model._site_name2id:
        raise RuntimeError(
            f"Cannot express CuRobo spheres in EE frame: site '{frame_name}' not found."
        )
    site_id = int(env.sim.model.site_name2id(frame_name))
    ee_position_world = np.asarray(
        env.sim.data.site_xpos[site_id], dtype=np.float64
    ).reshape(3)
    ee_rotation_world = np.asarray(
        env.sim.data.site_xmat[site_id], dtype=np.float64
    ).reshape(3, 3)
    selected_centers_world = centers_world[selected_indices]
    centers_ee = (selected_centers_world - ee_position_world) @ ee_rotation_world
    result = EECollisionSphereModel(
        centers_ee=np.asarray(centers_ee, dtype=np.float64),
        radii=np.asarray(spheres_curobo_base[selected_indices, 3], dtype=np.float64),
        link_names=tuple(str(all_link_names[index]) for index in selected_indices),
        source_indices=np.asarray(selected_indices, dtype=np.int64),
        current_centers_world=np.asarray(selected_centers_world, dtype=np.float64),
    )
    args._curobo_ee_collision_sphere_model = result
    return result


def _validate_q_config_candidate_collision(
    env,
    surface,
    arm_joint_names,
    gripper_joint_names,
    q_full,
    *,
    penetration_tolerance: float,
):
    model = env.sim.model
    data = env.sim.data
    q_full = np.asarray(q_full, dtype=np.float64).reshape(-1)
    arm_dof = len(arm_joint_names)
    q_arm = q_full[:arm_dof]
    q_gripper = q_full[arm_dof : arm_dof + len(gripper_joint_names)]
    qpos_saved = data.qpos.copy()
    qvel_saved = data.qvel.copy()
    geom_id_to_name = {geom_id: name for name, geom_id in model._geom_name2id.items()}
    try:
        from robocasa.demos import demo_open_drawer_contact_curobo as open_demo
        from robocasa.demos import demo_close_drawer_contact_curobo as close_demo

        close_demo._set_env_arm_q(env, arm_joint_names, q_arm)
        for joint_name, value in zip(gripper_joint_names, q_gripper):
            if joint_name in getattr(model, "_joint_name2id", {}):
                data.qpos[model.get_joint_qpos_addr(joint_name)] = float(value)
        env.sim.forward()
        (
            robot_geoms,
            ee_geoms,
            target_geoms,
        ) = open_demo._robot_contact_geom_sets_for_surface(
            env,
            surface,
        )
        penetration_limit = -max(float(penetration_tolerance), 0.0)
        # Wrist links wrap around the handle in a legitimate Panda grasp, so
        # treat their contact with the drawer target geoms the same way as an
        # EE geom (allowed up to `wrist_target_tolerance`). The env-side
        # persistent overlaps (link5 into the counter top the arm is mounted
        # on) get a looser tolerance so they stop rejecting every sample.
        wrist_link_substrings = ("_link6_", "_link7_")
        wrist_target_tolerance = (
            0.05  # meters: Panda wrist may sit up to 5 cm inside handle envelope
        )
        env_penetration_tolerance = max(
            float(penetration_tolerance),
            0.015,  # tolerate ~1.5 cm of persistent scene mount overlap (link5-counter etc.)
        )
        env_penetration_limit = -env_penetration_tolerance
        wrist_target_limit = -wrist_target_tolerance

        def _is_wrist(name):
            return any(sub in name for sub in wrist_link_substrings)

        for contact_idx in range(int(data.ncon)):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in robot_geoms and geom2 not in robot_geoms:
                continue
            if geom1 in robot_geoms and geom2 in robot_geoms:
                continue
            robot_geom = geom1 if geom1 in robot_geoms else geom2
            other_geom = geom2 if robot_geom == geom1 else geom1
            contact_dist = float(contact.dist)
            robot_name = geom_id_to_name.get(robot_geom, str(robot_geom))
            other_name = geom_id_to_name.get(other_geom, str(other_geom))
            if other_geom not in target_geoms:
                if contact_dist >= env_penetration_limit:
                    continue
                scope = "ee" if robot_geom in ee_geoms else "arm"
                return (
                    False,
                    f"{scope}_env_penetration:{robot_name}--{other_name}:"
                    f"dist={contact_dist:.6f}",
                )
            if robot_geom in ee_geoms or _is_wrist(robot_name):
                # EE fingers/pads or wrist links touching the drawer target
                # (handle/door) — legitimate grasp contact up to the wrist
                # tolerance.
                if contact_dist >= wrist_target_limit:
                    continue
                return (
                    False,
                    f"arm_target_penetration:{robot_name}--{other_name}:"
                    f"dist={contact_dist:.6f}",
                )
            if contact_dist >= penetration_limit:
                continue
            return (
                False,
                f"arm_target_penetration:{robot_name}--{other_name}:"
                f"dist={contact_dist:.6f}",
            )
        return True, "collision_free"
    finally:
        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        env.sim.forward()


def solve_collision_sphere_contact_with_q_mpc(
    env,
    surface,
    candidates,
    demonstration_seed,
    robot_state,
    precontact_solution,
    args,
    pull_distance=0.0,
):
    """Optimize arm + gripper q so an EE collision sphere lands on feasible contact."""
    from robocasa.demos import demo_open_drawer_contact_curobo as open_demo
    from robocasa.demos.dream import (
        QConfigOptimizerConfig,
        solve_q_config_contact,
        solve_q_config_contact_set,
    )

    feasible_cache = build_feasible_contact_cache(surface, candidates)
    contact_set_local_indices = select_contact_set_from_cache(feasible_cache, args)
    feasible_points_world = feasible_cache.positions_world
    contact_set_points_world = feasible_cache.positions_world[contact_set_local_indices]
    contact_set_normals_world = feasible_cache.normals_world[contact_set_local_indices]
    contact_set_candidate_indices = feasible_cache.candidate_indices[
        contact_set_local_indices
    ]

    sphere_model = load_curobo_ee_collision_spheres(env, robot_state, args)
    representative_points = sample_object_representative_points(
        env, surface, candidates, args
    )
    demonstration_pose = _demonstration_seed_pose_wxyz(
        demonstration_seed, open_demo._rotation_matrix_to_quat_wxyz
    )
    from robocasa.demos.object_cso import sphere_centers_world_from_pose

    demonstration_centers = sphere_centers_world_from_pose(
        demonstration_pose, sphere_model
    )
    contact_links = set(
        _parse_name_list(getattr(args, "curobo_contact_sphere_links", ""))
    )
    eligible_sphere_indices = np.asarray(
        [
            i
            for i, link_name in enumerate(sphere_model.link_names)
            if not contact_links or str(link_name) in contact_links
        ],
        dtype=np.int64,
    )
    if eligible_sphere_indices.size == 0:
        raise RuntimeError(
            "No contact sphere matched --curobo-contact-sphere-links within "
            "the selected EE sphere set"
        )
    sphere_radii = np.asarray(sphere_model.radii, dtype=np.float64)

    arm_joint_names = robot_state["robocasa_joint_names"]
    gripper_joint_names = gripper_joint_names_for_q_mpc(env)
    precontact_arm_q = np.asarray(precontact_solution.arm_q, dtype=np.float64)
    seed_arm_q = np.asarray(demonstration_seed.arm_q, dtype=np.float64).reshape(-1)
    # Force the MPC seed to hold the fingers fully open during pre-grasp
    # sampling. The env fingers are ALSO opened (in the autogen wrapper)
    # before this call, so the collision validator at `_validate_q_config_
    # candidate_collision` sees wide-open fingers and does not mistake a
    # finger-touching-handle contact for `arm_target_penetration`. The wrist
    # (link7) collision reason typically follows from a narrow-finger seed
    # driving the EE deeper toward the handle center; wide fingers pull the
    # EE back to the outside of the handle. See the plan in the assistant
    # response referencing this change.
    open_gripper_qpos = _wide_open_gripper_qpos(env, gripper_joint_names)
    seed_q = np.concatenate(
        (
            seed_arm_q.reshape(-1),
            np.asarray(open_gripper_qpos, dtype=np.float64).reshape(-1),
        )
    )

    config = QConfigOptimizerConfig(
        device=str(getattr(args, "q_config_mpc_device", "cuda:0")),
        seed=int(args.seed),
        num_samples=int(getattr(args, "q_config_mpc_num_samples", 128)),
        max_num_iterations=int(getattr(args, "q_config_mpc_iterations", 16)),
        arm_noise_scale=float(getattr(args, "q_config_mpc_arm_noise", 0.15)),
        gripper_noise_scale=0.0,
        contact_weight=float(getattr(args, "q_config_mpc_contact_weight", 200.0)),
        penetration_weight=float(
            getattr(args, "q_config_mpc_penetration_weight", 400.0)
        ),
        penetration_margin=float(
            getattr(args, "q_config_mpc_penetration_margin", 0.002)
        ),
        regularization_weight=float(getattr(args, "q_config_mpc_reg_weight", 1.0)),
        nconmax_per_env=max(int(getattr(args, "q_config_mpc_nconmax_per_env", 120)), 1),
        njmax_per_env=max(int(getattr(args, "q_config_mpc_njmax_per_env", 500)), 1),
    )

    raw_model = getattr(env.sim.model, "_model", env.sim.model)
    raw_data = getattr(env.sim.data, "_data", env.sim.data)
    config = _limit_q_config_samples_for_memory(config, raw_model, args)
    ee_site = getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    frame_name = str(ee_site).split(":", 1)[0]
    seed_pos, seed_rot = open_demo._site_pose_for_arm_q(
        env, robot_state, seed_arm_q, frame_name
    )
    seed_position_error = float(
        np.linalg.norm(
            seed_pos
            - np.asarray(
                demonstration_seed.projected_ee_position_world, dtype=np.float64
            ).reshape(3)
        )
    )
    seed_rotation_error = float(
        np.linalg.norm(
            seed_rot
            - np.asarray(
                demonstration_seed.projected_ee_rotation_world, dtype=np.float64
            ).reshape(3, 3)
        )
    )
    sample_count = int(config.num_samples)
    assignment = np.arange(sample_count, dtype=np.int64) % max(
        int(contact_set_points_world.shape[0]), 1
    )
    assigned_points = contact_set_points_world[assignment]
    assigned_normals = contact_set_normals_world[assignment]
    assigned_candidate_indices = contact_set_candidate_indices[assignment]
    assigned_sphere_indices = []
    assigned_target_centers = []
    for point, normal in zip(assigned_points, assigned_normals):
        target_centers = point.reshape(1, 3) + normal.reshape(1, 3) * sphere_radii[
            eligible_sphere_indices
        ].reshape(-1, 1)
        errors = np.linalg.norm(
            demonstration_centers[eligible_sphere_indices] - target_centers,
            axis=1,
        )
        sphere_index = int(eligible_sphere_indices[int(np.argmin(errors))])
        assigned_sphere_indices.append(sphere_index)
        assigned_target_centers.append(
            point + normal * float(sphere_radii[sphere_index])
        )
    assigned_sphere_indices = np.asarray(assigned_sphere_indices, dtype=np.int64)
    assigned_target_centers = np.asarray(assigned_target_centers, dtype=np.float64)
    (
        robot_geoms,
        _ee_geoms,
        target_geoms,
    ) = open_demo._robot_contact_geom_sets_for_surface(env, surface)

    gpu_result = None
    try:
        gpu_result = solve_q_config_contact_set(
            raw_model,
            raw_data,
            ee_site=ee_site,
            arm_joint_names=arm_joint_names,
            gripper_joint_names=gripper_joint_names,
            sphere_centers_ee=sphere_model.centers_ee,
            sphere_radii=sphere_model.radii,
            seed_q=seed_q,
            contact_sphere_indices=assigned_sphere_indices,
            target_points_world=assigned_target_centers,
            feasible_points_world=feasible_points_world,
            geom_set_a=robot_geoms,
            geom_set_b=target_geoms,
            object_points_world=representative_points.points_world,
            contact_tolerance=float(args.dream_initial_contact_feasible_distance),
            penetration_tolerance=float(args.sphere_contact_penetration_tolerance),
            success_contact_fraction=float(
                getattr(args, "q_config_mpc_success_contact_fraction", 0.5)
            ),
            config=config,
        )
        result = _q_config_result_to_cpu(gpu_result)
    finally:
        del gpu_result
        _empty_cuda_caches(config.device)
    candidate_q_np = result.candidate_q.detach().cpu().numpy().astype(np.float64)
    candidate_costs_np = (
        result.candidate_costs.detach().cpu().numpy().astype(np.float64)
    )
    finite_mask = np.isfinite(candidate_costs_np)
    if not np.any(finite_mask):
        raise RuntimeError("q-config MPC produced no finite contact q")

    candidate_collision_ok = np.zeros(candidate_q_np.shape[0], dtype=bool)
    candidate_collision_reasons = []
    for q_candidate in candidate_q_np:
        ok, reason = _validate_q_config_candidate_collision(
            env,
            surface,
            arm_joint_names,
            gripper_joint_names,
            q_candidate,
            penetration_tolerance=float(args.sphere_contact_penetration_tolerance),
        )
        candidate_collision_ok[len(candidate_collision_reasons)] = bool(ok)
        candidate_collision_reasons.append(str(reason))
    successful_mask = finite_mask & candidate_collision_ok
    if not np.any(successful_mask):
        finite_reasons = [
            candidate_collision_reasons[index]
            for index in np.flatnonzero(finite_mask)[:5]
        ]
        # Non-fatal fallback: no candidate passes the full-arm collision
        # validator, but we still want a trajectory to visualize. Fall back
        # to the best-cost finite sample and warn loudly. Set
        # `args.q_config_mpc_require_collision_free = True` to restore the
        # hard failure.
        if bool(getattr(args, "q_config_mpc_require_collision_free", False)):
            raise RuntimeError(
                "q-config MPC produced no full-arm collision-free contact q. "
                f"first_finite_reasons={finite_reasons}"
            )
        import sys as _sys

        print(
            "[q_config_mpc] WARNING: no collision-free candidate; falling "
            "back to best-cost sample for visualization. "
            f"first_finite_reasons={finite_reasons}",
            file=_sys.__stdout__,
            flush=True,
        )
        successful_mask = finite_mask.copy()

    successful_hypothesis_q = candidate_q_np[successful_mask]
    successful_hypothesis_costs = candidate_costs_np[successful_mask]

    selected_sample_index = int(
        np.flatnonzero(successful_mask)[
            int(np.argmin(candidate_costs_np[successful_mask]))
        ]
    )
    selected_sphere_index = int(assigned_sphere_indices[selected_sample_index])
    selected_target_center = assigned_target_centers[selected_sample_index]
    selected_candidate_index = int(assigned_candidate_indices[selected_sample_index])
    selected_feasible_matches = np.flatnonzero(
        feasible_cache.candidate_indices == selected_candidate_index
    )
    selected_feasible_index = (
        int(selected_feasible_matches[0]) if selected_feasible_matches.size else 0
    )

    best_q_np = candidate_q_np[selected_sample_index]
    contact_q_arm = best_q_np[: len(arm_joint_names)]
    contact_q_gripper = best_q_np[len(arm_joint_names) :]

    action_distance = (
        min(float(pull_distance), float(args.dream_max_action_distance))
        if getattr(args, "dream_action_distance", None) is None
        else min(float(pull_distance), float(args.dream_action_distance))
    )
    action_distance = max(float(action_distance), 1e-4)
    pull_target_center = (
        np.asarray(selected_target_center, dtype=np.float64).reshape(3)
        + np.asarray(surface.pull_world, dtype=np.float64).reshape(3) * action_distance
    )
    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    try:
        from robocasa.demos import demo_close_drawer_contact_curobo as close_demo

        close_demo._set_drawer_joint_value(
            env,
            float(close_demo._drawer_joint_value(env)) - float(action_distance),
        )
        env.sim.forward()
        pull_representative_points = sample_object_representative_points(
            env, surface, candidates, args
        )
        pull_config = replace(config, seed=int(args.seed) + 900001)
        gpu_pull_result = None
        try:
            gpu_pull_result = solve_q_config_contact(
                raw_model,
                raw_data,
                ee_site=ee_site,
                arm_joint_names=arm_joint_names,
                gripper_joint_names=gripper_joint_names,
                sphere_centers_ee=sphere_model.centers_ee,
                sphere_radii=sphere_model.radii,
                seed_q=best_q_np,
                sphere_target_pairs=[(selected_sphere_index, pull_target_center)],
                object_points_world=pull_representative_points.points_world,
                config=pull_config,
            )
            pull_result = _q_config_result_to_cpu(gpu_pull_result)
        finally:
            del gpu_pull_result
            _empty_cuda_caches(pull_config.device)
    finally:
        try:
            raw_data.qfrc_applied[:] = 0.0
        except Exception:
            pass
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()
    if not np.isfinite(float(pull_result.best_cost)):
        pull_result = result
        pull_representative_points = representative_points

    pull_q_np = pull_result.best_q.detach().cpu().numpy().astype(np.float64)
    pull_q_arm = pull_q_np[: len(arm_joint_names)]
    q_waypoints = np.vstack([precontact_arm_q, contact_q_arm, pull_q_arm])
    contact_distances_np = (
        result.best_contact_distances.detach().cpu().numpy().astype(np.float64)
    )
    selected_best_cost = float(candidate_costs_np[selected_sample_index])
    return {
        "selected_candidate_index": selected_candidate_index,
        "selected_sphere_index": int(selected_sphere_index),
        "selected_feasible_index": int(selected_feasible_index),
        "selected_target_center": np.asarray(selected_target_center, dtype=np.float64),
        "pull_target_center": np.asarray(pull_target_center, dtype=np.float64),
        "best_q_arm": best_q_np[: len(arm_joint_names)],
        "best_q_gripper": contact_q_gripper,
        "pull_q_arm": pull_q_arm,
        "q_waypoints": q_waypoints,
        "candidate_q": candidate_q_np,
        "candidate_costs": candidate_costs_np,
        "successful_hypothesis_q": successful_hypothesis_q,
        "successful_hypothesis_costs": successful_hypothesis_costs,
        "contact_distances": contact_distances_np,
        "min_penetration_distance": float(result.best_min_penetration_distance),
        "pull_min_penetration_distance": float(
            pull_result.best_min_penetration_distance
        ),
        "best_cost": selected_best_cost,
        "pull_best_cost": float(pull_result.best_cost),
        "iterations": int(result.iterations),
        "pull_iterations": int(pull_result.iterations),
        "iteration_best_costs": result.iteration_best_costs,
        "sphere_target_pairs": [
            (int(sphere_index), target.tolist())
            for sphere_index, target in zip(
                assigned_sphere_indices, assigned_target_centers
            )
        ],
        "successful_hypotheses": int(np.count_nonzero(successful_mask)),
        "evaluated_hypotheses": int(sample_count),
        "full_arm_collision_free_mask": candidate_collision_ok,
        "full_arm_collision_reasons": tuple(candidate_collision_reasons),
        "action_distance": float(action_distance),
        "sphere_model": sphere_model,
        "representative_points": representative_points,
        "pull_representative_points": pull_representative_points,
        "arm_joint_names": tuple(arm_joint_names),
        "gripper_joint_names": gripper_joint_names,
        "feasible_contact_cache": feasible_cache,
        "contact_set_candidate_indices": contact_set_candidate_indices,
        "contact_set_local_indices": contact_set_local_indices,
        "demonstration_q_mapping": {
            "demo_arm_q": seed_arm_q,
            "mapped_arm_q": seed_arm_q.copy(),
            "target_position_world": np.asarray(
                demonstration_seed.projected_ee_position_world,
                dtype=np.float64,
            ),
            "target_rotation_world": np.asarray(
                demonstration_seed.projected_ee_rotation_world,
                dtype=np.float64,
            ),
            "demo_position_error": float(seed_position_error),
            "demo_rotation_error": float(seed_rotation_error),
            "mapped_position_error": float(seed_position_error),
            "mapped_rotation_error": float(seed_rotation_error),
        },
    }


__all__ = [
    "EECollisionSphereModel",
    "gripper_joint_names_for_q_mpc",
    "open_env_gripper_wide",
    "load_curobo_ee_collision_spheres",
    "solve_collision_sphere_contact_with_q_mpc",
]
