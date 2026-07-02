from dataclasses import dataclass

import numpy as np

import robocasa.demos.demo_close_drawer_contact_curobo as close_demo


@dataclass
class BaseAlignmentResult:
    applied: bool
    reason: str
    yaw_delta: float
    translation_delta: np.ndarray
    initial_position_error: float
    initial_rotation_error: float
    final_position_error: float
    final_rotation_error: float


@dataclass
class PreContactMinkSolution:
    arm_q: np.ndarray
    target_position_world: np.ndarray
    target_rotation_world: np.ndarray
    actual_position_world: np.ndarray
    actual_rotation_world: np.ndarray
    position_error: float
    rotation_error: float
    collision_free: bool
    collision_reason: str


def _import_mink():
    import mink

    return mink


def _make_pose_matrix(pos, rot):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return pose


def _frame_pose_from_configuration(configuration, frame_name, frame_type="site"):
    transform = configuration.get_transform_frame_to_world(frame_name, frame_type)
    matrix = transform.as_matrix()
    return (
        np.asarray(matrix[:3, 3], dtype=np.float64),
        np.asarray(matrix[:3, :3], dtype=np.float64),
    )


def _current_robot_model_q(env, robot_model):
    return close_demo._current_robot_model_q(env, robot_model)


def _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names):
    return close_demo._arm_q_from_robot_model_q(
        robot_model,
        q_robot,
        arm_joint_names,
    )


def _robot_model_q_with_arm(env, robot_model, arm_joint_names, arm_q):
    q_robot = _current_robot_model_q(env, robot_model)
    for joint_name, value in zip(arm_joint_names, np.asarray(arm_q).reshape(-1)):
        if close_demo._mj_has_name(robot_model, "joint", joint_name):
            address = int(robot_model.joint(joint_name).qposadr[0])
            q_robot[address] = float(value)
    return q_robot


def _make_posture_cost(robot_model, arm_joint_names, args):
    return close_demo._make_mink_posture_cost(robot_model, arm_joint_names, args)


def solve_frame_pose(
    env,
    frame_name,
    target_pos,
    target_rot,
    q_start,
    q_posture,
    posture_cost,
    args,
):
    mink = _import_mink()
    robot_model = env.robots[0].robot_model.mujoco_model
    close_demo._sync_mink_base_pose(env, robot_model)

    configuration = mink.Configuration(robot_model)
    configuration.update(np.asarray(q_start, dtype=np.float64).copy())

    posture_task = mink.PostureTask(
        robot_model,
        cost=posture_cost,
        lm_damping=args.mink_posture_lm_damping,
    )
    posture_task.set_target(np.asarray(q_posture, dtype=np.float64).copy())

    frame_task = mink.FrameTask(
        frame_name=frame_name,
        frame_type="site",
        position_cost=args.mink_position_cost,
        orientation_cost=args.mink_orientation_cost,
        lm_damping=args.mink_frame_lm_damping,
    )
    frame_task.set_target(
        mink.SE3.from_matrix(_make_pose_matrix(target_pos, target_rot))
    )

    last_pos_error = np.inf
    for _ in range(int(args.mink_max_iters)):
        velocity = mink.solve_ik(
            configuration,
            [posture_task, frame_task],
            float(args.mink_dt),
            args.mink_solver,
            float(args.mink_damping),
        )
        configuration.integrate_inplace(velocity, float(args.mink_dt))
        actual_pos, _ = _frame_pose_from_configuration(configuration, frame_name)
        last_pos_error = float(np.linalg.norm(actual_pos - target_pos))
        if last_pos_error <= float(args.mink_position_tolerance):
            break
    return configuration.q.copy(), last_pos_error


def _site_pose_for_arm_q(env, arm_joint_names, q_arm, frame_name):
    model = env.sim.model
    data = env.sim.data
    if frame_name not in model._site_name2id:
        raise RuntimeError(f"Cannot read site pose: site '{frame_name}' not found")
    qpos_saved = data.qpos.copy()
    qvel_saved = data.qvel.copy()
    try:
        close_demo._set_env_arm_q(
            env,
            arm_joint_names,
            np.asarray(q_arm, dtype=np.float64).reshape(len(arm_joint_names)),
        )
        env.sim.forward()
        site_id = model.site_name2id(frame_name)
        return (
            np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
            np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
        )
    finally:
        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        env.sim.forward()


def _yaw_from_rot(rot):
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def _base_joint_addresses(model):
    names = (
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_side",
        "mobilebase0_joint_mobile_yaw",
    )
    if not all(name in model._joint_name2id for name in names):
        return None
    return tuple(int(model.get_joint_qpos_addr(name)) for name in names)


def align_base_to_demo_ee_pose(
    env,
    demonstration_seed,
    robot_state,
    args,
):
    if not bool(getattr(args, "align_base_to_demonstration_ee", True)):
        return BaseAlignmentResult(
            applied=False,
            reason="disabled",
            yaw_delta=0.0,
            translation_delta=np.zeros(2, dtype=np.float64),
            initial_position_error=float("nan"),
            initial_rotation_error=float("nan"),
            final_position_error=float("nan"),
            final_rotation_error=float("nan"),
        )

    addresses = _base_joint_addresses(env.sim.model)
    if addresses is None:
        return BaseAlignmentResult(
            applied=False,
            reason="mobile_base_joints_missing",
            yaw_delta=0.0,
            translation_delta=np.zeros(2, dtype=np.float64),
            initial_position_error=float("nan"),
            initial_rotation_error=float("nan"),
            final_position_error=float("nan"),
            final_rotation_error=float("nan"),
        )

    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    demo_arm_q = np.asarray(demonstration_seed.arm_q, dtype=np.float64).reshape(-1)
    target_pos = np.asarray(
        demonstration_seed.projected_ee_position_world, dtype=np.float64
    ).reshape(3)
    target_rot = np.asarray(
        demonstration_seed.projected_ee_rotation_world, dtype=np.float64
    ).reshape(3, 3)

    initial_pos, initial_rot = _site_pose_for_arm_q(
        env,
        arm_joint_names,
        demo_arm_q,
        frame_name,
    )
    initial_position_error = float(np.linalg.norm(initial_pos - target_pos))
    initial_rotation_error = float(np.linalg.norm(initial_rot - target_rot))

    qpos_saved = env.sim.data.qpos.copy()
    qvel_saved = env.sim.data.qvel.copy()
    forward_addr, side_addr, yaw_addr = addresses
    original_base_q = env.sim.data.qpos[[forward_addr, side_addr, yaw_addr]].copy()
    yaw_delta = _yaw_from_rot(target_rot @ initial_rot.T)
    max_yaw_delta = float(getattr(args, "demonstration_base_max_yaw_delta", np.pi))
    yaw_delta = float(np.clip(yaw_delta, -max_yaw_delta, max_yaw_delta))
    translation_delta = np.zeros(2, dtype=np.float64)

    try:
        env.sim.data.qpos[yaw_addr] = float(env.sim.data.qpos[yaw_addr] + yaw_delta)
        env.sim.forward()
        slide_addrs = (forward_addr, side_addr)
        eps = 1e-4
        for _ in range(max(int(getattr(args, "demonstration_base_align_iters", 2)), 1)):
            pos, _ = _site_pose_for_arm_q(env, arm_joint_names, demo_arm_q, frame_name)
            error_xy = target_pos[:2] - pos[:2]
            if float(np.linalg.norm(error_xy)) <= float(
                getattr(args, "demonstration_base_position_tolerance", 0.003)
            ):
                break
            jac = np.zeros((2, 2), dtype=np.float64)
            base_before = env.sim.data.qpos[list(slide_addrs)].copy()
            for col, addr in enumerate(slide_addrs):
                env.sim.data.qpos[addr] += eps
                env.sim.forward()
                bumped_pos, _ = _site_pose_for_arm_q(
                    env,
                    arm_joint_names,
                    demo_arm_q,
                    frame_name,
                )
                jac[:, col] = (bumped_pos[:2] - pos[:2]) / eps
                env.sim.data.qpos[list(slide_addrs)] = base_before
                env.sim.forward()
            dq, *_ = np.linalg.lstsq(jac, error_xy, rcond=None)
            max_step = float(
                getattr(args, "demonstration_base_max_translation_step", 0.5)
            )
            norm = float(np.linalg.norm(dq))
            if norm > max_step:
                dq *= max_step / max(norm, 1e-9)
            env.sim.data.qpos[forward_addr] += float(dq[0])
            env.sim.data.qpos[side_addr] += float(dq[1])
            translation_delta += dq
            env.sim.forward()

        final_pos, final_rot = _site_pose_for_arm_q(
            env,
            arm_joint_names,
            demo_arm_q,
            frame_name,
        )
        final_position_error = float(np.linalg.norm(final_pos - target_pos))
        final_rotation_error = float(np.linalg.norm(final_rot - target_rot))
        final_base_q = env.sim.data.qpos[[forward_addr, side_addr, yaw_addr]].copy()
    except Exception:
        env.sim.data.qpos[:] = qpos_saved
        env.sim.data.qvel[:] = qvel_saved
        env.sim.forward()
        raise

    env.sim.data.qpos[:] = qpos_saved
    env.sim.data.qvel[:] = qvel_saved
    env.sim.data.qpos[[forward_addr, side_addr, yaw_addr]] = final_base_q
    env.sim.forward()

    applied = not np.allclose(final_base_q, original_base_q)
    return BaseAlignmentResult(
        applied=bool(applied),
        reason="aligned",
        yaw_delta=float(yaw_delta),
        translation_delta=np.asarray(translation_delta, dtype=np.float64),
        initial_position_error=initial_position_error,
        initial_rotation_error=initial_rotation_error,
        final_position_error=final_position_error,
        final_rotation_error=final_rotation_error,
    )


def solve_precontact_pose(
    env,
    panel,
    robot_state,
    target_position_world,
    target_rotation_world,
    seed_arm_q,
    args,
    collision_checker=None,
):
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    robot_model = env.robots[0].robot_model.mujoco_model
    arm_joint_names = tuple(robot_state["robocasa_joint_names"])
    q_start = _robot_model_q_with_arm(
        env,
        robot_model,
        arm_joint_names,
        seed_arm_q,
    )
    q_posture = _robot_model_q_with_arm(
        env,
        robot_model,
        arm_joint_names,
        np.asarray(robot_state["q"], dtype=np.float64),
    )
    posture_cost = _make_posture_cost(robot_model, arm_joint_names, args)
    q_robot, position_error = solve_frame_pose(
        env,
        frame_name,
        np.asarray(target_position_world, dtype=np.float64).reshape(3),
        np.asarray(target_rotation_world, dtype=np.float64).reshape(3, 3),
        q_start,
        q_posture,
        posture_cost,
        args,
    )
    q_arm = _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
    actual_pos, actual_rot = _site_pose_for_arm_q(
        env,
        arm_joint_names,
        q_arm,
        frame_name,
    )
    rotation_error = float(
        np.linalg.norm(
            actual_rot
            - np.asarray(target_rotation_world, dtype=np.float64).reshape(3, 3)
        )
    )
    if collision_checker is None:
        collision_free, collision_reason = close_demo._check_arm_q_collision(
            env,
            panel,
            arm_joint_names,
            q_arm,
            allowed_ee_geom_name=None,
            penetration_tolerance=float(args.mink_collision_penetration_tolerance),
        )
    else:
        collision_free, collision_reason = collision_checker(q_arm)
    return PreContactMinkSolution(
        arm_q=np.asarray(q_arm, dtype=np.float64),
        target_position_world=np.asarray(target_position_world, dtype=np.float64),
        target_rotation_world=np.asarray(target_rotation_world, dtype=np.float64),
        actual_position_world=np.asarray(actual_pos, dtype=np.float64),
        actual_rotation_world=np.asarray(actual_rot, dtype=np.float64),
        position_error=float(position_error),
        rotation_error=rotation_error,
        collision_free=bool(collision_free),
        collision_reason=str(collision_reason),
    )
