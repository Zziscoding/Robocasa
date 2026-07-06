"""Open-drawer cuRobo trajectory-planning stage.

Assembles ``OpenDrawerStage`` objects from the MIQP/DAQP/pre-grasp results
in the context, flattens them to target gripper poses, and calls
``demo_close_drawer_contact_curobo.plan_with_curobo`` (shared with the
close_drawer pipeline) for final arm trajectory planning.

The OpenDrawerStage is built via
``demo_open_drawer_contact_curobo._solve_stage`` — exactly as the monolithic
reference pipeline does — so that waypoint target poses are consistent.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

import numpy as np

from .context import PipelineContext, autogen_print


def _build_open_drawer_stage(ctx: PipelineContext, args: Any) -> Any:
    """Construct an ``OpenDrawerStage`` from the final pre-grasp solution."""
    from robocasa.demos.demo_open_drawer_contact_curobo import OpenDrawerStage
    from robocasa.demos.demo_close_drawer_contact_curobo import (
        _drawer_joint_value,
    )

    sol = ctx.pregrasping_solution
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    drawer_q = float(_drawer_joint_value(ctx.env))

    # Recompute target_gripper_poses from the accepted pre-grasp (precontact +
    # contact only — grasp does not have a pull phase).
    contact_pos = np.asarray(sol.actual_position_world, dtype=np.float64).reshape(3)
    contact_rot = np.asarray(sol.actual_rotation_world, dtype=np.float64).reshape(3, 3)
    approach = np.asarray(
        getattr(ctx.surface, "approach_world", (1.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    approach_norm = approach / max(float(np.linalg.norm(approach)), 1e-9)
    precontact_pos = contact_pos + approach_norm * float(
        getattr(args, "precontact_distance", 0.04)
    )

    gripper_opening = float(getattr(sol, "gripper_opening", 0.04))

    # Grasp-rollout forces the contact position to be best-effort if there was
    # no cold sphere-based IK; keep the sphere model as None so we go through
    # the "no-q_config" path in _solve_stage. The grasp path handles planning
    # outside that helper; here we just synthesize an OpenDrawerStage shape.
    stage = OpenDrawerStage(
        name="grasp",
        surface_name=str(getattr(ctx.surface, "name", "handle_inner")),
        start_drawer_q=drawer_q,
        pull_distance=0.0,  # no pull in pure-grasp mode
        selected_contact_index=int(ctx.pregrasping_pair_index),
        selected_contact_world=contact_pos,
        selected_contact_local=contact_pos,  # same (in the object frame)
        selected_contact_cost=0.0,
        mink_solution=_MinkSolutionShim(
            arm_q=np.asarray(sol.arm_q, dtype=np.float64).reshape(7),
            target_position_world=np.asarray(
                sol.target_position_world, dtype=np.float64
            ).reshape(3),
            target_rotation_world=np.asarray(
                sol.target_rotation_world, dtype=np.float64
            ).reshape(3, 3),
            actual_position_world=contact_pos,
            actual_rotation_world=contact_rot,
            position_error=float(sol.position_error),
            rotation_error=float(sol.rotation_error),
            collision_free=bool(sol.collision_free),
            collision_reason=str(sol.collision_reason),
            gripper_opening=gripper_opening,
            contact_offset_ee=np.zeros(3, dtype=np.float64),
            contact_frame=frame_name,
            q_waypoints=np.zeros((0, 7), dtype=np.float64),
            target_gripper_poses=[
                ("precontact", precontact_pos, contact_rot),
                ("contact", contact_pos, contact_rot),
            ],
            contact_position_error=float(sol.position_error),
        ),
        candidates=list(ctx.candidates),
        mink_reports=[],
    )
    return stage


class _MinkSolutionShim:
    """Minimal shim mirroring the fields of ``MinkContactPoseSolution``.

    The open-drawer planner does not produce a full q-MPC solution like the
    close_drawer path; the pre-grasp solver gives us a single (precontact,
    contact) pair instead. This shim exposes the attributes
    ``_all_target_gripper_poses`` and ``plan_with_curobo`` look for.
    """

    def __init__(
        self,
        *,
        arm_q: np.ndarray,
        target_position_world: np.ndarray,
        target_rotation_world: np.ndarray,
        actual_position_world: np.ndarray,
        actual_rotation_world: np.ndarray,
        position_error: float,
        rotation_error: float,
        collision_free: bool,
        collision_reason: str,
        gripper_opening: float,
        contact_offset_ee: np.ndarray,
        contact_frame: str,
        q_waypoints: np.ndarray,
        target_gripper_poses: list,
        contact_position_error: float,
    ):
        self.arm_q = arm_q
        self.target_position_world = target_position_world
        self.target_rotation_world = target_rotation_world
        self.actual_position_world = actual_position_world
        self.actual_rotation_world = actual_rotation_world
        self.position_error = position_error
        self.rotation_error = rotation_error
        self.collision_free = collision_free
        self.collision_reason = collision_reason
        self.gripper_opening = gripper_opening
        self.contact_offset_ee = contact_offset_ee
        self.contact_frame = contact_frame
        self.q_waypoints = q_waypoints
        self.target_gripper_poses = target_gripper_poses
        self.contact_position_error = contact_position_error


def plan_open_drawer_trajectory_stage(
    ctx: PipelineContext, args: Any
) -> PipelineContext:
    """Plan a cuRobo arm trajectory through the precontact → contact waypoints."""
    if ctx.pregrasping_solution is None:
        autogen_print("stage=plan_open_drawer_skipped reason=no_pregrasping_solution")
        return ctx

    stage = _build_open_drawer_stage(ctx, args)
    ctx.open_stages = [stage]

    from robocasa.demos.demo_close_drawer_contact_curobo import (
        gripper_pose_to_curobo_hand_pose,
        plan_with_curobo,
    )
    from robocasa.demos.demo_open_drawer_contact_curobo import (
        _all_target_gripper_poses,
    )

    target_gripper_poses = _all_target_gripper_poses(ctx.open_stages)
    autogen_print(f"stage=plan_open_drawer_poses n_poses={len(target_gripper_poses)}")
    target_hand_poses = [
        (name, *gripper_pose_to_curobo_hand_pose(pos, rot, ctx.robot_state))
        for name, pos, rot in target_gripper_poses
    ]

    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(
            open(os.devnull, "w")
        ), contextlib.redirect_stderr(open(os.devnull, "w")):
            q_traj, segments = plan_with_curobo(
                ctx.robot_state,
                target_hand_poses,
                args,
                env=ctx.env,
            )
    except Exception as exc:
        autogen_print(
            f"curobo_status=failed_nonfatal "
            f"curobo_time={time.perf_counter() - started:.6f} "
            f"error={type(exc).__name__}: {exc}"
        )
        return ctx

    ctx.q_traj = q_traj
    ctx.segments = list(segments) if segments else []
    autogen_print(
        f"curobo_time={time.perf_counter() - started:.6f} "
        f"successful_trajectories={len(ctx.segments)}"
    )
    return ctx
