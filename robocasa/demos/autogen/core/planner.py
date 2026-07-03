"""Trajectory planning: cuRobo arm trajectory from contact solutions.

Builds the target hand poses from the selected contact solution and calls
``demo_close_drawer_contact_curobo.plan_with_curobo`` (with the mink
q-waypoints as joint-space guidance when available).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robocasa.demos.demo_close_drawer_contact_curobo import (  # noqa: E402
    build_target_gripper_poses,
    gripper_pose_to_curobo_hand_pose,
    plan_with_curobo,
)

from .context import PipelineContext, autogen_print


def plan_trajectory_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Plan a cuRobo arm trajectory through the contact solution waypoints.

    Populates ``ctx.q_traj`` and ``ctx.segments``.
    """
    if ctx.mink_solution is None:
        autogen_print("stage=plan_trajectory_skipped reason=no_solution")
        return ctx

    selected = ctx.candidates[ctx.mink_solution.drawer_candidate_index]
    target_gripper_poses = build_target_gripper_poses(
        ctx.panel, selected, ctx.push_distance, args
    )
    target_hand_poses = [
        (name, *gripper_pose_to_curobo_hand_pose(pos, rot, ctx.robot_state))
        for name, pos, rot in target_gripper_poses
    ]

    mink_q_waypoints = None
    if bool(getattr(args, "curobo_mink_joint_space", True)):
        q_waypoints = np.asarray(
            getattr(ctx.mink_solution, "q_waypoints", np.zeros((0, 7))),
            dtype=np.float64,
        )
        if q_waypoints.size:
            q_waypoints = q_waypoints.reshape(-1, 7)
        if q_waypoints.shape[0] > 0:
            mink_q_waypoints = q_waypoints

    started = time.perf_counter()
    try:
        q_traj, segments = plan_with_curobo(
            ctx.robot_state,
            target_hand_poses,
            args,
            env=ctx.env,
            q_waypoints=mink_q_waypoints,
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
