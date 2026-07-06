"""Visualization popup stages for the open-drawer grasp pipeline.

Thin wrapper stages around the popups in ``demo_open_drawer_autogen``. Each
takes ``(ctx, args)`` and reads what it needs from the context — keeping the
stage signature uniform with the close-drawer viz stages.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from .context import PipelineContext, autogen_print


def visualize_contact_pairs_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Popup viewer showing MIQP contact pairs as colored spheres."""
    if not bool(getattr(args, "autogen_visualize_contact_pairs", True)):
        return ctx
    if not ctx.miqp_pairs:
        return ctx
    from robocasa.demos import demo_open_drawer_autogen as open_autogen

    try:
        with contextlib.redirect_stdout(
            open(os.devnull, "w")
        ), contextlib.redirect_stderr(open(os.devnull, "w")):
            open_autogen._visualize_contact_pairs_popup(ctx.env, ctx.miqp_pairs, args)
    except Exception as exc:
        autogen_print(f"viz=contact_pairs_error error={exc!r}")
    return ctx


def visualize_mink_q_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Popup viewer of ghost Panda hands at the pre-contact q waypoints.

    Uses the pre-grasp arm-q stored on ``ctx.pregrasping_solution`` as a single
    frame (the open_drawer path does not produce multi-waypoint q lists like
    close_drawer — instead it has a single pre-grasp solution).
    """
    if not bool(getattr(args, "autogen_visualize_mink_poses", True)):
        return ctx
    if ctx.pregrasping_solution is None:
        return ctx
    from robocasa.demos import demo_open_drawer_autogen as open_autogen

    try:
        q_waypoints = np_atleast_1d(ctx.pregrasping_solution.arm_q).reshape(-1, 7)
        with contextlib.redirect_stdout(
            open(os.devnull, "w")
        ), contextlib.redirect_stderr(open(os.devnull, "w")):
            open_autogen._visualize_mink_q_poses_popup(
                ctx.env,
                q_waypoints,
                ctx.robot_state,
                args,
            )
    except Exception as exc:
        autogen_print(f"viz=mink_q_error error={exc!r}")
    return ctx


def visualize_grasp_precontact_stage(
    ctx: PipelineContext, args: Any
) -> PipelineContext:
    """Ghost popup of the DAQP skeleton poses.

    Renders ghosts at every accepted pregrasp candidate. The accepted pose
    is highlighted green; others are colored by rollout score.
    """
    if not bool(getattr(args, "autogen_visualize_grasp_precontact", True)):
        return ctx
    if not ctx.grasp_skeleton_poses:
        return ctx
    from robocasa.demos import demo_open_drawer_autogen as open_autogen

    # Build the (q_arm, gripper_opening, rollout_score, is_accepted) tuples
    # that the ghost popup expects. We re-rollout-evaluate to recover the score
    # per pose; cheap enough since the rollout is ~15 steps.
    from robocasa.demos.demo_close_drawer_contact_curobo import (
        _drawer_joint_value,
    )

    drawer_q = float(_drawer_joint_value(ctx.env))
    accepted_index = int(ctx.pregrasping_pair_index)

    # Delegate to the MIQP-skeleton stage's dedicated skeleton-popup (it knows
    # how to build SkeletonPose ghosts from the DAQP results). The precontact
    # ghosts popup called from `_solve_grasp_precontact_autogen` requires
    # per-pose rollout scores we don't recompute here, so this path uses the
    # lighter `visualize_skeleton_poses` hook in `demo_open_drawer_autogen`.
    from robocasa.demos import ee_skelton

    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    try:
        skeleton = ee_skelton.build_panda_skeleton(ctx.env, frame_name)
        poses = [sp for _, sp, _ in ctx.grasp_skeleton_poses]
        _viz_backup = getattr(args, "autogen_visualize_skeleton_poses", True)
        try:
            args.autogen_visualize_skeleton_poses = True
            ee_skelton.visualize_skeleton_poses(
                ctx.env, frame_name, skeleton, poses, args
            )
        finally:
            args.autogen_visualize_skeleton_poses = _viz_backup
    except Exception as exc:
        autogen_print(f"viz=grasp_precontact_error error={exc!r}")
    return ctx
