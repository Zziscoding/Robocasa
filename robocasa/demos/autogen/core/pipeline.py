"""Pipeline assembly + task registry.

A *pipeline* is an ordered list of stage callables ``(ctx, args) -> ctx``.
The ``TASK_REGISTRY`` maps a task name to its pipeline. Adding a new task
means registering a new entry here — the stages themselves live in the
``core`` modules and are reused across tasks.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

from . import (
    candidates,
    grasp_contacts,
    grasp_env,
    grasp_miqp_skeleton,
    grasp_planner,
    grasp_pregrasp,
    grasp_viz,
    planner,
    scene,
    skeleton,
    step2,
    viz,
)
from .context import PipelineContext, autogen_print

Stage = Callable[[PipelineContext, argparse.Namespace], PipelineContext]


def _close_drawer_pipeline(args: argparse.Namespace) -> list[Stage]:
    """Build the close-drawer pipeline: COACD → DAQP → step-2 → rollout → cuRobo."""
    return [
        lambda ctx, a=args: _build_env(ctx, a),
        _set_panel,
        _set_robot_state,
        candidates.evaluate_contacts_stage,
        skeleton.solve_skeleton_poses_stage,
        step2.solve_step2_stage,
        lambda ctx, a=args: _viz_mink_poses(ctx, a),
        lambda ctx, a=args: _viz_floating_ee(ctx, a),
        planner.plan_trajectory_stage,
        lambda ctx, a=args: _viz_contact_pd(ctx, a),
    ]


def _open_drawer_grasp_pipeline(args: argparse.Namespace) -> list[Stage]:
    """Build the open-drawer *grasp* pipeline.

    OPENDRAWER env → handle surface → demonstration seed → COACD contacts →
    MIQP contact pairs → dual-finger DAQP skeleton → pre-grasp mink/MPPI →
    grasp rollout filter → cuRobo trajectory.
    """
    return [
        grasp_env.build_open_drawer_env_stage,
        grasp_env.set_panel_stage,
        grasp_env.set_robot_state_stage,
        grasp_env.set_surface_stage,
        grasp_env.load_demonstration_seed_stage,
        grasp_contacts.evaluate_open_contacts_stage,
        grasp_viz.visualize_contact_pairs_stage,
        grasp_miqp_skeleton.miqp_and_skeleton_stage,
        grasp_pregrasp.solve_pregrasp_stage,
        grasp_viz.visualize_grasp_precontact_stage,
        grasp_planner.plan_open_drawer_trajectory_stage,
    ]


def _build_env(ctx: PipelineContext, args: argparse.Namespace) -> PipelineContext:
    ctx.env = scene.build_env(args)
    return ctx


def _set_panel(ctx: PipelineContext, args: argparse.Namespace) -> PipelineContext:
    ctx.panel = scene.get_panel(ctx.env)
    return ctx


def _set_robot_state(ctx: PipelineContext, args: argparse.Namespace) -> PipelineContext:
    ctx.robot_state = scene.get_robot_state(ctx.env)
    return ctx


def _viz_mink_poses(ctx: PipelineContext, args: argparse.Namespace) -> PipelineContext:
    try:
        viz.visualize_mink_q_poses_popup(ctx, args)
    except Exception as exc:
        autogen_print(f"[autogen] mink-q popup skipped: {exc}")
    return ctx


def _viz_floating_ee(ctx: PipelineContext, args: argparse.Namespace) -> PipelineContext:
    try:
        viz.visualize_floating_ee_poses_popup(ctx, args)
    except Exception as exc:
        autogen_print(f"[autogen] floating-EE popup skipped: {exc}")
    return ctx


def _viz_contact_pd(ctx: PipelineContext, args: argparse.Namespace) -> PipelineContext:
    try:
        viz.visualize_contact_marker_with_physical_pd(ctx, args)
    except Exception as exc:
        autogen_print(f"[autogen] contact PD popup skipped: {exc}")
    return ctx


# --- Task registry ----------------------------------------------------------
TASK_REGISTRY: dict[str, Callable[[argparse.Namespace], list[Stage]]] = {
    "close_drawer": _close_drawer_pipeline,
    "open_drawer": _open_drawer_grasp_pipeline,
}


def run_pipeline(
    ctx: PipelineContext,
    args: argparse.Namespace,
    stages: list[Stage],
) -> PipelineContext:
    """Execute the pipeline stages in order."""
    for stage in stages:
        ctx = stage(ctx, args)
    return ctx


def get_stages(task_name: str, args: argparse.Namespace) -> list[Stage]:
    """Look up the pipeline for ``task_name`` and return its stages."""
    builder = TASK_REGISTRY.get(task_name)
    if builder is None:
        raise ValueError(
            f"Unknown task {task_name!r}. Available: {list(TASK_REGISTRY)}"
        )
    return builder(args)
