"""Open-drawer env + handle-surface + demonstration-seed stages.

These stages build the OPENDRAWER MuJoCo env (different from the close-drawer
env used in the push pipeline) and extract the env-level data the rest of the
grasp pipeline needs:

* ``ctx.env`` — the OPENDRAWER env (from ``open_drawer.viewer.create_open_drawer_env``).
* ``ctx.panel`` — the drawer ``PanelFrame`` (reuses ``close_demo.get_panel_frame``).
* ``ctx.robot_state`` — the robot arm state dict (reuses ``close_demo.get_robot_arm_state``).
* ``ctx.surface`` — the handle inner surface in the drawer frame
  (``open_drawer.scene_process.make_handle_inner_surface``).
* ``ctx.demonstration_seed`` — projected demonstration seed
  (``demo_open_drawer_contact_curobo._load_and_project_demonstration_seed``).

The scene-processing step is the open_drawer-specific one that seeds the
projection parameters used by the demonstration loading.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from .context import PipelineContext, autogen_print


_OPEN_DEPS_LOADED = False


def _ensure_open_deps() -> None:
    """Import the open_drawer sub-tree quietly (it's heavy and noisy)."""
    global _OPEN_DEPS_LOADED
    if _OPEN_DEPS_LOADED:
        return
    _OPEN_DEPS_LOADED = True

    # Silence the open_drawer import banner / mujoco warnings.
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
        open(os.devnull, "w")
    ):
        import robocasa.demos.open_drawer.scene as _  # noqa: F401
    autogen_print("[grasp_env] open_drawer runtime loaded")


def build_open_drawer_env_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Build the OPENDRAWER env (not the close_drawer env)."""
    _ensure_open_deps()
    from robocasa.demos.open_drawer.viewer import create_open_drawer_env

    autogen_print("stage=build_open_drawer_env")
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
        open(os.devnull, "w")
    ):
        ctx.env = create_open_drawer_env(args)
    return ctx


def set_panel_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Get the drawer ``PanelFrame`` from the OPENDRAWER env."""
    from robocasa.demos.demo_close_drawer_contact_curobo import (
        get_panel_frame,
    )

    autogen_print("stage=set_panel")
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
        open(os.devnull, "w")
    ):
        ctx.panel = get_panel_frame(ctx.env)
    return ctx


def set_robot_state_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Get the robot arm state dict from the OPENDRAWER env."""
    from robocasa.demos.demo_close_drawer_contact_curobo import (
        get_robot_arm_state,
    )

    autogen_print("stage=set_robot_state")
    ctx.robot_state = get_robot_arm_state(ctx.env)
    return ctx


def set_surface_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Build the handle inner surface in the drawer frame."""
    import robocasa.demos.open_drawer.scene_process as open_scene_process
    import robocasa.demos.demo_open_drawer_contact_curobo as open_demo

    autogen_print("stage=set_surface")
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
        open(os.devnull, "w")
    ):
        # must run the open_drawer-specific scene processing first
        open_scene_process.initialize_scene_processing(ctx.env, args)
        ctx.surface = open_scene_process.make_handle_inner_surface(ctx.env, ctx.panel)
    return ctx


def load_demonstration_seed_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Project the demonstration episode seed into the current env frame."""
    import robocasa.demos.demo_open_drawer_contact_curobo as open_demo

    autogen_print("stage=load_demonstration_seed")
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
        open(os.devnull, "w")
    ):
        ctx.demonstration_seed = open_demo._load_and_project_demonstration_seed(
            ctx.env, args
        )
    return ctx
