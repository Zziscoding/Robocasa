"""Scene construction: build env, extract panel frame, get robot state.

Thin wrappers around ``demo_close_drawer_contact_curobo`` so the pipeline
can call them without importing the demo's module-level globals.
"""

from __future__ import annotations

from typing import Any

from robocasa.demos.demo_close_drawer_contact_curobo import (  # noqa: E402
    create_close_drawer_env,
    get_panel_frame,
    get_robot_arm_state,
)


def build_env(args: Any) -> Any:
    """Build the close-drawer MuJoCo env."""
    return create_close_drawer_env(args)


def get_panel(env: Any) -> Any:
    """Extract the drawer ``PanelFrame`` from the env."""
    return get_panel_frame(env)


def get_robot_state(env: Any) -> dict:
    """Get the robot arm state dict (joint names, q, base pose, ...)."""
    return get_robot_arm_state(env)
