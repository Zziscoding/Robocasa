"""Pipeline context: carries all intermediate state between autogen stages."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


def autogen_print(message: str) -> None:
    """Print to the real stdout (bypass any redirection)."""
    print(message, file=sys.__stdout__, flush=True)


@dataclass
class PipelineContext:
    """Mutable container shared by every pipeline stage.

    Each stage reads what it needs and writes its outputs back so the next
    stage can consume them.  This keeps the stage signatures uniform
    ``(ctx, args) -> ctx`` while avoiding a sprawling parameter list.
    """

    # --- scene ---
    env: Any = None
    panel: Any = None
    robot_state: dict = field(default_factory=dict)

    # --- contact candidates ---
    candidates: list = field(default_factory=list)
    selected: Any = None
    push_distance: float = 0.0
    feasible_cache: Any = None

    # --- skeleton poses (DAQP output) ---
    # List of (candidate_index, SkeletonPose) tuples, theta-diverse ordered.
    skeleton_poses: list = field(default_factory=list)

    # --- step-2 solutions ---
    # Final accepted MinkContactSolution objects.
    solutions: list = field(default_factory=list)
    # Per-attempt reports (MinkContactAttemptReport).
    reports: list = field(default_factory=list)
    # Refined floating-EE poses: list of (pos, rot, gripper, candidate_index).
    refined_poses: list = field(default_factory=list)
    # The single best solution (selected by cost).
    mink_solution: Any = None

    # --- trajectory ---
    q_traj: Any = None
    segments: list = field(default_factory=list)

    # --- scratch / debug ---
    # Any stage may stash diagnostics here for the logger.
    diagnostics: dict = field(default_factory=dict)
