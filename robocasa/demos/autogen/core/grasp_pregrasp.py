"""Pre-grasp solver stage for the open-drawer grasp pipeline.

Wraps ``demo_open_drawer_autogen._solve_grasp_precontact_autogen`` which runs the
final refinement step: given the dual-finger DAQP skeleton poses from the
previous stage, solve a pre-grasp arm-q (either via mink or MPPI) and run the
grasp-closure rollout that scores ``force_closure_cost``.

Populates ``ctx.pregrasping_solution`` (a ``PreContactMinkSolution``) and
``ctx.pregrasping_pair_index``.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

from .context import PipelineContext, autogen_print


def solve_pregrasp_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Select the best pre-grasp (mink or MPPI) + score force-closure."""
    from robocasa.demos import demo_open_drawer_autogen as open_autogen

    if not ctx.grasp_skeleton_poses:
        autogen_print("stage=solve_pregrasp_skipped reason=no skeleton poses")
        return ctx

    started = time.perf_counter()
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
        open(os.devnull, "w")
    ):
        solution, pair_index = open_autogen._solve_grasp_precontact_autogen(
            ctx.env,
            ctx.surface,
            ctx.candidates,
            ctx.demonstration_seed,
            ctx.robot_state,
            args,
        )

    elapsed = time.perf_counter() - started
    ctx.pregrasping_solution = solution
    ctx.pregrasping_pair_index = int(pair_index)
    autogen_print(
        f"stage=solve_pregrasp_done "
        f"pair_index={int(pair_index)} "
        f"elapsed_s={elapsed:.3f}"
    )
    return ctx
