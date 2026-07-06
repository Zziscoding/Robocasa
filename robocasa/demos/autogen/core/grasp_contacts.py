"""Contact-candidate generation stage for the open-drawer (grasp) pipeline.

Wraps ``demo_open_drawer_autogen.evaluate_open_contacts`` which internally runs
the COACD-based candidate-generation + friction-cone filtering, exactly as the
monolithic ``demo_open_drawer_autogen._solve_stage_autogen`` would — so the
modular pipeline produces identical candidates.

Also stashes the cached handle-mesh pose (``mesh_path``, ``obj_pos``, ``obj_quat``)
set by ``_export_grasp_handle_mesh`` on ``args._autogen_grasp_*`` so the
downstream rollout can score force closure on the same geometry.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

from .context import PipelineContext, autogen_print


def evaluate_open_contacts_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Run COACD candidate generation over the handle inner surface.

    Populates ``ctx.candidates``, ``ctx.selected``, ``ctx.push_distance``,
    ``ctx.feasible_cache`` (the ``AutogenFeasibleContactCache``) and, via the
    cached side effect, ``args._autogen_grasp_*`` mesh fields.
    """
    from robocasa.demos import demo_open_drawer_contact_curobo as open_demo
    from robocasa.demos.demo_open_drawer_autogen import evaluate_open_contacts

    # total pull distance for the TARGET_OPEN_FARCTION open
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
        open(os.devnull, "w")
    ):
        pull_distance = open_demo._target_pull_distance(ctx.env, args)

    autogen_print(
        f"stage=evaluate_open_contacts " f"pull_distance={float(pull_distance):.6f}"
    )
    started = time.perf_counter()
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
        open(os.devnull, "w")
    ):
        candidates, selected = evaluate_open_contacts(
            ctx.env, ctx.surface, pull_distance, args
        )

    elapsed = time.perf_counter() - started
    feasible_count = sum(1 for c in candidates if bool(getattr(c, "feasible", False)))
    ctx.candidates = list(candidates)
    ctx.selected = selected
    ctx.push_distance = float(pull_distance)
    ctx.feasible_cache = getattr(args, "_autogen_feasible_cache", None)
    autogen_print(
        f"stage=evaluate_open_contacts_done "
        f"candidates={len(ctx.candidates)} "
        f"feasible={feasible_count} "
        f"elapsed_s={elapsed:.3f}"
    )
    return ctx
