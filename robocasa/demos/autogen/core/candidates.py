"""Contact candidate generation: COACD decomposition + feasible filtering.

Delegates to ``demo_close_drawer_contact_curobo.evaluate_contacts`` which
internally runs ``_build_autogen_contact_candidates`` (COACD convex
decomposition of the drawer handle/panel geoms, uniform sampling over the
convex parts, and a batched contact-optimizer solve that labels each sample
as feasible or not).
"""

from __future__ import annotations

from typing import Any

from robocasa.demos.demo_close_drawer_contact_curobo import (  # noqa: E402
    evaluate_contacts,
)

from .context import PipelineContext, autogen_print


def evaluate_contacts_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Populate ``ctx.candidates``, ``ctx.selected``, ``ctx.push_distance``,
    ``ctx.feasible_cache`` from the env + panel.
    """
    autogen_print(
        "stage=evaluate_contacts "
        f"contact_cost_threshold={float(getattr(args, 'contact_cost_threshold', 0.35)):.4f}"
    )
    candidates, selected, push_distance = evaluate_contacts(ctx.env, ctx.panel, args)
    ctx.candidates = list(candidates)
    ctx.selected = selected
    ctx.push_distance = float(push_distance)
    ctx.feasible_cache = getattr(args, "_autogen_feasible_cache", None)
    autogen_print(
        f"stage=evaluate_contacts_done "
        f"candidates={len(ctx.candidates)} "
        f"feasible={sum(1 for c in ctx.candidates if bool(getattr(c, 'feasible', False)))} "
        f"push_distance={ctx.push_distance:.6f}"
    )
    return ctx
