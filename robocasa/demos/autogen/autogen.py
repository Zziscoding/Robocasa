"""Unified autogen entry point.

Reads a YAML config and dispatches to the right manipulation pipeline:

1. Load YAML (defaults to ``open_drawer_contact_curobo.yaml``).
   The YAML's ``task_name`` field (e.g. ``open_drawer`` / ``close_drawer``)
   and/or ``drawer_action`` field selects the scene + action.
2. Run the world-model prehensile-need probe (close_drawer path only — the
   open_drawer path is always grasp; its probe is a no-op).
3. Look up the task pipeline in ``core.pipeline.TASK_REGISTRY`` and execute
   the corresponding stages:

   * ``close_drawer`` → COACD → feasible candidates → DAQP skeleton poses →
     MPPI/mink step-2 → rollout filtering → cuRobo trajectory, with 3 popup
     visualizations.
   * ``open_drawer`` → COACD → MIQP contact pairs → dual-finger DAQP
     skeleton → pre-grasp mink/MPPI + grasp rollout filter → cuRobo
     trajectory, with 2 popup visualizations.

Adding a new task means registering a new builder in
``core.pipeline.TASK_REGISTRY`` — no changes in this file are required.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("ROBOCASA_ALLOW_VERSION_MISMATCH", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from robocasa.demos.demo_open_drawer_contact_curobo import (
    _apply_config_overrides,
    _load_yaml_config,
)
from robocasa.demos import world_model as wm

from .core.context import PipelineContext, autogen_print
from .core.pipeline import get_stages, run_pipeline


_DEFAULT_YAML = Path(__file__).with_name("open_drawer_contact_curobo.yaml")


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified autogen demo (task selected by YAML)."
    )
    parser.add_argument("--config", type=str, default=str(_DEFAULT_YAML))
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Override task_name from the YAML (e.g. open_drawer, close_drawer).",
    )
    return parser.parse_args()


def _resolve_task(config: dict, cli_override: str | None) -> tuple[str, str]:
    """Return (task_name, drawer_action). task_name is the high-level label."""
    task = (cli_override or config.get("task_name") or "").strip().lower()
    if not task:
        drawer_action = str(config.get("drawer_action", "open")).strip().lower()
        task = f"{drawer_action}_drawer"
    if task.startswith("close"):
        return "close_drawer", "close"
    return "open_drawer", "open"


def _run_world_model_probe(args, task_name: str) -> dict:
    """Build env + feasible cache via the open-drawer autogen flow and probe.

    The open-drawer autogen pipeline already does scene construction +
    COACD-based feasible candidate generation. We hijack its solver entry to
    extract the candidates, run the prehensile-need probe, and return early
    instead of going into curobo planning.
    """
    from robocasa.demos import demo_open_drawer_autogen as open_autogen
    from robocasa.demos import demo_open_drawer_contact_curobo as open_demo

    probe_result: dict = {}

    def _evaluate_and_probe(env, surface, pull_distance, args_):
        candidates, selected = open_autogen.evaluate_open_contacts(
            env, surface, pull_distance, args_
        )
        feasible_points = np.asarray(
            [c.world_point for c in candidates if bool(c.feasible)],
            dtype=np.float64,
        ).reshape(-1, 3)
        result = wm.evaluate_prehensile_need(
            env,
            surface,
            feasible_points,
            pull_direction_world=getattr(surface, "pull_world", (1.0, 0.0, 0.0)),
            args=args_,
            task_name=task_name,
            expected_displacement=float(pull_distance),
        )
        probe_result.update(result)
        raise _ProbeDone()

    base_globals = open_demo.main.__globals__
    saved_eval = base_globals.get("evaluate_open_contacts")
    saved_parse = base_globals.get("parse_args")
    base_globals["evaluate_open_contacts"] = _evaluate_and_probe
    base_globals["parse_args"] = lambda: args
    try:
        try:
            open_demo.main()
        except _ProbeDone:
            pass
    finally:
        if saved_eval is not None:
            base_globals["evaluate_open_contacts"] = saved_eval
        if saved_parse is not None:
            base_globals["parse_args"] = saved_parse
    return probe_result


class _ProbeDone(Exception):
    pass


def _merge_overrides(
    autogen_args: argparse.Namespace, yaml_args: argparse.Namespace
) -> argparse.Namespace:
    """Merge YAML-supplied keys not already set on the autogen namespace.

    The autogen parser already inherits / overrides the demo-level defaults,
    so we only need to merge the YAML config fields the caller actually
    explicitly passed (i.e. flags not absorbed through the CLI poppy parser).
    """
    for key, value in vars(yaml_args).items():
        if not hasattr(autogen_args, key):
            setattr(autogen_args, key, value)
    return autogen_args


def _dispatch(args: argparse.Namespace, task_name: str) -> None:
    """Run any task pipeline by name."""
    autogen_print(f"dispatch=task task_name={task_name}")
    ctx = PipelineContext()
    stages = get_stages(task_name, args)
    started = time.perf_counter()
    try:
        run_pipeline(ctx, args, stages)
    finally:
        elapsed = time.perf_counter() - started
        autogen_print(f"pipeline_done task={task_name} elapsed_s={elapsed:.3f}")
        try:
            ctx.env.close()
        except Exception:
            pass


def _dispatch_close_drawer(args: argparse.Namespace) -> None:
    """Parse close-drawer args and run the push pipeline."""
    autogen_print("dispatch=close_drawer")
    from .core.args import parse_autogen_args
    from robocasa.demos.demo_close_drawer_contact_curobo import (
        parse_args as close_parse_args,
    )

    autogen_args = _merge_overrides(parse_autogen_args(close_parse_args), args)
    _dispatch(autogen_args, "close_drawer")


def _dispatch_open_drawer(args: argparse.Namespace) -> None:
    """Parse open-drawer args and run the grasp pipeline."""
    autogen_print("dispatch=open_drawer (grasp pipeline)")
    from .core.args import parse_open_autogen_args
    from robocasa.demos.demo_open_drawer_contact_curobo import (
        parse_args as open_parse_args,
    )

    autogen_args = _merge_overrides(parse_open_autogen_args(open_parse_args), args)
    _dispatch(autogen_args, "open_drawer")


def main() -> None:
    cli = _parse_cli()
    config = _load_yaml_config(cli.config)
    config.setdefault(
        "scene_cache_dir", str(REPO_ROOT / "outputs" / "scene_point_cache")
    )
    _apply_config_overrides(config, cli.overrides)
    task_name, drawer_action = _resolve_task(config, cli.task_name)
    config["drawer_action"] = drawer_action
    config.setdefault("task_name", task_name)

    # World-model probe knobs (with sensible defaults).
    config.setdefault("world_model_horizon_steps", 80)
    config.setdefault("world_model_sim_dt", 0.01)
    config.setdefault("world_model_applied_force", 30.0)
    config.setdefault("world_model_success_ratio", 0.3)
    config.setdefault("world_model_device", "cuda:0")

    args = argparse.Namespace(**config)
    autogen_print(f"task_name={task_name} drawer_action={drawer_action}")

    probe = _run_world_model_probe(args, task_name)
    use_prehens = int(probe.get("use_prehens", 0))
    autogen_print(
        f"world_model.use_prehens={use_prehens} "
        f"ratio={probe.get('ratio', float('nan')):.3f}"
    )

    if task_name == "close_drawer":
        _dispatch_close_drawer(args)
    elif task_name == "open_drawer":
        _dispatch_open_drawer(args)
    else:
        autogen_print(
            f"dispatch=no-op (task_name={task_name!r}, " f"use_prehens={use_prehens})"
        )


if __name__ == "__main__":
    main()
