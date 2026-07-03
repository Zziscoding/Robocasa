"""Unified autogen entry point.

Reads a YAML config and dispatches to the right manipulation pipeline:

1. Load YAML (defaults to ``open_drawer_contact_curobo.yaml``). The YAML's
   ``task_name`` field (e.g. ``open_drawer`` / ``close_drawer``) and/or
   ``drawer_action`` field selects scene + action.
2. Run the world-model prehensile-need probe.
3. If ``use_prehens == 0`` → invoke the modular close-drawer pipeline
   (COACD → feasible candidates → DAQP skeleton poses → MPPI/mink step-2 →
   rollout filtering → cuRobo trajectory, with 3 popup visualizations).
   If ``use_prehens == 1`` → prehensile pipeline placeholder (TODO).
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


def _dispatch_nonprehensile(args: argparse.Namespace) -> None:
    """Run the modular close-drawer pipeline."""
    autogen_print("dispatch=nonprehensile (modular close_drawer pipeline)")
    from .core.args import parse_autogen_args
    from robocasa.demos.demo_close_drawer_contact_curobo import (
        parse_args as close_parse_args,
    )

    # Parse args with autogen-specific defaults.
    autogen_args = parse_autogen_args(close_parse_args)

    # Merge any YAML-supplied overrides.
    for key, value in vars(args).items():
        if not hasattr(autogen_args, key):
            setattr(autogen_args, key, value)

    ctx = PipelineContext()
    stages = get_stages("close_drawer", autogen_args)
    started = time.perf_counter()
    try:
        run_pipeline(ctx, autogen_args, stages)
    finally:
        elapsed = time.perf_counter() - started
        autogen_print(f"pipeline_done elapsed_s={elapsed:.3f}")
        try:
            ctx.env.close()
        except Exception:
            pass


def _dispatch_prehensile() -> None:
    autogen_print("dispatch=prehensile (TODO: implement)")


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

    if use_prehens == 0:
        _dispatch_nonprehensile(args)
    else:
        _dispatch_prehensile()


if __name__ == "__main__":
    main()
