"""Unified autogen entry point.

Reads a YAML config and dispatches to the right manipulation pipeline:

1. Load YAML (defaults to ``open_drawer_contact_curobo.yaml``). The YAML's
   ``task_name`` field (e.g. ``open_drawer`` / ``close_drawer``) and/or
   ``drawer_action`` field selects scene + action.
2. Build feasible contact candidates (delegated to the open-drawer autogen
   pipeline, which is the generic COACD-based candidate generator).
3. Run :func:`world_model.evaluate_prehensile_need` on those candidates.
4. If ``use_prehens == 0`` → invoke ``demo_close_drawer_autogen.main`` (the
   nonprehensile push pipeline). If ``use_prehens == 1`` → prehensile pipeline
   placeholder (TODO).
"""

from __future__ import annotations

import argparse
import os
import sys
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


_DEFAULT_YAML = Path(__file__).with_name("open_drawer_contact_curobo.yaml")


def _stdout_print(message: str) -> None:
    print(f"[autogen] {message}", file=sys.__stdout__, flush=True)


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
        # Short-circuit: raise a sentinel to stop the open-drawer pipeline
        # cleanly after probing.
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


def _dispatch_nonprehensile() -> None:
    from robocasa.demos import demo_close_drawer_autogen

    _stdout_print("dispatch=nonprehensile (close_drawer_autogen)")
    demo_close_drawer_autogen.main()


def _dispatch_prehensile() -> None:
    _stdout_print("dispatch=prehensile (TODO: implement)")
    # Intentionally left as a stub per spec.


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
    _stdout_print(f"task_name={task_name} drawer_action={drawer_action}")

    probe = _run_world_model_probe(args, task_name)
    use_prehens = int(probe.get("use_prehens", 0))
    _stdout_print(
        f"world_model.use_prehens={use_prehens} "
        f"ratio={probe.get('ratio', float('nan')):.3f}"
    )

    if use_prehens == 0:
        _dispatch_nonprehensile()
    else:
        _dispatch_prehensile()


if __name__ == "__main__":
    main()
