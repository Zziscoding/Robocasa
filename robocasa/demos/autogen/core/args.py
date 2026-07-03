"""Argument parsing for the autogen pipeline.

Parses the autogen-specific CLI flags (visualization popups, tqdm, DAQP
workers, MPPI knobs, etc.) and applies the same default overrides that
``demo_close_drawer_autogen._parse_close_autogen_args`` applies, so the
modular pipeline behaves identically to the reference script.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_autogen_args(original_parse_args) -> argparse.Namespace:
    """Parse autogen-specific flags, then call the underlying demo parser.

    Mirrors ``demo_close_drawer_autogen._parse_close_autogen_args`` so the
    modular pipeline sees the same defaults.
    """
    parser = argparse.ArgumentParser(add_help=False)
    # --- visualization popups ---
    parser.add_argument(
        "--autogen-visualize-mink-poses",
        dest="autogen_visualize_mink_poses",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-mink-poses",
        dest="autogen_visualize_mink_poses",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_mink_poses=False)
    parser.add_argument("--autogen-mink-ghost-alpha", type=float, default=0.28)
    parser.add_argument(
        "--autogen-mink-popup-camera-distance", type=float, default=0.85
    )
    parser.add_argument(
        "--autogen-mink-popup-camera-azimuth", type=float, default=135.0
    )
    parser.add_argument(
        "--autogen-mink-popup-camera-elevation", type=float, default=-25.0
    )
    parser.add_argument("--autogen-mink-popup-fps", type=float, default=30.0)

    parser.add_argument(
        "--autogen-visualize-skeleton",
        dest="autogen_visualize_skeleton",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-skeleton",
        dest="autogen_visualize_skeleton",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_skeleton=True)
    parser.add_argument(
        "--autogen-visualize-skeleton-poses",
        dest="autogen_visualize_skeleton_poses",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-skeleton-poses",
        dest="autogen_visualize_skeleton_poses",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_skeleton_poses=True)
    parser.add_argument(
        "--autogen-visualize-execution",
        dest="autogen_visualize_execution",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-execution",
        dest="autogen_visualize_execution",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_execution=True)

    parser.add_argument(
        "--autogen-physical-pd-visualization",
        dest="autogen_physical_pd_visualization",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-physical-pd-visualization",
        dest="autogen_physical_pd_visualization",
        action="store_false",
    )
    parser.set_defaults(autogen_physical_pd_visualization=True)
    parser.add_argument("--autogen-pd-kp", type=float, default=350.0)
    parser.add_argument("--autogen-pd-kd", type=float, default=35.0)
    parser.add_argument("--autogen-pd-target-dt", type=float, default=0.02)
    parser.add_argument("--autogen-pd-hold-seconds", type=float, default=1.0)

    parser.add_argument(
        "--autogen-visualize-floating-ee",
        dest="autogen_visualize_floating_ee",
        action="store_true",
    )
    parser.add_argument(
        "--no-autogen-visualize-floating-ee",
        dest="autogen_visualize_floating_ee",
        action="store_false",
    )
    parser.set_defaults(autogen_visualize_floating_ee=True)
    parser.add_argument("--autogen-visualize-floating-ee-limit", type=int, default=12)
    parser.add_argument("--autogen-floating-ee-ghost-alpha", type=float, default=0.32)

    # --- parallelism ---
    parser.add_argument("--autogen-mink-parallel-workers", type=int, default=None)
    parser.add_argument("--autogen-mink-q-checker-workers", type=int, default=None)
    parser.add_argument("--autogen-mink-q-worlds-per-worker", type=int, default=None)

    # --- step-2 solver ---
    parser.add_argument(
        "--solve_step2",
        "--solve-step2",
        dest="solve_step2",
        type=str,
        default=None,
        choices=("MPPI", "mink", "mppi", "MINK"),
    )

    popup_args, remaining = parser.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = original_parse_args()
    finally:
        sys.argv = old_argv

    for key, value in vars(popup_args).items():
        if value is None:
            continue
        setattr(args, key, value)

    # --- defaults that mirror demo_close_drawer_autogen ---
    if not any(str(arg).startswith("--mink-solver") for arg in old_argv[1:]):
        args.mink_solver = "daqp"

    user_set_mink_workers = any(
        str(arg) == "--autogen-mink-parallel-workers"
        or str(arg).startswith("--autogen-mink-parallel-workers=")
        for arg in old_argv[1:]
    )
    args.autogen_visualize_skeleton = False
    args.autogen_visualize_skeleton_poses = True
    args.autogen_use_current_ee_rotation = False
    args.autogen_allow_current_rotation_fallback = True
    if not hasattr(args, "autogen_skeleton_disable_handle_convex"):
        args.autogen_skeleton_disable_handle_convex = False
    args.autogen_step2_verbose_poses = False
    args.autogen_skeleton_daqp_verbose = False
    if "--autogen-mink-q-debug" not in old_argv:
        args.autogen_mink_q_debug = False
    args.autogen_skeleton_object_penetration_tol = min(
        float(getattr(args, "autogen_skeleton_object_penetration_tol", 0.001)),
        0.001,
    )
    args.autogen_skeleton_clearance_tolerance = min(
        float(getattr(args, "autogen_skeleton_clearance_tolerance", 0.001)),
        0.001,
    )
    args.autogen_qmppi_penetration_threshold = min(
        float(getattr(args, "autogen_qmppi_penetration_threshold", 0.005)),
        float(getattr(args, "contact_standoff", 0.005)),
        float(getattr(args, "mink_collision_penetration_tolerance", 0.02)),
        0.001,
    )
    args.mink_collision_penetration_tolerance = min(
        float(getattr(args, "mink_collision_penetration_tolerance", 0.02)),
        0.001,
    )
    args.autogen_qmppi_accept_object_improvement_only = True
    args.autogen_qmppi_penetration_weight = 500.0
    args.autogen_qmppi_contact_weight = 0.0
    if str(getattr(args, "solve_step2", "MPPI")).strip().lower() == "mppi":
        args.autogen_skip_mink_q_after_mppi = True
    default_mink_workers = max(1, (os.cpu_count() or 1))
    if not user_set_mink_workers:
        args.autogen_mink_parallel_workers = default_mink_workers
    if (
        str(getattr(args, "solve_step2", "MPPI")).strip().lower() == "mink"
        and not bool(getattr(args, "autogen_mink_q_debug", False))
        and not user_set_mink_workers
    ):
        args.autogen_mink_parallel_workers = max(2, default_mink_workers)
    args.curobo_use_mujoco_world = True
    args.curobo_world_exclude_target_drawer = True
    args.curobo_world_padding = min(
        float(getattr(args, "curobo_world_padding", 0.005)), 0.0
    )
    if bool(getattr(args, "autogen_visualize_execution", True)):
        args.visualize_contact = True
    args.save_output = False
    args.save_trajectory_videos = False
    if float(getattr(args, "contact_cost_threshold", 0.10)) == 0.10:
        args.contact_cost_threshold = 0.35
    if not hasattr(args, "autogen_panel_edge_margin"):
        args.autogen_panel_edge_margin = 0.015
    if float(getattr(args, "autogen_panel_edge_margin", 0.015)) in (0.015, 0.03):
        args.autogen_panel_edge_margin = 0.05
    if not hasattr(args, "autogen_panel_edge_fraction"):
        args.autogen_panel_edge_fraction = 0.28
    if float(getattr(args, "autogen_panel_edge_fraction", 0.18)) == 0.18:
        args.autogen_panel_edge_fraction = 0.28
    if not hasattr(args, "autogen_panel_top_edge_fraction"):
        args.autogen_panel_top_edge_fraction = 0.38
    if not hasattr(args, "autogen_skeleton_pose_variants_per_contact"):
        args.autogen_skeleton_pose_variants_per_contact = 8
    elif int(getattr(args, "autogen_skeleton_pose_variants_per_contact", 4)) == 4:
        args.autogen_skeleton_pose_variants_per_contact = 8
    if not hasattr(args, "autogen_skeleton_pose_min_theta_separation"):
        args.autogen_skeleton_pose_min_theta_separation = float(np.pi / 6.0)
    if not hasattr(args, "autogen_visualize_skeleton_pose_limit"):
        args.autogen_visualize_skeleton_pose_limit = 120
    return args
