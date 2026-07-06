"""Argument parsing for the autogen pipeline.

Parses the autogen-specific CLI flags (visualization popups, tqdm, DAQP
workers, MPPI knobs, etc.) and merges them into the underlying demo's
parsed args. Two demo-specific back-ends are supported:

* ``parse_autogen_args`` — close-drawer (``demo_close_drawer_contact_curobo``)
* ``parse_open_autogen_args`` — open-drawer / grasp
  (``demo_open_drawer_contact_curobo``)

Both share the ``_build_autogen_parser`` popup-flag scaffolding; only the
defaults block differs. Defaults are supplied via the ``defaults`` argument
so callers can inject demo-specific values without forking the file.
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


# ---------------------------------------------------------------------------
# Default-override blocks per demo. The close_drawer block mirrors
# ``demo_close_drawer_autogen._parse_close_autogen_args``; the open_drawer
# block is derived from ``demo_open_drawer_autogen.parse_args``.
# ---------------------------------------------------------------------------

_CLOSE_DRAWER_DEFAULTS: dict = {
    "mink_solver": "daqp",
    "contact_cost_threshold": 0.35,
    "autogen_panel_edge_margin": 0.05,
    "autogen_panel_edge_fraction": 0.28,
    "autogen_panel_top_edge_fraction": 0.38,
    "autogen_skeleton_pose_variants_per_contact": 8,
    "autogen_skeleton_pose_min_theta_separation": float(np.pi / 6.0),
    "autogen_visualize_skeleton_pose_limit": 120,
}

_OPEN_DRAWER_DEFAULTS: dict = {
    # --- open_drawer_contact_curobo base defaults ---
    "mink_arm_posture_cost": 0.02,
    "mink_locked_dof_cost": 200.0,
    "object_representative_point_count": 2048,
    "object_representative_min_per_geom": 16,
    "curobo_trajopt_tsteps": 32,
    "curobo_interpolation_dt": 0.02,
    "curobo_ik_seeds": 16,
    "curobo_graph_seeds": 2,
    "curobo_trajopt_seeds": 2,
    "curobo_max_attempts": 2,
    "curobo_enable_graph_attempt": 1,
    "disable_curobo_self_collision": False,
    "disable_curobo_cuda_graph": False,
    "curobo_world_padding": 0.005,
    "curobo_world_exclude_geoms": "",
    "curobo_world_exclude_bodies": "",
    "curobo_world_max_obstacles": None,
    "curobo_joint_enable_graph": False,
    "curobo_joint_enable_graph_attempt": None,
    "curobo_joint_disable_graph_attempt": None,
    "curobo_joint_max_attempts": 6,
    "curobo_joint_timeout": 5.0,
    "curobo_joint_retry_graph": True,
    "curobo_joint_graph_max_attempts": 2,
    "curobo_joint_graph_timeout": 8.0,
    "curobo_joint_enable_finetune_trajopt": False,
    "curobo_joint_check_start_validity": False,
    # --- autogen-specific defaults ---
    "autogen_object_point_count": 384,
    "autogen_handle_subdivide_max_edge": 0.005,
    "autogen_gripper_candidate_count": 50,
    "autogen_initial_pose_count": 200,
    "autogen_mink_max_attempts": 200,
    "autogen_mink_parallel_workers": max(1, (os.cpu_count() or 1)),
    "autogen_skeleton_parallel_workers": max(1, (os.cpu_count() or 1)),
    "autogen_coacd_threshold": 0.05,
    "autogen_coacd_max_convex_hull": 32,
    "autogen_coacd_preprocess_mode": "auto",
    "autogen_coacd_preprocess_resolution": 30,
    "autogen_coacd_resolution": 2000,
    "autogen_coacd_mcts_nodes": 20,
    "autogen_coacd_mcts_iterations": 100,
    "autogen_coacd_mcts_max_depth": 3,
    "autogen_coacd_max_ch_vertex": 256,
    "autogen_visualize_mink_poses": True,
    "autogen_mink_ghost_alpha": 0.28,
    "autogen_mink_popup_camera_distance": 0.85,
    "autogen_mink_popup_camera_azimuth": 135.0,
    "autogen_mink_popup_camera_elevation": -25.0,
    "autogen_mink_popup_fps": 30.0,
    "grasp_num_pairs": 200,
    "grasp_accept_threshold": 1.0,
    "grasp_rollout_steps": 15,
    "grasp_closed_opening": 0.005,
    "grasp_mppi_samples": 256,
    "grasp_mppi_iterations": 6,
    "grasp_precontact_solver": "mink",  # "mink" | "mppi" | "auto"
    "grasp_skeleton_parallel": True,
    "grasp_skeleton_max_workers": 8,
    # --- dual-finger DAQP skeleton solver ---
    "autogen_dual_theta_count": 6,
    "autogen_dual_n_lift": 1,
    "autogen_dual_n_g": 1,
    "autogen_dual_seed": 7,
    "autogen_dual_max_candidates": 0,
    "autogen_dual_min_theta_separation": 0.0,
    "autogen_dual_debug": False,
    "autogen_visualize_contact_pairs": True,
    "autogen_contact_pair_radius": 0.004,
    "autogen_dual_object_penetration_tol": 0.005,
    "autogen_dual_object_margin": 0.0,
    "autogen_grasp_max_skeleton_poses": 512,
    "autogen_visualize_grasp_skeleton_poses": True,
    "autogen_visualize_grasp_precontact": True,
    "autogen_visualize_mink_poses": True,
    "autogen_visualize_skeleton_poses": True,
    "autogen_visualize_skeleton_preview": False,
    "debug": False,
}

# Debug-mode overrides (applied unless the user explicitly set them).
_OPEN_DRAWER_DEBUG_OVERRIDES: dict = {
    "autogen_dual_theta_count": 2,
    "autogen_dual_n_lift": 1,
    "autogen_dual_n_g": 1,
    "autogen_dual_debug": True,
    "autogen_skeleton_segment_samples": 2,
    "autogen_skeleton_n_random": 1,
    "autogen_skeleton_theta_count": 2,
    "autogen_visualize_mink_poses": False,
    "autogen_visualize_skeleton_poses": False,
    "autogen_visualize_skeleton_preview": False,
    "autogen_visualize_contact_pairs": False,
    "autogen_visualize_grasp_skeleton_poses": False,
    "grasp_num_pairs": 32,
}


def _build_autogen_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all autogen-specific visualization flags.

    Shared by both close-drawer and open-drawer parsers — the underlying
    demo's arguments are added by the caller-provided ``original_parse_args``
    so callers can keep using the demo's full parser.
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
    return parser


def _apply_close_drawer_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    """Apply the close-drawer autogen defaults to the parsed args.

    Mutates ``args`` in place. Mirrors the behavior of
    ``demo_close_drawer_autogen._parse_close_autogen_args``.
    """
    for key, value in _CLOSE_DRAWER_DEFAULTS.items():
        if not hasattr(args, key) or getattr(args, key) in (None,):
            setattr(args, key, value)
    # contact_cost_threshold: only bump the demo default, not user-overridden values
    if float(getattr(args, "contact_cost_threshold", 0.10)) == 0.10:
        args.contact_cost_threshold = 0.35

    args.autogen_visualize_skeleton = False
    args.autogen_visualize_skeleton_poses = True
    args.autogen_use_current_ee_rotation = False
    args.autogen_allow_current_rotation_fallback = True
    if not hasattr(args, "autogen_skeleton_disable_handle_convex"):
        args.autogen_skeleton_disable_handle_convex = False
    args.autogen_step2_verbose_poses = False
    args.autogen_skeleton_daqp_verbose = False
    if "--autogen-mink-q-debug" not in argv:
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

    user_set_mink_workers = any(
        str(arg) == "--autogen-mink-parallel-workers"
        or str(arg).startswith("--autogen-mink-parallel-workers=")
        for arg in argv
    )
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
        args.autogen_skeleton_pose_variants_per_contact = 4
    elif int(getattr(args, "autogen_skeleton_pose_variants_per_contact", 4)) == 4:
        args.autogen_skeleton_pose_variants_per_contact = 8
    if not hasattr(args, "autogen_skeleton_pose_min_theta_separation"):
        args.autogen_skeleton_pose_min_theta_separation = float(np.pi / 6.0)
    if not hasattr(args, "autogen_visualize_skeleton_pose_limit"):
        args.autogen_visualize_skeleton_pose_limit = 120


def _apply_open_drawer_defaults(
    args: argparse.Namespace,
    argv: list[str],
    precontact_distance: float,
) -> None:
    """Apply the open-drawer (grasp) autogen defaults.

    Mutates ``args`` in place. Derived from
    ``demo_open_drawer_autogen.parse_args``.
    """
    for key, value in _OPEN_DRAWER_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    # autogen_precontact_lift falls back to YAML precontact_distance when set
    args.autogen_precontact_lift = precontact_distance

    parser = _build_autogen_parser()
    parser.set_defaults(
        autogen_visualize_mink_poses=True,
        autogen_visualize_skeleton_poses=True,
        autogen_visualize_skeleton_preview=False,
        autogen_visualize_contact_pairs=True,
        autogen_visualize_grasp_skeleton_poses=True,
    )
    cli_defaults = vars(parser.parse_args([]))
    _cli_override_keys = {key.split("=", 1)[0].replace("-", "_") for key in argv}
    # Merge popup defaults only for flags the user did NOT explicitly set
    for key, value in cli_defaults.items():
        if key in _cli_override_keys:
            continue
        setattr(args, key, value)

    args.save_output = False
    args.save_trajectory_videos = False


def parse_autogen_args(original_parse_args) -> argparse.Namespace:
    """Parse autogen-specific flags, then call the underlying close-drawer demo parser.

    Mirrors ``demo_close_drawer_autogen._parse_close_autogen_args`` so the
    modular pipeline behaves identically to the reference script.
    """
    parser = _build_autogen_parser()
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

    _apply_close_drawer_defaults(args, old_argv[1:])
    return args


def parse_open_autogen_args(original_parse_args) -> argparse.Namespace:
    """Parse autogen-specific flags, then call the underlying open-drawer demo parser.

    Injects the autogen visualization defaults and the grasp-specific knobs
    the underlying ``demo_open_drawer_contact_curobo`` parser does not set,
    so the modular pipeline can drive the open-drawer / grasp flow without
    going through ``demo_open_drawer_autogen.parse_args``.
    """
    parser = _build_autogen_parser()
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

    precontact_distance = float(getattr(args, "precontact_distance", 0.04))
    _apply_open_drawer_defaults(args, old_argv[1:], precontact_distance)

    if bool(getattr(args, "debug", False)):
        _cli_override_keys = {
            key.split("=", 1)[0].replace("-", "_") for key in old_argv[1:]
        }
        for key, value in _OPEN_DRAWER_DEBUG_OVERRIDES.items():
            if key not in _cli_override_keys:
                setattr(args, key, value)

    return args
