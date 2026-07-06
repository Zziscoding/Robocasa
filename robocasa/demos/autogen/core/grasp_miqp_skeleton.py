"""Contact-pair MIQP + dual-finger DAQP skeleton stage for open-drawer grasp.

This stage composes the three sub-steps the monolithic
``demo_open_drawer_autogen._solve_grasp_precontact_autogen`` would run BEFORE
the final pre-grasp solver:

  1. **MIQP** — rank up to ``grasp_num_pairs`` feasible contact pairs on the
     coACD handle mesh that was cached by ``evaluate_open_contacts_stage``.
     Imported from ``example_code.grasping.miqp_grasping.solve_grasping_contact_pairs``.
  2. **Skeleton solve** — run the dual-finger DAQP skeleton solver
     (``demo_open_drawer_autogen._solve_skeleton_grasp_all``) over every
     candidate contact pair × mirror angle. Returns
     ``[(pair_index, SkeletonPose, mirror_tag), ...]``.
  3. **Skeleton-popup** — optionally render the DAQP solutions as ghosts.

These are merged into one stage because they share the skeleton scene pool
(``args.autogen_skeleton_scene_pool``) which is expensive to build twice.

Populates ``ctx.grasp_skeleton_poses``.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

import numpy as np

from .context import PipelineContext, autogen_print


def miqp_and_skeleton_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Run the MIQP + dual-finger DAQP skeleton pipeline over the feasible cache."""
    from robocasa.demos import demo_open_drawer_autogen as open_autogen
    from robocasa.demos.demo_open_drawer_contact_curobo import (
        _drawer_joint_value,
    )

    feasible_cache = ctx.feasible_cache
    if feasible_cache is None:
        feasible_cache = getattr(args, "_autogen_feasible_cache", None)
    if feasible_cache is None:
        autogen_print("stage=miqp_and_skeleton_skipped reason=no feasible_cache")
        return ctx

    # --- 2. MIQP: contact pairs on the cached handle mesh ---
    mesh_path = getattr(feasible_cache, "_grasp_handle_mesh_path", None)
    if mesh_path is None:
        mesh_path = getattr(args, "_autogen_grasp_handle_mesh_path", None)
    obj_pos = getattr(feasible_cache, "_grasp_obj_pos", None)
    obj_quat = getattr(feasible_cache, "_grasp_obj_quat_wxyz", None)
    obj_scale = getattr(feasible_cache, "_grasp_obj_scale", None)

    started = time.perf_counter()
    grasp_pairs = []
    if mesh_path is not None and obj_pos is not None and obj_quat is not None:
        with contextlib.redirect_stdout(
            open(os.devnull, "w")
        ), contextlib.redirect_stderr(open(os.devnull, "w")):
            from robocasa.demos.example_code.grasping.miqp_grasping import (
                solve_grasping_contact_pairs,
            )

            try:
                grasp_pairs = solve_grasping_contact_pairs(
                    mesh_path=mesh_path,
                    obj_pos=np.asarray(obj_pos, dtype=np.float64),
                    obj_quat=np.asarray(obj_quat, dtype=np.float64),
                    obj_scale=np.asarray(
                        obj_scale if obj_scale is not None else np.ones(3),
                        dtype=np.float64,
                    ),
                    num_pairs=int(getattr(args, "grasp_num_pairs", 200)),
                    min_pairs=int(
                        getattr(
                            args,
                            "grasp_min_pairs",
                            16 if bool(getattr(args, "debug", False)) else 256,
                        )
                    ),
                    sample_budget=int(
                        getattr(
                            args,
                            "grasp_miqp_sample_budget",
                            32 if bool(getattr(args, "debug", False)) else 120,
                        )
                    ),
                    verbose=True,
                    debug=bool(getattr(args, "debug", False)),
                )
            except Exception as exc:
                autogen_print(f"stage=miqp_and_skeleton_miqp_failed error={exc!r}")
    else:
        autogen_print("stage=miqp_and_skeleton_skipped reason=missing mesh pose cache")

    autogen_print(
        f"stage=miqp_and_skeleton_miqp_done "
        f"pairs={len(grasp_pairs)} "
        f"elapsed_s={time.perf_counter() - started:.3f}"
    )
    ctx.miqp_pairs = list(grasp_pairs)

    # --- 3. DAQP skeleton solve (parallel over pairs × mirrors) ---
    if not grasp_pairs:
        autogen_print("stage=miqp_and_skeleton_no_pairs")
        ctx.grasp_skeleton_poses = []
        return ctx

    from robocasa.demos import ee_skelton

    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    skeleton = ee_skelton.build_panda_skeleton(ctx.env, frame_name)
    demo_ee_rotation = np.asarray(
        ctx.demonstration_seed.projected_ee_rotation_world, dtype=np.float64
    ).reshape(3, 3)
    handle_convex_equations = getattr(feasible_cache, "handle_convex_equations", None)
    if handle_convex_equations is None:
        handle_convex_equations = getattr(
            args, "_autogen_handle_convex_equations", None
        )

    # lazy-init the shared skeleton scene pool (also used by the pre-grasp solver)
    if getattr(args, "autogen_skeleton_scene_pool", None) is None:
        try:
            from robocasa.demos.skelton_scene import SkeletonScenePool

            pool_workers = min(
                max(1, int(getattr(args, "grasp_skeleton_max_workers", 8))),
                max(1, len(grasp_pairs) * 3),
            )
            args.autogen_skeleton_scene_pool = SkeletonScenePool.from_env(
                ctx.env, num_workers=pool_workers
            )
        except Exception as exc:
            autogen_print(
                f"stage=miqp_and_skeleton_scene_pool_unavailable error={exc!r}"
            )
            args.autogen_skeleton_scene_pool = None
    scene_pool = getattr(args, "autogen_skeleton_scene_pool", None)
    if scene_pool is not None:
        try:
            scene_pool.reset()
        except Exception:
            pass

    skel_started = time.perf_counter()
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(
        open(os.devnull, "w")
    ):
        ctx.grasp_skeleton_poses = open_autogen._solve_skeleton_grasp_all(
            ctx.env,
            skeleton,
            ctx.miqp_pairs,
            demo_ee_rotation,
            handle_convex_equations,
            tuple(
                gid for gid in open_autogen._iter_scene_geom_ids(ctx.env, ctx.surface)
            ),
            args,
        )
    autogen_print(
        f"stage=miqp_and_skeleton_done "
        f"count={len(ctx.grasp_skeleton_poses)} "
        f"elapsed_s={time.perf_counter() - skel_started:.3f}"
    )

    # --- 4. Optional skeleton-popup ---
    if bool(getattr(args, "autogen_visualize_grasp_skeleton_poses", False)):
        from .grasp_viz import _visualize_grasp_skeleton_poses_popup

        try:
            _visualize_grasp_skeleton_poses_popup(ctx, args)
        except Exception as exc:
            autogen_print(f"stage=miqp_and_skeleton_viz_error error={exc!r}")

    return ctx
