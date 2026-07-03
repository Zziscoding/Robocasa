"""Skeleton pose solving: parallel DAQP over feasible contact candidates.

Port of the DAQP portion of
``demo_close_drawer_contact_curobo._solve_contact_poses_with_skeleton``:
build the EE skeleton, select interior feasible points, build
(candidate x primitive x finger) jobs, solve them in parallel with a
tqdm progress bar, then order the results by theta-diversity.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
from typing import Any

import numpy as np

from robocasa.demos import ee_skelton  # noqa: E402
from robocasa.demos.demo_close_drawer_contact_curobo import (  # noqa: E402
    _current_site_pose,
    _scene_geom_ids_for_skeleton,
    _select_close_panel_interior_feasible_points,
    _theta_diverse_skeleton_pose_order,
)

from .context import PipelineContext, autogen_print


def _solve_job(
    job: tuple,
    env: Any,
    skeleton: Any,
    object_eqs: Any,
    scene_geom_ids: Any,
    initial_rot: np.ndarray,
    args: Any,
    use_current_rotation: bool,
) -> tuple:
    """Solve a single (candidate, point, normal, primitive, finger) DAQP job."""
    candidate_index, point, normal, primitive, finger = job
    try:
        poses = ee_skelton.solve_skeleton_pose_candidates(
            env,
            skeleton,
            point,
            normal,
            finger=finger,
            contact_primitive=primitive,
            object_convex_equations=object_eqs,
            object_convex_equation_mask=None,
            scene_geom_ids=scene_geom_ids,
            initial_ee_rotation_world=initial_rot if use_current_rotation else None,
            args=args,
            max_candidates=max(
                1, int(getattr(args, "autogen_skeleton_pose_variants_per_contact", 4))
            ),
            min_theta_separation=float(
                getattr(args, "autogen_skeleton_pose_min_theta_separation", np.pi / 6.0)
            ),
        )
        return int(candidate_index), list(poses), None
    except Exception as exc:
        return int(candidate_index), [], f"{exc.__class__.__name__}:{exc}"


def solve_skeleton_poses_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Build skeleton poses for all feasible candidates via parallel DAQP.

    Populates ``ctx.skeleton_poses`` with ``(candidate_index, SkeletonPose)``
    tuples ordered by theta-diversity.
    """
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    feasible_cache = ctx.feasible_cache
    if feasible_cache is None:
        feasible_cache = getattr(args, "_autogen_feasible_cache", None)
    if feasible_cache is None:
        raise RuntimeError("feasible_cache is missing; run evaluate_contacts first.")

    skeleton = ee_skelton.build_panda_skeleton(ctx.env, frame_name)
    if bool(getattr(args, "autogen_visualize_skeleton", False)):
        ee_skelton.visualize_skeleton_and_ee(ctx.env, frame_name, skeleton, args)

    select_count = int(getattr(args, "autogen_initial_pose_count", 200))
    (
        local_ids,
        points_world,
        normals_world,
    ) = _select_close_panel_interior_feasible_points(
        feasible_cache,
        ctx.panel,
        select_count,
        int(args.seed),
        args,
    )
    feasible_candidate_indices = np.asarray(
        feasible_cache.candidate_indices, dtype=np.int64
    )
    scene_geom_ids = _scene_geom_ids_for_skeleton(ctx.env, ctx.panel)
    object_eqs = getattr(feasible_cache, "handle_convex_equations", None)
    initial_rot = _current_site_pose(ctx.env, frame_name)[1]

    def _collect(use_current_rotation: bool) -> list:
        variants_per_contact = max(
            1, int(getattr(args, "autogen_skeleton_pose_variants_per_contact", 4))
        )
        primitive_specs = (
            ("left_finger", "left"),
            ("right_finger", "right"),
            ("hand", "left"),
        )
        jobs = []
        for local_id, point, normal in zip(local_ids, points_world, normals_world):
            candidate_index = int(feasible_candidate_indices[int(local_id)])
            for primitive, finger in primitive_specs:
                jobs.append(
                    (
                        int(candidate_index),
                        np.asarray(point, dtype=np.float64).reshape(3).copy(),
                        np.asarray(normal, dtype=np.float64).reshape(3).copy(),
                        str(primitive),
                        str(finger),
                    )
                )
        workers = int(getattr(args, "autogen_mink_parallel_workers", 1) or 1)
        active_workers = max(1, min(workers, len(jobs) or 1))
        verbose_daqp = bool(getattr(args, "autogen_skeleton_daqp_verbose", True))
        if verbose_daqp:
            autogen_print(
                "SKELETON_POSE_SOLVER "
                f"backend=daqp workers={active_workers} "
                f"jobs={len(jobs)} "
                f"use_current_rotation={bool(use_current_rotation)}"
            )
        use_tqdm = (not verbose_daqp) and len(jobs) > 0
        pbar = None
        if use_tqdm:
            try:
                from tqdm import tqdm as _tqdm

                pbar = _tqdm(
                    total=len(jobs),
                    desc=f"daqp (workers={active_workers})",
                    unit="job",
                    file=sys.__stdout__,
                    dynamic_ncols=True,
                    miniters=1,
                    mininterval=0.2,
                    leave=True,
                )
            except Exception:
                pbar = None

        def _do_job(job):
            return _solve_job(
                job,
                ctx.env,
                skeleton,
                object_eqs,
                scene_geom_ids,
                initial_rot,
                args,
                use_current_rotation,
            )

        if workers <= 1 or len(jobs) <= 1:
            results = []
            for job in jobs:
                results.append(_do_job(job))
                if pbar is not None:
                    pbar.update(1)
        else:
            results_by_index = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=active_workers
            ) as executor:
                future_by_index = {
                    executor.submit(_do_job, job): idx for idx, job in enumerate(jobs)
                }
                for future in concurrent.futures.as_completed(future_by_index):
                    results_by_index[future_by_index[future]] = future.result()
                    if pbar is not None:
                        pbar.update(1)
            results = [results_by_index[idx] for idx in range(len(jobs))]
        if pbar is not None:
            pbar.close()

        collected = []
        failures: dict[str, int] = {}
        for candidate_index, poses, error in results:
            if error is not None:
                failures[error] = failures.get(error, 0) + 1
                continue
            for pose in poses:
                collected.append((candidate_index, pose))
        if failures and verbose_daqp:
            autogen_print(f"SKELETON_POSE_SOLVER_FAILURES {failures}")
        return collected

    started = time.perf_counter()
    skeleton_poses = _collect(
        bool(getattr(args, "autogen_use_current_ee_rotation", False))
    )
    allow_rotation_fallback = bool(
        getattr(args, "autogen_allow_current_rotation_fallback", True)
    )
    if (
        not skeleton_poses
        and not bool(getattr(args, "autogen_use_current_ee_rotation", False))
        and allow_rotation_fallback
    ):
        autogen_print(
            "close_autogen_patch=skeleton_rotation_fallback=current_ee_rotation"
        )
        skeleton_poses = _collect(True)
    skeleton_poses = _theta_diverse_skeleton_pose_order(skeleton_poses)
    elapsed = time.perf_counter() - started
    autogen_print(
        f"stage=skeleton_poses_done "
        f"count={len(skeleton_poses)} "
        f"elapsed_s={elapsed:.3f}"
    )
    ctx.skeleton_poses = skeleton_poses
    return ctx
