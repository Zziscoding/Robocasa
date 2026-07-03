"""Step-2 solver: refine skeleton poses into feasible arm-q solutions.

Two backends (selected by ``args.solve_step2``):

* **MPPI** — ``FloatingEEMPPI`` refines the EE pose in a stripped scene,
  ``FloatingEERollout`` checks object-cost improvement, then a final
  mink IK recovers the arm-q.
* **mink** — ``mink_q.solve_skeleton_precontact_q_parallel`` (or the batched
  variant) solves IK + retreat directly, then ``FloatingEERollout`` filters.

The implementation mirrors the step-2 loop in
``demo_close_drawer_contact_curobo._solve_contact_poses_with_skeleton``.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import mujoco
import numpy as np

from robocasa.demos import ee_skelton, mink_q  # noqa: E402
from robocasa.demos.demo_close_drawer_contact_curobo import (  # noqa: E402
    _arm_q_from_robot_model_q,
    _check_arm_q_collision,
    _current_robot_model_q,
    _current_site_pose,
    _drawer_joint_value,
    _make_mink_posture_cost,
    _normalize,
    _set_drawer_joint_value,
    _skeleton_solution_to_contact_solution,
    _solve_mink_frame_pose,
    _strict_robot_drawer_penetration,
)
from robocasa.demos.ee_floating_mppi import (
    FloatingEEConfig,
    FloatingEEMPPI,
)  # noqa: E402
from robocasa.demos.rollout import FloatingEERollout, RolloutConfig  # noqa: E402

from .context import PipelineContext, autogen_print


def _candidate_force_direction_world(
    candidate_index: int,
    candidate: Any,
    panel: Any,
    feasible_cache: Any,
    feasible_row_by_candidate: dict,
) -> np.ndarray:
    """Compute the approach force world for a candidate (mirrors close_demo)."""
    fallback = _normalize(panel.push_world, fallback=[1.0, 0.0, 0.0])
    row = feasible_row_by_candidate.get(int(candidate_index))
    if row is None:
        return fallback
    try:
        lam = np.asarray(
            getattr(candidate, "lam", np.zeros(3)), dtype=np.float64
        ).reshape(-1)
        if lam.size < 3:
            return fallback
        normal = np.asarray(
            feasible_cache.normals_world[row], dtype=np.float64
        ).reshape(3)
        t1 = np.asarray(feasible_cache.tangents1_world[row], dtype=np.float64).reshape(
            3
        )
        t2 = np.asarray(feasible_cache.tangents2_world[row], dtype=np.float64).reshape(
            3
        )
        direction = lam[0] * normal + lam[1] * t1 + lam[2] * t2
        if float(np.dot(direction, panel.push_world)) < 0.0:
            direction = -direction
        return _normalize(direction, fallback=fallback)
    except Exception:
        return fallback


def _build_floating_mppi(ctx: PipelineContext, args: Any) -> FloatingEEMPPI | None:
    """Construct the FloatingEEMPPI solver for the close-drawer task."""
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    _drawer = getattr(ctx.env, "drawer", None)
    _drawer_body_id = -1
    _drawer_slide_axis_world = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    _drawer_qpos_addr = None
    if _drawer is not None and len(_drawer.door_joint_names) > 0:
        try:
            _dj = _drawer.door_joint_names[0]
            _drawer_qpos_addr = int(ctx.env.sim.model.get_joint_qpos_addr(_dj))
            _jid = ctx.env.sim.model.joint_name2id(_dj)
            _drawer_slide_axis_world = np.asarray(
                ctx.env.sim.model.jnt_axis[_jid], dtype=np.float64
            ).reshape(3)
            _drawer_body_id = int(ctx.env.sim.model.jnt_bodyid[_jid])
        except Exception:
            _drawer_body_id = -1
    drawer_q_now = float(_drawer_joint_value(ctx.env))
    _drawer_q_target = min(float(drawer_q_now) + float(ctx.push_distance), 0.0)
    _current_drawer_body_pos = (
        np.asarray(ctx.env.sim.data.body_xpos[_drawer_body_id], dtype=np.float64)
        if _drawer_body_id >= 0
        else np.zeros(3)
    )
    _target_object_pos = (
        _current_drawer_body_pos
        + (_drawer_q_target - float(drawer_q_now)) * _drawer_slide_axis_world
    )
    _approach_world = np.asarray(ctx.panel.push_world, dtype=np.float64).reshape(3)
    try:
        floating_mppi = FloatingEEMPPI(
            ctx.env,
            hand_xml_path=None,
            finger_geom_names=(),
            object_body_id=int(_drawer_body_id),
            ee_site_name=frame_name,
            config=FloatingEEConfig(
                seed=int(getattr(args, "autogen_qmppi_seed", args.seed)),
                num_samples=int(getattr(args, "autogen_qmppi_num_samples", 256)),
                max_num_iterations=int(
                    getattr(args, "autogen_qmppi_num_iterations", 6)
                ),
                elite_ratio=float(getattr(args, "autogen_qmppi_elite_ratio", 0.1)),
                temperature=float(getattr(args, "autogen_qmppi_temperature", 1.0)),
                gripper_noise_scale=float(
                    getattr(args, "autogen_qmppi_gripper_noise_scale", 0.005)
                ),
                gripper_min=float(getattr(args, "autogen_qmppi_gripper_min", 0.0)),
                gripper_max=float(getattr(args, "autogen_qmppi_gripper_max", 0.08)),
                pen_threshold=float(
                    getattr(
                        args,
                        "autogen_qmppi_penetration_threshold",
                        min(
                            float(getattr(args, "contact_standoff", 0.005)),
                            float(
                                getattr(
                                    args,
                                    "mink_collision_penetration_tolerance",
                                    0.02,
                                )
                            ),
                        ),
                    )
                ),
                contact_tolerance=float(
                    getattr(args, "autogen_qmppi_contact_tolerance", 0.002)
                ),
                pen_weight=float(
                    getattr(args, "autogen_qmppi_penetration_weight", 500.0)
                ),
                contact_weight=float(
                    getattr(args, "autogen_qmppi_contact_weight", 0.0)
                ),
                track_pos_weight=float(
                    getattr(args, "autogen_qmppi_pos_weight", 200.0)
                ),
                track_rot_weight=float(getattr(args, "autogen_qmppi_rot_weight", 20.0)),
                track_gripper_weight=float(
                    getattr(args, "autogen_qmppi_gripper_tracking_weight", 50.0)
                ),
                accept_object_improvement_only=bool(
                    getattr(args, "autogen_qmppi_accept_object_improvement_only", True)
                ),
            ),
            approach_world=_approach_world,
            target_object_position=_target_object_pos,
            drawer_qpos_addr=_drawer_qpos_addr,
            drawer_qpos_value=float(drawer_q_now),
        )
        return floating_mppi
    except Exception as exc:
        sys.stderr.write(
            f"[autogen] FloatingEEMPPI unavailable ({exc!r}); "
            "step-2 MPPI will be skipped.\n"
        )
        sys.stderr.flush()
        return None


def _get_rollout(
    ctx: PipelineContext,
    args: Any,
    force_direction: np.ndarray,
    rollouts_by_direction: dict,
    target_object_pos: np.ndarray,
) -> FloatingEERollout | None:
    """Lazily build / cache a FloatingEERollout for a given approach direction."""
    key = tuple(
        np.round(np.asarray(force_direction, dtype=np.float64).reshape(3), 6).tolist()
    )
    cached = rollouts_by_direction.get(key)
    if cached is not None:
        return cached
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    _drawer = getattr(ctx.env, "drawer", None)
    _drawer_body_id = -1
    if _drawer is not None and len(_drawer.door_joint_names) > 0:
        try:
            _dj = _drawer.door_joint_names[0]
            _jid = ctx.env.sim.model.joint_name2id(_dj)
            _drawer_body_id = int(ctx.env.sim.model.jnt_bodyid[_jid])
        except Exception:
            _drawer_body_id = -1
    try:
        r = FloatingEERollout(
            ctx.env,
            hand_xml_path=None,
            finger_geom_names=(),
            object_body_id=int(_drawer_body_id),
            ee_site_name=frame_name,
            config=RolloutConfig(
                sim_dt=float(getattr(args, "autogen_qmppi_sim_dt", 0.05)),
                horizon_steps=int(getattr(args, "autogen_qmppi_horizon_steps", 30)),
                approach_total_distance=float(
                    getattr(
                        args,
                        "autogen_qmppi_approach_total_distance",
                        float(ctx.push_distance),
                    )
                ),
                object_improvement_eps=float(
                    getattr(args, "autogen_qmppi_object_improvement_eps", 1e-5)
                ),
                score_accept_threshold=float(
                    getattr(args, "autogen_qmppi_score_accept_threshold", 0.15)
                ),
            ),
            approach_world=_normalize(force_direction, fallback=ctx.panel.push_world),
            target_object_position=target_object_pos,
        )
        rollouts_by_direction[key] = r
        return r
    except Exception as exc:
        sys.stderr.write(
            f"[autogen] FloatingEERollout unavailable ({exc!r}); "
            "object-improvement gating disabled for this direction.\n"
        )
        sys.stderr.flush()
        rollouts_by_direction[key] = None
        return None


def solve_step2_stage(ctx: PipelineContext, args: Any) -> PipelineContext:
    """Run the step-2 solver over all skeleton poses.

    Populates ``ctx.solutions``, ``ctx.reports``, ``ctx.refined_poses``.
    """
    frame_name = str(
        getattr(args, "mink_contact_frame", "gripper0_right_grip_site")
    ).split(":")[0]
    robot_model = ctx.env.robots[0].robot_model.mujoco_model
    arm_joint_names = tuple(ctx.robot_state["robocasa_joint_names"])
    q_robot_start = _current_robot_model_q(ctx.env, robot_model)
    q_posture = q_robot_start.copy()
    posture_cost = _make_mink_posture_cost(robot_model, arm_joint_names, args)
    solve_step2 = str(getattr(args, "solve_step2", "MPPI")).strip().lower()
    if solve_step2 not in ("mppi", "mink"):
        raise ValueError(
            f"Unsupported solve_step2={getattr(args, 'solve_step2', None)!r}; "
            "expected MPPI or mink"
        )

    feasible_cache = ctx.feasible_cache
    feasible_row_by_candidate = {}
    if feasible_cache is not None:
        feasible_candidate_indices = np.asarray(
            feasible_cache.candidate_indices, dtype=np.int64
        )
        feasible_row_by_candidate = {
            int(ci): int(ri) for ri, ci in enumerate(feasible_candidate_indices)
        }

    # Shuffle order for fairness.
    rng = np.random.default_rng(int(args.seed) + 29003)
    order = np.arange(len(ctx.skeleton_poses), dtype=np.int64)
    rng.shuffle(order)
    if solve_step2 == "mink":
        max_attempts = len(order)
    else:
        max_attempts = min(
            int(getattr(args, "autogen_mink_max_attempts", len(order))),
            len(order),
        )

    pos_tol = float(
        getattr(args, "autogen_accept_position_tolerance", args.mink_position_tolerance)
    )
    pen_tol = float(args.mink_collision_penetration_tolerance)

    # --- FloatingEEMPPI setup ---
    floating_mppi = None
    if solve_step2 == "mppi":
        floating_mppi = _build_floating_mppi(ctx, args)

    # --- Rollout cache ---
    _drawer = getattr(ctx.env, "drawer", None)
    _drawer_body_id = -1
    _drawer_slide_axis_world = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    if _drawer is not None and len(_drawer.door_joint_names) > 0:
        try:
            _dj = _drawer.door_joint_names[0]
            _jid = ctx.env.sim.model.joint_name2id(_dj)
            _drawer_slide_axis_world = np.asarray(
                ctx.env.sim.model.jnt_axis[_jid], dtype=np.float64
            ).reshape(3)
            _drawer_body_id = int(ctx.env.sim.model.jnt_bodyid[_jid])
        except Exception:
            _drawer_body_id = -1
    drawer_q_now = float(_drawer_joint_value(ctx.env))
    _drawer_q_target = min(float(drawer_q_now) + float(ctx.push_distance), 0.0)
    _current_drawer_body_pos = (
        np.asarray(ctx.env.sim.data.body_xpos[_drawer_body_id], dtype=np.float64)
        if _drawer_body_id >= 0
        else np.zeros(3)
    )
    _target_object_pos = (
        _current_drawer_body_pos
        + (_drawer_q_target - float(drawer_q_now)) * _drawer_slide_axis_world
    )
    rollouts_by_direction: dict[tuple, FloatingEERollout | None] = {}

    # --- mink-q full-scene checker ---
    _mink_q_checker = None
    if solve_step2 == "mink" and max_attempts > 0:
        try:
            from robocasa.demos.full_scene_mjwarp import FullSceneCollisionCheckerPool

            _mink_q_requested_workers = max(
                int(getattr(args, "autogen_mink_parallel_workers", 1) or 1),
                1,
            )
            _checker_worker_arg = getattr(args, "autogen_mink_q_checker_workers", None)
            if _checker_worker_arg is None:
                _mink_q_workers = min(_mink_q_requested_workers, max_attempts, 8)
            else:
                _mink_q_workers = min(
                    max(int(_checker_worker_arg), 1),
                    _mink_q_requested_workers,
                    max_attempts,
                )
            _mink_q_workers = max(int(_mink_q_workers), 1)
            _mink_q_worlds_per_worker = int(
                getattr(args, "autogen_mink_q_worlds_per_worker", 1) or 1
            )
            _mink_q_scene_pen_tol = min(
                float(getattr(args, "mink_collision_penetration_tolerance", 0.0)),
                1e-4,
            )
            _mink_q_checker = FullSceneCollisionCheckerPool.from_env(
                ctx.env,
                arm_joint_names=arm_joint_names,
                frame_name=frame_name,
                panel=ctx.panel,
                num_workers=_mink_q_workers,
                allowed_ee_geom_name=None,
                penetration_tolerance=_mink_q_scene_pen_tol,
                device=getattr(args, "autogen_mink_q_mjwarp_device", "cuda:0"),
                nconmax_per_env=int(
                    getattr(args, "autogen_mink_q_mjwarp_nconmax", 256)
                ),
                njmax_per_env=int(getattr(args, "autogen_mink_q_mjwarp_njmax", 1024)),
                prefer_comfree=bool(
                    getattr(args, "autogen_mink_q_mjwarp_comfree", True)
                ),
                nworld_per_worker=_mink_q_worlds_per_worker,
            )
            try:
                _mink_q_checker.warmup()
            except Exception:
                pass
        except Exception as exc:
            sys.stderr.write(
                f"[autogen] FullSceneMjWarp checker unavailable ({exc!r}); "
                "falling back to env.sim-backed checks.\n"
            )
            sys.stderr.flush()
            _mink_q_checker = None

    # --- tqdm bar ---
    _step2_verbose_poses = bool(getattr(args, "autogen_step2_verbose_poses", True))
    _tqdm_pbar = None
    if not _step2_verbose_poses:
        step2_desc = "mink" if solve_step2 == "mink" else f"step2[{solve_step2}]"
        try:
            from tqdm import tqdm as _tqdm

            _tqdm_pbar = _tqdm(
                order[:max_attempts],
                desc=step2_desc,
                unit="skel",
                file=sys.__stdout__,
                dynamic_ncols=True,
                miniters=1,
                mininterval=0.2,
                leave=True,
            )
        except Exception:
            _tqdm_pbar = None

    yellow = "\033[93m"
    reset_color = "\033[0m"
    solutions: list = []
    reports: list = []
    refined_poses: list = []
    successful_floating_ee_count = 0
    rejected_floating_ee_count = 0
    mink_q_accepted_count = 0
    mink_q_rejected_count = 0
    pending_mink_q_solutions: list = []

    def _pbar_post(tag: str, candidate_index: int, **kw) -> None:
        if _tqdm_pbar is None:
            return
        now_s = time.perf_counter()
        acc = (
            mink_q_accepted_count
            if solve_step2 == "mink"
            else successful_floating_ee_count
        )
        rej = (
            mink_q_rejected_count
            if solve_step2 == "mink"
            else rejected_floating_ee_count
        )
        parts = [f"cand={int(candidate_index)}", tag]
        for k, v in kw.items():
            if v is None:
                continue
            try:
                parts.append(f"{k}={float(v):.4f}")
            except (TypeError, ValueError):
                parts.append(f"{k}={v}")
        parts.append(f"acc={acc}")
        parts.append(f"rej={rej}")
        _tqdm_pbar.set_postfix_str(" ".join(parts))

    autogen_print(
        f"STEP2_SOLVER "
        f"{'MPPI=active' if solve_step2 == 'mppi' else 'MPPI=inactive'} "
        f"{'mink=active' if solve_step2 == 'mink' else 'mink=inactive'} "
        f"mink_solver={getattr(args, 'mink_solver', None)}"
    )

    started = time.perf_counter()
    for attempt_id, pose_index in enumerate(order[:max_attempts]):
        candidate_index, skeleton_pose = ctx.skeleton_poses[int(pose_index)]
        candidate = ctx.candidates[int(candidate_index)]
        feasible_row = feasible_row_by_candidate.get(int(candidate_index))
        try:
            _skel_pos = np.asarray(skeleton_pose.ee_position, dtype=np.float64)
            _skel_rot = np.asarray(skeleton_pose.ee_rotation, dtype=np.float64)
            _skel_g = float(
                getattr(
                    skeleton_pose,
                    "gripper_opening",
                    ee_skelton.PANDA_DEFAULT_GRIPPER_OPENING,
                )
            )
            _accept_improvement_only = bool(
                getattr(args, "autogen_qmppi_accept_object_improvement_only", True)
            )
            _score_thresh = float(
                getattr(args, "autogen_qmppi_score_accept_threshold", 0.15)
            )
            force_direction = _candidate_force_direction_world(
                candidate_index,
                candidate,
                ctx.panel,
                feasible_cache,
                feasible_row_by_candidate,
            )

            if solve_step2 == "mppi" and floating_mppi is not None:
                _skel_quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(_skel_quat, _skel_rot.reshape(9))
                selected_point = (
                    np.asarray(
                        feasible_cache.positions_world[int(feasible_row)],
                        dtype=np.float64,
                    )
                    if feasible_row is not None
                    else np.asarray(candidate.world_point, dtype=np.float64)
                )
                floating_mppi.selected_contact_point_world = selected_point.reshape(3)
                floating_mppi.approach_world = force_direction.reshape(3)
                _result = floating_mppi.solve(_skel_pos, _skel_quat, _skel_g)
                refined_pos = np.asarray(_result.ee_position, dtype=np.float64)
                refined_rot = np.asarray(_result.ee_rotation, dtype=np.float64)
                refined_g = float(_result.gripper_opening)
                max_pen = float(_result.pen_cost)

                _rollout = _get_rollout(
                    ctx,
                    args,
                    force_direction,
                    rollouts_by_direction,
                    _target_object_pos,
                )
                _obj_delta = 0.0
                _score_norm = float("nan")
                if _rollout is not None:
                    _rollout_res = _rollout.run(refined_pos, refined_rot, refined_g)
                    _obj_delta = float(_rollout_res.object_cost_delta)
                    _score_norm = float(_rollout_res.score_normalized)
                    obj_improved = bool(
                        np.isfinite(_score_norm) and _score_norm > _score_thresh
                    )
                else:
                    obj_improved = False
                final_accepted = (
                    obj_improved
                    if _accept_improvement_only
                    else bool(_result.accepted) and obj_improved
                )
                if not final_accepted:
                    rejected_floating_ee_count += 1
                    _pbar_post(
                        "MPPI_REJECT",
                        candidate_index,
                        pen=_result.pen_cost,
                        obj=_obj_delta,
                    )
                    if _tqdm_pbar is not None:
                        _tqdm_pbar.update(1)
                    continue
                refined_poses.append(
                    (
                        refined_pos.copy(),
                        refined_rot.copy(),
                        refined_g,
                        int(candidate_index),
                    )
                )
                successful_floating_ee_count += 1
                # Recover arm-q via mink IK.
                q_robot_refined, pos_err_val = _solve_mink_frame_pose(
                    ctx.env,
                    frame_name,
                    refined_pos,
                    refined_rot,
                    q_robot_start,
                    q_posture,
                    posture_cost,
                    args,
                )
                q_best = _arm_q_from_robot_model_q(
                    robot_model, q_robot_refined, arm_joint_names
                )
                g_best = refined_g
                pos_err = float(pos_err_val)
                rot_err = 0.0
                actual_pen, _ = _strict_robot_drawer_penetration(
                    ctx.env, arm_joint_names, q_best
                )
                max_pen = max(float(max_pen), float(actual_pen))
                if _tqdm_pbar is not None:
                    _tqdm_pbar.update(1)
            elif solve_step2 == "mppi":
                rejected_floating_ee_count += 1
                if _tqdm_pbar is not None:
                    _tqdm_pbar.update(1)
                continue
            else:
                # mink path
                retreat_normal = np.asarray(
                    getattr(
                        skeleton_pose,
                        "contact_normal_world",
                        ctx.panel.outward_world,
                    ),
                    dtype=np.float64,
                ).reshape(3)
                if float(np.dot(retreat_normal, ctx.panel.outward_world)) < 0.0:
                    retreat_normal = -retreat_normal
                result = mink_q.solve_skeleton_precontact_q_parallel(
                    ctx.env,
                    robot_model=robot_model,
                    arm_joint_names=arm_joint_names,
                    frame_name=frame_name,
                    skeleton_pose=skeleton_pose,
                    q_start=q_robot_start,
                    q_posture=q_posture,
                    posture_cost=posture_cost,
                    args=args,
                    retreat_direction_world=retreat_normal,
                    penetration_checker=lambda q_arm: _strict_robot_drawer_penetration(
                        ctx.env, arm_joint_names, q_arm
                    ),
                    scene_collision_checker=lambda q_arm: _check_arm_q_collision(
                        ctx.env,
                        ctx.panel,
                        arm_joint_names,
                        q_arm,
                        allowed_ee_geom_name=None,
                        penetration_tolerance=min(
                            float(
                                getattr(
                                    args, "mink_collision_penetration_tolerance", 0.0
                                )
                            ),
                            1e-4,
                        ),
                    ),
                    scene_checker=_mink_q_checker,
                    max_workers=getattr(args, "autogen_mink_parallel_workers", None),
                )
                if not bool(result.collision_free):
                    mink_q_rejected_count += 1
                    _pbar_post(
                        "MINK_Q_REJECT",
                        candidate_index,
                        pen=float(result.max_penetration),
                        pos=float(result.position_error),
                    )
                    if _tqdm_pbar is not None:
                        _tqdm_pbar.update(1)
                    continue
                q_best = np.asarray(result.arm_q, dtype=np.float64).reshape(7)
                g_best = _skel_g
                pos_err = float(result.position_error)
                rot_err = float(result.rotation_error)
                max_pen = float(result.max_penetration)
                refined_pos = np.asarray(result.actual_position_world, dtype=np.float64)
                refined_rot = np.asarray(result.actual_rotation_world, dtype=np.float64)
                refined_g = _skel_g
                mink_q_accepted_count += 1
                pending_mink_q_solutions.append(
                    {
                        "candidate_index": int(candidate_index),
                        "candidate": candidate,
                        "skeleton_pose": skeleton_pose,
                        "q_best": q_best,
                        "g_best": g_best,
                        "pos_err": pos_err,
                        "rot_err": rot_err,
                        "max_pen": max_pen,
                        "refined_pos": refined_pos.copy(),
                        "refined_rot": refined_rot.copy(),
                        "refined_g": refined_g,
                        "force_direction": force_direction.copy(),
                        "retreat_distance": float(result.retreat_distance),
                    }
                )
                _pbar_post(
                    "MINK_Q_ACCEPT",
                    candidate_index,
                    pen=float(max_pen),
                    pos=float(pos_err),
                )
                if _tqdm_pbar is not None:
                    _tqdm_pbar.update(1)
                continue

            collision_free = bool(float(max_pen) <= pen_tol)
            solution = _skeleton_solution_to_contact_solution(
                ctx.env,
                ctx.panel,
                candidate,
                int(candidate_index),
                ctx.push_distance,
                ctx.robot_state,
                frame_name,
                skeleton_pose,
                q_best,
                g_best,
                pos_err,
                rot_err,
                max_pen,
                args,
            )
            ok = bool(float(pos_err) <= pos_tol and collision_free)
            reports.append(
                type(
                    "Report",
                    (),
                    {
                        "drawer_candidate_index": int(candidate_index),
                        "status": "success" if ok else "failed",
                        "reason": "success" if ok else "penetration",
                    },
                )()
            )
            if ok:
                solutions.append(solution)
        except Exception as exc:
            if solve_step2 == "mppi":
                rejected_floating_ee_count += 1
            else:
                mink_q_rejected_count += 1
            if _tqdm_pbar is not None:
                _tqdm_pbar.update(1)
            autogen_print(
                f"{yellow}STEP2_EXCEPTION "
                f"candidate={int(candidate_index)} "
                f"type={exc.__class__.__name__} "
                f"message={str(exc)[:200]}{reset_color}"
            )
            continue

    if _tqdm_pbar is not None:
        _tqdm_pbar.close()

    # --- mink rollout filter ---
    if solve_step2 == "mink" and pending_mink_q_solutions:
        mink_rollout_rejected_count = 0
        try:
            from tqdm import tqdm as _tqdm

            rollout_iter = _tqdm(
                pending_mink_q_solutions,
                desc="rollout[mink]",
                unit="q",
                file=sys.__stdout__,
                dynamic_ncols=True,
                miniters=1,
                mininterval=0.2,
                leave=True,
            )
        except Exception:
            rollout_iter = pending_mink_q_solutions
        for entry in rollout_iter:
            candidate_index = int(entry["candidate_index"])
            candidate = entry["candidate"]
            skeleton_pose = entry["skeleton_pose"]
            refined_pos = np.asarray(entry["refined_pos"], dtype=np.float64).reshape(3)
            refined_rot = np.asarray(entry["refined_rot"], dtype=np.float64).reshape(
                3, 3
            )
            refined_g = float(entry["refined_g"])
            pos_err = float(entry["pos_err"])
            rot_err = float(entry["rot_err"])
            max_pen = float(entry["max_pen"])
            _rollout = _get_rollout(
                ctx,
                args,
                entry["force_direction"],
                rollouts_by_direction,
                _target_object_pos,
            )
            _obj_delta = float("nan")
            _score_norm = float("nan")
            if _rollout is not None:
                _rollout_res = _rollout.run(refined_pos, refined_rot, refined_g)
                _obj_delta = float(_rollout_res.object_cost_delta)
                _score_norm = float(_rollout_res.score_normalized)
                obj_improved = bool(
                    np.isfinite(_score_norm) and _score_norm > _score_thresh
                )
            else:
                obj_improved = False
            if not obj_improved:
                mink_rollout_rejected_count += 1
                continue
            refined_poses.append(
                (
                    refined_pos.copy(),
                    refined_rot.copy(),
                    refined_g,
                    int(candidate_index),
                )
            )
            collision_free = bool(float(max_pen) <= pen_tol)
            solution = _skeleton_solution_to_contact_solution(
                ctx.env,
                ctx.panel,
                candidate,
                int(candidate_index),
                ctx.push_distance,
                ctx.robot_state,
                frame_name,
                skeleton_pose,
                np.asarray(entry["q_best"], dtype=np.float64).reshape(7),
                float(entry["g_best"]),
                pos_err,
                rot_err,
                max_pen,
                args,
                contact_anchor_pose=(
                    np.asarray(skeleton_pose.ee_position, dtype=np.float64).reshape(3),
                    np.asarray(skeleton_pose.ee_rotation, dtype=np.float64).reshape(
                        3, 3
                    ),
                ),
            )
            ok = bool(float(pos_err) <= pos_tol and collision_free)
            if ok:
                solutions.append(solution)
        autogen_print(
            f"stage=rollout_filter_done "
            f"accepted={len(solutions)} "
            f"rejected={mink_rollout_rejected_count}"
        )

    if _mink_q_checker is not None:
        try:
            _mink_q_checker.close()
        except Exception:
            pass

    elapsed = time.perf_counter() - started
    autogen_print(
        f"{yellow}step2[{solve_step2}] elapsed_s={elapsed:.3f} "
        f"feasible_q={len(solutions)}{reset_color}"
    )

    ctx.solutions = solutions
    ctx.reports = reports
    ctx.refined_poses = refined_poses
    if solutions:
        ctx.mink_solution = min(
            solutions,
            key=lambda s: (
                0 if getattr(s, "collision_free", False) else 1,
                getattr(s, "contact_position_error", float("inf")),
                getattr(s, "drawer_contact_cost", float("inf")),
            ),
        )
    return ctx
