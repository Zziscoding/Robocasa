from __future__ import annotations

import concurrent.futures
import hashlib
import multiprocessing
import os
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


_MINK_MODEL_SYNC_LOCK = threading.Lock()


# --- Process pool for mink IK -------------------------------------------------
# The IK worker used to run inside a ThreadPoolExecutor, but the mink solver is
# pure Python + numpy and hits the GIL; N threads collapse to ~1 core. We now
# lazily create a ProcessPoolExecutor and route parallel IK jobs through it.
# Because MuJoCo MjData is not picklable and env.sim carries live state, the
# process worker rebuilds its own robot-only MjModel from a cached .mjb file.
# If anything fails (model can't be serialized, spawn fails, etc.) callers fall
# back to the thread pool so the demo keeps running.
_PROCESS_POOL: "concurrent.futures.ProcessPoolExecutor | None" = None
_PROCESS_POOL_LOCK = threading.Lock()
_PROCESS_POOL_DISABLED = False
_PROCESS_POOL_MODEL_KEY: "str | None" = None
_PROCESS_POOL_MODEL_FILES: dict[str, str] = {}

# Per-worker state (populated in the child by _init_ik_worker).
_WORKER_ROBOT_MODEL = None
_WORKER_ROBOT_MODEL_PATH: "str | None" = None


def _init_ik_worker(model_path: str) -> None:
    """ProcessPoolExecutor initializer: build robot_model once per process."""
    global _WORKER_ROBOT_MODEL, _WORKER_ROBOT_MODEL_PATH
    try:
        import mujoco

        _WORKER_ROBOT_MODEL_PATH = str(model_path)
        _WORKER_ROBOT_MODEL = mujoco.MjModel.from_binary_path(str(model_path))
    except Exception as exc:  # pragma: no cover -- init failure surfaced to main
        _WORKER_ROBOT_MODEL = None
        _WORKER_ROBOT_MODEL_PATH = None
        print(f"[mink_q] worker init failed: {exc!r}", file=sys.stderr)


def _model_bytes_key(robot_model) -> str:
    """Compact signature for a robot-only MuJoCo model after base-pose sync."""
    digest = hashlib.sha1()
    for value in (
        getattr(robot_model, "nq", 0),
        getattr(robot_model, "nv", 0),
        getattr(robot_model, "nbody", 0),
        getattr(robot_model, "njnt", 0),
        getattr(robot_model, "nsite", 0),
        getattr(robot_model, "ngeom", 0),
    ):
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"|")
    for name in ("qpos0", "body_pos", "body_quat", "jnt_range", "names"):
        value = getattr(robot_model, name, None)
        if value is None:
            continue
        if isinstance(value, (bytes, bytearray)):
            digest.update(bytes(value))
        else:
            arr = np.ascontiguousarray(np.asarray(value))
            digest.update(str(arr.dtype).encode("ascii"))
            digest.update(str(arr.shape).encode("ascii"))
            digest.update(arr.tobytes())
    return digest.hexdigest()


def _prepare_process_model(env, robot_model) -> tuple[str, str] | None:
    """Export the robot-only MjModel to a persistent MJB for process workers."""
    try:
        import mujoco
    except Exception:
        return None
    with _MINK_MODEL_SYNC_LOCK:
        _sync_mink_base_pose(env, robot_model)
        key = _model_bytes_key(robot_model)
        path = _PROCESS_POOL_MODEL_FILES.get(key)
        if path is not None and os.path.exists(path):
            return key, path
        fd, path = tempfile.mkstemp(prefix="mink_robot_model_", suffix=".mjb")
        os.close(fd)
        try:
            mujoco.mj_saveModel(robot_model, path, None)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            return None
        _PROCESS_POOL_MODEL_FILES[key] = path
        return key, path


def _get_or_create_process_pool(
    model_key: str, model_path: str, requested_workers: int | None = None
) -> "concurrent.futures.ProcessPoolExecutor | None":
    """Return the shared process pool (creating it lazily), or None on failure."""
    global _PROCESS_POOL, _PROCESS_POOL_DISABLED, _PROCESS_POOL_MODEL_KEY
    if _PROCESS_POOL_DISABLED:
        return None
    with _PROCESS_POOL_LOCK:
        if _PROCESS_POOL_DISABLED:
            return None
        # If the model changed identity (e.g. a new env was constructed) we
        # tear down the old pool so the new MJB gets pushed to fresh workers.
        if _PROCESS_POOL is not None and _PROCESS_POOL_MODEL_KEY != model_key:
            try:
                _PROCESS_POOL.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            _PROCESS_POOL = None
            _PROCESS_POOL_MODEL_KEY = None
        if _PROCESS_POOL is None:
            try:
                # Use fork on Linux so workers do not re-import the demo script
                # as __main__.  The IK worker builds its own robot-only MjModel
                # from MJB and does not touch the parent env.sim MjData, so this
                # avoids spawn-time robocasa/robosuite import side effects while
                # keeping worker state independent.
                try:
                    ctx = multiprocessing.get_context("fork")
                except ValueError:
                    ctx = multiprocessing.get_context("spawn")
                cpu_cap = os.cpu_count() or 2
                if requested_workers is not None and int(requested_workers) > 0:
                    max_workers = max(1, min(int(requested_workers), cpu_cap))
                else:
                    max_workers = max(1, min(16, cpu_cap))
                _PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=ctx,
                    initializer=_init_ik_worker,
                    initargs=(model_path,),
                )
                _PROCESS_POOL_MODEL_KEY = model_key
            except Exception as exc:
                warnings.warn(
                    f"[mink_q] ProcessPoolExecutor creation failed ({exc!r}); "
                    "falling back to ThreadPoolExecutor.",
                    RuntimeWarning,
                )
                _PROCESS_POOL = None
                _PROCESS_POOL_DISABLED = True
                return None
        return _PROCESS_POOL


def _disable_process_pool(reason: str) -> None:
    global _PROCESS_POOL, _PROCESS_POOL_DISABLED, _PROCESS_POOL_MODEL_KEY
    with _PROCESS_POOL_LOCK:
        if _PROCESS_POOL is not None:
            try:
                _PROCESS_POOL.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        _PROCESS_POOL = None
        _PROCESS_POOL_MODEL_KEY = None
        _PROCESS_POOL_DISABLED = True
    warnings.warn(
        f"[mink_q] process pool disabled; falling back to threads: {reason}",
        RuntimeWarning,
    )


def _process_worker_solve(payload: dict) -> tuple:
    """Picklable worker: runs mink IK in a child process using the pre-built
    per-process robot_model. Payload must contain only picklable primitives."""
    import mink  # noqa: F401 -- ensure mink is importable in worker

    global _WORKER_ROBOT_MODEL
    if _WORKER_ROBOT_MODEL is None:
        # Late init in case the initializer never ran (shouldn't happen).
        model_path = payload.get("model_path")
        if model_path:
            _init_ik_worker(str(model_path))
    robot_model = _WORKER_ROBOT_MODEL
    if robot_model is None:
        raise RuntimeError("mink_q worker: robot_model unavailable")

    configuration = mink.Configuration(robot_model)
    configuration.update(np.asarray(payload["q_start"], dtype=np.float64).copy())

    posture_task = mink.PostureTask(
        robot_model,
        cost=np.asarray(payload["posture_cost"], dtype=np.float64),
        lm_damping=float(payload["mink_posture_lm_damping"]),
    )
    posture_task.set_target(np.asarray(payload["q_posture"], dtype=np.float64).copy())

    frame_task = mink.FrameTask(
        frame_name=str(payload["frame_name"]),
        frame_type="site",
        position_cost=float(payload["mink_position_cost"]),
        orientation_cost=float(payload["mink_orientation_cost"]),
        lm_damping=float(payload["mink_frame_lm_damping"]),
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(payload["target_rot"], dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(payload["target_pos"], dtype=np.float64).reshape(3)
    frame_task.set_target(mink.SE3.from_matrix(pose))

    tasks = [posture_task, frame_task]
    last_pos_error = float("inf")
    for _ in range(int(payload["mink_max_iters"])):
        velocity = mink.solve_ik(
            configuration,
            tasks,
            float(payload["mink_dt"]),
            payload["mink_solver"],
            float(payload["mink_damping"]),
        )
        configuration.integrate_inplace(velocity, float(payload["mink_dt"]))
        try:
            transform = configuration.get_transform_frame_to_world(
                str(payload["frame_name"]), "site"
            )
            actual_pos = np.asarray(transform.as_matrix()[:3, 3], dtype=np.float64)
        except Exception:
            break
        last_pos_error = float(
            np.linalg.norm(actual_pos - np.asarray(payload["target_pos"]))
        )
        if last_pos_error <= float(payload["mink_position_tolerance"]):
            break
    return configuration.q.copy(), last_pos_error


def _build_worker_payload(
    frame_name, target_pos, target_rot, q_start, q_posture, posture_cost, args
) -> dict:
    return {
        "frame_name": str(frame_name),
        "target_pos": np.asarray(target_pos, dtype=np.float64).copy(),
        "target_rot": np.asarray(target_rot, dtype=np.float64).copy(),
        "q_start": np.asarray(q_start, dtype=np.float64).copy(),
        "q_posture": np.asarray(q_posture, dtype=np.float64).copy(),
        "posture_cost": np.asarray(posture_cost, dtype=np.float64).copy(),
        "mink_posture_lm_damping": float(args.mink_posture_lm_damping),
        "mink_position_cost": float(args.mink_position_cost),
        "mink_orientation_cost": float(args.mink_orientation_cost),
        "mink_frame_lm_damping": float(args.mink_frame_lm_damping),
        "mink_max_iters": int(args.mink_max_iters),
        "mink_dt": float(args.mink_dt),
        "mink_solver": args.mink_solver,
        "mink_damping": float(args.mink_damping),
        "mink_position_tolerance": float(args.mink_position_tolerance),
    }


@dataclass(frozen=True)
class MinkQAttempt:
    target_position_world: np.ndarray
    actual_position_world: np.ndarray
    actual_rotation_world: np.ndarray
    retreat_distance: float
    position_error: float
    rotation_error: float
    max_penetration: float
    collision_free: bool
    reason: str


@dataclass(frozen=True)
class MinkQResult:
    arm_q: np.ndarray
    robot_q: np.ndarray
    target_position_world: np.ndarray
    target_rotation_world: np.ndarray
    actual_position_world: np.ndarray
    actual_rotation_world: np.ndarray
    position_error: float
    rotation_error: float
    max_penetration: float
    collision_free: bool
    collision_reason: str
    retreat_distance: float
    attempts: tuple[MinkQAttempt, ...]


def _import_mink():
    import mink

    return mink


def _normalize(vec, fallback=(1.0, 0.0, 0.0)):
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        return arr / norm
    return np.asarray(fallback, dtype=np.float64).reshape(3)


def _sync_mink_base_pose(env, robot_model) -> None:
    full_model = env.sim.model
    for body_name in ("robot0_base", "robot0_link0"):
        try:
            robot_body = robot_model.body(body_name)
            full_body = full_model.body(body_name)
        except Exception:
            continue
        robot_body.pos = full_body.pos
        robot_body.quat = full_body.quat


def _make_pose_matrix(pos, rot):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return pose


def _frame_pose_from_configuration(configuration, frame_name):
    transform = configuration.get_transform_frame_to_world(frame_name, "site")
    matrix = transform.as_matrix()
    return (
        np.asarray(matrix[:3, 3], dtype=np.float64),
        np.asarray(matrix[:3, :3], dtype=np.float64),
    )


def _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names):
    q = []
    for joint_name in arm_joint_names:
        addr = int(robot_model.joint(joint_name).qposadr[0])
        q.append(float(q_robot[addr]))
    return np.asarray(q, dtype=np.float64)


def _site_pose_for_arm_q(env, arm_joint_names, q_arm, frame_name):
    model = env.sim.model
    data = env.sim.data
    qpos_saved = data.qpos.copy()
    qvel_saved = data.qvel.copy()
    try:
        for joint_name, value in zip(arm_joint_names, q_arm):
            data.qpos[model.get_joint_qpos_addr(joint_name)] = float(value)
        env.sim.forward()
        site_id = int(model.site_name2id(frame_name))
        return (
            np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3).copy(),
            np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
        )
    finally:
        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        env.sim.forward()


def _rotation_error(target_rot, actual_rot):
    rel = np.asarray(target_rot, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        actual_rot, dtype=np.float64
    ).reshape(3, 3)
    c = 0.5 * (float(np.trace(rel)) - 1.0)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _parse_multipliers(value):
    if value is None:
        return (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(float(item) for item in items) or (1.0,)
    try:
        return tuple(float(item) for item in value) or (1.0,)
    except TypeError:
        return (float(value),)


def _solve_frame_pose(
    env,
    robot_model,
    frame_name,
    target_pos,
    target_rot,
    q_start,
    q_posture,
    posture_cost,
    args,
):
    mink = _import_mink()
    _sync_mink_base_pose(env, robot_model)

    configuration = mink.Configuration(robot_model)
    configuration.update(np.asarray(q_start, dtype=np.float64).copy())

    posture_task = mink.PostureTask(
        robot_model,
        cost=np.asarray(posture_cost, dtype=np.float64),
        lm_damping=float(args.mink_posture_lm_damping),
    )
    posture_task.set_target(np.asarray(q_posture, dtype=np.float64).copy())

    frame_task = mink.FrameTask(
        frame_name=str(frame_name),
        frame_type="site",
        position_cost=float(args.mink_position_cost),
        orientation_cost=float(args.mink_orientation_cost),
        lm_damping=float(args.mink_frame_lm_damping),
    )
    frame_task.set_target(
        mink.SE3.from_matrix(_make_pose_matrix(target_pos, target_rot))
    )

    last_pos_error = float("inf")
    for _ in range(int(args.mink_max_iters)):
        velocity = mink.solve_ik(
            configuration,
            [posture_task, frame_task],
            float(args.mink_dt),
            args.mink_solver,
            float(args.mink_damping),
        )
        configuration.integrate_inplace(velocity, float(args.mink_dt))
        actual_pos, _ = _frame_pose_from_configuration(configuration, frame_name)
        last_pos_error = float(np.linalg.norm(actual_pos - target_pos))
        if last_pos_error <= float(args.mink_position_tolerance):
            break
    return configuration.q.copy(), last_pos_error


def _solve_frame_pose_parallel_worker(
    env,
    robot_model,
    frame_name,
    target_pos,
    target_rot,
    q_start,
    q_posture,
    posture_cost,
    args,
    *,
    sync_base_pose: Callable | None = None,
    pose_matrix_fn: Callable | None = None,
    frame_pose_fn: Callable | None = None,
    skip_sync_base_pose: bool = False,
):
    """Thread worker for independent mink frame IK targets."""
    mink = _import_mink()
    sync_fn = _sync_mink_base_pose if sync_base_pose is None else sync_base_pose
    pose_fn = _make_pose_matrix if pose_matrix_fn is None else pose_matrix_fn
    frame_fn = (
        _frame_pose_from_configuration if frame_pose_fn is None else frame_pose_fn
    )
    # The base pose is a property of env.sim (not the per-job target) so callers
    # that batch many workers over one env state should sync ONCE outside the
    # pool and pass skip_sync_base_pose=True. The lock below is only taken on
    # the serial path — under the previous unconditional-lock design, every
    # ThreadPool worker serialized on this mutex, reducing N workers to ~1.
    if not skip_sync_base_pose:
        with _MINK_MODEL_SYNC_LOCK:
            sync_fn(env, robot_model)

    configuration = mink.Configuration(robot_model)
    configuration.update(np.asarray(q_start, dtype=np.float64).copy())

    posture_task = mink.PostureTask(
        robot_model,
        cost=np.asarray(posture_cost, dtype=np.float64),
        lm_damping=float(args.mink_posture_lm_damping),
    )
    posture_task.set_target(np.asarray(q_posture, dtype=np.float64).copy())

    frame_task = mink.FrameTask(
        frame_name=str(frame_name),
        frame_type="site",
        position_cost=float(args.mink_position_cost),
        orientation_cost=float(args.mink_orientation_cost),
        lm_damping=float(args.mink_frame_lm_damping),
    )
    frame_task.set_target(mink.SE3.from_matrix(pose_fn(target_pos, target_rot)))
    tasks = [posture_task, frame_task]

    last_pos_error = float("inf")
    for _ in range(int(args.mink_max_iters)):
        velocity = mink.solve_ik(
            configuration,
            tasks,
            float(args.mink_dt),
            args.mink_solver,
            float(args.mink_damping),
        )
        configuration.integrate_inplace(velocity, float(args.mink_dt))
        try:
            actual_pos, _ = frame_fn(configuration, frame_name, "site")
        except TypeError:
            actual_pos, _ = frame_fn(configuration, frame_name)
        last_pos_error = float(np.linalg.norm(actual_pos - target_pos))
        if last_pos_error <= float(args.mink_position_tolerance):
            break
    return configuration.q.copy(), last_pos_error


def solve_skeleton_precontact_q(
    env,
    *,
    robot_model,
    arm_joint_names: Sequence[str],
    frame_name: str,
    skeleton_pose,
    q_start: np.ndarray,
    q_posture: np.ndarray,
    posture_cost: np.ndarray,
    args,
    retreat_direction_world: np.ndarray | None = None,
    penetration_checker: Callable[[np.ndarray], tuple[float, str]] | None = None,
    scene_collision_checker: Callable[[np.ndarray], tuple[bool, str]] | None = None,
    scene_checker: Any | None = None,
) -> MinkQResult:
    """Solve a collision-free arm q near a skeleton contact pose.

    The skeleton pose is treated as a contact pose. We therefore evaluate that
    pose and a short sequence of poses retreated along the contact normal, using
    a stricter penetration tolerance than the later contact-stage tolerance.
    """
    arm_joint_names = tuple(arm_joint_names)
    target_rot = np.asarray(skeleton_pose.ee_rotation, dtype=np.float64).reshape(3, 3)
    contact_pos = np.asarray(skeleton_pose.ee_position, dtype=np.float64).reshape(3)

    fallback_dir = np.asarray(
        getattr(skeleton_pose, "contact_normal_world", np.array([1.0, 0.0, 0.0])),
        dtype=np.float64,
    ).reshape(3)
    retreat_dir = _normalize(
        fallback_dir if retreat_direction_world is None else retreat_direction_world,
        fallback=fallback_dir,
    )

    base_distance = float(
        getattr(
            args,
            "mink_q_precontact_distance",
            min(max(float(getattr(args, "contact_standoff", 0.005)), 0.003), 0.008),
        )
    )
    multipliers = _parse_multipliers(
        getattr(args, "mink_q_retreat_distance_multipliers", None)
    )
    position_tolerance = float(
        getattr(args, "mink_q_position_tolerance", float(args.mink_position_tolerance))
        or float(args.mink_position_tolerance)
    )
    raw_pen_tol = getattr(args, "mink_q_collision_penetration_tolerance", None)
    if raw_pen_tol is None:
        penetration_tolerance = min(
            float(getattr(args, "mink_collision_penetration_tolerance", 0.0)),
            1e-4,
        )
    else:
        penetration_tolerance = float(raw_pen_tol)

    if scene_checker is not None:
        return solve_skeleton_precontact_q_parallel(
            env,
            robot_model=robot_model,
            arm_joint_names=arm_joint_names,
            frame_name=frame_name,
            skeleton_pose=skeleton_pose,
            q_start=q_start,
            q_posture=q_posture,
            posture_cost=posture_cost,
            args=args,
            retreat_direction_world=retreat_direction_world,
            penetration_checker=penetration_checker,
            scene_collision_checker=scene_collision_checker,
            max_workers=1,
            scene_checker=scene_checker,
        )

    best: MinkQResult | None = None
    attempts: list[MinkQAttempt] = []
    for multiplier in multipliers:
        retreat_distance = max(float(base_distance) * float(multiplier), 0.0)
        target_pos = contact_pos + retreat_dir * retreat_distance
        try:
            q_robot, position_error = _solve_frame_pose(
                env,
                robot_model,
                frame_name,
                target_pos,
                target_rot,
                q_start,
                q_posture,
                posture_cost,
                args,
            )
            q_arm = _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
            actual_pos, actual_rot = _site_pose_for_arm_q(
                env, arm_joint_names, q_arm, frame_name
            )
            rotation_error = _rotation_error(target_rot, actual_rot)
            scene_ok = True
            scene_reason = "collision_free"
            if scene_collision_checker is not None:
                scene_ok, scene_reason = scene_collision_checker(q_arm)
            max_pen = 0.0
            pen_reason = "penetration_checker_unavailable"
            if penetration_checker is not None:
                max_pen, pen_reason = penetration_checker(q_arm)
            collision_free = bool(scene_ok and float(max_pen) <= penetration_tolerance)
            if not scene_ok:
                reason = str(scene_reason)
            elif float(max_pen) > penetration_tolerance:
                reason = f"drawer_penetration:{pen_reason}"
            elif float(position_error) > position_tolerance:
                reason = "position_error"
            else:
                reason = "success"
        except Exception as exc:
            q_robot = np.asarray(q_start, dtype=np.float64).copy()
            q_arm = _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
            actual_pos = np.full(3, np.nan, dtype=np.float64)
            actual_rot = np.eye(3, dtype=np.float64)
            position_error = float("inf")
            rotation_error = float("inf")
            max_pen = float("inf")
            collision_free = False
            reason = f"ik_exception:{exc.__class__.__name__}"

        attempt = MinkQAttempt(
            target_position_world=target_pos.copy(),
            actual_position_world=np.asarray(actual_pos, dtype=np.float64).copy(),
            actual_rotation_world=np.asarray(actual_rot, dtype=np.float64).copy(),
            retreat_distance=float(retreat_distance),
            position_error=float(position_error),
            rotation_error=float(rotation_error),
            max_penetration=float(max_pen),
            collision_free=bool(collision_free),
            reason=str(reason),
        )
        attempts.append(attempt)
        result = MinkQResult(
            arm_q=np.asarray(q_arm, dtype=np.float64).reshape(len(arm_joint_names)),
            robot_q=np.asarray(q_robot, dtype=np.float64).copy(),
            target_position_world=target_pos.copy(),
            target_rotation_world=target_rot.copy(),
            actual_position_world=np.asarray(actual_pos, dtype=np.float64).copy(),
            actual_rotation_world=np.asarray(actual_rot, dtype=np.float64).copy(),
            position_error=float(position_error),
            rotation_error=float(rotation_error),
            max_penetration=float(max_pen),
            collision_free=bool(
                collision_free and float(position_error) <= position_tolerance
            ),
            collision_reason=str(reason),
            retreat_distance=float(retreat_distance),
            attempts=tuple(attempts),
        )
        if best is None:
            best = result
        else:
            best_key = (
                0 if best.collision_free else 1,
                best.max_penetration,
                best.position_error,
                best.retreat_distance,
            )
            result_key = (
                0 if result.collision_free else 1,
                result.max_penetration,
                result.position_error,
                result.retreat_distance,
            )
            if result_key < best_key:
                best = result
        if result.collision_free:
            return result

    if best is None:
        raise RuntimeError("mink_q received no retreat-distance candidates")
    return MinkQResult(
        arm_q=best.arm_q,
        robot_q=best.robot_q,
        target_position_world=best.target_position_world,
        target_rotation_world=best.target_rotation_world,
        actual_position_world=best.actual_position_world,
        actual_rotation_world=best.actual_rotation_world,
        position_error=best.position_error,
        rotation_error=best.rotation_error,
        max_penetration=best.max_penetration,
        collision_free=best.collision_free,
        collision_reason=best.collision_reason,
        retreat_distance=best.retreat_distance,
        attempts=tuple(attempts),
    )


_solve_skeleton_precontact_q_serial_impl = solve_skeleton_precontact_q


def solve_skeleton_precontact_q_parallel(
    env,
    *,
    robot_model,
    arm_joint_names: Sequence[str],
    frame_name: str,
    skeleton_pose,
    q_start: np.ndarray,
    q_posture: np.ndarray,
    posture_cost: np.ndarray,
    args,
    retreat_direction_world: np.ndarray | None = None,
    penetration_checker: Callable[[np.ndarray], tuple[float, str]] | None = None,
    scene_collision_checker: Callable[[np.ndarray], tuple[bool, str]] | None = None,
    max_workers: int | None = None,
    scene_checker: Any | None = None,
) -> MinkQResult:
    """Parallel variant of :func:`solve_skeleton_precontact_q`.

    The independent retreat-distance IK targets are solved concurrently. The
    If ``scene_checker`` is supplied, all IK-solved candidates are staged into
    independent mjwarp/comfree worlds and collision-read in one batched forward.
    """
    workers = int(max_workers or getattr(args, "autogen_mink_parallel_workers", 1))
    if workers <= 1 and scene_checker is None:
        return solve_skeleton_precontact_q(
            env,
            robot_model=robot_model,
            arm_joint_names=arm_joint_names,
            frame_name=frame_name,
            skeleton_pose=skeleton_pose,
            q_start=q_start,
            q_posture=q_posture,
            posture_cost=posture_cost,
            args=args,
            retreat_direction_world=retreat_direction_world,
            penetration_checker=penetration_checker,
            scene_collision_checker=scene_collision_checker,
        )

    arm_joint_names = tuple(arm_joint_names)
    target_rot = np.asarray(skeleton_pose.ee_rotation, dtype=np.float64).reshape(3, 3)
    contact_pos = np.asarray(skeleton_pose.ee_position, dtype=np.float64).reshape(3)
    fallback_dir = np.asarray(
        getattr(skeleton_pose, "contact_normal_world", np.array([1.0, 0.0, 0.0])),
        dtype=np.float64,
    ).reshape(3)
    retreat_dir = _normalize(
        fallback_dir if retreat_direction_world is None else retreat_direction_world,
        fallback=fallback_dir,
    )
    base_distance = float(
        getattr(
            args,
            "mink_q_precontact_distance",
            min(max(float(getattr(args, "contact_standoff", 0.005)), 0.003), 0.008),
        )
    )
    multipliers = _parse_multipliers(
        getattr(args, "mink_q_retreat_distance_multipliers", None)
    )
    position_tolerance = float(
        getattr(args, "mink_q_position_tolerance", float(args.mink_position_tolerance))
        or float(args.mink_position_tolerance)
    )
    raw_pen_tol = getattr(args, "mink_q_collision_penetration_tolerance", None)
    penetration_tolerance = (
        min(float(getattr(args, "mink_collision_penetration_tolerance", 0.0)), 1e-4)
        if raw_pen_tol is None
        else float(raw_pen_tol)
    )

    jobs = []
    for multiplier in multipliers:
        retreat_distance = max(float(base_distance) * float(multiplier), 0.0)
        target_pos = contact_pos + retreat_dir * retreat_distance
        jobs.append((float(multiplier), float(retreat_distance), target_pos.copy()))
    if not jobs:
        raise RuntimeError("mink_q received no retreat-distance candidates")

    def _run(job):
        try:
            q_robot, position_error = _solve_frame_pose_parallel_worker(
                env,
                robot_model,
                frame_name,
                job[2],
                target_rot,
                q_start,
                q_posture,
                posture_cost,
                args,
                skip_sync_base_pose=True,
            )
            return job, q_robot, float(position_error), None
        except Exception as exc:
            return job, np.asarray(q_start, dtype=np.float64).copy(), float("inf"), exc

    # Sync base pose ONCE before dispatch: env.sim is not touched inside
    # workers, and all workers share robot_model, so a per-worker sync would
    # both serialize (lock) and race (shared model writes).
    with _MINK_MODEL_SYNC_LOCK:
        _sync_mink_base_pose(env, robot_model)

    results = []
    if workers <= 1:
        results = [_run(job) for job in jobs]
    else:
        # Try process pool first; fall back to threads on any failure.
        used_process_pool = False
        model_spec = _prepare_process_model(env, robot_model)
        if model_spec is not None:
            pool = _get_or_create_process_pool(
                model_spec[0], model_spec[1], requested_workers=workers
            )
            if pool is not None:
                try:
                    proc_futures = {}
                    for job in jobs:
                        payload = _build_worker_payload(
                            frame_name,
                            job[2],
                            target_rot,
                            q_start,
                            q_posture,
                            posture_cost,
                            args,
                        )
                        proc_futures[pool.submit(_process_worker_solve, payload)] = job
                    for future in concurrent.futures.as_completed(proc_futures):
                        job = proc_futures[future]
                        try:
                            q_robot, position_error = future.result()
                            results.append((job, q_robot, float(position_error), None))
                        except Exception as exc:
                            results.append(
                                (
                                    job,
                                    np.asarray(q_start, dtype=np.float64).copy(),
                                    float("inf"),
                                    exc,
                                )
                            )
                    used_process_pool = True
                except Exception as exc:
                    _disable_process_pool(f"submit/collect failed: {exc!r}")
                    results = []
            if (
                used_process_pool
                and results
                and all(item[3] is not None for item in results)
            ):
                _disable_process_pool("all process IK jobs failed")
                used_process_pool = False
                results = []
        if not used_process_pool:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(workers, len(jobs)))
            ) as executor:
                futures = [executor.submit(_run, job) for job in jobs]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
    result_by_multiplier = {
        job[0]: (job, q_robot, position_error, exc)
        for job, q_robot, position_error, exc in results
    }

    # If a full-scene mjwarp checker was supplied, evaluate every IK-solved
    # candidate without touching shared env.sim.  The worker-pool checker gives
    # each worker a private one-world backend; the older checker API stages all
    # candidates into one batched backend.  Candidates whose IK raised an
    # exception have no q_arm to check and fall through to the failure branch.
    staged_world: dict[int, int] = {}
    pooled_report: dict[int, Any] = {}
    if scene_checker is not None:
        if hasattr(scene_checker, "evaluate_candidates_threadsafe"):
            q_entries = []
            for job_index, job in enumerate(jobs):
                multiplier = job[0]
                _, q_robot, _, exc = result_by_multiplier[multiplier]
                if exc is not None:
                    continue
                q_arm = _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
                q_entries.append((job_index, q_arm))
            if q_entries:
                reports = scene_checker.evaluate_candidates_threadsafe(
                    [q_arm for _, q_arm in q_entries]
                )
                if len(reports) != len(q_entries):
                    raise RuntimeError(
                        "scene_checker.evaluate_candidates_threadsafe returned "
                        f"{len(reports)} reports for {len(q_entries)} candidates"
                    )
                pooled_report = {
                    job_index: report
                    for (job_index, _), report in zip(q_entries, reports)
                }
        elif hasattr(scene_checker, "evaluate_candidates"):
            q_entries = []
            for job_index, job in enumerate(jobs):
                multiplier = job[0]
                _, q_robot, _, exc = result_by_multiplier[multiplier]
                if exc is not None:
                    continue
                q_arm = _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
                q_entries.append((job_index, q_arm))
            if q_entries:
                reports = scene_checker.evaluate_candidates(
                    [q_arm for _, q_arm in q_entries]
                )
                if len(reports) != len(q_entries):
                    raise RuntimeError(
                        "scene_checker.evaluate_candidates returned "
                        f"{len(reports)} reports for {len(q_entries)} candidates"
                    )
                pooled_report = {
                    job_index: report
                    for (job_index, _), report in zip(q_entries, reports)
                }
        else:
            for job_index, job in enumerate(jobs):
                multiplier = job[0]
                _, q_robot, _, exc = result_by_multiplier[multiplier]
                if exc is not None:
                    continue
                q_arm = _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
                staged_world[job_index] = scene_checker.submit(q_arm)
            if staged_world:
                scene_checker.evaluate()

    best: MinkQResult | None = None
    attempts: list[MinkQAttempt] = []
    for job_index, job in enumerate(jobs):
        multiplier, retreat_distance, target_pos = job
        _, q_robot, position_error, exc = result_by_multiplier[multiplier]
        q_arm = _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
        world_id = staged_world.get(job_index)
        staged = world_id is not None
        if exc is None and job_index in pooled_report:
            report = pooled_report[job_index]
            actual_pos = np.asarray(report.ee_position, dtype=np.float64).reshape(3)
            actual_rot = np.asarray(report.ee_rotation, dtype=np.float64).reshape(3, 3)
            rotation_error = _rotation_error(target_rot, actual_rot)
            scene_ok = bool(report.scene_collision_free)
            scene_reason = str(report.scene_reason)
            max_pen = float(report.max_penetration)
            pen_reason = str(report.penetration_reason)
            collision_free = bool(scene_ok and float(max_pen) <= penetration_tolerance)
            if not scene_ok:
                reason = str(scene_reason)
            elif float(max_pen) > penetration_tolerance:
                reason = f"drawer_penetration:{pen_reason}"
            elif float(position_error) > position_tolerance:
                reason = "position_error"
            else:
                reason = "success"
        elif exc is None and staged:
            actual_pos, actual_rot = scene_checker.ee_pose(world_id)
            actual_pos = np.asarray(actual_pos, dtype=np.float64).reshape(3)
            actual_rot = np.asarray(actual_rot, dtype=np.float64).reshape(3, 3)
            rotation_error = _rotation_error(target_rot, actual_rot)
            scene_ok, scene_reason = scene_checker.scene_collision(world_id)
            max_pen, pen_reason = scene_checker.penetration(world_id)
            collision_free = bool(scene_ok and float(max_pen) <= penetration_tolerance)
            if not scene_ok:
                reason = str(scene_reason)
            elif float(max_pen) > penetration_tolerance:
                reason = f"drawer_penetration:{pen_reason}"
            elif float(position_error) > position_tolerance:
                reason = "position_error"
            else:
                reason = "success"
        elif exc is None:
            actual_pos, actual_rot = _site_pose_for_arm_q(
                env,
                arm_joint_names,
                q_arm,
                frame_name,
            )
            rotation_error = _rotation_error(target_rot, actual_rot)
            scene_ok = True
            scene_reason = "collision_free"
            if scene_collision_checker is not None:
                scene_ok, scene_reason = scene_collision_checker(q_arm)
            max_pen = 0.0
            pen_reason = "penetration_checker_unavailable"
            if penetration_checker is not None:
                max_pen, pen_reason = penetration_checker(q_arm)
            collision_free = bool(scene_ok and float(max_pen) <= penetration_tolerance)
            if not scene_ok:
                reason = str(scene_reason)
            elif float(max_pen) > penetration_tolerance:
                reason = f"drawer_penetration:{pen_reason}"
            elif float(position_error) > position_tolerance:
                reason = "position_error"
            else:
                reason = "success"
        else:
            actual_pos = np.full(3, np.nan, dtype=np.float64)
            actual_rot = np.eye(3, dtype=np.float64)
            rotation_error = float("inf")
            max_pen = float("inf")
            collision_free = False
            reason = f"ik_exception:{exc.__class__.__name__}"

        attempt = MinkQAttempt(
            target_position_world=target_pos.copy(),
            actual_position_world=np.asarray(actual_pos, dtype=np.float64).copy(),
            actual_rotation_world=np.asarray(actual_rot, dtype=np.float64).copy(),
            retreat_distance=float(retreat_distance),
            position_error=float(position_error),
            rotation_error=float(rotation_error),
            max_penetration=float(max_pen),
            collision_free=bool(collision_free),
            reason=str(reason),
        )
        attempts.append(attempt)
        result = MinkQResult(
            arm_q=np.asarray(q_arm, dtype=np.float64).reshape(len(arm_joint_names)),
            robot_q=np.asarray(q_robot, dtype=np.float64).copy(),
            target_position_world=target_pos.copy(),
            target_rotation_world=target_rot.copy(),
            actual_position_world=np.asarray(actual_pos, dtype=np.float64).copy(),
            actual_rotation_world=np.asarray(actual_rot, dtype=np.float64).copy(),
            position_error=float(position_error),
            rotation_error=float(rotation_error),
            max_penetration=float(max_pen),
            collision_free=bool(
                collision_free and float(position_error) <= position_tolerance
            ),
            collision_reason=str(reason),
            retreat_distance=float(retreat_distance),
            attempts=tuple(attempts),
        )
        if best is None:
            best = result
        else:
            best_key = (
                0 if best.collision_free else 1,
                best.max_penetration,
                best.position_error,
                best.retreat_distance,
            )
            result_key = (
                0 if result.collision_free else 1,
                result.max_penetration,
                result.position_error,
                result.retreat_distance,
            )
            if result_key < best_key:
                best = result
        if result.collision_free:
            return result

    if best is None:
        raise RuntimeError("mink_q received no retreat-distance candidates")
    return MinkQResult(
        arm_q=best.arm_q,
        robot_q=best.robot_q,
        target_position_world=best.target_position_world,
        target_rotation_world=best.target_rotation_world,
        actual_position_world=best.actual_position_world,
        actual_rotation_world=best.actual_rotation_world,
        position_error=best.position_error,
        rotation_error=best.rotation_error,
        max_penetration=best.max_penetration,
        collision_free=best.collision_free,
        collision_reason=best.collision_reason,
        retreat_distance=best.retreat_distance,
        attempts=tuple(attempts),
    )


def solve_skeleton_precontact_q_batch(
    env,
    *,
    robot_model,
    arm_joint_names: Sequence[str],
    frame_name: str,
    pose_entries: Sequence[tuple[int, Any, np.ndarray | None]],
    q_start: np.ndarray,
    q_posture: np.ndarray,
    posture_cost: np.ndarray,
    args,
    scene_checker: Any | None = None,
    penetration_checker: Callable[[np.ndarray], tuple[float, str]] | None = None,
    scene_collision_checker: Callable[[np.ndarray], tuple[bool, str]] | None = None,
    max_workers: int | None = None,
) -> dict[int, MinkQResult]:
    """Solve many skeleton precontact IK problems in one process-pool batch.

    ``scene_checker`` is expected to be the full-scene checker pool from
    ``full_scene_mjwarp.py``.  It is intentionally evaluated after all IK jobs
    finish so GPU full-scene collision checks can be batched across poses.
    """
    entries = [
        (int(pose_index), skeleton_pose, retreat_direction_world)
        for pose_index, skeleton_pose, retreat_direction_world in pose_entries
    ]
    if not entries:
        return {}
    workers = int(max_workers or getattr(args, "autogen_mink_parallel_workers", 1))
    workers = max(workers, 1)
    arm_joint_names = tuple(arm_joint_names)
    base_distance = float(
        getattr(
            args,
            "mink_q_precontact_distance",
            min(max(float(getattr(args, "contact_standoff", 0.005)), 0.003), 0.008),
        )
    )
    multipliers = _parse_multipliers(
        getattr(args, "mink_q_retreat_distance_multipliers", None)
    )
    if not multipliers:
        raise RuntimeError("mink_q received no retreat-distance candidates")
    position_tolerance = float(
        getattr(args, "mink_q_position_tolerance", float(args.mink_position_tolerance))
        or float(args.mink_position_tolerance)
    )
    raw_pen_tol = getattr(args, "mink_q_collision_penetration_tolerance", None)
    penetration_tolerance = (
        min(float(getattr(args, "mink_collision_penetration_tolerance", 0.0)), 1e-4)
        if raw_pen_tol is None
        else float(raw_pen_tol)
    )

    jobs: list[dict[str, Any]] = []
    entry_by_pose_index = {}
    for pose_index, skeleton_pose, retreat_direction_world in entries:
        entry_by_pose_index[int(pose_index)] = skeleton_pose
        target_rot = np.asarray(skeleton_pose.ee_rotation, dtype=np.float64).reshape(
            3, 3
        )
        contact_pos = np.asarray(skeleton_pose.ee_position, dtype=np.float64).reshape(3)
        fallback_dir = np.asarray(
            getattr(skeleton_pose, "contact_normal_world", np.array([1.0, 0.0, 0.0])),
            dtype=np.float64,
        ).reshape(3)
        retreat_dir = _normalize(
            fallback_dir
            if retreat_direction_world is None
            else retreat_direction_world,
            fallback=fallback_dir,
        )
        for local_job_index, multiplier in enumerate(multipliers):
            retreat_distance = max(float(base_distance) * float(multiplier), 0.0)
            target_pos = contact_pos + retreat_dir * retreat_distance
            jobs.append(
                {
                    "global_index": len(jobs),
                    "local_job_index": int(local_job_index),
                    "pose_index": int(pose_index),
                    "multiplier": float(multiplier),
                    "retreat_distance": float(retreat_distance),
                    "target_pos": np.asarray(target_pos, dtype=np.float64).copy(),
                    "target_rot": np.asarray(target_rot, dtype=np.float64).copy(),
                }
            )
    if not jobs:
        return {}

    def _thread_run(job: dict[str, Any]):
        try:
            q_robot, position_error = _solve_frame_pose_parallel_worker(
                env,
                robot_model,
                frame_name,
                job["target_pos"],
                job["target_rot"],
                q_start,
                q_posture,
                posture_cost,
                args,
                skip_sync_base_pose=True,
            )
            return int(job["global_index"]), q_robot, float(position_error), None
        except Exception as exc:
            return (
                int(job["global_index"]),
                np.asarray(q_start, dtype=np.float64).copy(),
                float("inf"),
                exc,
            )

    with _MINK_MODEL_SYNC_LOCK:
        _sync_mink_base_pose(env, robot_model)

    results_by_global: dict[int, tuple[np.ndarray, float, Exception | None]] = {}
    used_process_pool = False
    if workers > 1:
        model_spec = _prepare_process_model(env, robot_model)
        if model_spec is not None:
            pool = _get_or_create_process_pool(
                model_spec[0], model_spec[1], requested_workers=workers
            )
            if pool is not None:
                try:
                    futures = {}
                    for job in jobs:
                        payload = _build_worker_payload(
                            frame_name,
                            job["target_pos"],
                            job["target_rot"],
                            q_start,
                            q_posture,
                            posture_cost,
                            args,
                        )
                        futures[pool.submit(_process_worker_solve, payload)] = int(
                            job["global_index"]
                        )
                    for future in concurrent.futures.as_completed(futures):
                        global_index = int(futures[future])
                        try:
                            q_robot, position_error = future.result()
                            results_by_global[global_index] = (
                                np.asarray(q_robot, dtype=np.float64).copy(),
                                float(position_error),
                                None,
                            )
                        except Exception as exc:
                            results_by_global[global_index] = (
                                np.asarray(q_start, dtype=np.float64).copy(),
                                float("inf"),
                                exc,
                            )
                    used_process_pool = True
                except Exception as exc:
                    _disable_process_pool(f"batch submit/collect failed: {exc!r}")
                    results_by_global.clear()
            if (
                used_process_pool
                and results_by_global
                and all(item[2] is not None for item in results_by_global.values())
            ):
                _disable_process_pool("all batch process IK jobs failed")
                used_process_pool = False
                results_by_global.clear()

    if not used_process_pool:
        if workers <= 1:
            thread_results = [_thread_run(job) for job in jobs]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(workers, len(jobs)))
            ) as executor:
                futures = [executor.submit(_thread_run, job) for job in jobs]
                thread_results = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]
        for global_index, q_robot, position_error, exc in thread_results:
            results_by_global[int(global_index)] = (
                np.asarray(q_robot, dtype=np.float64).copy(),
                float(position_error),
                exc,
            )

    report_by_global: dict[int, Any] = {}
    if scene_checker is not None:
        q_entries: list[tuple[int, np.ndarray]] = []
        for job in jobs:
            q_robot, _, exc = results_by_global[int(job["global_index"])]
            if exc is not None:
                continue
            q_arm = _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
            q_entries.append((int(job["global_index"]), q_arm))
        if q_entries:
            if hasattr(scene_checker, "evaluate_candidates"):
                collision_reports = scene_checker.evaluate_candidates(
                    [q_arm for _, q_arm in q_entries]
                )
            elif hasattr(scene_checker, "evaluate_candidates_threadsafe"):
                collision_reports = scene_checker.evaluate_candidates_threadsafe(
                    [q_arm for _, q_arm in q_entries]
                )
            else:
                collision_reports = []
                staged_world = {}
                scene_checker.reset()
                for global_index, q_arm in q_entries:
                    staged_world[int(global_index)] = scene_checker.submit(q_arm)
                scene_checker.evaluate()
                for global_index, _ in q_entries:
                    world_id = staged_world[int(global_index)]
                    max_pen, pen_reason = scene_checker.penetration(world_id)
                    scene_ok, scene_reason = scene_checker.scene_collision(world_id)
                    ee_pos, ee_rot = scene_checker.ee_pose(world_id)
                    collision_reports.append(
                        type(
                            "_MinkQCollisionReport",
                            (),
                            {
                                "max_penetration": float(max_pen),
                                "penetration_reason": str(pen_reason),
                                "scene_collision_free": bool(scene_ok),
                                "scene_reason": str(scene_reason),
                                "ee_position": np.asarray(ee_pos, dtype=np.float64),
                                "ee_rotation": np.asarray(ee_rot, dtype=np.float64),
                            },
                        )()
                    )
            if len(collision_reports) != len(q_entries):
                raise RuntimeError(
                    "scene_checker returned "
                    f"{len(collision_reports)} reports for {len(q_entries)} candidates"
                )
            report_by_global = {
                global_index: report
                for (global_index, _), report in zip(q_entries, collision_reports)
            }

    jobs_by_pose: dict[int, list[dict[str, Any]]] = {}
    for job in jobs:
        jobs_by_pose.setdefault(int(job["pose_index"]), []).append(job)

    results: dict[int, MinkQResult] = {}
    for pose_index, pose_jobs in jobs_by_pose.items():
        pose_jobs = sorted(pose_jobs, key=lambda item: int(item["local_job_index"]))
        best: MinkQResult | None = None
        attempts: list[MinkQAttempt] = []
        for job in pose_jobs:
            global_index = int(job["global_index"])
            q_robot, position_error, exc = results_by_global[global_index]
            q_arm = _arm_q_from_robot_model_q(robot_model, q_robot, arm_joint_names)
            target_rot = np.asarray(job["target_rot"], dtype=np.float64).reshape(3, 3)
            target_pos = np.asarray(job["target_pos"], dtype=np.float64).reshape(3)
            if exc is None and global_index in report_by_global:
                report = report_by_global[global_index]
                actual_pos = np.asarray(report.ee_position, dtype=np.float64).reshape(3)
                actual_rot = np.asarray(report.ee_rotation, dtype=np.float64).reshape(
                    3, 3
                )
                rotation_error = _rotation_error(target_rot, actual_rot)
                scene_ok = bool(report.scene_collision_free)
                scene_reason = str(report.scene_reason)
                max_pen = float(report.max_penetration)
                pen_reason = str(report.penetration_reason)
                collision_free = bool(scene_ok and max_pen <= penetration_tolerance)
                if not scene_ok:
                    reason = scene_reason
                elif max_pen > penetration_tolerance:
                    reason = f"drawer_penetration:{pen_reason}"
                elif float(position_error) > position_tolerance:
                    reason = "position_error"
                else:
                    reason = "success"
            elif exc is None:
                actual_pos, actual_rot = _site_pose_for_arm_q(
                    env, arm_joint_names, q_arm, frame_name
                )
                rotation_error = _rotation_error(target_rot, actual_rot)
                scene_ok = True
                scene_reason = "collision_free"
                if scene_collision_checker is not None:
                    scene_ok, scene_reason = scene_collision_checker(q_arm)
                max_pen = 0.0
                pen_reason = "penetration_checker_unavailable"
                if penetration_checker is not None:
                    max_pen, pen_reason = penetration_checker(q_arm)
                collision_free = bool(scene_ok and max_pen <= penetration_tolerance)
                if not scene_ok:
                    reason = str(scene_reason)
                elif max_pen > penetration_tolerance:
                    reason = f"drawer_penetration:{pen_reason}"
                elif float(position_error) > position_tolerance:
                    reason = "position_error"
                else:
                    reason = "success"
            else:
                actual_pos = np.full(3, np.nan, dtype=np.float64)
                actual_rot = np.eye(3, dtype=np.float64)
                rotation_error = float("inf")
                max_pen = float("inf")
                collision_free = False
                reason = f"ik_exception:{exc.__class__.__name__}"

            attempt = MinkQAttempt(
                target_position_world=target_pos.copy(),
                actual_position_world=np.asarray(actual_pos, dtype=np.float64).copy(),
                actual_rotation_world=np.asarray(actual_rot, dtype=np.float64).copy(),
                retreat_distance=float(job["retreat_distance"]),
                position_error=float(position_error),
                rotation_error=float(rotation_error),
                max_penetration=float(max_pen),
                collision_free=bool(collision_free),
                reason=str(reason),
            )
            attempts.append(attempt)
            result = MinkQResult(
                arm_q=np.asarray(q_arm, dtype=np.float64).reshape(len(arm_joint_names)),
                robot_q=np.asarray(q_robot, dtype=np.float64).copy(),
                target_position_world=target_pos.copy(),
                target_rotation_world=target_rot.copy(),
                actual_position_world=np.asarray(actual_pos, dtype=np.float64).copy(),
                actual_rotation_world=np.asarray(actual_rot, dtype=np.float64).copy(),
                position_error=float(position_error),
                rotation_error=float(rotation_error),
                max_penetration=float(max_pen),
                collision_free=bool(
                    collision_free and float(position_error) <= position_tolerance
                ),
                collision_reason=str(reason),
                retreat_distance=float(job["retreat_distance"]),
                attempts=tuple(attempts),
            )
            if best is None:
                best = result
            else:
                best_key = (
                    0 if best.collision_free else 1,
                    best.max_penetration,
                    best.position_error,
                    best.retreat_distance,
                )
                result_key = (
                    0 if result.collision_free else 1,
                    result.max_penetration,
                    result.position_error,
                    result.retreat_distance,
                )
                if result_key < best_key:
                    best = result
            if result.collision_free:
                break
        if best is None:
            continue
        results[int(pose_index)] = MinkQResult(
            arm_q=best.arm_q,
            robot_q=best.robot_q,
            target_position_world=best.target_position_world,
            target_rotation_world=best.target_rotation_world,
            actual_position_world=best.actual_position_world,
            actual_rotation_world=best.actual_rotation_world,
            position_error=best.position_error,
            rotation_error=best.rotation_error,
            max_penetration=best.max_penetration,
            collision_free=best.collision_free,
            collision_reason=best.collision_reason,
            retreat_distance=best.retreat_distance,
            attempts=tuple(attempts),
        )
    missing = sorted(set(entry_by_pose_index) - set(results))
    if missing:
        raise RuntimeError(
            f"mink_q batch produced no result for pose indices {missing}"
        )
    return results


def solve_mink_for_drawer_candidate_parallel(
    env,
    panel,
    candidate,
    candidate_index,
    push_distance,
    robot_model,
    arm_joint_names,
    frame_name,
    q_initial,
    q_posture,
    posture_cost,
    contact_offsets,
    roll_angles,
    base_rot,
    args,
    *,
    close_ops,
    original_solver: Callable | None = None,
    status_printer: Callable[[str], None] | None = None,
    max_workers: int | None = None,
):
    """Parallelize close-drawer contact-pose IK candidates.

    This helper keeps close-demo-specific construction behind ``close_ops`` so
    the reusable parallel mink logic lives here without importing the demo at
    module import time.
    """
    workers = int(max_workers or getattr(args, "autogen_mink_parallel_workers", 1))
    if workers <= 1:
        if original_solver is None:
            raise ValueError("original_solver is required when workers <= 1")
        return original_solver(
            env,
            panel,
            candidate,
            candidate_index,
            push_distance,
            robot_model,
            arm_joint_names,
            frame_name,
            q_initial,
            q_posture,
            posture_cost,
            contact_offsets,
            roll_angles,
            base_rot,
            args,
        )

    report_cls = close_ops.MinkContactAttemptReport
    if args.require_feasible_contact and not candidate.feasible:
        report = report_cls(
            drawer_candidate_index=int(candidate_index),
            drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
            drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
            drawer_contact_cost=float(candidate.cost),
            contact_feasible=bool(candidate.feasible),
            status="failed",
            reason="contact_cost_above_threshold",
            best_ee_sample_index=-1,
            best_ee_sample_name="",
            best_position_error=float("inf"),
            best_collision_free=False,
        )
        return None, report

    jobs = []
    for ee_sample_index, (
        ee_sample_name,
        contact_offset,
        ee_contact_geom_name,
    ) in enumerate(contact_offsets):
        if ee_sample_name == "gripper_center" and not args.mink_include_grip_site:
            continue
        for roll_angle in roll_angles:
            target_rot = base_rot @ close_ops._rot_about_axis(
                [1.0, 0.0, 0.0], roll_angle
            )
            desired_contact_point = (
                candidate.world_point + panel.outward_world * args.contact_standoff
            )
            target_frame_pos = desired_contact_point - target_rot @ contact_offset
            jobs.append(
                (
                    int(ee_sample_index),
                    str(ee_sample_name),
                    str(ee_contact_geom_name),
                    np.asarray(contact_offset, dtype=np.float64).copy(),
                    float(roll_angle),
                    np.asarray(target_rot, dtype=np.float64).copy(),
                    np.asarray(target_frame_pos, dtype=np.float64).copy(),
                )
            )

    if not jobs:
        report = report_cls(
            drawer_candidate_index=int(candidate_index),
            drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
            drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
            drawer_contact_cost=float(candidate.cost),
            contact_feasible=bool(candidate.feasible),
            status="failed",
            reason="not_attempted",
            best_ee_sample_index=-1,
            best_ee_sample_name="",
            best_position_error=float("inf"),
            best_collision_free=False,
        )
        return None, report

    def _run(job):
        try:
            q_contact, pos_error = _solve_frame_pose_parallel_worker(
                env,
                robot_model,
                frame_name,
                job[6],
                job[5],
                q_initial,
                q_posture,
                posture_cost,
                args,
                sync_base_pose=close_ops._sync_mink_base_pose,
                pose_matrix_fn=close_ops._make_pose_matrix,
                frame_pose_fn=close_ops._frame_pose_from_configuration,
                skip_sync_base_pose=True,
            )
            return job, q_contact, float(pos_error), None
        except Exception as exc:
            return job, None, float("inf"), f"ik_exception:{exc.__class__.__name__}"

    # Sync base pose once before dispatch — see solve_skeleton_precontact_q_parallel.
    with _MINK_MODEL_SYNC_LOCK:
        close_ops._sync_mink_base_pose(env, robot_model)

    results = []
    started = time.perf_counter()
    used_process_pool = False
    model_spec = _prepare_process_model(env, robot_model)
    if model_spec is not None:
        pool = _get_or_create_process_pool(
            model_spec[0], model_spec[1], requested_workers=workers
        )
        if pool is not None:
            try:
                proc_futures = {}
                for job in jobs:
                    payload = _build_worker_payload(
                        frame_name,
                        job[6],
                        job[5],
                        q_initial,
                        q_posture,
                        posture_cost,
                        args,
                    )
                    proc_futures[pool.submit(_process_worker_solve, payload)] = job
                for future in concurrent.futures.as_completed(proc_futures):
                    job = proc_futures[future]
                    try:
                        q_contact, pos_error = future.result()
                        results.append((job, q_contact, float(pos_error), None))
                    except Exception as exc:
                        results.append(
                            (
                                job,
                                None,
                                float("inf"),
                                f"ik_exception:{exc.__class__.__name__}",
                            )
                        )
                used_process_pool = True
            except Exception as exc:
                _disable_process_pool(f"submit/collect failed: {exc!r}")
                results = []
        if (
            used_process_pool
            and results
            and all(item[3] is not None for item in results)
        ):
            _disable_process_pool("all process IK jobs failed")
            used_process_pool = False
            results = []
    if not used_process_pool:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(workers, len(jobs)))
        ) as executor:
            futures = [executor.submit(_run, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    result_by_key = {
        (job[0], job[4]): (job, q_contact, pos_error, error)
        for job, q_contact, pos_error, error in results
    }

    best = None
    best_score = np.inf
    best_reason = "not_attempted"
    for job in jobs:
        (
            ee_sample_index,
            ee_sample_name,
            ee_contact_geom_name,
            contact_offset,
            roll_angle,
            target_rot,
            _,
        ) = job
        _, q_contact, pos_error, error = result_by_key[(ee_sample_index, roll_angle)]
        if error is not None:
            best_reason = str(error)
            continue

        q_arm = close_ops._arm_q_from_robot_model_q(
            robot_model, q_contact, arm_joint_names
        )
        collision_free, collision_reason = close_ops._check_arm_q_collision(
            env,
            panel,
            arm_joint_names,
            q_arm,
            allowed_ee_geom_name=ee_contact_geom_name,
            penetration_tolerance=args.mink_collision_penetration_tolerance,
        )
        within_tolerance = bool(pos_error <= args.mink_position_tolerance)
        score = float(pos_error) + (
            0.0 if collision_free else args.mink_collision_penalty
        )
        if score < best_score:
            best_score = score
            best = (
                ee_sample_index,
                ee_sample_name,
                ee_contact_geom_name,
                contact_offset,
                roll_angle,
                target_rot,
                q_contact,
                float(pos_error),
                bool(collision_free),
                collision_reason,
            )
            if not within_tolerance:
                best_reason = "position_error"
            elif not collision_free:
                best_reason = collision_reason
            else:
                best_reason = "success"

        if within_tolerance and collision_free:
            solution = close_ops._build_mink_solution(
                env,
                panel,
                candidate,
                candidate_index,
                push_distance,
                robot_model,
                arm_joint_names,
                frame_name,
                ee_sample_index,
                ee_sample_name,
                ee_contact_geom_name,
                contact_offset,
                target_rot,
                roll_angle,
                q_initial,
                q_posture,
                posture_cost,
                pos_error,
                collision_free,
                args,
            )
            if (
                solution.contact_position_error <= args.mink_position_tolerance
                and solution.collision_free
            ):
                if status_printer is not None:
                    status_printer(
                        "mink_parallel_candidate "
                        f"candidate={int(candidate_index)} workers={max(1, min(workers, len(jobs)))} "
                        f"jobs={len(jobs)} elapsed_s={time.perf_counter() - started:.3f}"
                    )
                report = report_cls(
                    drawer_candidate_index=int(candidate_index),
                    drawer_contact_world=np.asarray(
                        candidate.world_point, dtype=np.float64
                    ),
                    drawer_contact_local=np.asarray(
                        candidate.local_point, dtype=np.float64
                    ),
                    drawer_contact_cost=float(candidate.cost),
                    contact_feasible=bool(candidate.feasible),
                    status="success",
                    reason="success",
                    best_ee_sample_index=int(ee_sample_index),
                    best_ee_sample_name=str(ee_sample_name),
                    best_position_error=float(solution.contact_position_error),
                    best_collision_free=True,
                )
                return solution, report
            best_reason = (
                "position_error"
                if solution.contact_position_error > args.mink_position_tolerance
                else "trajectory_collision"
            )

    if best is None:
        report = report_cls(
            drawer_candidate_index=int(candidate_index),
            drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
            drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
            drawer_contact_cost=float(candidate.cost),
            contact_feasible=bool(candidate.feasible),
            status="failed",
            reason=best_reason,
            best_ee_sample_index=-1,
            best_ee_sample_name="",
            best_position_error=float("inf"),
            best_collision_free=False,
        )
        return None, report

    (
        ee_sample_index,
        ee_sample_name,
        _,
        _,
        _,
        _,
        _,
        pos_error,
        collision_free,
        collision_reason,
    ) = best
    if pos_error > args.mink_position_tolerance:
        reason = "position_error"
    elif collision_free:
        reason = best_reason if best_reason != "success" else "trajectory_collision"
    else:
        reason = collision_reason
    report = report_cls(
        drawer_candidate_index=int(candidate_index),
        drawer_contact_world=np.asarray(candidate.world_point, dtype=np.float64),
        drawer_contact_local=np.asarray(candidate.local_point, dtype=np.float64),
        drawer_contact_cost=float(candidate.cost),
        contact_feasible=bool(candidate.feasible),
        status="failed",
        reason=reason,
        best_ee_sample_index=int(ee_sample_index),
        best_ee_sample_name=str(ee_sample_name),
        best_position_error=float(pos_error),
        best_collision_free=bool(collision_free),
    )
    return None, report


__all__ = [
    "MinkQAttempt",
    "MinkQResult",
    "solve_mink_for_drawer_candidate_parallel",
    "solve_skeleton_precontact_q",
    "solve_skeleton_precontact_q_parallel",
]
