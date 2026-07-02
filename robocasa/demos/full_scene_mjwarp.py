"""Full-scene mjwarp backend for parallel collision / penetration queries.

The existing rollout / MPPI stack (``rollout.py``, ``ee_floating_mppi.py``) builds a
*stripped* floating-hand + target-object contact model and runs it on
mjwarp/comfree.  For the skeleton pre-contact ``q`` search in ``mink_q.py`` we
need the *complete* robot operating inside the *complete* kitchen scene, because
the two checkers we must honour —

* ``_strict_robot_drawer_penetration`` — report the deepest overlap between the
  robot+gripper collision geoms and the drawer door geoms, and
* ``_check_arm_q_collision`` — report whether any robot geom penetrates the rest
  of the scene (excluding the single EE/drawer contact we *want* to keep),

both read contact information off the full MuJoCo model.  Today those two checkers
are wired to the single shared ``env.sim`` and evaluated *serially* per candidate
so the parallel solve in :func:`robocasa.demos.mink_q.solve_skeleton_precontact_q_parallel`
still contends on one MuJoCo environment.

This module clones the full ``env.sim.model``/``env.sim.data`` into an
mjwarp/comfree :class:`Data` with ``nworld = N``.  Every candidate arm ``q`` is
written into its *own* world, a single batched ``forward``/``collision`` is run and
the per-world penetration / collision answer is read back — so the ``N`` candidate
queries are truly independent and run in parallel on the GPU.

Typical usage (from ``mink_q.solve_skeleton_precontact_q_parallel``)::

    scene = FullSceneMjWarp.from_env(
        env,
        arm_joint_names=arm_joint_names,
        frame_name=frame_name,
        panel=panel,
        nworld=len(multipliers),
    )
    checker = scene.checker(
        penetration_tolerance=mink_q_scene_pen_tol,
    )
    # later, for each IK'd candidate q_arm:
    report = checker.submit(q_arm)          # stage into the next free world
    pen, reason = checker.penetration(i)    # robot-drawer penetration, world i
    ok, reason = checker.scene_collision(i) # robot-vs-scene, world i
"""

from __future__ import annotations

import sys
import concurrent.futures
import gc
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import mujoco
import numpy as np
import torch

from robocasa.demos.ee_floating_mppi import (  # noqa: E402
    _backend_tensor,
    _import_comfree_backend,
    _import_mjwarp_backend,
)

_BACKEND_IMPORT_LOCK = threading.Lock()
_BACKEND_BUILD_LOCK = threading.Lock()


def _drop_partial_comfree_modules() -> None:
    for name in list(sys.modules):
        if name == "comfree_warp" or name.startswith("comfree_warp."):
            del sys.modules[name]


def _import_comfree_backend_threadsafe():
    with _BACKEND_IMPORT_LOCK:
        try:
            return _import_comfree_backend()
        except ImportError as exc:
            if "partially initialized module" not in str(exc):
                raise
            _drop_partial_comfree_modules()
            return _import_comfree_backend()


def _import_mjwarp_backend_threadsafe():
    with _BACKEND_IMPORT_LOCK:
        return _import_mjwarp_backend()


# ---------------------------------------------------------------------------
# Geom-set helpers (mirrors demo_close_drawer_contact_curobo semantics)
# ---------------------------------------------------------------------------


def _env_geom_name(env, geom_id: int) -> str:
    """Resolve a geom id to its name using the env's cached name table."""
    model = env.sim.model
    table = getattr(model, "_geom_name2id", None)
    if table is not None:
        for name, gid in table.items():
            if int(gid) == int(geom_id):
                return name
    return mujoco.mj_id2name(
        getattr(model, "_model", model), mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)
    )


def _robot_geom_sets(env, panel):
    """Return ``(robot_geoms, ee_geoms, allowed_drawer_geoms)`` id sets.

    ``robot_geoms`` = ``robot0_*`` + ``gripper0_right_*`` collision geoms.
    ``ee_geoms``    = ``gripper0_right_*`` collision geoms.
    ``allowed_drawer_geoms`` = the target panel geom + ``{drawer}_door_g*`` geoms
    (the EE is *allowed* to touch these without it counting as a scene collision).
    """
    model = env.sim.model
    name2id = getattr(model, "_geom_name2id", None)
    if name2id is None:
        raw = getattr(model, "_model", model)
        name2id = {
            mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_GEOM, gid): gid
            for gid in range(int(raw.ngeom))
        }
    robot_geoms = {
        int(gid)
        for name, gid in name2id.items()
        if name.startswith("robot0_") and "collision" in name
    }
    ee_geoms = {
        int(gid)
        for name, gid in name2id.items()
        if name.startswith("gripper0_right_") and "collision" in name
    }
    robot_geoms.update(ee_geoms)
    drawer_name = env.drawer.name
    allowed_drawer_geoms = {
        int(gid)
        for name, gid in name2id.items()
        if name == panel.geom_name or name.startswith(f"{drawer_name}_door_g")
    }
    return robot_geoms, ee_geoms, allowed_drawer_geoms


def _strict_robot_drawer_geom_ids(env) -> set[int]:
    """Robot geom ids used by ``_strict_robot_drawer_penetration``."""
    model = env.sim.model
    name2id = getattr(model, "_geom_name2id", None)
    if name2id is None:
        raw = getattr(model, "_model", model)
        name2id = {
            mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_GEOM, gid): gid
            for gid in range(int(raw.ngeom))
        }
    return {
        int(gid)
        for name, gid in name2id.items()
        if (name.startswith("robot0_") or name.startswith("gripper0_"))
        and "collision" in name
    }


def _drawer_geom_ids(env):
    """Drawer door geom ids (excluding inflating handle/knob fixtures)."""
    model = env.sim.model
    raw = getattr(model, "_model", model)
    name2id = getattr(model, "_geom_name2id", None)
    if name2id is None:
        name2id = {
            mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_GEOM, gid): gid
            for gid in range(int(raw.ngeom))
        }
    drawer_name = env.drawer.name
    inflating = set()
    for name in name2id:
        if not (
            name.startswith(f"{drawer_name}_door_g")
            or name.startswith(f"{drawer_name}_door")
        ):
            continue
        low = name.lower()
        if "handle" in low or "knob" in low:
            inflating.add(name)
    return {
        int(gid)
        for name, gid in name2id.items()
        if (
            name.startswith(f"{drawer_name}_door_")
            or name.startswith(f"{drawer_name}_door")
        )
        and name not in inflating
    }


# ---------------------------------------------------------------------------
# Per-candidate collision report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateCollisionReport:
    """Collision / penetration read-out for a single staged candidate.

    Mirrors the tuple shape returned by the two ``env.sim`` checkers in
    ``demo_close_drawer_contact_curobo.py`` so the ``mink_q`` loop can consume it
    without branching on the backend.
    """

    max_penetration: float
    penetration_reason: str
    scene_collision_free: bool
    scene_reason: str
    # EE-site world pose after staging this candidate (recomposed from the full
    # scene so that ``_site_pose_for_arm_q`` no longer has to touch env.sim).
    ee_position: np.ndarray  # (3,) float64
    ee_rotation: np.ndarray  # (3, 3) float64


@dataclass(frozen=True)
class _MujocoReferenceReport:
    max_penetration: float
    penetration_reason: str
    scene_collision_free: bool
    scene_reason: str
    ee_position: np.ndarray
    ee_rotation: np.ndarray
    ncon: int


@dataclass(frozen=True)
class _PoolItem:
    scene: "FullSceneMjWarp"
    checker: "FullSceneCollisionChecker"


class _LazyPoolItem:
    """One lazily constructed worker-owned collision checker."""

    def __init__(self, factory: Callable[[], _PoolItem]) -> None:
        self._factory = factory
        self._item: _PoolItem | None = None
        self._lock = threading.Lock()

    def get(self) -> _PoolItem:
        item = self._item
        if item is not None:
            return item
        with self._lock:
            item = self._item
            if item is None:
                item = self._factory()
                self._item = item
            return item


# ---------------------------------------------------------------------------
# The staged, batched collision checker
# ---------------------------------------------------------------------------


class FullSceneCollisionChecker:
    """Batched checker that stages arm ``q`` candidates into independent worlds.

    Instantiate via :meth:`FullSceneMjWarp.checker`.  Each :meth:`submit` arm ``q``
    is written into the next free world (only the arm-joint qpos entries change —
    the drawer, base and the rest of the scene stay at the env's current state).
    After all candidates are staged, call :meth:`evaluate` once to run a single
    batched forward+collision, then read results per-world with
    :meth:`penetration` / :meth:`scene_collision`.
    """

    def __init__(
        self,
        scene: "FullSceneMjWarp",
        *,
        allowed_ee_geom_name: str | None,
        penetration_tolerance: float,
    ) -> None:
        self.scene = scene
        self.allowed_ee_geom_name = allowed_ee_geom_name
        self.penetration_tolerance = float(penetration_tolerance)
        self._staged: list[np.ndarray] = []

    # ----- staging ---------------------------------------------------------

    def submit(self, q_arm: np.ndarray) -> int:
        """Stage ``q_arm`` (7,) into the next free world.  Returns its world id."""
        idx = len(self._staged)
        if idx >= self.scene.nworld:
            raise IndexError(
                f"checker capacity exhausted ({self.scene.nworld} worlds staged)"
            )
        self._staged.append(np.asarray(q_arm, dtype=np.float64).reshape(7).copy())
        return idx

    def reset(self) -> None:
        """Drop all staged candidates without re-allocating the backend."""
        self._staged.clear()

    def refresh_base(self) -> None:
        """Re-read env's current qpos/qvel as the per-world baseline.

        The ``env.sim`` checkers they replace read env's *live* state on every
        call, so the drawer (and any other joint) may have moved since the
        checker was built.  Calling this once before :meth:`evaluate` restores
        that semantics with a single O(nq) read that covers every world.
        """
        scene = self.scene
        scene._raw_data = getattr(scene.env.sim.data, "_data", scene.env.sim.data)
        scene.data_cpu = scene._raw_data
        base = np.asarray(scene._raw_data.qpos, dtype=np.float64).reshape(-1).copy()
        base_qvel = (
            np.asarray(scene._raw_data.qvel, dtype=np.float64).reshape(-1).copy()
        )
        scene._base_qpos = torch.as_tensor(
            base, dtype=torch.float32, device=scene.torch_device
        )
        scene._base_qvel = torch.as_tensor(
            base_qvel, dtype=torch.float32, device=scene.torch_device
        )
        qpos = _backend_tensor(scene.data.qpos)
        qpos[:] = scene._base_qpos.unsqueeze(0)
        qvel = _backend_tensor(scene.data.qvel)
        qvel[:] = scene._base_qvel.unsqueeze(0)

    # ----- batched evaluation ---------------------------------------------

    def evaluate(self) -> None:
        """Write every staged candidate into its own world and run forward."""
        scene = self.scene
        count = len(self._staged)
        if count == 0:
            self._out = _EvalOutput.empty(scene)
            return
        # Refresh the baseline from env so a moved drawer / scene is honoured
        # (mirrors the per-call env.read in the checkers we replace).
        self.refresh_base()
        qpos = _backend_tensor(scene.data.qpos)  # (nworld, nq)
        # Baseline state = env's current qpos, tiled across worlds.
        base = scene._base_qpos.to(qpos.device, dtype=qpos.dtype)  # (nq,)
        qpos[:count] = base.unsqueeze(0)
        addrs = scene._arm_qpos_addrs.to(qpos.device, dtype=torch.long)  # (7,)
        q_arms = torch.as_tensor(
            np.asarray(self._staged, dtype=np.float32),
            device=qpos.device,
            dtype=qpos.dtype,
        )  # (count, 7)
        rows = torch.arange(count, device=qpos.device, dtype=torch.long)
        qpos[rows[:, None], addrs[None, :]] = q_arms
        qvel = _backend_tensor(scene.data.qvel)
        base_qvel = scene._base_qvel.to(qvel.device, dtype=qvel.dtype)
        qvel[:count] = base_qvel.unsqueeze(0)
        scene.forward()
        self._out = _read_eval_output(
            scene,
            count,
            self.allowed_ee_geom_name,
            self.penetration_tolerance,
        )

    # ----- per-world readback ---------------------------------------------

    def count(self) -> int:
        return len(self._staged)

    def penetration(self, world_id: int) -> tuple[float, str]:
        """Per-world max robot-drawer penetration (m) and worst-pair reason."""
        out = self._out
        return float(out.robot_drawer_pen[world_id]), str(
            out.robot_drawer_reason[world_id]
        )

    def scene_collision(self, world_id: int) -> tuple[bool, str]:
        """Per-world robot-vs-scene collision result + reason string."""
        out = self._out
        return bool(out.scene_ok[world_id]), str(out.scene_reason[world_id])

    def ee_pose(self, world_id: int) -> tuple[np.ndarray, np.ndarray]:
        """EE-site world position (3,) and rotation (3,3) for this candidate."""
        out = self._out
        return out.ee_pos[world_id].copy(), out.ee_rot[world_id].copy()


class FullSceneCollisionCheckerPool:
    """Pool of one-world full-scene mjwarp checkers.

    This is the worker-owned variant used by the threaded mink path: each worker
    gets its own mjwarp/comfree ``Data`` object, so contact readout never shares a
    mutable backend environment with another worker.
    """

    def __init__(
        self,
        items: Sequence[_PoolItem | _LazyPoolItem],
        *,
        debug_compare_env: bool = False,
        debug_limit: int = 12,
    ) -> None:
        if not items:
            raise ValueError(
                "FullSceneCollisionCheckerPool requires at least one checker"
            )
        self._items = tuple(items)
        self.num_workers = len(self._items)
        self.debug_compare_env = bool(debug_compare_env)
        self.debug_limit = max(int(debug_limit), 0)
        self._debug_printed = 0
        self._available = queue.Queue()
        for index in range(self.num_workers):
            self._available.put(index)

    @classmethod
    def from_env(
        cls,
        env,
        *,
        arm_joint_names: Sequence[str],
        frame_name: str,
        panel,
        num_workers: int,
        allowed_ee_geom_name: str | None = None,
        penetration_tolerance: float = 0.0,
        device: str = "cuda:0",
        nconmax_per_env: int = 256,
        njmax_per_env: int = 1024,
        prefer_comfree: bool = True,
        contact_stiffness: float = 0.2,
        contact_damping: float = 0.001,
        debug_compare_env: bool = False,
        debug_limit: int = 12,
        lazy: bool = True,
        nworld_per_worker: int = 1,
    ) -> "FullSceneCollisionCheckerPool":
        count = max(int(num_workers), 1)
        worlds_per_worker = max(int(nworld_per_worker), 1)
        items: list[_PoolItem | _LazyPoolItem] = []

        def _make_item() -> _PoolItem:
            scene = FullSceneMjWarp(
                env,
                arm_joint_names=arm_joint_names,
                frame_name=frame_name,
                panel=panel,
                nworld=worlds_per_worker,
                device=device,
                nconmax_per_env=nconmax_per_env,
                njmax_per_env=njmax_per_env,
                prefer_comfree=prefer_comfree,
                contact_stiffness=contact_stiffness,
                contact_damping=contact_damping,
            )
            return _PoolItem(
                scene=scene,
                checker=scene.checker(
                    allowed_ee_geom_name=allowed_ee_geom_name,
                    penetration_tolerance=penetration_tolerance,
                ),
            )

        for _ in range(count):
            if lazy:
                items.append(_LazyPoolItem(_make_item))
            else:
                items.append(_make_item())
        return cls(
            items,
            debug_compare_env=debug_compare_env,
            debug_limit=debug_limit,
        )

    def _item_at(self, index: int) -> _PoolItem:
        slot = self._items[int(index)]
        if isinstance(slot, _LazyPoolItem):
            return slot.get()
        return slot

    def constructed_worker_count(self) -> int:
        """Number of worker backend envs actually constructed so far."""
        count = 0
        for slot in self._items:
            if isinstance(slot, _LazyPoolItem):
                count += int(slot._item is not None)
            else:
                count += 1
        return count

    def reset(self) -> None:
        for index in range(self.num_workers):
            slot = self._items[index]
            if isinstance(slot, _LazyPoolItem) and slot._item is None:
                continue
            item = self._item_at(index)
            item.checker.reset()

    def warmup(self, *, parallel: bool = True, show_progress: bool = True) -> None:
        """Eagerly construct every lazy pool item.

        Without this, the *first* outer worker to borrow each slot pays the
        per-item ``_build_backend`` cost (which sits behind a module-level
        ``_BACKEND_BUILD_LOCK`` because the cone swap mutates the shared
        ``model_cpu``). If N outer workers all borrow simultaneously, N-1
        block on the lock and the outer parallel dispatch appears to hang.
        Do it up-front with a progress bar so the stall is visible and
        happens once.
        """
        lazy_slots = [
            (index, slot)
            for index, slot in enumerate(self._items)
            if isinstance(slot, _LazyPoolItem) and slot._item is None
        ]
        if not lazy_slots:
            return
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm as _tqdm

                pbar = _tqdm(
                    total=len(lazy_slots),
                    desc=f"mjwarp warmup (workers={len(lazy_slots)})",
                    unit="env",
                    file=sys.__stdout__,
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:
                pbar = None

        def _init(pair):
            _, slot = pair
            slot.get()
            return True

        try:
            if parallel and len(lazy_slots) > 1:
                # Backend build is serialized on _BACKEND_BUILD_LOCK, but the
                # ThreadPool still helps by overlapping the surrounding
                # post-build device transfers (torch tensor allocation, mask
                # precomputation) that run outside the lock.
                import concurrent.futures as _cf

                with _cf.ThreadPoolExecutor(max_workers=len(lazy_slots)) as executor:
                    for _ in executor.map(_init, lazy_slots):
                        if pbar is not None:
                            pbar.update(1)
            else:
                for pair in lazy_slots:
                    _init(pair)
                    if pbar is not None:
                        pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

    def close(self) -> None:
        """Release worker-owned backend objects before later CUDA users run."""
        for slot in self._items:
            item = slot._item if isinstance(slot, _LazyPoolItem) else slot
            if item is None:
                continue
            try:
                item.checker.reset()
            except Exception:
                pass
            try:
                item.scene.close()
            except Exception:
                pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        gc.collect()

    def _evaluate_with_item(
        self, item: _PoolItem, q_arm: np.ndarray
    ) -> CandidateCollisionReport:
        return self._evaluate_batch_with_item(item, [q_arm])[0]

    def _evaluate_batch_with_item(
        self, item: _PoolItem, q_arms: Sequence[np.ndarray]
    ) -> list[CandidateCollisionReport]:
        q_arms = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in q_arms]
        if not q_arms:
            return []
        if len(q_arms) > int(item.scene.nworld):
            raise ValueError(
                f"batch size {len(q_arms)} exceeds checker capacity {int(item.scene.nworld)}"
            )
        checker = item.checker
        checker.reset()
        for q_arm in q_arms:
            checker.submit(q_arm)
        checker.evaluate()
        reports: list[CandidateCollisionReport] = []
        for world_id in range(len(q_arms)):
            max_pen, pen_reason = checker.penetration(world_id)
            scene_ok, scene_reason = checker.scene_collision(world_id)
            ee_pos, ee_rot = checker.ee_pose(world_id)
            reports.append(
                CandidateCollisionReport(
                    max_penetration=float(max_pen),
                    penetration_reason=str(pen_reason),
                    scene_collision_free=bool(scene_ok),
                    scene_reason=str(scene_reason),
                    ee_position=np.asarray(ee_pos, dtype=np.float64).reshape(3),
                    ee_rotation=np.asarray(ee_rot, dtype=np.float64).reshape(3, 3),
                )
            )
        return reports

    def evaluate_candidate_threadsafe(
        self, q_arm: np.ndarray
    ) -> CandidateCollisionReport:
        """Evaluate one candidate using an exclusively borrowed checker."""
        q_arm = np.asarray(q_arm, dtype=np.float64).reshape(7).copy()
        worker_id = self._available.get()
        try:
            return self._evaluate_with_item(self._item_at(int(worker_id)), q_arm)
        finally:
            self._available.put(worker_id)

    def evaluate_candidates_threadsafe(
        self, q_arms: Sequence[np.ndarray]
    ) -> list[CandidateCollisionReport]:
        """Evaluate candidates while allowing concurrent callers to share this pool."""
        q_arms = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in q_arms]
        if not q_arms:
            return []
        worker_id = self._available.get()
        try:
            item = self._item_at(int(worker_id))
            capacity = max(int(item.scene.nworld), 1)
            reports: list[CandidateCollisionReport] = []
            for start in range(0, len(q_arms), capacity):
                reports.extend(
                    self._evaluate_batch_with_item(
                        item, q_arms[start : start + capacity]
                    )
                )
        finally:
            self._available.put(worker_id)
        if self.debug_compare_env and self._debug_printed < self.debug_limit:
            self._debug_compare_reports(q_arms, reports)
        return reports

    def evaluate_candidates(
        self, q_arms: Sequence[np.ndarray]
    ) -> list[CandidateCollisionReport]:
        q_arms = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in q_arms]
        reports: list[CandidateCollisionReport | None] = [None] * len(q_arms)
        if not q_arms:
            return []

        def _run(worker_id: int, indexed_qs: list[tuple[int, np.ndarray]]) -> None:
            item = self._item_at(worker_id)
            capacity = max(int(item.scene.nworld), 1)
            for start in range(0, len(indexed_qs), capacity):
                chunk = indexed_qs[start : start + capacity]
                chunk_reports = self._evaluate_batch_with_item(
                    item, [q_arm for _, q_arm in chunk]
                )
                for (candidate_index, _), report in zip(chunk, chunk_reports):
                    reports[candidate_index] = report

        buckets: list[list[tuple[int, np.ndarray]]] = [
            [] for _ in range(self.num_workers)
        ]
        for index, q_arm in enumerate(q_arms):
            buckets[index % self.num_workers].append((index, q_arm))

        active = [(i, bucket) for i, bucket in enumerate(buckets) if bucket]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = [
                executor.submit(_run, worker_id, bucket) for worker_id, bucket in active
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        missing = [index for index, report in enumerate(reports) if report is None]
        if missing:
            raise RuntimeError(
                f"collision checker pool produced no report for candidates {missing}"
            )
        final_reports = [report for report in reports if report is not None]
        if self.debug_compare_env and self._debug_printed < self.debug_limit:
            self._debug_compare_reports(q_arms, final_reports)
        return final_reports

    def _debug_compare_reports(
        self,
        q_arms: Sequence[np.ndarray],
        reports: Sequence[CandidateCollisionReport],
    ) -> None:
        """Compare worker mjwarp reports against the shared MuJoCo env.

        This intentionally runs after worker threads finish because the MuJoCo
        env is shared and its qpos/contact buffers are mutable.
        """
        if not q_arms or not reports:
            return
        first_item = self._item_at(0)
        scene = first_item.scene
        remaining = self.debug_limit - self._debug_printed
        for local_index, (q_arm, report) in enumerate(zip(q_arms, reports)):
            if remaining <= 0:
                break
            ref = _mujoco_reference_report(
                scene.env,
                scene.panel,
                scene.arm_joint_names,
                q_arm,
                scene.frame_name,
                allowed_ee_geom_name=first_item.checker.allowed_ee_geom_name,
                penetration_tolerance=first_item.checker.penetration_tolerance,
            )
            pos_delta = float(
                np.linalg.norm(
                    np.asarray(report.ee_position, dtype=np.float64).reshape(3)
                    - ref.ee_position
                )
            )
            rot_delta = _rotation_delta(
                np.asarray(report.ee_rotation, dtype=np.float64).reshape(3, 3),
                ref.ee_rotation,
            )
            pen_delta = abs(float(report.max_penetration) - float(ref.max_penetration))
            mismatch = (
                bool(report.scene_collision_free) != bool(ref.scene_collision_free)
                or pen_delta > 1e-4
                or pos_delta > 1e-4
                or rot_delta > 1e-3
            )
            if mismatch or self._debug_printed == 0:
                print(
                    "[full_scene_mjwarp][debug] "
                    f"candidate={local_index} backend={scene.backend_kind} "
                    f"mismatch={bool(mismatch)} pos_delta={pos_delta:.6g} "
                    f"rot_delta={rot_delta:.6g} pen_delta={pen_delta:.6g}",
                    file=sys.__stdout__,
                    flush=True,
                )
                print(
                    "[full_scene_mjwarp][debug] "
                    f"mjwarp scene_ok={bool(report.scene_collision_free)} "
                    f"scene_reason={report.scene_reason} "
                    f"pen={float(report.max_penetration):.6g} "
                    f"pen_reason={report.penetration_reason} "
                    f"ee_pos={np.asarray(report.ee_position).tolist()}",
                    file=sys.__stdout__,
                    flush=True,
                )
                print(
                    "[full_scene_mjwarp][debug] "
                    f"mujoco ncon={int(ref.ncon)} scene_ok={bool(ref.scene_collision_free)} "
                    f"scene_reason={ref.scene_reason} "
                    f"pen={float(ref.max_penetration):.6g} "
                    f"pen_reason={ref.penetration_reason} "
                    f"ee_pos={ref.ee_position.tolist()}",
                    file=sys.__stdout__,
                    flush=True,
                )
                self._debug_printed += 1
                remaining -= 1


# ---------------------------------------------------------------------------
# Full-scene mjwarp wrapper
# ---------------------------------------------------------------------------


@dataclass
class _EvalOutput:
    robot_drawer_pen: np.ndarray  # (nworld,) float64
    robot_drawer_reason: list[str]  # (nworld,)
    scene_ok: np.ndarray  # (nworld,) bool
    scene_reason: list[str]  # (nworld,)
    ee_pos: np.ndarray  # (nworld, 3) float64
    ee_rot: np.ndarray  # (nworld, 3, 3) float64

    @staticmethod
    def empty(scene: "FullSceneMjWarp") -> "_EvalOutput":
        n = scene.nworld
        return _EvalOutput(
            robot_drawer_pen=np.zeros(n, dtype=np.float64),
            robot_drawer_reason=[""] * n,
            scene_ok=np.ones(n, dtype=bool),
            scene_reason=["collision_free"] * n,
            ee_pos=np.zeros((n, 3), dtype=np.float64),
            ee_rot=np.tile(np.eye(3, dtype=np.float64), (n, 1, 1)),
        )


class FullSceneMjWarp:
    """Full robot + full scene cloned into an nworld mjwarp/comfree Data.

    Builds the *complete* env model (not a stripped floating-hand contact scene)
    so that the standard ``env.sim`` collision semantics — robot geoms, gripper EE
    geoms, drawer door geoms and kitchen fixture geoms — are all present and can
    be queried per-world.
    """

    def __init__(
        self,
        env,
        *,
        arm_joint_names: Sequence[str],
        frame_name: str,
        panel,
        nworld: int,
        device: str = "cuda:0",
        nconmax_per_env: int = 256,
        njmax_per_env: int = 1024,
        prefer_comfree: bool = True,
        contact_stiffness: float = 0.2,
        contact_damping: float = 0.001,
    ) -> None:
        self.env = env
        self.panel = panel
        self.arm_joint_names = tuple(arm_joint_names)
        self.frame_name = str(frame_name)
        self.nworld = int(nworld)
        self.device = device
        self.nconmax_per_env = int(nconmax_per_env)
        self.njmax_per_env = int(njmax_per_env)
        self.prefer_comfree = bool(prefer_comfree)
        self.contact_stiffness = float(contact_stiffness)
        self.contact_damping = float(contact_damping)

        self._raw_model = getattr(env.sim.model, "_model", env.sim.model)
        self._raw_data = getattr(env.sim.data, "_data", env.sim.data)
        self.model_cpu = self._raw_model
        self.data_cpu = self._raw_data

        # Arm-joint qpos addresses inside the *full* model (all 7 are hinge
        # joints → one qpos slot each).
        self._arm_qpos_addrs = torch.as_tensor(
            [
                int(env.sim.model.get_joint_qpos_addr(name))
                for name in self.arm_joint_names
            ],
            dtype=torch.long,
        )
        # EE site id in the full model.
        self._site_id = int(
            mujoco.mj_name2id(
                self._raw_model, mujoco.mjtObj.mjOBJ_SITE, self.frame_name
            )
        )

        # Named geom sets for the two checkers.
        robot_geoms, ee_geoms, allowed_drawer_geoms = _robot_geom_sets(env, panel)
        strict_robot_drawer_geoms = _strict_robot_drawer_geom_ids(env)
        drawer_geoms = _drawer_geom_ids(env)
        ngeom = int(self._raw_model.ngeom)
        self._mask_robot = self._bool_mask(robot_geoms, ngeom)
        self._mask_robot_drawer = self._bool_mask(strict_robot_drawer_geoms, ngeom)
        self._mask_ee = self._bool_mask(ee_geoms, ngeom)
        self._mask_drawer = self._bool_mask(drawer_geoms, ngeom)
        self._mask_allowed_drawer = self._bool_mask(allowed_drawer_geoms, ngeom)
        self._allowed_drawer_geoms = allowed_drawer_geoms

        self._build_backend()

    # ----- construction helpers -------------------------------------------

    @staticmethod
    def _bool_mask(ids: set[int], ngeom: int) -> torch.Tensor:
        mask = torch.zeros(ngeom, dtype=torch.bool)
        if ids:
            mask[torch.as_tensor(sorted(ids), dtype=torch.long)] = True
        return mask

    def _build_backend(self) -> None:
        nworld = self.nworld
        backend_kind = None
        cfg_stiffness = self.contact_stiffness
        cfg_damping = self.contact_damping
        with _BACKEND_BUILD_LOCK:
            original_cone = int(self.model_cpu.opt.cone)
            try:
                # comfree_core/mjwarp full contact readout currently requires
                # pyramidal contact constraints. Convert only the model snapshot
                # uploaded to the backend, then restore the live MuJoCo env.
                # This block touches shared env.sim model/data, so lazy worker
                # construction must be serialized even though later evaluations
                # run on independent backend Data objects.
                self.model_cpu.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
                mujoco.mj_forward(self.model_cpu, self.data_cpu)

                # Size the contact buffer from the actual env state we clone
                # (with a margin): the full kitchen scene can carry hundreds of
                # simultaneous contacts at rest, and staging the arm into the
                # drawer can grow that count. Passing anything below ``mjd.ncon``
                # to put_data raises, so we bump the requested floor to fit the
                # live pyramidal data.
                baseline_ncon = int(getattr(self._raw_data, "ncon", 0))
                baseline_nefc = int(getattr(self._raw_data, "nefc", 0))
                nconmax = max(int(self.nconmax_per_env), int(baseline_ncon * 1.5) + 128)
                # nefc on the full kitchen scene is large (joint limits,
                # tendons, equality constraints) — njmax must cover it or
                # put_data raises.
                njmax = max(int(self.njmax_per_env), int(baseline_nefc * 1.5) + 256)
                if self.prefer_comfree:
                    try:
                        (
                            self.wp,
                            self.cfwarp,
                            step_fn,
                        ) = _import_comfree_backend_threadsafe()
                        self.model = self.cfwarp.put_model(
                            self.model_cpu,
                            comfree_stiffness=cfg_stiffness,
                            comfree_damping=cfg_damping,
                        )
                        self.data = self.cfwarp.put_data(
                            self.model_cpu,
                            self.data_cpu,
                            nworld=nworld,
                            nconmax=nconmax,
                            njmax=njmax,
                        )
                        self._step_fn = lambda: step_fn(self.model, self.data)
                        self._forward_fn = lambda: self.cfwarp.forward(
                            self.model, self.data
                        )
                        backend_kind = "comfree"
                    except Exception as exc:
                        sys.stderr.write(
                            f"[full_scene_mjwarp] Comfree backend unavailable, "
                            f"falling back to native mjwarp: {exc!r}\n"
                        )
                        sys.stderr.flush()
                if backend_kind is None:
                    self.wp, mjwarp, step_fn = _import_mjwarp_backend_threadsafe()
                    self.cfwarp = mjwarp
                    self.model = mjwarp.put_model(self.model_cpu)
                    self.data = mjwarp.put_data(
                        self.model_cpu,
                        self.data_cpu,
                        nworld=nworld,
                        nconmax=nconmax,
                        njmax=njmax,
                    )
                    self._step_fn = lambda: step_fn(self.model, self.data)
                    self._forward_fn = lambda: mjwarp.forward(self.model, self.data)
                    backend_kind = "mjwarp"
            finally:
                self.model_cpu.opt.cone = original_cone
                mujoco.mj_forward(self.model_cpu, self.data_cpu)
        self.backend_kind = backend_kind

        # Precompute the baseline qpos (env's current state) on the backend
        # device; per-world staging only overwrites the 7 arm-joint slots.
        self.torch_device = torch.device(self.device)
        base = np.asarray(self._raw_data.qpos, dtype=np.float64).reshape(-1).copy()
        base_qvel = np.asarray(self._raw_data.qvel, dtype=np.float64).reshape(-1).copy()
        self._base_qpos = torch.as_tensor(
            base, dtype=torch.float32, device=self.torch_device
        )
        self._base_qvel = torch.as_tensor(
            base_qvel, dtype=torch.float32, device=self.torch_device
        )
        qpos = _backend_tensor(self.data.qpos)
        qpos[:] = self._base_qpos.unsqueeze(0)
        qvel = _backend_tensor(self.data.qvel)
        qvel[:] = self._base_qvel.unsqueeze(0)

    # ----- public API ------------------------------------------------------

    def forward(self) -> None:
        self._forward_fn()

    def step(self) -> None:
        self._step_fn()

    def checker(
        self,
        *,
        allowed_ee_geom_name: str | None = None,
        penetration_tolerance: float = 0.0,
    ) -> FullSceneCollisionChecker:
        """Return a staged batched checker bound to this scene."""
        return FullSceneCollisionChecker(
            self,
            allowed_ee_geom_name=allowed_ee_geom_name,
            penetration_tolerance=penetration_tolerance,
        )

    def close(self) -> None:
        """Best-effort deterministic release of backend device allocations."""
        try:
            if hasattr(self, "wp"):
                self.wp.synchronize()
        except Exception:
            pass
        for name in (
            "_step_fn",
            "_forward_fn",
            "data",
            "model",
            "_base_qpos",
            "_base_qvel",
            "_arm_qpos_addrs",
            "_mask_robot",
            "_mask_robot_drawer",
            "_mask_ee",
            "_mask_drawer",
            "_mask_allowed_drawer",
        ):
            if hasattr(self, name):
                try:
                    setattr(self, name, None)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Batched contact read-out (GPU), mirroring the two env.sim checkers
# ---------------------------------------------------------------------------


def _read_eval_output(
    scene: FullSceneMjWarp,
    count: int,
    allowed_ee_geom_name: str | None,
    penetration_tolerance: float,
) -> _EvalOutput:
    """Read contacts once off the batched data and split per-world answers."""
    ngeom = int(scene.model_cpu.ngeom)

    contact = scene.data.contact
    dist_raw = _backend_tensor(contact.dist)
    dist = dist_raw.reshape(-1)
    device = dist.device
    geom = _contact_geom_tensor(contact).to(device).long().reshape(-1, 2)
    worldid = (
        _contact_worldid_tensor(contact, dist, scene.nworld)
        .to(device)
        .long()
        .reshape(-1)
    )
    total_contacts = min(int(dist.numel()), int(geom.shape[0]), int(worldid.numel()))
    dist = dist[:total_contacts]
    geom = geom[:total_contacts]
    worldid = worldid[:total_contacts]
    active = _active_contact_mask(
        scene.data,
        dist_raw,
        worldid,
        scene.nworld,
        total_contacts,
        device,
    )

    mask_robot = scene._mask_robot.to(device)  # (ngeom,)
    mask_robot_drawer = scene._mask_robot_drawer.to(device)  # (ngeom,)
    mask_ee = scene._mask_ee.to(device)  # (ngeom,)
    mask_drawer = scene._mask_drawer.to(device)  # (ngeom,)
    mask_allowed_drawer = scene._mask_allowed_drawer.to(device)  # (ngeom,)

    # The EE geom we are *allowed* to touch the drawer with.  If the caller
    # names one, only that single EE geom is exempted; otherwise every
    # ``gripper0_right_*`` EE geom is (matching _check_arm_q_collision).
    if allowed_ee_geom_name is not None:
        allowed_ee_ids = {
            int(
                mujoco.mj_name2id(
                    scene._raw_model, mujoco.mjtObj.mjOBJ_GEOM, allowed_ee_geom_name
                )
            )
        }
    else:
        allowed_ee_ids = None
    if allowed_ee_ids:
        mask_allowed_ee = FullSceneMjWarp._bool_mask(allowed_ee_ids, ngeom).to(device)
    else:
        mask_allowed_ee = mask_ee

    valid = active & (geom[:, 0] >= 0) & (geom[:, 1] >= 0)

    safe_g0 = geom[:, 0].clamp(0, ngeom - 1)
    safe_g1 = geom[:, 1].clamp(0, ngeom - 1)

    # ---- robot-drawer penetration (_strict_robot_drawer_penetration) ------
    pair_rd = (mask_robot_drawer[safe_g0] & mask_drawer[safe_g1]) | (
        mask_drawer[safe_g0] & mask_robot_drawer[safe_g1]
    )
    pen_rd = torch.where(
        valid & pair_rd,
        torch.clamp(-dist, min=0.0),
        torch.zeros_like(dist),
    )
    pen_per_world = torch.zeros(scene.nworld, device=device, dtype=torch.float64)
    if pen_rd.any():
        pen_per_world.scatter_reduce_(
            0,
            worldid.clamp(0, scene.nworld - 1),
            pen_rd.to(torch.float64),
            reduce="amax",
            include_self=True,
        )

    # ---- robot-vs-scene collision (_check_arm_q_collision) ---------------
    # For each active contact involving exactly one robot geom, decide whether
    # it is the *allowed* EE/drawer contact or a real scene collision.
    g0_robot = mask_robot[safe_g0]
    g1_robot = mask_robot[safe_g1]
    g0_ee_allowed = mask_allowed_ee[safe_g0]
    g1_ee_allowed = mask_allowed_ee[safe_g1]
    g0_adrawer = mask_allowed_drawer[safe_g0]
    g1_adrawer = mask_allowed_drawer[safe_g1]

    robot_involved = valid & (g0_robot ^ g1_robot)
    # "allowed" = robot-EE geom touching allowed-drawer geom, in either order.
    allowed_contact = valid & (
        (g0_ee_allowed & g1_adrawer) | (g1_ee_allowed & g0_adrawer)
    )
    bad = (
        robot_involved
        & ~allowed_contact
        & (dist < -max(float(penetration_tolerance), 0.0))
    )
    scene_ok = torch.ones(scene.nworld, device=device, dtype=torch.bool)
    if bad.any():
        bad_worlds = worldid[bad].clamp(0, scene.nworld - 1).unique()
        scene_ok[bad_worlds] = False

    # ---- EE-site world pose per world ------------------------------------
    site_xpos = _backend_tensor(scene.data.site_xpos)  # (nworld, nsite, 3)
    site_xmat = _backend_tensor(scene.data.site_xmat)  # (nworld, nsite, 3, 3)
    sid = scene._site_id
    ee_pos_t = site_xpos[:, sid, :]  # (nworld, 3)
    ee_rot_t = site_xmat[:, sid, :, :]  # (nworld, 3, 3)

    # Move only the evaluated worlds to CPU as numpy.
    pen_np = pen_per_world[:count].detach().cpu().numpy().astype(np.float64)
    ok_np = scene_ok[:count].detach().cpu().numpy().astype(bool)
    ee_pos_np = ee_pos_t[:count].detach().cpu().numpy().astype(np.float64)
    ee_rot_np = ee_rot_t[:count].detach().cpu().numpy().astype(np.float64)

    # Build human-readable worst-case reasons by scanning the active contacts
    # on the CPU (small absolute count; only needed for logging / reason text).
    robot_drawer_reason = ["collision_free"] * count
    scene_reason = ["collision_free"] * count
    if bool(active.any().item()):
        _fill_reasons(
            count,
            active,
            pen_rd,
            bad,
            dist,
            geom,
            worldid,
            scene._raw_model,
            scene._raw_data,
            pen_np,
            ok_np,
            robot_drawer_reason,
            scene_reason,
        )

    return _EvalOutput(
        robot_drawer_pen=pen_np,
        robot_drawer_reason=robot_drawer_reason,
        scene_ok=ok_np,
        scene_reason=scene_reason,
        ee_pos=ee_pos_np,
        ee_rot=ee_rot_np,
    )


def _rotation_delta(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = np.asarray(rot_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        rot_b, dtype=np.float64
    ).reshape(3, 3)
    c = 0.5 * (float(np.trace(rel)) - 1.0)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def _mujoco_reference_report(
    env,
    panel,
    arm_joint_names: Sequence[str],
    q_arm: np.ndarray,
    frame_name: str,
    *,
    allowed_ee_geom_name: str | None,
    penetration_tolerance: float,
) -> _MujocoReferenceReport:
    """Evaluate one candidate with the shared MuJoCo env semantics."""
    model = env.sim.model
    data = env.sim.data
    qpos_saved = np.asarray(data.qpos, dtype=np.float64).copy()
    qvel_saved = np.asarray(data.qvel, dtype=np.float64).copy()
    geom_id_to_name = {
        int(geom_id): name for name, geom_id in model._geom_name2id.items()
    }
    try:
        for joint_name, value in zip(arm_joint_names, q_arm):
            data.qpos[model.get_joint_qpos_addr(joint_name)] = float(value)
        env.sim.forward()

        site_id = int(model.site_name2id(frame_name))
        ee_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3).copy()
        ee_rot = (
            np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
        )

        robot_geoms, ee_geoms, allowed_drawer_geoms = _robot_geom_sets(env, panel)
        allowed_ee_geoms = set(ee_geoms)
        if (
            allowed_ee_geom_name is not None
            and allowed_ee_geom_name in model._geom_name2id
        ):
            allowed_ee_geoms = {int(model.geom_name2id(allowed_ee_geom_name))}

        robot_drawer_geoms = _strict_robot_drawer_geom_ids(env)
        drawer_geoms = _drawer_geom_ids(env)

        max_pen = 0.0
        pen_reason = "collision_free"
        scene_ok = True
        scene_reason = "collision_free"
        for contact_idx in range(int(data.ncon)):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            dist = float(contact.dist)
            drawer_pair = (geom1 in robot_drawer_geoms and geom2 in drawer_geoms) or (
                geom2 in robot_drawer_geoms and geom1 in drawer_geoms
            )
            if drawer_pair:
                penetration = max(0.0, -dist)
                if penetration > max_pen:
                    max_pen = penetration
                    pen_reason = (
                        f"{geom_id_to_name.get(geom1, str(geom1))}"
                        f"--{geom_id_to_name.get(geom2, str(geom2))}:"
                        f"dist={dist:.6f}"
                    )

            if not scene_ok:
                continue
            if geom1 not in robot_geoms and geom2 not in robot_geoms:
                continue
            if geom1 in robot_geoms and geom2 in robot_geoms:
                continue
            robot_geom = geom1 if geom1 in robot_geoms else geom2
            other_geom = geom2 if robot_geom == geom1 else geom1
            if robot_geom in allowed_ee_geoms and other_geom in allowed_drawer_geoms:
                continue
            if dist >= -max(float(penetration_tolerance), 0.0):
                continue
            scene_ok = False
            scene_reason = (
                f"collision:{geom_id_to_name.get(robot_geom, str(robot_geom))}"
                f"--{geom_id_to_name.get(other_geom, str(other_geom))}:"
                f"dist={dist:.6f}"
            )

        return _MujocoReferenceReport(
            max_penetration=float(max_pen),
            penetration_reason=str(pen_reason),
            scene_collision_free=bool(scene_ok),
            scene_reason=str(scene_reason),
            ee_position=ee_pos,
            ee_rotation=ee_rot,
            ncon=int(data.ncon),
        )
    finally:
        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        env.sim.forward()


def _contact_geom_tensor(contact) -> torch.Tensor:
    """Return contact geom pairs across comfree/mjwarp field variants."""
    if hasattr(contact, "geom"):
        return _backend_tensor(contact.geom)
    if hasattr(contact, "geom1") and hasattr(contact, "geom2"):
        g1 = _backend_tensor(contact.geom1).reshape(-1)
        g2 = _backend_tensor(contact.geom2).reshape(-1)
        return torch.stack([g1, g2], dim=-1)
    raise AttributeError("backend contact object exposes neither geom nor geom1/geom2")


def _contact_worldid_tensor(contact, dist: torch.Tensor, nworld: int) -> torch.Tensor:
    """Return per-contact world ids, falling back only for single-world data."""
    value = getattr(contact, "worldid", getattr(contact, "world_id", None))
    if value is not None:
        return _backend_tensor(value)
    if int(nworld) == 1:
        return torch.zeros_like(dist, dtype=torch.long)
    raise AttributeError("backend contact object does not expose worldid/world_id")


def _active_contact_count(data, total_contacts: int) -> int:
    """Read the active flat contact count if available; otherwise trust masks."""
    for attr in ("nacon", "ncon"):
        if not hasattr(data, attr):
            continue
        value = getattr(data, attr)
        try:
            arr = _backend_tensor(value).reshape(-1)
            if int(arr.numel()) > 0:
                return min(int(arr[0].item()), int(total_contacts))
        except Exception:
            try:
                return min(int(value), int(total_contacts))
            except Exception:
                pass
    return int(total_contacts)


def _contact_count_tensor(
    data, nworld: int, device: torch.device
) -> torch.Tensor | None:
    """Return per-world active contact counts when the backend exposes them."""
    for attr in ("ncon", "nacon"):
        if not hasattr(data, attr):
            continue
        value = getattr(data, attr)
        try:
            arr = _backend_tensor(value).to(device).reshape(-1).long()
        except Exception:
            try:
                scalar = int(value)
            except Exception:
                continue
            return torch.full((int(nworld),), scalar, device=device, dtype=torch.long)
        if int(arr.numel()) >= int(nworld):
            return arr[: int(nworld)]
        if int(arr.numel()) == 1:
            return torch.full(
                (int(nworld),), int(arr[0].item()), device=device, dtype=torch.long
            )
    return None


def _active_contact_mask(
    data,
    dist_raw: torch.Tensor,
    worldid: torch.Tensor,
    nworld: int,
    total_contacts: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a flat active-contact mask for scalar or per-world contact buffers."""
    total_contacts = int(total_contacts)
    counts = _contact_count_tensor(data, int(nworld), device)
    if counts is None:
        return torch.ones(total_contacts, device=device, dtype=torch.bool)

    flat_index = torch.arange(total_contacts, device=device, dtype=torch.long)
    if int(nworld) == 1:
        return flat_index < int(counts[0].item())

    slots_per_world = 0
    shape = tuple(int(v) for v in getattr(dist_raw, "shape", ()))
    if len(shape) >= 2 and shape[0] == int(nworld):
        slots_per_world = int(np.prod(shape[1:]))
    elif total_contacts % int(nworld) == 0:
        slots_per_world = total_contacts // int(nworld)

    if slots_per_world > 0:
        local_index = flat_index % int(slots_per_world)
        safe_world = worldid[:total_contacts].clamp(0, int(nworld) - 1)
        return local_index < counts[safe_world].clamp_min(0)

    # Fallback for compact contact arrays with only a global active count.
    return flat_index < int(torch.sum(counts.clamp_min(0)).item())


def _fill_reasons(
    count: int,
    active: torch.Tensor,
    pen_rd: torch.Tensor,
    bad: torch.Tensor,
    dist: torch.Tensor,
    geom: torch.Tensor,
    worldid: torch.Tensor,
    raw_model: mujoco.MjModel,
    raw_data: mujoco.MjData,
    pen_np: np.ndarray,
    ok_np: np.ndarray,
    robot_drawer_reason: list[str],
    scene_reason: list[str],
) -> None:
    """Populate worst-pair reason strings per world (CPU, logging only)."""
    if not (pen_rd.any().item() or bad.any().item()):
        return
    active_indices = torch.nonzero(active.reshape(-1), as_tuple=False).reshape(-1)
    if int(active_indices.numel()) == 0:
        return
    dist_cpu = dist[active_indices].detach().cpu().numpy()
    geom_cpu = geom[active_indices].detach().cpu().numpy()
    world_cpu = worldid[active_indices].detach().cpu().numpy()
    pen_cpu = pen_rd[active_indices].detach().cpu().numpy()
    bad_cpu = bad[active_indices].detach().cpu().numpy()
    for k in range(int(active_indices.numel())):
        w = int(world_cpu[k])
        if w >= count:
            continue
        g1 = int(geom_cpu[k, 0])
        g2 = int(geom_cpu[k, 2 - 1])
        n1 = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1)
        n2 = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2)
        d = float(dist_cpu[k])
        if pen_cpu[k] > 0.0 and pen_cpu[k] >= pen_np[w] - 1e-9:
            robot_drawer_reason[w] = f"{n1}--{n2}:dist={d:.6f}"
        if bad_cpu[k] and not ok_np[w] and scene_reason[w] == "collision_free":
            scene_reason[w] = f"collision:{n1}--{n2}:dist={d:.6f}"


__all__ = [
    "CandidateCollisionReport",
    "FullSceneCollisionChecker",
    "FullSceneCollisionCheckerPool",
    "FullSceneMjWarp",
]
