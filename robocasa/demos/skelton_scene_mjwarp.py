"""Per-worker mujoco ``MjData`` clones for parallel DAQP skeleton solves.

Despite the file name (mirrors :mod:`full_scene_mjwarp`), this uses plain
mujoco ``MjData``, not the mjwarp/comfree GPU backend: the DAQP inner loop in
:func:`robocasa.demos.ee_skelton.solve_skeleton_pose` uses ``mujoco.mj_ray``
for per-sample directional scene distances
(``ee_skelton.py:842``), and mjwarp does not expose an ``mj_ray`` equivalent.
Batched ``contact.dist`` has different semantics (convex-convex penetration
depth vs. point-along-direction ray distance) and would change the QP's
linear scene constraints.

The pool shares one read-only ``MjModel`` (the env's compiled model, which is
immutable after compilation and therefore safe to share across threads) and
gives each worker its own ``MjData``, 1:1 cloned from ``env.sim.data`` and
``mj_forward``-ed so ``geom_xpos/geom_xmat`` match the shared env exactly.
Each DAQP batch begins with :meth:`reset` that re-syncs every worker's data
from env's current state — this mirrors what
:meth:`full_scene_mjwarp.FullSceneCollisionChecker.refresh_base` does for the
mink ``q`` path.
"""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager

import mujoco


class SkeletonScenePool:
    """Per-worker cloned ``MjData`` for read-only ``mj_ray`` in DAQP.

    Usage::

        pool = SkeletonScenePool.from_env(env, num_workers=8)
        pool.reset()                            # once per DAQP batch
        with pool.borrow() as (model, data):
            dist = mujoco.mj_ray(model, data, ...)
    """

    def __init__(
        self,
        env,
        model: mujoco.MjModel,
        datas: list[mujoco.MjData],
    ) -> None:
        self.env = env
        self.model = model
        self._datas = list(datas)
        self.num_workers = len(self._datas)
        self._lock = threading.Lock()
        self._free: queue.Queue[int] = queue.Queue()
        for i in range(self.num_workers):
            self._free.put(i)

    @classmethod
    def from_env(cls, env, *, num_workers: int) -> "SkeletonScenePool":
        model = getattr(env.sim.model, "_model", env.sim.model)
        src_data = getattr(env.sim.data, "_data", env.sim.data)
        n = max(int(num_workers), 1)
        datas: list[mujoco.MjData] = []
        for _ in range(n):
            d = mujoco.MjData(model)
            mujoco.mj_copyData(d, model, src_data)
            mujoco.mj_forward(model, d)
            datas.append(d)
        return cls(env, model, datas)

    def reset(self) -> None:
        """Re-sync every worker's ``MjData`` from env's current state.

        Call once at the start of each DAQP batch so the drawer / kitchen
        geoms reflect any joint movement that happened between batches.
        """
        src_data = getattr(self.env.sim.data, "_data", self.env.sim.data)
        for d in self._datas:
            mujoco.mj_copyData(d, self.model, src_data)
            mujoco.mj_forward(self.model, d)

    @contextmanager
    def borrow(self):
        """Check out one worker's ``(model, data)`` pair for the current thread."""
        worker_id = self._free.get()
        try:
            yield self.model, self._datas[worker_id]
        finally:
            self._free.put(worker_id)

    def get(self, worker_id: int) -> tuple[mujoco.MjModel, mujoco.MjData]:
        return self.model, self._datas[worker_id]


__all__ = ["SkeletonScenePool"]
