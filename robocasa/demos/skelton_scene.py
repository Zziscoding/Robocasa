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

import numpy as np
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

    def check_penetration(
        self,
        samples_w,
        radii,
        exclude_geom_ids=(),
        margin: float = 0.0,
    ):
        """Per-sample signed-distance / penetration check via ``mj_ray``.

        For every sample (a sphere of ``radii[i]``) we cast six axis-aligned
        rays outward. If two opposite rays both hit the *same* scene geom
        (not in ``exclude_geom_ids``) with total travel <
        ``2 * (radius + margin)`` the sample sits inside that geom.

        Returns ``(signed_distance, hit_geom_id)`` with one entry per sample.
        ``signed_distance`` is negative on penetration (``-half_overlap``) and
        ``+inf`` when no scene geom is enclosing the sample.  Uses this pool's
        worker-0 ``MjData`` — call :meth:`reset` first if the env state
        changed since ``from_env``.
        """
        samples = np.ascontiguousarray(
            np.asarray(samples_w, dtype=np.float64).reshape(-1, 3)
        )
        radii = np.asarray(radii, dtype=np.float64).reshape(-1)
        if radii.size != samples.shape[0]:
            raise ValueError("radii size must match samples")
        exclude = set(int(g) for g in (exclude_geom_ids or ()))
        model = self.model
        data = self._datas[0]
        geomgroup = np.ones(6, dtype=np.uint8)
        gid_scratch = np.zeros(1, dtype=np.int32)
        n_geoms = int(model.ngeom)
        signed = np.full(samples.shape[0], np.inf, dtype=np.float64)
        hit_geom = np.full(samples.shape[0], -1, dtype=np.int64)
        axes = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float64,
        )
        for si in range(samples.shape[0]):
            origin = np.ascontiguousarray(samples[si])
            per_axis_hit = {}
            for ai in range(6):
                gid_scratch[0] = -1
                try:
                    dist_val = mujoco.mj_ray(
                        model,
                        data,
                        origin,
                        np.ascontiguousarray(axes[ai]),
                        geomgroup,
                        1,
                        -1,
                        gid_scratch,
                    )
                except Exception:
                    dist_val = -1.0
                    gid_scratch[0] = -1
                gid = int(gid_scratch[0])
                if (
                    float(dist_val) > 0.0
                    and gid >= 0
                    and gid < n_geoms
                    and gid not in exclude
                ):
                    per_axis_hit[ai] = (float(dist_val), gid)
            # Pair opposite rays: (+x,-x) (+y,-y) (+z,-z) -> (0,1) (2,3) (4,5)
            for a, b in ((0, 1), (2, 3), (4, 5)):
                if a not in per_axis_hit or b not in per_axis_hit:
                    continue
                d_a, gid_a = per_axis_hit[a]
                d_b, gid_b = per_axis_hit[b]
                if gid_a != gid_b:
                    continue
                span = d_a + d_b
                # Sample is inside geom gid_a. Approx signed distance = -min(d_a, d_b).
                sd = -min(d_a, d_b)
                if sd < signed[si]:
                    signed[si] = sd
                    hit_geom[si] = gid_a
            # Non-penetrating: report closest ray hit as positive clearance
            if not np.isfinite(signed[si]) or signed[si] >= 0.0:
                best = np.inf
                best_gid = -1
                for _, (d_val, gid) in per_axis_hit.items():
                    clr = d_val - float(radii[si])
                    if clr < best:
                        best = clr
                        best_gid = gid
                if np.isfinite(best):
                    signed[si] = best
                    hit_geom[si] = best_gid
        return signed, hit_geom

    def geom_name(self, geom_id: int) -> str:
        try:
            return (
                mujoco.mj_id2name(
                    self.model, int(mujoco.mjtObj.mjOBJ_GEOM), int(geom_id)
                )
                or f"geom{int(geom_id)}"
            )
        except Exception:
            return f"geom{int(geom_id)}"


__all__ = ["SkeletonScenePool"]
