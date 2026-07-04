"""MIQP-based grasping contact-pair selection.

Wraps :class:`LambdaContactControlOptimizer` from ``mlqp_point_v2.py`` to
select the top-N (=50 by default) feasible contact pairs on an object mesh,
ranked by their force-closure cost.  Mirrors the grasp-selection logic used
by ``bigrasp.py`` but exposes a single functional entry point that can be
dropped into the ``demo_open_drawer_autogen.py`` pipeline.

The solver works entirely in the **object-local** frame: contact points,
normals and tangent frames are expressed relative to the object mesh.
Use :func:`_local_to_world` (or the helper that already lives in the
caller) to convert back to world coordinates.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent  # .../demos/example_code/grasping
DEMOS_ROOT = CURRENT_DIR.parents[1]  # .../demos
REPO_ROOT = CURRENT_DIR.parents[3]  # .../wm/vlm/robocasa

# mlqp_point_v2.py's first import attempt is `from project_point import ...`,
# so we need REPO_ROOT / DEMOS_ROOT (where project_point.py lives) AND the
# grasping subdir itself on sys.path BEFORE mlqp_point_v2 is imported.
for _p in (DEMOS_ROOT, REPO_ROOT, CURRENT_DIR):
    _ps = str(_p)
    if _ps in sys.path:
        sys.path.remove(_ps)
    sys.path.insert(0, _ps)


def _normalize(vec, eps=1e-9):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(vec))
    if n < eps:
        return np.zeros_like(vec)
    return vec / n


def _quat_wxyz_to_matrix(quat_wxyz):
    """wxyz quaternion -> 3x3 rotation matrix."""
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat_wxyz = quat_wxyz / max(float(np.linalg.norm(quat_wxyz)), 1e-9)
    # scipy uses xyzw
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float64)


def _decompose_handle_coacd(
    mesh_path: str,
    threshold: float = 0.05,
    max_convex_hull: int = 32,
    resolution: int = 2000,
    mcts_nodes: int = 20,
    mcts_iterations: int = 100,
    mcts_max_depth: int = 3,
    preprocess_mode: str = "auto",
    preprocess_resolution: int = 30,
    max_ch_vertex: int = 256,
    seed: int = 0,
) -> str:
    """Decompose *mesh_path* with COACD and write the combined parts to a new STL.

    The handle mesh that robocasa ships is typically a single trimesh whose
    concave "wrap-around" cavity has been filled in by the asset-export step
    (or by trimesh's implicit convex-hull fill when the STL is non-watertight).
    Sampling contact points on that directly deposits points inside the cavity
    (the ``handle空隙`` the user sees).  COACD re-decomposes the mesh into a set
    of convex pieces whose **union** reproduces the original surface *without*
    filling the cavity, letting the sampler place points on the real handle
    surface instead.

    Parameters
    ----------
    mesh_path : str
        Absolute path to the input STL/OBJ mesh.
    threshold, max_convex_hull, resolution, mcts_nodes, mcts_iterations,
        mcts_max_depth, preprocess_mode, preprocess_resolution, max_ch_vertex
        Forwarded to ``coacd.run_coacd``.  See the COACD documentation for
        details; the defaults mirror ``demo_close_drawer_autogen.py``.
    seed : int
        RNG seed for COACD.

    Returns
    -------
    new_mesh_path : str
        Path to the temporary STL that contains the combined COACD parts.
        The caller is responsible for deleting this file (it is created with
        ``delete=False``).
    """
    import tempfile

    import coacd
    import trimesh

    mesh = trimesh.load_mesh(mesh_path)
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    if mesh.faces.shape[0] < 4:
        # Too few faces to decompose — fall back to the original mesh.
        return mesh_path

    source = coacd.Mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    kwargs = {
        "threshold": float(threshold),
        "max_convex_hull": int(max_convex_hull),
        "preprocess_mode": str(preprocess_mode),
        "preprocess_resolution": int(preprocess_resolution),
        "resolution": int(resolution),
        "mcts_nodes": int(mcts_nodes),
        "mcts_iterations": int(mcts_iterations),
        "mcts_max_depth": int(mcts_max_depth),
        "max_ch_vertex": int(max_ch_vertex),
        "seed": int(seed),
    }
    try:
        parts = coacd.run_coacd(source, **kwargs)
    except TypeError:
        # Older COACD builds don't accept ``seed``.
        kwargs.pop("seed", None)
        parts = coacd.run_coacd(source, **kwargs)

    if not parts:
        return mesh_path

    combined = trimesh.util.concatenate(
        [
            trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            for verts, faces in parts
        ]
    )
    combined.remove_unreferenced_vertices()

    tmp = tempfile.NamedTemporaryFile(prefix="miqp_coacd_", suffix=".stl", delete=False)
    tmp.close()
    combined.export(tmp.name)
    return tmp.name


def _verify_force_closure_direct(
    optimizer,
    contact_indices: np.ndarray,
    disturbance_wrench_count: int = 200,
    threshold: float = 2.0,
    seed: int = 0,
) -> dict:
    """Verify force closure by explicitly testing random disturbance wrenches.

    The DAQP force-closure cost only measures how well the contact forces can
    cancel the 12 canonical disturbance wrenches (±identity).  A pair can score
    well on those 12 yet still fail against other directions, which is why some
    "valid" pairs look wrong visually.  This helper generates additional random
    unit wrenches on the 6-D sphere and re-runs the QP solve for each of them.
    If any residual exceeds ``threshold`` the pair is rejected.

    Returns
    -------
    dict with keys ``valid`` (bool), ``max_residual`` (float),
    ``mean_residual`` (float), ``n_tested`` (int).
    """
    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    extra_wrenches = rng.standard_normal((int(disturbance_wrench_count), 6))
    norms = np.linalg.norm(extra_wrenches, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    extra_wrenches = extra_wrenches / norms

    # Disturbance wrenches are unit vectors in the *unscaled* wrench space,
    # exactly like the 12 canonical ±identity wrenches used by the original
    # QP.  The residual is therefore directly comparable to the per-column
    # cost the QP reports.
    G = optimizer._stack_grasp_matrices(contact_indices, scaled=True)
    G = np.asarray(G, dtype=np.float64).reshape(6, -1)
    nc = optimizer.num_grasp_contacts
    n = 3 * nc
    reg = float(optimizer.force_reg_weight)
    beta = float(optimizer.beta)

    H = 2.0 * (G.T @ G + reg * np.eye(n, dtype=np.float64))
    A, blower, bupper = optimizer._get_qp_cone_matrix(fn_lb=0.0, fn_ub=1.0)

    residuals = []
    for k in range(extra_wrenches.shape[0]):
        d_k = np.asarray(extra_wrenches[k], dtype=np.float64).reshape(6)
        g_k = -2.0 * beta * (G.T @ d_k)
        x_k, status = optimizer._solve_qp_daqp(H, g_k, A, bupper, blower)
        if status != 1:
            return {
                "valid": False,
                "max_residual": float("inf"),
                "mean_residual": float("inf"),
                "n_tested": int(k + 1),
            }
        residual = beta * d_k - G @ x_k
        residuals.append(float(residual @ residual))

    max_res = float(np.max(residuals)) if residuals else 0.0
    mean_res = float(np.mean(residuals)) if residuals else 0.0
    return {
        "valid": max_res <= float(threshold),
        "max_residual": max_res,
        "mean_residual": mean_res,
        "n_tested": len(residuals),
    }


def solve_grasping_contact_pairs(
    mesh_path: str,
    obj_pos: np.ndarray,
    obj_quat: np.ndarray,
    obj_scale: np.ndarray = np.ones(3),
    num_pairs: int = 512,
    min_pairs: int = 256,
    mu_arm_obj: float = 0.9,
    friction_cone_edges: int = 8,
    min_pair_distance: float = 0.015,
    sample_budget: int = 120,
    region_anchor_count: int = 100,
    region_radius: float = 0.08,
    top_region_pairs: int = 10,
    preselect_region_pairs: int = 300,
    overlap_penalty_weight: float = 1500.0,
    antipodal_penalty_weight: float = 25.0,
    centerline_penalty_weight: float = 6.0,
    support_axis_penalty_weight: float = 12.0,
    curvature_penalty_weight: float = 1.5,
    normal_consistency_min: float = 0.6,
    accessibility_alignment_min: float = -0.2,
    support_surface_normal: Optional[np.ndarray] = None,
    support_surface_point: Optional[np.ndarray] = None,
    support_surface_clearance: float = 0.0,
    nlp_solver: str = "daqp",
    obj_mass: float = 0.01,
    top_candidate_count: Optional[int] = None,
    seed: int = 0,
    verbose: bool = False,
    debug: bool = False,
    # --- Handle concavity fix: COACD decomposition ---------------------------
    use_coacd: bool = False,
    coacd_threshold: float = 0.05,
    coacd_max_convex_hull: int = 32,
    coacd_resolution: int = 2000,
    coacd_mcts_nodes: int = 20,
    coacd_mcts_iterations: int = 100,
    coacd_mcts_max_depth: int = 3,
    coacd_preprocess_mode: str = "auto",
    coacd_preprocess_resolution: int = 30,
    coacd_max_ch_vertex: int = 256,
    # --- DEPRECATED: DAQP force-closure verification -------------------------
    # These args are kept for backward-compatible callers but the QP-based
    # verification has been replaced by a closed-form directional-force test
    # (each contact's force points along the pair-connecting line; the pair
    # passes iff both forces lie inside their Coulomb friction cones).
    disturbance_wrench_count: int = 0,
    force_closure_verify: bool = False,
    force_closure_verify_threshold: float = 2.0,
) -> list[dict]:
    """Select the top-N feasible contact pairs on a mesh, ranked by force-closure cost.

    Parameters
    ----------
    mesh_path : str
        Path to the object mesh file (stl/obj/ply).
    obj_pos : (3,) array
        World-frame object position.
    obj_quat : (4,) array
        World-frame object quaternion (wxyz convention).
    obj_scale : (3,) array
        Mesh scale factors applied by ProjectionPoint.
    num_pairs : int
        Number of best pairs to return.
    mu_arm_obj : float
        Finger-object friction coefficient.
    friction_cone_edges : int
        Polygonal approximation of the friction cone.
    min_pair_distance : float
        Minimum allowed distance between the two contact points (m).
    sample_budget : int
        Number of surface samples to draw from the mesh.
    region_anchor_count : int
        Number of FPS anchors for region sampling.  Set to 100 to produce a
        denser coverage of the handle surface and more candidate pairs.
    region_radius : float
        Radius of each candidate region around an anchor.
    top_region_pairs : int
        Number of top-ranked region groups to keep for contact-pair generation.
        Increasing this directly raises the number of candidate pairs.
    preselect_region_pairs : int
        Number of coarse-scored region groups to keep before the expensive
        GWB ranking stage.  Should be >= ``top_region_pairs``.
    use_coacd : bool
        If True, decompose the mesh with COACD before sampling.  This prevents
        the handle's concave cavity from being "filled in" by trimesh's
        convex-hull auto-filling (see ``_decompose_handle_coacd``).
    coacd_threshold : float
        COACD concavity threshold (lower → more convex parts).
    coacd_max_convex_hull : int
        Maximum number of convex parts COACD may emit.
    coacd_resolution : int
        COACD voxel resolution for the preprocessing step.
    coacd_mcts_nodes : int
        COACD MCTS node count.
    coacd_mcts_iterations : int
        COACD MCTS iteration count.
    coacd_mcts_max_depth : int
        COACD MCTS max tree depth.
    coacd_preprocess_mode : str
        COACD preprocessing mode (``"auto"``, ``"on"``, ``"off"``).
    coacd_preprocess_resolution : int
        COACD preprocessing resolution.
    coacd_max_ch_vertex : int
        Maximum vertices per convex-hull part in COACD.
    disturbance_wrench_count : int
        Number of random disturbance wrenches used to verify force closure
        beyond the standard ±identity axes.  Higher values give a stricter
        force-closure test (default 200).
    force_closure_verify : bool
        If True, run a secondary force-closure verification pass on the
        top-ranked pairs and discard pairs that fail.
    force_closure_verify_threshold : float
        Maximum allowed squared residual per disturbance column for a pair
        to pass the verification pass.  The residual is computed in the same
        (scaled) wrench space as the original QP, so ``2.0`` means the
        worst-case random disturbance can be resisted with residual norm
        below ``sqrt(2.0) ≈ 1.41``.  Lower values enforce stricter force
        closure at the cost of discarding more candidates.

    Returns
    -------
    pairs : list of dict
        ``num_pairs`` entries (or fewer if not enough valid pairs found).
        Each dict contains:

        - ``contact_indices`` – (2,) int indices into the optimizer's
          sample-point array.
        - ``contact_points_world`` – (2, 3) world-frame contact positions.
        - ``contact_points_local`` – (2, 3) object-local contact positions.
        - ``normals_world`` – (2, 3) inward surface normals in world frame.
        - ``normals_local`` – (2, 3) inward surface normals in object frame.
        - ``tangent1_world``, ``tangent2_world`` – (2, 3) world tangent frames.
        - ``tangent1_local``, ``tangent2_local`` – (2, 3) object tangent frames.
        - ``force_closure_cost`` – float, force-closure residual cost.
        - ``total_cost`` – float, overall score (lower is better).
        - ``valid`` – bool, whether the pair satisfies friction-cone + equilibrium.
        - ``antipodal_margin`` – float, force-closure geometric margin.
    """
    # Import mlqp_point_v2 here so its `from project_point import ...`
    # resolves against the demos root (and the current subdir).
    sys.path.insert(0, str(CURRENT_DIR.parent))
    sys.path.insert(0, str(CURRENT_DIR))
    from robocasa.demos.example_code.grasping.mlqp_point_v2 import (
        LambdaContactControlOptimizer,
    )

    del top_candidate_count, top_region_pairs, preselect_region_pairs
    del region_anchor_count, region_radius, overlap_penalty_weight
    del antipodal_penalty_weight, centerline_penalty_weight
    del support_axis_penalty_weight, curvature_penalty_weight
    del normal_consistency_min, accessibility_alignment_min
    del support_surface_normal, support_surface_point, support_surface_clearance
    del disturbance_wrench_count, force_closure_verify
    del force_closure_verify_threshold, obj_mass

    t0 = time.perf_counter()

    # --- 1. Optional COACD decomposition of the input mesh -------------------
    actual_mesh_path = str(mesh_path)
    coacd_mesh_path = None
    if use_coacd:
        if verbose:
            print(
                f"[miqp_grasping] running COACD decomposition on {mesh_path}",
                flush=True,
            )
        coacd_mesh_path = _decompose_handle_coacd(
            mesh_path=str(mesh_path),
            threshold=float(coacd_threshold),
            max_convex_hull=int(coacd_max_convex_hull),
            resolution=int(coacd_resolution),
            mcts_nodes=int(coacd_mcts_nodes),
            mcts_iterations=int(coacd_mcts_iterations),
            mcts_max_depth=int(coacd_mcts_max_depth),
            preprocess_mode=str(coacd_preprocess_mode),
            preprocess_resolution=int(coacd_preprocess_resolution),
            max_ch_vertex=int(coacd_max_ch_vertex),
            seed=int(seed),
        )
        actual_mesh_path = coacd_mesh_path

    # --- 2. Build the optimizer purely to reuse its surface sampling  --------
    # (sample_point / normal / t1 / t2). The 6-D QP force-closure ranking is
    # bypassed entirely — see step 3 below.
    optimizer = LambdaContactControlOptimizer(
        mesh_path=actual_mesh_path,
        obj_mass=1e-3,
        arm_friction=float(mu_arm_obj),
        sample_num=int(sample_budget),
        scale_factors=tuple(float(s) for s in np.asarray(obj_scale).reshape(3)),
        num_grasp_contacts=2,
        friction_cone_edges=int(friction_cone_edges),
        min_pair_distance=float(min_pair_distance),
        nlp_solver=str(nlp_solver),
        static_nlp_solver=str(nlp_solver),
    )

    sample_point = np.asarray(optimizer.sample_point, dtype=np.float64)
    normal = np.asarray(optimizer.normal, dtype=np.float64)
    t1_arr = np.asarray(optimizer.t1, dtype=np.float64)
    t2_arr = np.asarray(optimizer.t2, dtype=np.float64)
    n_samples = int(sample_point.shape[0])

    if n_samples < 2:
        if coacd_mesh_path is not None and coacd_mesh_path != str(mesh_path):
            import os as _os

            try:
                _os.unlink(coacd_mesh_path)
            except OSError:
                pass
        return []

    obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
    obj_rot = _quat_wxyz_to_matrix(obj_quat)

    # --- 3. Closed-form directional-force friction-cone test -----------------
    # For every unordered pair (i, j), fix the contact forces to point along
    # the line connecting the two contacts:
    #     f_i = normalize(p_j - p_i),   f_j = -f_i
    # Translation and torque balance are automatic (two colinear opposing
    # forces). The pair is force-closable iff each fixed force lies inside
    # its Coulomb friction cone at that contact:
    #     dot(f_i, n_i) > 0 AND ||f_i_lateral|| <= mu * dot(f_i, n_i)
    # where n_i is the inward surface normal.  Ranking is by cone margin
    # (larger = more centered in the friction cone).
    mu = max(float(mu_arm_obj), 1e-6)
    min_dist = max(float(min_pair_distance), 0.0)
    # Broadcast pairwise vectors.
    idx_i, idx_j = np.triu_indices(n_samples, k=1)
    p_i = sample_point[idx_i]
    p_j = sample_point[idx_j]
    n_i = normal[idx_i]
    n_j = normal[idx_j]

    diff = p_j - p_i
    dist = np.linalg.norm(diff, axis=1)
    # Force direction on contact i: from i toward j (both are unit-normalized).
    with np.errstate(invalid="ignore", divide="ignore"):
        f_i_dir = diff / np.where(dist[:, None] > 1e-12, dist[:, None], 1.0)
    # Contact i: force is +f_i_dir; contact j: force is -f_i_dir.
    # Cone check uses the *inward* surface normal (points into the object),
    # which is what optimizer.normal already stores.
    cos_i = np.einsum("ij,ij->i", f_i_dir, n_i)
    cos_j = np.einsum("ij,ij->i", -f_i_dir, n_j)
    # Cone margin: cos(angle) between force and inward normal must exceed
    # the friction-cone half-aperture cos(atan(mu)) = 1/sqrt(1+mu²).
    cone_threshold = 1.0 / float(np.sqrt(1.0 + mu * mu))

    cone_i_ok = cos_i >= cone_threshold
    cone_j_ok = cos_j >= cone_threshold
    dist_ok = dist >= min_dist
    valid_mask = cone_i_ok & cone_j_ok & dist_ok

    if debug or verbose:
        total_pairs = int(idx_i.size)
        reject_dist = int(np.sum(~dist_ok))
        reject_cone_i = int(np.sum(dist_ok & ~cone_i_ok))
        reject_cone_j = int(np.sum(dist_ok & cone_i_ok & ~cone_j_ok))
        print(
            f"[miqp_grasping] pair sweep: total={total_pairs} "
            f"reject_too_close={reject_dist} "
            f"reject_cone_i={reject_cone_i} "
            f"reject_cone_j={reject_cone_j} "
            f"valid={int(valid_mask.sum())} "
            f"(mu={mu:.3f}, cone_threshold={cone_threshold:.4f}, "
            f"min_dist={min_dist:.4f})",
            flush=True,
        )

    valid_indices = np.nonzero(valid_mask)[0]
    if valid_indices.size == 0:
        if verbose:
            print(
                "[miqp_grasping] no valid pairs after friction-cone test.",
                flush=True,
            )
        if coacd_mesh_path is not None and coacd_mesh_path != str(mesh_path):
            import os as _os

            try:
                _os.unlink(coacd_mesh_path)
            except OSError:
                pass
        return []

    # Rank surviving pairs by the WORST of the two cone-alignment cosines
    # (larger = both forces more centered in cone). Break ties by the sum
    # so nearly-antipodal pairs win over slightly-off ones.
    cone_min = np.minimum(cos_i[valid_indices], cos_j[valid_indices])
    cone_sum = cos_i[valid_indices] + cos_j[valid_indices]
    order = np.lexsort((-cone_sum, -cone_min))
    ranked = valid_indices[order]

    max_return = int(max(min_pairs, num_pairs))
    ranked = ranked[:max_return]

    if verbose and int(ranked.size) < int(min_pairs):
        print(
            f"[miqp_grasping] only {int(ranked.size)} valid pairs available "
            f"(requested min_pairs={int(min_pairs)}).",
            flush=True,
        )

    result: list[dict] = []
    for pair_idx in ranked:
        i = int(idx_i[int(pair_idx)])
        j = int(idx_j[int(pair_idx)])
        pts_local = np.stack([sample_point[i], sample_point[j]], axis=0)
        nrm_local = np.stack([normal[i], normal[j]], axis=0)
        t1_local = np.stack([t1_arr[i], t1_arr[j]], axis=0)
        t2_local = np.stack([t2_arr[i], t2_arr[j]], axis=0)

        pts_world = (obj_rot @ pts_local.T).T + obj_pos[None, :]
        nrm_world = (obj_rot @ nrm_local.T).T
        t1_world = (obj_rot @ t1_local.T).T
        t2_world = (obj_rot @ t2_local.T).T

        margin_i = float(cos_i[int(pair_idx)])
        margin_j = float(cos_j[int(pair_idx)])
        pair_margin = float(min(margin_i, margin_j))

        result.append(
            {
                "contact_indices": np.array([i, j], dtype=int),
                "contact_points_world": pts_world,
                "contact_points_local": pts_local,
                "normals_world": nrm_world,
                "normals_local": nrm_local,
                "tangent1_world": t1_world,
                "tangent2_world": t2_world,
                "tangent1_local": t1_local,
                "tangent2_local": t2_local,
                # Preserve legacy keys; use (1 - cone_margin) so "lower is
                # better" matches the old force_closure_cost semantics.
                "force_closure_cost": float(1.0 - pair_margin),
                "total_cost": float(1.0 - pair_margin),
                "valid": True,
                "antipodal_margin": pair_margin,
                "cone_cosine_i": margin_i,
                "cone_cosine_j": margin_j,
                "pair_distance": float(dist[int(pair_idx)]),
            }
        )

    elapsed = time.perf_counter() - t0

    if coacd_mesh_path is not None and coacd_mesh_path != str(mesh_path):
        import os as _os

        try:
            _os.unlink(coacd_mesh_path)
        except OSError:
            pass

    if verbose:
        print(
            f"[miqp_grasping] selected {len(result)} / {max_return} pairs "
            f"(min_pairs={int(min_pairs)}, n_samples={n_samples}) "
            f"in {elapsed:.3f}s",
            flush=True,
        )

    return result


def evaluate_force_closure_at_points(
    mesh_path: str,
    obj_scale: np.ndarray,
    contact_indices: np.ndarray,
    obj_quat: Optional[np.ndarray] = None,
    external_wrench: Optional[np.ndarray] = None,
    nlp_solver: str = "daqp",
    mu_arm_obj: float = 0.9,
    friction_cone_edges: int = 8,
    sample_budget: int = 120,
    verbose: bool = False,
) -> dict:
    """Evaluate force-closure cost for a specific pair of contact indices.

    Used by :class:`GraspRollout` to re-evaluate closure quality at the
    actually-reached finger positions.

    Parameters
    ----------
    mesh_path : str
    obj_scale : (3,) array
    contact_indices : (2,) array
    obj_quat : (4,) or None — if None, identity rotation is assumed.
    external_wrench : (6,) or None — additional wrench (e.g. gravity) in the
        object frame, already scaled if desired.
    nlp_solver, mu_arm_obj, friction_cone_edges, sample_budget :
        forwarded to the underlying optimizer.

    Returns
    -------
    dict with keys ``valid``, ``force_closure_cost``, ``total_cost``,
    ``contact_forces``, ``witness_forces``.
    """
    from robocasa.demos.example_code.grasping.mlqp_point_v2 import (
        LambdaContactControlOptimizer,
    )

    optimizer = LambdaContactControlOptimizer(
        mesh_path=str(mesh_path),
        scale_factors=tuple(float(s) for s in np.asarray(obj_scale).reshape(3)),
        num_grasp_contacts=2,
        arm_friction=float(mu_arm_obj),
        friction_cone_edges=int(friction_cone_edges),
        sample_num=int(sample_budget),
        nlp_solver=str(nlp_solver),
        static_nlp_solver=str(nlp_solver),
    )

    contact_indices = np.asarray(contact_indices, dtype=int).reshape(-1)
    if contact_indices.size != 2:
        return {
            "valid": False,
            "force_closure_cost": float("inf"),
            "total_cost": float("inf"),
            "contact_forces": None,
            "witness_forces": None,
        }

    kwargs = {}
    if obj_quat is not None:
        kwargs["object_rot"] = _quat_wxyz_to_matrix(obj_quat)
    if external_wrench is not None:
        kwargs["external_wrench"] = np.asarray(
            external_wrench, dtype=np.float64
        ).reshape(6)

    result = optimizer._evaluate_force_closure_candidate(contact_indices, **kwargs)
    if verbose:
        print(
            f"[miqp_grasping] fc_cost={result.get('force_closure_cost', float('inf')):.6f} "
            f"valid={result.get('valid', False)}",
            flush=True,
        )
    return {
        "valid": bool(result.get("valid", False)),
        "force_closure_cost": float(result.get("force_closure_cost", float("inf"))),
        "total_cost": float(result.get("total_cost", float("inf"))),
        "contact_forces": result.get("witness_contact_forces_local", None),
        "witness_forces": result.get("witness_force_vectors_local", None),
    }


__all__ = [
    "solve_grasping_contact_pairs",
    "evaluate_force_closure_at_points",
]
