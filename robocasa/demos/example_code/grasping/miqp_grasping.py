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
    num_pairs: int = 50,
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
    # --- DAQP force-closure verification ------------------------------------
    disturbance_wrench_count: int = 200,
    force_closure_verify: bool = True,
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

    t0 = time.perf_counter()

    # --- 2. Handle concavity fix: optionally decompose the mesh with COACD ---
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
        if verbose:
            print(
                f"[miqp_grasping] COACD mesh written to {coacd_mesh_path}",
                flush=True,
            )

    optimizer = LambdaContactControlOptimizer(
        mesh_path=actual_mesh_path,
        obj_mass=float(obj_mass),
        arm_friction=float(mu_arm_obj),
        sample_num=int(sample_budget),
        scale_factors=tuple(float(s) for s in np.asarray(obj_scale).reshape(3)),
        num_grasp_contacts=2,
        friction_cone_edges=int(friction_cone_edges),
        min_pair_distance=float(min_pair_distance),
        region_anchor_count=int(region_anchor_count),
        region_radius=float(region_radius),
        top_region_pairs=int(top_region_pairs),
        preselect_region_pairs=int(preselect_region_pairs),
        overlap_penalty_weight=float(overlap_penalty_weight),
        antipodal_penalty_weight=float(antipodal_penalty_weight),
        centerline_penalty_weight=float(centerline_penalty_weight),
        support_axis_penalty_weight=float(support_axis_penalty_weight),
        curvature_penalty_weight=float(curvature_penalty_weight),
        normal_consistency_min=float(normal_consistency_min),
        accessibility_alignment_min=float(accessibility_alignment_min),
        nlp_solver=str(nlp_solver),
        static_nlp_solver=str(nlp_solver),
    )

    if support_surface_normal is not None:
        optimizer.set_support_surface(
            support_surface_point=support_surface_point,
            support_surface_normal=_normalize(support_surface_normal),
            support_surface_clearance=float(support_surface_clearance),
        )

    obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
    obj_rot = _quat_wxyz_to_matrix(obj_quat)

    if top_candidate_count is None:
        top_candidate_count = num_pairs

    cache = optimizer.precompute_contact_search_cache(
        object_pos=obj_pos,
        object_rot=obj_rot,
        top_candidate_count=int(top_candidate_count),
        gravity_world=(0.0, 0.0, -9.81),
    )

    candidate_entries = cache.get("candidate_entries", [])
    if not candidate_entries:
        if verbose:
            print("[miqp_grasping] no feasible contact pairs found.", flush=True)
        return []

    result = []
    sample_point = np.asarray(optimizer.sample_point, dtype=np.float64)
    normal = np.asarray(optimizer.normal, dtype=np.float64)
    t1 = np.asarray(optimizer.t1, dtype=np.float64)
    t2 = np.asarray(optimizer.t2, dtype=np.float64)

    n_verified = 0
    n_passed = 0
    verify_t0 = time.perf_counter()

    for entry in candidate_entries:
        if len(result) >= num_pairs:
            break

        contact_indices = np.asarray(
            entry.get("contact_indices", []), dtype=int
        ).reshape(-1)
        if contact_indices.size != 2:
            continue

        entry_valid = bool(
            np.isfinite(entry.get("offline_force_closure_cost", float("inf")))
        )

        # --- 3. Optional secondary force-closure verification -----------------
        if entry_valid and force_closure_verify and disturbance_wrench_count > 0:
            n_verified += 1
            verify_res = _verify_force_closure_direct(
                optimizer,
                contact_indices,
                disturbance_wrench_count=int(disturbance_wrench_count),
                threshold=float(force_closure_verify_threshold),
                seed=int(seed)
                + int(contact_indices[0]) * 7919
                + int(contact_indices[1]) * 104729,
            )
            if not verify_res["valid"]:
                entry_valid = False
                if verbose:
                    print(
                        f"[miqp_grasping] pair {contact_indices.tolist()} failed "
                        f".verify (max_residual={verify_res['max_residual']:.4e}, "
                        f"mean={verify_res['mean_residual']:.4e}, "
                        f"n_tested={verify_res['n_tested']})",
                        flush=True,
                    )
            else:
                n_passed += 1

        if not entry_valid:
            continue

        # Object-local geometry.
        pts_local = sample_point[contact_indices]  # (2, 3)
        nrm_local = normal[contact_indices]  # (2, 3)
        t1_local = t1[contact_indices]  # (2, 3)
        t2_local = t2[contact_indices]  # (2, 3)

        # World-frame geometry.
        pts_world = (obj_rot @ pts_local.T).T + obj_pos[None, :]
        nrm_world = (obj_rot @ nrm_local.T).T
        t1_world = (obj_rot @ t1_local.T).T
        t2_world = (obj_rot @ t2_local.T).T

        result.append(
            {
                "contact_indices": contact_indices.copy(),
                "contact_points_world": pts_world,
                "contact_points_local": pts_local,
                "normals_world": nrm_world,
                "normals_local": nrm_local,
                "tangent1_world": t1_world,
                "tangent2_world": t2_world,
                "tangent1_local": t1_local,
                "tangent2_local": t2_local,
                "force_closure_cost": float(
                    entry.get("offline_force_closure_cost", float("inf"))
                ),
                "total_cost": float(
                    entry.get("offline_force_closure_total_cost", float("inf"))
                ),
                "valid": True,
                "antipodal_margin": float(
                    entry.get("antipodal_margin", 0.0)
                    if "antipodal_margin" in entry
                    else entry.get("offline_force_closure_cost", float("inf"))
                ),
            }
        )

    verify_elapsed = time.perf_counter() - verify_t0
    elapsed = time.perf_counter() - t0

    # Clean up the temporary COACD mesh file if one was created.
    if coacd_mesh_path is not None and coacd_mesh_path != str(mesh_path):
        import os

        try:
            os.unlink(coacd_mesh_path)
        except OSError:
            pass

    if verbose:
        print(
            f"[miqp_grasping] selected {len(result)} / {num_pairs} pairs "
            f"({n_verified} verified, {n_passed} passed) "
            f"in {elapsed:.3f}s (verify {verify_elapsed:.3f}s)",
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
