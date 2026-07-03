import argparse
import ctypes
import itertools
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import casadi as cs
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    import open3d as o3d
except ImportError:
    o3d = None

os.environ["SNOPT_LICENSE"] = "/home/lab423/opt_ws/libsnopt7/snopt7.lic"

try:
    from project_point import ProjectionPoint
except ImportError:
    from planning.project_point import ProjectionPoint


OBJECT_ASSET_DIR = Path(__file__).resolve().parents[1] / "envs" / "assets" / "objects"
_EPS = 1e-9
_O3D_OBJECT_BASE_COLOR = np.array([0.82, 0.82, 0.85], dtype=np.float64)
_O3D_OBJECT_TRANSPARENCY = 0.30
_O3D_OBJECT_OPACITY = 1.0 - _O3D_OBJECT_TRANSPARENCY
_ACADOS_TEMPLATE_SYMBOLS = None
_ACADOS_IMPORT_FAILURE = None
_ACADOS_EXPORT_VERSION = "v1"
_ACADOS_CASADI_FALLBACK_WARNED = False
_ACADOS_PRELOADED_SHARED_LIBS = []
_ACADOS_STATUS_LABELS = {
    -1: "ACADOS_UNKNOWN",
    0: "ACADOS_SUCCESS",
    1: "ACADOS_NAN_DETECTED",
    2: "ACADOS_MAXITER",
    3: "ACADOS_MINSTEP",
    4: "ACADOS_QP_FAILURE",
    5: "ACADOS_READY",
    6: "ACADOS_UNBOUNDED",
    7: "ACADOS_TIMEOUT",
    8: "ACADOS_QPSCALING_BOUNDS_NOT_SATISFIED",
    9: "ACADOS_INFEASIBLE",
}


def _repo_root_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shared_lib_ext():
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt":
        return ".dll"
    return ".so"


def _acados_root_candidates():
    repo_root = _repo_root_dir()
    candidates = []
    acados_source_dir = os.environ.get("ACADOS_SOURCE_DIR")
    if acados_source_dir:
        candidates.append(os.path.abspath(acados_source_dir))
    candidates.append(
        os.path.abspath(os.path.join(repo_root, "..", "thirdparty", "acados"))
    )
    candidates.append(os.path.abspath(os.path.join(repo_root, "thirdparty", "acados")))

    deduped = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _install_deprecated_sphinx_shim():
    import types

    if "deprecated.sphinx" in sys.modules:
        return

    deprecated_module = types.ModuleType("deprecated")
    sphinx_module = types.ModuleType("deprecated.sphinx")

    def _deprecated(*args, **kwargs):
        def _decorator(obj):
            return obj

        return _decorator

    sphinx_module.deprecated = _deprecated
    deprecated_module.sphinx = sphinx_module
    sys.modules.setdefault("deprecated", deprecated_module)
    sys.modules.setdefault("deprecated.sphinx", sphinx_module)


def _preload_acados_shared_libraries(acados_root):
    if os.name == "nt":
        return

    lib_dir = os.path.join(acados_root, "lib")
    if not os.path.isdir(lib_dir):
        return

    load_mode = getattr(ctypes, "RTLD_GLOBAL", None)
    shared_lib_names = [
        "libblasfeo.so.0",
        "libblasfeo.so",
        "libhpipm.so",
        "libqpOASES_e.so",
        "libdaqp.so",
        "libosqp.so",
        "libacados.so",
    ]

    for lib_name in shared_lib_names:
        lib_path = os.path.join(lib_dir, lib_name)
        if not os.path.isfile(lib_path):
            continue
        try:
            if load_mode is None:
                handle = ctypes.CDLL(lib_path)
            else:
                handle = ctypes.CDLL(lib_path, mode=load_mode)
        except OSError:
            continue
        _ACADOS_PRELOADED_SHARED_LIBS.append(handle)


def _bootstrap_local_acados_python_interface():
    for acados_root in _acados_root_candidates():
        interface_root = os.path.join(acados_root, "interfaces", "acados_template")
        package_init = os.path.join(interface_root, "acados_template", "__init__.py")
        if not os.path.isfile(package_init):
            continue

        os.environ.setdefault("ACADOS_SOURCE_DIR", acados_root)
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

        acados_lib_dir = os.path.join(acados_root, "lib")
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        ld_entries = [entry for entry in ld_library_path.split(":") if entry]
        if acados_lib_dir not in ld_entries:
            ld_entries.append(acados_lib_dir)
            os.environ["LD_LIBRARY_PATH"] = ":".join(ld_entries)
        _preload_acados_shared_libraries(acados_root)

        if interface_root not in sys.path:
            sys.path.insert(0, interface_root)
        return


def _import_acados_template_symbols():
    global _ACADOS_TEMPLATE_SYMBOLS, _ACADOS_IMPORT_FAILURE

    if _ACADOS_TEMPLATE_SYMBOLS is not None:
        return _ACADOS_TEMPLATE_SYMBOLS
    if _ACADOS_IMPORT_FAILURE is not None:
        raise ImportError(
            "Failed to import acados_template earlier. "
            "See the chained exception for the original cause."
        ) from _ACADOS_IMPORT_FAILURE

    try:
        from deprecated.sphinx import deprecated as _unused_deprecated  # noqa: F401
    except Exception:
        _install_deprecated_sphinx_shim()

    _bootstrap_local_acados_python_interface()

    try:
        from acados_template import (
            ACADOS_INFTY,
            AcadosModel,
            AcadosOcp,
            AcadosOcpSolver,
        )
    except Exception as exc:
        _ACADOS_IMPORT_FAILURE = exc
        raise ImportError(
            "Unable to import acados_template. "
            "The local acados source tree was checked, but Python still could not "
            "load the interface. Make sure the acados Python dependencies are "
            "available and that the local acados build is complete."
        ) from exc

    _ACADOS_TEMPLATE_SYMBOLS = (
        AcadosModel,
        AcadosOcp,
        AcadosOcpSolver,
        ACADOS_INFTY,
    )
    return _ACADOS_TEMPLATE_SYMBOLS


def _ensure_acados_renderer_available():
    tera_candidates = []
    tera_path = os.environ.get("TERA_PATH")
    if tera_path:
        tera_candidates.append(os.path.abspath(tera_path))
    tera_on_path = shutil.which("t_renderer")
    if tera_on_path:
        tera_candidates.append(os.path.abspath(tera_on_path))
    for acados_root in _acados_root_candidates():
        tera_candidates.append(os.path.join(acados_root, "bin", "t_renderer"))
        tera_candidates.append(
            os.path.join(
                acados_root,
                "interfaces",
                "acados_template",
                "tera_renderer",
                "target",
                "release",
                "t_renderer",
            )
        )
        tera_candidates.append(
            os.path.join(
                acados_root,
                "interfaces",
                "acados_template",
                "tera_renderer",
                "target",
                "debug",
                "t_renderer",
            )
        )

    for candidate in tera_candidates:
        if os.path.isfile(candidate):
            os.environ["TERA_PATH"] = candidate
            return candidate

    searched = "\n".join(
        f"  - {candidate}" for candidate in tera_candidates if candidate
    )
    raise RuntimeError(
        "acados was selected, but the tera renderer executable is missing.\n"
        "Looked for it in:\n"
        f"{searched}\n"
        "Install the acados tera renderer or point TERA_PATH to an existing "
        "t_renderer binary before using the acados backend."
    )


def _format_acados_status(status_code, sqp_iter=None):
    label = _ACADOS_STATUS_LABELS.get(
        int(status_code), f"ACADOS_STATUS_{int(status_code)}"
    )
    if sqp_iter is None:
        return label
    return f"{label} (sqp_iter={int(sqp_iter)})"


def _warn_acados_casadi_fallback(reason):
    global _ACADOS_CASADI_FALLBACK_WARNED
    if _ACADOS_CASADI_FALLBACK_WARNED:
        return
    _ACADOS_CASADI_FALLBACK_WARNED = True
    warnings.warn(
        "Falling back from the acados generated solver to the existing CasADi Opti path. "
        f"Reason: {reason}",
        RuntimeWarning,
    )


def _normalize_nlp_solver_name(solver_name, env_var, default_solver):
    if solver_name is None:
        solver_name = os.environ.get(env_var, default_solver)
    solver_name = str(solver_name).strip().lower()
    if solver_name not in {"ipopt", "snopt", "acados", "daqp"}:
        raise ValueError(
            f"Unsupported NLP solver '{solver_name}'. Expected 'ipopt', 'snopt', 'acados' or 'daqp'."
        )
    return solver_name


def _as_numpy(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    if hasattr(value, "full"):
        return np.asarray(value.full(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _normalize(vec, axis=None):
    vec = np.asarray(vec, dtype=np.float64)
    if axis is None:
        norm = float(np.linalg.norm(vec))
        if norm < _EPS:
            return np.zeros_like(vec)
        return vec / norm

    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    norm = np.where(norm < _EPS, 1.0, norm)
    return vec / norm


def _farthest_point_sampling(points, num_samples, seed_index=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or num_samples <= 0:
        return np.zeros((0,), dtype=int)

    num_samples = min(int(num_samples), int(points.shape[0]))
    selected = np.empty(num_samples, dtype=int)

    if seed_index is None:
        center = np.mean(points, axis=0, keepdims=True)
        seed_index = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    seed_index = int(np.clip(seed_index, 0, points.shape[0] - 1))

    selected[0] = seed_index
    min_dist2 = np.sum((points - points[seed_index]) ** 2, axis=1)
    min_dist2[seed_index] = -1.0

    for i in range(1, num_samples):
        next_idx = int(np.argmax(min_dist2))
        selected[i] = next_idx
        dist2 = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
        min_dist2[selected[: i + 1]] = -1.0

    return selected


def _rotation_from_z(direction):
    direction = _normalize(direction)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot_val = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))

    if dot_val > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if dot_val < -1.0 + 1e-8:
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )

    axis = np.cross(z_axis, direction)
    axis = _normalize(axis)
    angle = np.arccos(dot_val)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )


def _orthonormal_frame_from_normal(normal):
    normal = _normalize(normal)
    arbitrary_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(arbitrary_dir, normal))) > 0.9:
        arbitrary_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    tangent1 = arbitrary_dir - np.dot(arbitrary_dir, normal) * normal
    tangent1 = _normalize(tangent1)
    tangent2 = _normalize(np.cross(normal, tangent1))
    return normal, tangent1, tangent2


def _quat_wxyz_to_matrix(quat_wxyz):
    quat_wxyz = _as_numpy(quat_wxyz).reshape(4)
    quat_wxyz = quat_wxyz / max(float(np.linalg.norm(quat_wxyz)), _EPS)
    quat_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )
    return Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float64)


class LambdaContactControlOptimizer:
    def __init__(
        self,
        mesh_path,
        obj_mass=0.01,
        arm_friction=0.9,
        contact_stiffness=12.5,
        time_step=0.01,
        max_contacts=10,
        sample_num=70,
        pos_coef=1.0,
        ori_coef=0.0005,
        scale_factors=(1.0, 1.0, 1.0),
        mppi_samples=96,
        mppi_iterations=4,
        mppi_horizon=4,
        mppi_lambda=1.0,
        mppi_noise_sigma=0.01,
        mppi_noise_decay=0.85,
        mppi_elite_frac=0.1,
        neighbor_k=12,
        min_pair_distance=0.015,
        path_tracking_weight=5.0,
        overlap_penalty_weight=1500.0,
        antipodal_penalty_weight=25.0,
        centerline_penalty_weight=6.0,
        support_axis_penalty_weight=12.0,
        force_reg_weight=1e-5,
        device=None,
        num_grasp_contacts=2,
        region_anchor_count=200,
        region_radius=0.08,
        region_max_points=256,
        region_contact_samples=5,
        top_region_pairs=3,
        preselect_region_pairs=200,
        friction_cone_edges=8,
        gwb_wrench_count=1000,
        beta=1.0,
        gamma=0.2,
        concavity_tol=None,
        normal_consistency_min=0.6,
        accessibility_alignment_min=-0.2,
        max_point_combination_eval=256,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=0.0,
        support_surface_normal_alignment_threshold=0.25,
        curvature_neighbor_k=8,
        region_max_mean_curvature=0.12,
        region_max_point_curvature=0.25,
        curvature_penalty_weight=1.5,
        static_equilibrium_cost_weight=0.0,
        static_equilibrium_residual_weight=0.0,
        gravity_world=(0.0, 0.0, -9.81),
        nlp_solver=None,
        static_nlp_solver=None,
    ):
        self.mesh_path = str(mesh_path)
        self.m = float(obj_mass)
        self.mu_arm_obj = float(arm_friction)
        self.K_contact = float(contact_stiffness)
        self.h = float(time_step)
        self.max_contacts = int(max_contacts)
        self.sample_budget = int(sample_num)
        self.pos_coef = float(pos_coef)
        self.ori_coef = float(ori_coef)
        self.force_reg_weight = float(force_reg_weight)
        self.min_pair_distance = float(min_pair_distance)
        self.overlap_penalty_weight = float(overlap_penalty_weight)
        self.antipodal_penalty_weight = float(antipodal_penalty_weight)
        self.centerline_penalty_weight = float(centerline_penalty_weight)
        self.support_axis_penalty_weight = float(support_axis_penalty_weight)

        # Legacy parameters are kept in the signature so the class remains easy to
        # drop into the existing codebase, but the new selection logic no longer
        # relies on the old MPPI rollout.
        self.mppi_samples = int(mppi_samples)
        self.mppi_iterations = int(mppi_iterations)
        self.mppi_horizon = int(mppi_horizon)
        self.mppi_lambda = float(mppi_lambda)
        self.mppi_noise_sigma = float(mppi_noise_sigma)
        self.mppi_noise_decay = float(mppi_noise_decay)
        self.mppi_elite_frac = float(mppi_elite_frac)
        self.neighbor_k = int(neighbor_k)
        self.path_tracking_weight = float(path_tracking_weight)
        self.device_ = device
        self.num_grasp_contacts = max(2, int(num_grasp_contacts))
        self.max_point_combination_eval = max(1, int(max_point_combination_eval))
        self.nlp_solver = _normalize_nlp_solver_name(nlp_solver, "LCC_SOLVER", "ipopt")
        self.static_nlp_solver = _normalize_nlp_solver_name(
            static_nlp_solver,
            "LCC_STATIC_SOLVER",
            "snopt",
        )

        self.pp = ProjectionPoint(self.mesh_path, scale_factors)
        self.mesh = self.pp.scaled_mesh
        self.mesh_centroid = np.asarray(self.mesh.centroid, dtype=np.float64)
        bounds = np.asarray(self.mesh.bounds, dtype=np.float64)
        self.mesh_diag = float(np.linalg.norm(bounds[1] - bounds[0]))
        self.mesh_diag = max(self.mesh_diag, 1e-3)
        self.characteristic_length = max(0.5 * self.mesh_diag, 1e-3)
        self.wrench_scale = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0 / self.characteristic_length,
                1.0 / self.characteristic_length,
                1.0 / self.characteristic_length,
            ],
            dtype=np.float64,
        )

        self.region_anchor_count = int(region_anchor_count)
        self.region_radius = min(float(region_radius), 0.35 * self.mesh_diag)
        self.region_radius = max(self.region_radius, 0.02 * self.mesh_diag)
        self.region_max_points = int(region_max_points)
        self.region_contact_samples = int(region_contact_samples)
        self.top_region_pairs = int(top_region_pairs)
        self.preselect_region_pairs = int(max(preselect_region_pairs, top_region_pairs))
        self.friction_cone_edges = int(max(4, friction_cone_edges))
        self.gwb_wrench_count = int(max(64, gwb_wrench_count))
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.normal_consistency_min = float(normal_consistency_min)
        self.accessibility_alignment_min = float(accessibility_alignment_min)
        self.concavity_tol = (
            float(concavity_tol)
            if concavity_tol is not None
            else 0.25 * self.region_radius
        )
        self.region_pair_min_distance = max(
            self.min_pair_distance, 0.5 * self.region_radius
        )
        self.support_surface_clearance = max(0.0, float(support_surface_clearance))
        self.support_surface_normal_alignment_threshold = float(
            np.clip(support_surface_normal_alignment_threshold, 0.0, 1.0)
        )
        self.curvature_neighbor_k = max(3, int(curvature_neighbor_k))
        self.region_max_mean_curvature = max(0.0, float(region_max_mean_curvature))
        self.region_max_point_curvature = max(0.0, float(region_max_point_curvature))
        self.curvature_penalty_weight = max(0.0, float(curvature_penalty_weight))
        self.static_equilibrium_cost_weight = max(
            0.0, float(static_equilibrium_cost_weight)
        )
        self.static_equilibrium_residual_weight = max(
            0.0, float(static_equilibrium_residual_weight)
        )
        self.gravity_world = _as_numpy(gravity_world).reshape(3)
        self.support_surface_point = None
        self.support_surface_normal = None
        self.set_support_surface(
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
        )

        self.sampling_frame = self._build_surface_frames()
        self.sample_point = np.asarray(self.sampling_frame["points"], dtype=np.float64)
        self.normal = np.asarray(
            self.sampling_frame["normals"], dtype=np.float64
        )  # inward normals
        self.t1 = np.asarray(self.sampling_frame["tangent1"], dtype=np.float64)
        self.t2 = np.asarray(self.sampling_frame["tangent2"], dtype=np.float64)
        self.outward_normal = -self.normal
        self.sample_num = int(self.sample_point.shape[0])
        self.point_idx = np.arange(self.sample_num, dtype=int)
        self.sample_kdtree = cKDTree(self.sample_point)
        self.point_curvature = self._estimate_point_curvature()

        self.convex_hull = self.mesh.convex_hull
        self.convex_hull_points = self._sample_convex_hull_surface()
        self.convex_hull_tree = cKDTree(self.convex_hull_points)

        self.J_tilde = np.zeros((4 * self.max_contacts, 6), dtype=np.float64)
        self.friction_primitives = self._build_friction_primitives()
        (
            self.disturbance_wrenches,
            self.disturbance_labels,
        ) = self._build_disturbance_wrenches()

        self.last_region_results = []
        self.last_available_idx = self.point_idx.copy()
        self.last_candidate_point_groups = []
        self.last_grasp_result = None
        self.last_static_equilibrium_result = None
        self.last_region_timing = {}
        self.last_contact_search_timing = {}
        self.precomputed_contact_search_cache = None
        self.enable_timing_prints = False
        self.last_object_pos = None
        self.last_object_rot = None
        self.force_closure_fn = None
        self.static_equilibrium_fn = None
        self.force_closure_solver_bundle = None
        self.static_equilibrium_solver_bundle = None
        self.force_closure_solver_backend = None
        self.static_equilibrium_solver_backend = None

        self._precompile_optimization_function()
        self._precompile_static_equilibrium_function()

    def _build_surface_frames(self):
        target_count = max(8, int(self.sample_budget))
        mesh = self.mesh

        try:
            points, face_idx = trimesh.sample.sample_surface_even(
                mesh, target_count, seed=0
            )
            if points.shape[0] < max(8, int(0.6 * target_count)):
                raise ValueError("surface-even sampling returned too few points")
        except Exception:
            points, face_idx = trimesh.sample.sample_surface(mesh, target_count, seed=0)

        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        face_idx = np.asarray(face_idx, dtype=int).reshape(-1)
        outward_normals = np.asarray(
            mesh.face_normals[face_idx], dtype=np.float64
        ).reshape(-1, 3)
        inward_normals = -_normalize(outward_normals, axis=1)

        normals = np.zeros_like(points)
        tangent1 = np.zeros_like(points)
        tangent2 = np.zeros_like(points)

        for idx in range(points.shape[0]):
            n, t1, t2 = _orthonormal_frame_from_normal(inward_normals[idx])
            normals[idx] = np.asarray(n, dtype=np.float64)
            tangent1[idx] = np.asarray(t1, dtype=np.float64)
            tangent2[idx] = np.asarray(t2, dtype=np.float64)

        return {
            "points": points,
            "normals": normals,
            "tangent1": tangent1,
            "tangent2": tangent2,
        }

    def _sample_convex_hull_surface(self):
        target_count = int(
            np.clip(max(1024, 2 * len(self.convex_hull.vertices)), 1024, 4096)
        )
        try:
            hull_points, _ = trimesh.sample.sample_surface_even(
                self.convex_hull, target_count
            )
        except Exception:
            hull_points = np.asarray(self.convex_hull.vertices, dtype=np.float64)
        return np.asarray(hull_points, dtype=np.float64)

    def _estimate_point_curvature(self):
        if self.sample_num <= 1:
            return np.zeros((self.sample_num,), dtype=np.float64)

        query_k = min(self.sample_num, self.curvature_neighbor_k + 1)
        _, neighbor_idx = self.sample_kdtree.query(self.sample_point, k=query_k)
        neighbor_idx = np.asarray(neighbor_idx, dtype=int)
        if neighbor_idx.ndim == 1:
            neighbor_idx = neighbor_idx.reshape(-1, 1)
        if neighbor_idx.shape[1] <= 1:
            return np.zeros((self.sample_num,), dtype=np.float64)

        neighbor_idx = neighbor_idx[:, 1:]
        neighbor_normals = self.normal[neighbor_idx]
        ref_normals = self.normal[:, None, :]
        normal_dot = np.clip(np.sum(ref_normals * neighbor_normals, axis=2), -1.0, 1.0)
        curvature = 0.5 * np.mean(1.0 - normal_dot, axis=1)
        return np.asarray(curvature, dtype=np.float64)

    def _filter_candidate_indices_by_curvature(self, candidate_idx, min_count=None):
        candidate_idx = self._sanitize_point_indices(candidate_idx)
        if candidate_idx.size == 0:
            return candidate_idx

        min_count = (
            self.num_grasp_contacts if min_count is None else max(1, int(min_count))
        )
        filtered_idx = candidate_idx[
            self.point_curvature[candidate_idx] <= self.region_max_point_curvature
        ]
        if filtered_idx.size >= min_count:
            return np.asarray(filtered_idx, dtype=int)
        return candidate_idx

    def set_timing_print_enabled(self, enabled):
        self.enable_timing_prints = bool(enabled)

    def _record_search_timing(self, name, timing):
        recorded = {"name": str(name)}
        for key, value in dict(timing).items():
            if isinstance(value, np.ndarray):
                recorded[key] = np.asarray(value, dtype=np.float64).copy()
            elif isinstance(value, (np.floating, float)):
                recorded[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                recorded[key] = int(value)
            elif isinstance(value, (str, bool)):
                recorded[key] = value
            else:
                recorded[key] = value

        self.last_contact_search_timing = recorded
        if name == "get_best_regions":
            self.last_region_timing = dict(recorded)

        if self.enable_timing_prints:
            if name == "get_best_regions":
                print(
                    "[timing:mlqp] get_best_regions "
                    f"total={float(recorded.get('wall_time', 0.0)):.4f}s "
                    f"filter={float(recorded.get('candidate_filter_time', 0.0)):.4f}s "
                    f"sample={float(recorded.get('sample_candidate_regions_time', 0.0)):.4f}s "
                    f"coarse={float(recorded.get('coarse_group_time', 0.0)):.4f}s "
                    f"gwb={float(recorded.get('gwb_score_time', 0.0)):.4f}s "
                    f"groups={int(recorded.get('group_count', 0))}"
                )
            elif name in {"get_best_grasp", "get_best_grasp_from_regions"}:
                print(
                    f"[timing:mlqp] {name} "
                    f"total={float(recorded.get('wall_time', 0.0)):.4f}s "
                    f"mode={recorded.get('mode', 'unknown')} "
                    f"prepare={float(recorded.get('prepare_candidate_entries_time', 0.0)):.4f}s "
                    f"eval={float(recorded.get('evaluate_candidates_time', 0.0)):.4f}s "
                    f"force_closure={float(recorded.get('force_closure_solve_time', 0.0)):.4f}s "
                    f"candidates={int(recorded.get('candidate_entry_count', 0))}"
                )
            elif name == "precompute_contact_search_cache":
                print(
                    "[timing:mlqp] precompute_contact_search_cache "
                    f"total={float(recorded.get('wall_time', 0.0)):.4f}s "
                    f"get_best_regions={float(recorded.get('get_best_regions_time', 0.0)):.4f}s "
                    f"prepare={float(recorded.get('prepare_candidate_entries_time', 0.0)):.4f}s "
                    f"rank={float(recorded.get('rank_candidate_entries_time', 0.0)):.4f}s "
                    f"force_closure={float(recorded.get('force_closure_solve_time', 0.0)):.4f}s "
                    f"candidates={int(recorded.get('candidate_entry_count_before_limit', 0))}"
                    f"->{int(recorded.get('candidate_entry_count', 0))}"
                )
        return recorded

    def _build_candidate_entries_from_region_groups(self, region_groups):
        normalized_groups = self._normalize_region_groups(region_groups)
        candidate_entries = []
        candidate_point_groups = []

        for region_group in normalized_groups:
            raw_sample_groups = region_group.get("region_sample_indices", [])
            sample_groups = self._prepare_contact_sample_groups(raw_sample_groups)
            sample_groups = [
                np.asarray(group, dtype=int).reshape(-1) for group in sample_groups
            ]
            candidate_point_groups.append(sample_groups)
            if not sample_groups or any(group.size == 0 for group in sample_groups):
                continue

            for contact_indices in itertools.product(
                *[group.tolist() for group in sample_groups]
            ):
                candidate_entries.append(
                    {
                        "contact_indices": np.asarray(contact_indices, dtype=int),
                        "region_group": region_group,
                    }
                )

        return normalized_groups, candidate_entries, candidate_point_groups

    def _select_top_candidate_entries_for_cache(
        self,
        candidate_entries,
        top_candidate_count=None,
        object_rot=None,
        external_wrench=None,
        target_pose=None,
        gravity_world=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        candidate_entries = list(candidate_entries)
        raw_count = int(len(candidate_entries))
        if raw_count == 0:
            return candidate_entries, {
                "candidate_entry_count_before_limit": 0,
                "candidate_entry_count_after_limit": 0,
                "evaluate_candidates_time": 0.0,
                "force_closure_solve_time": 0.0,
            }

        if top_candidate_count is None:
            limit = None
        else:
            limit = int(top_candidate_count)
            if limit <= 0:
                limit = None

        # Always evaluate force-closure for every candidate so the downstream
        # caller can rely on ``offline_force_closure_cost`` being present.
        # The early-return below is only a true skip when there is NOTHING to
        # evaluate; the old ``raw_count <= limit`` branch left entries without
        # a force-closure cost, which caused every pair to be filtered out.
        effective_limit = limit if limit is not None else raw_count
        effective_limit = max(effective_limit, 0)

        eval_t0 = time.perf_counter()
        force_closure_solve_time = 0.0
        ranked_entries = []

        for entry_idx, entry in enumerate(candidate_entries):
            result = self._evaluate_force_closure_candidate(
                entry["contact_indices"],
                region_group=entry.get("region_group", None),
                object_rot=object_rot,
                external_wrench=external_wrench,
                target_pose=target_pose,
                gravity_world=gravity_world,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
            force_closure_solve_time += float(result.get("solve_time", 0.0))

            ranked_entry = dict(entry)
            ranked_entry["offline_force_closure_total_cost"] = float(
                result.get("total_cost", float("inf"))
            )
            ranked_entry["offline_force_closure_cost"] = float(
                result.get("force_closure_cost", float("inf"))
            )
            ranked_entry["offline_region_score"] = float(
                result.get("region_score", 0.0)
            )
            ranked_entry["offline_min_contact_distance"] = float(
                result.get("min_contact_distance", 0.0)
            )
            ranked_entry["offline_force_closure_solver_status"] = result.get(
                "solver_status",
                None,
            )
            ranked_entry["offline_force_closure_solver_backend"] = result.get(
                "solver_backend",
                self.force_closure_solver_backend,
            )
            ranked_entries.append(
                (
                    (
                        ranked_entry["offline_force_closure_total_cost"],
                        -ranked_entry["offline_region_score"],
                        -ranked_entry["offline_min_contact_distance"],
                        int(entry_idx),
                    ),
                    ranked_entry,
                )
            )

        ranked_entries.sort(key=lambda item: item[0])
        limited_entries = [item[1] for item in ranked_entries[:effective_limit]]
        return limited_entries, {
            "candidate_entry_count_before_limit": raw_count,
            "candidate_entry_count_after_limit": int(len(limited_entries)),
            "evaluate_candidates_time": float(time.perf_counter() - eval_t0),
            "force_closure_solve_time": float(force_closure_solve_time),
        }

    def precompute_contact_search_cache(
        self,
        visible_face_idx=None,
        object_pos=None,
        object_rot=None,
        external_wrench=None,
        target_pose=None,
        gravity_world=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
        top_candidate_count=None,
    ):
        precompute_t0 = time.perf_counter()
        region_groups = self.get_best_regions(
            visible_face_idx=visible_face_idx,
            top_k=self.top_region_pairs,
            object_pos=object_pos,
            object_rot=object_rot,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )

        prepare_t0 = time.perf_counter()
        (
            prepared_region_groups,
            candidate_entries,
            candidate_point_groups,
        ) = self._build_candidate_entries_from_region_groups(region_groups)
        prepare_elapsed = time.perf_counter() - prepare_t0
        candidate_rank_info = {
            "candidate_entry_count_before_limit": int(len(candidate_entries)),
            "candidate_entry_count_after_limit": int(len(candidate_entries)),
            "evaluate_candidates_time": 0.0,
            "force_closure_solve_time": 0.0,
        }
        (
            candidate_entries,
            candidate_rank_info,
        ) = self._select_top_candidate_entries_for_cache(
            candidate_entries,
            top_candidate_count=top_candidate_count,
            object_rot=object_rot,
            external_wrench=external_wrench,
            target_pose=target_pose,
            gravity_world=gravity_world,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        wall_elapsed = time.perf_counter() - precompute_t0

        cache = {
            "region_groups": prepared_region_groups,
            "candidate_entries": candidate_entries,
            "candidate_point_groups": candidate_point_groups,
            "candidate_entry_count_before_limit": int(
                candidate_rank_info.get(
                    "candidate_entry_count_before_limit", len(candidate_entries)
                )
            ),
            "candidate_entry_count": int(
                candidate_rank_info.get(
                    "candidate_entry_count_after_limit", len(candidate_entries)
                )
            ),
            "candidate_entry_limit": (
                None
                if top_candidate_count is None or int(top_candidate_count) <= 0
                else int(top_candidate_count)
            ),
            "timing": self._record_search_timing(
                "precompute_contact_search_cache",
                {
                    "mode": "offline_precompute",
                    "get_best_regions_time": float(
                        self.last_region_timing.get("wall_time", 0.0)
                    ),
                    "prepare_candidate_entries_time": float(prepare_elapsed),
                    "rank_candidate_entries_time": float(
                        candidate_rank_info.get("evaluate_candidates_time", 0.0)
                    ),
                    "force_closure_solve_time": float(
                        candidate_rank_info.get("force_closure_solve_time", 0.0)
                    ),
                    "candidate_entry_count_before_limit": int(
                        candidate_rank_info.get(
                            "candidate_entry_count_before_limit", len(candidate_entries)
                        )
                    ),
                    "candidate_entry_count": int(
                        candidate_rank_info.get(
                            "candidate_entry_count_after_limit", len(candidate_entries)
                        )
                    ),
                    "region_group_count": len(prepared_region_groups),
                    "wall_time": float(wall_elapsed),
                },
            ),
        }
        self.precomputed_contact_search_cache = cache
        self.last_region_results = prepared_region_groups
        self.last_candidate_point_groups = candidate_point_groups
        return cache

    @staticmethod
    def _get_casadi_solver_config(solver_name, tol):
        solver_name = str(solver_name).strip().lower()
        p_opts = {"print_time": False, "jit": False}
        if solver_name == "snopt":
            s_opts = {
                "Major print level": 0,
                "Minor print level": 0,
                "Print file": 0,
                "Summary file": 0,
                "print_level": 0,
                "Major iterations limit": 300,
                "Minor iterations limit": 300,
            }
        else:
            solver_name = "ipopt"
            s_opts = {
                "max_iter": 300,
                "tol": float(tol),
                "print_level": 0,
                "sb": "yes",
            }
        return solver_name, p_opts, s_opts

    def _get_solver_config(self, solver_name=None):
        solver_name = self.nlp_solver if solver_name is None else solver_name
        if solver_name == "acados":
            solver_name = "ipopt"
        return self._get_casadi_solver_config(solver_name, tol=1e-6)

    def _configure_solver(self, opti, solver_name=None):
        solver_name, p_opts, s_opts = self._get_solver_config(solver_name=solver_name)
        opti.solver(solver_name, p_opts, s_opts)

    def _get_static_solver_config(self, solver_name=None):
        solver_name = self.static_nlp_solver if solver_name is None else solver_name
        if solver_name == "acados":
            solver_name = "ipopt"
        return self._get_casadi_solver_config(solver_name, tol=1e-8)

    def _configure_static_solver(self, opti, solver_name=None):
        solver_name, p_opts, s_opts = self._get_static_solver_config(
            solver_name=solver_name
        )
        opti.solver(solver_name, p_opts, s_opts)

    @staticmethod
    def _get_snopt_v1_options():
        p_opts = {
            "print_time": False,
            "jit": False,
            "snopt": {
                "Total real workspace": 500000,
                "Total integer workspace": 500000,
                "Total character workspace": 500000,
            },
        }
        s_opts = {
            "Major print level": 0,
            "Minor print level": 0,
            "Print file": 1,
            "Summary file": 0,
            "print_level": 0,
        }
        return p_opts, s_opts

    def _configure_snopt_v1(self, opti):
        p_opts, s_opts = self._get_snopt_v1_options()
        opti.solver("snopt", p_opts, s_opts)

    def update_Jacobian(self, J_tilde=None):
        required_rows = 4 * self.max_contacts
        if J_tilde is None:
            return self.J_tilde

        J_tilde = np.asarray(J_tilde, dtype=np.float64)[:, :6]
        padded = np.zeros((required_rows, 6), dtype=np.float64)
        valid_rows = min(required_rows, J_tilde.shape[0])
        padded[:valid_rows] = J_tilde[:valid_rows]
        self.J_tilde = padded
        return self.J_tilde

    def _build_friction_primitives(self):
        theta = np.linspace(
            0.0, 2.0 * np.pi, self.friction_cone_edges, endpoint=False, dtype=np.float64
        )
        tangential = self.mu_arm_obj * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        primitives = np.column_stack([np.ones_like(theta), tangential])
        return np.asarray(primitives, dtype=np.float64)

    def _build_friction_cone_normals(self):
        """Inward face-normals of the polyhedral friction cone.

        The cone is generated by the extreme rays ``friction_primitives`` (each
        row is ``[1, mu*cos(theta), mu*sin(theta)]``), i.e.
        ``C = { sum_j alpha_j p_j : alpha_j >= 0 }``.  The inward normal of the
        face spanned by adjacent rays ``p_j, p_{j+1}`` is their cross product
        ``p_j x p_{j+1}`` -- its ``fn`` component is ``mu**2 * sin(2pi/P) > 0``,
        confirming it points into the cone.

        Returns
        -------
        normals : (P, 3) array, one inward normal per face (``P`` = n edges).
        """
        prim = self.friction_primitives  # (P, 3)
        P = prim.shape[0]
        normals = np.zeros((P, 3), dtype=np.float64)
        for j in range(P):
            n = np.cross(prim[j], prim[(j + 1) % P])
            if n[0] < 0.0:  # safety: enforce inward (fn > 0)
                n = -n
            normals[j] = n
        return normals

    def _build_qp_cone_matrix(self, fn_lb, fn_ub):
        """Build the QP inequality matrix for the polyhedral friction cone.

        Parameters
        ----------
        fn_lb, fn_ub : float
            Lower / upper bound on each contact's normal force.

        Returns
        -------
        A : (M, 3*nc)      ``blower <= A x <= bupper``
        blower, bupper : (M,)
        """
        nc = self.num_grasp_contacts
        n = 3 * nc
        normals = self._build_friction_cone_normals()  # (P, 3)
        P = normals.shape[0]

        rows, lower, upper = [], [], []

        # Friction cone:  n_j . f_c >= 0   (inside the polyhedral cone)
        for c in range(nc):
            base = 3 * c
            for j in range(P):
                row = np.zeros(n, dtype=np.float64)
                row[base : base + 3] = normals[j]
                rows.append(row)
                lower.append(0.0)
                upper.append(1e30)

        # Normal-force box:  fn_lb <= fn_c <= fn_ub
        for c in range(nc):
            row = np.zeros(n, dtype=np.float64)
            row[3 * c] = 1.0
            rows.append(row)
            lower.append(float(fn_lb))
            upper.append(float(fn_ub))

        # Normal-force sum:  sum_c fn_c >= gamma   (==  -sum fn_c <= -gamma)
        row = np.zeros(n, dtype=np.float64)
        for c in range(nc):
            row[3 * c] = -1.0
        rows.append(row)
        lower.append(-1e30)
        upper.append(-float(self.gamma))

        A = np.vstack(rows).astype(np.float64)
        return A, np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)

    def _get_qp_cone_matrix(self, fn_lb, fn_ub):
        """Cache the cone matrix -- it depends only on nc, mu, gamma, (fn_lb,fn_ub)."""
        cache = getattr(self, "_qp_cone_cache", None)
        if cache is None:
            cache = {}
            self._qp_cone_cache = cache
        key = (float(fn_lb), float(fn_ub))
        if key not in cache:
            cache[key] = self._build_qp_cone_matrix(fn_lb, fn_ub)
        return cache[key]

    @staticmethod
    def _solve_qp_daqp(H, f, A, bupper, blower):
        """Solve ``min 0.5 x'Hx + f'x  s.t. blower <= A x <= bupper`` via DAQP.

        Variable bounds are folded into ``A`` as identity rows by the caller, so
        ``A``, ``bupper``, ``blower`` are all consistent in row-count.

        Returns (x, status) with status == 1 on success.
        """
        try:
            import daqp
        except ImportError as exc:
            raise RuntimeError(
                "DAQP is required for the 'daqp' backend. "
                "Install it with: pip install daqp"
            ) from exc

        H = np.ascontiguousarray(np.asarray(H, dtype=np.float64))
        f = np.ascontiguousarray(np.asarray(f, dtype=np.float64).reshape(-1))
        A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
        bupper = np.ascontiguousarray(np.asarray(bupper, dtype=np.float64).reshape(-1))
        blower = np.ascontiguousarray(np.asarray(blower, dtype=np.float64).reshape(-1))
        sense = np.zeros(A.shape[0], dtype=np.int32)

        try:
            x, _fval, exitflag, _info = daqp.solve(H, f, A, bupper, blower, sense)
        except Exception:
            return np.zeros(H.shape[0], dtype=np.float64), -99
        status = 1 if int(exitflag) == 1 else int(exitflag)
        return np.asarray(x, dtype=np.float64).reshape(-1), status

    def _build_disturbance_wrenches(self):
        base = np.eye(6, dtype=np.float64)
        disturbances = np.vstack([base, -base])
        labels = [
            "+Fx",
            "+Fy",
            "+Fz",
            "+Tx",
            "+Ty",
            "+Tz",
            "-Fx",
            "-Fy",
            "-Fz",
            "-Tx",
            "-Ty",
            "-Tz",
        ]
        return disturbances, labels

    def _precompile_optimization_function(self):
        if self.nlp_solver == "daqp":
            # Pure-QP backend: no CasADi function is built; the solver is
            # dispatched at call-time via the "daqp_qp" backend tag.
            self.force_closure_solver_bundle = None
            self.force_closure_fn = None
            self.force_closure_solver_backend = "daqp_qp"
            return

        if self.nlp_solver == "acados":
            try:
                bundle = self._build_force_closure_acados_bundle()
            except Exception as exc:
                _warn_acados_casadi_fallback(str(exc))
                bundle = self._build_force_closure_casadi_function(solver_name="ipopt")
                bundle["backend"] = "casadi_ipopt_fallback_from_acados"
            self.force_closure_solver_bundle = bundle
            self.force_closure_fn = bundle.get("fn", None)
            self.force_closure_solver_backend = str(bundle.get("backend", "acados"))
            return

        bundle = self._build_force_closure_casadi_function()
        self.force_closure_solver_bundle = bundle
        self.force_closure_fn = bundle["fn"]
        self.force_closure_solver_backend = str(bundle["backend"])

    def _precompile_static_equilibrium_function(self):
        if self.static_nlp_solver == "daqp":
            self.static_equilibrium_solver_bundle = None
            self.static_equilibrium_fn = None
            self.static_equilibrium_solver_backend = "daqp_qp"
            return

        if self.static_nlp_solver == "acados":
            try:
                bundle = self._build_static_equilibrium_acados_bundle()
            except Exception as exc:
                _warn_acados_casadi_fallback(str(exc))
                bundle = self._build_static_equilibrium_casadi_function(
                    solver_name="ipopt"
                )
                bundle["backend"] = "casadi_ipopt_fallback_from_acados"
            self.static_equilibrium_solver_bundle = bundle
            self.static_equilibrium_fn = bundle.get("fn", None)
            self.static_equilibrium_solver_backend = str(
                bundle.get("backend", "acados")
            )
            return

        bundle = self._build_static_equilibrium_casadi_function()
        self.static_equilibrium_solver_bundle = bundle
        self.static_equilibrium_fn = bundle["fn"]
        self.static_equilibrium_solver_backend = str(bundle["backend"])

    def _build_force_closure_casadi_function(self, solver_name=None):
        opti = cs.Opti()

        n_dist = int(self.disturbance_wrenches.shape[0])
        G = opti.parameter(6, 3 * self.num_grasp_contacts)
        f = opti.variable(3 * self.num_grasp_contacts, n_dist)
        disturbance_matrix = cs.DM(self.disturbance_wrenches.T)

        opti.set_initial(f, 0.1)

        cost_terms = []
        response_terms = []
        objective = 0

        for disturbance_idx in range(n_dist):
            wrench_response = G @ f[:, disturbance_idx]
            residual = (
                self.beta * disturbance_matrix[:, disturbance_idx] - wrench_response
            )
            response_terms.append(wrench_response)
            cost_terms.append(cs.sumsqr(residual))
            objective += cost_terms[-1]

            normal_force_sum = 0
            for contact_idx in range(self.num_grasp_contacts):
                f_contact = f[3 * contact_idx : 3 * (contact_idx + 1), disturbance_idx]
                self._add_friction_cone_constraints(opti, f_contact)
                normal_force_sum += f_contact[0]
            opti.subject_to(normal_force_sum >= self.gamma)

        objective += self.force_reg_weight * cs.sumsqr(f)
        opti.minimize(objective)

        resolved_solver_name, _, _ = self._get_solver_config(solver_name=solver_name)
        self._configure_solver(opti, solver_name=resolved_solver_name)

        return {
            "backend": f"casadi_{resolved_solver_name}",
            "fn": opti.to_function(
                "force_closure_fn",
                [G],
                [f, objective, cs.vertcat(*cost_terms), cs.hcat(response_terms)],
                ["G"],
                ["f_opt", "cost", "cost_terms", "wrench_response"],
            ),
        }

    def _build_static_equilibrium_casadi_function(self, solver_name=None):
        opti = cs.Opti()

        G = opti.parameter(6, 3 * self.num_grasp_contacts)
        wrench_ext = opti.parameter(6)
        f = opti.variable(3 * self.num_grasp_contacts)

        opti.set_initial(f, 0.05)

        residual = wrench_ext + G @ f
        objective = cs.sumsqr(residual) + self.force_reg_weight * cs.sumsqr(f)
        opti.minimize(objective)

        normal_force_sum = 0
        for contact_idx in range(self.num_grasp_contacts):
            f_contact = f[3 * contact_idx : 3 * (contact_idx + 1)]
            self._add_static_friction_cone_constraints(opti, f_contact)
            normal_force_sum += f_contact[0]
        opti.subject_to(normal_force_sum >= self.gamma)

        resolved_solver_name, _, _ = self._get_static_solver_config(
            solver_name=solver_name
        )
        self._configure_static_solver(opti, solver_name=resolved_solver_name)

        return {
            "backend": f"casadi_{resolved_solver_name}",
            "fn": opti.to_function(
                "static_equilibrium_fn",
                [G, wrench_ext],
                [f, residual, objective],
                ["G", "wrench_ext"],
                ["f_opt", "residual", "cost"],
            ),
        }

    def _pack_force_closure_parameter_vector(self, G):
        return np.asarray(G, dtype=np.float64).reshape(-1, order="F")

    def _pack_static_equilibrium_parameter_vector(self, G, wrench_ext):
        return np.concatenate(
            [
                np.asarray(G, dtype=np.float64).reshape(-1, order="F"),
                np.asarray(wrench_ext, dtype=np.float64).reshape(6),
            ]
        )

    def _project_contact_force_slice(self, force_slice, normal_lb, normal_ub):
        projected = (
            np.asarray(force_slice, dtype=np.float64)
            .reshape(3 * self.num_grasp_contacts)
            .copy()
        )
        for contact_idx in range(self.num_grasp_contacts):
            base = 3 * contact_idx
            normal_force = float(np.clip(projected[base], normal_lb, normal_ub))
            tangential = projected[base + 1 : base + 3]
            tangential_norm = float(np.linalg.norm(tangential))
            tangential_limit = max(self.mu_arm_obj * normal_force, 0.0)
            if tangential_norm > max(tangential_limit, _EPS):
                tangential = tangential * (tangential_limit / tangential_norm)
            elif tangential_limit <= _EPS:
                tangential = np.zeros_like(tangential)
            projected[base] = normal_force
            projected[base + 1 : base + 3] = tangential
        return projected

    def _enforce_normal_force_sum(self, force_slice, normal_lb, normal_ub):
        projected = self._project_contact_force_slice(force_slice, normal_lb, normal_ub)
        normal_indices = np.arange(0, projected.size, 3, dtype=int)
        normal_forces = projected[normal_indices]
        deficit = float(self.gamma - np.sum(normal_forces))
        if deficit > _EPS:
            headroom = np.maximum(normal_ub - normal_forces, 0.0)
            total_headroom = float(np.sum(headroom))
            if total_headroom > _EPS:
                projected[normal_indices] = np.clip(
                    normal_forces + deficit * headroom / total_headroom,
                    normal_lb,
                    normal_ub,
                )
                projected = self._project_contact_force_slice(
                    projected, normal_lb, normal_ub
                )
        return projected

    def _project_force_closure_initial_guess(self, x_flat):
        x_matrix = np.asarray(x_flat, dtype=np.float64).reshape(
            3 * self.num_grasp_contacts,
            int(self.disturbance_wrenches.shape[0]),
            order="F",
        )
        for disturbance_idx in range(x_matrix.shape[1]):
            x_matrix[:, disturbance_idx] = self._enforce_normal_force_sum(
                x_matrix[:, disturbance_idx],
                normal_lb=0.0,
                normal_ub=1.0,
            )
        return x_matrix.reshape(-1, order="F")

    def _project_static_equilibrium_initial_guess(self, x_flat):
        return self._enforce_normal_force_sum(
            x_flat,
            normal_lb=0.001,
            normal_ub=2.0,
        )

    def _build_force_closure_acados_bundle(self):
        (
            AcadosModel,
            AcadosOcp,
            AcadosOcpSolver,
            ACADOS_INFTY,
        ) = _import_acados_template_symbols()

        n_dist = int(self.disturbance_wrenches.shape[0])
        force_dim = 3 * self.num_grasp_contacts
        total_force_dim = force_dim * n_dist

        f_flat = cs.SX.sym("f_flat", total_force_dim)
        p = cs.SX.sym("p", 6 * force_dim)
        G = cs.reshape(p, 6, force_dim)
        f = cs.reshape(f_flat, force_dim, n_dist)
        disturbance_matrix = cs.DM(self.disturbance_wrenches.T)

        objective = 0
        cost_terms = []
        response_terms = []
        nonlinear_constraints = []
        normal_force_indices = []

        for disturbance_idx in range(n_dist):
            force_vec = f[:, disturbance_idx]
            wrench_response = G @ force_vec
            residual = (
                self.beta * disturbance_matrix[:, disturbance_idx] - wrench_response
            )
            response_terms.append(wrench_response)
            cost_term = cs.sumsqr(residual)
            cost_terms.append(cost_term)
            objective += cost_term

            normal_force_sum = 0
            for contact_idx in range(self.num_grasp_contacts):
                base = disturbance_idx * force_dim + 3 * contact_idx
                normal_force_indices.append(base)
                force_local = force_vec[3 * contact_idx : 3 * (contact_idx + 1)]
                nonlinear_constraints.append(
                    cs.sumsqr(force_local[1:3])
                    - (self.mu_arm_obj * force_local[0]) ** 2
                )
                normal_force_sum += force_local[0]
            nonlinear_constraints.append(self.gamma - normal_force_sum)

        objective += self.force_reg_weight * cs.sumsqr(f_flat)

        model = AcadosModel()
        model.name = (
            f"mlqp_point_v2_force_closure_{_ACADOS_EXPORT_VERSION}_"
            f"c{self.num_grasp_contacts}_d{n_dist}"
        )
        model.x = f_flat
        model.u = cs.SX.sym("u", 0, 0)
        model.disc_dyn_expr = f_flat
        model.p = p
        model.cost_expr_ext_cost_e = objective
        model.con_h_expr_e = cs.vertcat(*nonlinear_constraints)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros((6 * force_dim,), dtype=np.float64)
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.constraints.idxbx_e = np.asarray(normal_force_indices, dtype=np.int64)
        ocp.constraints.lbx_e = np.zeros((len(normal_force_indices),), dtype=np.float64)
        ocp.constraints.ubx_e = np.ones((len(normal_force_indices),), dtype=np.float64)
        ocp.constraints.lh_e = -ACADOS_INFTY * np.ones(
            (len(nonlinear_constraints),), dtype=np.float64
        )
        ocp.constraints.uh_e = np.zeros((len(nonlinear_constraints),), dtype=np.float64)

        ocp.solver_options.N_horizon = 0
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.regularize_method = "MIRROR"
        ocp.solver_options.nlp_solver_ext_qp_res = 1
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.qp_solver_iter_max = 400
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.print_level = 0

        code_export_directory = os.path.join("/tmp", f"{model.name}_codegen")
        json_file = os.path.join(code_export_directory, f"{model.name}.json")
        shared_lib_path = os.path.join(
            code_export_directory,
            f"libacados_ocp_solver_{model.name}{_shared_lib_ext()}",
        )
        os.makedirs(code_export_directory, exist_ok=True)
        ocp.code_gen_opts.code_export_directory = code_export_directory

        can_reuse_existing_solver = os.path.isfile(json_file) and os.path.isfile(
            shared_lib_path
        )
        if can_reuse_existing_solver:
            solver = AcadosOcpSolver(
                ocp,
                json_file=json_file,
                generate=False,
                build=False,
                check_reuse_possible=False,
                verbose=False,
            )
        else:
            try:
                _ensure_acados_renderer_available()
                solver = AcadosOcpSolver(
                    ocp,
                    json_file=json_file,
                    generate=True,
                    build=True,
                    check_reuse_possible=True,
                    verbose=False,
                )
            except Exception as exc:
                _warn_acados_casadi_fallback(str(exc))
                fallback_bundle = self._build_force_closure_casadi_function(
                    solver_name="ipopt"
                )
                fallback_bundle["backend"] = "casadi_ipopt_fallback_from_acados"
                return fallback_bundle

        eval_fun = cs.Function(
            f"{model.name}_eval",
            [f_flat, p],
            [f, objective, cs.vertcat(*cost_terms), cs.hcat(response_terms)],
        )

        return {
            "backend": "acados",
            "solver": solver,
            "eval_fun": eval_fun,
            "last_x_solution": np.full((total_force_dim,), 0.1, dtype=np.float64),
        }

    def _build_static_equilibrium_acados_bundle(self):
        (
            AcadosModel,
            AcadosOcp,
            AcadosOcpSolver,
            ACADOS_INFTY,
        ) = _import_acados_template_symbols()

        force_dim = 3 * self.num_grasp_contacts
        f_flat = cs.SX.sym("f_flat", force_dim)
        p = cs.SX.sym("p", 6 * force_dim + 6)
        G = cs.reshape(p[: 6 * force_dim], 6, force_dim)
        wrench_ext = p[6 * force_dim : 6 * force_dim + 6]

        residual = wrench_ext + G @ f_flat
        objective = cs.sumsqr(residual) + self.force_reg_weight * cs.sumsqr(f_flat)

        nonlinear_constraints = []
        normal_force_indices = []
        normal_force_sum = 0
        for contact_idx in range(self.num_grasp_contacts):
            base = 3 * contact_idx
            normal_force_indices.append(base)
            force_local = f_flat[base : base + 3]
            nonlinear_constraints.append(
                cs.sumsqr(force_local[1:3]) - (self.mu_arm_obj * force_local[0]) ** 2
            )
            normal_force_sum += force_local[0]
        nonlinear_constraints.append(self.gamma - normal_force_sum)

        model = AcadosModel()
        model.name = (
            f"mlqp_point_v2_static_equilibrium_{_ACADOS_EXPORT_VERSION}_"
            f"c{self.num_grasp_contacts}"
        )
        model.x = f_flat
        model.u = cs.SX.sym("u", 0, 0)
        model.disc_dyn_expr = f_flat
        model.p = p
        model.cost_expr_ext_cost_e = objective
        model.con_h_expr_e = cs.vertcat(*nonlinear_constraints)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros((6 * force_dim + 6,), dtype=np.float64)
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.constraints.idxbx_e = np.asarray(normal_force_indices, dtype=np.int64)
        ocp.constraints.lbx_e = np.full(
            (len(normal_force_indices),), 0.001, dtype=np.float64
        )
        ocp.constraints.ubx_e = np.full(
            (len(normal_force_indices),), 2.0, dtype=np.float64
        )
        ocp.constraints.lh_e = -ACADOS_INFTY * np.ones(
            (len(nonlinear_constraints),), dtype=np.float64
        )
        ocp.constraints.uh_e = np.zeros((len(nonlinear_constraints),), dtype=np.float64)

        ocp.solver_options.N_horizon = 0
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.regularize_method = "MIRROR"
        ocp.solver_options.nlp_solver_ext_qp_res = 1
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.qp_solver_iter_max = 400
        ocp.solver_options.tol = 1e-6
        ocp.solver_options.print_level = 0

        code_export_directory = os.path.join("/tmp", f"{model.name}_codegen")
        json_file = os.path.join(code_export_directory, f"{model.name}.json")
        shared_lib_path = os.path.join(
            code_export_directory,
            f"libacados_ocp_solver_{model.name}{_shared_lib_ext()}",
        )
        os.makedirs(code_export_directory, exist_ok=True)
        ocp.code_gen_opts.code_export_directory = code_export_directory

        can_reuse_existing_solver = os.path.isfile(json_file) and os.path.isfile(
            shared_lib_path
        )
        if can_reuse_existing_solver:
            solver = AcadosOcpSolver(
                ocp,
                json_file=json_file,
                generate=False,
                build=False,
                check_reuse_possible=False,
                verbose=False,
            )
        else:
            try:
                _ensure_acados_renderer_available()
                solver = AcadosOcpSolver(
                    ocp,
                    json_file=json_file,
                    generate=True,
                    build=True,
                    check_reuse_possible=True,
                    verbose=False,
                )
            except Exception as exc:
                _warn_acados_casadi_fallback(str(exc))
                fallback_bundle = self._build_static_equilibrium_casadi_function(
                    solver_name="ipopt"
                )
                fallback_bundle["backend"] = "casadi_ipopt_fallback_from_acados"
                return fallback_bundle

        eval_fun = cs.Function(
            f"{model.name}_eval",
            [f_flat, p],
            [f_flat, residual, objective],
        )

        return {
            "backend": "acados",
            "solver": solver,
            "eval_fun": eval_fun,
            "last_x_solution": np.full((force_dim,), 0.05, dtype=np.float64),
        }

    def _solve_force_closure_with_acados(self, G):
        bundle = self.force_closure_solver_bundle
        param_vector = self._pack_force_closure_parameter_vector(G)
        solver = bundle["solver"]
        x_initial = self._project_force_closure_initial_guess(bundle["last_x_solution"])

        status = None
        sqp_iter = None
        try:
            solver.set(0, "p", param_vector)
            solver.set(0, "x", x_initial)
            status = solver.solve()
            x_solution = np.asarray(solver.get(0, "x"), dtype=np.float64).reshape(-1)
            try:
                sqp_iter = int(solver.get_stats("sqp_iter"))
            except Exception:
                sqp_iter = None
            solver_status = _format_acados_status(status, sqp_iter=sqp_iter)
        except Exception:
            x_solution = x_initial
            solver_status = "ACADOS_EXCEPTION"

        f_opt, cost, cost_terms, wrench_response = bundle["eval_fun"](
            x_solution, param_vector
        )
        bundle["last_x_solution"] = np.asarray(x_solution, dtype=np.float64).reshape(-1)
        return {
            "f_opt": f_opt,
            "cost": cost,
            "cost_terms": cost_terms,
            "wrench_response": wrench_response,
            "solver_status": solver_status,
            "solver_backend": str(bundle.get("backend", "acados")),
        }

    def _solve_static_equilibrium_with_acados(self, G, wrench_ext):
        bundle = self.static_equilibrium_solver_bundle
        param_vector = self._pack_static_equilibrium_parameter_vector(G, wrench_ext)
        solver = bundle["solver"]
        x_initial = self._project_static_equilibrium_initial_guess(
            bundle["last_x_solution"]
        )

        status = None
        sqp_iter = None
        try:
            solver.set(0, "p", param_vector)
            solver.set(0, "x", x_initial)
            status = solver.solve()
            x_solution = np.asarray(solver.get(0, "x"), dtype=np.float64).reshape(-1)
            try:
                sqp_iter = int(solver.get_stats("sqp_iter"))
            except Exception:
                sqp_iter = None
            solver_status = _format_acados_status(status, sqp_iter=sqp_iter)
        except Exception:
            x_solution = x_initial
            solver_status = "ACADOS_EXCEPTION"

        f_opt, residual, cost = bundle["eval_fun"](x_solution, param_vector)
        bundle["last_x_solution"] = np.asarray(x_solution, dtype=np.float64).reshape(-1)
        return {
            "f_opt": f_opt,
            "residual": residual,
            "cost": cost,
            "solver_status": solver_status,
            "solver_backend": str(bundle.get("backend", "acados")),
        }

    def _add_friction_cone_constraints(self, opti, force_local):
        opti.subject_to(force_local[0] >= 0.0)
        opti.subject_to(force_local[0] <= 1.0)
        opti.subject_to(
            cs.sumsqr(force_local[1:3]) <= (self.mu_arm_obj * force_local[0]) ** 2
        )

    def _add_static_friction_cone_constraints(self, opti, force_local):
        opti.subject_to(force_local[0] >= 0.001)
        opti.subject_to(force_local[0] <= 2.0)
        opti.subject_to(
            cs.sumsqr(force_local[1:3]) <= (self.mu_arm_obj * force_local[0]) ** 2
        )

    def _sanitize_point_indices(self, indices):
        if indices is None:
            return self.point_idx.copy()

        indices = np.asarray(indices, dtype=int).reshape(-1)
        if indices.size == 0:
            return np.zeros((0,), dtype=int)
        indices = indices[(indices >= 0) & (indices < self.sample_num)]
        if indices.size == 0:
            return np.zeros((0,), dtype=int)
        return np.unique(indices)

    def set_support_surface(
        self,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        if support_surface_clearance is not None:
            self.support_surface_clearance = max(0.0, float(support_surface_clearance))
        if support_surface_normal_alignment_threshold is not None:
            self.support_surface_normal_alignment_threshold = float(
                np.clip(support_surface_normal_alignment_threshold, 0.0, 1.0)
            )

        if support_surface_point is None and support_surface_normal is None:
            if (
                support_surface_clearance is None
                and support_surface_normal_alignment_threshold is None
            ):
                self.support_surface_point = None
                self.support_surface_normal = None
            return

        normal = self.support_surface_normal
        if support_surface_normal is not None:
            normal = _normalize(_as_numpy(support_surface_normal).reshape(3))
        if normal is None:
            raise ValueError(
                "support_surface_normal is required when configuring a support surface."
            )
        if np.linalg.norm(normal) < 1e-8:
            raise ValueError("support_surface_normal must be a non-zero 3D vector.")

        self.support_surface_point = (
            None
            if support_surface_point is None
            else _as_numpy(support_surface_point).reshape(3)
        )
        self.support_surface_normal = normal

    def _resolve_support_surface(
        self,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        point = self.support_surface_point
        if support_surface_point is not None:
            point = _as_numpy(support_surface_point).reshape(3)

        normal = self.support_surface_normal
        if support_surface_normal is not None:
            normal = _normalize(_as_numpy(support_surface_normal).reshape(3))
        if normal is None:
            if support_surface_point is not None:
                raise ValueError(
                    "support_surface_normal is required when support_surface_point is provided."
                )
            return None
        if np.linalg.norm(normal) < 1e-8:
            raise ValueError("support_surface_normal must be a non-zero 3D vector.")

        clearance = self.support_surface_clearance
        if support_surface_clearance is not None:
            clearance = max(0.0, float(support_surface_clearance))

        normal_alignment_threshold = self.support_surface_normal_alignment_threshold
        if support_surface_normal_alignment_threshold is not None:
            normal_alignment_threshold = float(
                np.clip(support_surface_normal_alignment_threshold, 0.0, 1.0)
            )

        return {
            "point": None if point is None else np.asarray(point, dtype=np.float64),
            "normal": np.asarray(normal, dtype=np.float64),
            "clearance": float(clearance),
            "normal_alignment_threshold": float(normal_alignment_threshold),
        }

    def get_contact_candidate_indices(
        self,
        visible_face_idx=None,
        object_pos=None,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        candidate_idx = self._sanitize_point_indices(visible_face_idx)
        if candidate_idx.size == 0:
            return candidate_idx

        support_surface = self._resolve_support_surface(
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        if support_surface is None:
            return candidate_idx

        if object_pos is not None and object_rot is not None:
            self.last_object_pos = _as_numpy(object_pos).reshape(3)
            self.last_object_rot = _as_numpy(object_rot).reshape(3, 3)
        elif self.last_object_pos is not None and self.last_object_rot is not None:
            object_pos = self.last_object_pos
            object_rot = self.last_object_rot

        if object_pos is None or object_rot is None:
            raise ValueError(
                "object_pos and object_rot are required when filtering contact candidates with a support surface."
            )

        object_pos = _as_numpy(object_pos).reshape(3)
        object_rot = _as_numpy(object_rot).reshape(3, 3)
        support_normal = support_surface["normal"]

        candidate_points_world = (
            object_rot @ self.sample_point[candidate_idx].T
        ).T + object_pos[None, :]
        candidate_outward_world = (object_rot @ self.outward_normal[candidate_idx].T).T

        if support_surface["point"] is None:
            # If no plane point is given, treat the lowest points along the support
            # normal as the current support band.
            all_points_world = (object_rot @ self.sample_point.T).T + object_pos[
                None, :
            ]
            support_level = float(np.min(all_points_world @ support_normal))
        else:
            support_level = float(np.dot(support_surface["point"], support_normal))

        signed_height = candidate_points_world @ support_normal - support_level
        support_facing = (candidate_outward_world @ support_normal) <= -support_surface[
            "normal_alignment_threshold"
        ]
        blocked_mask = (
            signed_height <= support_surface["clearance"] + _EPS
        ) & support_facing
        return candidate_idx[~blocked_mask]

    def _build_region_from_anchor(
        self, anchor_idx, candidate_idx, candidate_tree, region_radius, max_points
    ):
        anchor_idx = int(anchor_idx)
        anchor_local = int(np.where(candidate_idx == anchor_idx)[0][0])
        neighbor_local = candidate_tree.query_ball_point(
            self.sample_point[anchor_idx], r=region_radius
        )
        neighbor_local = np.asarray(neighbor_local, dtype=int).reshape(-1)
        if neighbor_local.size == 0:
            neighbor_local = np.array([anchor_local], dtype=int)

        region_idx = candidate_idx[neighbor_local]
        region_points = self.sample_point[region_idx]
        dist2 = np.sum(
            (region_points - self.sample_point[anchor_idx][None]) ** 2, axis=1
        )
        order = np.argsort(dist2)
        region_idx = region_idx[order[:max_points]]
        region_points = self.sample_point[region_idx]

        region_outward = self.outward_normal[region_idx]
        mean_outward = _normalize(np.mean(region_outward, axis=0))
        mean_inward = -mean_outward
        normal_consistency = float(
            np.mean(np.clip(region_outward @ mean_outward, -1.0, 1.0))
        )
        center = np.mean(region_points, axis=0)
        radial = _normalize(center - self.mesh_centroid)
        accessibility = float(np.clip(np.dot(mean_outward, radial), -1.0, 1.0))
        hull_distance = float(
            np.mean(self.convex_hull_tree.query(region_points, k=1)[0])
        )
        region_curvature = self.point_curvature[region_idx]
        curvature_mean = float(np.mean(region_curvature))
        curvature_max = float(np.max(region_curvature))
        anchor_curvature = float(self.point_curvature[anchor_idx])
        is_high_curvature = (
            curvature_mean > self.region_max_mean_curvature
            or curvature_max > self.region_max_point_curvature
        )

        is_concave = (
            hull_distance > self.concavity_tol
            or normal_consistency < self.normal_consistency_min
            or accessibility < self.accessibility_alignment_min
            or is_high_curvature
        )
        quality = (
            1.25 * normal_consistency
            + max(accessibility, 0.0)
            - hull_distance / max(self.concavity_tol, 1e-6)
            - self.curvature_penalty_weight * (curvature_mean + 0.5 * curvature_max)
        )

        return {
            "anchor_idx": anchor_idx,
            "point_indices": np.asarray(region_idx, dtype=int),
            "center": np.asarray(center, dtype=np.float64),
            "mean_inward_normal": np.asarray(mean_inward, dtype=np.float64),
            "mean_outward_normal": np.asarray(mean_outward, dtype=np.float64),
            "normal_consistency": normal_consistency,
            "accessibility": accessibility,
            "hull_distance": hull_distance,
            "anchor_curvature": anchor_curvature,
            "curvature_mean": curvature_mean,
            "curvature_max": curvature_max,
            "is_high_curvature": bool(is_high_curvature),
            "quality": float(quality),
            "is_concave": bool(is_concave),
        }

    def _deduplicate_regions(self, regions):
        if not regions:
            return []

        sorted_regions = sorted(regions, key=lambda item: item["quality"], reverse=True)
        selected = []
        for region in sorted_regions:
            keep = True
            region_points = region["point_indices"]
            for chosen in selected:
                center_distance = np.linalg.norm(region["center"] - chosen["center"])
                normal_dot = float(
                    np.clip(
                        np.dot(
                            region["mean_inward_normal"], chosen["mean_inward_normal"]
                        ),
                        -1.0,
                        1.0,
                    )
                )
                overlap = np.intersect1d(region_points, chosen["point_indices"]).size
                overlap_ratio = overlap / max(
                    1, min(region_points.size, chosen["point_indices"].size)
                )
                if (
                    center_distance < 0.35 * self.region_radius
                    and normal_dot > 0.9
                    and overlap_ratio > 0.6
                ):
                    keep = False
                    break
            if keep:
                selected.append(region)
        return selected

    def sample_candidate_region_on_surface(
        self,
        visible_face_idx=None,
        anchor_count=None,
        region_radius=None,
        max_points_per_region=None,
        object_pos=None,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        candidate_idx = self.get_contact_candidate_indices(
            visible_face_idx=visible_face_idx,
            object_pos=object_pos,
            object_rot=object_rot,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        if candidate_idx.size == 0:
            return []
        candidate_idx = self._filter_candidate_indices_by_curvature(candidate_idx)

        region_radius = (
            self.region_radius if region_radius is None else float(region_radius)
        )
        max_points_per_region = (
            self.region_max_points
            if max_points_per_region is None
            else int(max_points_per_region)
        )
        anchor_count = (
            self.region_anchor_count if anchor_count is None else int(anchor_count)
        )
        anchor_count = min(anchor_count, candidate_idx.size)

        candidate_points = self.sample_point[candidate_idx]
        candidate_tree = cKDTree(candidate_points)
        anchor_local_idx = _farthest_point_sampling(candidate_points, anchor_count)

        raw_regions = []
        valid_regions = []
        for local_anchor in anchor_local_idx:
            anchor_idx = int(candidate_idx[int(local_anchor)])
            region = self._build_region_from_anchor(
                anchor_idx,
                candidate_idx,
                candidate_tree,
                region_radius,
                max_points_per_region,
            )
            raw_regions.append(region)
            if not region["is_concave"]:
                valid_regions.append(region)

        regions = valid_regions if valid_regions else raw_regions
        regions = self._deduplicate_regions(regions)
        regions.sort(key=lambda item: item["quality"], reverse=True)
        return regions

    def _select_farthest_point_set(self, indices, num_points=None):
        indices = self._sanitize_point_indices(indices)
        if indices.size == 0:
            return None

        num_points = self.num_grasp_contacts if num_points is None else int(num_points)
        num_points = min(max(1, num_points), indices.size)
        local_idx = _farthest_point_sampling(self.sample_point[indices], num_points)
        return indices[np.asarray(local_idx, dtype=int)]

    def sample_points_from_region(self, region, num_points=None):
        if region is None:
            return np.zeros((0,), dtype=int)

        indices = np.asarray(region["point_indices"], dtype=int)
        if indices.size == 0:
            return indices

        num_points = (
            self.region_contact_samples if num_points is None else int(num_points)
        )
        num_points = min(num_points, indices.size)
        anchor_hits = np.where(indices == int(region["anchor_idx"]))[0]
        seed_index = int(anchor_hits[0]) if anchor_hits.size else None
        sampled_local = _farthest_point_sampling(
            self.sample_point[indices], num_points, seed_index=seed_index
        )
        return indices[sampled_local]

    def _scale_wrench(self, wrench):
        wrench = _as_numpy(wrench).reshape(6)
        return self.wrench_scale * wrench

    def _unscale_wrench(self, wrench):
        wrench = _as_numpy(wrench).reshape(6)
        return wrench / np.maximum(self.wrench_scale, _EPS)

    def _default_gravity_wrench_local(self):
        gravity_force_local = self.m * self.gravity_world
        return np.hstack([gravity_force_local, np.zeros(3, dtype=np.float64)])

    def build_gravity_wrench_local(self, target_pose=None, gravity_world=None):
        if target_pose is None:
            return self._default_gravity_wrench_local()

        pose_or_quat = _as_numpy(target_pose).reshape(-1)
        if pose_or_quat.size == 7:
            rotation_matrix = _quat_wxyz_to_matrix(pose_or_quat[3:7])
        elif pose_or_quat.size == 4:
            rotation_matrix = _quat_wxyz_to_matrix(pose_or_quat)
        elif pose_or_quat.size == 9:
            rotation_matrix = pose_or_quat.reshape(3, 3)
        else:
            raise ValueError(
                "target_pose must contain either 7 values [xyz, quat_wxyz], "
                "4 values quat_wxyz, or 9 values for a 3x3 rotation matrix."
            )

        gravity_world = (
            self.gravity_world
            if gravity_world is None
            else _as_numpy(gravity_world).reshape(3)
        )
        gravity_force_local = rotation_matrix.T @ (self.m * gravity_world)
        return np.hstack([gravity_force_local, np.zeros(3, dtype=np.float64)])

    def _resolve_external_wrench(
        self, external_wrench=None, target_pose=None, gravity_world=None
    ):
        total_wrench = np.zeros(6, dtype=np.float64)
        has_wrench = False

        if external_wrench is not None:
            external_wrench = _as_numpy(external_wrench)
            if external_wrench.size:
                total_wrench += np.sum(external_wrench.reshape(-1, 6), axis=0)
            has_wrench = True

        if target_pose is not None:
            total_wrench += self.build_gravity_wrench_local(
                target_pose=target_pose,
                gravity_world=gravity_world,
            )
            has_wrench = True

        if not has_wrench:
            return None
        return np.asarray(total_wrench, dtype=np.float64).reshape(6)

    @staticmethod
    def _coerce_target_pose(target_pose):
        if target_pose is None:
            return None
        target_pose = _as_numpy(target_pose).reshape(-1)
        if target_pose.size in {4, 7, 9}:
            return target_pose
        return None

    def _compute_static_equilibrium_penalty(
        self,
        static_result,
        cost_weight=None,
        residual_weight=None,
    ):
        cost_weight = (
            self.static_equilibrium_cost_weight
            if cost_weight is None
            else max(0.0, float(cost_weight))
        )
        residual_weight = (
            self.static_equilibrium_residual_weight
            if residual_weight is None
            else max(0.0, float(residual_weight))
        )

        if cost_weight <= 0.0 and residual_weight <= 0.0:
            return 0.0
        if static_result is None:
            return float("inf")

        static_cost = float(static_result.get("cost", float("inf")))
        scaled_residual_norm = float(
            static_result.get("scaled_residual_norm", float("inf"))
        )
        if not np.isfinite(static_cost) or not np.isfinite(scaled_residual_norm):
            return float("inf")
        return float(cost_weight * static_cost + residual_weight * scaled_residual_norm)

    def _stack_grasp_matrices(self, contact_indices, scaled=True):
        contact_indices = np.asarray(contact_indices, dtype=int).reshape(-1)
        return np.hstack(
            [self._grasp_matrix(int(idx), scaled=scaled) for idx in contact_indices]
        )

    def _grasp_matrix(self, idx, scaled=True):
        idx = int(idx)
        p = self.sample_point[idx]
        n = self.normal[idx]
        d = self.t1[idx]
        e = self.t2[idx]

        G = np.zeros((6, 3), dtype=np.float64)
        G[:3, 0] = n
        G[:3, 1] = d
        G[:3, 2] = e
        G[3:, 0] = np.cross(p, n)
        G[3:, 1] = np.cross(p, d)
        G[3:, 2] = np.cross(p, e)
        if scaled:
            return self.wrench_scale[:, None] * G
        return G

    def _primitive_wrenches_for_indices(self, indices):
        indices = np.asarray(indices, dtype=int).reshape(-1)
        primitive_wrenches = np.zeros(
            (indices.size, self.friction_primitives.shape[0], 6), dtype=np.float64
        )
        for row, idx in enumerate(indices):
            primitive_wrenches[row] = (
                self.friction_primitives @ self._grasp_matrix(int(idx)).T
            )
        return primitive_wrenches

    def _sample_boundary_wrenches(self, primitive_sets, sample_count, seed):
        primitive_sets = [np.asarray(ps, dtype=np.float64) for ps in primitive_sets]
        if not primitive_sets:
            return np.zeros((0, 6), dtype=np.float64)

        total_combo_count = 1
        counts = []
        for primitive_set in primitive_sets:
            count = int(primitive_set.shape[0])
            counts.append(count)
            total_combo_count *= max(count, 1)

        if total_combo_count <= sample_count:
            boundary_wrenches = []
            for combo in itertools.product(*[range(count) for count in counts]):
                wrench = np.zeros(6, dtype=np.float64)
                for primitive_set, combo_idx in zip(primitive_sets, combo):
                    wrench += primitive_set[int(combo_idx)]
                boundary_wrenches.append(wrench)
            return np.asarray(boundary_wrenches, dtype=np.float64)

        rng = np.random.default_rng(int(seed) % (2**32 - 1))
        boundary_wrenches = np.zeros((sample_count, 6), dtype=np.float64)
        for sample_idx in range(sample_count):
            wrench = np.zeros(6, dtype=np.float64)
            for primitive_set in primitive_sets:
                primitive_idx = int(rng.integers(primitive_set.shape[0]))
                wrench += primitive_set[primitive_idx]
            boundary_wrenches[sample_idx] = wrench
        return boundary_wrenches

    def _estimate_region_group_gwb(self, region_group):
        region_group = list(region_group)
        region_sample_indices = [
            self.sample_points_from_region(region, self.region_contact_samples)
            for region in region_group
        ]
        primitive_sets = [
            self._primitive_wrenches_for_indices(sample_idx).reshape(-1, 6)
            for sample_idx in region_sample_indices
        ]
        seed = sum(int(region["anchor_idx"]) for region in region_group) + 97 * len(
            region_group
        )
        boundary_wrenches = self._sample_boundary_wrenches(
            primitive_sets,
            self.gwb_wrench_count,
            seed=seed,
        )

        projections = self.disturbance_wrenches @ boundary_wrenches.T
        disturbance_scores = np.max(projections, axis=1)
        stability_score = float(np.min(disturbance_scores))

        return {
            "regions": region_group,
            "region_sample_indices": region_sample_indices,
            "boundary_wrenches": boundary_wrenches,
            "disturbance_scores": disturbance_scores,
            "stability_score": stability_score,
        }

    def _compute_region_group_coarse_score(self, region_group):
        centers = np.stack([region["center"] for region in region_group], axis=0)
        normals = np.stack(
            [region["mean_inward_normal"] for region in region_group], axis=0
        )
        pairwise_distances = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=-1
        )
        triu = np.triu_indices(len(region_group), k=1)
        if triu[0].size == 0:
            return None

        pair_distances = pairwise_distances[triu]
        if np.any(pair_distances < self.region_pair_min_distance):
            return None

        normal_dots = np.clip(normals @ normals.T, -1.0, 1.0)
        normal_diversity = float(np.mean(1.0 - normal_dots[triu]))
        mean_distance = float(np.mean(pair_distances))
        min_distance = float(np.min(pair_distances))
        mean_quality = float(np.mean([region["quality"] for region in region_group]))
        mean_accessibility = float(
            np.mean([max(region["accessibility"], 0.0) for region in region_group])
        )

        coarse_score = (
            min_distance
            * mean_distance
            * max(mean_quality, 0.1)
            * (0.5 + normal_diversity + 0.5 * mean_accessibility)
        )
        return float(coarse_score)

    def _fallback_region_group(self, visible_face_idx):
        visible_face_idx = self._filter_candidate_indices_by_curvature(visible_face_idx)
        if visible_face_idx.size == 0:
            return []

        seed_indices = self._select_farthest_point_set(
            visible_face_idx, self.num_grasp_contacts
        )
        if seed_indices is None or len(seed_indices) == 0:
            return []

        candidate_tree = cKDTree(self.sample_point[visible_face_idx])
        regions = [
            self._build_region_from_anchor(
                int(anchor_idx),
                visible_face_idx,
                candidate_tree,
                self.region_radius,
                self.region_max_points,
            )
            for anchor_idx in np.asarray(seed_indices, dtype=int)
        ]
        fallback = self._estimate_region_group_gwb(regions)
        fallback["rank"] = 1
        fallback["coarse_score"] = 0.0
        return [fallback]

    def get_best_regions(
        self,
        visible_face_idx=None,
        top_k=3,
        object_pos=None,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        search_t0 = time.perf_counter()
        step_t0 = time.perf_counter()
        candidate_idx = self.get_contact_candidate_indices(
            visible_face_idx=visible_face_idx,
            object_pos=object_pos,
            object_rot=object_rot,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        candidate_filter_elapsed = time.perf_counter() - step_t0
        if candidate_idx.size == 0:
            self.last_region_results = []
            self._record_search_timing(
                "get_best_regions",
                {
                    "candidate_filter_time": float(candidate_filter_elapsed),
                    "sample_candidate_regions_time": 0.0,
                    "coarse_group_time": 0.0,
                    "gwb_score_time": 0.0,
                    "candidate_count": 0,
                    "group_count": 0,
                    "wall_time": float(time.perf_counter() - search_t0),
                },
            )
            return []

        step_t0 = time.perf_counter()
        candidate_regions = self.sample_candidate_region_on_surface(
            visible_face_idx=candidate_idx
        )
        sample_candidate_regions_elapsed = time.perf_counter() - step_t0
        if len(candidate_regions) < self.num_grasp_contacts:
            self.last_region_results = self._fallback_region_group(candidate_idx)
            self._record_search_timing(
                "get_best_regions",
                {
                    "candidate_filter_time": float(candidate_filter_elapsed),
                    "sample_candidate_regions_time": float(
                        sample_candidate_regions_elapsed
                    ),
                    "coarse_group_time": 0.0,
                    "gwb_score_time": 0.0,
                    "candidate_count": int(candidate_idx.size),
                    "group_count": len(self.last_region_results[: int(top_k)]),
                    "wall_time": float(time.perf_counter() - search_t0),
                },
            )
            return self.last_region_results[: int(top_k)]

        candidate_region_limit = min(
            len(candidate_regions),
            max(
                self.num_grasp_contacts,
                8,
                2 * self.num_grasp_contacts * self.top_region_pairs,
            ),
        )
        candidate_regions = candidate_regions[:candidate_region_limit]

        step_t0 = time.perf_counter()
        coarse_groups = []
        for combo in itertools.combinations(
            range(len(candidate_regions)), self.num_grasp_contacts
        ):
            region_group = [candidate_regions[idx] for idx in combo]
            coarse_score = self._compute_region_group_coarse_score(region_group)
            if coarse_score is None:
                continue
            coarse_groups.append((coarse_score, combo))
        coarse_group_elapsed = time.perf_counter() - step_t0

        if not coarse_groups:
            self.last_region_results = self._fallback_region_group(candidate_idx)
            self._record_search_timing(
                "get_best_regions",
                {
                    "candidate_filter_time": float(candidate_filter_elapsed),
                    "sample_candidate_regions_time": float(
                        sample_candidate_regions_elapsed
                    ),
                    "coarse_group_time": float(coarse_group_elapsed),
                    "gwb_score_time": 0.0,
                    "candidate_count": int(candidate_idx.size),
                    "group_count": len(self.last_region_results[: int(top_k)]),
                    "wall_time": float(time.perf_counter() - search_t0),
                },
            )
            return self.last_region_results[: int(top_k)]

        coarse_groups.sort(key=lambda item: item[0], reverse=True)
        coarse_groups = coarse_groups[: self.preselect_region_pairs]

        step_t0 = time.perf_counter()
        scored_groups = []
        for coarse_score, combo in coarse_groups:
            region_group = [candidate_regions[idx] for idx in combo]
            group_info = self._estimate_region_group_gwb(region_group)
            group_info["coarse_score"] = float(coarse_score)
            scored_groups.append(group_info)
        gwb_score_elapsed = time.perf_counter() - step_t0

        scored_groups.sort(
            key=lambda item: (item["stability_score"], item["coarse_score"]),
            reverse=True,
        )

        best_groups = scored_groups[: int(top_k)]
        for rank, group in enumerate(best_groups, start=1):
            group["rank"] = rank
            for region_idx, region in enumerate(group["regions"]):
                group[f"region{region_idx + 1}"] = region
                group[f"region{region_idx + 1}_sample_idx"] = np.asarray(
                    group["region_sample_indices"][region_idx],
                    dtype=int,
                )

        self.last_region_results = best_groups
        self._record_search_timing(
            "get_best_regions",
            {
                "candidate_filter_time": float(candidate_filter_elapsed),
                "sample_candidate_regions_time": float(
                    sample_candidate_regions_elapsed
                ),
                "coarse_group_time": float(coarse_group_elapsed),
                "gwb_score_time": float(gwb_score_elapsed),
                "candidate_count": int(candidate_idx.size),
                "group_count": len(best_groups),
                "wall_time": float(time.perf_counter() - search_t0),
            },
        )
        return best_groups

    def compute_two_point_force_closure_margin(self, idx1, idx2):
        idx1 = int(idx1)
        idx2 = int(idx2)
        if idx1 == idx2:
            return -np.inf

        p1 = self.sample_point[idx1]
        p2 = self.sample_point[idx2]
        delta = p2 - p1
        distance = float(np.linalg.norm(delta))
        if distance < max(self.min_pair_distance, 1e-6):
            return -np.inf

        line_12 = delta / distance
        friction_angle = np.arctan(self.mu_arm_obj)
        cos_phi = float(np.cos(friction_angle))

        margin_1 = float(np.dot(self.normal[idx1], line_12) - cos_phi)
        margin_2 = float(np.dot(self.normal[idx2], -line_12) - cos_phi)
        return min(margin_1, margin_2)

    def _compute_contact_set_metrics(
        self,
        contact_indices,
        object_rot=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        contact_indices = np.asarray(contact_indices, dtype=int).reshape(-1)
        points = self.sample_point[contact_indices]
        normals = self.normal[contact_indices]
        pairwise_distances = np.linalg.norm(
            points[:, None, :] - points[None, :, :], axis=-1
        )
        triu = np.triu_indices(contact_indices.size, k=1)

        overlap_penalty = 0.0
        if triu[0].size:
            too_close = pairwise_distances[triu] < self.min_pair_distance
            if np.any(too_close):
                overlap_penalty = self.overlap_penalty_weight * float(
                    np.sum(
                        (self.min_pair_distance - pairwise_distances[triu][too_close])
                        ** 2
                    )
                )

        if contact_indices.size == 2:
            geometry_margin = self.compute_two_point_force_closure_margin(
                contact_indices[0], contact_indices[1]
            )
            geometry_penalty = 0.0
            if not np.isfinite(geometry_margin) or geometry_margin < 0.0:
                geometry_penalty = self.antipodal_penalty_weight * (
                    max(-geometry_margin, 1e-4) ** 2
                    if np.isfinite(geometry_margin)
                    else 1.0
                )
            normal_diversity = float(
                1.0 - np.clip(np.dot(normals[0], normals[1]), -1.0, 1.0)
            )

            pair_direction = _normalize(points[1] - points[0])
            centroid_line_offset = float(
                np.linalg.norm(np.cross(pair_direction, self.mesh_centroid - points[0]))
                / max(self.mesh_diag, 1e-6)
            )
            geometry_penalty += self.centerline_penalty_weight * (
                centroid_line_offset**2
            )

            support_axis_alignment = 0.0
            support_surface = self._resolve_support_surface(
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
            if support_surface is not None and object_rot is not None:
                pair_direction_world = _normalize(
                    _as_numpy(object_rot).reshape(3, 3) @ pair_direction
                )
                support_axis_alignment = float(
                    abs(np.dot(pair_direction_world, support_surface["normal"]))
                )
                geometry_penalty += self.support_axis_penalty_weight * (
                    support_axis_alignment**2
                )
        else:
            pair_margins = [
                self.compute_two_point_force_closure_margin(
                    contact_indices[i], contact_indices[j]
                )
                for i, j in zip(*triu)
            ]
            geometry_margin = float(np.max(pair_margins)) if pair_margins else -np.inf
            normal_dots = np.clip(normals @ normals.T, -1.0, 1.0)
            normal_diversity = (
                float(np.mean(1.0 - normal_dots[triu])) if triu[0].size else 0.0
            )
            target_diversity = 0.8
            geometry_penalty = (
                self.antipodal_penalty_weight
                * max(0.0, target_diversity - normal_diversity) ** 2
            )
            centroid_line_offset = 0.0
            support_axis_alignment = 0.0

        min_distance = float(np.min(pairwise_distances[triu])) if triu[0].size else 0.0
        return {
            "geometry_margin": float(geometry_margin),
            "geometry_penalty": float(geometry_penalty),
            "overlap_penalty": float(overlap_penalty),
            "normal_diversity": float(normal_diversity),
            "min_contact_distance": float(min_distance),
            "centroid_line_offset": float(centroid_line_offset),
            "support_axis_alignment": float(support_axis_alignment),
        }

    def _local_force_to_object_force(self, idx, force_local):
        idx = int(idx)
        force_local = _as_numpy(force_local).reshape(3)
        return (
            self.normal[idx] * force_local[0]
            + self.t1[idx] * force_local[1]
            + self.t2[idx] * force_local[2]
        )

    def _forces_from_solution_matrix(self, force_matrix):
        force_matrix = _as_numpy(force_matrix).reshape(3 * self.num_grasp_contacts, -1)
        return np.stack(
            [
                force_matrix[3 * contact_idx : 3 * (contact_idx + 1)]
                for contact_idx in range(self.num_grasp_contacts)
            ],
            axis=0,
        )

    def _forces_from_solution_vector(self, force_vector):
        force_vector = _as_numpy(force_vector).reshape(3 * self.num_grasp_contacts)
        return np.stack(
            [
                force_vector[3 * contact_idx : 3 * (contact_idx + 1)]
                for contact_idx in range(self.num_grasp_contacts)
            ],
            axis=0,
        )

    def _local_force_batch_to_object(self, contact_indices, contact_forces_local):
        contact_indices = np.asarray(contact_indices, dtype=int).reshape(-1)
        contact_forces_local = _as_numpy(contact_forces_local)
        return np.stack(
            [
                self._local_force_to_object_force(int(idx), contact_forces_local[i])
                for i, idx in enumerate(contact_indices)
            ],
            axis=0,
        )

    def _result_contact_wrench(self, result, r_obj_to_world=None):
        if result is None:
            wrench_info = {
                "contact_points_local": np.zeros((0, 3), dtype=np.float64),
                "contact_force_components_local": np.zeros((0, 3), dtype=np.float64),
                "contact_force_vectors_local": np.zeros((0, 3), dtype=np.float64),
                "contact_torques_local": np.zeros((0, 3), dtype=np.float64),
                "contact_force_local": np.zeros(3, dtype=np.float64),
                "contact_torque_local": np.zeros(3, dtype=np.float64),
            }
            if r_obj_to_world is not None:
                wrench_info["contact_force_world"] = np.zeros(3, dtype=np.float64)
                wrench_info["contact_torque_world"] = np.zeros(3, dtype=np.float64)
            return wrench_info

        contact_indices = np.asarray(
            result.get("contact_indices", []), dtype=int
        ).reshape(-1)
        contact_points_local = np.asarray(
            result.get("contact_points", np.zeros((0, 3), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1, 3)
        if contact_points_local.shape[0] != contact_indices.size:
            contact_points_local = (
                self.sample_point[contact_indices]
                if contact_indices.size
                else np.zeros((0, 3), dtype=np.float64)
            )

        contact_force_components_local = np.asarray(
            result.get(
                "chosen_contact_force_local",
                result.get(
                    "witness_contact_forces_local", np.zeros((0, 3), dtype=np.float64)
                ),
            ),
            dtype=np.float64,
        ).reshape(-1, 3)

        contact_force_vectors_local = np.asarray(
            result.get("force_vectors_local", np.zeros((0, 3), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1, 3)
        if (
            contact_force_vectors_local.shape[0] != contact_indices.size
            and contact_indices.size
            and contact_force_components_local.shape[0] == contact_indices.size
        ):
            contact_force_vectors_local = self._local_force_batch_to_object(
                contact_indices,
                contact_force_components_local,
            )
        elif contact_force_vectors_local.shape[0] != contact_indices.size:
            contact_force_vectors_local = np.zeros(
                (contact_indices.size, 3), dtype=np.float64
            )

        if contact_points_local.shape[0]:
            contact_torques_local = np.cross(
                contact_points_local, contact_force_vectors_local
            )
            total_force_local = np.sum(contact_force_vectors_local, axis=0)
            total_torque_local = np.sum(contact_torques_local, axis=0)
        else:
            contact_torques_local = np.zeros((0, 3), dtype=np.float64)
            total_force_local = np.zeros(3, dtype=np.float64)
            total_torque_local = np.zeros(3, dtype=np.float64)

        wrench_info = {
            "contact_points_local": contact_points_local,
            "contact_force_components_local": contact_force_components_local,
            "contact_force_vectors_local": contact_force_vectors_local,
            "contact_torques_local": contact_torques_local,
            "contact_force_local": np.asarray(
                total_force_local, dtype=np.float64
            ).reshape(3),
            "contact_torque_local": np.asarray(
                total_torque_local, dtype=np.float64
            ).reshape(3),
        }

        if r_obj_to_world is not None:
            r_obj_to_world = _as_numpy(r_obj_to_world).reshape(3, 3)
            wrench_info["contact_force_world"] = (
                r_obj_to_world @ wrench_info["contact_force_local"]
            )
            wrench_info["contact_torque_world"] = (
                r_obj_to_world @ wrench_info["contact_torque_local"]
            )

        return wrench_info

    def _solve_force_closure_qp_daqp(self, G):
        """Solve the force-closure SOCP as a pure QP via DAQP.

        The problem is separable across the ``n_dist`` disturbance columns, so
        we solve ``n_dist`` independent small QPs (one per disturbance) that
        share the same Hessian ``H`` and constraint matrix ``A``.  The
        Lorentz-cone friction constraint is replaced by its polyhedral
        approximation (inward face-normals of ``friction_primitives``), which
        makes the problem a true QP suitable for DAQP.

        Objective for column k::

            min_{f_k}  ||beta * d_k - G @ f_k||**2 + reg * ||f_k||**2

        i.e.  ``0.5 f_k' H f_k + g_k' f_k``  with
        ``H = 2 (G'G + reg I)``,  ``g_k = -2 beta G' d_k``.
        """
        G = np.asarray(G, dtype=np.float64).reshape(6, -1)
        nc = self.num_grasp_contacts
        n = 3 * nc
        n_dist = int(self.disturbance_wrenches.shape[0])
        reg = float(self.force_reg_weight)
        beta = float(self.beta)

        H = 2.0 * (G.T @ G + reg * np.eye(n, dtype=np.float64))
        A, blower, bupper = self._get_qp_cone_matrix(fn_lb=0.0, fn_ub=1.0)

        f_opt = np.zeros((n, n_dist), dtype=np.float64)
        cost_terms = np.zeros(n_dist, dtype=np.float64)
        total_cost = 0.0
        any_fail = False

        for k in range(n_dist):
            d_k = np.asarray(self.disturbance_wrenches[k], dtype=np.float64).reshape(6)
            g_k = -2.0 * beta * (G.T @ d_k)
            x_k, status = self._solve_qp_daqp(H, g_k, A, bupper, blower)
            if status != 1:
                any_fail = True
                cost_terms[k] = float("inf")
                continue
            f_opt[:, k] = x_k
            residual = beta * d_k - G @ x_k
            cost_terms[k] = float(residual @ residual)
            total_cost += cost_terms[k]

        if any_fail:
            total_cost = float("inf")
        else:
            total_cost += reg * float(f_opt.ravel() @ f_opt.ravel())

        return {
            "f_opt": f_opt,
            "cost": float(total_cost),
            "cost_terms": cost_terms,
            "wrench_response": G @ f_opt,
            "solver_status": "daqp",
            "solver_backend": "daqp_qp",
        }

    def _solve_static_equilibrium_qp_daqp(self, G, wrench_ext):
        """Solve the static-equilibrium SOCP as a pure QP via DAQP.

        Objective::

            min_{f}  ||wrench_ext + G @ f||**2 + reg * ||f||**2

        i.e. ``0.5 f' H f + g' f`` with
        ``H = 2 (G'G + reg I)``,  ``g = 2 G' wrench_ext``.
        """
        G = np.asarray(G, dtype=np.float64).reshape(6, -1)
        n = 3 * self.num_grasp_contacts
        reg = float(self.force_reg_weight)

        H = 2.0 * (G.T @ G + reg * np.eye(n, dtype=np.float64))
        A, blower, bupper = self._get_qp_cone_matrix(fn_lb=0.001, fn_ub=2.0)

        w = np.asarray(wrench_ext, dtype=np.float64).reshape(6)
        g = 2.0 * (G.T @ w)
        x, status = self._solve_qp_daqp(H, g, A, bupper, blower)

        if status != 1:
            return {
                "f_opt": np.zeros(n, dtype=np.float64),
                "residual": np.full(6, np.inf, dtype=np.float64),
                "cost": float("inf"),
                "solver_status": "daqp",
                "solver_backend": "daqp_qp",
            }

        residual = w + G @ x
        cost = float(residual @ residual + reg * (x @ x))
        return {
            "f_opt": x,
            "residual": residual,
            "cost": cost,
            "solver_status": "daqp",
            "solver_backend": "daqp_qp",
        }

    def _solve_force_closure(self, G):
        if self.force_closure_solver_backend == "acados":
            return self._solve_force_closure_with_acados(G)
        if self.force_closure_solver_backend == "daqp_qp":
            return self._solve_force_closure_qp_daqp(G)

        try:
            sol = self.force_closure_fn(G=G)
        except RuntimeError:
            return None
        return {
            "f_opt": sol["f_opt"],
            "cost": sol["cost"],
            "cost_terms": sol["cost_terms"],
            "wrench_response": sol["wrench_response"],
            "solver_status": None,
            "solver_backend": str(self.force_closure_solver_backend or "casadi"),
        }

    def _solve_static_equilibrium(self, G, wrench_ext):
        if self.static_equilibrium_solver_backend == "acados":
            return self._solve_static_equilibrium_with_acados(G, wrench_ext)
        if self.static_equilibrium_solver_backend == "daqp_qp":
            return self._solve_static_equilibrium_qp_daqp(G, wrench_ext)

        try:
            sol = self.static_equilibrium_fn(G=G, wrench_ext=wrench_ext)
        except RuntimeError:
            return None
        return {
            "f_opt": sol["f_opt"],
            "residual": sol["residual"],
            "cost": sol["cost"],
            "solver_status": None,
            "solver_backend": str(self.static_equilibrium_solver_backend or "casadi"),
        }

    def solve_static_equilibrium(self, result=None, external_wrench=None):
        if result is None:
            result = self.last_grasp_result
        if result is None:
            raise ValueError(
                "No grasp result available. Call get_best_grasp() first or pass a result dict."
            )

        if isinstance(result, dict):
            contact_indices = np.asarray(result["contact_indices"], dtype=int).reshape(
                -1
            )
        else:
            contact_indices = np.asarray(result, dtype=int).reshape(-1)

        if external_wrench is None:
            external_wrench = self._default_gravity_wrench_local()
        external_wrench = _as_numpy(external_wrench).reshape(6)
        scaled_wrench_ext = self._scale_wrench(external_wrench)

        start_time = time.time()
        sol = self._solve_static_equilibrium(
            self._stack_grasp_matrices(contact_indices, scaled=True),
            scaled_wrench_ext,
        )
        solve_time = time.time() - start_time

        if sol is None:
            contact_forces_local = np.zeros((contact_indices.size, 3), dtype=np.float64)
            scaled_residual = np.full(6, np.inf, dtype=np.float64)
            cost = float("inf")
            valid = False
        else:
            contact_forces_local = self._forces_from_solution_vector(sol["f_opt"])
            scaled_residual = _as_numpy(sol["residual"]).reshape(6)
            cost = float(_as_numpy(sol["cost"]).reshape(-1)[0])
            valid = np.isfinite(cost)
        solver_status = None if sol is None else sol.get("solver_status", None)
        solver_backend = (
            self.static_equilibrium_solver_backend
            if sol is None
            else sol.get(
                "solver_backend",
                self.static_equilibrium_solver_backend,
            )
        )

        residual_unscaled = self._unscale_wrench(scaled_residual)
        force_vectors = self._local_force_batch_to_object(
            contact_indices, contact_forces_local
        )
        contact_torques_local = np.cross(
            self.sample_point[contact_indices], force_vectors
        )
        total_force_local = (
            np.sum(force_vectors, axis=0)
            if force_vectors.size
            else np.zeros(3, dtype=np.float64)
        )
        total_torque_local = (
            np.sum(contact_torques_local, axis=0)
            if contact_torques_local.size
            else np.zeros(3, dtype=np.float64)
        )

        static_result = {
            "valid": bool(valid),
            "contact_indices": np.asarray(contact_indices, dtype=int),
            "contact_points": self.sample_point[contact_indices],
            "contact_normals": self.normal[contact_indices],
            "external_wrench": external_wrench,
            "scaled_external_wrench": scaled_wrench_ext,
            "contact_forces_local": contact_forces_local,
            "force_vectors_local": force_vectors,
            "contact_torques_local": contact_torques_local,
            "contact_force_local": np.asarray(
                total_force_local, dtype=np.float64
            ).reshape(3),
            "contact_torque_local": np.asarray(
                total_torque_local, dtype=np.float64
            ).reshape(3),
            "residual_wrench": residual_unscaled,
            "scaled_residual_wrench": scaled_residual,
            "residual_norm": float(np.linalg.norm(residual_unscaled)),
            "scaled_residual_norm": float(np.linalg.norm(scaled_residual)),
            "cost": cost,
            "solve_time": float(solve_time),
            "solver_status": solver_status,
            "solver_backend": solver_backend,
            "static_nlp_solver": self.static_nlp_solver,
        }

        self.last_static_equilibrium_result = static_result
        if isinstance(result, dict):
            result["static_equilibrium"] = static_result
        return static_result

    def _get_visualization_result(
        self, result=None, use_static_equilibrium=False, external_wrench=None
    ):
        if result is None:
            result = self.last_grasp_result
        if result is None:
            raise ValueError("No grasp result available for visualization.")

        if not use_static_equilibrium:
            return result

        static_result = None
        if isinstance(result, dict):
            static_result = result.get("static_equilibrium", None)
            if static_result is not None and external_wrench is not None:
                cached_wrench = _as_numpy(
                    static_result.get("external_wrench", np.zeros(6))
                ).reshape(6)
                requested_wrench = _as_numpy(external_wrench).reshape(6)
                if not np.allclose(cached_wrench, requested_wrench):
                    static_result = None

        if static_result is None:
            static_result = self.solve_static_equilibrium(
                result=result, external_wrench=external_wrench
            )

        vis_result = dict(result) if isinstance(result, dict) else {}
        vis_result.update(static_result)
        return vis_result

    def _evaluate_force_closure_candidate(
        self,
        contact_indices,
        region_group=None,
        object_rot=None,
        external_wrench=None,
        target_pose=None,
        gravity_world=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        contact_indices = np.asarray(contact_indices, dtype=int).reshape(-1)
        metrics = self._compute_contact_set_metrics(
            contact_indices,
            object_rot=object_rot,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )

        start_time = time.time()
        sol = self._solve_force_closure(
            self._stack_grasp_matrices(contact_indices, scaled=True)
        )
        solve_time = time.time() - start_time

        if sol is None:
            force_closure_cost = float("inf")
            cost_terms = np.full(
                (self.disturbance_wrenches.shape[0],), np.inf, dtype=np.float64
            )
            local_contact_forces = np.zeros(
                (contact_indices.size, 3, self.disturbance_wrenches.shape[0]),
                dtype=np.float64,
            )
            wrench_response = np.zeros(
                (6, self.disturbance_wrenches.shape[0]), dtype=np.float64
            )
            valid = False
        else:
            force_closure_cost = float(_as_numpy(sol["cost"]).reshape(-1)[0])
            cost_terms = _as_numpy(sol["cost_terms"]).reshape(-1)
            local_contact_forces = self._forces_from_solution_matrix(sol["f_opt"])
            wrench_response = _as_numpy(sol["wrench_response"]).reshape(6, -1)
            valid = np.isfinite(force_closure_cost)
        solver_status = None if sol is None else sol.get("solver_status", None)
        solver_backend = (
            self.force_closure_solver_backend
            if sol is None
            else sol.get(
                "solver_backend",
                self.force_closure_solver_backend,
            )
        )

        worst_disturbance_idx = int(np.argmax(cost_terms)) if cost_terms.size else 0
        chosen_contact_force_local = (
            local_contact_forces[:, :, worst_disturbance_idx]
            if local_contact_forces.size
            else np.zeros((contact_indices.size, 3), dtype=np.float64)
        )
        force_vectors = self._local_force_batch_to_object(
            contact_indices, chosen_contact_force_local
        )
        resolved_external_wrench = self._resolve_external_wrench(
            external_wrench=external_wrench,
            target_pose=target_pose,
            gravity_world=gravity_world,
        )
        static_result = None
        static_equilibrium_penalty = 0.0
        if resolved_external_wrench is not None:
            static_result = self.solve_static_equilibrium(
                contact_indices,
                external_wrench=resolved_external_wrench,
            )
            static_equilibrium_penalty = self._compute_static_equilibrium_penalty(
                static_result
            )

        base_total_cost = (
            force_closure_cost
            + metrics["geometry_penalty"]
            + metrics["overlap_penalty"]
        )
        total_cost = base_total_cost + static_equilibrium_penalty
        result = {
            "valid": bool(valid and np.isfinite(total_cost)),
            "contact_indices": np.asarray(contact_indices, dtype=int),
            "contact_points": self.sample_point[contact_indices],
            "contact_normals": self.normal[contact_indices],
            "contact_tangent1": self.t1[contact_indices],
            "contact_tangent2": self.t2[contact_indices],
            "min_contact_distance": metrics["min_contact_distance"],
            "pair_distance": metrics["min_contact_distance"],
            "antipodal_margin": float(metrics["geometry_margin"]),
            "antipodal_penalty": float(metrics["geometry_penalty"]),
            "normal_diversity": float(metrics["normal_diversity"]),
            "overlap_penalty": float(metrics["overlap_penalty"]),
            "centroid_line_offset": float(metrics["centroid_line_offset"]),
            "support_axis_alignment": float(metrics["support_axis_alignment"]),
            "force_closure_cost": float(force_closure_cost),
            "base_total_cost": float(base_total_cost),
            "total_cost": float(total_cost),
            "solve_time": float(solve_time),
            "solver_status": solver_status,
            "solver_backend": solver_backend,
            "nlp_solver": self.nlp_solver,
            "local_contact_forces": local_contact_forces,
            "cost_terms": cost_terms,
            "wrench_response": wrench_response,
            "worst_disturbance_idx": worst_disturbance_idx,
            "worst_disturbance_label": self.disturbance_labels[worst_disturbance_idx],
            "worst_disturbance_wrench": self.disturbance_wrenches[
                worst_disturbance_idx
            ],
            "witness_contact_forces_local": chosen_contact_force_local,
            "witness_force_vectors_local": force_vectors,
            "force_vectors_local": force_vectors,
            "chosen_contact_force_local": chosen_contact_force_local,
        }
        if resolved_external_wrench is not None:
            result["external_wrench"] = np.asarray(
                resolved_external_wrench, dtype=np.float64
            ).reshape(6)
            result["static_equilibrium_penalty"] = float(static_equilibrium_penalty)
            result["static_equilibrium"] = static_result
            result["static_equilibrium_cost"] = float(
                static_result.get("cost", float("inf"))
            )
            result["static_equilibrium_residual_norm"] = float(
                static_result.get("residual_norm", float("inf"))
            )
            result["static_equilibrium_scaled_residual_norm"] = float(
                static_result.get("scaled_residual_norm", float("inf"))
            )

        if region_group is not None:
            result["region_rank"] = int(region_group.get("rank", 0))
            result["region_score"] = float(region_group.get("stability_score", 0.0))
            result["regions"] = region_group["regions"]
            result["region_sample_indices"] = [
                np.asarray(sample_idx, dtype=int)
                for sample_idx in region_group["region_sample_indices"]
            ]
            for region_idx, region in enumerate(region_group["regions"]):
                result[f"region{region_idx + 1}"] = region
                result[f"region{region_idx + 1}_sample_idx"] = np.asarray(
                    region_group["region_sample_indices"][region_idx],
                    dtype=int,
                )

        return result

    def _evaluate_force_closure_pair(
        self,
        idx1,
        idx2,
        region_pair=None,
        object_rot=None,
        external_wrench=None,
        target_pose=None,
        gravity_world=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        return self._evaluate_force_closure_candidate(
            [idx1, idx2],
            region_group=region_pair,
            object_rot=object_rot,
            external_wrench=external_wrench,
            target_pose=target_pose,
            gravity_world=gravity_world,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )

    def _prepare_contact_sample_groups(self, region_sample_groups):
        sample_groups = [
            np.asarray(group, dtype=int).reshape(-1) for group in region_sample_groups
        ]
        if not sample_groups:
            return sample_groups

        per_group_cap = max(
            1, int(round(self.max_point_combination_eval ** (1.0 / len(sample_groups))))
        )
        sample_groups = [
            group[: min(group.size, per_group_cap)] for group in sample_groups
        ]

        while (
            np.prod([max(group.size, 1) for group in sample_groups], dtype=np.int64)
            > self.max_point_combination_eval
        ):
            largest_group_idx = int(np.argmax([group.size for group in sample_groups]))
            if sample_groups[largest_group_idx].size <= 1:
                break
            sample_groups[largest_group_idx] = sample_groups[largest_group_idx][:-1]
        return sample_groups

    def get_best_grasp(
        self,
        visible_face_idx=None,
        object_pos=None,
        object_rot=None,
        external_wrench=None,
        target_pose=None,
        gravity_world=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
        candidate_cache=None,
    ):
        search_t0 = time.perf_counter()
        candidate_cache_used = candidate_cache is not None
        get_best_regions_elapsed = 0.0
        prepare_candidate_entries_elapsed = 0.0

        if candidate_cache_used:
            region_groups = self._normalize_region_groups(
                candidate_cache.get("region_groups", [])
            )
            candidate_entries = list(candidate_cache.get("candidate_entries", []))
            candidate_point_groups = list(
                candidate_cache.get("candidate_point_groups", [])
            )
        else:
            region_groups = self.get_best_regions(
                visible_face_idx=visible_face_idx,
                top_k=self.top_region_pairs,
                object_pos=object_pos,
                object_rot=object_rot,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
            get_best_regions_elapsed = float(
                self.last_region_timing.get("wall_time", 0.0)
            )
            if not region_groups:
                self.last_grasp_result = None
                self._record_search_timing(
                    "get_best_grasp",
                    {
                        "mode": "online_search",
                        "candidate_cache_used": False,
                        "get_best_regions_time": float(get_best_regions_elapsed),
                        "prepare_candidate_entries_time": 0.0,
                        "evaluate_candidates_time": 0.0,
                        "force_closure_solve_time": 0.0,
                        "candidate_entry_count": 0,
                        "region_group_count": 0,
                        "wall_time": float(time.perf_counter() - search_t0),
                    },
                )
                return None

            prepare_t0 = time.perf_counter()
            (
                region_groups,
                candidate_entries,
                candidate_point_groups,
            ) = self._build_candidate_entries_from_region_groups(region_groups)
            prepare_candidate_entries_elapsed = time.perf_counter() - prepare_t0

        self.last_region_results = region_groups
        self.last_candidate_point_groups = candidate_point_groups

        if not candidate_entries:
            self.last_grasp_result = None
            self._record_search_timing(
                "get_best_grasp",
                {
                    "mode": "precomputed_cache"
                    if candidate_cache_used
                    else "online_search",
                    "candidate_cache_used": bool(candidate_cache_used),
                    "get_best_regions_time": float(get_best_regions_elapsed),
                    "prepare_candidate_entries_time": float(
                        prepare_candidate_entries_elapsed
                    ),
                    "evaluate_candidates_time": 0.0,
                    "force_closure_solve_time": 0.0,
                    "candidate_entry_count": 0,
                    "region_group_count": len(region_groups),
                    "wall_time": float(time.perf_counter() - search_t0),
                },
            )
            return None

        eval_t0 = time.perf_counter()
        evaluated_candidates = []
        force_closure_solve_time = 0.0
        for entry in candidate_entries:
            result = self._evaluate_force_closure_candidate(
                entry["contact_indices"],
                region_group=entry.get("region_group", None),
                object_rot=object_rot,
                external_wrench=external_wrench,
                target_pose=target_pose,
                gravity_world=gravity_world,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
            force_closure_solve_time += float(result.get("solve_time", 0.0))
            evaluated_candidates.append(result)
        evaluate_candidates_elapsed = time.perf_counter() - eval_t0

        if not evaluated_candidates:
            self.last_grasp_result = None
            self._record_search_timing(
                "get_best_grasp",
                {
                    "mode": "precomputed_cache"
                    if candidate_cache_used
                    else "online_search",
                    "candidate_cache_used": bool(candidate_cache_used),
                    "get_best_regions_time": float(get_best_regions_elapsed),
                    "prepare_candidate_entries_time": float(
                        prepare_candidate_entries_elapsed
                    ),
                    "evaluate_candidates_time": float(evaluate_candidates_elapsed),
                    "force_closure_solve_time": float(force_closure_solve_time),
                    "candidate_entry_count": len(candidate_entries),
                    "region_group_count": len(region_groups),
                    "wall_time": float(time.perf_counter() - search_t0),
                },
            )
            return None

        best_result = min(
            evaluated_candidates,
            key=lambda item: (
                item["total_cost"],
                -item.get("region_score", 0.0),
                -item["min_contact_distance"],
            ),
        )
        best_result["evaluated_candidate_count"] = len(evaluated_candidates)
        best_result["top_region_pairs"] = region_groups
        best_result["candidate_cache_used"] = bool(candidate_cache_used)
        best_result["wall_time"] = float(time.perf_counter() - search_t0)
        best_result["search_timing"] = self._record_search_timing(
            "get_best_grasp",
            {
                "mode": "precomputed_cache"
                if candidate_cache_used
                else "online_search",
                "candidate_cache_used": bool(candidate_cache_used),
                "get_best_regions_time": float(get_best_regions_elapsed),
                "prepare_candidate_entries_time": float(
                    prepare_candidate_entries_elapsed
                ),
                "evaluate_candidates_time": float(evaluate_candidates_elapsed),
                "force_closure_solve_time": float(force_closure_solve_time),
                "candidate_entry_count": len(candidate_entries),
                "region_group_count": len(region_groups),
                "wall_time": float(best_result["wall_time"]),
            },
        )
        self.last_grasp_result = best_result
        return best_result

    def _normalize_region_groups(self, region_groups):
        if region_groups is None:
            return []
        if isinstance(region_groups, dict):
            return [region_groups]
        return [group for group in list(region_groups) if group is not None]

    def _prepare_fixed_region_group(self, region_group, candidate_idx):
        candidate_idx = self._sanitize_point_indices(candidate_idx)
        prepared_regions = []
        prepared_sample_groups = []
        raw_sample_groups = list(region_group.get("region_sample_indices", []))

        for region_idx, region in enumerate(region_group.get("regions", [])):
            region_point_idx = np.asarray(
                region.get("point_indices", []), dtype=int
            ).reshape(-1)
            filtered_point_idx = (
                region_point_idx[np.isin(region_point_idx, candidate_idx)]
                if candidate_idx.size
                else np.zeros((0,), dtype=int)
            )

            if filtered_point_idx.size == 0:
                fallback_group = (
                    np.asarray(raw_sample_groups[region_idx], dtype=int).reshape(-1)
                    if region_idx < len(raw_sample_groups)
                    else np.zeros((0,), dtype=int)
                )
                filtered_point_idx = (
                    fallback_group if fallback_group.size else region_point_idx
                )

            if filtered_point_idx.size == 0:
                return None

            prepared_region = dict(region)
            prepared_region["point_indices"] = np.asarray(filtered_point_idx, dtype=int)
            prepared_regions.append(prepared_region)

            sampled_idx = self.sample_points_from_region(
                prepared_region, self.region_contact_samples
            )
            if sampled_idx.size == 0:
                return None
            prepared_sample_groups.append(np.asarray(sampled_idx, dtype=int))

        if not prepared_regions:
            return None

        prepared_group = dict(region_group)
        prepared_group["regions"] = prepared_regions
        prepared_group["region_sample_indices"] = prepared_sample_groups
        for region_idx, region in enumerate(prepared_regions):
            prepared_group[f"region{region_idx + 1}"] = region
            prepared_group[f"region{region_idx + 1}_sample_idx"] = np.asarray(
                prepared_sample_groups[region_idx],
                dtype=int,
            )
        return prepared_group

    def get_best_grasp_from_regions(
        self,
        region_groups,
        visible_face_idx=None,
        object_pos=None,
        object_rot=None,
        external_wrench=None,
        target_pose=None,
        gravity_world=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        search_t0 = time.perf_counter()
        region_groups = self._normalize_region_groups(region_groups)
        if not region_groups:
            self.last_grasp_result = None
            self._record_search_timing(
                "get_best_grasp_from_regions",
                {
                    "mode": "fixed_region_groups",
                    "candidate_filter_time": 0.0,
                    "prepare_candidate_entries_time": 0.0,
                    "evaluate_candidates_time": 0.0,
                    "force_closure_solve_time": 0.0,
                    "candidate_entry_count": 0,
                    "region_group_count": 0,
                    "wall_time": float(time.perf_counter() - search_t0),
                },
            )
            return None

        candidate_filter_t0 = time.perf_counter()
        candidate_idx = self.get_contact_candidate_indices(
            visible_face_idx=visible_face_idx,
            object_pos=object_pos,
            object_rot=object_rot,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        candidate_filter_elapsed = time.perf_counter() - candidate_filter_t0
        if candidate_idx.size == 0:
            candidate_idx = self._sanitize_point_indices(visible_face_idx)
        if candidate_idx.size == 0:
            candidate_idx = self.point_idx.copy()

        self.last_available_idx = np.asarray(candidate_idx, dtype=int)

        prepare_t0 = time.perf_counter()
        prepared_region_groups = []
        candidate_entries = []
        for region_group in region_groups:
            prepared_group = self._prepare_fixed_region_group(
                region_group, candidate_idx
            )
            if prepared_group is None:
                continue

            prepared_region_groups.append(prepared_group)
            sample_groups = self._prepare_contact_sample_groups(
                prepared_group["region_sample_indices"]
            )
            if not sample_groups or any(group.size == 0 for group in sample_groups):
                continue

            for contact_indices in itertools.product(
                *[group.tolist() for group in sample_groups]
            ):
                candidate_entries.append(
                    {
                        "contact_indices": np.asarray(contact_indices, dtype=int),
                        "region_group": prepared_group,
                    }
                )
        prepare_candidate_entries_elapsed = time.perf_counter() - prepare_t0

        evaluated_candidates = []
        force_closure_solve_time = 0.0
        eval_t0 = time.perf_counter()
        for entry in candidate_entries:
            result = self._evaluate_force_closure_candidate(
                entry["contact_indices"],
                region_group=entry.get("region_group", None),
                object_rot=object_rot,
                external_wrench=external_wrench,
                target_pose=target_pose,
                gravity_world=gravity_world,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
            force_closure_solve_time += float(result.get("solve_time", 0.0))
            evaluated_candidates.append(result)
        evaluate_candidates_elapsed = time.perf_counter() - eval_t0

        self.last_region_results = (
            prepared_region_groups if prepared_region_groups else region_groups
        )
        self.last_candidate_point_groups = [
            [
                np.asarray(group, dtype=int)
                for group in prepared_group["region_sample_indices"]
            ]
            for prepared_group in prepared_region_groups
        ]

        if not evaluated_candidates:
            self.last_grasp_result = None
            self._record_search_timing(
                "get_best_grasp_from_regions",
                {
                    "mode": "fixed_region_groups",
                    "candidate_filter_time": float(candidate_filter_elapsed),
                    "prepare_candidate_entries_time": float(
                        prepare_candidate_entries_elapsed
                    ),
                    "evaluate_candidates_time": float(evaluate_candidates_elapsed),
                    "force_closure_solve_time": float(force_closure_solve_time),
                    "candidate_entry_count": len(candidate_entries),
                    "region_group_count": len(self.last_region_results),
                    "wall_time": float(time.perf_counter() - search_t0),
                },
            )
            return None

        best_result = min(
            evaluated_candidates,
            key=lambda item: (
                item["total_cost"],
                -item.get("region_score", 0.0),
                -item["min_contact_distance"],
            ),
        )
        best_result["evaluated_candidate_count"] = len(evaluated_candidates)
        best_result["top_region_pairs"] = (
            prepared_region_groups if prepared_region_groups else region_groups
        )
        best_result["fixed_region_groups"] = True
        best_result["wall_time"] = float(time.perf_counter() - search_t0)
        best_result["search_timing"] = self._record_search_timing(
            "get_best_grasp_from_regions",
            {
                "mode": "fixed_region_groups",
                "candidate_filter_time": float(candidate_filter_elapsed),
                "prepare_candidate_entries_time": float(
                    prepare_candidate_entries_elapsed
                ),
                "evaluate_candidates_time": float(evaluate_candidates_elapsed),
                "force_closure_solve_time": float(force_closure_solve_time),
                "candidate_entry_count": len(candidate_entries),
                "region_group_count": len(
                    prepared_region_groups if prepared_region_groups else region_groups
                ),
                "wall_time": float(best_result["wall_time"]),
            },
        )
        self.last_grasp_result = best_result
        return best_result

    def get_availble_point_idx(
        self,
        pos,
        R,
        target_pos,
        threshold=0.025,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
    ):
        pos = _as_numpy(pos).reshape(3)
        R = _as_numpy(R).reshape(3, 3)

        centers_world = (R @ self.sample_point.T).T + pos
        visible_mask = centers_world[:, 2] > float(threshold)
        visible_idx = np.where(visible_mask)[0]
        if visible_idx.size == 0:
            visible_idx = self.point_idx.copy()

        visible_idx = self.get_contact_candidate_indices(
            visible_face_idx=visible_idx,
            object_pos=pos,
            object_rot=R,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )
        if visible_idx.size == 0:
            visible_idx = self.get_contact_candidate_indices(
                visible_face_idx=self.point_idx,
                object_pos=pos,
                object_rot=R,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )

        self.last_available_idx = visible_idx
        best_regions = self.get_best_regions(
            visible_face_idx=visible_idx,
            top_k=self.top_region_pairs,
            object_pos=pos,
            object_rot=R,
            support_surface_point=support_surface_point,
            support_surface_normal=support_surface_normal,
            support_surface_clearance=support_surface_clearance,
            support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
        )

        grouped_samples = []
        flat_samples = []
        for region_group in best_regions:
            groups = [
                np.asarray(group, dtype=int)
                for group in region_group["region_sample_indices"]
            ]
            grouped_samples.append(groups)
            for group in groups:
                flat_samples.extend(group.tolist())

        self.last_candidate_point_groups = grouped_samples
        if not flat_samples:
            return visible_idx
        return np.unique(np.asarray(flat_samples, dtype=int))

    def choose_contact_points(
        self,
        x_d=None,
        current_x=None,
        tau_o=None,
        visible_face_idx=None,
        v_obj=None,
        object_pos=None,
        object_rot=None,
        target_pose=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
        fixed_region_groups=None,
        candidate_cache=None,
    ):
        target_pose = self._coerce_target_pose(
            target_pose if target_pose is not None else x_d
        )
        if candidate_cache is not None:
            result = self.get_best_grasp(
                visible_face_idx=visible_face_idx,
                object_pos=object_pos,
                object_rot=object_rot,
                external_wrench=tau_o,
                target_pose=target_pose,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
                candidate_cache=candidate_cache,
            )
        elif fixed_region_groups is None:
            result = self.get_best_grasp(
                visible_face_idx=visible_face_idx,
                object_pos=object_pos,
                object_rot=object_rot,
                external_wrench=tau_o,
                target_pose=target_pose,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
        else:
            result = self.get_best_grasp_from_regions(
                fixed_region_groups,
                visible_face_idx=visible_face_idx,
                object_pos=object_pos,
                object_rot=object_rot,
                external_wrench=tau_o,
                target_pose=target_pose,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
        if result is None:
            fallback_idx = int(self.point_idx[0])
            return (
                self.sample_point[fallback_idx],
                self.normal[fallback_idx],
                float("inf"),
                float("inf"),
                0.0,
            )

        return (
            result["contact_points"][0],
            result["contact_normals"][0],
            float(result["total_cost"]),
            float(result.get("region_score", 0.0)),
            float(result["antipodal_margin"]),
        )

    def choose_contact_set(
        self,
        x_d=None,
        current_x=None,
        tau_o=None,
        visible_face_idx=None,
        v_obj=None,
        object_pos=None,
        object_rot=None,
        target_pose=None,
        support_surface_point=None,
        support_surface_normal=None,
        support_surface_clearance=None,
        support_surface_normal_alignment_threshold=None,
        fixed_region_groups=None,
        candidate_cache=None,
    ):
        target_pose = self._coerce_target_pose(
            target_pose if target_pose is not None else x_d
        )
        del current_x
        del v_obj
        if candidate_cache is not None:
            result = self.get_best_grasp(
                visible_face_idx=visible_face_idx,
                object_pos=object_pos,
                object_rot=object_rot,
                external_wrench=tau_o,
                target_pose=target_pose,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
                candidate_cache=candidate_cache,
            )
        elif fixed_region_groups is None:
            result = self.get_best_grasp(
                visible_face_idx=visible_face_idx,
                object_pos=object_pos,
                object_rot=object_rot,
                external_wrench=tau_o,
                target_pose=target_pose,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
        else:
            result = self.get_best_grasp_from_regions(
                fixed_region_groups,
                visible_face_idx=visible_face_idx,
                object_pos=object_pos,
                object_rot=object_rot,
                external_wrench=tau_o,
                target_pose=target_pose,
                support_surface_point=support_surface_point,
                support_surface_normal=support_surface_normal,
                support_surface_clearance=support_surface_clearance,
                support_surface_normal_alignment_threshold=support_surface_normal_alignment_threshold,
            )
        if result is None:
            fallback_idx = int(self.point_idx[0])
            return (
                self.sample_point[[fallback_idx]],
                self.normal[[fallback_idx]],
                float("inf"),
                float("inf"),
                0.0,
            )

        return (
            result["contact_points"],
            result["contact_normals"],
            float(result["total_cost"]),
            float(result.get("region_score", 0.0)),
            float(result["antipodal_margin"]),
        )

    def optimize_dual_contact_input(
        self, x_d, current_x, tau_o, p_arm1, p_arm2, v_obj=None
    ):
        target_pose = self._coerce_target_pose(x_d) if tau_o is None else None
        idx1 = int(self.sample_kdtree.query(_as_numpy(p_arm1).reshape(3), k=1)[1])
        idx2 = int(self.sample_kdtree.query(_as_numpy(p_arm2).reshape(3), k=1)[1])
        result = self._evaluate_force_closure_candidate(
            [idx1, idx2],
            external_wrench=tau_o,
            target_pose=target_pose,
        )
        self.last_grasp_result = result
        return result

    def optimize_multi_contact_input(
        self,
        contact_points,
        x_d=None,
        current_x=None,
        tau_o=None,
        v_obj=None,
        r_obj_to_world=None,
    ):
        target_pose = self._coerce_target_pose(x_d) if tau_o is None else None
        del current_x
        del v_obj
        contact_points = _as_numpy(contact_points).reshape(-1, 3)
        contact_indices = np.array(
            [int(self.sample_kdtree.query(point, k=1)[1]) for point in contact_points],
            dtype=int,
        )
        result = self._evaluate_force_closure_candidate(
            contact_indices,
            external_wrench=tau_o,
            target_pose=target_pose,
        )
        result.update(
            self._result_contact_wrench(result, r_obj_to_world=r_obj_to_world)
        )
        self.last_grasp_result = result
        return result

    def optimize_control_input(
        self,
        x_d,
        current_x,
        tau_o,
        p_arm=None,
        v_obj=None,
        r_obj_to_world=None,
        return_world_torque=False,
    ):
        target_pose = self._coerce_target_pose(x_d) if tau_o is None else None
        del v_obj

        current_pose = _as_numpy(current_x).reshape(-1)
        if current_pose.size < 7:
            current_pose = np.pad(
                current_pose.astype(np.float64, copy=False), (0, 7 - current_pose.size)
            )
        else:
            current_pose = current_pose[:7].astype(np.float64, copy=False)

        if p_arm is None:
            best_grasp = self.get_best_grasp(
                self.last_available_idx,
                external_wrench=tau_o,
                target_pose=target_pose,
            )
            if best_grasp is None:
                point = self.sample_point[0]
                normal = self.normal[0]
                info = {
                    "solver_failed": True,
                    "selected_contacts": np.zeros((0,), dtype=int),
                }
                info.update(
                    self._result_contact_wrench(None, r_obj_to_world=r_obj_to_world)
                )
                if return_world_torque:
                    return (
                        point,
                        normal,
                        current_pose,
                        float("inf"),
                        info,
                        info.get("contact_torque_world", None),
                    )
                return point, normal, current_pose, float("inf"), info
            point = best_grasp["contact_points"][0]
            normal = best_grasp["contact_normals"][0]
            info = {
                "solver_failed": False,
                "selected_contacts": best_grasp["contact_indices"],
            }
            info.update(
                self._result_contact_wrench(best_grasp, r_obj_to_world=r_obj_to_world)
            )
            info["solver_status"] = best_grasp.get("solver_status", None)
            info["solver_backend"] = best_grasp.get(
                "solver_backend", self.force_closure_solver_backend
            )
            info["nlp_solver"] = best_grasp.get("nlp_solver", self.nlp_solver)
            if return_world_torque:
                return (
                    point,
                    normal,
                    current_pose,
                    float(best_grasp["total_cost"]),
                    info,
                    info.get("contact_torque_world", None),
                )
            return point, normal, current_pose, float(best_grasp["total_cost"]), info

        p_arm = _as_numpy(p_arm).reshape(-1, 3)
        contact_indices = np.array(
            [int(self.sample_kdtree.query(point, k=1)[1]) for point in p_arm],
            dtype=int,
        )

        if contact_indices.size >= 2:
            result = self._evaluate_force_closure_candidate(
                contact_indices,
                external_wrench=tau_o,
                target_pose=target_pose,
            )
            result.update(
                self._result_contact_wrench(result, r_obj_to_world=r_obj_to_world)
            )
            self.last_grasp_result = result
            point = np.asarray(result["contact_points"][0], dtype=np.float64).reshape(3)
            normal = np.asarray(result["contact_normals"][0], dtype=np.float64).reshape(
                3
            )
            info = {
                "solver_failed": not bool(result.get("valid", False)),
                "selected_contacts": np.asarray(
                    result.get("contact_indices", contact_indices), dtype=int
                ).reshape(-1),
                "query_contact_indices": contact_indices.copy(),
                "query_contact_points_local": p_arm.copy(),
                "solver_status": result.get("solver_status", None),
                "solver_backend": result.get(
                    "solver_backend", self.force_closure_solver_backend
                ),
                "nlp_solver": result.get("nlp_solver", self.nlp_solver),
            }
            info.update(
                self._result_contact_wrench(result, r_obj_to_world=r_obj_to_world)
            )
            objective_value = float(result.get("total_cost", float("inf")))
            if return_world_torque:
                return (
                    point,
                    normal,
                    current_pose,
                    objective_value,
                    info,
                    info.get("contact_torque_world", None),
                )
            return point, normal, current_pose, objective_value, info

        idx = int(contact_indices[0])
        point = self.sample_point[idx]
        normal = self.normal[idx]
        info = {
            "solver_failed": False,
            "nearest_surface_idx": idx,
            "selected_contacts": np.asarray([idx], dtype=int),
            "query_contact_indices": contact_indices.copy(),
            "query_contact_points_local": p_arm.copy(),
        }
        info.update(self._result_contact_wrench(None, r_obj_to_world=r_obj_to_world))
        info["contact_point_local"] = np.asarray(point, dtype=np.float64).reshape(3)
        if return_world_torque:
            return (
                point,
                normal,
                current_pose,
                0.0,
                info,
                info.get("contact_torque_world", None),
            )
        return point, normal, current_pose, 0.0, info

    @staticmethod
    def _make_colored_sphere(center, radius, color):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        sphere.translate(np.asarray(center, dtype=np.float64))
        return sphere

    @staticmethod
    def _make_arrow(start, vector, color, cylinder_radius, cone_radius):
        length = float(np.linalg.norm(vector))
        if length < 1e-8:
            return None

        cylinder_height = max(0.7 * length, 1e-4)
        cone_height = max(length - cylinder_height, 1e-4)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=float(cylinder_radius),
            cone_radius=float(cone_radius),
            cylinder_height=float(cylinder_height),
            cone_height=float(cone_height),
        )
        arrow.compute_vertex_normals()
        arrow.paint_uniform_color(color)
        arrow.rotate(_rotation_from_z(vector), center=np.zeros(3, dtype=np.float64))
        arrow.translate(np.asarray(start, dtype=np.float64))
        return arrow

    @staticmethod
    def _contact_palette():
        return [
            np.array([0.95, 0.45, 0.45]),
            np.array([0.25, 0.45, 0.95]),
            np.array([0.95, 0.75, 0.25]),
            np.array([0.20, 0.72, 0.78]),
            np.array([0.85, 0.35, 0.80]),
            np.array([0.55, 0.78, 0.30]),
        ]

    @staticmethod
    def _build_object_mesh_material():
        if o3d is None:
            return None
        if not hasattr(o3d.visualization, "rendering"):
            return None

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLitTransparency"
        material.base_color = [
            float(_O3D_OBJECT_BASE_COLOR[0]),
            float(_O3D_OBJECT_BASE_COLOR[1]),
            float(_O3D_OBJECT_BASE_COLOR[2]),
            float(_O3D_OBJECT_OPACITY),
        ]
        material.has_alpha = True
        return material

    def _build_draw_items(self, geometries):
        draw_items = []
        mesh_material = self._build_object_mesh_material()

        for geometry_idx, geometry in enumerate(geometries):
            draw_item = {
                "name": f"grasp_geometry_{geometry_idx:03d}",
                "geometry": geometry,
            }
            if geometry_idx == 0 and mesh_material is not None:
                draw_item["material"] = mesh_material
            draw_items.append(draw_item)

        return draw_items

    def build_grasp_visualization_geometries(
        self,
        result=None,
        point_radius=None,
        force_scale=None,
        include_normals=True,
        include_region_samples=True,
        use_static_equilibrium=False,
        external_wrench=None,
    ):
        if o3d is None:
            raise ImportError(
                "open3d is required for visualization but is not installed."
            )

        result = self._get_visualization_result(
            result=result,
            use_static_equilibrium=use_static_equilibrium,
            external_wrench=external_wrench,
        )

        mesh_o3d = self.pp.to_open3d_mesh()
        mesh_o3d.compute_vertex_normals()
        mesh_o3d.paint_uniform_color(_O3D_OBJECT_BASE_COLOR.tolist())

        bbox = mesh_o3d.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        diag = max(diag, 1e-3)
        if point_radius is None:
            point_radius = 0.015 * diag

        contact_points = np.asarray(result["contact_points"], dtype=np.float64)
        contact_normals = np.asarray(result["contact_normals"], dtype=np.float64)
        force_vectors = np.asarray(
            result.get("force_vectors_local", np.zeros_like(contact_points)),
            dtype=np.float64,
        )
        max_force = max(float(np.max(np.linalg.norm(force_vectors, axis=1))), 1e-6)
        if force_scale is None:
            force_scale = 0.18 * diag / max_force

        geometries = [mesh_o3d]
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.2 * diag, origin=[0.0, 0.0, 0.0]
            )
        )

        region_colors = self._contact_palette()
        region_sample_groups = result.get("region_sample_indices", None)
        if region_sample_groups is None:
            region_sample_groups = []
            region_idx = 1
            while f"region{region_idx}_sample_idx" in result:
                region_sample_groups.append(
                    np.asarray(result[f"region{region_idx}_sample_idx"], dtype=int)
                )
                region_idx += 1
        if include_region_samples:
            for i, region_sample_idx in enumerate(region_sample_groups):
                color = region_colors[i % len(region_colors)]
                region_points = self.sample_point[
                    np.asarray(region_sample_idx, dtype=int)
                ]
                if region_points.size == 0:
                    continue
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(region_points)
                pcd.paint_uniform_color((0.6 * color + 0.4).clip(0.0, 1.0))
                geometries.append(pcd)

        for i in range(contact_points.shape[0]):
            point = contact_points[i]
            normal = contact_normals[i]
            color = region_colors[i % len(region_colors)]
            force_vec = force_vectors[i] * force_scale

            geometries.append(self._make_colored_sphere(point, point_radius, color))

            arrow = self._make_arrow(
                point,
                force_vec,
                color,
                cylinder_radius=0.22 * point_radius,
                cone_radius=0.42 * point_radius,
            )
            if arrow is not None:
                geometries.append(arrow)

            if include_normals:
                normal_arrow = self._make_arrow(
                    point,
                    0.18 * diag * normal,
                    [0.15, 0.8, 0.25],
                    cylinder_radius=0.12 * point_radius,
                    cone_radius=0.26 * point_radius,
                )
                if normal_arrow is not None:
                    geometries.append(normal_arrow)

        return geometries

    def visual_grasp(
        self,
        result=None,
        point_radius=None,
        force_scale=None,
        include_normals=True,
        include_region_samples=True,
        use_static_equilibrium=False,
        external_wrench=None,
    ):
        geometries = self.build_grasp_visualization_geometries(
            result=result,
            point_radius=point_radius,
            force_scale=force_scale,
            include_normals=include_normals,
            include_region_samples=include_region_samples,
            use_static_equilibrium=use_static_equilibrium,
            external_wrench=external_wrench,
        )
        if hasattr(o3d.visualization, "draw") and hasattr(
            o3d.visualization, "rendering"
        ):
            o3d.visualization.draw(self._build_draw_items(geometries))
            return
        o3d.visualization.draw_geometries(geometries, mesh_show_back_face=True)

    def visualize_grasp_result(
        self,
        result=None,
        force_scale=None,
        normal_scale=None,
        point_radius=None,
        include_normals=True,
        include_target=True,
        use_static_equilibrium=False,
        external_wrench=None,
    ):
        del normal_scale
        del include_target
        self.visual_grasp(
            result=result,
            point_radius=point_radius,
            force_scale=force_scale,
            include_normals=include_normals,
            include_region_samples=True,
            use_static_equilibrium=use_static_equilibrium,
            external_wrench=external_wrench,
        )


def find_object_mesh(object_name, objects_dir=OBJECT_ASSET_DIR):
    objects_dir = Path(objects_dir)
    if not objects_dir.exists():
        raise FileNotFoundError(f"Objects directory does not exist: {objects_dir}")

    object_name = str(object_name).strip().lower()
    candidates = []
    for mesh_path in objects_dir.rglob("*"):
        if mesh_path.suffix.lower() not in {".stl", ".obj"}:
            continue
        stem = mesh_path.stem.lower()
        filename = mesh_path.name.lower()
        if stem == object_name or filename == object_name:
            score = 0
        elif object_name in stem or object_name in filename:
            score = 1
        else:
            continue
        format_bias = 0 if mesh_path.suffix.lower() == ".stl" else 1
        candidates.append((score, format_bias, len(mesh_path.name), mesh_path))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find mesh for object '{object_name}' under {objects_dir}."
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[0][3]


def build_default_target_pose(current_pose, lift_distance=0.2):
    current_pose = _as_numpy(current_pose).reshape(7).copy()
    target_pose = current_pose.copy()
    target_pose[2] += float(lift_distance)
    return target_pose


def run_grasp_prediction(args):
    mesh_path = find_object_mesh(args.object_name, args.objects_dir)
    optimizer = LambdaContactControlOptimizer(
        mesh_path=mesh_path,
        obj_mass=args.obj_mass,
        arm_friction=args.arm_friction,
        contact_stiffness=args.contact_stiffness,
        time_step=args.time_step,
        max_contacts=args.max_contacts,
        sample_num=args.sample_num,
        pos_coef=args.pos_coef,
        ori_coef=args.ori_coef,
        num_grasp_contacts=args.num_grasp_contacts,
        region_anchor_count=args.region_anchor_count,
        region_radius=args.region_radius,
        region_max_points=args.region_max_points,
        region_contact_samples=args.region_contact_samples,
        top_region_pairs=args.top_region_pairs,
        preselect_region_pairs=args.preselect_region_pairs,
        friction_cone_edges=args.friction_cone_edges,
        gwb_wrench_count=args.gwb_wrench_count,
        beta=args.beta,
        gamma=args.gamma,
        force_reg_weight=args.force_reg_weight,
        concavity_tol=args.concavity_tol,
        max_point_combination_eval=args.max_point_combination_eval,
        nlp_solver=args.solver,
        static_nlp_solver=(
            args.static_solver if args.static_solver is not None else args.solver
        ),
    )

    current_pose = np.asarray(args.current_pose, dtype=np.float64)
    target_pose = build_default_target_pose(current_pose, args.lift_distance)
    start_time = time.time()
    result = optimizer.get_best_grasp(np.arange(optimizer.sample_num, dtype=int))
    wall_time = time.time() - start_time
    if result is None:
        raise RuntimeError("Failed to produce a valid grasp candidate.")

    result["mesh_path"] = str(mesh_path)
    result["current_pose"] = current_pose
    result["target_pose"] = target_pose
    result["wall_time"] = wall_time
    return optimizer, result


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Region-based multi-contact grasp selection with GWB ranking and force-closure optimization."
    )
    parser.add_argument(
        "object_name",
        type=str,
        help="Object name used to search envs/assets/objects/*.stl or *.obj",
    )
    parser.add_argument("--objects-dir", type=str, default=str(OBJECT_ASSET_DIR))
    parser.add_argument(
        "--current-pose",
        nargs=7,
        type=float,
        default=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )
    parser.add_argument("--lift-distance", type=float, default=0.2)
    parser.add_argument("--obj-mass", type=float, default=0.01)
    parser.add_argument("--arm-friction", type=float, default=0.9)
    parser.add_argument("--contact-stiffness", type=float, default=12.5)
    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--max-contacts", type=int, default=10)
    parser.add_argument(
        "--sample-num",
        type=int,
        default=70,
        help="Legacy parameter kept for compatibility",
    )
    parser.add_argument(
        "--pos-coef",
        type=float,
        default=1.0,
        help="Legacy parameter kept for compatibility",
    )
    parser.add_argument(
        "--ori-coef",
        type=float,
        default=0.0005,
        help="Legacy parameter kept for compatibility",
    )
    parser.add_argument(
        "--num-grasp-contacts",
        type=int,
        default=3,
        help="Number of contact points used for grasp synthesis",
    )
    parser.add_argument("--region-anchor-count", type=int, default=200)
    parser.add_argument("--region-radius", type=float, default=0.08)
    parser.add_argument("--region-max-points", type=int, default=256)
    parser.add_argument("--region-contact-samples", type=int, default=5)
    parser.add_argument("--top-region-pairs", type=int, default=3)
    parser.add_argument("--preselect-region-pairs", type=int, default=200)
    parser.add_argument("--friction-cone-edges", type=int, default=8)
    parser.add_argument("--gwb-wrench-count", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--force-reg-weight", type=float, default=1e-5)
    parser.add_argument("--concavity-tol", type=float, default=None)
    parser.add_argument("--max-point-combination-eval", type=int, default=256)
    parser.add_argument(
        "--solver",
        type=str,
        choices=("ipopt", "snopt", "acados"),
        default=None,
        help="Optional NLP solver used by the force-closure search. Defaults to the current mlqp_point_v2 setting.",
    )
    parser.add_argument(
        "--static-solver",
        type=str,
        choices=("ipopt", "snopt", "acados"),
        default=None,
        help="Optional solver used by static-equilibrium solve. Defaults to --solver when provided, otherwise the current mlqp_point_v2 setting.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show object mesh, best contact set, and force arrows in Open3D",
    )
    parser.add_argument("--force-scale", type=float, default=None)
    parser.add_argument("--point-radius", type=float, default=None)
    parser.add_argument("--hide-normals", action="store_true")
    parser.add_argument("--hide-region-samples", action="store_true")
    parser.add_argument(
        "--use-static-equilibrium-force",
        action="store_true",
        help="Visualize the gravity static-equilibrium contact forces instead of the worst-disturbance witness forces",
    )
    return parser


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    cli_args = build_argparser().parse_args()
    optimizer, grasp_result = run_grasp_prediction(cli_args)
    static_result = optimizer.solve_static_equilibrium(grasp_result)

    print("\nBest grasp result")
    print(f"mesh_path: {grasp_result['mesh_path']}")
    print(f"num_grasp_contacts: {grasp_result['contact_indices'].shape[0]}")
    print(f"contact_indices: {grasp_result['contact_indices']}")
    print(f"contact_points_local:\n{grasp_result['contact_points']}")
    print(f"contact_normals_local:\n{grasp_result['contact_normals']}")
    print(f"region_rank: {grasp_result.get('region_rank', 0)}")
    print(f"region_score(GWB): {grasp_result.get('region_score', 0.0):.6f}")
    print(
        f"force_closure_solver: {grasp_result.get('solver_backend', optimizer.force_closure_solver_backend)}"
    )
    print(f"force_closure_status: {grasp_result.get('solver_status', None)}")
    print(f"force_closure_cost: {grasp_result['force_closure_cost']:.6f}")
    print(f"total_cost: {grasp_result['total_cost']:.6f}")
    print(f"min_contact_distance: {grasp_result['min_contact_distance']:.6f}")
    print(f"antipodal_margin: {grasp_result['antipodal_margin']:.6f}")
    print(f"worst_disturbance: {grasp_result['worst_disturbance_label']}")
    print(
        f"witness_contact_forces_local:\n{grasp_result['witness_contact_forces_local']}"
    )
    print(
        f"witness_force_vectors_local:\n{grasp_result['witness_force_vectors_local']}"
    )
    print(f"static_external_wrench_local:\n{static_result['external_wrench']}")
    print(f"static_contact_forces_local:\n{static_result['contact_forces_local']}")
    print(f"static_force_vectors_local:\n{static_result['force_vectors_local']}")
    print(f"static_residual_norm: {static_result['residual_norm']:.6f}")
    print(
        f"static_solver: {static_result.get('solver_backend', optimizer.static_equilibrium_solver_backend)}"
    )
    print(f"static_status: {static_result.get('solver_status', None)}")
    print(f"solve_time: {grasp_result['solve_time']:.6f}s")
    print(f"wall_time: {grasp_result['wall_time']:.6f}s")

    # if cli_args.visualize:
    optimizer.visual_grasp(
        result=grasp_result,
        point_radius=cli_args.point_radius,
        force_scale=cli_args.force_scale,
        include_normals=not cli_args.hide_normals,
        include_region_samples=not cli_args.hide_region_samples,
        use_static_equilibrium=cli_args.use_static_equilibrium_force,
    )
