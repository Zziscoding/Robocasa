import ctypes
import os
import shutil
import sys
import time
import warnings

import casadi as cs
import numpy as np
import trimesh
from scipy.spatial import cKDTree

os.environ["SNOPT_LICENSE"] = "/home/lab423/opt_ws/libsnopt7/snopt7.lic"
from robocasa.demos.open_drawer.project_point import ProjectionPoint


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


def _normalize_nlp_solver_name(solver_name):
    if solver_name is None:
        solver_name = os.environ.get("LCC_CABINET_SOLVER", "snopt")
    solver_name = str(solver_name).strip().lower()
    if solver_name not in {"ipopt", "snopt", "acados"}:
        raise ValueError(
            f"Unsupported NLP solver '{solver_name}'. Expected 'ipopt', 'snopt' or 'acados'."
        )
    return solver_name


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
        pos_coef=1,
        ori_coef=0.0005,
        scale_factors=[1.0, 1.0, 1.0],
        sampling_stl_specs=None,
        nlp_solver=None,
    ):
        self.m = obj_mass
        self.mu_arm_obj = arm_friction
        self.K_contact = contact_stiffness
        self.h = time_step
        self.max_contacts = max_contacts
        self.nlp_solver = _normalize_nlp_solver_name(nlp_solver)
        self.pp = ProjectionPoint(mesh_path, scale_factors)
        self.sampling_stl_specs = sampling_stl_specs or {}

        self.sample_num = sample_num
        self.sampling_frame = self.pp.sample_vertices_with_normals(
            num_samples=self.sample_num
        )
        self.sample_point = self.sampling_frame["points"]
        self.normal = self.sampling_frame["normals"]
        self.t1 = self.sampling_frame["tangent1"]
        self.t2 = self.sampling_frame["tangent2"]
        self.sample_num = self.sample_point.shape[0]

        self.J_tilde = np.zeros([4 * self.max_contacts, 6])

        self.obj_inertia = np.eye(6)
        self.obj_inertia[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia[3:, 3:] = 0.05 * np.eye(3)
        q = np.zeros((6, 6))
        q[:6, :6] = self.obj_inertia
        self.Q_inv = np.linalg.inv(q + 1e-8 * np.eye(q.shape[0]))

        self.pos_coef = pos_coef
        self.ori_coef = ori_coef
        self.point_idx = np.arange(self.sample_num)
        self.sampling_part_masks = self._build_sampling_part_masks()
        self._default_lam_upper_bound = 2.0
        self.init_utils()
        self._precompile_optimization_function()

    @staticmethod
    def _apply_sampling_spec(vertices, spec):
        transformed_vertices = np.asarray(vertices, dtype=np.float64).copy()

        scale_factors = spec.get("scale_factors")
        if scale_factors is not None:
            scale_factors = np.asarray(scale_factors, dtype=np.float64)
            if scale_factors.ndim == 0:
                transformed_vertices *= float(scale_factors)
            else:
                transformed_vertices *= scale_factors.reshape(1, 3)

        rotation = spec.get("rotation")
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
            transformed_vertices = (rotation @ transformed_vertices.T).T

        translation = spec.get("translation")
        if translation is not None:
            translation = np.asarray(translation, dtype=np.float64).reshape(1, 3)
            transformed_vertices += translation

        transform = spec.get("transform")
        if transform is not None:
            transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
            homogeneous_vertices = np.hstack(
                [transformed_vertices, np.ones((transformed_vertices.shape[0], 1))]
            )
            transformed_vertices = (transform @ homogeneous_vertices.T).T[:, :3]

        return transformed_vertices

    def _load_sampling_spec_vertices(self, spec):
        mesh_path = spec.get("mesh_path")
        if mesh_path is None:
            raise ValueError("Each sampling STL spec must provide a mesh_path.")

        mesh = trimesh.load_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.size == 0:
            raise ValueError(f"Sampling STL mesh has no vertices: {mesh_path}")
        return self._apply_sampling_spec(vertices, spec)

    def _build_sampling_part_masks(self):
        if not self.sampling_stl_specs:
            return {}

        sample_points = np.asarray(self.sample_point, dtype=np.float64)
        closest_part_dist = np.full(sample_points.shape[0], np.inf, dtype=np.float64)
        closest_part_name = np.empty(sample_points.shape[0], dtype=object)
        closest_part_name[:] = None

        for stl_name, spec in self.sampling_stl_specs.items():
            stl_vertices = self._load_sampling_spec_vertices(spec)
            stl_tree = cKDTree(stl_vertices)
            distances, _ = stl_tree.query(sample_points, k=1)
            update_mask = distances < closest_part_dist
            closest_part_dist[update_mask] = distances[update_mask]
            closest_part_name[update_mask] = stl_name

        sampling_part_masks = {}
        for stl_name in self.sampling_stl_specs:
            part_mask = closest_part_name == stl_name
            spec = self.sampling_stl_specs[stl_name]

            point_bounds_min = spec.get("point_bounds_min")
            point_bounds_max = spec.get("point_bounds_max")
            if point_bounds_min is not None:
                point_bounds_min = np.asarray(point_bounds_min, dtype=np.float64)
                part_mask = part_mask & np.all(
                    sample_points >= point_bounds_min, axis=1
                )
            if point_bounds_max is not None:
                point_bounds_max = np.asarray(point_bounds_max, dtype=np.float64)
                part_mask = part_mask & np.all(
                    sample_points <= point_bounds_max, axis=1
                )

            sampling_part_masks[stl_name] = np.where(part_mask)[0]

        return sampling_part_masks

    def update_Jacobian(self, J_tilde=None):
        required_rows = 4 * self.max_contacts
        if J_tilde is None:
            return

        j_tilde = np.asarray(J_tilde, dtype=np.float64)[:, :6]
        current_rows = j_tilde.shape[0]
        if current_rows < required_rows:
            padding = np.zeros((required_rows - current_rows, 6), dtype=np.float64)
            self.J_tilde = np.concatenate([j_tilde, padding], axis=0)
        else:
            self.J_tilde = j_tilde[:required_rows, :6]

    def _precompile_optimization_function(self):
        if self.nlp_solver == "acados":
            self._solver_bundle = self._build_acados_solver_bundle(
                default_lam_upper_bound=self._default_lam_upper_bound
            )
        else:
            self._solver_bundle = self._build_solver_bundle(
                solver_name=self.nlp_solver,
                default_lam_upper_bound=self._default_lam_upper_bound,
            )
        self.optimization_fn = self._solver_bundle.get("optimization_fn")

    def _build_solver_bundle(self, solver_name=None, default_lam_upper_bound=2.0):
        solver_name = _normalize_nlp_solver_name(solver_name)
        if solver_name == "acados":
            solver_name = "snopt"

        opti = cs.Opti()

        x_d = opti.parameter(7)
        current_x = opti.parameter(7)
        J_tilde = opti.parameter(4 * self.max_contacts, 6)
        tau_o_np = opti.parameter(6)
        p_arm = opti.parameter(3)
        n_arm = opti.parameter(3)
        t1 = opti.parameter(3)
        t2 = opti.parameter(3)
        curr_ori_coef = opti.parameter(1)
        lam_upper_bound = opti.parameter(1)
        R_contact = cs.horzcat(n_arm, t1, t2)

        opti.set_value(lam_upper_bound, float(default_lam_upper_bound))

        lam_arm = opti.variable(3)
        opti.set_initial(lam_arm, [0.01, 0.0, 0.0])

        J_arm_world = self.compute_contact_jacobian(p_arm)
        b = tau_o_np + cs.transpose(J_arm_world) @ (R_contact @ lam_arm)

        k_contact = (self.K_contact * self.h) * cs.MX.eye(4 * self.max_contacts)
        q_inv_b = cs.MX(self.Q_inv) @ b
        j_tilde_q_inv_b = J_tilde @ q_inv_b
        contact_force = -k_contact @ j_tilde_q_inv_b
        contact_force = cs.fmax(contact_force, 0.0)

        v_plus = (
            q_inv_b / self.h + cs.MX(self.Q_inv) @ J_tilde.T @ contact_force / self.h
        )
        x_plus = self.cs_qposInteg_(current_x, v_plus)

        position_error = x_plus[:3] - x_d[:3]
        orientation_error = 1.0 - cs.dot(x_plus[3:7], x_d[3:7]) ** 2
        objective = (
            self.pos_coef * cs.norm_2(position_error)
            + self.ori_coef * orientation_error
        )
        opti.minimize(objective)

        mu = self.mu_arm_obj
        opti.subject_to(lam_arm[1] <= mu * lam_arm[0])
        opti.subject_to(lam_arm[1] >= -mu * lam_arm[0])
        opti.subject_to(lam_arm[2] <= mu * lam_arm[0])
        opti.subject_to(lam_arm[2] >= -mu * lam_arm[0])
        opti.subject_to(lam_arm[0] >= 0.001)
        opti.subject_to(lam_arm[0] <= lam_upper_bound)

        p_opts = {"print_time": False, "jit": False}
        if solver_name == "snopt":
            s_opts = {
                "Major iterations limit": 200,
                "Minor iterations limit": 100,
                "Major print level": 0,
                "Minor print level": 0,
                "Print file": 0,
                "Summary file": 0,
                "Total real workspace": 500000,
                "Total integer workspace": 500000,
                "Total character workspace": 500000,
            }
            opti.solver("snopt", p_opts, s_opts)
        else:
            s_opts = {
                "max_iter": 200,
                "tol": 1e-6,
                "linear_solver": "mumps",
                "print_level": 0,
            }
            opti.solver("ipopt", p_opts, s_opts)

        return {
            "backend": f"casadi_opti_{solver_name}",
            "solver_name": solver_name,
            "default_lam_upper_bound": float(default_lam_upper_bound),
            "opti": opti,
            "x_d": x_d,
            "current_x": current_x,
            "J_tilde_param": J_tilde,
            "tau_o_param": tau_o_np,
            "p_arm_param": p_arm,
            "n_arm_param": n_arm,
            "t1_param": t1,
            "t2_param": t2,
            "curr_ori_coef_param": curr_ori_coef,
            "lam_upper_bound_param": lam_upper_bound,
            "lam_arm_var": lam_arm,
            "x_plus_expr": x_plus,
            "objective_expr": objective,
            "last_lam_solution": np.array([0.01, 0.0, 0.0], dtype=np.float64),
        }

    def init_utils(self):
        quat = cs.SX.sym("quat", 4)
        h_q_body = cs.vertcat(
            cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
            cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
            cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]),
        )
        self.cs_qmat_body_fn_ = cs.Function("cs_qmat_body_fn", [quat], [h_q_body.T])

        qvel = cs.SX.sym("qvel", 6)
        qpos = cs.SX.sym("qpos", 7)
        next_obj_pos = qpos[0:3] + self.h * qvel[0:3]
        next_obj_quat = (
            qpos[3:7] + 0.5 * self.h * self.cs_qmat_body_fn_(qpos[3:7]) @ qvel[3:6]
        )
        next_obj_quat = next_obj_quat / cs.norm_2(next_obj_quat)
        next_qpos = cs.vertcat(next_obj_pos, next_obj_quat)
        self.cs_qposInteg_ = cs.Function("cs_qposInte", [qpos, qvel], [next_qpos])

    @staticmethod
    def compute_contact_jacobian(p):
        J_c = cs.MX.zeros(3, 6)
        J_c[:3, :3] = cs.MX.eye(3)
        J_c[0, 4], J_c[0, 5] = p[2], -p[1]
        J_c[1, 3], J_c[1, 5] = -p[2], p[0]
        J_c[2, 3], J_c[2, 4] = p[1], -p[0]
        return J_c

    @staticmethod
    def compute_contact_jacobian_sx(p):
        J_c = cs.SX.zeros(3, 6)
        J_c[:3, :3] = cs.SX.eye(3)
        J_c[0, 4], J_c[0, 5] = p[2], -p[1]
        J_c[1, 3], J_c[1, 5] = -p[2], p[0]
        J_c[2, 3], J_c[2, 4] = p[1], -p[0]
        return J_c

    def _pack_acados_parameter_vector(
        self,
        x_d,
        current_x,
        tau_o,
        n_arm,
        t1,
        t2,
        p_arm,
        curr_ori_coef,
        lam_upper_bound,
    ):
        return np.concatenate(
            [
                np.asarray(x_d, dtype=np.float64).reshape(7),
                np.asarray(current_x, dtype=np.float64).reshape(7),
                np.asarray(self.J_tilde, dtype=np.float64).reshape(-1, order="F"),
                np.asarray(tau_o, dtype=np.float64).reshape(6),
                np.asarray(n_arm, dtype=np.float64).reshape(3),
                np.asarray(t1, dtype=np.float64).reshape(3),
                np.asarray(t2, dtype=np.float64).reshape(3),
                np.asarray(p_arm, dtype=np.float64).reshape(3),
                np.asarray(curr_ori_coef, dtype=np.float64).reshape(1),
                np.asarray([lam_upper_bound], dtype=np.float64).reshape(1),
            ]
        )

    def _build_acados_solver_bundle(self, default_lam_upper_bound=2.0):
        (
            AcadosModel,
            AcadosOcp,
            AcadosOcpSolver,
            ACADOS_INFTY,
        ) = _import_acados_template_symbols()

        lam_arm = cs.SX.sym("lam_arm", 3)
        param_dim = 2 * 7 + (4 * self.max_contacts) * 6 + 6 + 3 + 3 + 3 + 3 + 1 + 1
        p = cs.SX.sym("p", param_dim)

        cursor = 0

        def _take(size):
            nonlocal cursor
            chunk = p[cursor : cursor + size]
            cursor += size
            return chunk

        x_d = _take(7)
        current_x = _take(7)
        j_tilde_flat = _take((4 * self.max_contacts) * 6)
        J_tilde = cs.reshape(j_tilde_flat, 4 * self.max_contacts, 6)
        tau_o = _take(6)
        n_arm = _take(3)
        t1 = _take(3)
        t2 = _take(3)
        p_arm = _take(3)
        _curr_ori_coef = _take(1)
        lam_upper_bound = _take(1)[0]

        R_contact = cs.horzcat(n_arm, t1, t2)
        J_arm_world = self.compute_contact_jacobian_sx(p_arm)
        b = tau_o + cs.transpose(J_arm_world) @ (R_contact @ lam_arm)

        k_contact = (self.K_contact * self.h) * cs.DM.eye(4 * self.max_contacts)
        q_inv = cs.DM(self.Q_inv)
        q_inv_b = q_inv @ b
        j_tilde_q_inv_b = J_tilde @ q_inv_b
        contact_force = -k_contact @ j_tilde_q_inv_b
        contact_force = cs.fmax(contact_force, 0.0)

        v_plus = q_inv_b / self.h + q_inv @ J_tilde.T @ contact_force / self.h
        x_plus = self.cs_qposInteg_(current_x, v_plus)

        position_error = x_plus[:3] - x_d[:3]
        orientation_error = 1.0 - cs.dot(x_plus[3:7], x_d[3:7]) ** 2
        objective = (
            self.pos_coef * cs.norm_2(position_error)
            + self.ori_coef * orientation_error
        )

        model = AcadosModel()
        model.name = (
            f"mlqp_point_cabinet_{_ACADOS_EXPORT_VERSION}_"
            f"c{self.max_contacts}_s{self.sample_num}"
        )
        model.x = lam_arm
        model.u = cs.SX.sym("u", 0, 0)
        model.disc_dyn_expr = lam_arm
        model.p = p
        model.cost_expr_ext_cost_e = objective
        model.con_h_expr_e = cs.vertcat(
            lam_arm[1] - self.mu_arm_obj * lam_arm[0],
            -lam_arm[1] - self.mu_arm_obj * lam_arm[0],
            lam_arm[2] - self.mu_arm_obj * lam_arm[0],
            -lam_arm[2] - self.mu_arm_obj * lam_arm[0],
            lam_arm[0] - lam_upper_bound,
        )

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros((param_dim,), dtype=np.float64)
        ocp.cost.cost_type_e = "EXTERNAL"

        ocp.constraints.idxbx_e = np.array([0], dtype=np.int64)
        ocp.constraints.lbx_e = np.array([1e-3], dtype=np.float64)
        ocp.constraints.ubx_e = np.array([ACADOS_INFTY], dtype=np.float64)
        ocp.constraints.lh_e = -ACADOS_INFTY * np.ones((5,), dtype=np.float64)
        ocp.constraints.uh_e = np.zeros((5,), dtype=np.float64)

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
        backend = "acados"
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
                fallback_bundle = self._build_solver_bundle(
                    solver_name="snopt",
                    default_lam_upper_bound=default_lam_upper_bound,
                )
                fallback_bundle["backend"] = "casadi_opti_fallback_from_acados"
                return fallback_bundle

        eval_fun = cs.Function(
            f"{model.name}_eval",
            [lam_arm, p],
            [x_plus, objective],
        )

        return {
            "backend": backend,
            "default_lam_upper_bound": float(default_lam_upper_bound),
            "solver": solver,
            "eval_fun": eval_fun,
            "last_lam_solution": np.array([0.01, 0.0, 0.0], dtype=np.float64),
        }

    def _solve_once(
        self,
        x_d,
        current_x,
        tau_o,
        n_arm,
        t1,
        t2,
        p_arm,
        curr_ori_coef,
        lam_upper_bound=None,
    ):
        bundle = self._solver_bundle
        if lam_upper_bound is None:
            lam_upper_bound = float(
                bundle.get("default_lam_upper_bound", self._default_lam_upper_bound)
            )
        lam_upper_bound = float(max(lam_upper_bound, 1e-3))

        if bundle.get("backend") in {"acados", "acados_casadi"}:
            param_vector = self._pack_acados_parameter_vector(
                x_d=x_d,
                current_x=current_x,
                tau_o=tau_o,
                n_arm=n_arm,
                t1=t1,
                t2=t2,
                p_arm=p_arm,
                curr_ori_coef=curr_ori_coef,
                lam_upper_bound=lam_upper_bound,
            )

            solver = bundle["solver"]
            lam_initial = (
                np.asarray(bundle["last_lam_solution"], dtype=np.float64)
                .reshape(3)
                .copy()
            )
            lam_initial[0] = np.clip(lam_initial[0], 1e-3, lam_upper_bound)
            tangential_bound = self.mu_arm_obj * lam_initial[0]
            lam_initial[1] = np.clip(
                lam_initial[1], -tangential_bound, tangential_bound
            )
            lam_initial[2] = np.clip(
                lam_initial[2], -tangential_bound, tangential_bound
            )

            status = None
            sqp_iter = None
            try:
                solver.set(0, "p", param_vector)
                solver.set(0, "x", lam_initial)
                status = solver.solve()
                lam_arm = np.asarray(solver.get(0, "x"), dtype=np.float32).reshape(3)
                try:
                    sqp_iter = int(solver.get_stats("sqp_iter"))
                except Exception:
                    sqp_iter = None
            except Exception:
                lam_arm = lam_initial.astype(np.float32)
                solver_status = "ACADOS_EXCEPTION"
            else:
                solver_status = _format_acados_status(status, sqp_iter=sqp_iter)

            x_plus_val, objective_val = bundle["eval_fun"](
                lam_arm.astype(np.float64), param_vector
            )
            x_plus_opt = np.asarray(x_plus_val, dtype=np.float32).reshape(7)
            objective_value = float(
                np.asarray(objective_val, dtype=np.float64).reshape(())
            )

            bundle["last_lam_solution"] = lam_arm.astype(np.float64)
            return lam_arm, x_plus_opt, objective_value, solver_status

        opti = bundle["opti"]
        opti.set_value(bundle["x_d"], np.asarray(x_d, dtype=np.float64).reshape(7))
        opti.set_value(
            bundle["current_x"], np.asarray(current_x, dtype=np.float64).reshape(7)
        )
        opti.set_value(
            bundle["J_tilde_param"], np.asarray(self.J_tilde, dtype=np.float64)
        )
        opti.set_value(
            bundle["tau_o_param"], np.asarray(tau_o, dtype=np.float64).reshape(6)
        )
        opti.set_value(
            bundle["n_arm_param"], np.asarray(n_arm, dtype=np.float64).reshape(3)
        )
        opti.set_value(bundle["t1_param"], np.asarray(t1, dtype=np.float64).reshape(3))
        opti.set_value(bundle["t2_param"], np.asarray(t2, dtype=np.float64).reshape(3))
        opti.set_value(
            bundle["p_arm_param"], np.asarray(p_arm, dtype=np.float64).reshape(3)
        )
        opti.set_value(
            bundle["curr_ori_coef_param"],
            np.asarray(curr_ori_coef, dtype=np.float64).reshape(1),
        )
        opti.set_value(
            bundle["lam_upper_bound_param"],
            np.asarray([lam_upper_bound], dtype=np.float64),
        )

        lam_initial = (
            np.asarray(bundle["last_lam_solution"], dtype=np.float64).reshape(3).copy()
        )
        lam_initial[0] = np.clip(lam_initial[0], 1e-3, lam_upper_bound)
        tangential_bound = self.mu_arm_obj * lam_initial[0]
        lam_initial[1] = np.clip(lam_initial[1], -tangential_bound, tangential_bound)
        lam_initial[2] = np.clip(lam_initial[2], -tangential_bound, tangential_bound)
        opti.set_initial(bundle["lam_arm_var"], lam_initial)

        try:
            sol = opti.solve()
            lam_arm = np.asarray(
                sol.value(bundle["lam_arm_var"]), dtype=np.float32
            ).reshape(3)
            x_plus_opt = np.asarray(
                sol.value(bundle["x_plus_expr"]), dtype=np.float32
            ).reshape(7)
            objective_value = float(sol.value(bundle["objective_expr"]))
            solver_status = str(opti.stats().get("return_status", "Solve_Succeeded"))
        except RuntimeError:
            lam_arm = np.asarray(
                opti.debug.value(bundle["lam_arm_var"]), dtype=np.float32
            ).reshape(3)
            x_plus_opt = np.asarray(
                opti.debug.value(bundle["x_plus_expr"]), dtype=np.float32
            ).reshape(7)
            objective_value = float(opti.debug.value(bundle["objective_expr"]))
            solver_status = str(opti.stats().get("return_status", "Solve_Failed"))

        bundle["last_lam_solution"] = lam_arm.astype(np.float64)
        return lam_arm, x_plus_opt, objective_value, solver_status

    def optimize_control_input(self, x_d, current_x, tau_o, p_arm=None):
        if p_arm is None:
            p_arm = np.array([-1, 0, 0])

        ori_align_sq = cs.dot(current_x[3:7], x_d[3:7]) ** 2
        th = 0.85
        scale = 10
        curr_ori_coef = 1.0 + cs.tanh(scale * (ori_align_sq - th))

        closest_idx, n, t1, t2 = self.pp.project_point_to_mesh(p_arm)
        p_obj_local = self.pp.scaled_mesh.vertices[closest_idx]
        normal_obj_local = self.pp.scaled_mesh.vertex_normals[closest_idx]

        start_time = time.time()
        lam_arm, x_plus_opt, objective_value, solver_status = self._solve_once(
            x_d=x_d,
            current_x=current_x,
            tau_o=tau_o,
            n_arm=n,
            t1=t1,
            t2=t2,
            p_arm=p_obj_local,
            curr_ori_coef=curr_ori_coef,
        )

        solver_backend = self._solver_bundle.get("backend")
        info = {
            "solve_time": time.time() - start_time,
            "control_input": lam_arm,
            "resulting_pose": x_plus_opt,
            "solver_status": solver_status,
            "nlp_solver": self.nlp_solver,
            "solver_backend": str(solver_backend if solver_backend else "unknown"),
        }

        return p_obj_local, -normal_obj_local, x_plus_opt, objective_value, info

    def choose_contact_points(self, x_d, current_x, tau_o, visible_face_idx):
        if not len(visible_face_idx):
            return self.sample_point[0], self.normal[0], 1.5, 1.0, 0

        ori_align_sq = cs.dot(current_x[3:7], x_d[3:7]) ** 2
        th = 0.85
        scale = 10
        curr_ori_coef = 1.0 + cs.tanh(scale * (ori_align_sq - th))

        error_list = np.zeros(visible_face_idx.shape[0], dtype=np.float64)
        for i, idx in enumerate(visible_face_idx):
            _, _, objective_value, _ = self._solve_once(
                x_d=x_d,
                current_x=current_x,
                tau_o=tau_o,
                n_arm=self.normal[idx],
                t1=self.t1[idx],
                t2=self.t2[idx],
                p_arm=self.sample_point[idx],
                curr_ori_coef=curr_ori_coef,
            )
            error_list[i] = objective_value

        min_error = float(np.min(error_list))
        max_error = float(np.max(error_list))
        min_idx = visible_face_idx[int(np.argmin(error_list))]
        return (
            self.sample_point[min_idx],
            self.normal[min_idx],
            min_error,
            max_error,
            curr_ori_coef,
        )

    def get_availble_point_idx(
        self, pos, R, target_pos, threshold=0.025, stl_names=None
    ):
        centers_world = (R @ self.sample_point.T).T + pos
        common_mask = centers_world[:, 2] > threshold

        if stl_names is not None:
            if isinstance(stl_names, str):
                stl_names = [stl_names]

            allowed_mask = np.zeros(self.sample_num, dtype=bool)
            unknown_stl_names = []
            for stl_name in stl_names:
                if stl_name not in self.sampling_part_masks:
                    unknown_stl_names.append(stl_name)
                    continue
                allowed_mask[self.sampling_part_masks[stl_name]] = True

            if unknown_stl_names:
                known_stl_names = sorted(self.sampling_part_masks)
                raise ValueError(
                    f"Unknown sampling STL names {unknown_stl_names}. "
                    f"Known names: {known_stl_names}"
                )

            common_mask = common_mask & allowed_mask

        return np.where(common_mask)[0]
