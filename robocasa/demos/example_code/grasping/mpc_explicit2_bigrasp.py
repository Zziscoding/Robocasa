import ctypes
import os
import shutil
import sys
import warnings

import casadi as cs
import numpy as np

from models.explicit_model import ExplicitModel


_ACADOS_TEMPLATE_SYMBOLS = None
_ACADOS_IMPORT_FAILURE = None
_ACADOS_EXPORT_VERSION = "v2"
_ACADOS_IPOPT_FALLBACK_WARNED = False
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


def _warn_acados_ipopt_fallback(reason):
    global _ACADOS_IPOPT_FALLBACK_WARNED
    if _ACADOS_IPOPT_FALLBACK_WARNED:
        return
    _ACADOS_IPOPT_FALLBACK_WARNED = True
    warnings.warn(
        "Falling back from the acados generated planner solver to IPOPT. "
        f"Reason: {reason}",
        RuntimeWarning,
    )


def _normalize_planner_solver_name(solver_name):
    if solver_name is None:
        solver_name = "ipopt"
    solver_name = str(solver_name).strip().lower()
    if solver_name not in {"ipopt", "acados"}:
        raise ValueError(
            f"Unsupported planner solver '{solver_name}'. Expected 'ipopt' or 'acados'."
        )
    return solver_name


def _as_vector(value, size):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.full((size,), arr.item(), dtype=np.float64)
    if arr.size != size:
        raise ValueError(f"Expected size {size}, got {arr.size}.")
    return arr


class MPCExplicit:
    def __init__(self, param):
        self.param_ = param
        self.mpc_model = self.param_.mpc_model
        if self.mpc_model != "explicit":
            raise ValueError(f"Invalid model type: {self.mpc_model}")

        self._init_defaults()
        self.planner_solver_ = _normalize_planner_solver_name(
            getattr(self.param_, "planner_solver_", "ipopt")
        )
        self.param_.planner_solver_ = self.planner_solver_
        self.path_cost_fn, self.final_cost_fn = self._build_cost_fns()
        self.model = ExplicitModel(param)
        self.init_MPC()

    def _init_defaults(self):
        self.ipopt_max_iter_ = int(getattr(self.param_, "ipopt_max_iter_", 500))
        self.param_.mpc_u_lb_ = _as_vector(self.param_.mpc_u_lb_, self.param_.n_cmd_)
        self.param_.mpc_u_ub_ = _as_vector(self.param_.mpc_u_ub_, self.param_.n_cmd_)

        if not hasattr(self.param_, "mpc_q_lb_"):
            self.param_.mpc_q_lb_ = -1e7 * np.ones(
                self.param_.n_qpos_, dtype=np.float64
            )
        if not hasattr(self.param_, "mpc_q_ub_"):
            self.param_.mpc_q_ub_ = 1e7 * np.ones(self.param_.n_qpos_, dtype=np.float64)

        self.param_.mpc_q_lb_ = _as_vector(self.param_.mpc_q_lb_, self.param_.n_qpos_)
        self.param_.mpc_q_ub_ = _as_vector(self.param_.mpc_q_ub_, self.param_.n_qpos_)

    @staticmethod
    def _log_barrier(point, target, epsilon=1e-3):
        diff = point - target
        return cs.log(cs.dot(diff, diff) + epsilon)

    def _build_cost_fns(self):
        x = cs.SX.sym("x", self.param_.n_qpos_)
        u = cs.SX.sym("u", self.param_.n_cmd_)

        target_position = cs.SX.sym("target_position", 3)
        target_quaternion = cs.SX.sym("target_quaternion", 4)
        phi_vec = cs.SX.sym("phi_vec", self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym("jac_mat", self.param_.max_ncon_ * 4, self.param_.n_qvel_)

        verify_cost_param_1 = cs.SX.sym("verify_cost_1", 1)
        verify_cost_param_2 = cs.SX.sym("verify_cost_2", 1)
        virtual_point_1 = cs.SX.sym("virtual_point_1", 3)
        virtual_point_2 = cs.SX.sym("virtual_point_2", 3)
        contact_point_1 = cs.SX.sym("contact_point_1", 3)
        contact_point_2 = cs.SX.sym("contact_point_2", 3)
        curr_ori_coef_1 = cs.SX.sym("curr_ori_coef_1", 1)
        curr_ori_coef_2 = cs.SX.sym("curr_ori_coef_2", 1)
        desired_force_world = cs.SX.sym("desired_force_world", 3)
        desired_torque_world = cs.SX.sym("desired_torque_world", 3)
        robot_contact_force_map_flat = cs.SX.sym(
            "robot_contact_force_map_flat", 3 * self.param_.max_ncon_ * 4
        )
        robot_contact_torque_map_flat = cs.SX.sym(
            "robot_contact_torque_map_flat", 3 * self.param_.max_ncon_ * 4
        )
        execute_desired_wrench = cs.SX.sym("execute_desired_wrench", 1)

        cost_params = cs.vvcat(
            [
                target_position,
                target_quaternion,
                phi_vec,
                jac_mat,
                verify_cost_param_1,
                verify_cost_param_2,
                virtual_point_1,
                virtual_point_2,
                contact_point_1,
                contact_point_2,
                curr_ori_coef_1,
                curr_ori_coef_2,
                desired_force_world,
                desired_torque_world,
                robot_contact_force_map_flat,
                robot_contact_torque_map_flat,
                execute_desired_wrench,
            ]
        )

        obj_pos = x[0:3]
        left_tip = x[7:10]
        right_tip = x[10:13]
        control_cost = cs.sumsqr(u)

        contact_cost_1 = cs.sumsqr(obj_pos - left_tip)
        contact_cost_2 = cs.sumsqr(obj_pos - right_tip)

        contact_point_cost_1 = self._log_barrier(left_tip, contact_point_1)
        contact_point_cost_2 = self._log_barrier(right_tip, contact_point_2)
        virtual_point_cost_1 = self._log_barrier(left_tip, virtual_point_1)
        virtual_point_cost_2 = self._log_barrier(right_tip, virtual_point_2)

        reject_cost_1 = cs.if_else(
            cs.sumsqr(left_tip - contact_point_1) < self.param_.reject_dis,
            -contact_point_cost_1,
            0.0,
        )
        reject_cost_2 = cs.if_else(
            cs.sumsqr(right_tip - contact_point_2) < self.param_.reject_dis,
            -contact_point_cost_2,
            0.0,
        )

        attract_cost_1 = (
            self.param_.attract_coef * virtual_point_cost_1
            + self.param_.reject_coef * reject_cost_1
        )
        attract_cost_2 = (
            self.param_.attract_coef * virtual_point_cost_2
            + self.param_.reject_coef * reject_cost_2
        )

        base_cost_1 = (
            1.0 - verify_cost_param_1
        ) * attract_cost_1 + self.param_.contact_coef * verify_cost_param_1 * (
            self.param_.contact_cost_param * contact_cost_1
            + (1.0 - self.param_.contact_cost_param) * contact_point_cost_1
        )
        base_cost_2 = (
            1.0 - verify_cost_param_2
        ) * attract_cost_2 + self.param_.contact_coef * verify_cost_param_2 * (
            self.param_.contact_cost_param * contact_cost_2
            + (1.0 - self.param_.contact_cost_param) * contact_point_cost_2
        )

        q_inv = np.linalg.inv(self.param_.Q)
        b_o = cs.DM(self.param_.obj_mass_ * self.param_.gravity_)
        b_r = cs.DM(self.param_.robot_stiff_) @ u
        b = cs.vertcat(b_o, b_r)
        raw_contact_force = -self.param_.model_params * (jac_mat @ q_inv @ b + phi_vec)
        contact_force = cs.fmax(raw_contact_force, 0)
        robot_contact_force_map = cs.reshape(
            robot_contact_force_map_flat, 3, self.param_.max_ncon_ * 4
        )
        robot_contact_torque_map = cs.reshape(
            robot_contact_torque_map_flat, 3, self.param_.max_ncon_ * 4
        )
        predicted_contact_force_world = robot_contact_force_map @ contact_force
        predicted_contact_torque_world = robot_contact_torque_map @ contact_force
        desired_wrench_gate = (
            execute_desired_wrench * verify_cost_param_1 * verify_cost_param_2
        )
        force_tracking_cost = (
            float(getattr(self.param_, "planner_force_tracking_weight_", 1.0))
            * desired_wrench_gate
            * cs.sumsqr(predicted_contact_force_world - desired_force_world)
        )
        torque_tracking_cost = (
            float(getattr(self.param_, "planner_torque_tracking_weight_", 1.0))
            * desired_wrench_gate
            * cs.sumsqr(predicted_contact_torque_world - desired_torque_world)
        )

        path_cost = (
            base_cost_1
            + base_cost_2
            + 50.0 * control_cost
            + force_tracking_cost
            + torque_tracking_cost
        )

        terminal_virtual_cost_1 = cs.sumsqr(left_tip - virtual_point_1)
        terminal_virtual_cost_2 = cs.sumsqr(right_tip - virtual_point_2)
        terminal_contact_cost_1 = cs.sumsqr(left_tip - contact_point_1)
        terminal_contact_cost_2 = cs.sumsqr(right_tip - contact_point_2)

        final_cost_1 = (
            1.0 - verify_cost_param_1
        ) * terminal_virtual_cost_1 + verify_cost_param_1 * terminal_contact_cost_1
        final_cost_2 = (
            1.0 - verify_cost_param_2
        ) * terminal_virtual_cost_2 + verify_cost_param_2 * terminal_contact_cost_2
        final_cost = 10.0 * 500.0 * (final_cost_1 + final_cost_2)

        path_cost_fn = cs.Function(
            "path_cost_fn_bigrasp", [x, u, cost_params], [path_cost]
        )
        final_cost_fn = cs.Function(
            "final_cost_fn_bigrasp", [x, cost_params], [final_cost]
        )
        return path_cost_fn, final_cost_fn

    def _build_cost_param_vector(
        self,
        target_p,
        target_q,
        phi_vec,
        jac_mat,
        verify_cost_param_1,
        verify_cost_param_2,
        virtual_point_1,
        virtual_point_2,
        contact_point_1,
        contact_point_2,
        curr_ori_coef_1,
        curr_ori_coef_2,
        desired_force_world=None,
        desired_torque_world=None,
        robot_contact_force_map=None,
        robot_contact_torque_map=None,
        execute_desired_wrench=False,
    ):
        phi_vec = np.asarray(phi_vec, dtype=np.float64).reshape(
            self.param_.max_ncon_ * 4
        )
        jac_mat = np.asarray(jac_mat, dtype=np.float64).reshape(
            self.param_.max_ncon_ * 4,
            self.param_.n_qvel_,
        )
        if desired_force_world is None:
            desired_force_world = np.zeros(3, dtype=np.float64)
        else:
            desired_force_world = np.asarray(
                desired_force_world, dtype=np.float64
            ).reshape(3)
        if desired_torque_world is None:
            desired_torque_world = np.zeros(3, dtype=np.float64)
        else:
            desired_torque_world = np.asarray(
                desired_torque_world, dtype=np.float64
            ).reshape(3)
        if robot_contact_force_map is None:
            robot_contact_force_map = np.zeros(
                (3, self.param_.max_ncon_ * 4), dtype=np.float64
            )
        else:
            robot_contact_force_map = np.asarray(
                robot_contact_force_map,
                dtype=np.float64,
            ).reshape(3, self.param_.max_ncon_ * 4)
        if robot_contact_torque_map is None:
            robot_contact_torque_map = np.zeros(
                (3, self.param_.max_ncon_ * 4), dtype=np.float64
            )
        else:
            robot_contact_torque_map = np.asarray(
                robot_contact_torque_map,
                dtype=np.float64,
            ).reshape(3, self.param_.max_ncon_ * 4)
        execute_desired_wrench = float(bool(execute_desired_wrench))

        return np.concatenate(
            [
                np.asarray(target_p, dtype=np.float64).reshape(3),
                np.asarray(target_q, dtype=np.float64).reshape(4),
                phi_vec,
                jac_mat.reshape(-1, order="F"),
                np.asarray([verify_cost_param_1], dtype=np.float64),
                np.asarray([verify_cost_param_2], dtype=np.float64),
                np.asarray(virtual_point_1, dtype=np.float64).reshape(3),
                np.asarray(virtual_point_2, dtype=np.float64).reshape(3),
                np.asarray(contact_point_1, dtype=np.float64).reshape(3),
                np.asarray(contact_point_2, dtype=np.float64).reshape(3),
                np.asarray([curr_ori_coef_1], dtype=np.float64),
                np.asarray([curr_ori_coef_2], dtype=np.float64),
                desired_force_world,
                desired_torque_world,
                robot_contact_force_map.reshape(-1, order="F"),
                robot_contact_torque_map.reshape(-1, order="F"),
                np.asarray([execute_desired_wrench], dtype=np.float64),
            ],
            axis=0,
        )

    def _rollout_ctrl_guess(self, ctrls, curr_x, phi_vec, jac_mat):
        ctrls = np.asarray(ctrls, dtype=np.float64).reshape(
            self.param_.mpc_horizon_, self.param_.n_cmd_
        )
        qk = np.asarray(curr_x, dtype=np.float64).reshape(self.param_.n_qpos_)
        phi_vec = np.asarray(phi_vec, dtype=np.float64).reshape(
            self.param_.max_ncon_ * 4
        )
        jac_mat = np.asarray(jac_mat, dtype=np.float64).reshape(
            self.param_.max_ncon_ * 4, self.param_.n_qvel_
        )

        w0 = []
        rollout_q = []
        for uk in ctrls:
            pred_q = np.asarray(
                self.model.step_once_fn(
                    qk, uk, phi_vec, jac_mat, self.param_.model_params
                ).full()
            ).reshape(-1)
            w0.extend([uk, pred_q])
            rollout_q.append(pred_q)
            qk = pred_q

        x0 = np.concatenate(
            [np.asarray(block, dtype=np.float64).reshape(-1) for block in w0], axis=0
        )
        rollout_q = np.asarray(rollout_q, dtype=np.float64)
        return x0, rollout_q

    def _extract_ctrl_guess(self, sol_guess):
        if sol_guess is None:
            return None

        ctrl_guess = sol_guess.get("u_traj")
        if ctrl_guess is None:
            ctrl_guess = sol_guess.get("best_u_traj")
        if ctrl_guess is not None:
            return np.asarray(ctrl_guess, dtype=np.float64).reshape(
                self.param_.mpc_horizon_, self.param_.n_cmd_
            )

        stacked_guess = sol_guess.get("x0")
        if stacked_guess is None:
            return None

        stacked_guess = np.asarray(stacked_guess, dtype=np.float64).reshape(-1)
        block_dim = self.param_.n_cmd_ + self.param_.n_qpos_
        expected_size = self.param_.mpc_horizon_ * block_dim
        if stacked_guess.size != expected_size:
            return None

        sol_traj = stacked_guess.reshape(self.param_.mpc_horizon_, block_dim)
        return sol_traj[:, : self.param_.n_cmd_]

    def _prepare_sol_guess(self, sol_guess, curr_x, phi_vec, jac_mat):
        if sol_guess is None:
            return dict(
                x0=self.nlp_w0_,
                lam_x0=self.nlp_lam_x0_,
                lam_g0=self.nlp_lam_g0_,
            )

        if "x0" in sol_guess:
            x0 = np.asarray(sol_guess["x0"], dtype=np.float64).reshape(-1)
            if x0.size == int(self.nlp_w0_.shape[0]):
                return dict(
                    x0=x0,
                    lam_x0=sol_guess.get("lam_x0", self.nlp_lam_x0_),
                    lam_g0=sol_guess.get("lam_g0", self.nlp_lam_g0_),
                )

        ctrl_guess = self._extract_ctrl_guess(sol_guess)
        if ctrl_guess is not None:
            x0, _ = self._rollout_ctrl_guess(ctrl_guess, curr_x, phi_vec, jac_mat)
            return dict(
                x0=x0,
                lam_x0=self.nlp_lam_x0_,
                lam_g0=self.nlp_lam_g0_,
            )

        return dict(
            x0=self.nlp_w0_,
            lam_x0=self.nlp_lam_x0_,
            lam_g0=self.nlp_lam_g0_,
        )

    def _prepare_acados_guess(self, sol_guess, curr_x, phi_vec, jac_mat):
        ctrl_guess = self._extract_ctrl_guess(sol_guess)
        if ctrl_guess is None:
            ctrl_guess = np.zeros(
                (self.param_.mpc_horizon_, self.param_.n_cmd_), dtype=np.float64
            )

        _, rollout_q = self._rollout_ctrl_guess(ctrl_guess, curr_x, phi_vec, jac_mat)
        x_traj = np.vstack(
            [
                np.asarray(curr_x, dtype=np.float64).reshape(1, self.param_.n_qpos_),
                rollout_q,
            ]
        )
        return dict(u_traj=ctrl_guess, x_traj=x_traj)

    @staticmethod
    def _shift_traj(ctrls):
        shifted = np.zeros_like(ctrls)
        shifted[:-1] = ctrls[1:]
        shifted[-1] = ctrls[-1]
        return shifted

    def _build_shifted_sol_guess(
        self, opt_u_traj, curr_x, phi_vec, jac_mat, backend, opt_cost, solve_status
    ):
        opt_u_traj = np.asarray(opt_u_traj, dtype=np.float64).reshape(
            self.param_.mpc_horizon_, self.param_.n_cmd_
        )
        shifted_u_traj = self._shift_traj(opt_u_traj)
        shifted_x0, shifted_rollout_q = self._rollout_ctrl_guess(
            shifted_u_traj, curr_x, phi_vec, jac_mat
        )
        shifted_x_traj = np.vstack(
            [
                np.asarray(curr_x, dtype=np.float64).reshape(1, self.param_.n_qpos_),
                shifted_rollout_q,
            ]
        )
        return dict(
            x0=shifted_x0,
            u_traj=shifted_u_traj,
            best_u_traj=opt_u_traj,
            x_traj=shifted_x_traj,
            opt_cost=float(opt_cost),
            solver_backend=str(backend),
            solve_status=str(solve_status),
        )

    def _evaluate_rollout_cost(self, x_traj, u_traj, cost_params):
        x_traj = np.asarray(x_traj, dtype=np.float64).reshape(
            self.param_.mpc_horizon_ + 1, self.param_.n_qpos_
        )
        u_traj = np.asarray(u_traj, dtype=np.float64).reshape(
            self.param_.mpc_horizon_, self.param_.n_cmd_
        )
        cost_params = np.asarray(cost_params, dtype=np.float64).reshape(-1)

        total_cost = 0.0
        for stage_idx in range(self.param_.mpc_horizon_):
            total_cost += float(
                self.path_cost_fn(x_traj[stage_idx], u_traj[stage_idx], cost_params)
                .full()
                .item()
            )
        total_cost += float(self.final_cost_fn(x_traj[-1], cost_params).full().item())
        return total_cost

    def _build_acados_param_vector(self, phi_vec, jac_mat, cost_params):
        phi_vec = np.asarray(phi_vec, dtype=np.float64).reshape(
            self.param_.max_ncon_ * 4
        )
        jac_mat = np.asarray(jac_mat, dtype=np.float64).reshape(
            self.param_.max_ncon_ * 4, self.param_.n_qvel_
        )
        cost_params = np.asarray(cost_params, dtype=np.float64).reshape(-1)
        model_param = np.asarray([self.param_.model_params], dtype=np.float64)

        return np.asarray(
            self.acados_params_fn_(phi_vec, jac_mat, cost_params, model_param),
            dtype=np.float64,
        ).reshape(-1)

    def _plan_once_ipopt(
        self,
        curr_x,
        phi_vec,
        jac_mat,
        cost_params,
        sol_guess,
        requested_solver,
        fallback_reason=None,
    ):
        warm_start = self._prepare_sol_guess(sol_guess, curr_x, phi_vec, jac_mat)

        nlp_param = self.nlp_params_fn_(
            curr_x, phi_vec, jac_mat, cost_params, self.param_.model_params
        )
        nlp_lbw, nlp_ubw = self.nlp_bounds_fn_(
            self.param_.mpc_u_lb_,
            self.param_.mpc_u_ub_,
            self.param_.mpc_q_lb_,
            self.param_.mpc_q_ub_,
        )

        raw_sol = self.ipopt_solver(
            x0=warm_start["x0"],
            lam_x0=warm_start["lam_x0"],
            lam_g0=warm_start["lam_g0"],
            lbx=nlp_lbw,
            ubx=nlp_ubw,
            lbg=0.0,
            ubg=0.0,
            p=nlp_param,
        )

        w_opt = raw_sol["x"].full().flatten()
        cost_opt = raw_sol["f"].full().flatten()
        sol_traj = np.reshape(
            w_opt, (self.param_.mpc_horizon_, self.param_.n_cmd_ + self.param_.n_qpos_)
        )
        opt_u_traj = sol_traj[:, : self.param_.n_cmd_]
        rollout_q = sol_traj[:, self.param_.n_cmd_ :]

        solve_status = self.ipopt_solver.stats()["return_status"]
        shifted_sol_guess = self._build_shifted_sol_guess(
            opt_u_traj,
            curr_x,
            phi_vec,
            jac_mat,
            backend="ipopt",
            opt_cost=raw_sol["f"].full().item(),
            solve_status=solve_status,
        )
        shifted_sol_guess["lam_x0"] = self.nlp_lam_x0_
        shifted_sol_guess["lam_g0"] = self.nlp_lam_g0_

        result = dict(
            action=opt_u_traj[0, :],
            rollout_q=rollout_q,
            sol_guess=shifted_sol_guess,
            cost_opt=cost_opt,
            solve_status=solve_status,
            solver_backend="ipopt",
            requested_solver=str(requested_solver),
        )
        if fallback_reason is not None:
            result["fallback_reason"] = str(fallback_reason)
        return result

    def _plan_once_acados(self, curr_x, phi_vec, jac_mat, cost_params, sol_guess):
        bundle = self.acados_solver_bundle_
        if bundle is None:
            return None

        curr_x = np.asarray(curr_x, dtype=np.float64).reshape(self.param_.n_qpos_)
        warm_start = self._prepare_acados_guess(sol_guess, curr_x, phi_vec, jac_mat)
        stage_param_vector = self._build_acados_param_vector(
            phi_vec, jac_mat, cost_params
        )
        solver = bundle["solver"]

        try:
            for stage_idx in range(self.param_.mpc_horizon_):
                solver.set(stage_idx, "p", stage_param_vector)
                solver.set(stage_idx, "u", warm_start["u_traj"][stage_idx])
                solver.set(stage_idx, "x", warm_start["x_traj"][stage_idx])

            solver.set(self.param_.mpc_horizon_, "p", stage_param_vector)
            solver.set(
                self.param_.mpc_horizon_,
                "x",
                warm_start["x_traj"][self.param_.mpc_horizon_],
            )
            solver.set(0, "lbx", curr_x)
            solver.set(0, "ubx", curr_x)

            status = solver.solve()
            try:
                sqp_iter = int(solver.get_stats("sqp_iter"))
            except Exception:
                sqp_iter = None

            x_traj = np.asarray(
                [
                    solver.get(stage_idx, "x")
                    for stage_idx in range(self.param_.mpc_horizon_ + 1)
                ],
                dtype=np.float64,
            ).reshape(self.param_.mpc_horizon_ + 1, self.param_.n_qpos_)
            u_traj = np.asarray(
                [
                    solver.get(stage_idx, "u")
                    for stage_idx in range(self.param_.mpc_horizon_)
                ],
                dtype=np.float64,
            ).reshape(self.param_.mpc_horizon_, self.param_.n_cmd_)
        except Exception as exc:
            _warn_acados_ipopt_fallback(f"planner solve exception: {exc}")
            return None

        if not np.all(np.isfinite(x_traj)) or not np.all(np.isfinite(u_traj)):
            _warn_acados_ipopt_fallback("planner returned non-finite iterates.")
            return None

        solve_status = _format_acados_status(status, sqp_iter=sqp_iter)
        cost_value = self._evaluate_rollout_cost(x_traj, u_traj, cost_params)
        bundle["last_x_solution"] = x_traj.copy()
        bundle["last_u_solution"] = u_traj.copy()

        return dict(
            action=u_traj[0, :],
            rollout_q=x_traj[1:, :],
            sol_guess=self._build_shifted_sol_guess(
                u_traj,
                curr_x,
                phi_vec,
                jac_mat,
                backend="acados",
                opt_cost=cost_value,
                solve_status=solve_status,
            ),
            cost_opt=np.asarray([cost_value], dtype=np.float64),
            solve_status=solve_status,
            solver_backend="acados",
            requested_solver="acados",
        )

    def plan_once(
        self,
        target_p,
        target_q,
        curr_x,
        phi_vec,
        jac_mat,
        verify_cost_param=None,
        virtual_point=None,
        contact_point=None,
        curr_ori_coef=None,
        sol_guess=None,
        verify_cost_param_1=None,
        verify_cost_param_2=None,
        virtual_point_1=None,
        virtual_point_2=None,
        contact_point_1=None,
        contact_point_2=None,
        curr_ori_coef_1=None,
        curr_ori_coef_2=None,
        desired_force_world=None,
        desired_torque_world=None,
        robot_contact_force_map=None,
        robot_contact_torque_map=None,
        execute_desired_wrench=False,
        solver_name=None,
    ):
        if verify_cost_param_1 is None or verify_cost_param_2 is None:
            raise ValueError(
                "planning/mpc_explicit2_bigrasp.py implements the dual-contact bigrasp objective. "
                "Please provide verify_cost_param_1/2, virtual_point_1/2, and contact_point_1/2."
            )

        curr_x = np.asarray(curr_x, dtype=np.float64).reshape(self.param_.n_qpos_)
        phi_vec = np.asarray(phi_vec, dtype=np.float64).reshape(
            self.param_.max_ncon_ * 4
        )
        jac_mat = np.asarray(jac_mat, dtype=np.float64).reshape(
            self.param_.max_ncon_ * 4, self.param_.n_qvel_
        )

        target_p = np.asarray(target_p, dtype=np.float64).reshape(3)
        target_q = np.asarray(target_q, dtype=np.float64).reshape(4)
        verify_cost_param_1 = float(verify_cost_param_1)
        verify_cost_param_2 = float(verify_cost_param_2)
        virtual_point_1 = np.asarray(virtual_point_1, dtype=np.float64).reshape(3)
        virtual_point_2 = np.asarray(virtual_point_2, dtype=np.float64).reshape(3)
        contact_point_1 = np.asarray(contact_point_1, dtype=np.float64).reshape(3)
        contact_point_2 = np.asarray(contact_point_2, dtype=np.float64).reshape(3)
        curr_ori_coef_1 = 0.0 if curr_ori_coef_1 is None else float(curr_ori_coef_1)
        curr_ori_coef_2 = 0.0 if curr_ori_coef_2 is None else float(curr_ori_coef_2)

        cost_params = self._build_cost_param_vector(
            target_p,
            target_q,
            phi_vec,
            jac_mat,
            verify_cost_param_1,
            verify_cost_param_2,
            virtual_point_1,
            virtual_point_2,
            contact_point_1,
            contact_point_2,
            curr_ori_coef_1,
            curr_ori_coef_2,
            desired_force_world=desired_force_world,
            desired_torque_world=desired_torque_world,
            robot_contact_force_map=robot_contact_force_map,
            robot_contact_torque_map=robot_contact_torque_map,
            execute_desired_wrench=execute_desired_wrench,
        )

        requested_solver = _normalize_planner_solver_name(
            self.planner_solver_ if solver_name is None else solver_name
        )

        if requested_solver == "acados":
            if self.acados_solver_bundle_ is None and not self.acados_build_attempted_:
                self.acados_build_attempted_ = True
                self.acados_solver_bundle_ = self._build_acados_solver_bundle()
            acados_result = self._plan_once_acados(
                curr_x, phi_vec, jac_mat, cost_params, sol_guess
            )
            if acados_result is not None:
                return acados_result
            return self._plan_once_ipopt(
                curr_x,
                phi_vec,
                jac_mat,
                cost_params,
                sol_guess,
                requested_solver=requested_solver,
                fallback_reason="acados_unavailable_or_failed",
            )

        return self._plan_once_ipopt(
            curr_x,
            phi_vec,
            jac_mat,
            cost_params,
            sol_guess,
            requested_solver=requested_solver,
        )

    def _build_acados_solver_bundle(self):
        try:
            (
                AcadosModel,
                AcadosOcp,
                AcadosOcpSolver,
                ACADOS_INFTY,
            ) = _import_acados_template_symbols()
        except Exception as exc:
            _warn_acados_ipopt_fallback(str(exc))
            return None

        model_params = cs.SX.sym("model_param", 1)
        phi_vec = cs.SX.sym("phi_vec", self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym("jac_mat", self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        cost_params = cs.SX.sym("cost_params", self.path_cost_fn.size_in(2))
        x = cs.SX.sym("x", self.param_.n_qpos_)
        u = cs.SX.sym("u", self.param_.n_cmd_)
        stage_params = cs.vvcat([phi_vec, jac_mat, cost_params, model_params])

        model = AcadosModel()
        model.name = (
            f"mpc_explicit2_bigrasp_"
            f"{_ACADOS_EXPORT_VERSION}_h{self.param_.mpc_horizon_}_c{self.param_.max_ncon_}"
        )
        model.x = x
        model.u = u
        model.p = stage_params
        model.disc_dyn_expr = self.model.step_once_fn(
            x, u, phi_vec, jac_mat, model_params
        )
        model.cost_expr_ext_cost = self.path_cost_fn(x, u, cost_params)
        model.cost_expr_ext_cost_e = self.final_cost_fn(x, cost_params)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros((int(stage_params.size1()),), dtype=np.float64)
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        idxu = np.arange(self.param_.n_cmd_, dtype=np.int64)
        idxx = np.arange(self.param_.n_qpos_, dtype=np.int64)
        zero_x = np.zeros((self.param_.n_qpos_,), dtype=np.float64)

        ocp.constraints.idxbu = idxu
        ocp.constraints.lbu = np.asarray(self.param_.mpc_u_lb_, dtype=np.float64).copy()
        ocp.constraints.ubu = np.asarray(self.param_.mpc_u_ub_, dtype=np.float64).copy()

        ocp.constraints.idxbx = idxx
        ocp.constraints.lbx = np.asarray(self.param_.mpc_q_lb_, dtype=np.float64).copy()
        ocp.constraints.ubx = np.asarray(self.param_.mpc_q_ub_, dtype=np.float64).copy()

        ocp.constraints.idxbx_e = idxx
        ocp.constraints.lbx_e = np.asarray(
            self.param_.mpc_q_lb_, dtype=np.float64
        ).copy()
        ocp.constraints.ubx_e = np.asarray(
            self.param_.mpc_q_ub_, dtype=np.float64
        ).copy()

        ocp.constraints.x0 = zero_x.copy()
        ocp.constraints.idxbx_0 = idxx
        ocp.constraints.lbx_0 = zero_x.copy()
        ocp.constraints.ubx_0 = zero_x.copy()

        ocp.solver_options.N_horizon = int(self.param_.mpc_horizon_)
        ocp.solver_options.tf = float(self.param_.h_ * self.param_.mpc_horizon_)
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.regularize_method = "MIRROR"
        ocp.solver_options.nlp_solver_ext_qp_res = 1
        ocp.solver_options.nlp_solver_max_iter = 20
        ocp.solver_options.qp_solver_iter_max = 400
        ocp.solver_options.tol = 1e-4
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
                _warn_acados_ipopt_fallback(str(exc))
                return None

        return {
            "backend": "acados",
            "solver": solver,
            "last_x_solution": np.zeros(
                (self.param_.mpc_horizon_ + 1, self.param_.n_qpos_), dtype=np.float64
            ),
            "last_u_solution": np.zeros(
                (self.param_.mpc_horizon_, self.param_.n_cmd_), dtype=np.float64
            ),
        }

    def init_MPC(self):
        model_params = cs.SX.sym("model_param", 1)
        phi_vec = cs.SX.sym("phi_vec", self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym("jac_mat", self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        cost_params = cs.SX.sym("cost_params", self.path_cost_fn.size_in(2))

        lbu = cs.SX.sym("lbu", self.param_.n_cmd_)
        ubu = cs.SX.sym("ubu", self.param_.n_cmd_)
        lbq = cs.SX.sym("lbq", self.param_.n_qpos_)
        ubq = cs.SX.sym("ubq", self.param_.n_qpos_)

        w, w0, lbw, ubw, g = [], [], [], [], []
        j = 0.0
        q0 = cs.SX.sym("q0", self.param_.n_qpos_)
        qk = q0

        for k in range(self.param_.mpc_horizon_):
            uk = cs.SX.sym(f"u{k}", self.param_.n_cmd_)
            w += [uk]
            lbw += [lbu]
            ubw += [ubu]
            w0 += [cs.DM.zeros(self.param_.n_cmd_)]

            pred_q = self.model.step_once_fn(qk, uk, phi_vec, jac_mat, model_params)
            j += self.path_cost_fn(qk, uk, cost_params)

            qk = cs.SX.sym(f"q{k + 1}", self.param_.n_qpos_)
            w += [qk]
            w0 += [cs.DM.zeros(self.param_.n_qpos_)]
            lbw += [lbq]
            ubw += [ubq]
            g += [pred_q - qk]

        j += self.final_cost_fn(qk, cost_params)

        nlp_params = cs.vvcat([q0, phi_vec, jac_mat, cost_params, model_params])
        nlp_prog = {"f": j, "x": cs.vcat(w), "g": cs.vcat(g), "p": nlp_params}
        nlp_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": self.ipopt_max_iter_,
            "ipopt.tol": 1e-4,
            "ipopt.linear_solver": "mumps",
        }
        self.ipopt_solver = cs.nlpsol("solver", "ipopt", nlp_prog, nlp_opts)

        self.nlp_w0_ = cs.vcat(w0)
        self.nlp_lam_x0_ = cs.DM.zeros(self.nlp_w0_.shape)
        self.nlp_lam_g0_ = cs.DM.zeros(cs.vcat(g).shape)
        self.nlp_bounds_fn_ = cs.Function(
            "nlp_bounds_fn", [lbu, ubu, lbq, ubq], [cs.vcat(lbw), cs.vvcat(ubw)]
        )
        self.nlp_params_fn_ = cs.Function(
            "nlp_params_fn",
            [q0, phi_vec, jac_mat, cost_params, model_params],
            [nlp_params],
        )
        self.acados_params_fn_ = cs.Function(
            "acados_params_fn_bigrasp",
            [phi_vec, jac_mat, cost_params, model_params],
            [cs.vvcat([phi_vec, jac_mat, cost_params, model_params])],
        )

        self.acados_solver_bundle_ = None
        self.acados_build_attempted_ = False
        if self.planner_solver_ == "acados":
            self.acados_build_attempted_ = True
            self.acados_solver_bundle_ = self._build_acados_solver_bundle()
