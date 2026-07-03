import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from scipy.linalg import pinv
from scipy.spatial.transform import Rotation

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
)
sys.path.append(parent_dir)

from robocasa.demos.open_drawer.params import (
    DRAWER_SAMPLING_HANDLE_NAME,
    DRAWER_SAMPLING_FRONT_PANEL_NAME,
    ExplicitMPCParams,
)
from robocasa.demos.open_drawer.mpc_explicit2 import MPCExplicit
from robocasa.demos.open_drawer.mpc_implicit import MPCImplicit
from robocasa.demos.open_drawer.screenshot import (
    PeriodicSVGScreenshotRecorder,
    build_free_camera_config,
    build_free_camera_config_from_position,
    create_mujoco_mp4_recorder,
)
from utils import metrics, rotations

DEFAULT_DRAWER_JOINT_FRICTIONLOSS = 10.0


TIMING_WINDOW = 10
TIMING_METRICS = [
    "exchange cost time",
    "cso solving time",
    "rs cost time",
    "cito cost time",
    "cpo cost time",
]
timing_history = {name: [0.0] * TIMING_WINDOW for name in TIMING_METRICS}


def update_timing_stats(metric_name, value):
    history = timing_history[metric_name]
    history.pop(0)
    history.append(float(value))
    history_array = np.asarray(history, dtype=np.float64)
    mean_value = float(np.mean(history_array))
    std_value = float(np.std(history_array))
    print(
        f"{metric_name} = {float(value):.6f} s | "
        f"mean(10) = {mean_value:.6f} s | std(10) = {std_value:.6f} s"
    )


def add_se3_noise(pos, quat, sigma_p, sigma_r):
    dp = np.random.randn(3) * sigma_p
    pos_noisy = pos + dp

    dtheta = np.random.randn(3) * sigma_r
    angle = np.linalg.norm(dtheta)

    if angle < 1e-12:
        dq = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = dtheta / angle
        dq = np.hstack([np.cos(angle / 2.0), axis * np.sin(angle / 2.0)])

    quat_noisy = np.zeros(4)
    mujoco.mju_mulQuat(quat_noisy, quat, dq)
    quat_noisy /= np.linalg.norm(quat_noisy)

    return pos_noisy, quat_noisy


def _filter_point_indices_to_upper_world_z_region(
    optimizer,
    pos,
    rotation_matrix,
    candidate_idx,
    upper_fraction=0.2,
):
    candidate_idx = np.asarray(candidate_idx, dtype=np.int32).reshape(-1)
    if candidate_idx.size == 0:
        return candidate_idx

    upper_fraction = float(upper_fraction)
    if upper_fraction <= 1e-6:
        return candidate_idx
    upper_fraction = min(upper_fraction, 1.0)
    if upper_fraction >= 1.0:
        return candidate_idx

    sample_points = np.asarray(optimizer.sample_point, dtype=np.float64)
    centers_world_all = (
        np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3) @ sample_points.T
    ).T + np.asarray(pos, dtype=np.float64).reshape(3)
    z_coords_all = centers_world_all[:, 2]
    z_min = float(np.min(z_coords_all))
    z_max = float(np.max(z_coords_all))
    z_span = z_max - z_min
    if z_span <= 1e-9:
        return candidate_idx

    z_cutoff = z_max - upper_fraction * z_span
    keep_mask = centers_world_all[candidate_idx, 2] >= z_cutoff
    filtered_idx = candidate_idx[keep_mask]
    return filtered_idx if filtered_idx.size > 0 else candidate_idx


def build_default_drawer_scene_camera_config(env):
    drawer_pos, _ = env.get_object_frame()
    ee_pos, _ = env.get_end_effector_pos()
    lookat = 0.5 * (drawer_pos + ee_pos)
    lookat[2] += 0.05

    camera_position = lookat + np.array(
        [-0.2, -1.35, 0.28],
        dtype=np.float64,
    )
    return build_free_camera_config_from_position(camera_position, lookat)


class MjSimulator:
    def __init__(self, param, model_path=None, visualize=True):
        self.param_ = param
        self.model_path_ = model_path or self.param_.model_path_
        print("model_path = ", self.model_path_)

        self.model = mujoco.MjModel.from_xml_path(self.model_path_)
        self.data = mujoco.MjData(self.model)

        self.break_out_signal_ = False
        self.dyn_paused_ = False
        self.viewer_ = None

        self.robot_joint_names_ = [f"joint{i}" for i in range(1, 8)]
        self.robot_joint_ids_ = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.robot_joint_names_
            ],
            dtype=int,
        )
        self.robot_qpos_adr_ = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.robot_joint_ids_],
            dtype=int,
        )
        self.robot_qvel_adr_ = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.robot_joint_ids_],
            dtype=int,
        )

        self.drawer_joint_id_ = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            self.param_.drawer_joint_name_,
        )
        self.drawer_qpos_adr_ = int(self.model.jnt_qposadr[self.drawer_joint_id_])
        self.drawer_qvel_adr_ = int(self.model.jnt_dofadr[self.drawer_joint_id_])
        self.drawer_joint_frictionloss_ = float(
            getattr(
                self.param_,
                "drawer_joint_frictionloss_",
                DEFAULT_DRAWER_JOINT_FRICTIONLOSS,
            )
        )
        self.model.dof_frictionloss[
            self.drawer_qvel_adr_
        ] = self.drawer_joint_frictionloss_

        self.drawer_body_id_ = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.param_.drawer_body_name_,
        )
        self.robot_base_body_id_ = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link0"
        )
        self.fingertip_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip"
        )
        self.fingertip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "attachment"
        )

        self.torque_limits = np.array([87, 87, 87, 87, 12, 12, 12], dtype=np.float64)
        self.home_q = np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            dtype=np.float64,
        )
        self.low_height = self.param_.table_height + 0.02

        self.cartesian_stiffness = 2 * np.diag([500, 500, 500, 50, 50, 50])
        self.cartesian_damping = 2 * np.sqrt(self.cartesian_stiffness)
        self.nullspace_stiffness = 10.0
        self.activate_tool_compensation = False
        self.tool_compensation_force = np.zeros(6)

        self.reset_mj_env()
        self.set_goal(self.param_.target_p_, self.param_.target_q_)

        if visualize:
            self.viewer_ = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self.keyboardCallback,
            )
            camera_config = build_default_drawer_scene_camera_config(self)
            self.viewer_.cam.lookat[:] = camera_config.lookat
            self.viewer_.cam.distance = camera_config.distance
            self.viewer_.cam.azimuth = camera_config.azimuth_deg
            self.viewer_.cam.elevation = camera_config.elevation_deg
            self.viewer_.sync()

        p, R = self.get_end_effector_pos()
        self.position_d = p.copy()
        self.orientation_d = R.copy()
        self.p_d = p.copy()
        self.R_d = R.copy()
        self.nominal_R_d = R.copy()

    def _set_drawer_offset(self, offset):
        self.data.qpos[self.drawer_qpos_adr_] = float(offset)
        self.data.qvel[self.drawer_qvel_adr_] = 0.0

    def _set_robot_joint_vector(self, joint_pos):
        for qpos_adr, joint_value in zip(self.robot_qpos_adr_, joint_pos):
            self.data.qpos[qpos_adr] = float(joint_value)
        for qvel_adr in self.robot_qvel_adr_:
            self.data.qvel[qvel_adr] = 0.0

    def keyboardCallback(self, keycode):
        try:
            key = chr(keycode)
        except ValueError:
            return
        if key == " ":
            self.dyn_paused_ = not self.dyn_paused_
            print("simulation paused!" if self.dyn_paused_ else "simulation resumed!")
        elif key == "Ā":
            self.break_out_signal_ = True

    def reset_mj_env(self):
        mujoco.mj_resetData(self.model, self.data)
        self._set_drawer_offset(self.param_.drawer_initial_offset_)
        self._set_robot_joint_vector(self.param_.init_robot_qpos_)
        self.data.ctrl[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_object_frame(self):
        mujoco.mj_forward(self.model, self.data)
        drawer_body = self.data.body(self.drawer_body_id_)
        drawer_pos = drawer_body.xpos.copy()
        drawer_rot = drawer_body.xmat.reshape(3, 3).copy()
        object_pos = drawer_pos + drawer_rot @ self.param_.drawer_reference_local_
        return object_pos, drawer_rot

    def get_end_effector_pos(self):
        mujoco.mj_forward(self.model, self.data)
        p = self.data.geom(self.fingertip_geom_id).xpos.copy()
        R = self.data.body(self.fingertip_id).xmat.reshape(3, 3).copy()
        return p, R

    def get_robot_forward_direction(self):
        mujoco.mj_forward(self.model, self.data)
        if self.robot_base_body_id_ < 0:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        base_rot = self.data.body(self.robot_base_body_id_).xmat.reshape(3, 3).copy()
        forward_dir = base_rot[:, 0]
        forward_dir[2] = 0.0
        forward_norm = np.linalg.norm(forward_dir)
        if forward_norm < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return forward_dir / forward_norm

    def set_goal(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            self.model.body("goal").pos = goal_pos
        if goal_quat is not None:
            self.model.body("goal").quat = goal_quat
        mujoco.mj_forward(self.model, self.data)

    def get_nominal_desired_rotation(self):
        return self.nominal_R_d.copy()

    def get_world_x_tilted_rotation(self, tilt_angle_rad):
        world_x_tilt = Rotation.from_rotvec(
            np.array([float(tilt_angle_rad), 0.0, 0.0], dtype=np.float64)
        ).as_matrix()
        return world_x_tilt @ self.nominal_R_d

    def step(self, cmd, target_rot=None):
        p_curr, _ = self.get_end_effector_pos()
        print("p_curr = ", p_curr)
        self.p_d = p_curr + cmd
        # self.p_d = self.p_d + cmd
        # self.p_d[2] = self.p_d[2] if self.p_d[2] > self.low_height else self.low_height
        if target_rot is not None:
            self.R_d = np.asarray(target_rot, dtype=np.float64).reshape(3, 3).copy()
        self.set_desired_pose(self.p_d, self.R_d)
        tau_d = self.compute_cartesian_impedance_control()
        self.set_control_torque(tau_d)

        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        if self.viewer_ is not None:
            self.viewer_.sync()

    def set_control_torque(self, control_torque):
        self.data.ctrl[:] = np.clip(
            control_torque, -self.torque_limits, self.torque_limits
        )

    def get_state(self):
        mujoco.mj_forward(self.model, self.data)
        obj_pos, obj_rot = self.get_object_frame()
        quat_xyzw = Rotation.from_matrix(obj_rot).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float64,
        )
        end_pos = self.get_end_effector_pos()[0].copy()
        return np.hstack([obj_pos, quat_wxyz, end_pos])

    def get_drawer_offset(self):
        mujoco.mj_forward(self.model, self.data)
        return float(self.data.qpos[self.drawer_qpos_adr_])

    def get_current_joint_position(self):
        mujoco.mj_forward(self.model, self.data)
        return np.array(
            [self.data.qpos[qadr] for qadr in self.robot_qpos_adr_],
            dtype=np.float64,
        )

    def get_current_joint_velocity(self):
        mujoco.mj_forward(self.model, self.data)
        return np.array(
            [self.data.qvel[vadr] for vadr in self.robot_qvel_adr_],
            dtype=np.float64,
        )

    def show_target(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            self.model.body("marker").pos = goal_pos
        if goal_quat is not None:
            self.model.body("marker").quat = goal_quat
        mujoco.mj_forward(self.model, self.data)

    def show_contact_point(self, goal_pos=None):
        if goal_pos is not None:
            self.model.body("obj_point").pos = goal_pos
        mujoco.mj_forward(self.model, self.data)

    def show_contact_point1(self, goal_pos=None):
        if goal_pos is not None:
            self.model.body("contact_point1").pos = goal_pos
        mujoco.mj_forward(self.model, self.data)

    def get_jacobian(self):
        mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        point = self.data.geom(self.fingertip_geom_id).xpos.copy()
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.fingertip_id)
        return np.vstack(
            [
                jacp[:, self.robot_qvel_adr_],
                jacr[:, self.robot_qvel_adr_],
            ]
        )

    def compute_cartesian_impedance_control(self):
        q = self.get_current_joint_position()
        dq = self.get_current_joint_velocity()
        jacobian = self.get_jacobian()
        p, R_current = self.get_end_effector_pos()

        error = np.zeros(6)
        error[:3] = p - self.position_d

        R_d = self.orientation_d
        R_error = R_current.T @ R_d
        error_quat = Rotation.from_matrix(R_error).as_quat()
        if error_quat[3] < 0:
            error_quat = -error_quat

        error[3:] = -R_current @ error_quat[:3]
        velocity = jacobian @ dq

        F_ee_des = -self.cartesian_stiffness @ error - self.cartesian_damping @ velocity
        tau_task = jacobian.T @ F_ee_des

        jacobian_pinv = pinv(jacobian.T)
        nullspace_proj = np.eye(7) - jacobian.T @ jacobian_pinv
        tau_nullspace = nullspace_proj @ (
            self.nullspace_stiffness * (self.home_q - q)
            - 2 * np.sqrt(self.nullspace_stiffness) * dq
        )

        if self.activate_tool_compensation:
            tau_tool = jacobian.T @ self.tool_compensation_force
        else:
            tau_tool = np.zeros(7)

        tau_d = tau_task + tau_nullspace + tau_tool
        tau_d = np.clip(tau_d, -self.torque_limits, self.torque_limits)
        return tau_d

    def set_cartesian_stiffness(self, stiffness):
        self.cartesian_stiffness = stiffness
        self.cartesian_damping = 2 * np.sqrt(self.cartesian_stiffness)

    def set_nullspace_stiffness(self, stiffness):
        self.nullspace_stiffness = stiffness

    def set_tool_compensation(self, force, activate=True):
        self.tool_compensation_force = force
        self.activate_tool_compensation = activate

    def set_desired_pose(self, position, orientation):
        self.position_d = position.copy()
        self.orientation_d = orientation.copy()
        # self.show_target(position, Rotation.from_matrix(orientation).as_quat())

    def close(self):
        if self.viewer_ is not None:
            self.viewer_.close()
            self.viewer_ = None


class Contact:
    def __init__(self, param):
        self.param_ = param

    def _is_robot_side(self, body_name, geom_name):
        if geom_name in {"rod", "fingertip"}:
            return True
        if body_name == "attachment":
            return True
        return body_name.startswith("link")

    def _build_object_jacobian(self, point_local):
        J = np.zeros((3, self.param_.n_qvel_), dtype=np.float64)
        J[:, :3] = np.eye(3)
        J[0, 4] = point_local[2]
        J[0, 5] = -point_local[1]
        J[1, 3] = -point_local[2]
        J[1, 5] = point_local[0]
        J[2, 3] = point_local[1]
        J[2, 4] = -point_local[0]
        return J

    def _build_robot_jacobian(self):
        J = np.zeros((3, self.param_.n_qvel_), dtype=np.float64)
        J[:, 6:] = np.eye(3)
        return J

    def detect_once(self, simulator: MjSimulator):
        mujoco.mj_forward(simulator.model, simulator.data)
        mujoco.mj_collision(simulator.model, simulator.data)

        n_con = simulator.data.ncon
        contacts = simulator.data.contact

        con_phi_list = []
        con_frame_list = []
        con_pos_list = []
        con_jac_list = []
        con_jac_env_list = []
        con_phi_env_list = []
        if_contact = False

        obj_pos, obj_rot = simulator.get_object_frame()
        robot_contact_jacobian = self._build_robot_jacobian()

        for i in range(n_con):
            contact_i = contacts[i]

            geom1_name = (
                mujoco.mj_id2name(
                    simulator.model, mujoco.mjtObj.mjOBJ_GEOM, contact_i.geom1
                )
                or ""
            )
            geom2_name = (
                mujoco.mj_id2name(
                    simulator.model, mujoco.mjtObj.mjOBJ_GEOM, contact_i.geom2
                )
                or ""
            )
            body1_id = simulator.model.geom_bodyid[contact_i.geom1]
            body2_id = simulator.model.geom_bodyid[contact_i.geom2]
            body1_name = (
                mujoco.mj_id2name(simulator.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                or ""
            )
            body2_name = (
                mujoco.mj_id2name(simulator.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                or ""
            )

            object_is_first = body1_name == self.param_.drawer_body_name_
            object_is_second = body2_name == self.param_.drawer_body_name_
            if not (object_is_first or object_is_second):
                continue

            if object_is_first:
                other_body_name = body2_name
                other_geom_name = geom2_name
            else:
                other_body_name = body1_name
                other_geom_name = geom1_name

            con_pos = contact_i.pos.copy()
            con_dist = contact_i.dist * 0.5
            con_mu = self.param_.mu_object_

            con_frame = contact_i.frame.reshape((-1, 3)).T
            con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

            con_pos_local = obj_rot.T @ (con_pos - obj_pos)
            object_jacobian = self._build_object_jacobian(con_pos_local)

            con_jacp_obj = con_frame_pmd.T @ object_jacobian
            if self._is_robot_side(other_body_name, other_geom_name):
                con_jacp_other = con_frame_pmd.T @ robot_contact_jacobian
                if_contact = True
            else:
                con_jacp_other = np.zeros((4, self.param_.n_qvel_), dtype=np.float64)

            con_jacp = con_jacp_obj - con_jacp_other
            con_jac = con_jacp[0] + con_mu * con_jacp[1:]

            if not self._is_robot_side(other_body_name, other_geom_name):
                con_pos_list.append(con_pos_local)
                con_jac_env_list.append(con_jac)
                con_phi_env_list.append(con_dist)

            con_phi_list.append(con_dist)
            con_frame_list.append(con_frame)
            con_jac_list.append(con_jac)

        phi_vec, jac_mat = self.reformat(
            dict(
                con_pos_list=con_pos_list,
                con_phi_list=con_phi_list,
                con_frame_list=con_frame_list,
                con_jac_list=con_jac_list,
            )
        )
        _, jac_mat_env = self.reformat(
            dict(
                con_phi_list=con_phi_env_list,
                con_jac_list=con_jac_env_list,
            )
        )

        print("con_phi_list = ", con_phi_list, "con_phi_env_list = ", con_phi_env_list)
        return phi_vec, jac_mat, con_pos_list, jac_mat_env, if_contact

    def reformat(self, contacts=None):
        con_jac_list = contacts["con_jac_list"]
        con_phi_list = contacts["con_phi_list"]

        phi_vec = np.ones((self.param_.max_ncon_ * 4,))
        jac_mat = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_mj_v_))
        for i in range(len(con_phi_list[: self.param_.max_ncon_])):
            phi_vec[4 * i : 4 * i + 4] = con_phi_list[i]
            jac_mat[4 * i : 4 * i + 4] = con_jac_list[i]

        return phi_vec, jac_mat


parser = argparse.ArgumentParser()
parser.add_argument("--obj", type=str, default="study_table_drawer", help="name of obj")
parser.add_argument(
    "--attract_coef", type=float, default=0.5, help="coef of attract function"
)
parser.add_argument(
    "--reject_coef", type=float, default=0.001, help="coef of reject function"
)
parser.add_argument(
    "--contact_coef", type=float, default=0.7, help="coef of contact function"
)
parser.add_argument(
    "--contact_cost_param",
    type=float,
    default=0,
    help="mass center or project point attract",
)
parser.add_argument("--model_param", type=float, default=7, help="model param")
parser.add_argument("--reject_dis", type=float, default=0.005, help="reject radius")
parser.add_argument(
    "--attract_point_comp",
    type=float,
    default=0.085,
    help="distance compensation of attract point",  # 向上偏移量
)
parser.add_argument(
    "--attract_point_forward_comp",
    type=float,
    default=0.05,
    help="forward offset of attract point along the robot-front direction; only used during front-panel sampling",  # 向前偏移量
)
parser.add_argument(
    "--ground_height_threshold",
    type=float,
    default=0.012,
    help="threshold of sample points height",
)
parser.add_argument(
    "--sampling_world_z_upper_fraction",
    type=float,
    default=-1,
    help=(
        "after visibility filtering, keep only the top fraction of candidate sample "
        "points measured by world-frame z on the active stage mesh; set <= 0 to disable"
    ),
)
parser.add_argument(
    "--sample_num", type=int, default=120, help="number of sample point"
)
parser.add_argument(
    "--pos_coef", type=float, default=2, help="coef of position cost in mlqp_point"
)
parser.add_argument(
    "--ori_coef",
    type=float,
    default=0.00,
    help="coef of orientation cost in mlqp_point",
)
parser.add_argument(
    "--mlqp-solver",
    type=str,
    choices=["ipopt", "snopt", "acados"],
    default="acados",
    help=(
        "Solver backend used inside planning/mlqp_point_cabinet.py; "
        "choose from the CasADi IPOPT/SNOPT backends or the acados_template backend."
    ),
)
parser.add_argument(
    "--low_err_coef", type=float, default=0.1, help="coef of delta error"
)
parser.add_argument(
    "--upper_err_coef", type=float, default=0.6, help="coef of delta error"
)
parser.add_argument(
    "--handle_low_err_coef",
    type=float,
    default=0.1,
    help="low error coef used during handle sampling; falls back to --low_err_coef",
)
parser.add_argument(
    "--handle_upper_err_coef",
    type=float,
    default=0.6,
    help="upper error coef used during handle sampling; falls back to --upper_err_coef",
)
parser.add_argument(
    "--front_panel_low_err_coef",
    type=float,
    default=0.1,
    help="low error coef used during front-panel sampling; falls back to --low_err_coef",
)
parser.add_argument(
    "--front_panel_upper_err_coef",
    type=float,
    default=0.55,
    help="upper error coef used during front-panel sampling; falls back to --upper_err_coef",
)
parser.add_argument(
    "--drawer_open_distance",
    type=float,
    default=-0.20,
    help="target drawer slide displacement in meters",
)
parser.add_argument(
    "--drawer_joint_frictionloss",
    type=float,
    default=DEFAULT_DRAWER_JOINT_FRICTIONLOSS,
    help="MuJoCo frictionloss applied to the main drawer slide joint after loading the XML",
)
parser.add_argument(
    "--handle_only_open_distance",
    type=float,
    default=None,
    help="optional override for handle-only sampling distance; defaults to half of the target drawer opening distance",
)
parser.add_argument(
    "--pose_noise_pos",
    type=float,
    default=0.002,
    help="position noise std for observed drawer pose",
)
parser.add_argument(
    "--pose_noise_rot",
    type=float,
    default=0.02,
    help="rotation noise std for observed drawer pose",
)
parser.add_argument(
    "--virtual_track_world_x_tilt_deg",
    type=float,
    default=12.0,
    help="when verify_cost=0, tilt the desired end-effector orientation by this many degrees around world +x",
)
parser.add_argument(
    "--screenshot_interval_sec",
    type=float,
    default=0.1,
    help="save one SVG screenshot every N seconds of simulation time; set <= 0 to disable",
)
parser.add_argument(
    "--screenshot_width",
    type=int,
    default=1280,
    help="SVG screenshot width in pixels",
)
parser.add_argument(
    "--screenshot_height",
    type=int,
    default=960,
    help="SVG screenshot height in pixels",
)
parser.add_argument(
    "--screenshot_camera_x",
    type=float,
    default=None,
    help="optional screenshot camera world x position",
)
parser.add_argument(
    "--screenshot_camera_y",
    type=float,
    default=None,
    help="optional screenshot camera world y position",
)
parser.add_argument(
    "--screenshot_camera_z",
    type=float,
    default=None,
    help="optional screenshot camera world z position",
)
parser.add_argument(
    "--screenshot_lookat_x",
    type=float,
    default=None,
    help="optional screenshot camera lookat world x position",
)
parser.add_argument(
    "--screenshot_lookat_y",
    type=float,
    default=None,
    help="optional screenshot camera lookat world y position",
)
parser.add_argument(
    "--screenshot_lookat_z",
    type=float,
    default=None,
    help="optional screenshot camera lookat world z position",
)
parser.add_argument(
    "--screenshot_pitch_deg",
    type=float,
    default=None,
    help="optional screenshot camera pitch angle in degrees",
)
parser.add_argument(
    "--video-output-dir",
    type=str,
    default="/home/lab423/scsp/Franka-contact-face-detection-manipulation-main/outputs/videos_drawer",
    help=(
        "Directory for per-trial MP4 recordings. "
        "When set, each trial saves video.mp4 under a unique trial subdirectory."
    ),
)
parser.add_argument(
    "--video-fps",
    type=float,
    default=20.0,
    help="MP4 recording frame rate",
)
parser.add_argument(
    "--video-width",
    type=int,
    default=1280,
    help="MP4 recording width in pixels",
)
parser.add_argument(
    "--video-height",
    type=int,
    default=960,
    help="MP4 recording height in pixels",
)

args = parser.parse_args()


def build_stage_error_coef_config(args):
    handle_low_err_coef = float(
        args.handle_low_err_coef
        if args.handle_low_err_coef is not None
        else args.low_err_coef
    )
    handle_upper_err_coef = float(
        args.handle_upper_err_coef
        if args.handle_upper_err_coef is not None
        else args.upper_err_coef
    )
    stage_base_low_err_coef = {
        DRAWER_SAMPLING_HANDLE_NAME: handle_low_err_coef,
        DRAWER_SAMPLING_FRONT_PANEL_NAME: float(
            args.front_panel_low_err_coef
            if args.front_panel_low_err_coef is not None
            else args.low_err_coef
        ),
    }
    stage_base_upper_err_coef = {
        DRAWER_SAMPLING_HANDLE_NAME: handle_upper_err_coef,
        DRAWER_SAMPLING_FRONT_PANEL_NAME: float(
            args.front_panel_upper_err_coef
            if args.front_panel_upper_err_coef is not None
            else args.upper_err_coef
        ),
    }
    stage_base_low_err_coef["default"] = handle_low_err_coef
    stage_base_upper_err_coef["default"] = handle_upper_err_coef
    return stage_base_low_err_coef, stage_base_upper_err_coef


def get_stage_err_coefs(sampling_stage_name, stage_low_err_coef, stage_upper_err_coef):
    low_err_coef = stage_low_err_coef.get(
        sampling_stage_name, stage_low_err_coef["default"]
    )
    upper_err_coef = stage_upper_err_coef.get(
        sampling_stage_name, stage_upper_err_coef["default"]
    )
    return float(low_err_coef), float(upper_err_coef)


def _get_optional_vec3(args, attr_names):
    values = [getattr(args, attr_name) for attr_name in attr_names]
    provided = [value is not None for value in values]
    if any(provided) and not all(provided):
        raise ValueError(f"Expected either all or none of {attr_names}, got {values}")
    if not all(provided):
        return None
    return np.array(values, dtype=np.float64)


def build_screenshot_camera_config(env, args):
    default_camera_config = build_default_drawer_scene_camera_config(env)
    lookat = _get_optional_vec3(
        args,
        ("screenshot_lookat_x", "screenshot_lookat_y", "screenshot_lookat_z"),
    )
    if lookat is None:
        lookat = default_camera_config.lookat.copy()

    camera_position = _get_optional_vec3(
        args,
        ("screenshot_camera_x", "screenshot_camera_y", "screenshot_camera_z"),
    )
    if camera_position is not None:
        return build_free_camera_config_from_position(
            camera_position,
            lookat,
            pitch_deg=args.screenshot_pitch_deg,
        )

    distance = float(default_camera_config.distance)
    azimuth_deg = float(default_camera_config.azimuth_deg)
    elevation_deg = float(default_camera_config.elevation_deg)
    if args.screenshot_pitch_deg is not None:
        elevation_deg = float(args.screenshot_pitch_deg)

    return build_free_camera_config(
        lookat=lookat,
        distance=distance,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
    )


def _allocate_trial_record_dir(base_output_dir, trial_id):
    if not base_output_dir:
        return None

    base_output_dir = os.path.abspath(base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)

    base_name = f"trial_{int(trial_id):03d}"
    candidate_dir = os.path.join(base_output_dir, base_name)
    if not os.path.exists(candidate_dir):
        os.makedirs(candidate_dir, exist_ok=False)
        return candidate_dir

    suffix = 1
    while True:
        candidate_dir = os.path.join(base_output_dir, f"{base_name}_{suffix:03d}")
        if not os.path.exists(candidate_dir):
            os.makedirs(candidate_dir, exist_ok=False)
            return candidate_dir
        suffix += 1


# -------------------------------
#       loop trials
# -------------------------------
save_flag = False
if save_flag:
    save_dir = "./examples/mpc/franka/oepn_drawer/save/"
    prefix_data_name = "ours_"
    save_data = dict()

trial_num = 20
success_pos_threshold = 0.01
success_quat_threshold = 0.08
consecutive_success_time_threshold = 20
max_rollout_length = 10000
success_rate = 0

trial_count = 0
while trial_count < trial_num:
    env = None
    screenshot_recorder = None
    video_recorder = None
    saved_video_path = None
    interrupted_trial = False
    trial_success = False
    rollout_step = max_rollout_length
    rollout_q_traj = []
    param = None

    try:
        # -------------------------------
        #        init parameters
        # -------------------------------
        param = ExplicitMPCParams(
            args,
            rand_seed=trial_count,
            target_type="translation",
            mpc_model="explicit",
        )
        param.drawer_joint_frictionloss_ = float(args.drawer_joint_frictionloss)
        print(
            f"target_drawer_offset = {param.drawer_target_offset_:.3f}, "
            f"target_p = {np.round(param.target_p_, 4)}, "
            f"target_q = {np.round(param.target_q_, 4)}"
        )

        # -------------------------------
        #        init contact
        # -------------------------------
        contact = Contact(param)

        # -------------------------------
        #        init envs
        # -------------------------------
        env = MjSimulator(param)
        screenshot_camera_config = build_screenshot_camera_config(env, args)

        if args.screenshot_interval_sec > 0.0:
            screenshot_dir = os.path.join(
                current_dir,
                "results",
                f"trial_{trial_count:03d}",
            )
            screenshot_recorder = PeriodicSVGScreenshotRecorder(
                env.model,
                output_dir=screenshot_dir,
                camera_config=screenshot_camera_config,
                interval_seconds=args.screenshot_interval_sec,
                width=args.screenshot_width,
                height=args.screenshot_height,
                filename_prefix="drawer",
            )
            screenshot_recorder.capture_if_due(env.data)
            print(f"SVG screenshots will be saved to {screenshot_dir}")

        record_dir = _allocate_trial_record_dir(args.video_output_dir, trial_count)
        if record_dir is not None:
            video_output_path = os.path.join(record_dir, "video.mp4")
            try:
                video_recorder = create_mujoco_mp4_recorder(
                    env.model,
                    output_path=video_output_path,
                    camera_config=screenshot_camera_config,
                    fps=args.video_fps,
                    width=args.video_width,
                    height=args.video_height,
                    capture_on_start=True,
                    wall_clock_timing=True,
                )
                video_recorder.capture_if_due(env.data)
                print(
                    "MP4 capture enabled: "
                    f"path={video_output_path}, fps={float(args.video_fps):.2f}, "
                    f"size={int(args.video_width)}x{int(args.video_height)}"
                )
            except RuntimeError as exc:
                video_recorder = None
                print(f"MP4 capture disabled: {exc}")

        # -------------------------------
        #        init planner
        # -------------------------------
        mpc = (
            MPCExplicit(param) if param.mpc_model == "explicit" else MPCImplicit(param)
        )

        # -------------------------------
        #        MPC rollout
        # -------------------------------
        rollout_step = 0
        consecutive_success_time = 0
        consecutive_detect_time = 0
        consecutive_contact_time = 0
        verify_cost = 0
        current_x = np.zeros(7)
        current_x[3] = 1

        (
            stage_base_low_err_coef,
            stage_base_upper_err_coef,
        ) = build_stage_error_coef_config(args)
        stage_upper_err_coef = dict(stage_base_upper_err_coef)

        previous_sampling_stage_name = None
        while rollout_step < max_rollout_length:
            if env.break_out_signal_:
                interrupted_trial = True
                print(
                    f"[trial {trial_count:03d}] break signal received, finalizing current recording."
                )
                break

            if not env.dyn_paused_:
                curr_q_real = env.get_state()

                noise_std = np.array([1e-3, 1e-3, 1e-3])
                noise = np.random.randn(*curr_q_real[7:10].shape) * noise_std
                curr_q = curr_q_real.copy()

                curr_q[7:10] = curr_q_real[7:10] + noise
                obj_pos, obj_quat = add_se3_noise(
                    curr_q_real[:3],
                    curr_q_real[3:7],
                    args.pose_noise_pos,
                    args.pose_noise_rot,
                )
                robot_pos = curr_q[7:10]

                rollout_q_traj.append(curr_q)

                # -----------------------
                #     contact detect
                # -----------------------
                (
                    phi_vec,
                    jac_mat,
                    con_point,
                    jac_mat_env,
                    if_contact,
                ) = contact.detect_once(env)
                quanternion = [curr_q[4], curr_q[5], curr_q[6], curr_q[3]]
                R_obj_to_world = Rotation.from_quat(quanternion).as_matrix()
                gravity = np.hstack(
                    [
                        R_obj_to_world.T @ param.gravity_[:3] * param.obj_mass_,
                        np.zeros(3),
                    ]
                )

                target_pos_ = param.target_p_ - obj_pos
                target_pos_[2] = 0

                target_quat_local = rotations.quaternion_multiply(
                    rotations.quaternion_conjugate(obj_quat),
                    param.target_q_,
                )
                target_pose_local = np.hstack(
                    [R_obj_to_world.T @ target_pos_, target_quat_local]
                )

                drawer_offset = env.get_drawer_offset()
                sampling_stage_name = param.get_sampling_stage_name(drawer_offset)
                if (
                    screenshot_recorder is not None
                    and previous_sampling_stage_name != sampling_stage_name
                    and sampling_stage_name == DRAWER_SAMPLING_FRONT_PANEL_NAME
                ):
                    screenshot_recorder.capture(
                        env.data,
                        label="front_panel_start",
                    )
                active_lambda_optimizer = param.get_active_lambda_optimizer(
                    drawer_offset
                )
                low_err_coef, upper_err_coef = get_stage_err_coefs(
                    sampling_stage_name,
                    stage_base_low_err_coef,
                    stage_upper_err_coef,
                )
                previous_sampling_stage_name = sampling_stage_name

                active_lambda_optimizer.update_Jacobian(jac_mat_env)
                st0 = time.time()
                visible_point_idx = active_lambda_optimizer.get_availble_point_idx(
                    obj_pos,
                    R_obj_to_world,
                    param.target_p_,
                    args.ground_height_threshold,
                )
                visible_point_idx = _filter_point_indices_to_upper_world_z_region(
                    active_lambda_optimizer,
                    obj_pos,
                    R_obj_to_world,
                    visible_point_idx,
                    upper_fraction=args.sampling_world_z_upper_fraction,
                )

                exchange_cost_time = time.time() - st0
                update_timing_stats("exchange cost time", exchange_cost_time)
                st1 = time.time()
                (
                    best_contact_point,
                    normal,
                    min_error,
                    max_error,
                    curr_ori_coef,
                ) = active_lambda_optimizer.choose_contact_points(
                    target_pose_local, current_x, gravity, visible_point_idx
                )
                cso_solving_time = (time.time() - st1) / 70
                update_timing_stats("cso solving time", cso_solving_time)
                print("best_contact_point = ", best_contact_point)
                attract_point = best_contact_point.copy()
                attract_point_world = R_obj_to_world @ attract_point + obj_pos
                best_contact_point_world = R_obj_to_world @ attract_point + obj_pos
                attract_point_world = best_contact_point_world.copy()
                attract_point_world[2] += args.attract_point_comp
                if sampling_stage_name == DRAWER_SAMPLING_FRONT_PANEL_NAME:
                    robot_forward_world = env.get_robot_forward_direction()
                    attract_point_world += (
                        args.attract_point_forward_comp * robot_forward_world
                    )

                local_point = R_obj_to_world.T @ (robot_pos - obj_pos)
                (
                    p_arm_local,
                    _,
                    _,
                    error,
                    _,
                ) = active_lambda_optimizer.optimize_control_input(
                    target_pose_local, current_x, gravity, local_point
                )
                p_arm_world = R_obj_to_world @ p_arm_local + obj_pos

                st2 = time.time()
                eps = 1e-6
                delta_error = max_error - min_error
                rel_impr = max_error - error
                if delta_error <= eps:
                    improvement = 0.0
                else:
                    improvement = rel_impr / delta_error
                rs_cost_time = time.time() - st2
                update_timing_stats("rs cost time", rs_cost_time)

                distance_to_attract_point = np.linalg.norm(
                    robot_pos - attract_point_world
                )
                if sampling_stage_name == DRAWER_SAMPLING_FRONT_PANEL_NAME:
                    verify_cost = int(distance_to_attract_point < 5e-2)
                    consecutive_detect_time = 0
                    consecutive_contact_time = 0
                    upper_err_coef = stage_base_upper_err_coef.get(
                        sampling_stage_name, stage_base_upper_err_coef["default"]
                    )
                else:
                    if verify_cost:
                        upper_err_coef = stage_base_upper_err_coef.get(
                            sampling_stage_name, stage_base_upper_err_coef["default"]
                        )
                    elif distance_to_attract_point < 5e-2:
                        upper_err_coef *= 0.9
                    stage_upper_err_coef[sampling_stage_name] = float(upper_err_coef)

                    consecutive_contact_time = (
                        consecutive_contact_time + int(if_contact) if verify_cost else 0
                    )
                    if (not verify_cost and improvement > upper_err_coef) or (
                        verify_cost and improvement <= low_err_coef
                    ):
                        consecutive_detect_time += 1
                    else:
                        consecutive_detect_time = 0

                    if not verify_cost and consecutive_detect_time >= 2:
                        verify_cost = 1
                    elif (
                        verify_cost
                        and consecutive_detect_time >= 10
                        and consecutive_contact_time >= 15
                    ):
                        verify_cost = 0

                print(
                    "min error:",
                    min_error,
                    "max error",
                    max_error,
                    "actual error:",
                    error,
                    "sampling_stage:",
                    sampling_stage_name,
                    "low_coef:",
                    low_err_coef,
                    "upper_coef:",
                    upper_err_coef,
                    "distance_to_attract:",
                    distance_to_attract_point,
                )
                print("verify cost:", verify_cost)
                env.show_target(best_contact_point_world)

                # -----------------------
                #        planning
                # -----------------------
                print("p_arm_world = ", p_arm_world)
                st3 = time.time()
                sol = mpc.plan_once(
                    param.target_p_,
                    param.target_q_,
                    curr_q,
                    phi_vec,
                    jac_mat,
                    verify_cost_param=verify_cost,
                    virtual_point=attract_point_world,
                    contact_point=best_contact_point_world,
                    curr_ori_coef=curr_ori_coef,
                    sol_guess=param.sol_guess_,
                )

                param.sol_guess_ = sol["sol_guess"]
                action = sol["action"]

                # -----------------------
                #        simulate
                # -----------------------
                if verify_cost == 0:
                    step_target_rot = env.get_world_x_tilted_rotation(
                        np.deg2rad(args.virtual_track_world_x_tilt_deg)
                    )
                else:
                    step_target_rot = env.get_nominal_desired_rotation()
                env.step(action, target_rot=step_target_rot)
                if screenshot_recorder is not None:
                    screenshot_recorder.capture_if_due(env.data)
                if video_recorder is not None:
                    video_recorder.capture_if_due(env.data)
                st_cpo = time.time()
                cito_cost_time = time.time() - st3
                cpo_cost_time = st_cpo - st0
                update_timing_stats("cito cost time", cito_cost_time)
                update_timing_stats("cpo cost time", cpo_cost_time)
                rollout_step = rollout_step + 1

                # -----------------------
                #        success check
                # -----------------------
                curr_q_after = env.get_state()
                if (
                    metrics.comp_pos_error(curr_q_after[:3], param.target_p_)
                    < success_pos_threshold
                    and metrics.comp_quat_error(curr_q_after[3:7], param.target_q_)
                    < success_quat_threshold
                ):
                    consecutive_success_time = consecutive_success_time + 1
                else:
                    consecutive_success_time = 0

                # -----------------------
                #       early termination
                # -----------------------
                if consecutive_success_time > consecutive_success_time_threshold:
                    trial_success = True
                    break

        if trial_success:
            print(f"[trial {trial_count:03d}] success, finalizing MP4 recording.")
        elif interrupted_trial:
            print(f"[trial {trial_count:03d}] interrupted, finalizing MP4 recording.")
    except KeyboardInterrupt:
        interrupted_trial = True
        print(
            f"\n[trial {trial_count:03d}] Ctrl+C received. Finalizing current MP4 recording..."
        )
        if env is not None:
            env.break_out_signal_ = True
    finally:
        if video_recorder is not None:
            try:
                saved_video_path = video_recorder.close()
            except Exception as exc:
                print(
                    f"[trial {trial_count:03d}] warning: failed to finalize MP4: {exc}"
                )
                saved_video_path = None
        if screenshot_recorder is not None:
            screenshot_recorder.close()
        if env is not None:
            env.close()

    if saved_video_path is not None:
        print(f"[trial {trial_count:03d}] saved mp4: {saved_video_path}")

    # -------------------------------
    #        save data
    # -------------------------------
    if save_flag and param is not None:
        save_data.update(target_obj_pos=param.target_p_)
        save_data.update(target_obj_quat=param.target_q_)
        save_data.update(rollout_traj=np.array(rollout_q_traj))
        save_data.update(success=bool(trial_success))
        metrics.save_data(
            save_data,
            data_name=prefix_data_name + "trial_" + str(trial_count) + "_rollout",
            save_dir=save_dir,
        )

    success_rate = success_rate + (1 if trial_success else 0)
    trial_count = trial_count + 1
    if interrupted_trial:
        break

print(
    f"Success rate over {trial_num} trials: {success_rate}/{trial_num} = {success_rate / trial_num:.2%}"
)
