import numpy as np
import casadi as cs
import trimesh

from pathlib import Path
from scipy.spatial.transform import Rotation
from xml.etree import ElementTree as ET

from robocasa.demos.open_drawer.mlqp_point_cabinet import LambdaContactControlOptimizer


REPO_ROOT = Path(__file__).resolve().parents[4]
STUDYTABLE_MESH_DIR = REPO_ROOT / "furniture_sim" / "studyTable" / "meshes"
ENV_XML_PATH = REPO_ROOT / "envs" / "xmls" / "env_study_table_drawer.xml"

STUDYTABLE_SCALE = np.float64(0.8)
STUDYTABLE_WORLD_POS = np.array([0.90, 0.0, 0.0], dtype=np.float64)
STUDYTABLE_WORLD_YAW = -np.pi / 2.0
STUDYTABLE_BODY_NAME = "obj"

DRAWER_BODY_NAME = "studyTable_Drawer"
DRAWER_JOINT_NAME = "studyTable_Drawer_Joint"
DRAWER_SAMPLING_HANDLE_NAME = "studyTable_Drawer_Handle"
DRAWER_SAMPLING_FRONT_PANEL_NAME = "studyTable_Drawer_FrontPanel"
DRAWER_BODY_LOCAL_POS = STUDYTABLE_SCALE * np.array(
    [0.0, 0.0, 0.655],
    dtype=np.float64,
)
DRAWER_JOINT_AXIS_LOCAL = np.array([0.0, 1.0, 0.0], dtype=np.float64)
DRAWER_MESH_SCALE = np.full(3, 0.01 * STUDYTABLE_SCALE, dtype=np.float64)
DRAWER_FRONT_PANEL_HALF_SIZE = np.array([0.192, 0.008, 0.04], dtype=np.float64)
DRAWER_FRONT_PANEL_POS_LOCAL = np.array([0.0, -0.2, 0.0], dtype=np.float64)
DRAWER_FRONT_PANEL_SUBDIVISIONS = 4
DRAWER_FRONT_PANEL_SAMPLING_INSET = np.array([0.024, 0.0, 0.008], dtype=np.float64)

# This matches the studyTable visual mesh orientation in MuJoCo for contact sampling.
DRAWER_VISUAL_ROTATION_LOCAL = Rotation.from_euler(
    "xzy",
    [-1.57, 0.0, 3.14],
).as_matrix()


def quat_xyzw_to_wxyz(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def quat_wxyz_to_xyzw(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    return np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float64,
    )


def load_studytable_pose_from_xml(xml_path):
    xml_path = Path(xml_path)
    default_pos = STUDYTABLE_WORLD_POS.copy()
    default_yaw = float(STUDYTABLE_WORLD_YAW)

    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return default_pos, default_yaw

    obj_body = root.find(".//body[@name='obj']")
    if obj_body is None:
        return default_pos, default_yaw

    pos_attr = obj_body.get("pos")
    if pos_attr:
        obj_pos = np.fromstring(pos_attr, sep=" ", dtype=np.float64)
        if obj_pos.shape != (3,):
            obj_pos = default_pos
    else:
        obj_pos = default_pos

    if obj_body.get("euler"):
        euler = np.fromstring(obj_body.get("euler"), sep=" ", dtype=np.float64)
        if euler.shape == (3,):
            yaw = float(euler[2])
        else:
            yaw = default_yaw
    elif obj_body.get("quat"):
        quat_wxyz = np.fromstring(obj_body.get("quat"), sep=" ", dtype=np.float64)
        if quat_wxyz.shape == (4,):
            yaw = float(
                Rotation.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).as_euler("xyz")[2]
            )
        else:
            yaw = default_yaw
    else:
        yaw = default_yaw

    return obj_pos, yaw


def _transform_drawer_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_scale(DRAWER_MESH_SCALE)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = DRAWER_VISUAL_ROTATION_LOCAL
    mesh.apply_transform(transform)
    return mesh


def _load_transformed_drawer_visual_meshes():
    drawer_wood_mesh = _transform_drawer_mesh(
        trimesh.load_mesh(STUDYTABLE_MESH_DIR / "studyTable_Drawer_Wood.stl")
    )
    drawer_handle_mesh = _transform_drawer_mesh(
        trimesh.load_mesh(STUDYTABLE_MESH_DIR / "studyTable_Drawer_Handle.stl")
    )
    return drawer_wood_mesh, drawer_handle_mesh


def _build_rect_surface_mesh(half_extent_x, half_extent_z, center, subdivisions):
    half_extent_x = float(half_extent_x)
    half_extent_z = float(half_extent_z)
    center = np.asarray(center, dtype=np.float64)

    vertices = np.array(
        [
            [center[0] - half_extent_x, center[1], center[2] - half_extent_z],
            [center[0] + half_extent_x, center[1], center[2] - half_extent_z],
            [center[0] + half_extent_x, center[1], center[2] + half_extent_z],
            [center[0] - half_extent_x, center[1], center[2] + half_extent_z],
        ],
        dtype=np.float64,
    )
    # Triangle winding gives a -Y normal, which matches the outward front face.
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    rect_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    for _ in range(int(subdivisions)):
        rect_mesh = rect_mesh.subdivide()
    return rect_mesh


def _build_drawer_front_panel_sampling_mesh() -> trimesh.Trimesh:
    sampling_half_size = (
        DRAWER_FRONT_PANEL_HALF_SIZE - DRAWER_FRONT_PANEL_SAMPLING_INSET
    )
    if np.any(sampling_half_size[[0, 2]] <= 0.0):
        raise ValueError("Front panel sampling inset is too large for the panel size.")

    front_face_center = DRAWER_FRONT_PANEL_POS_LOCAL.copy()
    front_face_center[1] -= DRAWER_FRONT_PANEL_HALF_SIZE[1]
    front_panel_mesh = _build_rect_surface_mesh(
        sampling_half_size[0],
        sampling_half_size[2],
        front_face_center,
        DRAWER_FRONT_PANEL_SUBDIVISIONS,
    )
    return front_panel_mesh


def build_drawer_reference_local():
    drawer_wood_mesh, drawer_handle_mesh = _load_transformed_drawer_visual_meshes()
    reference_mesh = trimesh.util.concatenate([drawer_wood_mesh, drawer_handle_mesh])
    return np.asarray(reference_mesh.centroid, dtype=np.float64)


def build_drawer_stage_sampling_mesh_paths(reference_local):
    reference_local = np.asarray(reference_local, dtype=np.float64)

    _, drawer_handle_mesh = _load_transformed_drawer_visual_meshes()
    handle_sampling_mesh = drawer_handle_mesh.copy()
    handle_sampling_mesh.apply_translation(-reference_local)

    front_panel_sampling_mesh = _build_drawer_front_panel_sampling_mesh()
    front_panel_sampling_mesh.apply_translation(-reference_local)

    return {
        DRAWER_SAMPLING_HANDLE_NAME: _export_sampling_mesh(
            handle_sampling_mesh,
            Path("/tmp") / "studytable_drawer_handle_only_sampling.stl",
        ),
        DRAWER_SAMPLING_FRONT_PANEL_NAME: _export_sampling_mesh(
            front_panel_sampling_mesh,
            Path("/tmp") / "studytable_drawer_front_panel_only_sampling.stl",
        ),
    }


def _build_drawer_front_panel_mesh() -> trimesh.Trimesh:
    # Backward-compatible alias for the dedicated front-panel sampling mesh.
    return _build_drawer_front_panel_sampling_mesh()


def _export_sampling_mesh(mesh: trimesh.Trimesh, export_path) -> str:
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(export_path)
    return str(export_path)


def build_drawer_sampling_mesh(export_path=None):
    reference_local = build_drawer_reference_local()
    _, drawer_handle_mesh = _load_transformed_drawer_visual_meshes()
    front_panel_mesh = _build_drawer_front_panel_sampling_mesh()
    sampling_mesh = trimesh.util.concatenate([drawer_handle_mesh, front_panel_mesh])
    sampling_mesh.apply_translation(-reference_local)

    if export_path is None:
        export_path = Path("/tmp") / "studytable_drawer_front_panel_handle_combined.stl"
    export_path = _export_sampling_mesh(sampling_mesh, export_path)

    return export_path, reference_local.astype(np.float32)


def build_drawer_sampling_stl_specs(reference_local):
    reference_local = np.asarray(reference_local, dtype=np.float64)
    translation = (-reference_local).astype(np.float64)

    handle_spec = {
        "scale_factors": DRAWER_MESH_SCALE,
        "rotation": DRAWER_VISUAL_ROTATION_LOCAL,
        "translation": translation,
    }
    front_panel_export_path = _export_sampling_mesh(
        _build_drawer_front_panel_mesh(),
        Path("/tmp") / "studytable_drawer_front_panel_sampling.stl",
    )
    return {
        DRAWER_SAMPLING_HANDLE_NAME: {
            "mesh_path": str(STUDYTABLE_MESH_DIR / "studyTable_Drawer_Handle.stl"),
            **handle_spec,
        },
        DRAWER_SAMPLING_FRONT_PANEL_NAME: {
            "mesh_path": front_panel_export_path,
            "translation": translation,
        },
    }


def compute_drawer_reference_pose(
    studytable_pos,
    studytable_yaw,
    drawer_offset,
    reference_local,
    drawer_body_local_pos=DRAWER_BODY_LOCAL_POS,
    drawer_joint_axis_local=DRAWER_JOINT_AXIS_LOCAL,
):
    studytable_pos = np.asarray(studytable_pos, dtype=np.float64)
    reference_local = np.asarray(reference_local, dtype=np.float64)
    drawer_body_local_pos = np.asarray(drawer_body_local_pos, dtype=np.float64)
    drawer_joint_axis_local = np.asarray(drawer_joint_axis_local, dtype=np.float64)

    studytable_rot = Rotation.from_euler("z", float(studytable_yaw)).as_matrix()
    drawer_rot_world = studytable_rot

    drawer_origin_world = studytable_pos + studytable_rot @ (
        drawer_body_local_pos + float(drawer_offset) * drawer_joint_axis_local
    )
    reference_world = drawer_origin_world + drawer_rot_world @ reference_local
    reference_quat_wxyz = quat_xyzw_to_wxyz(
        Rotation.from_matrix(drawer_rot_world).as_quat()
    ).astype(np.float32)

    return reference_world.astype(np.float32), reference_quat_wxyz


class ExplicitMPCParams:
    def __init__(
        self,
        args,
        rand_seed=1,
        target_type="translation",
        mpc_model="explicit",
    ):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters
        # ---------------------------------------------------------------------------------------------
        self.contact_cost_param = args.contact_cost_param
        self.attract_coef = args.attract_coef
        self.reject_coef = args.reject_coef
        self.contact_coef = args.contact_coef
        self.reject_dis = args.reject_dis

        self.repo_root_ = str(REPO_ROOT)
        self.model_path_ = "./envs/xmls/env_study_table_drawer.xml"
        studytable_world_pos, studytable_world_yaw = load_studytable_pose_from_xml(
            ENV_XML_PATH
        )
        self.drawer_reference_local_ = build_drawer_reference_local().astype(np.float32)
        self.stage_sampling_mesh_paths_ = build_drawer_stage_sampling_mesh_paths(
            self.drawer_reference_local_
        )
        self.handle_mesh_path_ = self.stage_sampling_mesh_paths_[
            DRAWER_SAMPLING_HANDLE_NAME
        ]
        self.front_panel_mesh_path_ = self.stage_sampling_mesh_paths_[
            DRAWER_SAMPLING_FRONT_PANEL_NAME
        ]
        self.mesh_path_ = self.handle_mesh_path_
        self.object_names_ = ["drawer_wood", "drawer_handle"]
        self.object_body_names_ = [DRAWER_BODY_NAME]

        self.studytable_body_name_ = STUDYTABLE_BODY_NAME
        self.drawer_body_name_ = DRAWER_BODY_NAME
        self.drawer_joint_name_ = DRAWER_JOINT_NAME
        self.studytable_pos_ = np.asarray(studytable_world_pos, dtype=np.float32)
        self.studytable_yaw_ = np.float32(studytable_world_yaw)
        self.drawer_body_local_pos_ = DRAWER_BODY_LOCAL_POS.astype(np.float32)
        self.drawer_joint_axis_local_ = DRAWER_JOINT_AXIS_LOCAL.astype(np.float32)
        self.drawer_initial_offset_ = 0.0
        self.drawer_target_offset_ = float(
            getattr(args, "drawer_open_distance", -0.25 * STUDYTABLE_SCALE)
        )
        target_open_distance = abs(
            self.drawer_target_offset_ - self.drawer_initial_offset_
        )
        requested_switch_distance = getattr(args, "handle_only_open_distance", None)
        if requested_switch_distance is None:
            self.handle_only_open_distance_ = 0.5 * target_open_distance
        else:
            self.handle_only_open_distance_ = float(requested_switch_distance)
        self.sampling_stl_specs_ = build_drawer_sampling_stl_specs(
            self.drawer_reference_local_
        )

        self.h_ = 0.05
        self.frame_skip_ = int(10)

        # system dimensions:
        self.n_robot_qpos_ = 3
        self.n_qpos_ = 10
        self.n_qvel_ = 9
        self.n_cmd_ = 3

        self.jc_kp_ = 200
        self.jc_damping_ = 10
        self.proximity_threshold_ = 0.1
        self.fingertip_geoms = ["rod", "fingertip"]

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(100 + rand_seed)

        self.table_height = 0.40
        self.init_robot_qpos_ = np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            dtype=np.float32,
        )

        init_pos, init_quat = compute_drawer_reference_pose(
            self.studytable_pos_,
            self.studytable_yaw_,
            self.drawer_initial_offset_,
            self.drawer_reference_local_,
            self.drawer_body_local_pos_,
            self.drawer_joint_axis_local_,
        )
        self.init_obj_qpos_ = np.hstack((init_pos, init_quat)).astype(np.float32)

        if target_type == "translation":
            target_pos, target_quat = compute_drawer_reference_pose(
                self.studytable_pos_,
                self.studytable_yaw_,
                self.drawer_target_offset_,
                self.drawer_reference_local_,
                self.drawer_body_local_pos_,
                self.drawer_joint_axis_local_,
            )
            self.target_p_ = target_pos.astype(np.float32)
            self.target_q_ = target_quat.astype(np.float32)
        else:
            raise ValueError(f"Target type {target_type} not supported")

        # ---------------------------------------------------------------------------------------------
        #      contact parameters
        # ---------------------------------------------------------------------------------------------
        self.mu_object_ = 0.55
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 15

        # ---------------------------------------------------------------------------------------------
        #      models parameters
        # ---------------------------------------------------------------------------------------------
        self.obj_inertia_ = np.identity(6, dtype=np.float32)
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3, dtype=np.float32)
        self.obj_inertia_[3:, 3:] = 0.05 * np.eye(3, dtype=np.float32)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [300]).astype(np.float32)

        Q = np.zeros((self.n_qvel_, self.n_qvel_), dtype=np.float32)
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q

        self.obj_mass_ = np.float32(2.0 * STUDYTABLE_SCALE**3)
        self.gravity_ = np.array([0.0, 0.0, -9.8, 0.0, 0.0, 0.0], dtype=np.float32)

        self.model_params = args.model_param

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_horizon_ = 5
        self.ipopt_max_iter_ = 100
        self.mpc_model = mpc_model

        self.mpc_u_lb_ = -0.1
        self.mpc_u_ub_ = 0.1
        fingertip_q_lb = np.array([-3, -3, self.table_height + 0.02], dtype=np.float32)
        fingertip_q_ub = np.array([3, 3, 3], dtype=np.float32)
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), fingertip_q_lb)).astype(
            np.float32
        )
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), fingertip_q_ub)).astype(
            np.float32
        )
        self.sol_guess_ = None
        self.comple_relax = 0.1
        self.max_env_contacts_ = 8
        optimizer_kwargs = dict(
            obj_mass=self.obj_mass_,
            arm_friction=self.mu_object_,
            contact_stiffness=self.model_params,
            time_step=self.h_ * 10,
            max_contacts=self.max_ncon_,
            sample_num=args.sample_num,
            pos_coef=args.pos_coef,
            ori_coef=args.ori_coef,
            nlp_solver=getattr(args, "mlqp_solver", "snopt"),
        )
        self.handle_lambda_optimizer_ = LambdaContactControlOptimizer(
            mesh_path=self.handle_mesh_path_,
            **optimizer_kwargs,
        )
        self.front_panel_lambda_optimizer_ = LambdaContactControlOptimizer(
            mesh_path=self.front_panel_mesh_path_,
            **optimizer_kwargs,
        )
        self.lambda_optimizer = self.handle_lambda_optimizer_

    def get_sampling_stage_name(self, drawer_offset):
        drawer_open_amount = abs(
            float(drawer_offset) - float(self.drawer_initial_offset_)
        )
        if drawer_open_amount < self.handle_only_open_distance_:
            return DRAWER_SAMPLING_HANDLE_NAME
        return DRAWER_SAMPLING_FRONT_PANEL_NAME

    def get_sampling_stl_names(self, drawer_offset):
        return [self.get_sampling_stage_name(drawer_offset)]

    def get_active_lambda_optimizer(self, drawer_offset):
        if (
            self.get_sampling_stage_name(drawer_offset)
            == DRAWER_SAMPLING_FRONT_PANEL_NAME
        ):
            return self.front_panel_lambda_optimizer_
        return self.handle_lambda_optimizer_

    @staticmethod
    def calculate_rotation_quaternion(x, target_position):
        direction = x[:3] - target_position
        direction = direction / cs.norm_2(direction)

        angle = cs.arctan2(direction[1], direction[0])
        half_angle = angle / 2.0
        w = cs.cos(half_angle)
        z = cs.sin(half_angle)
        return [w, 0, 0, z]

    def init_cost_fns(self):
        x = cs.SX.sym("x", self.n_qpos_)
        u = cs.SX.sym("u", self.n_cmd_)

        target_position = cs.SX.sym("target_position", 3)
        target_quaternion = cs.SX.sym("target_quaternion", 4)
        position_cost = cs.sumsqr(x[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(x[3:7], target_quaternion) ** 2
        contact_cost = cs.sumsqr(x[0:3] - x[7:10])
        control_cost = cs.sumsqr(u)
        virtual_point = cs.SX.sym("virtual_point", 3)
        contact_point = cs.SX.sym("contact point", 3)
        curr_ori_coef = cs.SX.sym("curr_ori_coef", 1)

        phi_vec = cs.SX.sym("phi_vec", self.max_ncon_ * 4)
        jac_mat = cs.SX.sym("jac_mat", self.max_ncon_ * 4, self.n_qvel_)
        verify_cost_param = cs.SX.sym("verify_cost_param", 1)
        virtual_point_cost = self.log_barrier_function(x, virtual_point)
        contact_point_cost = self.log_barrier_function(x, contact_point)

        reject_cost = cs.if_else(
            cs.sumsqr(x[7:10] - contact_point) < self.reject_dis,
            -self.log_barrier_function(x, contact_point),
            0.0,
        )
        attract_cost = (
            self.attract_coef * virtual_point_cost + self.reject_coef * reject_cost
        )

        cost_param = cs.vvcat(
            [
                target_position,
                target_quaternion,
                phi_vec,
                jac_mat,
                verify_cost_param,
                virtual_point,
                contact_point,
                curr_ori_coef,
            ]
        )

        base_cost = (
            1 - verify_cost_param
        ) * attract_cost + self.contact_coef * verify_cost_param * (
            self.contact_cost_param * contact_cost
            + (1 - self.contact_cost_param) * contact_point_cost
        )
        final_cost = 500 * position_cost * curr_ori_coef + 5.0 * quaternion_cost

        path_cost_fn = cs.Function(
            "path_cost_fn",
            [x, u, cost_param],
            [base_cost + 50 * control_cost * verify_cost_param],
        )
        final_cost_fn = cs.Function(
            "final_cost_fn",
            [x, cost_param],
            [10 * final_cost * verify_cost_param],
        )

        return path_cost_fn, final_cost_fn

    @staticmethod
    def log_barrier_function(x, virtual_point, epsilon=1e-3):
        diff = x[7:10] - virtual_point
        squared_norm = cs.dot(diff, diff) + epsilon
        barrier_cost = cs.log(squared_norm)
        return barrier_cost
