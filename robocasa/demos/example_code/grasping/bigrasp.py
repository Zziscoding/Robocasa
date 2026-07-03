import argparse
import copy
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import trimesh
from scipy.linalg import pinv
from scipy.spatial.transform import Rotation


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[3]
CUROBO_SRC_ROOT = REPO_ROOT.parent / "thirdparty" / "curobo" / "src"
for path in (REPO_ROOT, CUROBO_SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    import torch
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    _HAS_CUROBO = True
    _CUROBO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime dependency
    torch = None
    Cuboid = None
    WorldConfig = None
    Pose = None
    JointState = None
    IKSolver = None
    IKSolverConfig = None
    _HAS_CUROBO = False
    _CUROBO_IMPORT_ERROR = exc

from robocasa.demos.example_code.grasping.mlqp_point_v2 import (
    LambdaContactControlOptimizer,
)

# from planning.mpppi_explicit import MPPIExplicit
from robocasa.demos.example_code.grasping.mpc_explicit2_bigrasp import MPCExplicit

PANDA_XML_PATH = REPO_ROOT / "envs" / "xmls" / "panda_nohand.xml"
GENERATED_SCENE_PATH = REPO_ROOT / "envs" / "xmls" / "_generated_bigrasp_scene.xml"
OBJECT_ASSET_DIR = REPO_ROOT / "envs" / "assets" / "objects"

PANDA_HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
TIP_RADIUS = 0.01
TIP_CENTER_OFFSET = 0.06
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
DEFAULT_SCALE_MAP = {
    "stanford_bunny2": np.array([1.5, 1.5, 1.5], dtype=np.float64),
    "rubber_duck": np.array([1.3, 1.4, 1.4], dtype=np.float64),
    "Wolf_Duck": np.array([0.002, 0.002, 0.002], dtype=np.float64),
}
DEFAULT_OBJECT_SCALE_BOOST = 1.2
ARM_OBSTACLE_SEGMENTS = (
    ("link6", "link5", "link6", 0.12),
    ("link7", "link6", "link7", 0.10),
    ("fingertip", "attachment", "tip_center", 0.07),
)
ARM_OBSTACLE_LENGTH_PADDING = 0.06
PANDA_TORQUE_LIMITS = np.array(
    [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64
)


def _normalize(vec, eps=1e-9):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def _project_to_plane(vec, normal):
    vec = np.asarray(vec, dtype=np.float64)
    normal = _normalize(normal)
    return vec - np.dot(vec, normal) * normal


def _project_to_rotation_matrix(rotation_matrix):
    u, _, vh = np.linalg.svd(np.asarray(rotation_matrix, dtype=np.float64))
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected


def _rotation_error(current_rot, target_rot):
    current_rot = np.asarray(current_rot, dtype=np.float64).reshape(3, 3)
    target_rot = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    return 0.5 * (
        np.cross(current_rot[:, 0], target_rot[:, 0])
        + np.cross(current_rot[:, 1], target_rot[:, 1])
        + np.cross(current_rot[:, 2], target_rot[:, 2])
    )


def parse_bool_arg(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _slerp_rotation_matrix(start_rot, end_rot, alpha):
    start_rot = _project_to_rotation_matrix(start_rot)
    end_rot = _project_to_rotation_matrix(end_rot)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 1e-9:
        return start_rot
    if alpha >= 1.0 - 1e-9:
        return end_rot

    q0 = Rotation.from_matrix(start_rot).as_quat()
    q1 = Rotation.from_matrix(end_rot).as_quat()
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        q /= max(np.linalg.norm(q), 1e-9)
        return _project_to_rotation_matrix(Rotation.from_quat(q).as_matrix())

    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta)) / max(sin_theta_0, 1e-9)
    s1 = sin_theta / max(sin_theta_0, 1e-9)
    q = s0 * q0 + s1 * q1
    q /= max(np.linalg.norm(q), 1e-9)
    return _project_to_rotation_matrix(Rotation.from_quat(q).as_matrix())


def _rotate_vector_toward(source_vec, target_vec, max_angle):
    source_vec = _normalize(source_vec)
    target_vec = _normalize(target_vec)
    max_angle = max(float(max_angle), 0.0)
    if np.linalg.norm(source_vec) < 1e-9:
        return (
            target_vec
            if np.linalg.norm(target_vec) >= 1e-9
            else np.array([0.0, 0.0, -1.0], dtype=np.float64)
        )
    if np.linalg.norm(target_vec) < 1e-9 or max_angle <= 1e-9:
        return source_vec

    dot = float(np.clip(np.dot(source_vec, target_vec), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle <= 1e-9:
        return source_vec

    step_angle = min(angle, max_angle)
    rot_axis = np.cross(source_vec, target_vec)
    axis_norm = float(np.linalg.norm(rot_axis))
    if axis_norm < 1e-9:
        rot_axis = _project_to_plane(
            np.array([1.0, 0.0, 0.0], dtype=np.float64), source_vec
        )
        if np.linalg.norm(rot_axis) < 1e-9:
            rot_axis = _project_to_plane(
                np.array([0.0, 1.0, 0.0], dtype=np.float64), source_vec
            )
        axis_norm = float(np.linalg.norm(rot_axis))
        if axis_norm < 1e-9:
            return source_vec
    rot_axis = rot_axis / axis_norm

    rotated = (
        source_vec * np.cos(step_angle)
        + np.cross(rot_axis, source_vec) * np.sin(step_angle)
        + rot_axis * np.dot(rot_axis, source_vec) * (1.0 - np.cos(step_angle))
    )
    return _normalize(rotated)


def _joint_index(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    if match is None:
        raise ValueError(f"Unable to extract joint index from name: {name}")
    return int(match.group(1))


def _tensor_to_numpy(value) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value) -> float:
    return float(_tensor_to_numpy(value).reshape(-1)[0])


def quat_wxyz_to_mat(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat_wxyz(rotation_matrix):
    rotation_matrix = _project_to_rotation_matrix(rotation_matrix)
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rotation_matrix.reshape(-1))
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def quat_from_axis_angle(axis, angle):
    axis = _normalize(axis)
    if np.linalg.norm(axis) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half = 0.5 * float(angle)
    return np.array(
        [
            np.cos(half),
            axis[0] * np.sin(half),
            axis[1] * np.sin(half),
            axis[2] * np.sin(half),
        ],
        dtype=np.float64,
    )


def quat_from_yaw(yaw):
    return quat_from_axis_angle([0.0, 0.0, 1.0], yaw)


def _extract_pose_from_kinematics(kinematics_state):
    ee_pose = getattr(kinematics_state, "ee_pose", None)
    if ee_pose is not None:
        pos = _tensor_to_numpy(ee_pose.position).reshape(-1, 3)[0]
        quat = _tensor_to_numpy(ee_pose.quaternion).reshape(-1, 4)[0]
        return pos.astype(np.float64), quat.astype(np.float64)

    pos = _tensor_to_numpy(kinematics_state.ee_pos_seq).reshape(-1, 3)[0]
    quat = _tensor_to_numpy(kinematics_state.ee_quat_seq).reshape(-1, 4)[0]
    return pos.astype(np.float64), quat.astype(np.float64)


def build_curobo_state(arm_q_mj, curobo_joint_names, default_joint_vector):
    arm_q_mj = np.asarray(arm_q_mj, dtype=np.float64).reshape(-1)
    joint_vec = (
        np.asarray(default_joint_vector, dtype=np.float64)
        .reshape(len(curobo_joint_names))
        .copy()
    )
    arm_map = {
        _joint_index(f"joint{joint_idx + 1}"): float(value)
        for joint_idx, value in enumerate(arm_q_mj)
    }
    for idx, joint_name in enumerate(curobo_joint_names):
        if "finger" in joint_name:
            continue
        joint_vec[idx] = arm_map[_joint_index(joint_name)]
    return joint_vec


def extract_mujoco_arm_configuration(curobo_joint_vector, curobo_joint_names):
    curobo_joint_vector = np.asarray(curobo_joint_vector, dtype=np.float64).reshape(
        len(curobo_joint_names)
    )
    arm_map = {
        _joint_index(name): value
        for name, value in zip(curobo_joint_names, curobo_joint_vector)
        if "finger" not in name
    }
    return np.array([arm_map[idx] for idx in range(1, 8)], dtype=np.float64)


def make_joint_state(controller, joint_vector, joint_names=None):
    joint_names = controller.joint_names if joint_names is None else joint_names
    tensor = torch.tensor(
        np.asarray(joint_vector, dtype=np.float32).reshape(1, -1),
        device=controller.tensor_args.device,
        dtype=controller.tensor_args.dtype,
    )
    return JointState.from_position(tensor, joint_names=joint_names)


def make_pose(controller, position, quat_wxyz):
    pos_t = torch.tensor(
        np.asarray(position, dtype=np.float32).reshape(1, 3),
        device=controller.tensor_args.device,
        dtype=controller.tensor_args.dtype,
    )
    quat_t = torch.tensor(
        np.asarray(quat_wxyz, dtype=np.float32).reshape(1, 4),
        device=controller.tensor_args.device,
        dtype=controller.tensor_args.dtype,
    )
    return Pose(position=pos_t, quaternion=quat_t)


def make_joint_tensor(controller, joint_vector, extra_dim=False):
    shape = (1, 1, -1) if extra_dim else (1, -1)
    return torch.tensor(
        np.asarray(joint_vector, dtype=np.float32).reshape(shape),
        device=controller.tensor_args.device,
        dtype=controller.tensor_args.dtype,
    )


def resolve_mesh_path(obj_name=None, mesh_path=None):
    if mesh_path:
        path = Path(mesh_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Mesh file does not exist: {path}")
        return path

    if not obj_name:
        raise ValueError("Either --obj or --mesh must be provided.")

    candidates = [
        OBJECT_ASSET_DIR / f"{obj_name}{suffix}"
        for suffix in ("", ".stl", ".obj", ".ply", ".off")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find mesh for object '{obj_name}' in {OBJECT_ASSET_DIR}"
    )


def load_mesh_bounds(mesh_path, scale):
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    mesh.apply_scale(np.asarray(scale, dtype=np.float64))
    return np.asarray(mesh.bounds, dtype=np.float64)


def format_vec(vec):
    return " ".join(
        f"{float(v):.8f}" for v in np.asarray(vec, dtype=np.float64).reshape(-1)
    )


def prefix_robot_tree(element, prefix):
    element = copy.deepcopy(element)
    queue = [element]
    while queue:
        node = queue.pop()
        if "name" in node.attrib:
            node.attrib["name"] = f"{prefix}{node.attrib['name']}"
        queue.extend(list(node))
    return element


def prefix_reference_attributes(element, prefix):
    element = copy.deepcopy(element)
    rename_keys = (
        "joint",
        "joint1",
        "joint2",
        "body",
        "body1",
        "body2",
        "site",
        "geom",
        "geom1",
        "geom2",
        "camera",
        "light",
    )
    queue = [element]
    while queue:
        node = queue.pop()
        for key in rename_keys:
            if key in node.attrib:
                node.attrib[key] = f"{prefix}{node.attrib[key]}"
        if "name" in node.attrib:
            node.attrib["name"] = f"{prefix}{node.attrib['name']}"
        queue.extend(list(node))
    return element


def build_bimanual_scene_xml(
    mesh_path,
    mesh_scale,
    object_mass,
    object_friction,
    object_pos,
    object_quat,
    scene_center_x,
    robot_span,
    pedestal_size=(0.05, 0.07, 0.06),
    pedestal_pos=(0.58, 0.0, 0.06),
    mujoco_timestep=0.01,
    scene_output_path=GENERATED_SCENE_PATH,
):
    panda_root = ET.parse(PANDA_XML_PATH).getroot()

    root = ET.Element("mujoco", {"model": "dual panda bigrasp"})
    root.append(
        ET.Element(
            "compiler", {"angle": "radian", "meshdir": "assets", "autolimits": "true"}
        )
    )
    root.append(
        ET.Element(
            "option",
            {
                "integrator": "implicitfast",
                "impratio": "10",
                "timestep": f"{float(mujoco_timestep):.8f}",
            },
        )
    )
    root.append(
        ET.Element(
            "statistic", {"center": f"{scene_center_x:.4f} 0 0.35", "extent": "1.2"}
        )
    )

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "diffuse": "0.6 0.6 0.6",
            "ambient": "0.25 0.25 0.25",
            "specular": "0.1 0.1 0.1",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-25"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.30 0.45 0.60",
            "rgb2": "0.02 0.03 0.04",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.20 0.28 0.34",
            "rgb2": "0.12 0.16 0.20",
            "markrgb": "0.85 0.85 0.85",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )
    ET.SubElement(
        asset, "material", {"name": "pedestal_mat", "rgba": "0.28 0.30 0.34 1"}
    )
    ET.SubElement(asset, "material", {"name": "obj_mat", "rgba": "0.88 0.52 0.22 1"})
    ET.SubElement(
        asset, "material", {"name": "ghost_obj_mat", "rgba": "0.88 0.52 0.22 0.20"}
    )
    ET.SubElement(
        asset, "material", {"name": "left_marker_mat", "rgba": "0.15 0.75 0.95 1"}
    )
    ET.SubElement(
        asset, "material", {"name": "right_marker_mat", "rgba": "0.95 0.25 0.35 1"}
    )
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "object_mesh",
            "file": str(Path(mesh_path).resolve()),
            "scale": format_vec(mesh_scale),
        },
    )
    for child in list(panda_root.find("asset")):
        asset.append(copy.deepcopy(child))

    default = ET.SubElement(root, "default")
    for child in list(panda_root.find("default")):
        default.append(copy.deepcopy(child))

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"pos": "0.4 -0.2 1.6", "dir": "0 0 -1", "directional": "true"},
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": "1.15 -0.70 0.65",
            "xyaxes": "0.51 0.86 0.00 -0.30 0.18 0.94",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "size": "0 0 0.05",
            "type": "plane",
            "material": "groundplane",
        },
    )

    ET.SubElement(
        ET.SubElement(
            worldbody,
            "body",
            {"name": "pedestal", "pos": format_vec(pedestal_pos)},
        ),
        "geom",
        {
            "name": "pedestal_geom",
            "type": "box",
            "size": format_vec(pedestal_size),
            "material": "pedestal_mat",
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.0 0.08 0.01",
            "condim": "3",
        },
    )

    goal_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "goal",
            "pos": format_vec(object_pos),
            "quat": format_vec(object_quat),
        },
    )
    ET.SubElement(
        goal_body,
        "geom",
        {
            "name": "goal_geom",
            "type": "mesh",
            "mesh": "object_mesh",
            "material": "ghost_obj_mat",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    obj_body = ET.SubElement(worldbody, "body", {"name": "obj"})
    ET.SubElement(obj_body, "freejoint", {"name": "obj_freejoint"})
    ET.SubElement(
        obj_body,
        "geom",
        {
            "name": "obj",
            "type": "mesh",
            "mesh": "object_mesh",
            "material": "obj_mat",
            "mass": f"{float(object_mass):.8f}",
            "condim": "6",
            "friction": f"{float(object_friction):.8f} 0.08 0.01",
            "solimp": "0.9 0.95 0.01",
            "solref": "0.02 1.0",
            "margin": "0.0",
            "gap": "0.0",
        },
    )

    marker_specs = [
        ("obj_point", "1 1 0 1"),
        ("contact_point1", "0 1 0 1"),
        ("contact_point2", "0 0 1 1"),
        ("left_goal", "0.15 0.75 0.95 1"),
        ("right_goal", "0.95 0.25 0.35 1"),
        ("left_desired_tip", "0.72 0.32 0.92 1"),
        ("right_desired_tip", "0.72 0.32 0.92 1"),
    ]
    for name, rgba in marker_specs:
        marker_body = ET.SubElement(
            worldbody, "body", {"name": name, "pos": format_vec(object_pos)}
        )
        ET.SubElement(
            marker_body,
            "geom",
            {
                "name": f"{name}_geom",
                "type": "sphere",
                "size": "0.008",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
            },
        )

    ghost_specs = [
        ("left_ghost_tip", "0.15 0.75 0.95 0.28"),
        ("right_ghost_tip", "0.95 0.25 0.35 0.28"),
    ]
    for name, rgba in ghost_specs:
        ghost_body = ET.SubElement(
            worldbody, "body", {"name": name, "pos": format_vec(object_pos)}
        )
        ET.SubElement(
            ghost_body,
            "geom",
            {
                "name": f"{name}_rod",
                "type": "cylinder",
                "size": "0.005 0.03",
                "pos": "0 0 0.03",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
            },
        )
        ET.SubElement(
            ghost_body,
            "geom",
            {
                "name": f"{name}_sphere",
                "type": "sphere",
                "size": "0.010",
                "pos": "0 0 0.06",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
            },
        )

    robot_body = panda_root.find("./worldbody/body[@name='link0']")
    if robot_body is None:
        raise RuntimeError(f"Failed to find Panda root body in {PANDA_XML_PATH}")
    robot_contact = panda_root.find("contact")
    left_root = prefix_robot_tree(robot_body, "left_")
    left_root.attrib["pos"] = format_vec([scene_center_x - 0.5 * robot_span, 0.0, 0.0])
    left_root.attrib["quat"] = "1 0 0 0"
    left_attachment = left_root.find(".//body[@name='left_attachment']")
    if left_attachment is not None:
        ET.SubElement(
            left_attachment,
            "site",
            {"name": "left_tip_center", "pos": "0 0 0.06", "size": "0.002"},
        )

    right_root = prefix_robot_tree(robot_body, "right_")
    right_root.attrib["pos"] = format_vec([scene_center_x + 0.5 * robot_span, 0.0, 0.0])
    right_root.attrib["quat"] = "0 0 0 1"
    right_attachment = right_root.find(".//body[@name='right_attachment']")
    if right_attachment is not None:
        ET.SubElement(
            right_attachment,
            "site",
            {"name": "right_tip_center", "pos": "0 0 0.06", "size": "0.002"},
        )

    worldbody.append(left_root)
    worldbody.append(right_root)

    actuator = ET.SubElement(root, "actuator")
    for prefix in ("left_", "right_"):
        for joint_idx, torque_limit in enumerate(PANDA_TORQUE_LIMITS, start=1):
            ET.SubElement(
                actuator,
                "motor",
                {
                    "name": f"{prefix}actuator{joint_idx}",
                    "joint": f"{prefix}joint{joint_idx}",
                    "ctrllimited": "true",
                    "ctrlrange": f"{-float(torque_limit):.8f} {float(torque_limit):.8f}",
                },
            )

    contact = ET.SubElement(root, "contact")
    for child in list(robot_contact):
        contact.append(prefix_reference_attributes(child, "left_"))
    for child in list(robot_contact):
        contact.append(prefix_reference_attributes(child, "right_"))

    tree = ET.ElementTree(root)
    scene_output_path = Path(scene_output_path)
    scene_output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(scene_output_path, encoding="utf-8", xml_declaration=False)
    return scene_output_path


@dataclass
class ArmHandles:
    prefix: str
    joint_ids: np.ndarray
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    actuator_ids: np.ndarray
    body_id: int
    tip_geom_id: int
    tip_site_id: int
    ghost_body_id: int
    base_pos: np.ndarray
    base_rot: np.ndarray
    body_ids_by_name: dict
    ik_solver: object = None
    curobo_joint_names: tuple = ()
    retract_cfg: np.ndarray = None
    static_world_with_pedestal: object = None
    static_world_floor_only: object = None
    static_world: object = None
    current_world: object = None
    current_world_mode: str = "with_pedestal"
    torque_limits: np.ndarray = None
    home_q: np.ndarray = None
    cartesian_stiffness: np.ndarray = None
    cartesian_damping: np.ndarray = None
    nullspace_stiffness: float = 10.0
    position_d: np.ndarray = None
    orientation_d: np.ndarray = None
    p_d: np.ndarray = None
    R_d: np.ndarray = None


@dataclass
class ArmIkResult:
    q_mj: np.ndarray
    success: bool
    position_error: float
    rotation_error: float
    target_hand_pos_world: np.ndarray
    target_hand_rot_world: np.ndarray
    solved_hand_pos_world: np.ndarray
    solved_hand_rot_world: np.ndarray
    solved_tip_pos_world: np.ndarray
    solved_tip_rot_world: np.ndarray
    constraint_total: float = 0.0
    bound_constraint: float = 0.0
    world_constraint: float = 0.0
    static_world_constraint: float = 0.0
    self_constraint: float = 0.0
    failure_reason: str = ""


class DualArmPlanOnceParams:
    def __init__(self, args, obj_mass):
        self.contact_cost_param = float(args.planner_contact_cost_param)
        self.attract_coef = float(args.planner_attract_coef)
        self.reject_coef = float(args.planner_reject_coef)
        self.contact_coef = float(args.planner_contact_coef)
        self.reject_dis = float(args.planner_reject_distance)

        self.h_ = float(args.planner_dt)
        self.n_robot_qpos_ = 6
        self.n_qpos_ = 13
        self.n_qvel_ = 12
        self.n_cmd_ = 6
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = int(args.planner_max_contacts)

        self.obj_inertia_ = np.identity(6, dtype=np.float32)
        self.obj_inertia_[0:3, 0:3] = float(args.planner_object_inertia_pos) * np.eye(
            3, dtype=np.float32
        )
        self.obj_inertia_[3:, 3:] = float(args.planner_object_inertia_rot) * np.eye(
            3, dtype=np.float32
        )
        self.robot_stiff_ = np.diag(
            self.n_cmd_ * [float(args.planner_robot_stiffness)]
        ).astype(np.float32)

        self.Q = np.zeros((self.n_qvel_, self.n_qvel_), dtype=np.float32)
        self.Q[:6, :6] = self.obj_inertia_
        self.Q[6:, 6:] = self.robot_stiff_

        self.obj_mass_ = float(obj_mass)
        self.gravity_ = np.array([0.0, 0.0, -9.8, 0.0, 0.0, 0.0], dtype=np.float32)
        self.model_params = float(args.contact_stiffness)

        self.mpc_horizon_ = int(args.planner_horizon)
        self.mpc_model = "explicit"
        self.planner_solver_ = str(args.planner_solver).strip().lower()
        self.mpc_u_lb_ = -float(args.planner_cmd_limit)
        self.mpc_u_ub_ = float(args.planner_cmd_limit)

        self.sol_guess_ = None
        self.mppi_samples_ = int(args.mppi_samples)
        self.mppi_iterations_ = int(args.mppi_iterations)
        self.mppi_init_iterations_ = int(args.mppi_init_iterations)
        self.mppi_lambda_ = float(args.mppi_lambda)
        self.mppi_noise_sigma_ = float(args.mppi_noise_sigma)
        self.mppi_noise_decay_ = float(args.mppi_noise_decay)
        self.mppi_elite_frac_ = float(args.mppi_elite_frac)
        self.mppi_use_torch_compile_ = bool(args.mppi_use_torch_compile)
        default_mppi_device = (
            "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
        )
        self.mppi_device_ = str(args.mppi_device or default_mppi_device)


class BimanualPandaGrasper:
    def __init__(self, args):
        if not _HAS_CUROBO:
            raise ImportError(
                "Failed to import cuRobo. Make sure cuRobo is installed or "
                f"{CUROBO_SRC_ROOT} is available on PYTHONPATH. "
                f"Original error: {_CUROBO_IMPORT_ERROR!r}"
            )

        self.args = args
        logging.getLogger("curobo").setLevel(logging.WARNING)
        self.mesh_path = resolve_mesh_path(args.obj, args.mesh)
        self.mesh_scale = self._resolve_mesh_scale(args)
        self.mesh_bounds = load_mesh_bounds(self.mesh_path, self.mesh_scale)
        self.pedestal_size = np.asarray(args.pedestal_size, dtype=np.float64).copy()
        self.pedestal_pos = np.asarray(args.pedestal_pos, dtype=np.float64).copy()
        self.pedestal_pos[2] += float(args.initial_object_lift)

        support_top = self.pedestal_pos[2] + self.pedestal_size[2]
        self.support_top = float(support_top)
        self.support_surface_point = np.array(
            [self.pedestal_pos[0], self.pedestal_pos[1], self.support_top],
            dtype=np.float64,
        )
        self.support_surface_normal = WORLD_UP.copy()
        object_pos = np.array(
            [
                args.scene_center_x,
                0.0,
                support_top - self.mesh_bounds[0, 2] + args.object_z_offset,
            ],
            dtype=np.float64,
        )
        object_quat = quat_from_yaw(args.object_yaw)

        self.left_base_pos = np.array(
            [args.scene_center_x - 0.5 * args.robot_span, 0.0, 0.0],
            dtype=np.float64,
        )
        self.right_base_pos = np.array(
            [args.scene_center_x + 0.5 * args.robot_span, 0.0, 0.0],
            dtype=np.float64,
        )
        self.left_base_rot = np.eye(3, dtype=np.float64)
        self.right_base_rot = quat_wxyz_to_mat(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )

        self.scene_path = build_bimanual_scene_xml(
            mesh_path=self.mesh_path,
            mesh_scale=self.mesh_scale,
            object_mass=args.obj_mass,
            object_friction=args.object_friction,
            object_pos=object_pos,
            object_quat=object_quat,
            scene_center_x=args.scene_center_x,
            robot_span=args.robot_span,
            pedestal_size=self.pedestal_size,
            pedestal_pos=self.pedestal_pos,
            mujoco_timestep=args.mujoco_dt,
            scene_output_path=args.scene_output,
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = float(args.mujoco_dt)
        self.data = mujoco.MjData(self.model)

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.cam.distance = 1.8
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25

        self.left_arm = self._build_arm_handles(
            "left_", self.left_base_pos, self.left_base_rot
        )
        self.right_arm = self._build_arm_handles(
            "right_", self.right_base_pos, self.right_base_rot
        )
        self.obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "obj"
        )
        self.obj_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint"
        )
        self.obj_qpos_adr = int(self.model.jnt_qposadr[self.obj_joint_id])
        self.obj_dof_adr = int(self.model.jnt_dofadr[self.obj_joint_id])
        self.obj_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj"
        )
        self.marker_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (
                "obj_point",
                "contact_point1",
                "contact_point2",
                "left_goal",
                "right_goal",
                "left_desired_tip",
                "right_desired_tip",
                "goal",
            )
        }
        self.support_height_threshold = (
            self.pedestal_pos[2] + self.pedestal_size[2] + args.ground_height_margin
        )

        self.reset(object_pos, object_quat)
        optimizer_support_kwargs = {}
        if bool(args.optimizer_use_support_filter):
            optimizer_support_kwargs = {
                "support_surface_point": self.support_surface_point,
                "support_surface_normal": self.support_surface_normal,
                "support_surface_clearance": args.ground_height_margin,
                "support_surface_normal_alignment_threshold": args.support_normal_alignment_threshold,
            }
        self.optimizer = LambdaContactControlOptimizer(
            mesh_path=str(self.mesh_path),
            obj_mass=args.obj_mass,
            arm_friction=args.optimizer_arm_friction,
            contact_stiffness=args.contact_stiffness,
            time_step=self.model.opt.timestep,
            sample_num=args.sample_num,
            pos_coef=args.pos_coef,
            ori_coef=args.ori_coef,
            scale_factors=tuple(self.mesh_scale.tolist()),
            curvature_neighbor_k=args.optimizer_curvature_neighbor_k,
            region_max_mean_curvature=args.optimizer_max_region_mean_curvature,
            region_max_point_curvature=args.optimizer_max_point_curvature,
            curvature_penalty_weight=args.optimizer_curvature_penalty_weight,
            nlp_solver=args.solver,
            static_nlp_solver=args.solver,
            **optimizer_support_kwargs,
        )
        self.optimizer.set_timing_print_enabled(
            bool(getattr(args, "print_contact_timing", False))
        )
        self.command_dt = (
            float(self.model.opt.timestep)
            * max(int(self.args.mj_steps_per_command), 1)
            * max(int(self.args.command_substeps), 1)
        )
        self.plan_params = DualArmPlanOnceParams(self.args, args.obj_mass)
        self.planner = MPCExplicit(self.plan_params)
        self._setup_curobo()

    def _resolve_mesh_scale(self, args):
        if args.scale is not None:
            return np.asarray(args.scale, dtype=np.float64)
        key = self.mesh_path.stem
        base_scale = DEFAULT_SCALE_MAP.get(key, None)
        if base_scale is None:
            return np.ones(3, dtype=np.float64)
        return (
            DEFAULT_OBJECT_SCALE_BOOST * np.asarray(base_scale, dtype=np.float64).copy()
        )

    def _build_arm_handles(self, prefix, base_pos, base_rot):
        joint_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}joint{i}"
                )
                for i in range(1, 8)
            ],
            dtype=np.int32,
        )
        qpos_adr = np.array(
            [self.model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32
        )
        dof_adr = np.array(
            [self.model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32
        )
        actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}actuator{i}"
                )
                for i in range(1, 8)
            ],
            dtype=np.int32,
        )
        body_ids_by_name = {
            name: mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{name}"
            )
            for name in (
                "link0",
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "link7",
                "attachment",
            )
        }
        return ArmHandles(
            prefix=prefix,
            joint_ids=joint_ids,
            qpos_adr=qpos_adr,
            dof_adr=dof_adr,
            actuator_ids=actuator_ids,
            body_id=mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}attachment"
            ),
            tip_geom_id=mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}fingertip"
            ),
            tip_site_id=mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}tip_center"
            ),
            ghost_body_id=mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{prefix}ghost_tip",
            ),
            base_pos=np.asarray(base_pos, dtype=np.float64).copy(),
            base_rot=np.asarray(base_rot, dtype=np.float64).reshape(3, 3).copy(),
            body_ids_by_name=body_ids_by_name,
            torque_limits=PANDA_TORQUE_LIMITS.copy(),
            home_q=PANDA_HOME_Q.copy(),
            cartesian_stiffness=2.0
            * np.diag(
                [
                    float(self.args.cartesian_stiffness_pos),
                    float(self.args.cartesian_stiffness_pos),
                    float(self.args.cartesian_stiffness_pos),
                    float(self.args.cartesian_stiffness_rot),
                    float(self.args.cartesian_stiffness_rot),
                    float(self.args.cartesian_stiffness_rot),
                ]
            ).astype(np.float64),
            nullspace_stiffness=float(self.args.nullspace_stiffness),
        )

    def _build_curobo_world_config_dict(self, arm, include_pedestal=True):
        floor_world_pos = np.array(
            [self.args.scene_center_x, 0.0, -0.05], dtype=np.float64
        )
        floor_world_rot = np.eye(3, dtype=np.float64)

        floor_local_pos, floor_local_rot = self.world_pose_to_arm_frame(
            arm, floor_world_pos, floor_world_rot
        )
        cuboids = {
            "floor": {
                "dims": [4.0, 4.0, 0.1],
                "pose": [
                    *floor_local_pos.tolist(),
                    *mat_to_quat_wxyz(floor_local_rot).tolist(),
                ],
            }
        }
        if include_pedestal:
            pedestal_world_pos = self.pedestal_pos.copy()
            pedestal_world_rot = np.eye(3, dtype=np.float64)
            pedestal_local_pos, pedestal_local_rot = self.world_pose_to_arm_frame(
                arm,
                pedestal_world_pos,
                pedestal_world_rot,
            )
            cuboids["pedestal"] = {
                "dims": (2.0 * self.pedestal_size).tolist(),
                "pose": [
                    *pedestal_local_pos.tolist(),
                    *mat_to_quat_wxyz(pedestal_local_rot).tolist(),
                ],
            }

        return {"cuboid": cuboids}

    def _build_curobo_world_config(self, arm, include_pedestal=True):
        return WorldConfig.from_dict(
            self._build_curobo_world_config_dict(arm, include_pedestal=include_pedestal)
        )

    def _set_curobo_world_mode(self, world_mode):
        if world_mode not in ("with_pedestal", "floor_only"):
            raise ValueError(f"Unsupported cuRobo world mode: {world_mode}")

        for arm in (self.left_arm, self.right_arm):
            if world_mode == "floor_only":
                static_world = arm.static_world_floor_only
            else:
                static_world = arm.static_world_with_pedestal
            if static_world is None:
                raise RuntimeError(
                    f"cuRobo static world '{world_mode}' is not initialized for {arm.prefix}."
                )
            arm.static_world = static_world
            arm.current_world_mode = world_mode

    def _build_segment_world_rotation(self, segment_dir, tangent_hint=None):
        return self._build_contact_rotation(segment_dir, tangent_hint=tangent_hint)

    def _get_body_pose(self, body_id, data=None):
        data = self.data if data is None else data
        pos = data.body(body_id).xpos.copy()
        rot = data.body(body_id).xmat.reshape(3, 3).copy()
        return pos, rot

    def _get_obstacle_anchor_pose(self, arm, anchor_name, data=None):
        if anchor_name == "tip_center":
            return self.get_tip_pose(arm, data=data)
        body_id = arm.body_ids_by_name[anchor_name]
        return self._get_body_pose(body_id, data=data)

    def _build_other_arm_obstacle_cuboids(self, target_arm, obstacle_arm):
        cuboids = []
        for name, start_name, end_name, thickness in ARM_OBSTACLE_SEGMENTS:
            start_pos_world, _ = self._get_obstacle_anchor_pose(
                obstacle_arm, start_name
            )
            end_pos_world, _ = self._get_obstacle_anchor_pose(obstacle_arm, end_name)
            segment_world = end_pos_world - start_pos_world
            segment_length = float(np.linalg.norm(segment_world))
            if segment_length < 1e-5:
                continue
            center_world = 0.5 * (start_pos_world + end_pos_world)
            seg_rot_world = self._build_segment_world_rotation(
                segment_world, tangent_hint=WORLD_UP
            )
            center_local, seg_rot_local = self.world_pose_to_arm_frame(
                target_arm,
                center_world,
                seg_rot_world,
            )
            cuboids.append(
                Cuboid(
                    name=f"{obstacle_arm.prefix}{name}_obs",
                    pose=[
                        *center_local.tolist(),
                        *mat_to_quat_wxyz(seg_rot_local).tolist(),
                    ],
                    dims=[
                        float(thickness),
                        float(thickness),
                        float(segment_length + ARM_OBSTACLE_LENGTH_PADDING),
                    ],
                )
            )
        return cuboids

    def _visible_optimizer_point_indices(self, object_pos, object_rot):
        if not bool(self.args.optimizer_use_support_filter):
            return self.optimizer.point_idx.copy()
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        centers_world = (object_rot @ self.optimizer.sample_point.T).T + object_pos[
            None, :
        ]
        visible_idx = np.where(
            centers_world[:, 2] > float(self.support_height_threshold)
        )[0]
        if visible_idx.size == 0:
            visible_idx = self.optimizer.point_idx.copy()
        visible_idx = self.optimizer.get_contact_candidate_indices(
            visible_face_idx=visible_idx,
            object_pos=object_pos,
            object_rot=object_rot,
        )
        if visible_idx.size == 0:
            visible_idx = self.optimizer.get_contact_candidate_indices(
                visible_face_idx=self.optimizer.point_idx,
                object_pos=object_pos,
                object_rot=object_rot,
            )
        return np.asarray(visible_idx, dtype=int)

    def _update_inter_arm_worlds(self):
        for target_arm, obstacle_arm in (
            (self.left_arm, self.right_arm),
            (self.right_arm, self.left_arm),
        ):
            world = target_arm.static_world.clone()
            for obstacle in self._build_other_arm_obstacle_cuboids(
                target_arm, obstacle_arm
            ):
                world.add_obstacle(obstacle)
            target_arm.current_world = world
            target_arm.ik_solver.update_world(world)

    def _setup_curobo_arm(self, arm):
        world_config = self._build_curobo_world_config_dict(arm, include_pedestal=True)
        ik_config = IKSolverConfig.load_from_robot_config(
            self.args.curobo_robot_cfg,
            world_config,
            position_threshold=self.args.ik_pos_tol,
            rotation_threshold=self.args.ik_rot_tol,
            num_seeds=self.args.ik_num_seeds,
            self_collision_check=not self.args.disable_curobo_self_collision,
            self_collision_opt=not self.args.disable_curobo_self_collision,
            collision_cache={"obb": 16},
            collision_activation_distance=self.args.curobo_collision_activation_distance,
            use_cuda_graph=False,
            regularization=True,
        )
        arm.ik_solver = IKSolver(ik_config)
        arm.curobo_joint_names = tuple(arm.ik_solver.joint_names)
        get_retract_config = getattr(arm.ik_solver, "get_retract_config", None)
        if callable(get_retract_config):
            retract_cfg = get_retract_config()
        else:
            retract_cfg = arm.ik_solver.rollout_fn.dynamics_model.retract_config
        arm.retract_cfg = _tensor_to_numpy(retract_cfg).reshape(-1).astype(np.float64)
        arm.cartesian_damping = 2.0 * np.sqrt(arm.cartesian_stiffness)
        arm.static_world_with_pedestal = self._build_curobo_world_config(
            arm, include_pedestal=True
        )
        arm.static_world_floor_only = self._build_curobo_world_config(
            arm, include_pedestal=False
        )
        arm.static_world = arm.static_world_with_pedestal
        arm.current_world = arm.static_world.clone()
        arm.current_world_mode = "with_pedestal"
        current_tip_pos, current_tip_rot = self.get_tip_pose(arm)
        arm.position_d = current_tip_pos.copy()
        arm.orientation_d = current_tip_rot.copy()
        arm.p_d = current_tip_pos.copy()
        arm.R_d = current_tip_rot.copy()
        self.set_ghost_pose(arm, *self.get_hand_pose(arm))

    def _setup_curobo(self):
        self._setup_curobo_arm(self.left_arm)
        self._setup_curobo_arm(self.right_arm)
        self._update_inter_arm_worlds()

    def reset(self, object_pos, object_quat):
        self.data.qpos[self.left_arm.qpos_adr] = PANDA_HOME_Q
        self.data.qpos[self.right_arm.qpos_adr] = PANDA_HOME_Q
        self.data.ctrl[self.left_arm.actuator_ids] = 0.0
        self.data.ctrl[self.right_arm.actuator_ids] = 0.0
        self.data.qpos[self.obj_qpos_adr : self.obj_qpos_adr + 7] = np.hstack(
            [object_pos, object_quat]
        )
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.set_ghost_pose(self.left_arm, *self.get_hand_pose(self.left_arm))
        self.set_ghost_pose(self.right_arm, *self.get_hand_pose(self.right_arm))
        for arm in (self.left_arm, self.right_arm):
            tip_pos, tip_rot = self.get_tip_pose(arm)
            arm.position_d = tip_pos.copy()
            arm.orientation_d = tip_rot.copy()
            arm.p_d = tip_pos.copy()
            arm.R_d = tip_rot.copy()
        mujoco.mj_forward(self.model, self.data)
        self.sync_viewer()

    def _print_viewer_camera_state(self):
        if self.viewer is None:
            return
        cam = self.viewer.cam
        lookat = np.asarray(cam.lookat, dtype=np.float64).reshape(3)
        azimuth_rad = np.deg2rad(float(cam.azimuth))
        elevation_rad = np.deg2rad(float(cam.elevation))
        distance = float(cam.distance)
        camera_pos = np.array(
            [
                lookat[0] - distance * np.cos(elevation_rad) * np.cos(azimuth_rad),
                lookat[1] - distance * np.cos(elevation_rad) * np.sin(azimuth_rad),
                lookat[2] - distance * np.sin(elevation_rad),
            ],
            dtype=np.float64,
        )
        print(
            "viewer_camera "
            f"pos={np.array2string(camera_pos, precision=5)} "
            f"lookat={np.array2string(lookat, precision=5)} "
            f"distance={distance:.5f} "
            f"azimuth={float(cam.azimuth):.5f} "
            f"elevation={float(cam.elevation):.5f}"
        )

    def sync_viewer(self):
        if self.viewer is None:
            return
        self.viewer.sync()
        if bool(getattr(self.args, "print_viewer_camera_state", False)):
            self._print_viewer_camera_state()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def is_running(self):
        if self.viewer is None:
            return True
        is_running = getattr(self.viewer, "is_running", None)
        if callable(is_running):
            return bool(is_running())
        return True

    def get_object_pose(self):
        qpos = self.data.qpos[self.obj_qpos_adr : self.obj_qpos_adr + 7].copy()
        pos = qpos[:3]
        quat = qpos[3:]
        return pos, quat, quat_wxyz_to_mat(quat)

    def apply_object_wrench_world(self, force_world=None, torque_world=None):
        self.data.xfrc_applied[:] = 0.0
        if force_world is not None:
            self.data.xfrc_applied[self.obj_body_id, :3] = np.asarray(
                force_world,
                dtype=np.float64,
            ).reshape(3)
        if torque_world is not None:
            self.data.xfrc_applied[self.obj_body_id, 3:] = np.asarray(
                torque_world,
                dtype=np.float64,
            ).reshape(3)

    def _get_tip_geom_pose(self, arm, data=None):
        data = self.data if data is None else data
        geom = data.geom(arm.tip_geom_id)
        pos = np.asarray(geom.xpos, dtype=np.float64).copy()
        rot = np.asarray(geom.xmat, dtype=np.float64).reshape(3, 3).copy()
        return pos, rot

    def get_tip_pos(self, arm):
        pos, _ = self._get_tip_geom_pose(arm)
        return pos

    def get_tip_pose(self, arm, data=None):
        return self._get_tip_geom_pose(arm, data=data)

    def get_hand_pose(self, arm, data=None):
        data = self.data if data is None else data
        pos = data.body(arm.body_id).xpos.copy()
        rot = data.body(arm.body_id).xmat.reshape(3, 3).copy()
        return pos, rot

    def world_pose_to_arm_frame(self, arm, pos_world, rot_world):
        pos_world = np.asarray(pos_world, dtype=np.float64).reshape(3)
        rot_world = _project_to_rotation_matrix(rot_world)
        pos_local = arm.base_rot.T @ (pos_world - arm.base_pos)
        rot_local = arm.base_rot.T @ rot_world
        return pos_local, _project_to_rotation_matrix(rot_local)

    def arm_pose_to_world_frame(self, arm, pos_local, rot_local):
        pos_local = np.asarray(pos_local, dtype=np.float64).reshape(3)
        rot_local = _project_to_rotation_matrix(rot_local)
        pos_world = arm.base_pos + arm.base_rot @ pos_local
        rot_world = arm.base_rot @ rot_local
        return pos_world, _project_to_rotation_matrix(rot_world)

    def tip_target_to_hand_pose(self, tip_pos_world, tip_rot_world):
        tip_pos_world = np.asarray(tip_pos_world, dtype=np.float64).reshape(3)
        tip_rot_world = _project_to_rotation_matrix(tip_rot_world)
        hand_pos_world = tip_pos_world - tip_rot_world[:, 2] * TIP_CENTER_OFFSET
        return hand_pos_world, tip_rot_world

    def world_to_object(self, world_point):
        obj_pos, _, obj_rot = self.get_object_pose()
        return obj_rot.T @ (np.asarray(world_point, dtype=np.float64) - obj_pos)

    @staticmethod
    def _build_contact_rotation(approach_dir, tangent_hint=None, positive_y_hint=None):
        z_axis = _normalize(approach_dir)
        if np.linalg.norm(z_axis) < 1e-8:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        candidate_hints = []
        if tangent_hint is not None:
            candidate_hints.append(np.asarray(tangent_hint, dtype=np.float64))
        candidate_hints.extend(
            [
                WORLD_UP,
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            ]
        )

        x_axis = None
        for hint in candidate_hints:
            tangent = _project_to_plane(hint, z_axis)
            if np.linalg.norm(tangent) > 1e-6:
                x_axis = _normalize(tangent)
                break
        if x_axis is None:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        y_axis = _normalize(np.cross(z_axis, x_axis))
        if np.linalg.norm(y_axis) < 1e-8:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        if positive_y_hint is not None:
            positive_y_hint = _project_to_plane(
                np.asarray(positive_y_hint, dtype=np.float64), z_axis
            )
            if np.linalg.norm(positive_y_hint) > 1e-6:
                positive_y_hint = _normalize(positive_y_hint)
                # Use the free yaw around the fingertip z-axis to keep the object on
                # the local +y side, so the arm presses from +y toward -y instead of
                # contacting from the local -y side.
                if float(np.dot(y_axis, positive_y_hint)) < 0.0:
                    x_axis = -x_axis
                    y_axis = -y_axis

        x_axis = _normalize(np.cross(y_axis, z_axis))
        return _project_to_rotation_matrix(np.column_stack([x_axis, y_axis, z_axis]))

    @staticmethod
    def _max_normal_force(contact_items):
        if not contact_items:
            return 0.0
        return float(
            max(float(item.get("normal_force", 0.0)) for item in contact_items)
        )

    def _ordered_contact_data(
        self,
        contact_points_local,
        normals_local,
        object_pos,
        object_rot,
        reference_points_world=None,
        return_order=False,
    ):
        contact_points_local = np.asarray(
            contact_points_local, dtype=np.float64
        ).reshape(-1, 3)
        normals_local = np.asarray(normals_local, dtype=np.float64).reshape(-1, 3)
        contact_points_world = (object_rot @ contact_points_local.T).T + object_pos[
            None, :
        ]
        if reference_points_world is None:
            left_ref = self.get_tip_pos(self.left_arm)
            right_ref = self.get_tip_pos(self.right_arm)
        else:
            reference_points_world = np.asarray(
                reference_points_world, dtype=np.float64
            ).reshape(-1, 3)
            if reference_points_world.shape[0] != 2:
                raise ValueError(
                    f"Expected 2 reference points for contact ordering, got {reference_points_world.shape[0]}."
                )
            left_ref = reference_points_world[0]
            right_ref = reference_points_world[1]

        keep_cost = np.linalg.norm(left_ref - contact_points_world[0]) + np.linalg.norm(
            right_ref - contact_points_world[1]
        )
        swap_cost = np.linalg.norm(left_ref - contact_points_world[1]) + np.linalg.norm(
            right_ref - contact_points_world[0]
        )
        order = (
            np.array([0, 1], dtype=int)
            if keep_cost <= swap_cost
            else np.array([1, 0], dtype=int)
        )
        ordered = (
            contact_points_local[order],
            normals_local[order],
            contact_points_world[order],
        )
        if return_order:
            return (*ordered, order)
        return ordered

    def _compute_preferred_tip_approach(self, inward_normal_world):
        downward_dir = -WORLD_UP
        inward_normal_world = _normalize(inward_normal_world)
        max_pitch = min(float(self.args.fingertip_max_pitch), np.pi * 0.49)
        return _rotate_vector_toward(downward_dir, inward_normal_world, max_pitch)

    def _compute_fingertip_targets(
        self, contact_points_world, inward_normals_world, center_offset
    ):
        contact_points_world = np.asarray(
            contact_points_world, dtype=np.float64
        ).reshape(-1, 3)
        inward_normals_world = np.asarray(
            inward_normals_world, dtype=np.float64
        ).reshape(-1, 3)
        if contact_points_world.shape[0] != 2:
            raise ValueError(
                f"Expected exactly 2 contact points, got {contact_points_world.shape[0]}."
            )

        pair_axis = contact_points_world[1] - contact_points_world[0]
        if np.linalg.norm(pair_axis) < 1e-8:
            pair_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        pair_axis = _normalize(pair_axis)

        outward_normals_world = -inward_normals_world
        left_pos = contact_points_world[0] + outward_normals_world[0] * float(
            center_offset
        )
        right_pos = contact_points_world[1] + outward_normals_world[1] * float(
            center_offset
        )
        # Bias the fingertip to point mostly downward, while allowing a limited
        # inward pitch toward the contact normal to make the IK target easier to reach.
        left_approach = self._compute_preferred_tip_approach(inward_normals_world[0])
        right_approach = self._compute_preferred_tip_approach(inward_normals_world[1])
        left_rot = self._build_contact_rotation(
            left_approach,
            tangent_hint=pair_axis,
            positive_y_hint=inward_normals_world[0],
        )
        right_rot = self._build_contact_rotation(
            right_approach,
            tangent_hint=-pair_axis,
            positive_y_hint=inward_normals_world[1],
        )
        return (left_pos, left_rot), (right_pos, right_rot)

    def _targets_from_object_pose(
        self,
        contact_points_local,
        normals_local,
        object_pos,
        object_rot,
        center_offset,
    ):
        contact_points_world = (
            object_rot @ np.asarray(contact_points_local, dtype=np.float64).T
        ).T + object_pos[None, :]
        inward_normals_world = (
            object_rot @ np.asarray(normals_local, dtype=np.float64).T
        ).T
        return self._compute_fingertip_targets(
            contact_points_world, inward_normals_world, center_offset
        )

    def _stage_targets_from_object_pose(
        self,
        contact_points_local,
        normals_local,
        object_pos,
        object_rot,
        center_offset,
    ):
        contact_points_world = (
            object_rot @ np.asarray(contact_points_local, dtype=np.float64).T
        ).T + object_pos[None, :]
        inward_normals_world = (
            object_rot @ np.asarray(normals_local, dtype=np.float64).T
        ).T
        outward_normals_world = -inward_normals_world
        (left_pos, left_rot), (right_pos, right_rot) = self._compute_fingertip_targets(
            contact_points_world,
            inward_normals_world,
            center_offset,
        )
        return {
            "left_tip_pos": np.asarray(left_pos, dtype=np.float64),
            "left_tip_rot": _project_to_rotation_matrix(left_rot),
            "right_tip_pos": np.asarray(right_pos, dtype=np.float64),
            "right_tip_rot": _project_to_rotation_matrix(right_rot),
            "left_outward_normal": _normalize(outward_normals_world[0]),
            "right_outward_normal": _normalize(outward_normals_world[1]),
            "object_target_pos": np.asarray(object_pos, dtype=np.float64).copy(),
            "object_target_quat": mat_to_quat_wxyz(object_rot),
        }

    @staticmethod
    def _offset_points_along_normals(points_world, outward_normals_world, offset):
        points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        outward_normals_world = np.asarray(
            outward_normals_world, dtype=np.float64
        ).reshape(-1, 3)
        return points_world + float(offset) * np.vstack(
            [_normalize(normal) for normal in outward_normals_world]
        )

    @staticmethod
    def _contact_dissipation_factor(distance_rate, dissipation_velocity):
        dissipation_velocity = max(float(dissipation_velocity), 1e-9)
        s = float(distance_rate) / dissipation_velocity
        if s < 0.0:
            return 1.0 - s
        if s < 2.0:
            return 0.25 * (s - 2.0) * (s - 2.0)
        return 0.0

    @staticmethod
    def _compliant_normal_force(distance, stiffness, smoothing_factor):
        distance = float(distance)
        stiffness = max(float(stiffness), 1e-9)
        smoothing_factor = max(float(smoothing_factor), 0.0)
        if smoothing_factor <= 1e-9:
            return max(-stiffness * distance, 0.0)

        exponent = -distance / smoothing_factor
        if exponent >= 37.0:
            return max(-stiffness * distance, 0.0)
        return float(smoothing_factor * stiffness * np.log1p(np.exp(exponent)))

    @staticmethod
    def _distance_from_normal_force(target_force, stiffness, smoothing_factor):
        target_force = max(float(target_force), 0.0)
        stiffness = max(float(stiffness), 1e-9)
        smoothing_factor = max(float(smoothing_factor), 0.0)
        if target_force <= 1e-9:
            return 0.0
        if smoothing_factor <= 1e-9:
            return -target_force / stiffness

        scaled_force = target_force / (smoothing_factor * stiffness)
        if scaled_force >= 37.0:
            return -target_force / stiffness
        return float(-smoothing_factor * np.log(np.expm1(scaled_force)))

    def _build_force_control_targets(
        self,
        contact_points_world,
        outward_normals_world,
        target_normal_forces,
        previous_state=None,
        contact_stiffness=None,
        dissipation_velocity=None,
        stiction_velocity=None,
        smoothing_factor=None,
    ):
        contact_points_world = np.asarray(
            contact_points_world, dtype=np.float64
        ).reshape(2, 3)
        outward_normals_world = np.asarray(
            outward_normals_world, dtype=np.float64
        ).reshape(2, 3)
        target_normal_forces = np.asarray(
            target_normal_forces, dtype=np.float64
        ).reshape(2)

        default_force_control_stiffness = getattr(
            self.args, "force_control_stiffness", None
        )
        contact_stiffness = (
            float(default_force_control_stiffness)
            if contact_stiffness is None and default_force_control_stiffness is not None
            else contact_stiffness
        )
        contact_stiffness = max(
            float(1.0 if contact_stiffness is None else contact_stiffness), 1e-9
        )
        dissipation_velocity = float(
            getattr(self.args, "force_control_dissipation_velocity", 0.1)
            if dissipation_velocity is None
            else dissipation_velocity
        )
        stiction_velocity = float(
            getattr(self.args, "force_control_stiction_velocity", 0.05)
            if stiction_velocity is None
            else stiction_velocity
        )
        smoothing_factor = float(
            getattr(self.args, "force_control_smoothing", 0.0)
            if smoothing_factor is None
            else smoothing_factor
        )

        current_tip_positions = np.vstack(
            [
                self.get_tip_pos(self.left_arm),
                self.get_tip_pos(self.right_arm),
            ]
        )
        if previous_state is None:
            previous_tip_positions = current_tip_positions.copy()
            previous_contact_points_world = contact_points_world.copy()
        else:
            previous_tip_positions = np.asarray(
                previous_state.get("tip_positions_world", current_tip_positions),
                dtype=np.float64,
            ).reshape(2, 3)
            previous_contact_points_world = np.asarray(
                previous_state.get("contact_points_world", contact_points_world),
                dtype=np.float64,
            ).reshape(2, 3)

        dt = max(float(self.command_dt), float(self.model.opt.timestep), 1e-6)
        goal_points_world = np.zeros((2, 3), dtype=np.float64)
        desired_force_world = np.zeros((2, 3), dtype=np.float64)
        modeled_force_world = np.zeros((2, 3), dtype=np.float64)
        modeled_normal_forces = np.zeros(2, dtype=np.float64)
        modeled_tangential_forces = np.zeros(2, dtype=np.float64)
        modeled_distance = np.zeros(2, dtype=np.float64)
        modeled_compression = np.zeros(2, dtype=np.float64)
        modeled_distance_rate = np.zeros(2, dtype=np.float64)
        desired_offsets = np.zeros(2, dtype=np.float64)

        for idx in range(2):
            outward_normal = _normalize(outward_normals_world[idx])
            if np.linalg.norm(outward_normal) < 1e-9:
                outward_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            current_tip_pos = current_tip_positions[idx]
            previous_tip_pos = previous_tip_positions[idx]
            contact_point_world = contact_points_world[idx]
            previous_contact_point_world = previous_contact_points_world[idx]

            tip_velocity = (current_tip_pos - previous_tip_pos) / dt
            contact_point_velocity = (
                contact_point_world - previous_contact_point_world
            ) / dt
            relative_velocity = tip_velocity - contact_point_velocity

            distance = float(
                np.dot(current_tip_pos - contact_point_world, outward_normal)
                - TIP_RADIUS
            )
            distance_rate = float(np.dot(relative_velocity, outward_normal))
            dissipation_factor = self._contact_dissipation_factor(
                distance_rate, dissipation_velocity
            )
            compliant_force = self._compliant_normal_force(
                distance, contact_stiffness, smoothing_factor
            )
            normal_force = float(compliant_force * dissipation_factor)

            tangential_velocity = relative_velocity - distance_rate * outward_normal
            tangential_speed = float(np.linalg.norm(tangential_velocity))
            regularized_speed = np.sqrt(
                max(stiction_velocity, 0.0) ** 2 + tangential_speed**2
            )
            if regularized_speed > 1e-12:
                tangential_force_world = (
                    (tangential_velocity / regularized_speed)
                    * float(self.args.arm_friction)
                    * normal_force
                )
            else:
                tangential_force_world = np.zeros(3, dtype=np.float64)

            target_distance = self._distance_from_normal_force(
                target_normal_forces[idx],
                contact_stiffness,
                smoothing_factor,
            )
            target_distance = min(float(target_distance), 0.0)
            desired_offset = float(
                np.clip(TIP_RADIUS + target_distance, 0.001, TIP_RADIUS)
            )

            desired_force_world[idx] = -outward_normal * float(
                target_normal_forces[idx]
            )
            modeled_force_world[idx] = (
                -outward_normal * normal_force + tangential_force_world
            )
            modeled_normal_forces[idx] = normal_force
            modeled_tangential_forces[idx] = float(
                np.linalg.norm(tangential_force_world)
            )
            modeled_distance[idx] = distance
            modeled_compression[idx] = max(-distance, 0.0)
            modeled_distance_rate[idx] = distance_rate
            desired_offsets[idx] = desired_offset
            goal_points_world[idx] = (
                contact_point_world + outward_normal * desired_offset
            )

        next_state = {
            "tip_positions_world": current_tip_positions.copy(),
            "contact_points_world": contact_points_world.copy(),
        }
        debug = {
            "force_target_points_world": goal_points_world.copy(),
            "desired_contact_force_world": desired_force_world.copy(),
            "modeled_contact_force_world": modeled_force_world.copy(),
            "desired_normal_forces": target_normal_forces.copy(),
            "modeled_normal_forces": modeled_normal_forces.copy(),
            "modeled_tangential_forces": modeled_tangential_forces.copy(),
            "modeled_contact_distance": modeled_distance.copy(),
            "modeled_contact_compression": modeled_compression.copy(),
            "modeled_contact_distance_rate": modeled_distance_rate.copy(),
            "force_target_offsets": desired_offsets.copy(),
            "force_control_stiffness": float(contact_stiffness),
            "force_control_dissipation_velocity": float(dissipation_velocity),
            "force_control_stiction_velocity": float(stiction_velocity),
            "force_control_smoothing": float(smoothing_factor),
        }
        return goal_points_world, debug, next_state

    def _copy_nested_value(self, value):
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: self._copy_nested_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._copy_nested_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._copy_nested_value(item) for item in value)
        return copy.deepcopy(value)

    def _copy_contact_targets(self, targets):
        if targets is None:
            return {}
        return {
            key: self._copy_nested_value(value) for key, value in dict(targets).items()
        }

    def _optimizer_support_kwargs(self):
        if not bool(self.args.optimizer_use_support_filter):
            return {}
        return {
            "support_surface_point": self.support_surface_point,
            "support_surface_normal": self.support_surface_normal,
            "support_surface_clearance": float(self.args.ground_height_margin),
            "support_surface_normal_alignment_threshold": float(
                self.args.support_normal_alignment_threshold
            ),
        }

    def _gravity_wrench_local(self, object_rot=None):
        if object_rot is None:
            _, _, object_rot = self.get_object_pose()
        object_rot = _project_to_rotation_matrix(object_rot)
        gravity_force_world = np.array(
            [0.0, 0.0, -float(self.args.obj_mass) * 9.81], dtype=np.float64
        )
        gravity_force_local = object_rot.T @ gravity_force_world
        return np.hstack([gravity_force_local, np.zeros(3, dtype=np.float64)])

    def _best_object_wrench_world(self, optimizer_result, object_rot):
        if optimizer_result is None:
            return None, None

        object_rot = _project_to_rotation_matrix(object_rot)
        wrench_source = optimizer_result
        static_result = None
        if isinstance(optimizer_result, dict):
            static_result = optimizer_result.get("static_equilibrium", None)
        if isinstance(static_result, dict) and bool(static_result.get("valid", False)):
            wrench_source = static_result

        wrench_info = self.optimizer._result_contact_wrench(
            wrench_source,
            r_obj_to_world=object_rot,
        )
        force_world = np.asarray(
            wrench_info.get("contact_force_world", np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3)
        torque_world = np.asarray(
            wrench_info.get("contact_torque_world", np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3)
        return force_world, torque_world

    def _compute_lift_cost(self, static_result):
        if static_result is None:
            return 0.0

        lift_cost_weight = float(self.args.grasp_lift_cost_weight)
        lift_residual_weight = float(self.args.grasp_lift_residual_weight)
        if lift_cost_weight <= 0.0 and lift_residual_weight <= 0.0:
            return 0.0

        static_cost = float(static_result.get("cost", float("inf")))
        scaled_residual_norm = float(
            static_result.get("scaled_residual_norm", float("inf"))
        )
        if not np.isfinite(static_cost) or not np.isfinite(scaled_residual_norm):
            return float("inf")

        return (
            lift_cost_weight * static_cost + lift_residual_weight * scaled_residual_norm
        )

    def _precompute_contact_candidate_cache(
        self, object_pos, object_rot, external_wrench_local
    ):
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        visible_idx = self._visible_optimizer_point_indices(object_pos, object_rot)
        candidate_limit = int(getattr(self.args, "cached_contact_candidate_limit", 10))

        cache = self.optimizer.precompute_contact_search_cache(
            visible_face_idx=visible_idx,
            object_pos=object_pos,
            object_rot=object_rot,
            top_candidate_count=(candidate_limit if candidate_limit > 0 else None),
            **self._optimizer_support_kwargs(),
        )

        candidate_entries = list(cache.get("candidate_entries", []))
        lift_precompute_t0 = time.perf_counter()
        for entry in candidate_entries:
            entry["offline_lift_cost"] = 0.0
            contact_indices = np.asarray(
                entry.get("contact_indices", []), dtype=int
            ).reshape(-1)
            if contact_indices.size == 0:
                continue

            static_result = self.optimizer.solve_static_equilibrium(
                contact_indices,
                external_wrench=external_wrench_local,
            )
            entry["offline_static_equilibrium"] = self._copy_nested_value(static_result)
            entry["offline_lift_cost"] = float(self._compute_lift_cost(static_result))
            entry["offline_total_cost"] = float(
                float(entry.get("offline_force_closure_total_cost", float("inf")))
                + float(entry["offline_lift_cost"])
            )

        candidate_entries.sort(
            key=lambda entry: (
                float(entry.get("offline_total_cost", float("inf"))),
                -float(
                    entry.get(
                        "offline_region_score",
                        entry.get("region_group", {}).get("stability_score", 0.0),
                    )
                ),
                -float(entry.get("offline_min_contact_distance", 0.0)),
            )
        )

        cache["candidate_entries"] = candidate_entries
        cache["visible_face_idx"] = np.asarray(visible_idx, dtype=int).copy()
        cache["offline_lift_precompute_time"] = float(
            time.perf_counter() - lift_precompute_t0
        )
        cache["candidate_entry_count_before_limit"] = int(
            cache.get("candidate_entry_count_before_limit", len(candidate_entries))
        )
        cache["candidate_entry_count"] = int(len(candidate_entries))
        return cache

    def _evaluate_best_cached_grasp(self, candidate_cache, object_rot):
        if candidate_cache is None:
            return None

        candidate_entries = list(candidate_cache.get("candidate_entries", []))
        if not candidate_entries:
            self.optimizer.last_grasp_result = None
            return None

        object_rot = _project_to_rotation_matrix(object_rot)
        eval_t0 = time.perf_counter()
        force_closure_solve_time = 0.0
        evaluated_candidates = []
        support_kwargs = self._optimizer_support_kwargs()

        for entry in candidate_entries:
            result = self.optimizer._evaluate_force_closure_candidate(
                entry["contact_indices"],
                region_group=entry.get("region_group", None),
                object_rot=object_rot,
                **support_kwargs,
            )
            force_closure_solve_time += float(result.get("solve_time", 0.0))

            force_closure_total_cost = float(result.get("total_cost", float("inf")))
            offline_lift_cost = float(entry.get("offline_lift_cost", 0.0))
            result["base_total_cost"] = float(force_closure_total_cost)
            result["force_closure_total_cost"] = float(force_closure_total_cost)
            result["lift_cost"] = float(offline_lift_cost)
            result["offline_lift_cost"] = float(offline_lift_cost)
            result["total_cost"] = float(force_closure_total_cost + offline_lift_cost)
            result["valid"] = bool(
                result.get("valid", True) and np.isfinite(result["total_cost"])
            )

            static_result = entry.get("offline_static_equilibrium", None)
            if static_result is not None:
                copied_static_result = self._copy_nested_value(static_result)
                result["static_equilibrium"] = copied_static_result
                result["static_equilibrium_cost"] = float(
                    copied_static_result.get("cost", float("inf"))
                )
                result["static_equilibrium_residual_norm"] = float(
                    copied_static_result.get("residual_norm", float("inf"))
                )
                result["static_equilibrium_scaled_residual_norm"] = float(
                    copied_static_result.get("scaled_residual_norm", float("inf"))
                )

            evaluated_candidates.append(result)

        if not evaluated_candidates:
            self.optimizer.last_grasp_result = None
            return None

        finite_total_costs = [
            float(item["total_cost"])
            for item in evaluated_candidates
            if np.isfinite(float(item.get("total_cost", float("inf"))))
        ]

        if finite_total_costs:
            best_result = min(
                evaluated_candidates,
                key=lambda item: (
                    item["total_cost"],
                    -item.get("region_score", 0.0),
                    -item["min_contact_distance"],
                ),
            )
            evaluated_total_cost_min = float(np.min(finite_total_costs))
            evaluated_total_cost_max = float(np.max(finite_total_costs))
        else:
            best_result = min(
                evaluated_candidates,
                key=lambda item: (
                    item.get("force_closure_total_cost", float("inf")),
                    -item.get("region_score", 0.0),
                    -item["min_contact_distance"],
                ),
            )
            fallback_costs = [
                float(item.get("force_closure_total_cost", float("inf")))
                for item in evaluated_candidates
                if np.isfinite(
                    float(item.get("force_closure_total_cost", float("inf")))
                )
            ]
            if fallback_costs:
                evaluated_total_cost_min = float(np.min(fallback_costs))
                evaluated_total_cost_max = float(np.max(fallback_costs))
            else:
                evaluated_total_cost_min = float("inf")
                evaluated_total_cost_max = float("inf")

        wall_time = float(time.perf_counter() - eval_t0)
        best_result["evaluated_candidate_count"] = int(len(evaluated_candidates))
        best_result["candidate_cache_used"] = True
        best_result["top_region_pairs"] = self._copy_nested_value(
            candidate_cache.get("region_groups", [])
        )
        best_result["evaluated_total_cost_min"] = float(evaluated_total_cost_min)
        best_result["evaluated_total_cost_max"] = float(evaluated_total_cost_max)
        best_result["wall_time"] = wall_time
        best_result["search_timing"] = {
            "mode": "precomputed_cache_force_closure_only",
            "candidate_entry_count_before_limit": int(
                candidate_cache.get(
                    "candidate_entry_count_before_limit", len(candidate_entries)
                )
            ),
            "candidate_entry_count": int(len(candidate_entries)),
            "evaluate_candidates_time": wall_time,
            "force_closure_solve_time": float(force_closure_solve_time),
            "wall_time": wall_time,
        }
        self.optimizer.last_grasp_result = best_result
        return best_result

    def _build_contact_targets_from_local_contacts(
        self,
        contact_points_local,
        normals_local,
        object_pos,
        object_rot,
        result=None,
        previous_targets=None,
        virtual_offset=None,
        source="best",
        projection_vertex_indices=None,
    ):
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        virtual_offset = float(
            self.args.planner_attract_offset
            if virtual_offset is None
            else virtual_offset
        )

        contact_points_local = np.asarray(
            contact_points_local, dtype=np.float64
        ).reshape(2, 3)
        normals_local = np.asarray(normals_local, dtype=np.float64).reshape(2, 3)
        raw_contact_points_local = contact_points_local.copy()
        raw_normals_local = normals_local.copy()

        reference_points_world = None
        if previous_targets is not None:
            prev_world = np.asarray(
                previous_targets.get("contact_points_world", []), dtype=np.float64
            ).reshape(-1, 3)
            if prev_world.shape[0] == 2:
                reference_points_world = prev_world

        (
            contact_points_local,
            normals_local,
            contact_points_world,
            order,
        ) = self._ordered_contact_data(
            contact_points_local,
            normals_local,
            object_pos,
            object_rot,
            reference_points_world=reference_points_world,
            return_order=True,
        )
        inward_normals_world = (object_rot @ normals_local.T).T
        outward_normals_world = -inward_normals_world
        virtual_points_world = self._offset_points_along_normals(
            contact_points_world,
            outward_normals_world,
            virtual_offset,
        )

        contact_indices = np.zeros((0,), dtype=int)
        witness_contact_forces_local = np.zeros((0, 3), dtype=np.float64)
        witness_force_vectors_local = np.zeros((0, 3), dtype=np.float64)
        desired_contact_forces_local = np.zeros((0, 3), dtype=np.float64)
        desired_force_vectors_local = np.zeros((0, 3), dtype=np.float64)
        base_total_cost = float("inf")
        total_cost = float("inf")
        force_closure_cost = float("inf")
        lift_cost = 0.0
        region_score = 0.0
        antipodal_margin = 0.0
        static_result = None

        if result is not None:
            contact_indices = np.asarray(
                result.get("contact_indices", []), dtype=int
            ).reshape(-1)
            if contact_indices.shape[0] == order.shape[0]:
                contact_indices = contact_indices[order]

            witness_contact_forces_local = np.asarray(
                result.get(
                    "witness_contact_forces_local", np.zeros((0, 3), dtype=np.float64)
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            if witness_contact_forces_local.shape[0] == order.shape[0]:
                witness_contact_forces_local = witness_contact_forces_local[order]

            witness_force_vectors_local = np.asarray(
                result.get(
                    "witness_force_vectors_local", np.zeros((0, 3), dtype=np.float64)
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            if witness_force_vectors_local.shape[0] == order.shape[0]:
                witness_force_vectors_local = witness_force_vectors_local[order]

            static_result = result.get("static_equilibrium", None)
            if static_result is not None:
                desired_contact_forces_local = np.asarray(
                    static_result.get(
                        "contact_forces_local", np.zeros((0, 3), dtype=np.float64)
                    ),
                    dtype=np.float64,
                ).reshape(-1, 3)
                if desired_contact_forces_local.shape[0] == order.shape[0]:
                    desired_contact_forces_local = desired_contact_forces_local[order]

                desired_force_vectors_local = np.asarray(
                    static_result.get(
                        "force_vectors_local", np.zeros((0, 3), dtype=np.float64)
                    ),
                    dtype=np.float64,
                ).reshape(-1, 3)
                if desired_force_vectors_local.shape[0] == order.shape[0]:
                    desired_force_vectors_local = desired_force_vectors_local[order]

            if desired_contact_forces_local.shape[0] != 2:
                desired_contact_forces_local = witness_contact_forces_local.copy()
            if desired_force_vectors_local.shape[0] != 2:
                desired_force_vectors_local = witness_force_vectors_local.copy()

            base_total_cost = float(
                result.get("base_total_cost", result.get("total_cost", float("inf")))
            )
            total_cost = float(result.get("total_cost", base_total_cost))
            force_closure_cost = float(result.get("force_closure_cost", float("inf")))
            lift_cost = float(result.get("lift_cost", 0.0))
            region_score = float(result.get("region_score", 0.0))
            antipodal_margin = float(result.get("antipodal_margin", 0.0))

        targets = {
            "target_source": str(source),
            "contact_points_local": contact_points_local.copy(),
            "normals_local": normals_local.copy(),
            "contact_points_world": contact_points_world.copy(),
            "inward_normals_world": inward_normals_world.copy(),
            "outward_normals_world": outward_normals_world.copy(),
            "virtual_points_world": virtual_points_world.copy(),
            "left_virtual_point": virtual_points_world[0].copy(),
            "right_virtual_point": virtual_points_world[1].copy(),
            "object_pos": object_pos.copy(),
            "object_quat": mat_to_quat_wxyz(object_rot),
            "contact_indices": contact_indices.copy(),
            "raw_contact_points_local": raw_contact_points_local.copy(),
            "raw_normals_local": raw_normals_local.copy(),
            "contact_order": order.copy(),
            "witness_contact_forces_local": witness_contact_forces_local.copy(),
            "witness_force_vectors_local": witness_force_vectors_local.copy(),
            "desired_contact_forces_local": desired_contact_forces_local.copy(),
            "desired_force_vectors_local": desired_force_vectors_local.copy(),
            "base_total_cost": float(base_total_cost),
            "total_cost": float(total_cost),
            "force_closure_cost": float(force_closure_cost),
            "lift_cost": float(lift_cost),
            "region_score": float(region_score),
            "antipodal_margin": float(antipodal_margin),
        }

        if projection_vertex_indices is not None:
            projection_vertex_indices = np.asarray(
                projection_vertex_indices, dtype=int
            ).reshape(-1)
            if projection_vertex_indices.shape[0] == order.shape[0]:
                projection_vertex_indices = projection_vertex_indices[order]
            targets["projection_vertex_indices"] = projection_vertex_indices.copy()

        if result is not None:
            targets["search_timing"] = self._copy_nested_value(
                result.get("search_timing", {})
            )
            if static_result is not None:
                targets["static_equilibrium"] = self._copy_nested_value(static_result)

        if desired_force_vectors_local.shape[0] == 2:
            targets["desired_contact_force_world"] = (
                object_rot @ desired_force_vectors_local.T
            ).T
        if witness_force_vectors_local.shape[0] == 2:
            targets["witness_contact_force_world"] = (
                object_rot @ witness_force_vectors_local.T
            ).T

        return targets

    def _project_current_tip_contacts(self, object_pos, object_rot):
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)

        tip_positions_world = np.vstack(
            [
                self.get_tip_pos(self.left_arm),
                self.get_tip_pos(self.right_arm),
            ]
        )
        tip_positions_local = (object_rot.T @ (tip_positions_world - object_pos).T).T

        contact_points_local = np.zeros((2, 3), dtype=np.float64)
        normals_local = np.zeros((2, 3), dtype=np.float64)
        projection_vertex_indices = np.zeros((2,), dtype=int)

        for idx in range(2):
            (
                vertex_idx,
                inward_normal,
                _t1,
                _t2,
            ) = self.optimizer.pp.project_point_to_mesh(tip_positions_local[idx])
            projection_vertex_indices[idx] = int(vertex_idx)
            contact_points_local[idx] = np.asarray(
                self.optimizer.pp.scaled_mesh.vertices[int(vertex_idx)],
                dtype=np.float64,
            ).reshape(3)
            normals_local[idx] = _normalize(inward_normal)

        return (
            contact_points_local,
            normals_local,
            tip_positions_local,
            projection_vertex_indices,
        )

    def _evaluate_nearest_tip_grasp(
        self, object_pos, object_rot, previous_targets=None, virtual_offset=None
    ):
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        gravity_local = self._gravity_wrench_local(object_rot)
        (
            contact_points_local,
            normals_local,
            tip_positions_local,
            projection_vertex_indices,
        ) = self._project_current_tip_contacts(
            object_pos,
            object_rot,
        )

        eval_t0 = time.perf_counter()
        result = self.optimizer.optimize_multi_contact_input(contact_points_local)
        static_result = self.optimizer.solve_static_equilibrium(
            result, external_wrench=gravity_local
        )
        result["base_total_cost"] = float(result.get("total_cost", float("inf")))
        result["lift_cost"] = float(self._compute_lift_cost(static_result))
        result["total_cost"] = float(result["base_total_cost"] + result["lift_cost"])
        result["valid"] = bool(
            result.get("valid", True) and np.isfinite(result["total_cost"])
        )
        result["static_equilibrium"] = self._copy_nested_value(static_result)
        result["static_equilibrium_cost"] = float(
            static_result.get("cost", float("inf"))
        )
        result["static_equilibrium_residual_norm"] = float(
            static_result.get("residual_norm", float("inf"))
        )
        result["static_equilibrium_scaled_residual_norm"] = float(
            static_result.get("scaled_residual_norm", float("inf"))
        )
        result["wall_time"] = float(time.perf_counter() - eval_t0)
        result["search_timing"] = {
            "mode": "nearest_projection_pair",
            "candidate_entry_count": 1,
            "evaluate_candidates_time": float(result["wall_time"]),
            "force_closure_solve_time": float(result.get("solve_time", 0.0)),
            "static_equilibrium_solve_time": float(
                static_result.get("solve_time", 0.0)
            ),
            "wall_time": float(result["wall_time"]),
        }

        targets = self._build_contact_targets_from_local_contacts(
            contact_points_local,
            normals_local,
            object_pos,
            object_rot,
            result=result,
            previous_targets=previous_targets,
            virtual_offset=virtual_offset,
            source="nearest",
            projection_vertex_indices=projection_vertex_indices,
        )
        targets["tip_points_local"] = np.asarray(
            tip_positions_local, dtype=np.float64
        ).copy()
        return targets, result

    def _update_contact_selection_mode(
        self, state, best_valid, nearest_valid, improvement
    ):
        current_mode = str(state.get("mode", "best"))
        low_improvement = float(self.args.nearest_contact_low_improvement)
        high_improvement = max(
            low_improvement, float(self.args.nearest_contact_high_improvement)
        )
        switch_steps = max(int(self.args.nearest_contact_switch_steps), 1)

        if not nearest_valid:
            state["mode"] = "best"
            state["pending_mode"] = None
            state["pending_steps"] = 0
            return state["mode"], False
        if not best_valid:
            state["mode"] = "nearest"
            state["pending_mode"] = None
            state["pending_steps"] = 0
            return state["mode"], True

        if current_mode == "nearest":
            desired_mode = "best" if improvement <= low_improvement else "nearest"
        else:
            desired_mode = "nearest" if improvement >= high_improvement else "best"

        switched = False
        if desired_mode == current_mode:
            state["pending_mode"] = None
            state["pending_steps"] = 0
            return current_mode, switched

        pending_mode = state.get("pending_mode", None)
        pending_steps = int(state.get("pending_steps", 0))
        if pending_mode != desired_mode:
            pending_mode = desired_mode
            pending_steps = 1
        else:
            pending_steps += 1

        if pending_steps >= switch_steps:
            current_mode = desired_mode
            pending_mode = None
            pending_steps = 0
            switched = True

        state["mode"] = current_mode
        state["pending_mode"] = pending_mode
        state["pending_steps"] = pending_steps
        return current_mode, switched

    def _select_live_contact_targets(
        self,
        object_pos,
        object_rot,
        candidate_cache=None,
        previous_targets=None,
        virtual_offset=None,
        selection_state=None,
    ):
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        selection_state = {} if selection_state is None else selection_state
        selection_state.setdefault("mode", "best")
        selection_state.setdefault("pending_mode", None)
        selection_state.setdefault("pending_steps", 0)

        best_result = (
            self._evaluate_best_cached_grasp(candidate_cache, object_rot)
            if candidate_cache is not None
            else None
        )
        best_targets = None
        if best_result is not None:
            best_targets = self._build_contact_targets_from_local_contacts(
                best_result["contact_points"],
                best_result["contact_normals"],
                object_pos,
                object_rot,
                result=best_result,
                previous_targets=previous_targets,
                virtual_offset=virtual_offset,
                source="best",
            )

        nearest_targets, nearest_result = self._evaluate_nearest_tip_grasp(
            object_pos,
            object_rot,
            previous_targets=previous_targets,
            virtual_offset=virtual_offset,
        )

        best_cost = (
            float("inf")
            if best_result is None
            else float(best_result.get("total_cost", float("inf")))
        )
        nearest_cost = float(nearest_result.get("total_cost", float("inf")))
        candidate_cost_min = float("inf")
        candidate_cost_max = float("inf")
        if best_result is not None:
            candidate_cost_min = float(
                best_result.get("evaluated_total_cost_min", best_cost)
            )
            candidate_cost_max = float(
                best_result.get("evaluated_total_cost_max", best_cost)
            )

        best_valid = bool(
            best_result is not None
            and best_result.get("valid", False)
            and np.isfinite(best_cost)
        )
        nearest_valid = bool(
            nearest_result.get("valid", False) and np.isfinite(nearest_cost)
        )
        if not np.isfinite(candidate_cost_min):
            candidate_cost_min = best_cost
        if not np.isfinite(candidate_cost_max):
            candidate_cost_max = best_cost

        cost_span = float(max(candidate_cost_max - candidate_cost_min, 0.0))
        if not nearest_valid:
            improvement = float("-inf")
        elif not best_valid:
            improvement = 1.0
        elif cost_span <= 1e-9:
            improvement = 1.0 if nearest_cost <= candidate_cost_min + 1e-9 else 0.0
        else:
            improvement = float((candidate_cost_max - nearest_cost) / cost_span)

        selected_mode, mode_switched = self._update_contact_selection_mode(
            selection_state,
            best_valid=best_valid,
            nearest_valid=nearest_valid,
            improvement=improvement,
        )
        if selected_mode == "nearest" or best_targets is None:
            selected_targets = nearest_targets
            selected_result = nearest_result
        else:
            selected_targets = best_targets
            selected_result = best_result

        if selected_result is not None:
            self.optimizer.last_grasp_result = selected_result
            static_result = selected_result.get("static_equilibrium", None)
            if static_result is not None:
                self.optimizer.last_static_equilibrium_result = static_result

        selected_search_timing = {}
        if selected_targets is not None:
            selected_search_timing = self._copy_nested_value(
                selected_targets.get("search_timing", {})
            )

        selection_debug = {
            "selected_contact_source": str(selected_mode),
            "contact_selection_mode": str(selection_state.get("mode", selected_mode)),
            "contact_selection_pending_mode": selection_state.get("pending_mode", None),
            "contact_selection_pending_steps": int(
                selection_state.get("pending_steps", 0)
            ),
            "contact_selection_switched": bool(mode_switched),
            "best_contact_cost": float(best_cost),
            "best_contact_force_closure_cost": float(
                best_result.get("force_closure_cost", float("inf"))
                if best_result is not None
                else float("inf")
            ),
            "best_contact_lift_cost": float(
                best_result.get("lift_cost", 0.0) if best_result is not None else 0.0
            ),
            "nearest_contact_cost": float(nearest_cost),
            "nearest_contact_force_closure_cost": float(
                nearest_result.get("force_closure_cost", float("inf"))
            ),
            "nearest_contact_lift_cost": float(nearest_result.get("lift_cost", 0.0)),
            "candidate_cost_min": float(candidate_cost_min),
            "candidate_cost_max": float(candidate_cost_max),
            "nearest_contact_improvement": float(improvement),
            "best_contact_points_world": np.asarray(
                best_targets["contact_points_world"]
                if best_targets is not None
                else np.zeros((0, 3), dtype=np.float64),
                dtype=np.float64,
            ).copy(),
            "nearest_contact_points_world": np.asarray(
                nearest_targets["contact_points_world"],
                dtype=np.float64,
            ).copy(),
            "best_contact_points_local": np.asarray(
                best_targets["contact_points_local"]
                if best_targets is not None
                else np.zeros((0, 3), dtype=np.float64),
                dtype=np.float64,
            ).copy(),
            "nearest_contact_points_local": np.asarray(
                nearest_targets["contact_points_local"],
                dtype=np.float64,
            ).copy(),
            "selected_search_timing": selected_search_timing,
        }
        return self._copy_contact_targets(selected_targets), selection_debug

    def _get_fixed_optimizer_regions(self, object_pos, object_rot):
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        visible_idx = self._visible_optimizer_point_indices(object_pos, object_rot)
        return self.optimizer.get_best_regions(
            visible_face_idx=visible_idx,
            top_k=self.optimizer.top_region_pairs,
            object_pos=object_pos,
            object_rot=object_rot,
        )

    def _get_live_contact_targets(
        self,
        object_pos,
        object_rot,
        previous_targets=None,
        virtual_offset=None,
        fixed_region_groups=None,
    ):
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        virtual_offset = float(
            self.args.planner_attract_offset
            if virtual_offset is None
            else virtual_offset
        )

        visible_idx = self._visible_optimizer_point_indices(object_pos, object_rot)
        (
            contact_points_local,
            normals_local,
            total_cost,
            region_score,
            antipodal_margin,
        ) = self.optimizer.choose_contact_set(
            visible_face_idx=visible_idx,
            object_pos=object_pos,
            object_rot=object_rot,
            fixed_region_groups=fixed_region_groups,
        )

        contact_points_local = np.asarray(
            contact_points_local, dtype=np.float64
        ).reshape(-1, 3)
        normals_local = np.asarray(normals_local, dtype=np.float64).reshape(-1, 3)
        if contact_points_local.shape[0] != 2 or normals_local.shape[0] != 2:
            if previous_targets is None:
                raise RuntimeError(
                    f"Expected 2 optimizer contacts, got {contact_points_local.shape[0]}"
                )
            contact_points_local = np.asarray(
                previous_targets["contact_points_local"], dtype=np.float64
            ).reshape(2, 3)
            normals_local = np.asarray(
                previous_targets["normals_local"], dtype=np.float64
            ).reshape(2, 3)
        raw_contact_points_local = contact_points_local.copy()
        raw_normals_local = normals_local.copy()

        reference_points_world = None
        if previous_targets is not None:
            prev_world = np.asarray(
                previous_targets.get("contact_points_world", []), dtype=np.float64
            ).reshape(-1, 3)
            if prev_world.shape[0] == 2:
                reference_points_world = prev_world

        (
            contact_points_local,
            normals_local,
            contact_points_world,
            order,
        ) = self._ordered_contact_data(
            contact_points_local,
            normals_local,
            object_pos,
            object_rot,
            reference_points_world=reference_points_world,
            return_order=True,
        )
        inward_normals_world = (object_rot @ normals_local.T).T
        outward_normals_world = -inward_normals_world
        virtual_points_world = self._offset_points_along_normals(
            contact_points_world,
            outward_normals_world,
            virtual_offset,
        )
        grasp_result = self.optimizer.last_grasp_result
        contact_indices = np.zeros((0,), dtype=int)
        raw_contact_indices = np.zeros((0,), dtype=int)
        witness_contact_forces_local = np.zeros((0, 3), dtype=np.float64)
        witness_force_vectors_local = np.zeros((0, 3), dtype=np.float64)
        if grasp_result is not None:
            contact_indices = np.asarray(
                grasp_result.get("contact_indices", []), dtype=int
            ).reshape(-1)
            raw_contact_indices = contact_indices.copy()
            if contact_indices.shape[0] == order.shape[0]:
                contact_indices = contact_indices[order]

            witness_contact_forces_local = np.asarray(
                grasp_result.get(
                    "witness_contact_forces_local", np.zeros((0, 3), dtype=np.float64)
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            if witness_contact_forces_local.shape[0] == order.shape[0]:
                witness_contact_forces_local = witness_contact_forces_local[order]

            witness_force_vectors_local = np.asarray(
                grasp_result.get(
                    "witness_force_vectors_local", np.zeros((0, 3), dtype=np.float64)
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            if witness_force_vectors_local.shape[0] == order.shape[0]:
                witness_force_vectors_local = witness_force_vectors_local[order]

        return {
            "contact_points_local": contact_points_local.copy(),
            "normals_local": normals_local.copy(),
            "contact_points_world": contact_points_world.copy(),
            "inward_normals_world": inward_normals_world.copy(),
            "outward_normals_world": outward_normals_world.copy(),
            "virtual_points_world": virtual_points_world.copy(),
            "left_virtual_point": virtual_points_world[0].copy(),
            "right_virtual_point": virtual_points_world[1].copy(),
            "object_pos": object_pos.copy(),
            "object_quat": mat_to_quat_wxyz(object_rot),
            "contact_indices": contact_indices.copy(),
            "raw_contact_indices": raw_contact_indices.copy(),
            "raw_contact_points_local": raw_contact_points_local.copy(),
            "raw_normals_local": raw_normals_local.copy(),
            "contact_order": order.copy(),
            "witness_contact_forces_local": witness_contact_forces_local.copy(),
            "witness_force_vectors_local": witness_force_vectors_local.copy(),
            "total_cost": float(total_cost),
            "region_score": float(region_score),
            "antipodal_margin": float(antipodal_margin),
        }

    def _project_cached_contact_targets(
        self, cached_targets, object_pos, object_rot, virtual_offset=None
    ):
        cached_targets = self._copy_contact_targets(cached_targets)
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        virtual_offset = float(
            self.args.planner_attract_offset
            if virtual_offset is None
            else virtual_offset
        )

        contact_points_local = np.asarray(
            cached_targets["contact_points_local"], dtype=np.float64
        ).reshape(2, 3)
        normals_local = np.asarray(
            cached_targets["normals_local"], dtype=np.float64
        ).reshape(2, 3)
        contact_points_world = (object_rot @ contact_points_local.T).T + object_pos[
            None, :
        ]
        inward_normals_world = (object_rot @ normals_local.T).T
        outward_normals_world = -inward_normals_world
        virtual_points_world = self._offset_points_along_normals(
            contact_points_world,
            outward_normals_world,
            virtual_offset,
        )

        projected = self._copy_contact_targets(cached_targets)
        projected.update(
            {
                "contact_points_world": contact_points_world.copy(),
                "inward_normals_world": inward_normals_world.copy(),
                "outward_normals_world": outward_normals_world.copy(),
                "virtual_points_world": virtual_points_world.copy(),
                "left_virtual_point": virtual_points_world[0].copy(),
                "right_virtual_point": virtual_points_world[1].copy(),
                "object_pos": object_pos.copy(),
                "object_quat": mat_to_quat_wxyz(object_rot),
            }
        )

        desired_force_vectors_local = np.asarray(
            projected.get(
                "desired_force_vectors_local", np.zeros((0, 3), dtype=np.float64)
            ),
            dtype=np.float64,
        ).reshape(-1, 3)
        if desired_force_vectors_local.shape[0] == 2:
            projected["desired_contact_force_world"] = (
                object_rot @ desired_force_vectors_local.T
            ).T

        witness_force_vectors_local = np.asarray(
            projected.get(
                "witness_force_vectors_local", np.zeros((0, 3), dtype=np.float64)
            ),
            dtype=np.float64,
        ).reshape(-1, 3)
        if witness_force_vectors_local.shape[0] == 2:
            projected["witness_contact_force_world"] = (
                object_rot @ witness_force_vectors_local.T
            ).T
        return projected

    def _resolve_stage_object_target(
        self, object_target_fn, step, current_pos, current_quat, current_rot
    ):
        current_pos = np.asarray(current_pos, dtype=np.float64).reshape(3)
        current_quat = np.asarray(current_quat, dtype=np.float64).reshape(4)
        current_rot = _project_to_rotation_matrix(current_rot)
        if object_target_fn is None:
            return current_pos.copy(), current_quat.copy()

        target = object_target_fn(
            step, current_pos.copy(), current_quat.copy(), current_rot.copy()
        )
        if isinstance(target, dict):
            target_pos = np.asarray(
                target.get("object_target_pos", target.get("pos", current_pos)),
                dtype=np.float64,
            ).reshape(3)
            if "object_target_quat" in target:
                target_quat = np.asarray(
                    target["object_target_quat"], dtype=np.float64
                ).reshape(4)
            elif "quat" in target:
                target_quat = np.asarray(target["quat"], dtype=np.float64).reshape(4)
            elif "object_target_rot" in target:
                target_quat = mat_to_quat_wxyz(target["object_target_rot"])
            elif "rot" in target:
                target_quat = mat_to_quat_wxyz(target["rot"])
            else:
                target_quat = current_quat.copy()
            return target_pos, target_quat

        target_pos, target_quat = target
        return (
            np.asarray(target_pos, dtype=np.float64).reshape(3),
            np.asarray(target_quat, dtype=np.float64).reshape(4),
        )

    def _run_live_contact_plan_stage(
        self,
        label,
        max_steps,
        left_step_rot,
        right_step_rot,
        verify_cost_1,
        verify_cost_2,
        virtual_offset,
        goal_offset,
        pos_tol,
        rot_tol=None,
        success_fn=None,
        world_mode="with_pedestal",
        object_target_fn=None,
        initial_contact_targets=None,
        contact_candidate_cache=None,
        contact_selection_state=None,
        planner_target_fn=None,
    ):
        rot_tol = self.args.ik_rot_tol if rot_tol is None else float(rot_tol)
        if int(max_steps) <= 0:
            return False, {}
        if initial_contact_targets is None and contact_candidate_cache is None:
            raise ValueError(
                f"{label} requires cached contact targets from the initial optimizer solve."
            )

        self._set_curobo_world_mode(world_mode)
        planner_sol_guess = None
        last_report_step = -1
        info = {}
        cached_contact_targets = self._copy_contact_targets(initial_contact_targets)
        if contact_selection_state is None:
            contact_selection_state = {
                "mode": "best",
                "pending_mode": None,
                "pending_steps": 0,
            }
        left_step_rot = _project_to_rotation_matrix(left_step_rot)
        right_step_rot = _project_to_rotation_matrix(right_step_rot)
        stage_uses_viewer = self.viewer is not None
        timing_sum = {
            "contact_plan_wall": 0.0,
            "pre_plan_setup": 0.0,
            "planner_contacts": 0.0,
            "planner_solve": 0.0,
            "cartesian_step": 0.0,
            "post_update": 0.0,
            "wall_total": 0.0,
        }

        for step in range(max_steps):
            if not self.is_running():
                break

            loop_t0 = time.perf_counter()
            object_pos, object_quat, object_rot = self.get_object_pose()
            step_t0 = time.perf_counter()
            selection_debug = {}
            if contact_candidate_cache is not None:
                live_targets, selection_debug = self._select_live_contact_targets(
                    object_pos,
                    object_rot,
                    candidate_cache=contact_candidate_cache,
                    previous_targets=cached_contact_targets,
                    virtual_offset=virtual_offset,
                    selection_state=contact_selection_state,
                )
            else:
                live_targets = self._project_cached_contact_targets(
                    cached_contact_targets,
                    object_pos,
                    object_rot,
                    virtual_offset=virtual_offset,
                )
            cached_contact_targets = self._copy_contact_targets(live_targets)
            step_t1 = time.perf_counter()

            planner_contact_points_world = live_targets["contact_points_world"].copy()
            planner_virtual_points_world = live_targets["virtual_points_world"].copy()
            goal_points_world = self._offset_points_along_normals(
                live_targets["contact_points_world"],
                live_targets["outward_normals_world"],
                goal_offset,
            )
            stage_debug = {}
            post_step_debug_fn = None
            if planner_target_fn is not None:
                stage_targets = planner_target_fn(step, live_targets)
                if stage_targets is None:
                    stage_targets = {}
                if "goal_points_world" in stage_targets:
                    goal_points_world = np.asarray(
                        stage_targets["goal_points_world"], dtype=np.float64
                    ).reshape(2, 3)
                if "planner_contact_points_world" in stage_targets:
                    planner_contact_points_world = np.asarray(
                        stage_targets["planner_contact_points_world"],
                        dtype=np.float64,
                    ).reshape(2, 3)
                if "planner_virtual_points_world" in stage_targets:
                    planner_virtual_points_world = np.asarray(
                        stage_targets["planner_virtual_points_world"],
                        dtype=np.float64,
                    ).reshape(2, 3)
                if stage_targets.get("debug") is not None:
                    stage_debug = dict(stage_targets["debug"])
                post_step_debug_fn = stage_targets.get("post_step_debug_fn")
            object_target_pos, object_target_quat = self._resolve_stage_object_target(
                object_target_fn,
                step,
                object_pos,
                object_quat,
                object_rot,
            )
            step_t1a = time.perf_counter()

            curr_x = self.get_planner_state()
            phi_vec, jac_mat = self._detect_planner_contacts()
            step_t2 = time.perf_counter()
            planner_result = self.planner.plan_once(
                object_target_pos,
                object_target_quat,
                curr_x,
                phi_vec,
                jac_mat,
                sol_guess=planner_sol_guess,
                verify_cost_param_1=float(verify_cost_1),
                verify_cost_param_2=float(verify_cost_2),
                virtual_point_1=planner_virtual_points_world[0],
                virtual_point_2=planner_virtual_points_world[1],
                contact_point_1=planner_contact_points_world[0],
                contact_point_2=planner_contact_points_world[1],
            )
            step_t3 = time.perf_counter()
            planner_sol_guess = planner_result["sol_guess"]
            self.plan_params.sol_guess_ = planner_sol_guess
            planner_backend = str(
                planner_result.get("solver_backend", self.plan_params.planner_solver_)
            )
            planner_status = str(planner_result.get("solve_status", ""))
            action = np.asarray(planner_result["action"], dtype=np.float64).reshape(-1)
            if action.shape[0] != 6:
                raise RuntimeError(
                    f"Expected a 6D dual-arm plan_once action, got shape {action.shape}."
                )

            applied_object_torque_world = None
            if bool(self.args.test_force):
                _, applied_object_torque_world = self._best_object_wrench_world(
                    live_targets,
                    object_rot,
                )
            self.step_cartesian_action(
                action[:3],
                action[3:6],
                left_step_rot,
                right_step_rot,
                object_torque_world=applied_object_torque_world,
            )
            step_t4 = time.perf_counter()

            self.set_marker("contact_point1", live_targets["contact_points_world"][0])
            self.set_marker("contact_point2", live_targets["contact_points_world"][1])
            self.set_marker("left_goal", goal_points_world[0])
            self.set_marker("right_goal", goal_points_world[1])
            self.set_marker("obj_point", object_target_pos)
            self.set_marker("goal", object_target_pos, object_target_quat)
            mujoco.mj_forward(self.model, self.data)
            if self.viewer is not None:
                self.sync_viewer()
            step_t5 = time.perf_counter()
            if callable(post_step_debug_fn):
                post_step_debug = post_step_debug_fn()
                if post_step_debug is not None:
                    stage_debug = dict(post_step_debug)
            loop_t1 = time.perf_counter()

            timing = {
                "contact_plan_wall": step_t1 - step_t0,
                "pre_plan_setup": step_t1a - step_t1,
                "planner_contacts": step_t2 - step_t1a,
                "planner_solve": step_t3 - step_t2,
                "cartesian_step": step_t4 - step_t3,
                "post_update": step_t5 - step_t4,
                "wall_total": loop_t1 - loop_t0,
            }
            optimizer_search_timing = selection_debug.get("selected_search_timing", {})
            if optimizer_search_timing:
                timing["contact_force_closure_total"] = float(
                    optimizer_search_timing.get("force_closure_solve_time", 0.0)
                )
                timing["contact_static_total"] = float(
                    optimizer_search_timing.get("static_equilibrium_solve_time", 0.0)
                )
            for key, value in timing.items():
                if key not in timing_sum:
                    timing_sum[key] = 0.0
                timing_sum[key] += float(value)

            left_err, left_rot_err = self._tip_target_error(
                self.left_arm, goal_points_world[0], left_step_rot
            )
            right_err, right_rot_err = self._tip_target_error(
                self.right_arm, goal_points_world[1], right_step_rot
            )
            contacts = self.extract_object_contacts()
            info = {
                "step": step,
                "contacts": contacts,
                "left_force": self._max_normal_force(contacts["left"]),
                "right_force": self._max_normal_force(contacts["right"]),
                "left_pos_err": float(left_err),
                "right_pos_err": float(right_err),
                "left_rot_err": float(left_rot_err),
                "right_rot_err": float(right_rot_err),
                "left_goal_pos": goal_points_world[0].copy(),
                "right_goal_pos": goal_points_world[1].copy(),
                "left_goal_rot": left_step_rot.copy(),
                "right_goal_rot": right_step_rot.copy(),
                "left_planner_cmd": action[:3].copy(),
                "right_planner_cmd": action[3:6].copy(),
                "planner_backend": planner_backend,
                "planner_status": planner_status,
                "object_pos": self.get_object_pose()[0].copy(),
                "planner_object_target_pos": np.asarray(
                    object_target_pos, dtype=np.float64
                ).copy(),
                "planner_object_target_quat": np.asarray(
                    object_target_quat, dtype=np.float64
                ).copy(),
                "contact_points_local": live_targets["contact_points_local"].copy(),
                "contact_points_world": live_targets["contact_points_world"].copy(),
                "normals_local": live_targets["normals_local"].copy(),
                "inward_normals_world": live_targets["inward_normals_world"].copy(),
                "outward_normals_world": live_targets["outward_normals_world"].copy(),
                "planner_contact_points_world": planner_contact_points_world.copy(),
                "planner_virtual_points_world": planner_virtual_points_world.copy(),
                "left_virtual_point": planner_virtual_points_world[0].copy(),
                "right_virtual_point": planner_virtual_points_world[1].copy(),
                "grasp_cost": float(live_targets["total_cost"]),
                "force_closure_cost": float(
                    live_targets.get("force_closure_cost", float("inf"))
                ),
                "lift_cost": float(live_targets.get("lift_cost", 0.0)),
                "region_score": float(live_targets["region_score"]),
                "antipodal_margin": float(live_targets["antipodal_margin"]),
                "contact_search_timing": self._copy_nested_value(
                    live_targets.get("search_timing", {})
                ),
                "contact_targets": self._copy_contact_targets(live_targets),
                "timing": dict(timing),
            }
            for key, value in selection_debug.items():
                if isinstance(value, np.ndarray):
                    info[key] = value.copy()
                else:
                    info[key] = copy.deepcopy(value)
            for key, value in stage_debug.items():
                if isinstance(value, np.ndarray):
                    info[key] = value.copy()
                else:
                    info[key] = copy.deepcopy(value)
            if "modeled_normal_forces" in info:
                modeled_normal_forces = np.asarray(
                    info["modeled_normal_forces"], dtype=np.float64
                ).reshape(-1)
                if modeled_normal_forces.size == 2:
                    info["left_modeled_force"] = float(modeled_normal_forces[0])
                    info["right_modeled_force"] = float(modeled_normal_forces[1])

            loop_hz = 1.0 / max(timing["wall_total"], 1e-9)
            contact_search_timing = info.get("contact_search_timing", {})
            raw_candidate_entry_count = int(
                contact_search_timing.get(
                    "candidate_entry_count_before_limit",
                    contact_search_timing.get("candidate_entry_count", 0),
                )
            )
            candidate_entry_count = int(
                contact_search_timing.get("candidate_entry_count", 0)
            )
            fc_total_time = float(
                contact_search_timing.get("force_closure_solve_time", 0.0)
            )
            static_total_time = float(
                contact_search_timing.get("static_equilibrium_solve_time", 0.0)
            )
            print(
                f"[timing:{label}] step={step:04d} "
                f"contact_plan_wall={timing['contact_plan_wall']:.4f}s "
                f"pre_plan_setup={timing['pre_plan_setup']:.4f}s "
                f"planner_contacts={timing['planner_contacts']:.4f}s "
                f"planner_solve={timing['planner_solve']:.4f}s "
                f"cartesian_step={timing['cartesian_step']:.4f}s "
                f"post_update={timing['post_update']:.4f}s "
                f"wall_total={timing['wall_total']:.4f}s "
                f"hz={loop_hz:.2f} "
                f"planner={planner_backend} "
                f"contact_mode={contact_search_timing.get('mode', 'unknown')} "
                f"candidates={raw_candidate_entry_count}->{candidate_entry_count} "
                f"fc_total={fc_total_time:.4f}s "
                f"static_total={static_total_time:.4f}s"
            )

            if step == 0 or step == max_steps - 1 or step - last_report_step >= 40:
                modeled_force_text = ""
                if "left_modeled_force" in info and "right_modeled_force" in info:
                    modeled_force_text = f" modeled_force=({info['left_modeled_force']:.3f},{info['right_modeled_force']:.3f})"
                print(
                    f"[{label}] step={step:04d} "
                    f"left_err={left_err:.4f}/{left_rot_err:.4f} "
                    f"right_err={right_err:.4f}/{right_rot_err:.4f} "
                    f"left_force={info['left_force']:.3f} right_force={info['right_force']:.3f} "
                    f"{modeled_force_text} "
                    f"grasp_cost={info['grasp_cost']:.4f} "
                    f"fc={info['force_closure_cost']:.4f} "
                    f"lift={info['lift_cost']:.4f} "
                    f"source={info.get('selected_contact_source', 'fixed')} "
                    f"planner={planner_backend} "
                    f"time(contact_plan={timing['contact_plan_wall']:.4f}s "
                    f"setup={timing['pre_plan_setup']:.4f}s "
                    f"contacts={timing['planner_contacts']:.4f}s "
                    f"plan={timing['planner_solve']:.4f}s "
                    f"ctrl={timing['cartesian_step']:.4f}s "
                    f"post={timing['post_update']:.4f}s "
                    f"wall={timing['wall_total']:.4f}s)"
                )
                if step == 0:
                    print(
                        f"  stage_cfg: world_mode={world_mode} "
                        f"verify=({float(verify_cost_1):.1f},{float(verify_cost_2):.1f}) "
                        f"virtual_offset={float(virtual_offset):.4f} goal_offset={float(goal_offset):.4f} "
                        f"contacts={'online_select' if contact_candidate_cache is not None else 'projected_local'} "
                        f"viewer={'on' if stage_uses_viewer else 'off'}"
                    )
                if planner_status:
                    print(f"  planner_status: {planner_status}")
                if contact_search_timing:
                    print(
                        "  contact_timing: "
                        f"mode={contact_search_timing.get('mode', 'unknown')} "
                        f"candidates={candidate_entry_count} "
                        f"fc_total={fc_total_time:.4f}s "
                        f"static_total={static_total_time:.4f}s "
                        f"search_wall={float(contact_search_timing.get('wall_time', 0.0)):.4f}s"
                    )
                last_report_step = step

            pose_ok = (
                left_err < pos_tol
                and right_err < pos_tol
                and left_rot_err < rot_tol
                and right_rot_err < rot_tol
            )
            if success_fn is None:
                if pose_ok:
                    info["timing_avg"] = {
                        key: value / max(step + 1, 1)
                        for key, value in timing_sum.items()
                    }
                    return True, info
            elif success_fn(info):
                info["timing_avg"] = {
                    key: value / max(step + 1, 1) for key, value in timing_sum.items()
                }
                return True, info

        if max_steps > 0:
            info["timing_avg"] = {
                key: value / max(int(max_steps), 1) for key, value in timing_sum.items()
            }
        return False, info

    def _solve_stage_ik_pose_set(
        self, contact_points_local, normals_local, stage_specs
    ):
        stage_pose_set = {}
        original_world_mode = self.left_arm.current_world_mode
        for stage_name, spec in stage_specs.items():
            world_mode = str(spec.get("world_mode", "with_pedestal"))
            center_offset = float(spec["center_offset"])
            object_pos = np.asarray(spec["object_pos"], dtype=np.float64).reshape(3)
            object_rot = _project_to_rotation_matrix(spec["object_rot"])

            self._set_curobo_world_mode(world_mode)
            self._update_inter_arm_worlds()
            stage_targets = self._stage_targets_from_object_pose(
                contact_points_local,
                normals_local,
                object_pos,
                object_rot,
                center_offset=center_offset,
            )
            left_ik = self.solve_arm_ik(
                self.left_arm,
                stage_targets["left_tip_pos"],
                stage_targets["left_tip_rot"],
            )
            right_ik = self.solve_arm_ik(
                self.right_arm,
                stage_targets["right_tip_pos"],
                stage_targets["right_tip_rot"],
            )
            stage_pose_set[stage_name] = {
                "targets": stage_targets,
                "left": left_ik,
                "right": right_ik,
                "world_mode": world_mode,
            }

        self._set_curobo_world_mode(original_world_mode)
        self._update_inter_arm_worlds()
        return stage_pose_set

    def _build_stage2_tracking_rotations(self, stage_targets):
        def _keep_current_rotation(_step, _curr_pos, curr_rot):
            return _project_to_rotation_matrix(curr_rot)

        return _keep_current_rotation, _keep_current_rotation

    def _run_precomputed_dual_arm_plan_stage(
        self,
        label,
        max_steps,
        left_virtual_point,
        right_virtual_point,
        left_contact_ik,
        right_contact_ik,
        object_target_pos,
        object_target_quat,
        left_goal_pos,
        right_goal_pos,
        left_goal_rot,
        right_goal_rot,
        verify_cost_1,
        verify_cost_2,
        pos_tol,
        rot_tol=None,
        success_fn=None,
        world_mode="with_pedestal",
    ):
        rot_tol = self.args.ik_rot_tol if rot_tol is None else float(rot_tol)
        if int(max_steps) <= 0:
            return False, {}

        self._set_curobo_world_mode(world_mode)
        planner_sol_guess = None
        last_report_step = -1
        info = {}

        left_virtual_point = np.asarray(left_virtual_point, dtype=np.float64).reshape(3)
        right_virtual_point = np.asarray(right_virtual_point, dtype=np.float64).reshape(
            3
        )
        object_target_pos = np.asarray(object_target_pos, dtype=np.float64).reshape(3)
        object_target_quat = np.asarray(object_target_quat, dtype=np.float64).reshape(4)
        left_goal_pos = np.asarray(left_goal_pos, dtype=np.float64).reshape(3)
        right_goal_pos = np.asarray(right_goal_pos, dtype=np.float64).reshape(3)

        for step in range(max_steps):
            if not self.is_running():
                break

            self._update_inter_arm_worlds()
            left_curr_pos, left_curr_rot = self.get_tip_pose(self.left_arm)
            right_curr_pos, right_curr_rot = self.get_tip_pose(self.right_arm)
            left_step_rot = (
                left_goal_rot(step, left_curr_pos, left_curr_rot)
                if callable(left_goal_rot)
                else left_goal_rot
            )
            right_step_rot = (
                right_goal_rot(step, right_curr_pos, right_curr_rot)
                if callable(right_goal_rot)
                else right_goal_rot
            )
            left_step_rot = _project_to_rotation_matrix(left_step_rot)
            right_step_rot = _project_to_rotation_matrix(right_step_rot)

            curr_x = self.get_planner_state()
            phi_vec, jac_mat = self._detect_planner_contacts()
            planner_result = self.planner.plan_once(
                object_target_pos,
                object_target_quat,
                curr_x,
                phi_vec,
                jac_mat,
                sol_guess=planner_sol_guess,
                verify_cost_param_1=float(verify_cost_1),
                verify_cost_param_2=float(verify_cost_2),
                virtual_point_1=left_virtual_point,
                virtual_point_2=right_virtual_point,
                contact_point_1=left_contact_ik.solved_tip_pos_world,
                contact_point_2=right_contact_ik.solved_tip_pos_world,
            )
            planner_sol_guess = planner_result["sol_guess"]
            self.plan_params.sol_guess_ = planner_sol_guess
            action = np.asarray(planner_result["action"], dtype=np.float64).reshape(-1)
            if action.shape[0] != 6:
                raise RuntimeError(
                    f"Expected a 6D dual-arm plan_once action, got shape {action.shape}."
                )

            self.step_cartesian_action(
                action[:3],
                action[3:6],
                left_step_rot,
                right_step_rot,
            )

            self.set_marker("left_goal", left_goal_pos)
            self.set_marker("right_goal", right_goal_pos)
            self.set_marker("obj_point", object_target_pos)
            mujoco.mj_forward(self.model, self.data)
            if self.viewer is not None:
                self.sync_viewer()

            left_err, left_rot_err = self._tip_target_error(
                self.left_arm, left_goal_pos, left_step_rot
            )
            right_err, right_rot_err = self._tip_target_error(
                self.right_arm, right_goal_pos, right_step_rot
            )
            contacts = self.extract_object_contacts()
            info = {
                "step": step,
                "contacts": contacts,
                "left_force": self._max_normal_force(contacts["left"]),
                "right_force": self._max_normal_force(contacts["right"]),
                "left_pos_err": float(left_err),
                "right_pos_err": float(right_err),
                "left_rot_err": float(left_rot_err),
                "right_rot_err": float(right_rot_err),
                "left_goal_pos": left_goal_pos.copy(),
                "right_goal_pos": right_goal_pos.copy(),
                "left_goal_rot": left_step_rot.copy(),
                "right_goal_rot": right_step_rot.copy(),
                "left_goal_q_mj": np.asarray(
                    left_contact_ik.q_mj, dtype=np.float64
                ).copy(),
                "right_goal_q_mj": np.asarray(
                    right_contact_ik.q_mj, dtype=np.float64
                ).copy(),
                "left_ik_ok": bool(left_contact_ik.success),
                "right_ik_ok": bool(right_contact_ik.success),
                "left_ik_pos_err": float(left_contact_ik.position_error),
                "right_ik_pos_err": float(right_contact_ik.position_error),
                "left_ik_rot_err": float(left_contact_ik.rotation_error),
                "right_ik_rot_err": float(right_contact_ik.rotation_error),
                "left_ik_constraint_total": float(left_contact_ik.constraint_total),
                "right_ik_constraint_total": float(right_contact_ik.constraint_total),
                "left_ik_failure_reason": str(left_contact_ik.failure_reason),
                "right_ik_failure_reason": str(right_contact_ik.failure_reason),
                "object_pos": self.get_object_pose()[0].copy(),
                "left_planner_cmd": action[:3].copy(),
                "right_planner_cmd": action[3:6].copy(),
            }

            if step == 0 or step == max_steps - 1 or step - last_report_step >= 40:
                print(
                    f"[{label}] step={step:04d} "
                    f"left_err={left_err:.4f}/{left_rot_err:.4f} "
                    f"right_err={right_err:.4f}/{right_rot_err:.4f} "
                    f"left_force={info['left_force']:.3f} right_force={info['right_force']:.3f} "
                    f"left_ik={left_contact_ik.success} right_ik={right_contact_ik.success}"
                )
                if step == 0:
                    print(
                        f"  stage_cfg: world_mode={world_mode} "
                        f"verify=({float(verify_cost_1):.1f},{float(verify_cost_2):.1f}) planner=plan_once+impedance"
                    )
                last_report_step = step

            pose_ok = (
                left_err < pos_tol
                and right_err < pos_tol
                and left_rot_err < rot_tol
                and right_rot_err < rot_tol
            )
            if success_fn is None:
                if pose_ok:
                    return True, info
            elif success_fn(info):
                return True, info

        return False, info

    def set_marker(self, name, pos, quat=None):
        body_id = self.marker_body_ids[name]
        self.model.body_pos[body_id] = np.asarray(pos, dtype=np.float64)
        if quat is not None:
            self.model.body_quat[body_id] = np.asarray(quat, dtype=np.float64)

    def set_ghost_pose(self, arm, hand_pos_world, hand_rot_world):
        if arm.ghost_body_id < 0:
            return
        self.model.body_pos[arm.ghost_body_id] = np.asarray(
            hand_pos_world, dtype=np.float64
        )
        self.model.body_quat[arm.ghost_body_id] = mat_to_quat_wxyz(hand_rot_world)

    @staticmethod
    def _constraint_scalar(value):
        if value is None:
            return 0.0
        array = np.asarray(_tensor_to_numpy(value), dtype=np.float64)
        if array.size == 0:
            return 0.0
        return float(np.max(array))

    def _diagnose_ik_solution(self, arm, q_curobo):
        joint_state = make_joint_state(arm.ik_solver, q_curobo)
        rollout_fn = arm.ik_solver.rollout_fn
        aug_state = rollout_fn._get_augmented_state(joint_state)

        bound_constraint = self._constraint_scalar(
            rollout_fn.bound_constraint.forward(aug_state.state_seq)
        )

        world_constraint = 0.0
        static_world_constraint = 0.0
        primitive_constraint = getattr(
            rollout_fn, "primitive_collision_constraint", None
        )
        if primitive_constraint is not None and getattr(
            primitive_constraint, "enabled", False
        ):
            world_constraint = self._constraint_scalar(
                primitive_constraint.forward(
                    aug_state.robot_spheres, env_query_idx=None
                )
            )
            if arm.static_world is not None:
                restore_world = (
                    arm.current_world
                    if arm.current_world is not None
                    else arm.static_world
                )
                arm.ik_solver.update_world(arm.static_world)
                try:
                    static_world_constraint = self._constraint_scalar(
                        primitive_constraint.forward(
                            aug_state.robot_spheres, env_query_idx=None
                        )
                    )
                finally:
                    arm.ik_solver.update_world(restore_world)

        self_constraint = 0.0
        self_collision_constraint = getattr(
            rollout_fn, "robot_self_collision_constraint", None
        )
        if self_collision_constraint is not None and getattr(
            self_collision_constraint, "enabled", False
        ):
            self_constraint = self._constraint_scalar(
                self_collision_constraint.forward(aug_state.robot_spheres)
            )

        total_constraint = bound_constraint + world_constraint + self_constraint
        eps = 1e-6
        failure_reasons = []
        static_world_reason = (
            "floor clearance"
            if arm.current_world_mode == "floor_only"
            else "pedestal/floor clearance"
        )
        if bound_constraint > eps:
            failure_reasons.append("joint bound")
        if world_constraint > eps:
            if (
                static_world_constraint > eps
                and world_constraint > static_world_constraint + eps
            ):
                failure_reasons.append(f"{static_world_reason} + other-arm clearance")
            elif static_world_constraint > eps:
                failure_reasons.append(static_world_reason)
            else:
                failure_reasons.append("other-arm clearance")
        if self_constraint > eps:
            failure_reasons.append("self-collision clearance")
        if not failure_reasons and total_constraint > eps:
            failure_reasons.append("feasibility constraint")

        return {
            "constraint_total": total_constraint,
            "bound_constraint": bound_constraint,
            "world_constraint": world_constraint,
            "static_world_constraint": static_world_constraint,
            "self_constraint": self_constraint,
            "failure_reason": ", ".join(failure_reasons),
        }

    def _current_arm_q_curobo(self, arm):
        return build_curobo_state(
            self.data.qpos[arm.qpos_adr].copy(),
            arm.curobo_joint_names,
            arm.retract_cfg,
        )

    def solve_arm_ik(self, arm, target_tip_pos_world, target_tip_rot_world):
        target_hand_pos_world, target_hand_rot_world = self.tip_target_to_hand_pose(
            target_tip_pos_world,
            target_tip_rot_world,
        )
        target_hand_pos_local, target_hand_rot_local = self.world_pose_to_arm_frame(
            arm,
            target_hand_pos_world,
            target_hand_rot_world,
        )
        goal_pose = make_pose(
            arm.ik_solver,
            target_hand_pos_local,
            mat_to_quat_wxyz(target_hand_rot_local),
        )
        current_q_curobo = self._current_arm_q_curobo(arm)
        retract_cfg = make_joint_tensor(
            arm.ik_solver, current_q_curobo, extra_dim=False
        )
        seed_cfg = make_joint_tensor(arm.ik_solver, current_q_curobo, extra_dim=True)
        result = arm.ik_solver.solve_single(
            goal_pose,
            retract_config=retract_cfg,
            seed_config=seed_cfg,
            return_seeds=1,
            num_seeds=self.args.ik_num_seeds,
            use_nn_seed=False,
            newton_iters=self.args.ik_max_iters,
        )

        ik_success = bool(_tensor_to_numpy(result.success).reshape(-1)[0])
        ik_pos_err = float(_tensor_to_numpy(result.position_error).reshape(-1)[0])
        ik_rot_err = float(_tensor_to_numpy(result.rotation_error).reshape(-1)[0])
        solution = _tensor_to_numpy(result.solution).reshape(
            -1, len(arm.curobo_joint_names)
        )[0]
        q_mj = extract_mujoco_arm_configuration(solution, arm.curobo_joint_names)
        ik_diag = {
            "constraint_total": 0.0,
            "bound_constraint": 0.0,
            "world_constraint": 0.0,
            "static_world_constraint": 0.0,
            "self_constraint": 0.0,
            "failure_reason": "",
        }
        if not ik_success:
            ik_diag = self._diagnose_ik_solution(arm, solution)

        fk_state = arm.ik_solver.fk(
            torch.tensor(
                np.asarray(solution, dtype=np.float32).reshape(1, -1),
                device=arm.ik_solver.tensor_args.device,
                dtype=arm.ik_solver.tensor_args.dtype,
            )
        )
        solved_hand_pos_local, solved_hand_quat_local = _extract_pose_from_kinematics(
            fk_state
        )
        solved_hand_rot_local = quat_wxyz_to_mat(solved_hand_quat_local)
        solved_hand_pos_world, solved_hand_rot_world = self.arm_pose_to_world_frame(
            arm,
            solved_hand_pos_local,
            solved_hand_rot_local,
        )
        solved_tip_pos_world = (
            solved_hand_pos_world + solved_hand_rot_world[:, 2] * TIP_CENTER_OFFSET
        )
        solved_tip_rot_world = solved_hand_rot_world.copy()
        self.set_ghost_pose(arm, solved_hand_pos_world, solved_hand_rot_world)

        return ArmIkResult(
            q_mj=q_mj,
            success=ik_success,
            position_error=ik_pos_err,
            rotation_error=ik_rot_err,
            target_hand_pos_world=target_hand_pos_world,
            target_hand_rot_world=target_hand_rot_world,
            solved_hand_pos_world=solved_hand_pos_world,
            solved_hand_rot_world=solved_hand_rot_world,
            solved_tip_pos_world=solved_tip_pos_world,
            solved_tip_rot_world=solved_tip_rot_world,
            constraint_total=ik_diag["constraint_total"],
            bound_constraint=ik_diag["bound_constraint"],
            world_constraint=ik_diag["world_constraint"],
            static_world_constraint=ik_diag["static_world_constraint"],
            self_constraint=ik_diag["self_constraint"],
            failure_reason=ik_diag["failure_reason"],
        )

    def build_pose_goal_result(
        self, arm, target_tip_pos_world, target_tip_rot_world, reference_q_mj=None
    ):
        target_tip_pos_world = np.asarray(
            target_tip_pos_world, dtype=np.float64
        ).reshape(3)
        target_tip_rot_world = _project_to_rotation_matrix(target_tip_rot_world)
        target_hand_pos_world, target_hand_rot_world = self.tip_target_to_hand_pose(
            target_tip_pos_world,
            target_tip_rot_world,
        )
        self.set_ghost_pose(arm, target_hand_pos_world, target_hand_rot_world)
        if reference_q_mj is None:
            q_mj = self.data.qpos[arm.qpos_adr].copy()
        else:
            q_mj = np.asarray(reference_q_mj, dtype=np.float64).reshape(-1).copy()
        return ArmIkResult(
            q_mj=q_mj,
            success=True,
            position_error=0.0,
            rotation_error=0.0,
            target_hand_pos_world=target_hand_pos_world,
            target_hand_rot_world=target_hand_rot_world,
            solved_hand_pos_world=target_hand_pos_world.copy(),
            solved_hand_rot_world=target_hand_rot_world.copy(),
            solved_tip_pos_world=target_tip_pos_world.copy(),
            solved_tip_rot_world=target_tip_rot_world.copy(),
            constraint_total=0.0,
            bound_constraint=0.0,
            world_constraint=0.0,
            static_world_constraint=0.0,
            self_constraint=0.0,
            failure_reason="",
        )

    def get_planner_state(self):
        obj_pos, obj_quat, _ = self.get_object_pose()
        left_tip_pos = self.get_tip_pos(self.left_arm)
        right_tip_pos = self.get_tip_pos(self.right_arm)
        return np.hstack([obj_pos, obj_quat, left_tip_pos, right_tip_pos]).astype(
            np.float32
        )

    def _build_object_jacobian(self, point_local):
        jacobian = np.zeros((3, self.plan_params.n_qvel_), dtype=np.float64)
        jacobian[:, :3] = np.eye(3, dtype=np.float64)
        jacobian[0, 4] = point_local[2]
        jacobian[0, 5] = -point_local[1]
        jacobian[1, 3] = -point_local[2]
        jacobian[1, 5] = point_local[0]
        jacobian[2, 3] = point_local[1]
        jacobian[2, 4] = -point_local[0]
        return jacobian

    def _build_tip_jacobian_block(self, arm):
        jacobian = np.zeros((3, self.plan_params.n_qvel_), dtype=np.float64)
        if arm.prefix == "left_":
            jacobian[:, 6:9] = np.eye(3, dtype=np.float64)
        else:
            jacobian[:, 9:12] = np.eye(3, dtype=np.float64)
        return jacobian

    def _reformat_planner_contacts(self, con_phi_list=None, con_jac_list=None):
        con_phi_list = [] if con_phi_list is None else con_phi_list
        con_jac_list = [] if con_jac_list is None else con_jac_list
        phi_vec = np.ones((self.plan_params.max_ncon_ * 4,), dtype=np.float32)
        jac_mat = np.zeros(
            (self.plan_params.max_ncon_ * 4, self.plan_params.n_qvel_), dtype=np.float32
        )
        for i in range(min(len(con_phi_list), self.plan_params.max_ncon_)):
            phi_vec[4 * i : 4 * i + 4] = float(con_phi_list[i])
            jac_mat[4 * i : 4 * i + 4] = np.asarray(con_jac_list[i], dtype=np.float32)
        return phi_vec, jac_mat

    def _detect_planner_contacts(self):
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

        obj_pos, _, obj_rot = self.get_object_pose()
        left_robot_jacobian = self._build_tip_jacobian_block(self.left_arm)
        right_robot_jacobian = self._build_tip_jacobian_block(self.right_arm)

        con_phi_list = []
        con_jac_list = []
        for i in range(self.data.ncon):
            contact_i = self.data.contact[i]
            geom1 = int(contact_i.geom1)
            geom2 = int(contact_i.geom2)
            if self.obj_geom_id not in {geom1, geom2}:
                continue

            geom1_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
            )
            geom2_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
            )
            body1_id = int(self.model.geom_bodyid[geom1])
            body2_id = int(self.model.geom_bodyid[geom2])
            body1_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or ""
            )
            body2_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or ""
            )

            object_is_first = body1_name == "obj"
            other_body_name = body2_name if object_is_first else body1_name
            other_geom_name = geom2_name if object_is_first else geom1_name

            con_pos = np.asarray(contact_i.pos, dtype=np.float64).copy()
            con_dist = float(contact_i.dist) * 0.5
            con_mu = float(self.args.arm_friction)
            con_frame = np.asarray(contact_i.frame, dtype=np.float64).reshape((-1, 3)).T
            con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

            con_pos_local = obj_rot.T @ (con_pos - obj_pos)
            object_jacobian = self._build_object_jacobian(con_pos_local)
            con_jacp_obj = con_frame_pmd.T @ object_jacobian

            if other_body_name.startswith("left_"):
                con_jacp_other = con_frame_pmd.T @ left_robot_jacobian
            elif other_body_name.startswith("right_"):
                con_jacp_other = con_frame_pmd.T @ right_robot_jacobian
            else:
                con_jacp_other = np.zeros(
                    (5, self.plan_params.n_qvel_), dtype=np.float64
                )

            con_jac = con_jacp_obj - con_jacp_other
            con_jac = con_jac[0] + con_mu * con_jac[1:]
            con_phi_list.append(con_dist)
            con_jac_list.append(con_jac)

        return self._reformat_planner_contacts(con_phi_list, con_jac_list)

    def get_current_joint_position(self, arm, update_kinematics=False):
        if update_kinematics:
            mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.qpos[arm.qpos_adr], dtype=np.float64).copy()

    def get_current_joint_velocity(self, arm, update_kinematics=False):
        if update_kinematics:
            mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.qvel[arm.dof_adr], dtype=np.float64).copy()

    def get_arm_jacobian(self, arm, update_kinematics=False):
        if update_kinematics:
            mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        tip_pos, _ = self._get_tip_geom_pose(arm)
        tip_body_id = int(self.model.geom_bodyid[arm.tip_geom_id])
        mujoco.mj_jac(
            self.model,
            self.data,
            jacp=jacp,
            jacr=jacr,
            point=tip_pos,
            body=tip_body_id,
        )
        return np.vstack([jacp[:, arm.dof_adr], jacr[:, arm.dof_adr]])

    def set_control_torque(self, arm, torque):
        self.data.ctrl[arm.actuator_ids] = np.clip(
            np.asarray(torque, dtype=np.float64),
            -arm.torque_limits,
            arm.torque_limits,
        )

    def set_desired_tip_pose(self, arm, position, orientation):
        arm.position_d = np.asarray(position, dtype=np.float64).copy()
        arm.orientation_d = _project_to_rotation_matrix(orientation)
        arm.p_d = arm.position_d.copy()
        arm.R_d = arm.orientation_d.copy()

    def _compute_cartesian_impedance_control(
        self, arm, q=None, dq=None, jacobian=None, tip_pose=None
    ):
        q = (
            self.get_current_joint_position(arm)
            if q is None
            else np.asarray(q, dtype=np.float64).copy()
        )
        dq = (
            self.get_current_joint_velocity(arm)
            if dq is None
            else np.asarray(dq, dtype=np.float64).copy()
        )
        jacobian = (
            self.get_arm_jacobian(arm)
            if jacobian is None
            else np.asarray(jacobian, dtype=np.float64).copy()
        )
        if tip_pose is None:
            p_current, rot_current = self.get_tip_pose(arm)
        else:
            p_current, rot_current = tip_pose
            p_current = np.asarray(p_current, dtype=np.float64).copy()
            rot_current = np.asarray(rot_current, dtype=np.float64).reshape(3, 3).copy()

        error = np.zeros(6, dtype=np.float64)
        error[:3] = p_current - arm.position_d
        rot_error = rot_current.T @ arm.orientation_d
        error_quat = Rotation.from_matrix(rot_error).as_quat()
        if error_quat[3] < 0.0:
            error_quat = -error_quat
        error[3:] = -rot_current @ error_quat[:3]

        velocity = jacobian @ dq
        desired_wrench = (
            -arm.cartesian_stiffness @ error - arm.cartesian_damping @ velocity
        )
        tau_task = jacobian.T @ desired_wrench

        jacobian_pinv = pinv(jacobian.T)
        nullspace_proj = np.eye(7) - jacobian.T @ jacobian_pinv
        tau_nullspace = nullspace_proj @ (
            arm.nullspace_stiffness * (arm.home_q - q)
            - 2.0 * np.sqrt(arm.nullspace_stiffness) * dq
        )
        tau_bias = np.asarray(self.data.qfrc_bias[arm.dof_adr], dtype=np.float64).copy()
        tau = tau_task + tau_nullspace + tau_bias
        return np.clip(tau, -arm.torque_limits, arm.torque_limits)

    def _gather_arm_control_state(self, arm):
        q = np.asarray(self.data.qpos[arm.qpos_adr], dtype=np.float64).copy()
        dq = np.asarray(self.data.qvel[arm.dof_adr], dtype=np.float64).copy()
        tip_pose = self.get_tip_pose(arm)
        jacobian = self.get_arm_jacobian(arm)
        return {
            "q": q,
            "dq": dq,
            "tip_pose": tip_pose,
            "jacobian": jacobian,
        }

    def _compute_dual_arm_cartesian_impedance_control(self):
        mujoco.mj_forward(self.model, self.data)
        left_state = self._gather_arm_control_state(self.left_arm)
        right_state = self._gather_arm_control_state(self.right_arm)
        left_tau = self._compute_cartesian_impedance_control(
            self.left_arm,
            q=left_state["q"],
            dq=left_state["dq"],
            jacobian=left_state["jacobian"],
            tip_pose=left_state["tip_pose"],
        )
        right_tau = self._compute_cartesian_impedance_control(
            self.right_arm,
            q=right_state["q"],
            dq=right_state["dq"],
            jacobian=right_state["jacobian"],
            tip_pose=right_state["tip_pose"],
        )
        return left_tau, right_tau

    def _planner_tracking_steps(self):
        base_steps = max(
            int(self.args.mj_steps_per_command)
            * max(int(self.args.command_substeps), 1),
            1,
        )
        planner_steps = max(
            int(np.ceil(float(self.plan_params.h_) / float(self.model.opt.timestep))), 1
        )
        return max(base_steps, planner_steps)

    def step_cartesian_action(
        self,
        left_cmd,
        right_cmd,
        left_rot_world,
        right_rot_world,
        object_force_world=None,
        object_torque_world=None,
    ):
        left_curr_pos, left_curr_rot = self.get_tip_pose(self.left_arm)
        right_curr_pos, right_curr_rot = self.get_tip_pose(self.right_arm)
        left_target_pos = left_curr_pos + np.asarray(left_cmd, dtype=np.float64)
        right_target_pos = right_curr_pos + np.asarray(right_cmd, dtype=np.float64)
        self.set_marker("left_desired_tip", left_target_pos)
        self.set_marker("right_desired_tip", right_target_pos)

        num_steps = self._planner_tracking_steps()
        for step_idx in range(num_steps):
            alpha = float(step_idx + 1) / float(num_steps)
            left_interp_pos = left_curr_pos + alpha * (left_target_pos - left_curr_pos)
            right_interp_pos = right_curr_pos + alpha * (
                right_target_pos - right_curr_pos
            )
            left_interp_rot = _slerp_rotation_matrix(
                left_curr_rot, left_rot_world, alpha
            )
            right_interp_rot = _slerp_rotation_matrix(
                right_curr_rot, right_rot_world, alpha
            )

            self.set_desired_tip_pose(self.left_arm, left_interp_pos, left_interp_rot)
            self.set_desired_tip_pose(
                self.right_arm, right_interp_pos, right_interp_rot
            )
            left_tau, right_tau = self._compute_dual_arm_cartesian_impedance_control()
            self.set_control_torque(self.left_arm, left_tau)
            self.set_control_torque(self.right_arm, right_tau)
            self.apply_object_wrench_world(
                force_world=object_force_world,
                torque_world=object_torque_world,
            )
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if self.viewer is not None:
                self.sync_viewer()
            if self.args.real_time and self.viewer is not None:
                time.sleep(self.model.opt.timestep)

    def hold_current_pose(
        self, num_steps=1, object_force_world=None, object_torque_world=None
    ):
        for arm in (self.left_arm, self.right_arm):
            tip_pos, tip_rot = self.get_tip_pose(arm)
            self.set_desired_tip_pose(arm, tip_pos, tip_rot)

        for _ in range(max(int(num_steps), 1)):
            left_tau, right_tau = self._compute_dual_arm_cartesian_impedance_control()
            self.set_control_torque(self.left_arm, left_tau)
            self.set_control_torque(self.right_arm, right_tau)
            self.apply_object_wrench_world(
                force_world=object_force_world,
                torque_world=object_torque_world,
            )
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if self.viewer is not None:
                self.sync_viewer()
            if self.args.real_time and self.viewer is not None:
                time.sleep(self.model.opt.timestep)

    def extract_object_contacts(self):
        obj_pos, _, obj_rot = self.get_object_pose()
        contacts = {"left": [], "right": []}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom_ids = {int(contact.geom1), int(contact.geom2)}
            if self.obj_geom_id not in geom_ids:
                continue
            other_geom = (
                int(contact.geom2)
                if int(contact.geom1) == self.obj_geom_id
                else int(contact.geom1)
            )
            if other_geom == self.left_arm.tip_geom_id:
                key = "left"
            elif other_geom == self.right_arm.tip_geom_id:
                key = "right"
            else:
                continue

            world_pos = np.asarray(contact.pos, dtype=np.float64).copy()
            local_pos = obj_rot.T @ (world_pos - obj_pos)
            contact_force = np.zeros(6, dtype=np.float64)
            if hasattr(mujoco, "mj_contactForce"):
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)
            normal_force = float(abs(contact_force[0]))
            tangential_force = float(np.linalg.norm(contact_force[1:3]))
            contacts[key].append(
                {
                    "world_pos": world_pos,
                    "local_pos": local_pos,
                    "dist": float(contact.dist),
                    "force_local": contact_force.copy(),
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                }
            )
        return contacts

    def _tip_target_error(self, arm, target_pos, target_rot=None):
        current_pos, current_rot = self.get_tip_pose(arm)
        pos_err = float(
            np.linalg.norm(np.asarray(target_pos, dtype=np.float64) - current_pos)
        )
        rot_err = 0.0
        if target_rot is not None:
            rot_err = float(np.linalg.norm(_rotation_error(current_rot, target_rot)))
        return pos_err, rot_err

    def _run_dual_arm_stage(
        self,
        label,
        max_steps,
        target_fn,
        pos_tol,
        rot_tol=None,
        success_fn=None,
        solve_ik_once=False,
        use_ik=True,
        fixed_joint_goals=None,
        world_mode="with_pedestal",
        waypoint_offset=None,
    ):
        rot_tol = self.args.ik_rot_tol if rot_tol is None else float(rot_tol)
        if int(max_steps) <= 0:
            return False, {}

        self._set_curobo_world_mode(world_mode)
        last_report_step = -1
        info = {}
        cached_targets = None
        cached_ik = None
        planner_sol_guess = None
        waypoint_offset = (
            self.args.planner_attract_offset
            if waypoint_offset is None
            else float(waypoint_offset)
        )
        waypoint_done = waypoint_offset <= 1e-9
        if fixed_joint_goals is None:
            left_joint_goal = None
            right_joint_goal = None
        else:
            left_joint_goal, right_joint_goal = fixed_joint_goals
        for step in range(max_steps):
            if not self.is_running():
                break

            self._update_inter_arm_worlds()
            left_curr_pos, left_curr_rot = self.get_tip_pose(self.left_arm)
            right_curr_pos, right_curr_rot = self.get_tip_pose(self.right_arm)
            if solve_ik_once:
                if cached_targets is None:
                    cached_targets = target_fn(step)
                    left_target = cached_targets["left_tip_pos"]
                    left_rot = cached_targets["left_tip_rot"]
                    right_target = cached_targets["right_tip_pos"]
                    right_rot = cached_targets["right_tip_rot"]
                    if use_ik:
                        left_ik = self.solve_arm_ik(
                            self.left_arm, left_target, left_rot
                        )
                        right_ik = self.solve_arm_ik(
                            self.right_arm, right_target, right_rot
                        )
                    else:
                        left_ik = self.build_pose_goal_result(
                            self.left_arm,
                            left_target,
                            left_rot,
                            reference_q_mj=left_joint_goal,
                        )
                        right_ik = self.build_pose_goal_result(
                            self.right_arm,
                            right_target,
                            right_rot,
                            reference_q_mj=right_joint_goal,
                        )
                    cached_ik = (left_ik, right_ik)
                else:
                    left_target = cached_targets["left_tip_pos"]
                    left_rot = cached_targets["left_tip_rot"]
                    right_target = cached_targets["right_tip_pos"]
                    right_rot = cached_targets["right_tip_rot"]
                    left_ik, right_ik = cached_ik
                stage_targets = cached_targets
            else:
                stage_targets = target_fn(step)
                left_target = stage_targets["left_tip_pos"]
                left_rot = stage_targets["left_tip_rot"]
                right_target = stage_targets["right_tip_pos"]
                right_rot = stage_targets["right_tip_rot"]
                if use_ik:
                    left_ik = self.solve_arm_ik(self.left_arm, left_target, left_rot)
                    right_ik = self.solve_arm_ik(
                        self.right_arm, right_target, right_rot
                    )
                else:
                    left_ik = self.build_pose_goal_result(
                        self.left_arm,
                        left_target,
                        left_rot,
                        reference_q_mj=left_joint_goal,
                    )
                    right_ik = self.build_pose_goal_result(
                        self.right_arm,
                        right_target,
                        right_rot,
                        reference_q_mj=right_joint_goal,
                    )

            left_attract = (
                left_ik.solved_tip_pos_world
                + stage_targets["left_outward_normal"] * waypoint_offset
            )
            right_attract = (
                right_ik.solved_tip_pos_world
                + stage_targets["right_outward_normal"] * waypoint_offset
            )

            if not waypoint_done:
                left_waypoint_dist = float(
                    np.linalg.norm(self.get_tip_pos(self.left_arm) - left_attract)
                )
                right_waypoint_dist = float(
                    np.linalg.norm(self.get_tip_pos(self.right_arm) - right_attract)
                )
                if left_waypoint_dist < float(
                    self.args.planner_attract_tol
                ) and right_waypoint_dist < float(self.args.planner_attract_tol):
                    waypoint_done = True
                    planner_sol_guess = None

            waypoint_active = not waypoint_done
            left_step_target = (
                left_attract if waypoint_active else left_ik.solved_tip_pos_world
            )
            right_step_target = (
                right_attract if waypoint_active else right_ik.solved_tip_pos_world
            )
            left_step_rot = (
                left_curr_rot if waypoint_active else left_ik.solved_tip_rot_world
            )
            right_step_rot = (
                right_curr_rot if waypoint_active else right_ik.solved_tip_rot_world
            )

            curr_x = self.get_planner_state()
            phi_vec, jac_mat = self._detect_planner_contacts()
            object_target_pos = np.asarray(
                stage_targets.get("object_target_pos", curr_x[:3]), dtype=np.float64
            )
            object_target_quat = np.asarray(
                stage_targets.get("object_target_quat", curr_x[3:7]), dtype=np.float64
            )
            planner_result = self.planner.plan_once(
                object_target_pos,
                object_target_quat,
                curr_x,
                phi_vec,
                jac_mat,
                sol_guess=planner_sol_guess,
                verify_cost_param_1=0.0 if waypoint_active else 1.0,
                verify_cost_param_2=0.0 if waypoint_active else 1.0,
                virtual_point_1=left_attract,
                virtual_point_2=right_attract,
                contact_point_1=left_ik.solved_tip_pos_world,
                contact_point_2=right_ik.solved_tip_pos_world,
            )
            planner_sol_guess = planner_result["sol_guess"]
            self.plan_params.sol_guess_ = planner_sol_guess
            action = np.asarray(planner_result["action"], dtype=np.float64).reshape(-1)
            if action.shape[0] != 6:
                raise RuntimeError(
                    f"Expected a 6D dual-arm plan_once action, got shape {action.shape}."
                )
            self.step_cartesian_action(
                action[:3],
                action[3:6],
                left_step_rot,
                right_step_rot,
            )

            self.set_marker("left_goal", left_step_target)
            self.set_marker("right_goal", right_step_target)
            self.set_marker("obj_point", object_target_pos)
            mujoco.mj_forward(self.model, self.data)
            if self.viewer is not None:
                self.sync_viewer()

            left_err, left_rot_err = self._tip_target_error(
                self.left_arm,
                left_ik.solved_tip_pos_world,
                left_ik.solved_tip_rot_world,
            )
            right_err, right_rot_err = self._tip_target_error(
                self.right_arm,
                right_ik.solved_tip_pos_world,
                right_ik.solved_tip_rot_world,
            )
            left_plan_pose_err, _ = self._tip_target_error(
                self.left_arm,
                left_step_target,
                left_ik.solved_tip_rot_world,
            )
            right_plan_pose_err, _ = self._tip_target_error(
                self.right_arm,
                right_step_target,
                right_ik.solved_tip_rot_world,
            )
            contacts = self.extract_object_contacts()
            info = {
                "step": step,
                "contacts": contacts,
                "left_force": self._max_normal_force(contacts["left"]),
                "right_force": self._max_normal_force(contacts["right"]),
                "waypoint_active": waypoint_active,
                "left_waypoint_pos_err": float(
                    np.linalg.norm(self.get_tip_pos(self.left_arm) - left_attract)
                ),
                "right_waypoint_pos_err": float(
                    np.linalg.norm(self.get_tip_pos(self.right_arm) - right_attract)
                ),
                "left_pos_err": left_err,
                "right_pos_err": right_err,
                "left_rot_err": left_rot_err,
                "right_rot_err": right_rot_err,
                "left_ik_ok": bool(left_ik.success),
                "right_ik_ok": bool(right_ik.success),
                "left_ik_pos_err": float(left_ik.position_error),
                "right_ik_pos_err": float(right_ik.position_error),
                "left_ik_rot_err": float(left_ik.rotation_error),
                "right_ik_rot_err": float(right_ik.rotation_error),
                "left_ik_constraint_total": float(left_ik.constraint_total),
                "right_ik_constraint_total": float(right_ik.constraint_total),
                "left_ik_bound_constraint": float(left_ik.bound_constraint),
                "right_ik_bound_constraint": float(right_ik.bound_constraint),
                "left_ik_world_constraint": float(left_ik.world_constraint),
                "right_ik_world_constraint": float(right_ik.world_constraint),
                "left_ik_static_world_constraint": float(
                    left_ik.static_world_constraint
                ),
                "right_ik_static_world_constraint": float(
                    right_ik.static_world_constraint
                ),
                "left_ik_self_constraint": float(left_ik.self_constraint),
                "right_ik_self_constraint": float(right_ik.self_constraint),
                "left_ik_failure_reason": str(left_ik.failure_reason),
                "right_ik_failure_reason": str(right_ik.failure_reason),
                "left_mpc_pose_err": float(left_plan_pose_err),
                "right_mpc_pose_err": float(right_plan_pose_err),
                "left_goal_q_mj": left_ik.q_mj.copy(),
                "right_goal_q_mj": right_ik.q_mj.copy(),
                "object_pos": self.get_object_pose()[0].copy(),
                "left_planner_cmd": action[:3].copy(),
                "right_planner_cmd": action[3:6].copy(),
            }
            if step == 0 or step == max_steps - 1 or step - last_report_step >= 40:
                print(
                    f"[{label}] step={step:04d} "
                    f"left_err={left_err:.4f}/{left_rot_err:.4f} "
                    f"right_err={right_err:.4f}/{right_rot_err:.4f} "
                    f"left_force={info['left_force']:.3f} right_force={info['right_force']:.3f} "
                    f"left_ik={left_ik.success} right_ik={right_ik.success} "
                    f"left_ik_res={left_ik.position_error:.4f}/{left_ik.rotation_error:.4f} "
                    f"right_ik_res={right_ik.position_error:.4f}/{right_ik.rotation_error:.4f} "
                    f"left_mpc={left_plan_pose_err:.4f} right_mpc={right_plan_pose_err:.4f}"
                )
                if step == 0:
                    print(
                        f"  stage_cfg: world_mode={world_mode} "
                        f"use_ik={use_ik} planner=plan_once waypoint_offset={waypoint_offset:.4f}"
                    )
                if not left_ik.success and left_ik.failure_reason:
                    print(
                        "  left_ik_diag: "
                        f"reason={left_ik.failure_reason} "
                        f"constraint={left_ik.constraint_total:.4f} "
                        f"bound={left_ik.bound_constraint:.4f} "
                        f"world={left_ik.world_constraint:.4f} "
                        f"static_world={left_ik.static_world_constraint:.4f} "
                        f"self={left_ik.self_constraint:.4f}"
                    )
                if not right_ik.success and right_ik.failure_reason:
                    print(
                        "  right_ik_diag: "
                        f"reason={right_ik.failure_reason} "
                        f"constraint={right_ik.constraint_total:.4f} "
                        f"bound={right_ik.bound_constraint:.4f} "
                        f"world={right_ik.world_constraint:.4f} "
                        f"static_world={right_ik.static_world_constraint:.4f} "
                        f"self={right_ik.self_constraint:.4f}"
                    )
                if waypoint_active:
                    print(
                        "  waypoint_track: "
                        f"left={info['left_waypoint_pos_err']:.4f} "
                        f"right={info['right_waypoint_pos_err']:.4f}"
                    )
                last_report_step = step

            pose_ok = (
                left_err < pos_tol
                and right_err < pos_tol
                and left_rot_err < rot_tol
                and right_rot_err < rot_tol
            )
            if success_fn is None:
                if pose_ok:
                    return True, info
            elif success_fn(info):
                return True, info

        return False, info

    def run(self):
        obj_pos, obj_quat, obj_rot = self.get_object_pose()
        gravity_local = self._gravity_wrench_local(obj_rot)
        lift_delta = np.array([0.0, 0.0, self.args.lift_height], dtype=np.float64)
        touch_offset = TIP_RADIUS + self.args.touch_offset
        squeeze_offset = max(TIP_RADIUS - self.args.squeeze_depth, 0.001)

        contact_candidate_cache = self._precompute_contact_candidate_cache(
            obj_pos,
            obj_rot,
            gravity_local,
        )
        contact_selection_state = {
            "mode": "best",
            "pending_mode": None,
            "pending_steps": 0,
        }
        (
            initial_contact_targets,
            initial_selection_debug,
        ) = self._select_live_contact_targets(
            obj_pos,
            obj_rot,
            candidate_cache=contact_candidate_cache,
            previous_targets=None,
            virtual_offset=float(self.args.planner_attract_offset),
            selection_state=contact_selection_state,
        )
        contact_selection_state["pending_mode"] = None
        contact_selection_state["pending_steps"] = 0
        contact_points_local = initial_contact_targets["contact_points_local"]
        normals_local = initial_contact_targets["normals_local"]
        contact_points_world = initial_contact_targets["contact_points_world"]
        inward_normals_world = initial_contact_targets["inward_normals_world"]
        static_result = self._copy_nested_value(
            initial_contact_targets.get("static_equilibrium", None)
        )

        self.set_marker("contact_point1", contact_points_world[0])
        self.set_marker("contact_point2", contact_points_world[1])
        self.set_marker("goal", obj_pos, obj_quat)
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.sync_viewer()

        required_normal_force = (
            0.35 * self.args.obj_mass * 9.81
            if self.args.min_normal_force is None
            else float(self.args.min_normal_force)
        )
        if static_result is not None and static_result["valid"]:
            modeled_normal = float(
                np.max(
                    np.asarray(static_result["contact_forces_local"], dtype=np.float64)[
                        :, 0
                    ]
                )
            )
            required_normal_force = max(required_normal_force, 0.5 * modeled_normal)

        cached_contact_targets = self._copy_contact_targets(initial_contact_targets)
        contact_order = np.asarray(
            cached_contact_targets.get("contact_order", np.array([0, 1], dtype=int)),
            dtype=int,
        ).reshape(-1)

        desired_contact_forces_local = np.zeros((0, 3), dtype=np.float64)
        desired_force_vectors_local = np.zeros((0, 3), dtype=np.float64)
        if static_result is not None and static_result["valid"]:
            desired_contact_forces_local = np.asarray(
                static_result["contact_forces_local"], dtype=np.float64
            ).reshape(-1, 3)
            desired_force_vectors_local = np.asarray(
                static_result["force_vectors_local"], dtype=np.float64
            ).reshape(-1, 3)
            if (
                desired_contact_forces_local.shape[0] == contact_order.shape[0]
                and contact_order.shape[0] == 2
            ):
                desired_contact_forces_local = desired_contact_forces_local[
                    contact_order
                ]
            if (
                desired_force_vectors_local.shape[0] == contact_order.shape[0]
                and contact_order.shape[0] == 2
            ):
                desired_force_vectors_local = desired_force_vectors_local[contact_order]
        else:
            desired_contact_forces_local = np.asarray(
                cached_contact_targets.get(
                    "witness_contact_forces_local", np.zeros((0, 3), dtype=np.float64)
                ),
                dtype=np.float64,
            ).reshape(-1, 3)
            desired_force_vectors_local = np.asarray(
                cached_contact_targets.get(
                    "witness_force_vectors_local", np.zeros((0, 3), dtype=np.float64)
                ),
                dtype=np.float64,
            ).reshape(-1, 3)

        if desired_contact_forces_local.shape[0] == 2:
            cached_contact_targets[
                "desired_contact_forces_local"
            ] = desired_contact_forces_local.copy()
        if desired_force_vectors_local.shape[0] == 2:
            cached_contact_targets[
                "desired_force_vectors_local"
            ] = desired_force_vectors_local.copy()
        cached_contact_targets = self._project_cached_contact_targets(
            cached_contact_targets,
            obj_pos,
            obj_rot,
            virtual_offset=float(self.args.planner_attract_offset),
        )

        desired_normal_forces = np.full(2, required_normal_force, dtype=np.float64)
        if desired_contact_forces_local.shape[0] == 2:
            desired_normal_forces = np.maximum(
                desired_normal_forces,
                np.maximum(desired_contact_forces_local[:, 0], 0.0),
            )
        force_control_stiffness = (
            float(self.args.force_control_stiffness)
            if self.args.force_control_stiffness is not None
            else float(np.max(desired_normal_forces))
            / max(float(self.args.squeeze_depth), 1e-4)
        )
        force_control_stiffness = max(force_control_stiffness, 1e-6)

        print("Generated scene:", self.scene_path)
        print("Mesh:", self.mesh_path)
        print("Scale:", self.mesh_scale)
        print("Object pose:", obj_pos, obj_quat)
        print(
            "Requested solvers:",
            f"optimizer={self.args.solver} planner={self.plan_params.planner_solver_}",
        )
        print(
            "Offline candidate cache:",
            f"regions={len(contact_candidate_cache.get('region_groups', []))} "
            f"raw_candidates={int(contact_candidate_cache.get('candidate_entry_count_before_limit', 0))} "
            f"cached_candidates={int(contact_candidate_cache.get('candidate_entry_count', 0))} "
            f"limit={contact_candidate_cache.get('candidate_entry_limit', None)} "
            f"region_precompute={float(contact_candidate_cache.get('timing', {}).get('wall_time', 0.0)):.4f}s "
            f"lift_precompute={float(contact_candidate_cache.get('offline_lift_precompute_time', 0.0)):.4f}s",
        )
        raw_contact_points_local = np.asarray(
            initial_contact_targets.get(
                "raw_contact_points_local", contact_points_local
            ),
            dtype=np.float64,
        ).reshape(-1, 3)
        print("Raw optimizer contact points local:\n", raw_contact_points_local)
        print("Arm-assigned contact points local:\n", contact_points_local)
        print("Contact points world:\n", contact_points_world)
        print("Inward normals world:\n", inward_normals_world)
        if desired_contact_forces_local.shape[0] == 2:
            print("Desired contact forces local:\n", desired_contact_forces_local)
        if desired_force_vectors_local.shape[0] == 2:
            print(
                "Desired contact force vectors in object frame:\n",
                desired_force_vectors_local,
            )
        print(
            f"Grasp score: total_cost={float(initial_contact_targets['total_cost']):.6f}, "
            f"force_closure_cost={float(initial_contact_targets.get('force_closure_cost', float('inf'))):.6f}, "
            f"lift_cost={float(initial_contact_targets.get('lift_cost', 0.0)):.6f}, "
            f"region_score={float(initial_contact_targets['region_score']):.6f}, "
            f"antipodal_margin={float(initial_contact_targets['antipodal_margin']):.6f}"
        )
        print(
            "Initial contact selection:",
            f"source={initial_selection_debug.get('selected_contact_source', 'best')} "
            f"best_cost={float(initial_selection_debug.get('best_contact_cost', float('inf'))):.6f} "
            f"nearest_cost={float(initial_selection_debug.get('nearest_contact_cost', float('inf'))):.6f} "
            f"nearest_improvement={float(initial_selection_debug.get('nearest_contact_improvement', 0.0)):.4f}",
        )
        print(f"Required normal force per fingertip: {required_normal_force:.3f} N")
        print(
            f"Force-control model: k={force_control_stiffness:.3f} "
            f"dissipation_velocity={float(self.args.force_control_dissipation_velocity):.4f} "
            f"stiction_velocity={float(self.args.force_control_stiction_velocity):.4f} "
            f"smoothing={float(self.args.force_control_smoothing):.6f}"
        )
        if static_result is not None:
            print(
                f"Static equilibrium: valid={static_result['valid']} "
                f"residual_norm={static_result['residual_norm']:.6f} "
                f"solve_time={static_result['solve_time']:.4f}s"
            )

        print(
            "Stage 1: solve IK once for the initial contact fingertip poses and keep the solved rotations fixed"
        )
        self._set_curobo_world_mode("floor_only")
        self._update_inter_arm_worlds()
        initial_touch_targets = self._stage_targets_from_object_pose(
            contact_points_local,
            normals_local,
            obj_pos,
            obj_rot,
            center_offset=touch_offset,
        )
        left_touch_ik = self.solve_arm_ik(
            self.left_arm,
            initial_touch_targets["left_tip_pos"],
            initial_touch_targets["left_tip_rot"],
        )
        right_touch_ik = self.solve_arm_ik(
            self.right_arm,
            initial_touch_targets["right_tip_pos"],
            initial_touch_targets["right_tip_rot"],
        )
        left_fixed_rot = _project_to_rotation_matrix(left_touch_ik.solved_tip_rot_world)
        right_fixed_rot = _project_to_rotation_matrix(
            right_touch_ik.solved_tip_rot_world
        )
        print(
            f"  left(success={left_touch_ik.success}, pos={left_touch_ik.position_error:.4f}, rot={left_touch_ik.rotation_error:.4f}) "
            f"right(success={right_touch_ik.success}, pos={right_touch_ik.position_error:.4f}, rot={right_touch_ik.rotation_error:.4f})"
        )
        if not left_touch_ik.success and left_touch_ik.failure_reason:
            print(
                f"  left diag: reason={left_touch_ik.failure_reason} "
                f"constraint={left_touch_ik.constraint_total:.4f}"
            )
        if not right_touch_ik.success and right_touch_ik.failure_reason:
            print(
                f"  right diag: reason={right_touch_ik.failure_reason} "
                f"constraint={right_touch_ik.constraint_total:.4f}"
            )

        print(
            "Stage 2: reuse the offline candidate cache, evaluate force-closure online, and switch between cached best contacts and current nearest projections"
        )

        stage2_ok, _stage2_info = self._run_live_contact_plan_stage(
            "attract",
            self.args.approach_steps,
            left_fixed_rot,
            right_fixed_rot,
            verify_cost_1=0.0,
            verify_cost_2=0.0,
            virtual_offset=float(self.args.planner_attract_offset),
            goal_offset=float(self.args.planner_attract_offset),
            pos_tol=float(self.args.planner_attract_tol),
            rot_tol=max(float(self.args.ik_rot_tol), 0.25),
            world_mode="with_pedestal",
            initial_contact_targets=cached_contact_targets,
            contact_candidate_cache=contact_candidate_cache,
            contact_selection_state=contact_selection_state,
        )
        if not stage2_ok:
            print(
                "Attract stage reached its step limit; proceeding directly to the squeeze stage with the current pose."
            )
        if _stage2_info.get("contact_targets") is not None:
            cached_contact_targets = self._copy_contact_targets(
                _stage2_info["contact_targets"]
            )

        print(
            "Stage 4: skip the old verify-contact stage and directly squeeze until both fingertips apply the desired contact force"
        )
        stable_contact_steps = 0
        force_stage_state = {"state": None}

        def force_stage_targets(step, live_targets):
            del step
            (
                goal_points_world,
                force_debug,
                next_force_state,
            ) = self._build_force_control_targets(
                live_targets["contact_points_world"],
                live_targets["outward_normals_world"],
                desired_normal_forces,
                previous_state=force_stage_state["state"],
                contact_stiffness=force_control_stiffness,
                dissipation_velocity=float(
                    self.args.force_control_dissipation_velocity
                ),
                stiction_velocity=float(self.args.force_control_stiction_velocity),
                smoothing_factor=float(self.args.force_control_smoothing),
            )
            if "desired_contact_force_world" in live_targets:
                force_debug["desired_contact_force_world"] = np.asarray(
                    live_targets["desired_contact_force_world"],
                    dtype=np.float64,
                ).reshape(2, 3)
            force_stage_state["state"] = next_force_state

            def post_step_debug():
                (
                    _,
                    post_force_debug,
                    post_force_state,
                ) = self._build_force_control_targets(
                    live_targets["contact_points_world"],
                    live_targets["outward_normals_world"],
                    desired_normal_forces,
                    previous_state=next_force_state,
                    contact_stiffness=force_control_stiffness,
                    dissipation_velocity=float(
                        self.args.force_control_dissipation_velocity
                    ),
                    stiction_velocity=float(self.args.force_control_stiction_velocity),
                    smoothing_factor=float(self.args.force_control_smoothing),
                )
                if "desired_contact_force_world" in live_targets:
                    post_force_debug["desired_contact_force_world"] = np.asarray(
                        live_targets["desired_contact_force_world"],
                        dtype=np.float64,
                    ).reshape(2, 3)
                force_stage_state["state"] = post_force_state
                return post_force_debug

            return {
                "goal_points_world": goal_points_world.copy(),
                "planner_contact_points_world": goal_points_world.copy(),
                "planner_virtual_points_world": goal_points_world.copy(),
                "debug": force_debug,
                "post_step_debug_fn": post_step_debug,
            }

        def squeeze_success(info):
            nonlocal stable_contact_steps
            modeled_normal_forces = np.asarray(
                info.get("modeled_normal_forces", np.zeros(2, dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1)
            force_ready = modeled_normal_forces.size == 2 and bool(
                np.all(modeled_normal_forces >= desired_normal_forces)
            )
            stable_contact_steps = stable_contact_steps + 1 if force_ready else 0
            return stable_contact_steps >= self.args.contact_stable_steps

        squeeze_ok, squeeze_info = self._run_live_contact_plan_stage(
            "squeeze",
            self.args.squeeze_steps,
            left_fixed_rot,
            right_fixed_rot,
            verify_cost_1=1.0,
            verify_cost_2=1.0,
            virtual_offset=float(self.args.planner_attract_offset),
            goal_offset=squeeze_offset,
            pos_tol=self.args.target_tol * 1.5,
            rot_tol=self.args.ik_rot_tol * 1.5,
            success_fn=squeeze_success,
            world_mode="floor_only",
            initial_contact_targets=cached_contact_targets,
            contact_candidate_cache=contact_candidate_cache,
            contact_selection_state=contact_selection_state,
            planner_target_fn=force_stage_targets,
        )
        if not squeeze_ok and self.args.squeeze_extra_steps > 0:
            print(
                f"Squeeze stage did not reach the required force within {self.args.squeeze_steps} steps; "
                f"extending by {self.args.squeeze_extra_steps} more steps."
            )
            squeeze_ok, squeeze_info = self._run_live_contact_plan_stage(
                "squeeze-extend",
                self.args.squeeze_extra_steps,
                left_fixed_rot,
                right_fixed_rot,
                verify_cost_1=1.0,
                verify_cost_2=1.0,
                virtual_offset=float(self.args.planner_attract_offset),
                goal_offset=squeeze_offset,
                pos_tol=self.args.target_tol * 1.5,
                rot_tol=self.args.ik_rot_tol * 1.5,
                success_fn=squeeze_success,
                world_mode="floor_only",
                initial_contact_targets=cached_contact_targets,
                contact_candidate_cache=contact_candidate_cache,
                contact_selection_state=contact_selection_state,
                planner_target_fn=force_stage_targets,
            )
        if squeeze_info.get("contact_targets") is not None:
            cached_contact_targets = self._copy_contact_targets(
                squeeze_info["contact_targets"]
            )
        if not squeeze_ok:
            print("Squeeze stage did not reach the required bilateral contact force.")
            return

        contacts = squeeze_info["contacts"]
        for key, marker_name in (
            ("left", "contact_point1"),
            ("right", "contact_point2"),
        ):
            if contacts[key]:
                best_contact = max(
                    contacts[key], key=lambda item: item.get("normal_force", 0.0)
                )
                self.set_marker(marker_name, best_contact["world_pos"])
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.sync_viewer()

        print(
            "Measured contacts after squeeze:",
            {key: len(value) for key, value in contacts.items()},
        )
        if "modeled_normal_forces" in squeeze_info:
            modeled_normal_forces = np.asarray(
                squeeze_info["modeled_normal_forces"], dtype=np.float64
            ).reshape(2)
            print(
                "Modeled squeeze forces:",
                f"left={modeled_normal_forces[0]:.4f}N right={modeled_normal_forces[1]:.4f}N",
            )
        for side in ("left", "right"):
            if contacts[side]:
                best_contact = max(
                    contacts[side], key=lambda item: item.get("normal_force", 0.0)
                )
                print(
                    f"  {side}: world={np.array2string(best_contact['world_pos'], precision=4)} "
                    f"local={np.array2string(best_contact['local_pos'], precision=4)} "
                    f"dist={best_contact['dist']:.6f} "
                    f"normal_force={best_contact['normal_force']:.4f}"
                )

        print(
            "Stage 5: lift while keeping force-driven contact tracking and the initial IK rotations"
        )
        lift_start_pos, lift_start_quat, _ = self.get_object_pose()
        target_lift_height = (
            obj_pos[2] + self.args.lift_height - self.args.lift_success_margin
        )

        def lift_success(info):
            contact_force_ok = (
                info["left_force"] > 0.1 * required_normal_force
                and info["right_force"] > 0.1 * required_normal_force
            )
            return info["object_pos"][2] >= target_lift_height and contact_force_ok

        def lift_target(step, _curr_pos, _curr_quat, _curr_rot):
            alpha = min(1.0, float(step + 1) / max(1, self.args.lift_steps))
            desired_pos = lift_start_pos + alpha * lift_delta
            return desired_pos, lift_start_quat.copy()

        lift_ok, _ = self._run_live_contact_plan_stage(
            "lift",
            self.args.lift_steps,
            left_fixed_rot,
            right_fixed_rot,
            verify_cost_1=1.0,
            verify_cost_2=1.0,
            virtual_offset=float(self.args.planner_attract_offset),
            goal_offset=squeeze_offset,
            pos_tol=self.args.target_tol * 2.0,
            rot_tol=self.args.ik_rot_tol * 1.5,
            success_fn=lift_success,
            world_mode="floor_only",
            object_target_fn=lift_target,
            initial_contact_targets=cached_contact_targets,
            contact_candidate_cache=contact_candidate_cache,
            contact_selection_state=contact_selection_state,
            planner_target_fn=force_stage_targets,
        )

        final_obj_pos, final_obj_quat, _ = self.get_object_pose()
        lifted_height = final_obj_pos[2] - obj_pos[2]
        print(
            f"Lift result: success={lift_ok} final_height_gain={lifted_height:.4f} "
            f"target={self.args.lift_height:.4f}"
        )
        print("Final object pose:", final_obj_pos, final_obj_quat)

        if lift_ok and self.args.hold_steps > 0:
            hold_torque_world = None
            if bool(self.args.test_force):
                _, _, hold_rot = self.get_object_pose()
                _, hold_torque_world = self._best_object_wrench_world(
                    cached_contact_targets,
                    hold_rot,
                )
            self.hold_current_pose(
                self.args.hold_steps,
                object_torque_world=hold_torque_world,
            )


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Dual Panda MuJoCo grasp demo driven by mlqp_point_v2, cuRobo IK, and plan_once tracking."
    )
    parser.add_argument(
        "--obj",
        type=str,
        default="elephant",
        help="Object asset name in envs/assets/objects.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Absolute or relative mesh path. Overrides --obj.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs=3,
        default=None,
        help="Mesh scale factors sx sy sz.",
    )
    parser.add_argument(
        "--obj-mass",
        type=float,
        default=0.01,
        help="Object mass used in MuJoCo and grasp scoring.",
    )
    parser.add_argument(
        "--arm-friction",
        type=float,
        default=5.0,
        help="Friction coefficient used by the MuJoCo/planner contact model and the Stage 4/5 tangential force estimate.",
    )
    parser.add_argument(
        "--optimizer-arm-friction",
        type=float,
        default=0.9,
        help="Friction coefficient passed to mlqp_point_v2 contact selection. Matches planning/mlqp_point_v2.py default.",
    )
    parser.add_argument(
        "--object-friction",
        type=float,
        default=5.0,
        help="Sliding friction coefficient used by the MuJoCo object geom.",
    )
    parser.add_argument(
        "--contact-stiffness",
        type=float,
        default=12.5,
        help="Contact stiffness passed to mlqp_point_v2.",
    )
    parser.add_argument(
        "--sample-num",
        type=int,
        default=70,
        help="Surface samples used by mlqp_point_v2. Matches planning/mlqp_point_v2.py default.",
    )
    parser.add_argument(
        "--optimizer-use-support-filter",
        action="store_true",
        help="Enable the old support-aware candidate filtering before calling mlqp_point_v2. Disabled by default to match standalone mlqp_point_v2.py behavior.",
    )
    parser.add_argument(
        "--optimizer-curvature-neighbor-k",
        type=int,
        default=8,
        help="Neighbor count used to estimate local surface curvature in mlqp_point_v2.",
    )
    parser.add_argument(
        "--optimizer-max-region-mean-curvature",
        type=float,
        default=0.12,
        help="Reject candidate contact regions whose mean curvature score exceeds this value.",
    )
    parser.add_argument(
        "--optimizer-max-point-curvature",
        type=float,
        default=0.25,
        help="Reject candidate contact regions that contain points above this curvature score.",
    )
    parser.add_argument(
        "--optimizer-curvature-penalty-weight",
        type=float,
        default=1.5,
        help="Quality penalty weight applied to higher-curvature regions in mlqp_point_v2.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=("ipopt", "snopt", "acados"),
        default="acados",
        help="Solver backend used by mlqp_point_v2 for force-closure and static-equilibrium optimization.",
    )
    parser.add_argument(
        "--print-contact-timing",
        action="store_true",
        help="Enable detailed timing prints from mlqp_point_v2 region search / candidate evaluation.",
    )
    parser.add_argument(
        "--cached-contact-candidate-limit",
        type=int,
        default=10,
        help="Number of offline contact-pair candidates kept in the Stage 2 cache. Set <= 0 to keep all candidates.",
    )
    parser.add_argument(
        "--pos-coef",
        type=float,
        default=1.0,
        help="Position coefficient for mlqp_point_v2.",
    )
    parser.add_argument(
        "--ori-coef",
        type=float,
        default=0.0005,
        help="Orientation coefficient for mlqp_point_v2.",
    )
    parser.add_argument(
        "--scene-center-x",
        type=float,
        default=0.58,
        help="Midpoint between the two Panda bases.",
    )
    parser.add_argument(
        "--robot-span",
        type=float,
        default=0.75,
        help="Distance between the two Panda bases.",
    )  # 間距
    parser.add_argument(
        "--pedestal-pos",
        type=float,
        nargs=3,
        default=(0.58, 0.0, 0.06),
        help="Central pedestal position.",
    )
    parser.add_argument(
        "--pedestal-size",
        type=float,
        nargs=3,
        default=(0.05, 0.07, 0.06),
        help="Central pedestal half sizes.",
    )
    parser.add_argument(
        "--object-yaw", type=float, default=0.0, help="Initial object yaw in radians."
    )
    parser.add_argument(
        "--object-z-offset",
        type=float,
        default=0.0,
        help="Extra object height above the pedestal.",
    )
    parser.add_argument(
        "--initial-object-lift",
        type=float,
        default=0.2,
        help="Extra height added to the pedestal/support under the object.",
    )  # 臺子高度
    parser.add_argument(
        "--ground-height-margin",
        type=float,
        default=0.003,
        help="Margin above support top for visible point filtering.",
    )
    parser.add_argument(
        "--support-normal-alignment-threshold",
        type=float,
        default=0.25,
        help="Normal alignment threshold used to reject support-facing bottom contacts.",
    )
    parser.add_argument(
        "--pregrasp-offset",
        type=float,
        default=0.01,
        help="Extra stand-off beyond the fingertip radius.",
    )
    parser.add_argument(
        "--touch-offset",
        type=float,
        default=0.004,
        help="Stand-off used for initial touch.",
    )
    parser.add_argument(
        "--squeeze-depth",
        type=float,
        default=0.003,
        help="Inward squeeze depth relative to fingertip radius.",
    )
    parser.add_argument(
        "--fingertip-max-pitch",
        type=float,
        default=0.35,
        help="Maximum inward pitch, in radians, allowed away from the vertical-down fingertip pose.",
    )
    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.06,
        help="Lift distance after the squeeze stage.",
    )
    parser.add_argument(
        "--approach-steps",
        type=int,
        default=1000,
        help="Simulation steps for the pregrasp stage.",
    )
    parser.add_argument(
        "--touch-steps",
        type=int,
        default=320,
        help="Legacy option kept for CLI compatibility; the explicit touch/verify stage is no longer used.",
    )
    parser.add_argument(
        "--squeeze-steps",
        type=int,
        default=480,
        help="Simulation steps for the squeeze stage.",
    )
    parser.add_argument(
        "--squeeze-extra-steps",
        type=int,
        default=360,
        help="Extra squeeze steps automatically used if the first squeeze window is not enough.",
    )
    parser.add_argument(
        "--lift-steps",
        type=int,
        default=320,
        help="Simulation steps for the lift stage.",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=0,
        help="Extra simulation steps after lifting.",
    )
    parser.add_argument(
        "--target-tol",
        type=float,
        default=0.006,
        help="Tip target tolerance in meters.",
    )
    parser.add_argument(
        "--ik-pos-tol",
        type=float,
        default=0.0015,
        help="cuRobo IK position threshold in meters.",
    )
    parser.add_argument(
        "--ik-rot-tol",
        type=float,
        default=0.10,
        help="cuRobo IK rotation threshold in radians.",
    )
    parser.add_argument(
        "--ik-max-iters",
        type=int,
        default=120,
        help="Maximum cuRobo gradient iterations used by each IK solve.",
    )
    parser.add_argument(
        "--ik-num-seeds", type=int, default=32, help="Number of cuRobo IK seeds."
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=0.05,
        help="Legacy option kept for CLI compatibility; unused with cuRobo IK.",
    )
    parser.add_argument(
        "--ik-step-scale",
        type=float,
        default=0.7,
        help="Legacy option kept for CLI compatibility; unused with cuRobo IK.",
    )
    parser.add_argument(
        "--ik-home-weight",
        type=float,
        default=0.01,
        help="Legacy option kept for CLI compatibility; unused with cuRobo IK.",
    )
    parser.add_argument(
        "--ik-pos-weight",
        type=float,
        default=1.0,
        help="Legacy option kept for CLI compatibility; unused with cuRobo IK.",
    )
    parser.add_argument(
        "--ik-rot-weight",
        type=float,
        default=0.35,
        help="Legacy option kept for CLI compatibility; unused with cuRobo IK.",
    )
    parser.add_argument(
        "--curobo-robot-cfg",
        type=str,
        default="franka.yml",
        help="Robot config passed to cuRobo.",
    )
    parser.add_argument(
        "--mujoco-dt",
        type=float,
        default=0.01,
        help="MuJoCo simulation timestep in seconds.",
    )
    parser.add_argument(
        "--mj-steps-per-command",
        type=int,
        default=1,
        help="Number of MuJoCo steps executed after each MPC command.",
    )
    parser.add_argument(
        "--command-substeps",
        type=int,
        default=1,
        help="Multiplier used when matching cuRobo MPC dt to the MuJoCo control cadence.",
    )
    parser.add_argument(
        "--curobo-collision-activation-distance",
        type=float,
        default=0.06,
        help="Collision activation distance passed to cuRobo.",
    )
    parser.add_argument(
        "--disable-curobo-self-collision",
        action="store_true",
        help="Disable cuRobo self-collision checking.",
    )
    parser.add_argument(
        "--disable-curobo-cuda-graph",
        action="store_true",
        help="Disable cuRobo CUDA graph capture.",
    )
    parser.add_argument(
        "--pose-only-mpc",
        action="store_true",
        help="Legacy option kept for CLI compatibility; plan_once tracking is pose-based by default.",
    )
    parser.add_argument(
        "--planner-dt",
        type=float,
        default=0.01,
        help="Time step used by the plan_once object-motion model.",
    )
    parser.add_argument(
        "--planner-horizon", type=int, default=20, help="plan_once horizon length."
    )
    parser.add_argument(
        "--planner-solver",
        type=str,
        choices=("ipopt", "acados"),
        default="acados",
        help="Solver backend used by self.planner.plan_once.",
    )
    parser.add_argument(
        "--planner-max-contacts",
        type=int,
        default=15,
        help="Maximum object contacts modeled by plan_once.",
    )
    parser.add_argument(
        "--planner-cmd-limit",
        type=float,
        default=0.05,
        help="Per-step Cartesian delta limit in meters for each arm.",
    )
    parser.add_argument(
        "--planner-attract-offset",
        type=float,
        default=0.025,
        help="Outward offset from the IK fingertip pose used as the first attract waypoint.",
    )
    parser.add_argument(
        "--planner-attract-tol",
        type=float,
        default=0.03,
        help="Distance threshold for switching from attract points to the IK contact pose.",
    )
    parser.add_argument(
        "--planner-attract-coef",
        type=float,
        default=0.5,
        help="Attract cost coefficient for plan_once.",
    )
    parser.add_argument(
        "--planner-reject-coef",
        type=float,
        default=0.001,
        help="Reject cost coefficient for plan_once.",
    )
    parser.add_argument(
        "--planner-contact-coef",
        type=float,
        default=0.7,
        help="Contact cost coefficient for plan_once.",
    )
    parser.add_argument(
        "--planner-contact-cost-param",
        type=float,
        default=0.0,
        help="Blend factor inside the dual-arm contact cost.",
    )
    parser.add_argument(
        "--planner-reject-distance",
        type=float,
        default=0.005,
        help="Reject distance threshold used by plan_once.",
    )
    parser.add_argument(
        "--planner-object-inertia-pos",
        type=float,
        default=40.0,
        help="Translational object inertia weight used by plan_once.",
    )
    parser.add_argument(
        "--planner-object-inertia-rot",
        type=float,
        default=0.05,
        help="Rotational object inertia weight used by plan_once.",
    )
    parser.add_argument(
        "--planner-robot-stiffness",
        type=float,
        default=300.0,
        help="Cartesian point stiffness used by the plan_once robot model.",
    )
    parser.add_argument(
        "--mppi-samples",
        type=int,
        default=256,
        help="Number of sampled trajectories used by plan_once.",
    )
    parser.add_argument(
        "--mppi-iterations",
        type=int,
        default=4,
        help="Number of MPPI update iterations after warm start.",
    )
    parser.add_argument(
        "--mppi-init-iterations",
        type=int,
        default=8,
        help="Number of MPPI iterations used before a warm start exists.",
    )
    parser.add_argument(
        "--mppi-lambda", type=float, default=1.0, help="MPPI temperature."
    )
    parser.add_argument(
        "--mppi-noise-sigma",
        type=float,
        default=0.005,
        help="Action noise sigma for MPPI.",
    )
    parser.add_argument(
        "--mppi-noise-decay",
        type=float,
        default=0.85,
        help="Per-iteration MPPI noise decay.",
    )
    parser.add_argument(
        "--mppi-elite-frac",
        type=float,
        default=0.1,
        help="Elite fraction used by MPPI weighting.",
    )
    parser.add_argument(
        "--mppi-device",
        type=str,
        default=None,
        help="Torch device used by MPPI, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--mppi-use-torch-compile",
        action="store_true",
        help="Enable torch.compile for the MPPI kernels when available.",
    )
    parser.add_argument(
        "--cartesian-stiffness-pos",
        type=float,
        default=500.0,
        help="Translational stiffness used by the Cartesian impedance controller.",
    )
    parser.add_argument(
        "--cartesian-stiffness-rot",
        type=float,
        default=50.0,
        help="Rotational stiffness used by the Cartesian impedance controller.",
    )
    parser.add_argument(
        "--nullspace-stiffness",
        type=float,
        default=10.0,
        help="Nullspace stiffness used by the Cartesian impedance controller.",
    )
    parser.add_argument(
        "--grasp-lift-cost-weight",
        type=float,
        default=1.0,
        help="Weight on the offline static-equilibrium objective used to prefer contact pairs that can lift the object.",
    )
    parser.add_argument(
        "--grasp-lift-residual-weight",
        type=float,
        default=10.0,
        help="Weight on the offline static-equilibrium residual norm used to reject weak lift candidates.",
    )
    parser.add_argument(
        "--nearest-contact-low-improvement",
        type=float,
        default=0.75,
        help="SCSP-style lower hysteresis threshold for falling back from current nearest-point grasping to cached best contacts.",
    )
    parser.add_argument(
        "--nearest-contact-high-improvement",
        type=float,
        default=0.85,
        help="SCSP-style upper hysteresis threshold for switching from cached best contacts to the current nearest-point pair.",
    )
    parser.add_argument(
        "--nearest-contact-switch-steps",
        type=int,
        default=5,
        help="Number of consecutive planner iterations required before switching between nearest-point and cached best-contact modes.",
    )
    parser.add_argument(
        "--min-normal-force",
        type=float,
        default=None,
        help="Required normal force per fingertip before lifting. Defaults to a mass-based value.",
    )
    parser.add_argument(
        "--force-control-stiffness",
        type=float,
        default=None,
        help="Normal stiffness used by the Stage 4/5 spring model. Defaults to required_force / squeeze_depth.",
    )
    parser.add_argument(
        "--force-control-dissipation-velocity",
        type=float,
        default=0.1,
        help="Normal dissipation velocity used by the Stage 4/5 spring model.",
    )
    parser.add_argument(
        "--force-control-stiction-velocity",
        type=float,
        default=0.05,
        help="Tangential velocity regularization used by the Stage 4/5 spring model.",
    )
    parser.add_argument(
        "--force-control-smoothing",
        type=float,
        default=0.0,
        help="Optional softplus smoothing used by the Stage 4/5 spring model.",
    )
    parser.add_argument(
        "--contact-stable-steps",
        type=int,
        default=15,
        help="Number of consecutive squeeze steps that must satisfy the normal-force threshold.",
    )
    parser.add_argument(
        "--lift-success-margin",
        type=float,
        default=0.005,
        help="Allowed height error when deciding whether the lift succeeded.",
    )
    parser.add_argument(
        "--test_force",
        "--test-force",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_arg,
        help="If true, project the current best optimizer wrench to world coordinates and apply it directly to the object every simulation step.",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Launch the MuJoCo passive viewer."
    )
    parser.add_argument(
        "--print-viewer-camera-state",
        action="store_true",
        help="Print the passive-viewer camera state on every viewer sync. Disabled by default to avoid slowing down the control loop.",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Sleep to approximate real-time playback when visualizing.",
    )
    parser.add_argument(
        "--scene-output",
        type=Path,
        default=GENERATED_SCENE_PATH,
        help="Path where the generated dual-Panda scene XML will be written.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    grasper = BimanualPandaGrasper(args)
    try:
        grasper.run()
    finally:
        grasper.close()


if __name__ == "__main__":
    main()
