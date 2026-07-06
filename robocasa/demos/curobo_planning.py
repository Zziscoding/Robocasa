"""cuRobo planning helpers backed by MuJoCo scene geometry.

This module intentionally stays independent from the drawer demo logic.  It
converts the current MuJoCo environment into a cuRobo ``WorldConfig`` and
provides a cached joint-goal planner that plans directly from ``start_q`` to a
known ``goal_q``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import mujoco
import numpy as np


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat.reshape(9))
    return quat


def _geom_name(model, geom_id: int) -> str:
    try:
        return str(model.geom_id2name(int(geom_id)) or f"geom_{int(geom_id)}")
    except Exception:
        return f"geom_{int(geom_id)}"


def _body_name(model, body_id: int) -> str:
    try:
        return str(model.body_id2name(int(body_id)) or f"body_{int(body_id)}")
    except Exception:
        return f"body_{int(body_id)}"


def _split_names(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


def _robot_body_ids(env) -> set[int]:
    model = env.sim.model
    prefixes = {"robot0", "panda", "gripper0"}
    for robot in getattr(env, "robots", []) or []:
        robot_model = getattr(robot, "robot_model", None)
        prefix = getattr(robot_model, "naming_prefix", "")
        if prefix:
            prefixes.add(str(prefix).rstrip("_"))
    ids: set[int] = set()
    for body_id in range(int(model.nbody)):
        name = _body_name(model, body_id)
        if any(name.startswith(prefix) for prefix in prefixes):
            ids.add(int(body_id))
    return ids


def _body_is_in_subtree(model, body_id: int, roots: set[int]) -> bool:
    current = int(body_id)
    while current >= 0:
        if current in roots:
            return True
        parent = int(model.body_parentid[current]) if current < int(model.nbody) else -1
        if parent == current:
            break
        current = parent
    return False


_MESH_CONVEX_HULL_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _mesh_convex_hull_local(model, geom_id: int):
    """Return (vertices, faces) of the mesh's convex hull, in geom-local frame.

    MuJoCo's default mesh-mesh narrow phase uses the convex hull of each mesh
    (this is what mink / `data.contact[]` sees). cuRobo was previously given
    only an OBB enclosing the mesh, which is much looser than the hull and
    caused false start/goal-in-collision rejections for grasp poses that sit
    inside an OBB but outside the true hull (e.g. drawer handles with a
    finger-sized cavity). Returning the hull aligns cuRobo's collision world
    with MuJoCo's.

    Hulls only depend on the mesh asset, not on geom_xpos, so we cache by
    (id(model), mesh_id) — cheap and safe across re-plans.
    """

    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None
    cache_key = (id(model), mesh_id)
    cached = _MESH_CONVEX_HULL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    if vert_count < 4:
        return None
    vertices = np.asarray(
        model.mesh_vert[vert_start : vert_start + vert_count], dtype=np.float64
    )
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:
        return None
    try:
        hull = ConvexHull(vertices)
    except Exception:  # QhullError, degenerate (coplanar) verts, etc.
        return None
    hull_vertex_indices = np.asarray(hull.vertices, dtype=np.int64)
    hull_vertices = vertices[hull_vertex_indices]
    remap = -np.ones(vertices.shape[0], dtype=np.int64)
    remap[hull_vertex_indices] = np.arange(hull_vertex_indices.shape[0], dtype=np.int64)
    hull_faces = remap[np.asarray(hull.simplices, dtype=np.int64)]
    if np.any(hull_faces < 0):
        return None
    result = (hull_vertices.astype(np.float64), hull_faces.astype(np.int64))
    _MESH_CONVEX_HULL_CACHE[cache_key] = result
    return result


def _mesh_world_aabb(model, data, geom_id: int):
    """Tight oriented bounding box for a mesh geom in world coordinates.

    `model.mesh_vert` is already in the mesh's final compiled scale — MuJoCo
    bakes the asset `<mesh scale=...>` into the vertices at compile time. The
    previous implementation multiplied the vertices by `model.geom_size`, but
    for mesh geoms MuJoCo populates `geom_size` with the mesh AABB half-extents
    (NOT a scale factor), so that line double-scaled every vertex by the
    bounding-box half-size and produced a cuboid whose dims were completely
    detached from the real mesh extents. cuRobo then planned through obstacles
    that MuJoCo/mink saw as solid.

    We also return the OBB (axis-aligned in the GEOM frame, not the world
    frame). That keeps long thin handles tight even when rotated.
    """

    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    if vert_count <= 0:
        return None
    vertices = np.asarray(
        model.mesh_vert[vert_start : vert_start + vert_count], dtype=np.float64
    )
    # AABB in the mesh/geom local frame.
    lower_local = np.min(vertices, axis=0)
    upper_local = np.max(vertices, axis=0)
    center_local = 0.5 * (lower_local + upper_local)
    dims = np.maximum(upper_local - lower_local, 1e-4)
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3)
    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    center_world = pos + rot @ center_local
    return center_world, rot, dims


def _geom_as_cuboid_pose_dims(model, data, geom_id: int, *, padding: float):
    geom_type = int(model.geom_type[geom_id])
    if geom_type in (
        int(mujoco.mjtGeom.mjGEOM_PLANE),
        int(mujoco.mjtGeom.mjGEOM_HFIELD),
    ):
        return None
    pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3)
    rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64).reshape(3)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        dims = 2.0 * size
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        dims = np.full(3, 2.0 * float(size[0]), dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        dims = 2.0 * size
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        dims = np.array([2.0 * size[0], 2.0 * size[0], 2.0 * size[1]], dtype=np.float64)
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        dims = np.array(
            [2.0 * size[0], 2.0 * size[0], 2.0 * (size[1] + size[0])],
            dtype=np.float64,
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_box = _mesh_world_aabb(model, data, geom_id)
        if mesh_box is None:
            return None
        pos, rot, dims = mesh_box
    else:
        return None
    dims = np.maximum(np.asarray(dims, dtype=np.float64) + 2.0 * float(padding), 1e-4)
    quat = _matrix_to_quat_wxyz(rot)
    return pos, quat, dims


def _pose_in_frame(pos_world, rot_world, frame_pos_world, frame_rot_world):
    frame_pos_world = np.asarray(frame_pos_world, dtype=np.float64).reshape(3)
    frame_rot_world = np.asarray(frame_rot_world, dtype=np.float64).reshape(3, 3)
    pos_world = np.asarray(pos_world, dtype=np.float64).reshape(3)
    rot_world = np.asarray(rot_world, dtype=np.float64).reshape(3, 3)
    return (
        frame_rot_world.T @ (pos_world - frame_pos_world),
        frame_rot_world.T @ rot_world,
    )


@dataclass(frozen=True)
class CuroboWorldBuildResult:
    world_config: Any
    obstacle_names: tuple[str, ...]
    signature: str


def build_world_config_from_mujoco(
    env,
    *,
    frame_pos_world: np.ndarray | None = None,
    frame_rot_world: np.ndarray | None = None,
    include_geom_names: Iterable[str] | None = None,
    exclude_geom_names: Iterable[str] | None = None,
    exclude_body_names: Iterable[str] | None = None,
    exclude_robot: bool = True,
    padding: float = 0.005,
    max_obstacles: int | None = None,
) -> CuroboWorldBuildResult:
    """Build a cuRobo ``WorldConfig`` from the current MuJoCo geom poses.

    Non-box primitives and meshes are conservatively approximated as cuboids.
    Mesh geoms use a world-axis-aligned bounding box, which is intentionally
    conservative and cheap.
    """

    from curobo.geom.types import Cuboid, Mesh, WorldConfig

    model = env.sim.model
    data = env.sim.data
    if frame_pos_world is not None:
        frame_pos_world = np.asarray(frame_pos_world, dtype=np.float64).reshape(3)
    if frame_rot_world is not None:
        frame_rot_world = np.asarray(frame_rot_world, dtype=np.float64).reshape(3, 3)
    include = _split_names(include_geom_names)
    exclude = _split_names(exclude_geom_names)
    exclude_bodies = _split_names(exclude_body_names)
    robot_roots = _robot_body_ids(env) if exclude_robot else set()
    explicit_body_roots = {
        int(model.body_name2id(name))
        for name in exclude_bodies
        if name in getattr(model, "_body_name2id", {})
    }

    cuboids = []
    meshes = []
    names = []
    digest = hashlib.sha1()
    mesh_geom_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    for geom_id in range(int(model.ngeom)):
        name = _geom_name(model, geom_id)
        if include and name not in include:
            continue
        if name in exclude:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if exclude_robot and _body_is_in_subtree(model, body_id, robot_roots):
            continue
        if explicit_body_roots and _body_is_in_subtree(
            model, body_id, explicit_body_roots
        ):
            continue
        # Skip visual-only geoms (no collision filter bits). MuJoCo's convention
        # is that a geom contributes to collisions iff (contype & conaffinity)
        # is nonzero with some other geom; the cheap proxy is "either contype
        # or conaffinity is set". Visual-only meshes (e.g. *_vis duplicates)
        # have both at 0 and were previously inflating world_obstacle_count
        # and squeezing free space until cuRobo returned GRAPH_FAIL.
        if (
            int(model.geom_contype[geom_id]) == 0
            and int(model.geom_conaffinity[geom_id]) == 0
        ):
            continue

        is_mesh = int(model.geom_type[geom_id]) == mesh_geom_type
        hull = _mesh_convex_hull_local(model, geom_id) if is_mesh else None
        if is_mesh and hull is not None:
            hull_vertices_local, hull_faces = hull
            pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64).reshape(3)
            rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            quat = _matrix_to_quat_wxyz(rot)
            if frame_pos_world is not None and frame_rot_world is not None:
                pos, rot_frame = _pose_in_frame(
                    pos, rot, frame_pos_world, frame_rot_world
                )
                quat = _matrix_to_quat_wxyz(rot_frame)
            if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
                continue
            obstacle_name = f"mj_{geom_id}_{name}".replace("/", "_")
            meshes.append(
                Mesh(
                    name=obstacle_name,
                    pose=[*pos.tolist(), *quat.tolist()],
                    vertices=hull_vertices_local.tolist(),
                    faces=hull_faces.tolist(),
                )
            )
            names.append(obstacle_name)
            digest.update(obstacle_name.encode("utf-8"))
            digest.update(np.asarray(pos, dtype=np.float32).tobytes())
            digest.update(np.asarray(quat, dtype=np.float32).tobytes())
            digest.update(hull_vertices_local.astype(np.float32).tobytes())
            digest.update(hull_faces.astype(np.int32).tobytes())
            if max_obstacles is not None and (len(cuboids) + len(meshes)) >= int(
                max_obstacles
            ):
                break
            continue

        pose_dims = _geom_as_cuboid_pose_dims(model, data, geom_id, padding=padding)
        if pose_dims is None:
            continue
        pos, quat, dims = pose_dims
        if frame_pos_world is not None and frame_rot_world is not None:
            rot_world = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rot_world, quat)
            pos, rot_frame = _pose_in_frame(
                pos,
                rot_world.reshape(3, 3),
                frame_pos_world,
                frame_rot_world,
            )
            quat = _matrix_to_quat_wxyz(rot_frame)
        if not (
            np.all(np.isfinite(pos))
            and np.all(np.isfinite(quat))
            and np.all(np.isfinite(dims))
        ):
            continue
        if np.any(dims <= 0.0):
            continue
        obstacle_name = f"mj_{geom_id}_{name}".replace("/", "_")
        cuboids.append(
            Cuboid(
                name=obstacle_name,
                pose=[*pos.tolist(), *quat.tolist()],
                dims=dims.tolist(),
            )
        )
        names.append(obstacle_name)
        digest.update(obstacle_name.encode("utf-8"))
        digest.update(np.asarray(pos, dtype=np.float32).tobytes())
        digest.update(np.asarray(quat, dtype=np.float32).tobytes())
        digest.update(np.asarray(dims, dtype=np.float32).tobytes())
        if max_obstacles is not None and (len(cuboids) + len(meshes)) >= int(
            max_obstacles
        ):
            break
    return CuroboWorldBuildResult(
        world_config=WorldConfig(cuboid=cuboids, mesh=meshes),
        obstacle_names=tuple(names),
        signature=digest.hexdigest(),
    )


def _compute_curobo_base_pose_in_world(
    motion_gen,
    tensor_args,
    robot_state: Mapping[str, Any],
    curobo_joint_names: tuple[str, ...],
):
    from robocasa.demos import demo_close_drawer_contact_curobo as close_demo
    from curobo.types.state import JointState

    robosuite_joint_names = tuple(
        robot_state.get("robocasa_joint_names", close_demo.PANDA_JOINT_NAMES)
    )
    q_robosuite = np.asarray(robot_state["q"], dtype=np.float64).reshape(1, 7)
    q_curobo = close_demo._reorder_q(
        q_robosuite,
        robosuite_joint_names,
        curobo_joint_names,
    ).reshape(1, 7)
    current_state = JointState.from_position(tensor_args.to_device(q_curobo))
    current_kin = motion_gen.compute_kinematics(current_state)
    (
        current_ee_position,
        current_ee_quaternion,
    ) = close_demo._kinematic_model_ee_components(current_kin)
    curobo_hand_pos = close_demo._tensor_to_numpy(current_ee_position).reshape(-1, 3)[0]
    curobo_hand_quat = close_demo._tensor_to_numpy(current_ee_quaternion).reshape(
        -1, 4
    )[0]
    curobo_hand_rot = close_demo._matrix_from_quat_wxyz(curobo_hand_quat)

    robosuite_hand_pos_base = np.asarray(
        robot_state["hand_pos_base"], dtype=np.float64
    ).reshape(3)
    robosuite_hand_rot_base = np.asarray(
        robot_state["hand_rot_base"], dtype=np.float64
    ).reshape(3, 3)
    curobo_base_pos_in_robosuite, curobo_base_rot_in_robosuite = close_demo._pose_mul(
        robosuite_hand_pos_base,
        robosuite_hand_rot_base,
        *close_demo._pose_inv(curobo_hand_pos, curobo_hand_rot),
    )
    robosuite_base_pos_world = np.asarray(
        robot_state["base_pos"], dtype=np.float64
    ).reshape(3)
    robosuite_base_rot_world = np.asarray(
        robot_state["base_rot"], dtype=np.float64
    ).reshape(3, 3)
    curobo_base_pos_world = (
        robosuite_base_pos_world
        + robosuite_base_rot_world @ curobo_base_pos_in_robosuite
    )
    curobo_base_rot_world = robosuite_base_rot_world @ curobo_base_rot_in_robosuite
    return curobo_base_pos_world, curobo_base_rot_world


def cached_motion_gen_for_env(
    env,
    args,
    *,
    robot_state: Mapping[str, Any],
    cache_attr: str = "_curobo_joint_goal_planner",
    extra_exclude_body_names: Iterable[str] | None = None,
):
    """Return a cached cuRobo ``MotionGen`` configured with the MuJoCo world."""

    from robocasa.demos import demo_close_drawer_contact_curobo as close_demo

    close_demo._ensure_curobo_importable()
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    base_signature = (
        str(getattr(args, "curobo_robot_cfg", "franka.yml")),
        int(getattr(args, "curobo_trajopt_tsteps", 32)),
        float(getattr(args, "curobo_interpolation_dt", 0.02)),
        int(getattr(args, "curobo_ik_seeds", 16)),
        int(getattr(args, "curobo_graph_seeds", 2)),
        int(getattr(args, "curobo_trajopt_seeds", 2)),
        bool(getattr(args, "disable_curobo_self_collision", False)),
        bool(getattr(args, "disable_curobo_cuda_graph", False)),
        float(getattr(args, "curobo_collision_activation_distance", 0.005)),
    )
    cached = getattr(args, cache_attr, None)
    bootstrap_motion_gen = None
    bootstrap_tensor_args = None
    bootstrap_joint_names = None
    if cached is not None and tuple(cached.get("base_signature", ())) == base_signature:
        bootstrap_motion_gen = cached["motion_gen"]
        bootstrap_tensor_args = cached["tensor_args"]
        bootstrap_joint_names = tuple(cached["curobo_joint_names"])
    if bootstrap_motion_gen is None:
        bootstrap_tensor_args = TensorDeviceType()
        bootstrap_config = MotionGenConfig.load_from_robot_config(
            base_signature[0],
            None,
            bootstrap_tensor_args,
            trajopt_tsteps=base_signature[1],
            interpolation_dt=base_signature[2],
            use_cuda_graph=not base_signature[7],
            self_collision_check=not base_signature[6],
            self_collision_opt=not base_signature[6],
            num_ik_seeds=base_signature[3],
            num_graph_seeds=base_signature[4],
            num_trajopt_seeds=base_signature[5],
            collision_activation_distance=base_signature[8],
            collision_checker_type=None,
        )
        bootstrap_motion_gen = MotionGen(bootstrap_config)
        bootstrap_joint_names = tuple(
            close_demo._extract_curobo_joint_names(bootstrap_motion_gen)
        )

    curobo_base_pos_world, curobo_base_rot_world = _compute_curobo_base_pose_in_world(
        bootstrap_motion_gen,
        bootstrap_tensor_args,
        robot_state,
        tuple(bootstrap_joint_names),
    )
    base_exclude_bodies = _split_names(getattr(args, "curobo_world_exclude_bodies", ""))
    extra_exclude = _split_names(extra_exclude_body_names)
    merged_exclude_bodies = tuple(sorted(set(base_exclude_bodies) | set(extra_exclude)))
    world_result = build_world_config_from_mujoco(
        env,
        frame_pos_world=curobo_base_pos_world,
        frame_rot_world=curobo_base_rot_world,
        exclude_geom_names=getattr(args, "curobo_world_exclude_geoms", ""),
        exclude_body_names=merged_exclude_bodies,
        exclude_robot=True,
        padding=float(getattr(args, "curobo_world_padding", 0.005)),
        max_obstacles=getattr(args, "curobo_world_max_obstacles", None),
    )
    signature = (
        *base_signature,
        str(world_result.signature),
    )
    if cached is not None and cached.get("signature") == signature:
        return cached

    tensor_args = TensorDeviceType()
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        signature[0],
        world_result.world_config,
        tensor_args,
        trajopt_tsteps=signature[1],
        interpolation_dt=signature[2],
        use_cuda_graph=not signature[7],
        self_collision_check=not signature[6],
        self_collision_opt=not signature[6],
        num_ik_seeds=signature[3],
        num_graph_seeds=signature[4],
        num_trajopt_seeds=signature[5],
        collision_activation_distance=signature[8],
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(enable_graph=not signature[7])
    cached = {
        "signature": signature,
        "base_signature": base_signature,
        "tensor_args": tensor_args,
        "motion_gen": motion_gen,
        "curobo_joint_names": tuple(close_demo._extract_curobo_joint_names(motion_gen)),
        "world_obstacle_names": world_result.obstacle_names,
        "world_signature": world_result.signature,
        "curobo_base_pos_world": curobo_base_pos_world.tolist(),
        "curobo_base_quat_wxyz_world": _matrix_to_quat_wxyz(
            curobo_base_rot_world
        ).tolist(),
    }
    setattr(args, cache_attr, cached)
    return cached


def _joint_goal_failure_diagnostics(
    env,
    robot_state: Mapping[str, Any],
    q_start: np.ndarray,
    q_goal: np.ndarray,
    cached: Mapping[str, Any],
    *,
    name: str,
) -> str:
    from robocasa.demos import demo_close_drawer_contact_curobo as close_demo

    model = env.sim.model
    data = env.sim.data
    arm_joint_names = tuple(
        robot_state.get("robocasa_joint_names", close_demo.PANDA_JOINT_NAMES)
    )

    def geom_name(geom_id: int) -> str:
        try:
            return str(model.geom_id2name(int(geom_id)) or f"geom_{int(geom_id)}")
        except Exception:
            return f"geom_{int(geom_id)}"

    robot_geom_ids = {
        int(geom_id)
        for geom_name_value, geom_id in getattr(model, "_geom_name2id", {}).items()
        if (
            str(geom_name_value).startswith("robot0_")
            or str(geom_name_value).startswith("gripper0_")
            or str(geom_name_value).startswith("panda")
        )
        and "collision" in str(geom_name_value)
    }

    qpos_saved = data.qpos.copy()
    qvel_saved = data.qvel.copy()

    def contacts_for(q_arm: np.ndarray) -> list[str]:
        close_demo._set_env_arm_q(
            env, arm_joint_names, np.asarray(q_arm, dtype=np.float64)
        )
        env.sim.forward()
        contacts = []
        for contact_idx in range(int(data.ncon)):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (
                robot_geom_ids
                and geom1 not in robot_geom_ids
                and geom2 not in robot_geom_ids
            ):
                continue
            contacts.append((float(contact.dist), geom_name(geom1), geom_name(geom2)))
        contacts.sort(key=lambda item: item[0])
        return [
            f"{left}--{right}:dist={dist:.6f}" for dist, left, right in contacts[:8]
        ]

    try:
        start_contacts = contacts_for(q_start)
        goal_contacts = contacts_for(q_goal)
    finally:
        data.qpos[:] = qpos_saved
        data.qvel[:] = qvel_saved
        env.sim.forward()

    world_names = tuple(cached.get("world_obstacle_names", ()))
    lines = [
        f"[cuRobo joint-goal diagnostics] name={name}",
        "q_start_robosuite="
        + np.array2string(np.asarray(q_start, dtype=np.float64), precision=6),
        "q_goal_robosuite="
        + np.array2string(np.asarray(q_goal, dtype=np.float64), precision=6),
        f"world_obstacle_count={len(world_names)}",
        f"world_signature={cached.get('world_signature', '')}",
        "curobo_base_pos_world="
        + np.array2string(
            np.asarray(cached.get("curobo_base_pos_world", []), dtype=np.float64),
            precision=6,
        ),
        "mujoco_start_robot_contacts="
        + ("; ".join(start_contacts) if start_contacts else "none"),
        "mujoco_goal_robot_contacts="
        + ("; ".join(goal_contacts) if goal_contacts else "none"),
    ]
    if world_names:
        lines.append("world_obstacles_head=" + ", ".join(world_names[:20]))
    return "\n".join(lines)


def plan_joint_goal(
    env,
    robot_state: Mapping[str, Any],
    goal_q_robosuite: np.ndarray,
    args,
    *,
    name: str,
    extra_exclude_body_names: Iterable[str] | None = None,
):
    """Plan directly from current robosuite q to a known robosuite goal q."""

    from robocasa.demos import demo_close_drawer_contact_curobo as close_demo

    cached = cached_motion_gen_for_env(
        env,
        args,
        robot_state=robot_state,
        extra_exclude_body_names=extra_exclude_body_names,
    )
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    tensor_args = cached["tensor_args"]
    motion_gen = cached["motion_gen"]
    curobo_joint_names = tuple(cached["curobo_joint_names"])
    robosuite_joint_names = tuple(
        robot_state.get("robocasa_joint_names", close_demo.PANDA_JOINT_NAMES)
    )
    q_start_robosuite = np.asarray(robot_state["q"], dtype=np.float64).reshape(1, 7)
    q_goal_robosuite = np.asarray(goal_q_robosuite, dtype=np.float64).reshape(1, 7)
    q_start_curobo = close_demo._reorder_q(
        q_start_robosuite, robosuite_joint_names, curobo_joint_names
    ).reshape(1, 7)
    q_goal_curobo = close_demo._reorder_q(
        q_goal_robosuite, robosuite_joint_names, curobo_joint_names
    ).reshape(1, 7)
    graph_attempt = getattr(args, "curobo_joint_enable_graph_attempt", None)
    if graph_attempt is not None:
        graph_attempt = int(graph_attempt)
    start_state = JointState.from_position(tensor_args.to_device(q_start_curobo))
    goal_state = JointState.from_position(tensor_args.to_device(q_goal_curobo))
    result = motion_gen.plan_single_js(
        start_state,
        goal_state,
        plan_config=MotionGenPlanConfig(
            max_attempts=int(getattr(args, "curobo_joint_max_attempts", 6)),
            timeout=float(getattr(args, "curobo_joint_timeout", 5.0)),
            enable_graph_attempt=graph_attempt,
            enable_graph=bool(getattr(args, "curobo_joint_enable_graph", False)),
            disable_graph_attempt=getattr(
                args, "curobo_joint_disable_graph_attempt", None
            ),
            enable_finetune_trajopt=bool(
                getattr(args, "curobo_joint_enable_finetune_trajopt", False)
            ),
            check_start_validity=bool(
                getattr(args, "curobo_joint_check_start_validity", False)
            ),
        ),
    )
    success = bool(close_demo._tensor_to_numpy(result.success).reshape(-1)[0])
    if not success and bool(getattr(args, "curobo_joint_retry_graph", True)):
        result = motion_gen.plan_single_js(
            start_state,
            goal_state,
            plan_config=MotionGenPlanConfig(
                max_attempts=int(getattr(args, "curobo_joint_graph_max_attempts", 2)),
                timeout=float(getattr(args, "curobo_joint_graph_timeout", 8.0)),
                enable_graph=True,
                enable_graph_attempt=None,
                enable_finetune_trajopt=bool(
                    getattr(args, "curobo_joint_enable_finetune_trajopt", False)
                ),
                check_start_validity=bool(
                    getattr(args, "curobo_joint_check_start_validity", False)
                ),
            ),
        )
        success = bool(close_demo._tensor_to_numpy(result.success).reshape(-1)[0])
    if not success:
        diagnostics = _joint_goal_failure_diagnostics(
            env,
            robot_state,
            q_start_robosuite.reshape(7),
            q_goal_robosuite.reshape(7),
            cached,
            name=name,
        )
        # Non-fatal fallback: cuRobo could not find a feasible joint-space
        # trajopt path, but we still want SOMETHING to visualize. Emit a
        # straight-line joint-space interpolation from start to goal and
        # warn loudly. Set `args.curobo_joint_require_success = True` to
        # restore the hard failure.
        if bool(getattr(args, "curobo_joint_require_success", False)):
            raise RuntimeError(
                f"cuRobo joint-space plan failed for '{name}': {result.status}\n"
                f"{diagnostics}"
            )
        import sys as _sys

        print(
            f"[curobo_joint] WARNING: plan failed for '{name}' "
            f"(status={result.status}); falling back to linear joint-space "
            f"interpolation for visualization.\n{diagnostics}",
            file=_sys.__stdout__,
            flush=True,
        )
        fallback_steps = max(int(getattr(args, "curobo_joint_fallback_steps", 40)), 2)
        alphas = np.linspace(0.0, 1.0, fallback_steps, dtype=np.float64)[:, None]
        q_plan_robosuite = (1.0 - alphas) * q_start_robosuite.reshape(
            1, 7
        ) + alphas * q_goal_robosuite.reshape(1, 7)
        return q_plan_robosuite, [
            {
                "name": name,
                "steps": int(q_plan_robosuite.shape[0]),
                "planner": "linear_joint_interpolation_fallback",
                "curobo_joint_names": tuple(curobo_joint_names),
                "robosuite_joint_names": tuple(robosuite_joint_names),
                "goal_q_robosuite": q_goal_robosuite.reshape(7).tolist(),
                "world_obstacle_count": int(len(cached["world_obstacle_names"])),
                "world_signature": str(cached["world_signature"]),
                "curobo_base_pos_world": list(cached["curobo_base_pos_world"]),
                "curobo_base_quat_wxyz_world": list(
                    cached["curobo_base_quat_wxyz_world"]
                ),
                "status": f"fallback_after:{result.status}",
                "terminal_collision_allowed": True,
            }
        ]
    interpolated = result.get_interpolated_plan()
    if interpolated is None:
        q_plan_curobo = close_demo._tensor_to_numpy(getattr(result, "raw_plan", None))
    else:
        q_plan_curobo = close_demo._tensor_to_numpy(interpolated.position)
    q_plan_curobo = np.asarray(q_plan_curobo, dtype=np.float64).reshape(-1, 7)
    q_plan_robosuite = close_demo._reorder_q(
        q_plan_curobo, curobo_joint_names, robosuite_joint_names
    ).reshape(-1, 7)
    return q_plan_robosuite, [
        {
            "name": name,
            "steps": int(q_plan_robosuite.shape[0]),
            "planner": "curobo_joint_goal_world",
            "curobo_joint_names": tuple(curobo_joint_names),
            "robosuite_joint_names": tuple(robosuite_joint_names),
            "goal_q_robosuite": q_goal_robosuite.reshape(7).tolist(),
            "world_obstacle_count": int(len(cached["world_obstacle_names"])),
            "world_signature": str(cached["world_signature"]),
            "curobo_base_pos_world": list(cached["curobo_base_pos_world"]),
            "curobo_base_quat_wxyz_world": list(cached["curobo_base_quat_wxyz_world"]),
            "status": str(result.status),
            "terminal_collision_allowed": True,
        }
    ]
