"""MuJoCo viewer diagnostics for RoboCasa demos."""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _GhostGeom:
    geom_type: int
    size: np.ndarray
    local_pos: np.ndarray
    local_rot: np.ndarray


@dataclass
class PopupVisualizationControl:
    """Mutable Enter-key state shared by one sequence of popup windows."""

    skip_remaining: bool = False


def _quat_wxyz_to_matrix(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _raw_model_data(env):
    model = env.sim.model
    data = env.sim.data
    return (
        model._model if hasattr(model, "_model") else model,
        data._data if hasattr(data, "_data") else data,
    )


def _name_maps(env):
    model = env.sim.model
    body_name2id = getattr(model, "_body_name2id", {})
    geom_name2id = getattr(model, "_geom_name2id", {})
    body_id2name = {int(v): k for k, v in body_name2id.items()}
    geom_id2name = {int(v): k for k, v in geom_name2id.items()}
    return body_name2id, geom_name2id, body_id2name, geom_id2name


def _descendant_body_ids(raw_model, root_ids):
    result = {int(body_id) for body_id in root_ids}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, int(raw_model.nbody)):
            parent = int(raw_model.body_parentid[body_id])
            if parent in result and body_id not in result:
                result.add(body_id)
                changed = True
    return result


def _ghost_source_body_ids(env):
    raw_model, _ = _raw_model_data(env)
    body_name2id, _, body_id2name, _ = _name_maps(env)
    root_ids = [
        body_id
        for name, body_id in body_name2id.items()
        if any(
            token in name
            for token in (
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
                "right_hand",
            )
        )
    ]
    if root_ids:
        return _descendant_body_ids(raw_model, root_ids)
    return {
        body_id
        for body_id, name in body_id2name.items()
        if any(
            token in name for token in ("hand", "leftfinger", "rightfinger", "finger")
        )
    }


def _site_pose(env, site_name: str):
    model = env.sim.model
    data = env.sim.data
    if site_name not in model._site_name2id:
        raise RuntimeError(f"Cannot visualize ghost EE: site '{site_name}' not found.")
    site_id = int(model.site_name2id(site_name))
    return (
        np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3).copy(),
        np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def _mesh_bbox_half_center(
    raw_model, geom_id: int, fallback_size
) -> tuple[np.ndarray, np.ndarray]:
    try:
        mesh_id = int(raw_model.geom_dataid[geom_id])
        vadr = int(raw_model.mesh_vertadr[mesh_id])
        vnum = int(raw_model.mesh_vertnum[mesh_id])
        verts = np.asarray(raw_model.mesh_vert[vadr : vadr + vnum], dtype=np.float64)
        if verts.size:
            mesh_min = np.min(verts, axis=0)
            mesh_max = np.max(verts, axis=0)
            half = 0.5 * (mesh_max - mesh_min)
            center = 0.5 * (mesh_max + mesh_min)
            return (
                np.maximum(half, np.array([0.003, 0.003, 0.003], dtype=np.float64)),
                np.asarray(center, dtype=np.float64).reshape(3),
            )
    except Exception:
        pass
    return (
        np.maximum(
            np.asarray(fallback_size, dtype=np.float64).reshape(3),
            np.array([0.003, 0.003, 0.003], dtype=np.float64),
        ),
        np.zeros(3, dtype=np.float64),
    )


def _extract_hand_finger_ghost_geoms(env, ee_site_name: str):
    import mujoco

    raw_model, raw_data = _raw_model_data(env)
    _, _, _, geom_id2name = _name_maps(env)
    body_ids = _ghost_source_body_ids(env)
    site_pos, site_rot = _site_pose(env, ee_site_name)
    geoms: list[_GhostGeom] = []
    for geom_id in range(int(raw_model.ngeom)):
        body_id = int(raw_model.geom_bodyid[geom_id])
        geom_name = geom_id2name.get(geom_id, "")
        if body_id not in body_ids and not any(
            token in geom_name
            for token in ("panda_hand", "panda_leftfinger", "panda_rightfinger")
        ):
            continue
        geom_type = int(raw_model.geom_type[geom_id])
        size = np.asarray(raw_model.geom_size[geom_id], dtype=np.float64).copy()
        draw_type = geom_type
        local_center = np.zeros(3, dtype=np.float64)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            draw_type = int(mujoco.mjtGeom.mjGEOM_BOX)
            size, local_center = _mesh_bbox_half_center(raw_model, geom_id, size)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        geom_pos = np.asarray(raw_data.geom_xpos[geom_id], dtype=np.float64).reshape(3)
        geom_rot = np.asarray(raw_data.geom_xmat[geom_id], dtype=np.float64).reshape(
            3, 3
        )
        geom_pos = geom_pos + geom_rot @ local_center
        geoms.append(
            _GhostGeom(
                geom_type=draw_type,
                size=size,
                local_pos=site_rot.T @ (geom_pos - site_pos),
                local_rot=site_rot.T @ geom_rot,
            )
        )
    if not geoms:
        raise RuntimeError(
            "Cannot find panda hand/finger geoms for ghost EE visualization."
        )
    return geoms


def _ensure_viewer(env):
    if getattr(env, "viewer", None) is None:
        try:
            env.render()
        except Exception:
            pass
    try:
        env.viewer.update()
    except Exception:
        pass
    viewer = getattr(getattr(env, "viewer", None), "viewer", None)
    if viewer is None:
        raise RuntimeError(
            "MuJoCo viewer is unavailable. Create the env with has_renderer=True "
            "or run with a render/visualization flag."
        )
    return viewer


def _add_ghost_geom(scene, ghost: _GhostGeom, target_pos, target_rot, rgba):
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    geom_pos = target_pos + target_rot @ ghost.local_pos
    geom_rot = target_rot @ ghost.local_rot
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        ghost.geom_type,
        np.asarray(ghost.size, dtype=np.float64),
        np.asarray(geom_pos, dtype=np.float64),
        np.asarray(geom_rot, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_scene_sphere(scene, position, radius, rgba):
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(position, dtype=np.float64).reshape(3),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_scene_line(scene, start, end, width, rgba):
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        float(width),
        np.asarray(start, dtype=np.float64).reshape(3),
        np.asarray(end, dtype=np.float64).reshape(3),
    )
    scene.ngeom += 1


def _copy_mujoco_data(source, target):
    for name in (
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "eq_active",
        "userdata",
    ):
        if not hasattr(source, name) or not hasattr(target, name):
            continue
        source_value = np.asarray(getattr(source, name))
        target_value = getattr(target, name)
        if np.asarray(target_value).shape == source_value.shape:
            target_value[...] = source_value
    target.time = float(source.time)


def _clone_model_data(env):
    import mujoco

    raw_model, raw_data = _raw_model_data(env)
    handle = tempfile.NamedTemporaryFile(suffix=".mjb", delete=False)
    model_path = handle.name
    handle.close()
    try:
        mujoco.mj_saveModel(raw_model, model_path, None)
        model = mujoco.MjModel.from_binary_path(model_path)
    finally:
        try:
            os.unlink(model_path)
        except OSError:
            pass
    data = mujoco.MjData(model)
    _copy_mujoco_data(raw_data, data)
    mujoco.mj_forward(model, data)
    return model, data


def _is_enter_key(keycode):
    return int(keycode) in (10, 13, 257, 335)


def _run_popup(
    env,
    *,
    draw_scene,
    control: PopupVisualizationControl,
    camera_lookat,
    camera_distance,
    camera_azimuth,
    camera_elevation,
    fps,
):
    import mujoco.viewer

    if control.skip_remaining:
        return "skip"
    model, data = _clone_model_data(env)

    def key_callback(keycode):
        if _is_enter_key(keycode):
            control.skip_remaining = True

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.type = 0
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = np.asarray(camera_lookat, dtype=np.float64).reshape(3)
        viewer.cam.distance = float(camera_distance)
        viewer.cam.azimuth = float(camera_azimuth)
        viewer.cam.elevation = float(camera_elevation)
        while viewer.is_running() and not control.skip_remaining:
            viewer.user_scn.ngeom = 0
            draw_scene(viewer.user_scn)
            viewer.sync()
            time.sleep(1.0 / max(float(fps), 1.0))
    return "skip" if control.skip_remaining else "closed"


def visualize_feasible_graph_popup(
    env,
    points_world,
    edges,
    *,
    control=None,
    point_radius=0.008,
    line_width=3.0,
    camera_lookat=None,
    camera_distance=0.9,
    camera_azimuth=135.0,
    camera_elevation=-25.0,
    fps=30.0,
):
    """Show feasible points and non-penetrating graph edges in a popup."""

    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    control = control or PopupVisualizationControl()
    if points.shape[0] == 0 or control.skip_remaining:
        return control
    if camera_lookat is None:
        camera_lookat = points.mean(axis=0)

    def draw(scene):
        for point in points:
            _add_scene_sphere(
                scene,
                point,
                point_radius,
                np.asarray([0.1, 1.0, 0.2, 1.0], dtype=np.float32),
            )
        for first, second in edges:
            _add_scene_line(
                scene,
                points[int(first)],
                points[int(second)],
                line_width,
                np.asarray([0.0, 0.9, 0.1, 1.0], dtype=np.float32),
            )

    _run_popup(
        env,
        draw_scene=draw,
        control=control,
        camera_lookat=camera_lookat,
        camera_distance=camera_distance,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
        fps=fps,
    )
    return control


def _ghost_geoms_for_mode(
    env,
    ee_site_name,
    finger_joint_qpos_ids,
    finger_joint_qpos,
):
    raw_model, raw_data = _raw_model_data(env)
    saved_qpos = np.asarray(raw_data.qpos, dtype=np.float64).copy()
    try:
        if finger_joint_qpos_ids is not None:
            ids = np.asarray(finger_joint_qpos_ids, dtype=np.int64).reshape(-1)
            values = np.asarray(finger_joint_qpos, dtype=np.float64).reshape(-1)
            raw_data.qpos[ids] = values
            import mujoco

            mujoco.mj_forward(raw_model, raw_data)
        return _extract_hand_finger_ghost_geoms(env, ee_site_name)
    finally:
        raw_data.qpos[:] = saved_qpos
        import mujoco

        mujoco.mj_forward(raw_model, raw_data)


def visualize_ee_pose_solution_popups(
    env,
    solutions,
    *,
    ee_site_name,
    control=None,
    alpha=0.55,
    camera_distance=0.75,
    camera_azimuth=135.0,
    camera_elevation=-20.0,
    fps=30.0,
):
    """Show solved EE poses one popup at a time.

    Closing a window advances to the next solution. Pressing Enter skips all
    remaining solution windows and returns to the planner.
    """

    control = control or PopupVisualizationControl()
    palette = {
        "gripper_open": np.asarray([0.1, 0.75, 1.0, alpha], dtype=np.float32),
        "gripper_closed": np.asarray([1.0, 0.5, 0.1, alpha], dtype=np.float32),
        "collision_sphere": np.asarray([0.2, 0.85, 1.0, alpha], dtype=np.float32),
    }
    ghost_cache = {}
    for solution_index, solution in enumerate(solutions):
        if control.skip_remaining:
            break
        pose = np.asarray(solution["pose"], dtype=np.float64).reshape(7)
        mode = str(solution.get("mode", ""))
        finger_joint_qpos = solution.get("finger_joint_qpos")
        if finger_joint_qpos is None:
            finger_joint_qpos_key = ()
        else:
            finger_joint_qpos_key = tuple(
                np.asarray(
                    finger_joint_qpos,
                    dtype=np.float64,
                )
                .reshape(-1)
                .tolist()
            )
        cache_key = (
            mode,
            finger_joint_qpos_key,
        )
        show_ghost_ee = mode != "collision_sphere"
        if show_ghost_ee:
            if cache_key not in ghost_cache:
                ghost_cache[cache_key] = _ghost_geoms_for_mode(
                    env,
                    ee_site_name,
                    solution.get("finger_joint_qpos_ids"),
                    solution.get("finger_joint_qpos"),
                )
            ghost_geoms = ghost_cache[cache_key]
        else:
            ghost_geoms = ()
        color = palette.get(
            mode,
            np.asarray([0.2, 1.0, 0.3, alpha], dtype=np.float32),
        )
        print(
            "[mpc_pose_visualization] "
            f"solution={solution_index + 1}/{len(solutions)} "
            f"job={solution.get('job_name', '')} "
            f"candidate={int(solution.get('candidate_index', -1))} "
            f"cost={float(solution.get('cost', np.nan)):.6f}. "
            "Close the window for the next solution; press Enter to skip.",
            flush=True,
        )

        sphere_centers_ee = np.asarray(
            solution.get("collision_sphere_centers_ee", np.zeros((0, 3))),
            dtype=np.float64,
        ).reshape(-1, 3)
        sphere_radii = np.asarray(
            solution.get("collision_sphere_radii", np.zeros(0)),
            dtype=np.float64,
        ).reshape(-1)
        selected_sphere_index = int(solution.get("selected_sphere_index", -1))
        representative_points = np.asarray(
            solution.get("representative_points_world", np.zeros((0, 3))),
            dtype=np.float64,
        ).reshape(-1, 3)
        feasible_points = np.asarray(
            solution.get("feasible_points_world", np.zeros((0, 3))),
            dtype=np.float64,
        ).reshape(-1, 3)
        contact_point = np.asarray(
            solution.get("contact_point_world", np.full(3, np.nan)),
            dtype=np.float64,
        ).reshape(3)

        def draw(
            scene,
            pose=pose,
            ghost_geoms=ghost_geoms,
            color=color,
            sphere_centers_ee=sphere_centers_ee,
            sphere_radii=sphere_radii,
            selected_sphere_index=selected_sphere_index,
            representative_points=representative_points,
            feasible_points=feasible_points,
            contact_point=contact_point,
        ):
            target_position = pose[:3]
            target_rotation = _quat_wxyz_to_matrix(pose[3:])
            for ghost in ghost_geoms:
                _add_ghost_geom(
                    scene,
                    ghost,
                    target_position,
                    target_rotation,
                    color,
                )
            for point in representative_points:
                _add_scene_sphere(
                    scene,
                    point,
                    0.002,
                    np.asarray([0.65, 0.65, 0.65, 0.45], dtype=np.float32),
                )
            for point in feasible_points:
                _add_scene_sphere(
                    scene,
                    point,
                    0.006,
                    np.asarray([1.0, 0.82, 0.05, 0.9], dtype=np.float32),
                )
            for sphere_index, (center_ee, radius) in enumerate(
                zip(sphere_centers_ee, sphere_radii)
            ):
                center_world = target_position + target_rotation @ center_ee
                sphere_color = (
                    np.asarray([0.1, 1.0, 0.2, 0.6], dtype=np.float32)
                    if sphere_index == selected_sphere_index
                    else np.asarray([0.1, 0.65, 1.0, 0.28], dtype=np.float32)
                )
                _add_scene_sphere(
                    scene,
                    center_world,
                    float(radius),
                    sphere_color,
                )
            if np.all(np.isfinite(contact_point)):
                _add_scene_sphere(
                    scene,
                    contact_point,
                    0.01,
                    np.asarray([1.0, 0.1, 0.1, 1.0], dtype=np.float32),
                )

        _run_popup(
            env,
            draw_scene=draw,
            control=control,
            camera_lookat=(
                contact_point if np.all(np.isfinite(contact_point)) else pose[:3]
            ),
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            fps=fps,
        )
    return control


def _sample_pose_indices(poses, count: int, seed: int):
    pose_count = int(np.asarray(poses).reshape(-1, 7).shape[0])
    count = min(max(int(count), 1), pose_count)
    rng = np.random.default_rng(int(seed))
    return rng.choice(pose_count, size=count, replace=False)


def visualize_precontact_ghost_ees(
    env,
    precontact_poses,
    *,
    ee_site_name: str,
    seed: int = 0,
    count: int = 5,
    seconds: float = 0.0,
    fps: float = 30.0,
    alpha: float = 0.32,
    camera_lookat=None,
    camera_distance: float = 0.9,
    camera_azimuth: float = 135.0,
    camera_elevation: float = -25.0,
):
    """Visualize sampled pre-contact ghost EEs in the MuJoCo viewer.

    The ghost hand/finger geoms are drawn into ``viewer.user_scn`` only. They
    are not inserted into the physics model, so they have no collision volume.
    """

    poses = np.asarray(precontact_poses, dtype=np.float64).reshape(-1, 7)
    if poses.shape[0] == 0:
        return
    viewer = _ensure_viewer(env)
    ghost_geoms = _extract_hand_finger_ghost_geoms(env, ee_site_name)
    indices = _sample_pose_indices(poses, count=count, seed=seed)
    palette = np.asarray(
        [
            [0.1, 0.7, 1.0, alpha],
            [1.0, 0.55, 0.05, alpha],
            [0.4, 1.0, 0.25, alpha],
            [1.0, 0.2, 0.7, alpha],
            [0.9, 0.9, 0.1, alpha],
        ],
        dtype=np.float32,
    )
    if hasattr(viewer, "cam"):
        viewer.cam.type = 0
        viewer.cam.fixedcamid = -1
        if camera_lookat is None:
            camera_lookat = np.mean(poses[indices, :3], axis=0)
        viewer.cam.lookat[:] = np.asarray(camera_lookat, dtype=np.float64).reshape(3)
        viewer.cam.distance = float(camera_distance)
        viewer.cam.azimuth = float(camera_azimuth)
        viewer.cam.elevation = float(camera_elevation)

    started_at = time.time()
    while True:
        if hasattr(viewer, "is_running") and not viewer.is_running():
            break
        if hasattr(viewer, "user_scn"):
            viewer.user_scn.ngeom = 0
            for color_index, pose_index in enumerate(indices):
                pose = poses[int(pose_index)]
                target_pos = pose[:3]
                target_rot = _quat_wxyz_to_matrix(pose[3:])
                rgba = palette[color_index % len(palette)]
                for ghost in ghost_geoms:
                    _add_ghost_geom(
                        viewer.user_scn, ghost, target_pos, target_rot, rgba
                    )
        if hasattr(viewer, "sync"):
            viewer.sync()
        if seconds > 0.0 and time.time() - started_at >= float(seconds):
            break
        if seconds <= 0.0:
            time.sleep(1.0 / max(float(fps), 1.0))
            continue
        time.sleep(1.0 / max(float(fps), 1.0))


__all__ = [
    "PopupVisualizationControl",
    "visualize_ee_pose_solution_popups",
    "visualize_feasible_graph_popup",
    "visualize_precontact_ghost_ees",
]
