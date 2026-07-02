from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robocasa.demos.scene_process import MJWarpVisibility, build_or_load_scene_points


@dataclass
class OpenContactSurface:
    name: str
    center_world: np.ndarray
    rotation_world: np.ndarray
    half_size: np.ndarray
    approach_world: np.ndarray
    pull_world: np.ndarray
    geom_name: str
    allowed_geom_names: tuple[str, ...]
    contact_local_y: float
    force_normal_local: np.ndarray


@dataclass(frozen=True)
class SceneSurfaceSpec:
    name: str
    kind: str
    geom_prefix: str | None = None
    allowed_geom_prefix: str | None = None
    geom_name: str | None = None
    contact_local_axis: int = 1
    contact_local_sign: float = 1.0
    force_normal_local: tuple[float, float, float] = (0.0, -1.0, 0.0)


DEFAULT_DRAWER_SURFACE_SPECS = {
    "handle": SceneSurfaceSpec(
        name="handle",
        kind="geom_bounds_in_panel_frame",
        geom_prefix="{drawer_name}_door_handle_g",
        allowed_geom_prefix="{drawer_name}_door_handle",
    ),
    "panel_inner": SceneSurfaceSpec(
        name="panel_inner",
        kind="panel",
        geom_prefix="{drawer_name}_door_g",
    ),
}


def load_scene_yaml(path):
    import yaml

    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Scene YAML must contain a mapping: {path}")
    return data


def surface_specs_from_config(config):
    surfaces = dict(config.get("surfaces", config))
    specs = {}
    for name, raw_spec in surfaces.items():
        if isinstance(raw_spec, SceneSurfaceSpec):
            specs[str(name)] = raw_spec
            continue
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Surface spec for {name!r} must be a mapping")
        values = dict(raw_spec)
        values.setdefault("name", str(name))
        specs[str(name)] = SceneSurfaceSpec(**values)
    return specs


def _format_scene_name(pattern, **values):
    if pattern is None:
        return None
    return str(pattern).format(**values)


def _matching_geom_names(model, *, prefix=None, exact=None):
    if exact:
        names = [str(exact)] if str(exact) in model._geom_name2id else []
    else:
        names = [
            name
            for name in model._geom_name2id
            if prefix is not None and name.startswith(str(prefix))
        ]
    return tuple(sorted(names))


def _drawer_name(env):
    return env.drawer.name


def _handle_geom_names(env):
    drawer_name = _drawer_name(env)
    names = _matching_geom_names(
        env.sim.model,
        prefix=f"{drawer_name}_door_handle_g",
    )
    if not names:
        raise RuntimeError(f"Cannot find handle geoms for drawer '{drawer_name}'.")
    return names


def _handle_allowed_geom_names(env):
    drawer_name = _drawer_name(env)
    names = _matching_geom_names(
        env.sim.model,
        prefix=f"{drawer_name}_door_handle",
    )
    if not names:
        raise RuntimeError(f"Cannot find handle geoms for drawer '{drawer_name}'.")
    return names


def _surface_contact_local_y(half_size, spec):
    axis = int(spec.contact_local_axis)
    if axis != 1:
        raise ValueError("Open drawer surfaces currently expect contact_local_axis=1")
    return float(
        np.asarray(half_size, dtype=np.float64)[axis] * float(spec.contact_local_sign)
    )


def _make_geom_bounds_surface(env, panel, spec):
    model = env.sim.model
    data = env.sim.data
    drawer_name = _drawer_name(env)
    geom_prefix = _format_scene_name(spec.geom_prefix, drawer_name=drawer_name)
    geom_names = _matching_geom_names(
        model,
        prefix=geom_prefix,
        exact=_format_scene_name(spec.geom_name, drawer_name=drawer_name),
    )
    if not geom_names:
        raise RuntimeError(f"Cannot find geoms for scene surface '{spec.name}'.")

    corners_panel = []
    for name in geom_names:
        geom_id = model.geom_name2id(name)
        geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        geom_rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        geom_size = np.asarray(model.geom_size[geom_id], dtype=np.float64).copy()
        geom_size = np.maximum(
            geom_size, np.array([0.002, 0.002, 0.002], dtype=np.float64)
        )
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corner_world = geom_pos + geom_rot @ (
                        geom_size * np.array([sx, sy, sz], dtype=np.float64)
                    )
                    corners_panel.append(
                        panel.rotation_world.T @ (corner_world - panel.center_world)
                    )
    corners_panel = np.asarray(corners_panel, dtype=np.float64)
    min_corner = np.min(corners_panel, axis=0)
    max_corner = np.max(corners_panel, axis=0)
    center_local = 0.5 * (min_corner + max_corner)
    half_size = np.maximum(
        0.5 * (max_corner - min_corner),
        np.array([0.006, 0.006, 0.006], dtype=np.float64),
    )
    center_world = panel.center_world + panel.rotation_world @ center_local

    allowed_prefix = _format_scene_name(
        spec.allowed_geom_prefix, drawer_name=drawer_name
    )
    allowed_geom_names = (
        _matching_geom_names(model, prefix=allowed_prefix)
        if allowed_prefix
        else geom_names
    )
    return OpenContactSurface(
        name=spec.name,
        center_world=center_world,
        rotation_world=np.asarray(panel.rotation_world, dtype=np.float64).copy(),
        half_size=half_size,
        approach_world=np.asarray(panel.push_world, dtype=np.float64).copy(),
        pull_world=np.asarray(panel.outward_world, dtype=np.float64).copy(),
        geom_name=geom_names[0],
        allowed_geom_names=allowed_geom_names,
        contact_local_y=_surface_contact_local_y(half_size, spec),
        force_normal_local=np.asarray(spec.force_normal_local, dtype=np.float64),
    )


def _make_panel_surface(env, panel, spec):
    drawer_name = _drawer_name(env)
    prefix = _format_scene_name(spec.geom_prefix, drawer_name=drawer_name)
    panel_geom_names = tuple(
        sorted(
            name
            for name in env.sim.model._geom_name2id
            if name == panel.geom_name
            or (prefix is not None and name.startswith(prefix))
        )
    )
    half_size = np.asarray(panel.half_size, dtype=np.float64).copy()
    return OpenContactSurface(
        name=spec.name,
        center_world=np.asarray(panel.center_world, dtype=np.float64).copy(),
        rotation_world=np.asarray(panel.rotation_world, dtype=np.float64).copy(),
        half_size=half_size,
        approach_world=np.asarray(panel.push_world, dtype=np.float64).copy(),
        pull_world=np.asarray(panel.outward_world, dtype=np.float64).copy(),
        geom_name=panel.geom_name,
        allowed_geom_names=panel_geom_names,
        contact_local_y=_surface_contact_local_y(half_size, spec),
        force_normal_local=np.asarray(spec.force_normal_local, dtype=np.float64),
    )


def make_contact_surface(env, panel, surface="handle", spec=None):
    if spec is None:
        spec = DEFAULT_DRAWER_SURFACE_SPECS[str(surface)]
    elif isinstance(spec, dict):
        spec = SceneSurfaceSpec(**spec)
    if spec.kind == "geom_bounds_in_panel_frame":
        return _make_geom_bounds_surface(env, panel, spec)
    if spec.kind == "panel":
        return _make_panel_surface(env, panel, spec)
    raise ValueError(f"Unsupported scene surface kind: {spec.kind!r}")


def make_handle_inner_surface(env, panel):
    return make_contact_surface(env, panel, "handle")


def make_panel_inner_surface(env, panel):
    return make_contact_surface(env, panel, "panel_inner")


def current_surface_for_stage(env, stage, panel_getter):
    panel = panel_getter(env)
    if stage.surface_name == "handle":
        return make_handle_inner_surface(env, panel)
    return make_panel_inner_surface(env, panel)


def _scene_xml_root(env, args):
    attr_name = str(getattr(args, "scene_xml_root_attr", "drawer") or "drawer")
    root = getattr(env, attr_name, None)
    if root is None:
        raise RuntimeError(f"Scene XML root attribute not found on env: {attr_name!r}")
    if not hasattr(root, "get_xml"):
        raise RuntimeError(f"Scene XML root {attr_name!r} does not provide get_xml().")
    return root.get_xml()


def initialize_scene_processing(env, args):
    args._scene_point_cloud = None
    args._scene_visibility = None
    args._scene_runtime_data = env.sim.data
    if not args.scene_process:
        return
    point_cloud = build_or_load_scene_points(
        _scene_xml_root(env, args),
        env.sim.model,
        args.scene_cache_dir,
        points_per_link=args.scene_points_per_link,
        seed=args.seed,
        force=args.scene_force_rebuild,
    )
    visibility = MJWarpVisibility(
        env.sim.model,
        env.sim.data,
        point_cloud,
        device=args.scene_visibility_device,
        hit_tolerance=args.scene_visibility_hit_tolerance,
        use_bvh=not args.scene_disable_bvh,
        allow_cpu_fallback=not args.scene_require_mjwarp,
    )
    args._scene_point_cloud = point_cloud
    args._scene_visibility = visibility
    env._scene_point_cloud = point_cloud
    env._scene_visibility = visibility
    refresh_scene_visibility(env, args)


def refresh_scene_visibility(env, args):
    visibility = getattr(args, "_scene_visibility", None)
    if visibility is None:
        return None
    env.sim.forward()
    return visibility.update_from_camera(
        args.scene_visibility_camera,
        width=args.scene_visibility_width,
        height=args.scene_visibility_height,
    )


__all__ = [
    "DEFAULT_DRAWER_SURFACE_SPECS",
    "OpenContactSurface",
    "SceneSurfaceSpec",
    "current_surface_for_stage",
    "initialize_scene_processing",
    "load_scene_yaml",
    "make_contact_surface",
    "make_handle_inner_surface",
    "make_panel_inner_surface",
    "refresh_scene_visibility",
    "surface_specs_from_config",
]
