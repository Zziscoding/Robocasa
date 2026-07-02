"""Offline link point-cloud preprocessing and runtime visibility masks.

The offline path compiles an MJCF object, treats each MuJoCo body with geometry
as one link, merges that link's render geometry in the body-local frame,
decomposes it with CoACD, uniformly samples a fixed number of surface points,
and stores both the arrays and a SciPy KDTree.

At runtime :class:`MJWarpVisibility` casts all camera-to-point rays in one
MJWarp call.  Every link owns a uint8 mask with exactly the same length as its
point set: invisible is 0, visible is 1.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import pickle
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree


CACHE_VERSION = 2


def _raw_model(model: Any) -> mujoco.MjModel:
    return getattr(model, "_model", model)


def _raw_data(data: Any) -> mujoco.MjData:
    return getattr(data, "_data", data)


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, obj_type, int(index)) or ""


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "unnamed"


def _quat_matrix(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = max(float(np.linalg.norm([w, x, y, z])), 1e-12)
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_mesh(
    mesh: trimesh.Trimesh, position: np.ndarray, rotation: np.ndarray
) -> trimesh.Trimesh:
    result = mesh.copy()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    result.apply_transform(transform)
    return result


def _mesh_geom(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
    mesh_id = int(model.geom_dataid[geom_id])
    vertex_address = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_address = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vertex_address : vertex_address + vertex_count],
        dtype=np.float64,
    )
    faces = np.asarray(
        model.mesh_face[face_address : face_address + face_count],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _primitive_geom(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * size)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.apply_scale(size)
        return mesh
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(
            radius=float(size[0]), height=2.0 * float(size[1]), sections=32
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(
            radius=float(size[0]), height=2.0 * float(size[1]), count=[16, 16]
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        return _mesh_geom(model, geom_id)
    return None


def _geom_mesh_in_body(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    mesh = _primitive_geom(model, geom_id)
    if mesh is None or mesh.faces.size == 0:
        return None
    return _transform_mesh(
        mesh,
        np.asarray(model.geom_pos[geom_id], dtype=np.float64),
        _quat_matrix(model.geom_quat[geom_id]),
    )


def _select_body_geoms(model: mujoco.MjModel, body_id: int) -> list[int]:
    geom_ids = np.flatnonzero(np.asarray(model.geom_bodyid) == int(body_id)).tolist()
    geom_ids = [
        geom_id
        for geom_id in geom_ids
        if float(model.geom_rgba[geom_id, 3]) > 1e-5
        and int(model.geom_type[geom_id])
        not in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)
    ]
    visual = [geom_id for geom_id in geom_ids if int(model.geom_group[geom_id]) == 1]
    return visual or geom_ids


def _uniform_surface_samples(
    mesh: trimesh.Trimesh, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.shape[0] == 0:
        raise ValueError("Cannot sample a mesh without triangles")
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    doubled_area = np.linalg.norm(cross, axis=1)
    valid = doubled_area > 1e-12
    triangles = triangles[valid]
    cross = cross[valid]
    doubled_area = doubled_area[valid]
    if triangles.shape[0] == 0:
        raise ValueError("Cannot sample a zero-area mesh")
    face_ids = rng.choice(
        triangles.shape[0],
        size=int(count),
        replace=True,
        p=doubled_area / doubled_area.sum(),
    )
    chosen = triangles[face_ids]
    u = rng.random((count, 1))
    v = rng.random((count, 1))
    sqrt_u = np.sqrt(u)
    points = (
        (1.0 - sqrt_u) * chosen[:, 0]
        + sqrt_u * (1.0 - v) * chosen[:, 1]
        + sqrt_u * v * chosen[:, 2]
    )
    normals = cross[face_ids]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    return points.astype(np.float32), normals.astype(np.float32)


def _normalize3(
    vec: Sequence[float], fallback: Sequence[float] | None = None
) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        if fallback is None:
            raise ValueError("Cannot normalize near-zero vector.")
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return vec / norm


def _grid_shape_for_count(sample_count: int) -> tuple[int, int]:
    sample_count = max(int(sample_count), 1)
    if sample_count == 15:
        return 5, 3
    best = (sample_count, 1)
    best_score = sample_count
    for rows in range(1, int(np.sqrt(sample_count)) + 2):
        cols = int(np.ceil(sample_count / rows))
        score = abs(cols - rows) + (cols * rows - sample_count)
        if score < best_score:
            best = (cols, rows)
            best_score = score
    return best


def _face_grid_points(count: int) -> np.ndarray:
    base = np.array(
        [
            [0.0, 0.0],
            [-0.75, -0.75],
            [0.75, -0.35],
            [-0.35, 0.75],
            [0.75, 0.75],
        ],
        dtype=np.float64,
    )
    if count <= len(base):
        return base[:count]
    n = max(int(np.ceil(np.sqrt(count))), 2)
    values = np.linspace(-0.85, 0.85, n)
    grid = np.asarray([(u, v) for v in values for u in values], dtype=np.float64)
    return grid[:count]


def sample_handle_surface_candidates(surface, sample_count: int, margin: float):
    per_face = max(int(sample_count) // 4, 1)
    remainder = max(int(sample_count) - 4 * per_face, 0)
    face_counts = [
        per_face + (1 if face_idx < remainder else 0) for face_idx in range(4)
    ]
    hx = max(float(surface.half_size[0]) - float(margin), 1e-4)
    hy = max(float(surface.half_size[1]) - float(margin), 1e-4)
    hz = max(float(surface.half_size[2]) - float(margin), 1e-4)
    face_specs = (
        (
            "inner",
            1,
            hy,
            (0, hx),
            (2, hz),
            np.array([0.0, -1.0, 0.0]),
            surface.rotation_world[:, 1],
        ),
        (
            "outer",
            1,
            -hy,
            (0, hx),
            (2, hz),
            np.array([0.0, 1.0, 0.0]),
            -surface.rotation_world[:, 1],
        ),
        (
            "top",
            2,
            hz,
            (0, hx),
            (1, hy),
            np.array([0.0, 0.0, -1.0]),
            surface.rotation_world[:, 2],
        ),
        (
            "bottom",
            2,
            -hz,
            (0, hx),
            (1, hy),
            np.array([0.0, 0.0, 1.0]),
            -surface.rotation_world[:, 2],
        ),
    )
    local_points = []
    world_points = []
    force_normals = []
    approach_dirs = []
    face_names = []
    for face_count, (
        face_name,
        fixed_axis,
        fixed_value,
        axis_a,
        axis_b,
        normal,
        approach,
    ) in zip(face_counts, face_specs):
        for u, v in _face_grid_points(face_count):
            p_local = np.zeros(3, dtype=np.float64)
            p_local[fixed_axis] = fixed_value
            p_local[axis_a[0]] = float(u) * axis_a[1]
            p_local[axis_b[0]] = float(v) * axis_b[1]
            local_points.append(p_local)
            world_points.append(surface.center_world + surface.rotation_world @ p_local)
            force_normals.append(np.asarray(normal, dtype=np.float64))
            approach_dirs.append(_normalize3(approach))
            face_names.append(face_name)
    return (
        np.asarray(local_points, dtype=np.float64),
        np.asarray(world_points, dtype=np.float64),
        np.asarray(force_normals, dtype=np.float64),
        np.asarray(approach_dirs, dtype=np.float64),
        face_names,
    )


def sample_inner_surface_candidates(surface, sample_count: int, margin: float):
    if surface.name == "handle":
        return sample_handle_surface_candidates(surface, sample_count, margin)
    grid_x, grid_z = _grid_shape_for_count(sample_count)
    x_lim = max(float(surface.half_size[0]) - float(margin), 1e-4)
    z_lim = max(float(surface.half_size[2]) - float(margin), 1e-4)
    xs = np.linspace(-x_lim, x_lim, grid_x)
    zs = np.linspace(-z_lim, z_lim, grid_z)
    local = []
    world = []
    for z in zs:
        for x in xs:
            if len(local) >= sample_count:
                break
            p_local = np.array([x, float(surface.contact_local_y), z], dtype=np.float64)
            p_world = surface.center_world + surface.rotation_world @ p_local
            local.append(p_local)
            world.append(p_world)
    count = len(local)
    force_normals = np.repeat(surface.force_normal_local.reshape(1, 3), count, axis=0)
    approach_dirs = np.repeat(surface.approach_world.reshape(1, 3), count, axis=0)
    face_names = ["inner"] * count
    return (
        np.asarray(local, dtype=np.float64),
        np.asarray(world, dtype=np.float64),
        force_normals,
        approach_dirs,
        face_names,
    )


@dataclass(frozen=True)
class CoACDConfig:
    threshold: float = 0.05
    max_convex_hull: int = 32
    preprocess_mode: str = "auto"
    preprocess_resolution: int = 30
    resolution: int = 2000
    mcts_nodes: int = 20
    mcts_iterations: int = 100
    mcts_max_depth: int = 3
    max_ch_vertex: int = 256


@dataclass
class LinkPointCloud:
    name: str
    source_body_id: int
    geom_names: tuple[str, ...]
    points_local: np.ndarray
    normals_local: np.ndarray
    mask: np.ndarray
    tree: cKDTree
    runtime_body_id: int = -1
    runtime_geom_ids: tuple[int, ...] = ()

    def bind(self, model: Any) -> None:
        raw_model = _raw_model(model)
        self.runtime_body_id = _resolve_named_id(
            raw_model, mujoco.mjtObj.mjOBJ_BODY, self.name
        )
        geom_ids = []
        for geom_name in self.geom_names:
            geom_id = _resolve_named_id(
                raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name, required=False
            )
            if geom_id >= 0:
                geom_ids.append(geom_id)
        if self.runtime_body_id >= 0:
            # Rays may hit an overlapping collision geom before its visual
            # duplicate. Both belong to the same physical link and therefore
            # count as a visible hit.
            geom_ids.extend(
                np.flatnonzero(
                    np.asarray(raw_model.geom_bodyid) == self.runtime_body_id
                )
                .astype(int)
                .tolist()
            )
        self.runtime_geom_ids = tuple(sorted(set(geom_ids)))

    def world_points(self, data: Any) -> np.ndarray:
        if self.runtime_body_id < 0:
            raise RuntimeError(f"Link {self.name!r} is not bound to a runtime model")
        raw_data = _raw_data(data)
        rotation = np.asarray(
            raw_data.xmat[self.runtime_body_id], dtype=np.float64
        ).reshape(3, 3)
        position = np.asarray(raw_data.xpos[self.runtime_body_id], dtype=np.float64)
        return self.points_local @ rotation.T + position

    def world_normals(self, data: Any) -> np.ndarray:
        raw_data = _raw_data(data)
        rotation = np.asarray(
            raw_data.xmat[self.runtime_body_id], dtype=np.float64
        ).reshape(3, 3)
        return self.normals_local @ rotation.T

    def world_to_local(self, point: Sequence[float], data: Any) -> np.ndarray:
        raw_data = _raw_data(data)
        rotation = np.asarray(
            raw_data.xmat[self.runtime_body_id], dtype=np.float64
        ).reshape(3, 3)
        position = np.asarray(raw_data.xpos[self.runtime_body_id], dtype=np.float64)
        return rotation.T @ (np.asarray(point, dtype=np.float64) - position)


def _resolve_named_id(
    model: mujoco.MjModel,
    obj_type: mujoco.mjtObj,
    requested: str,
    *,
    required: bool = True,
) -> int:
    exact = mujoco.mj_name2id(model, obj_type, requested)
    if exact >= 0:
        return int(exact)
    count_by_type = {
        mujoco.mjtObj.mjOBJ_BODY: model.nbody,
        mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
        mujoco.mjtObj.mjOBJ_CAMERA: model.ncam,
    }
    candidates = []
    for index in range(count_by_type[obj_type]):
        candidate_name = _name(model, obj_type, index)
        if candidate_name and (
            candidate_name.endswith(requested) or requested.endswith(candidate_name)
        ):
            candidates.append(index)
    if len(candidates) == 1:
        return int(candidates[0])
    if required:
        raise ValueError(f"Cannot uniquely resolve {requested!r} in runtime model")
    return -1


class ScenePointCloud:
    """Cached point sets, one KDTree and one visibility mask per link."""

    def __init__(self, links: Sequence[LinkPointCloud], cache_dir: Path) -> None:
        self.links = list(links)
        self.cache_dir = Path(cache_dir)

    @classmethod
    def load(cls, cache_dir: str | Path) -> "ScenePointCloud":
        cache_dir = Path(cache_dir)
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        if int(manifest["cache_version"]) != CACHE_VERSION:
            raise ValueError("Scene point cache version mismatch")
        links = []
        for item in manifest["links"]:
            arrays = np.load(cache_dir / item["arrays"])
            points = np.asarray(arrays["points_local"], dtype=np.float32)
            normals = np.asarray(arrays["normals_local"], dtype=np.float32)
            try:
                with (cache_dir / item["tree"]).open("rb") as stream:
                    tree = pickle.load(stream)
                if int(tree.n) != points.shape[0]:
                    raise ValueError("KDTree point count mismatch")
            except Exception:
                # SciPy KDTree pickles are not guaranteed to be portable across
                # versions. The points remain the source of truth.
                tree = cKDTree(points)
            links.append(
                LinkPointCloud(
                    name=item["name"],
                    source_body_id=int(item["source_body_id"]),
                    geom_names=tuple(item["geom_names"]),
                    points_local=points,
                    normals_local=normals,
                    mask=np.zeros(points.shape[0], dtype=np.uint8),
                    tree=tree,
                )
            )
        return cls(links, cache_dir)

    def bind(self, model: Any) -> "ScenePointCloud":
        for link in self.links:
            link.bind(model)
        return self

    @property
    def size(self) -> int:
        return sum(link.points_local.shape[0] for link in self.links)

    @property
    def mask(self) -> np.ndarray:
        if not self.links:
            return np.zeros(0, dtype=np.uint8)
        return np.concatenate([link.mask for link in self.links])

    @property
    def masks(self) -> dict[str, np.ndarray]:
        """Live per-link masks; each array has the same length as its point set."""

        return {link.name: link.mask for link in self.links}

    def world_points(self, data: Any) -> np.ndarray:
        if not self.links:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate([link.world_points(data) for link in self.links], axis=0)

    def world_normals(self, data: Any) -> np.ndarray:
        if not self.links:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate([link.world_normals(data) for link in self.links], axis=0)

    def point_target_geom_ids(self) -> list[tuple[int, ...]]:
        result: list[tuple[int, ...]] = []
        for link in self.links:
            result.extend([link.runtime_geom_ids] * link.points_local.shape[0])
        return result

    def update_masks(self, flat_mask: np.ndarray) -> None:
        flat_mask = np.asarray(flat_mask, dtype=np.uint8).reshape(-1)
        if flat_mask.size != self.size:
            raise ValueError(f"Expected {self.size} mask values, got {flat_mask.size}")
        offset = 0
        for link in self.links:
            count = link.points_local.shape[0]
            link.mask[:] = flat_mask[offset : offset + count]
            offset += count

    def nearest(
        self,
        point_world: Sequence[float],
        data: Any,
        *,
        allowed_geom_names: Iterable[str] | None = None,
        visible_only: bool = False,
    ) -> tuple[LinkPointCloud, int, float] | None:
        allowed = set(allowed_geom_names or ())
        best = None
        for link in self.links:
            if allowed and not allowed.intersection(link.geom_names):
                continue
            local = link.world_to_local(point_world, data)
            if visible_only:
                indices = np.flatnonzero(link.mask)
                if not indices.size:
                    continue
                delta = link.points_local[indices] - local
                nearest_visible = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
                local_index = int(indices[nearest_visible])
                distance = float(np.linalg.norm(delta[nearest_visible]))
            else:
                distance, local_index = link.tree.query(local)
                local_index = int(local_index)
                distance = float(distance)
            if best is None or distance < best[2]:
                best = (link, local_index, distance)
        return best

    def summary(self) -> dict[str, Any]:
        return {
            "cache_dir": str(self.cache_dir),
            "point_count": self.size,
            "visible_count": int(self.mask.sum()),
            "links": [
                {
                    "name": link.name,
                    "point_count": int(link.points_local.shape[0]),
                    "visible_count": int(link.mask.sum()),
                    "geom_names": list(link.geom_names),
                }
                for link in self.links
            ],
        }


def _compile_source(xml_source: str | Path) -> tuple[mujoco.MjModel, str]:
    text_or_path = str(xml_source)
    if "<mujoco" in text_or_path:
        return mujoco.MjModel.from_xml_string(text_or_path), text_or_path
    source = Path(xml_source)
    if source.exists():
        text = source.read_text()
        return mujoco.MjModel.from_xml_path(str(source)), text
    raise FileNotFoundError(f"MJCF source does not exist: {xml_source}")


def _decompose(
    mesh: trimesh.Trimesh, config: CoACDConfig, seed: int
) -> list[trimesh.Trimesh]:
    try:
        coacd = importlib.import_module("coacd")
    except ImportError as exc:
        raise RuntimeError(
            "CoACD is required for offline scene preprocessing. Install the `coacd` package."
        ) from exc
    source = coacd.Mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    parts = coacd.run_coacd(source, seed=int(seed), **asdict(config))
    return [
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        for vertices, faces in parts
    ]


def build_link_point_cache(
    xml_source: str | Path,
    cache_root: str | Path,
    *,
    points_per_link: int = 2048,
    body_prefix: str | None = None,
    body_names: Sequence[str] | None = None,
    seed: int = 0,
    coacd_config: CoACDConfig | None = None,
    force: bool = False,
) -> Path:
    """Build or reuse a content-addressed link point/KDTree cache."""

    model, xml_text = _compile_source(xml_source)
    config = coacd_config or CoACDConfig()
    settings = {
        "cache_version": CACHE_VERSION,
        "points_per_link": int(points_per_link),
        "body_prefix": body_prefix,
        "body_names": list(body_names or ()),
        "seed": int(seed),
        "coacd": asdict(config),
    }
    digest = hashlib.sha256(
        xml_text.encode("utf-8") + json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    cache_dir = Path(cache_root) / digest
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return cache_dir

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hull_dir = cache_dir / "hulls"
    hull_dir.mkdir(exist_ok=True)
    requested_names = set(body_names or ())
    rng = np.random.default_rng(seed)
    manifest_links = []

    for body_id in range(1, model.nbody):
        body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not body_name:
            continue
        if body_prefix is not None and not body_name.startswith(body_prefix):
            continue
        if requested_names and body_name not in requested_names:
            continue
        geom_ids = _select_body_geoms(model, body_id)
        meshes = [
            mesh
            for geom_id in geom_ids
            if (mesh := _geom_mesh_in_body(model, geom_id)) is not None
        ]
        if not meshes:
            continue
        merged = trimesh.util.concatenate(meshes)
        merged.remove_unreferenced_vertices()
        if merged.faces.shape[0] < 4:
            continue
        parts = _decompose(merged, config, seed + body_id)
        # Keep samples on the original link surface. CoACD hulls are cached as
        # the link decomposition, but sampling their approximate outer surfaces
        # would make camera rays disagree with the MuJoCo render geometry.
        points, normals = _uniform_surface_samples(merged, int(points_per_link), rng)
        tree = cKDTree(points)
        stem = f"{body_id:04d}_{_safe_name(body_name)}"
        arrays_name = f"{stem}.npz"
        tree_name = f"{stem}.kdtree.pkl"
        np.savez_compressed(
            cache_dir / arrays_name,
            points_local=points,
            normals_local=normals,
        )
        with (cache_dir / tree_name).open("wb") as stream:
            pickle.dump(tree, stream, protocol=pickle.HIGHEST_PROTOCOL)
        for part_index, part in enumerate(parts):
            part.export(hull_dir / f"{stem}_{part_index:03d}.obj")
        manifest_links.append(
            {
                "name": body_name,
                "source_body_id": body_id,
                "geom_names": [
                    _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    for geom_id in geom_ids
                    if _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                ],
                "arrays": arrays_name,
                "tree": tree_name,
                "hull_count": len(parts),
            }
        )

    if not manifest_links:
        raise RuntimeError("No processable links were found in the supplied MJCF")
    manifest = {
        **settings,
        "source_sha256": hashlib.sha256(xml_text.encode("utf-8")).hexdigest(),
        "links": manifest_links,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (cache_dir / "source.xml").write_text(xml_text)
    return cache_dir


class MJWarpVisibility:
    """Batched camera visibility test backed by MJWarp rays."""

    def __init__(
        self,
        model: Any,
        data: Any,
        point_cloud: ScenePointCloud,
        *,
        device: str = "cuda:0",
        hit_tolerance: float = 0.01,
        front_face_epsilon: float = 0.0,
        use_bvh: bool = True,
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.model_cpu = _raw_model(model)
        self.data_cpu = _raw_data(data)
        self.point_cloud = point_cloud.bind(self.model_cpu)
        self.device = device
        self.hit_tolerance = float(hit_tolerance)
        self.front_face_epsilon = float(front_face_epsilon)
        self.backend_name = "mujoco"
        self.wp = None
        self.mjwarp = None
        self.model_wp = None
        self.data_wp = None
        self.render_context = None
        self._geomgroup_type = None
        self._ray_origin = None
        self._ray_direction = None
        self._ray_bodyexclude = None
        self._ray_distance = None
        self._ray_geom_id = None
        self._ray_normal = None
        self._allocate_backend(use_bvh, allow_cpu_fallback)

    def _allocate_backend(self, use_bvh: bool, allow_cpu_fallback: bool) -> None:
        try:
            wp = importlib.import_module("warp")
            try:
                mjwarp = importlib.import_module("mujoco_warp")
                ray_types = importlib.import_module("mujoco_warp._src.types")
            except Exception:
                repo_root = Path(__file__).resolve().parents[2]
                package_root = repo_root / "comfree_warp"
                if str(package_root) not in sys.path:
                    sys.path.insert(0, str(package_root))
                mjwarp = importlib.import_module("comfree_warp.mujoco_warp")
                ray_types = importlib.import_module(
                    "comfree_warp.mujoco_warp._src.types"
                )
            wp.init()
            wp.set_device(self.device)
            with wp.ScopedDevice(self.device):
                self.model_wp = mjwarp.put_model(self.model_cpu)
                self.data_wp = mjwarp.put_data(
                    self.model_cpu,
                    self.data_cpu,
                    nworld=1,
                    nconmax=max(int(self.data_cpu.ncon), 1),
                    njmax=max(int(self.data_cpu.nefc), 1),
                )
                if use_bvh:
                    try:
                        self.render_context = mjwarp.create_render_context(
                            self.model_cpu,
                            nworld=1,
                            enabled_geom_groups=[0, 1, 2, 3, 4, 5],
                            use_textures=False,
                            use_shadows=False,
                        )
                    except Exception:
                        self.render_context = None
                point_count = self.point_cloud.size
                self._ray_origin = wp.empty(
                    (1, point_count), dtype=wp.vec3, device=self.device
                )
                self._ray_direction = wp.empty(
                    (1, point_count), dtype=wp.vec3, device=self.device
                )
                self._ray_bodyexclude = wp.full(
                    point_count, -1, dtype=int, device=self.device
                )
                self._ray_distance = wp.empty(
                    (1, point_count), dtype=float, device=self.device
                )
                self._ray_geom_id = wp.empty(
                    (1, point_count), dtype=int, device=self.device
                )
                self._ray_normal = wp.empty(
                    (1, point_count), dtype=wp.vec3, device=self.device
                )
            self.wp = wp
            self.mjwarp = mjwarp
            self._geomgroup_type = ray_types.vec6
            self.backend_name = "mjwarp"
        except Exception as exc:
            if not allow_cpu_fallback:
                raise RuntimeError(
                    "MJWarp visibility backend initialization failed"
                ) from exc
            self.backend_error = repr(exc)

    def _frustum_mask(
        self,
        points: np.ndarray,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray | None,
        fovy_degrees: float | None,
        aspect: float,
    ) -> np.ndarray:
        if camera_rotation is None or fovy_degrees is None:
            return np.ones(points.shape[0], dtype=bool)
        camera_points = (points - camera_position) @ camera_rotation
        depth = -camera_points[:, 2]
        tan_y = math.tan(math.radians(float(fovy_degrees)) * 0.5)
        tan_x = tan_y * float(aspect)
        return (
            (depth > 1e-6)
            & (np.abs(camera_points[:, 1]) <= depth * tan_y)
            & (np.abs(camera_points[:, 0]) <= depth * tan_x)
        )

    def _update_mjwarp(
        self, camera_position: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        wp = self.wp
        assert wp is not None and self.mjwarp is not None
        count = directions.shape[0]
        origins = np.repeat(camera_position.reshape(1, 1, 3), count, axis=1)
        vectors = directions.reshape(1, count, 3)
        with wp.ScopedDevice(self.device):
            wp.copy(
                self.data_wp.geom_xpos,
                wp.array(
                    np.asarray(self.data_cpu.geom_xpos, dtype=np.float32).reshape(
                        1, self.model_cpu.ngeom, 3
                    ),
                    dtype=wp.vec3,
                    device=self.device,
                ),
            )
            wp.copy(
                self.data_wp.geom_xmat,
                wp.array(
                    np.asarray(self.data_cpu.geom_xmat, dtype=np.float32).reshape(
                        1, self.model_cpu.ngeom, 3, 3
                    ),
                    dtype=wp.mat33,
                    device=self.device,
                ),
            )
            wp.copy(
                self._ray_origin,
                wp.array(origins, dtype=wp.vec3, device=self.device),
            )
            wp.copy(
                self._ray_direction,
                wp.array(vectors, dtype=wp.vec3, device=self.device),
            )
            self.mjwarp.rays(
                self.model_wp,
                self.data_wp,
                self._ray_origin,
                self._ray_direction,
                self._geomgroup_type(-1, -1, -1, -1, -1, -1),
                True,
                self._ray_bodyexclude,
                self._ray_distance,
                self._ray_geom_id,
                self._ray_normal,
                self.render_context,
            )
            wp.synchronize()
            return self._ray_distance.numpy()[0], self._ray_geom_id.numpy()[0]

    def _update_mujoco(
        self, camera_position: np.ndarray, directions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.full(directions.shape[0], -1.0, dtype=np.float64)
        geom_ids = np.full(directions.shape[0], -1, dtype=np.int32)
        for index, direction in enumerate(directions):
            hit_geom = np.full(1, -1, dtype=np.int32)
            distances[index] = mujoco.mj_ray(
                self.model_cpu,
                self.data_cpu,
                camera_position,
                direction,
                None,
                1,
                -1,
                hit_geom,
            )
            geom_ids[index] = hit_geom[0]
        return distances, geom_ids

    def update(
        self,
        camera_position: Sequence[float],
        *,
        camera_rotation: np.ndarray | None = None,
        fovy_degrees: float | None = None,
        aspect: float = 1.0,
    ) -> np.ndarray:
        points = self.point_cloud.world_points(self.data_cpu)
        normals = self.point_cloud.world_normals(self.data_cpu)
        camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
        ray = points - camera_position
        point_distance = np.linalg.norm(ray, axis=1)
        directions = ray / point_distance[:, None].clip(min=1e-12)
        if self.backend_name == "mjwarp":
            hit_distance, hit_geom = self._update_mjwarp(camera_position, directions)
        else:
            hit_distance, hit_geom = self._update_mujoco(camera_position, directions)

        target_ids = self.point_cloud.point_target_geom_ids()
        target_hit = np.fromiter(
            (
                int(hit_geom[index]) in target_ids[index]
                for index in range(points.shape[0])
            ),
            dtype=bool,
            count=points.shape[0],
        )
        distance_match = (hit_distance >= 0.0) & (
            np.abs(hit_distance - point_distance) <= self.hit_tolerance
        )
        front_facing = (
            np.einsum("ij,ij->i", normals, camera_position.reshape(1, 3) - points)
            > self.front_face_epsilon
        )
        frustum = self._frustum_mask(
            points,
            camera_position,
            None
            if camera_rotation is None
            else np.asarray(camera_rotation).reshape(3, 3),
            fovy_degrees,
            aspect,
        )
        mask = (target_hit & distance_match & front_facing & frustum).astype(np.uint8)
        self.point_cloud.update_masks(mask)
        return mask

    def update_from_camera(
        self,
        camera_name: str,
        *,
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        camera_id = _resolve_named_id(
            self.model_cpu, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        camera_position = np.asarray(
            self.data_cpu.cam_xpos[camera_id], dtype=np.float64
        )
        camera_rotation = np.asarray(
            self.data_cpu.cam_xmat[camera_id], dtype=np.float64
        ).reshape(3, 3)
        return self.update(
            camera_position,
            camera_rotation=camera_rotation,
            fovy_degrees=float(self.model_cpu.cam_fovy[camera_id]),
            aspect=float(width) / max(float(height), 1.0),
        )


def build_or_load_scene_points(
    xml_source: str | Path,
    runtime_model: Any,
    cache_root: str | Path,
    *,
    points_per_link: int = 2048,
    seed: int = 0,
    force: bool = False,
    coacd_config: CoACDConfig | None = None,
) -> ScenePointCloud:
    cache_dir = build_link_point_cache(
        xml_source,
        cache_root,
        points_per_link=points_per_link,
        seed=seed,
        force=force,
        coacd_config=coacd_config,
    )
    return ScenePointCloud.load(cache_dir).bind(runtime_model)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--points-per-link", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cache_dir = build_link_point_cache(
        args.xml,
        args.cache_root,
        points_per_link=args.points_per_link,
        seed=args.seed,
        force=args.force,
    )
    cloud = ScenePointCloud.load(cache_dir)
    print(json.dumps(cloud.summary(), indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    "CACHE_VERSION",
    "CoACDConfig",
    "LinkPointCloud",
    "MJWarpVisibility",
    "ScenePointCloud",
    "build_link_point_cache",
    "build_or_load_scene_points",
]
