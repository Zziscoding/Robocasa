"""Isolated Open3D process used to avoid MuJoCo/Open3D GLX conflicts."""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import open3d as o3d


def _geometry(spec):
    kind = spec["kind"]
    if kind == "point_cloud":
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(spec["points"])
        result.colors = o3d.utility.Vector3dVector(spec["colors"])
        return result
    if kind == "line_set":
        result = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(spec["points"]),
            lines=o3d.utility.Vector2iVector(spec["lines"]),
        )
        result.colors = o3d.utility.Vector3dVector(spec["colors"])
        return result
    if kind == "triangle_mesh":
        result = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(spec["vertices"]),
            triangles=o3d.utility.Vector3iVector(spec["triangles"]),
        )
        result.compute_vertex_normals()
        return result
    raise ValueError(f"Unsupported serialized geometry kind: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    args = parser.parse_args()
    with open(args.bundle, "rb") as stream:
        bundle = pickle.load(stream)

    draw_items = []
    for spec in bundle["geometries"]:
        item = {"name": spec["name"], "geometry": _geometry(spec)}
        material_spec = spec.get("material")
        if material_spec is not None:
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = material_spec["shader"]
            material.base_color = np.asarray(
                material_spec["base_color"], dtype=np.float32
            )
            item["material"] = material
        draw_items.append(item)

    o3d.visualization.draw(
        draw_items,
        title=bundle["title"],
        width=bundle["width"],
        height=bundle["height"],
        show_skybox=False,
        bg_color=(0.03, 0.03, 0.03, 1.0),
        point_size=int(round(bundle["point_size"])),
        line_width=int(round(bundle["line_width"])),
    )


if __name__ == "__main__":
    main()
