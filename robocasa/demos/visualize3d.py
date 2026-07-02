"""Reusable Viser interfaces for RoboCasa point-cloud diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _points(value) -> np.ndarray:
    if value is None:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1, 3)


def _colors(value, count: int, default) -> np.ndarray:
    if value is None:
        return np.repeat(
            np.asarray(default, dtype=np.uint8).reshape(1, 3),
            count,
            axis=0,
        )
    colors = np.asarray(value)
    if colors.ndim == 1:
        colors = np.repeat(colors.reshape(1, 3), count, axis=0)
    if colors.shape != (count, 3):
        raise ValueError(
            f"point-cloud colors must have shape {(count, 3)}, got {colors.shape}"
        )
    if np.issubdtype(colors.dtype, np.floating) and colors.size:
        if float(np.nanmax(colors)) <= 1.0:
            colors = colors * 255.0
    return np.nan_to_num(colors).clip(0, 255).astype(np.uint8)


def show_pointcloud_scene(
    *,
    scene_points=None,
    scene_colors=None,
    ee_points=None,
    ee_colors=None,
    object_candidate_points=None,
    object_candidate_colors=None,
    line_segments: Sequence[dict] = (),
    diagnostic_points: Sequence[dict] = (),
    host: str = "0.0.0.0",
    port: int = 8080,
    scene_point_size: float = 0.002,
    ee_point_size: float = 0.003,
    candidate_point_size: float = 0.008,
    block: bool = True,
):
    """Show scene, end-effector, and object-candidate clouds in Viser.

    Arrays use world coordinates and RGB colors. ``line_segments`` entries
    accept ``name``, ``points`` with shape ``(N, 2, 3)``, optional ``colors``,
    and optional ``line_width``. ``diagnostic_points`` entries accept ``name``,
    ``points``, optional ``colors``, and optional ``point_size``.

    The returned server remains owned by the caller when ``block=False``.
    """

    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "Viser is required for 3D point-cloud visualization"
        ) from exc

    server = viser.ViserServer(host=host, port=int(port), verbose=False)
    try:
        clouds = (
            (
                "scene/pointcloud",
                _points(scene_points),
                scene_colors,
                (180, 180, 180),
                float(scene_point_size),
            ),
            (
                "ee/pointcloud",
                _points(ee_points),
                ee_colors,
                (255, 245, 0),
                float(ee_point_size),
            ),
            (
                "object/candidates",
                _points(object_candidate_points),
                object_candidate_colors,
                (255, 220, 0),
                float(candidate_point_size),
            ),
        )
        for name, points, colors, default_color, point_size in clouds:
            if points.shape[0] == 0:
                continue
            server.scene.add_point_cloud(
                name,
                points=points,
                colors=_colors(colors, points.shape[0], default_color),
                point_size=point_size,
                point_shape="rounded",
            )

        for index, spec in enumerate(diagnostic_points):
            points = _points(spec.get("points"))
            if points.shape[0] == 0:
                continue
            server.scene.add_point_cloud(
                str(spec.get("name", f"diagnostic/points_{index}")),
                points=points,
                colors=_colors(
                    spec.get("colors"),
                    points.shape[0],
                    spec.get("default_color", (40, 220, 255)),
                ),
                point_size=float(spec.get("point_size", candidate_point_size)),
                point_shape="rounded",
            )

        for index, spec in enumerate(line_segments):
            segments = np.asarray(spec.get("points", ()), dtype=np.float32).reshape(
                -1, 2, 3
            )
            if segments.shape[0] == 0:
                continue
            colors = spec.get("colors")
            if colors is None:
                color = np.asarray(
                    spec.get("default_color", (255, 245, 0)),
                    dtype=np.uint8,
                )
                colors = np.repeat(color.reshape(1, 1, 3), segments.shape[0], axis=0)
                colors = np.repeat(colors, 2, axis=1)
            else:
                colors = np.asarray(colors)
                if np.issubdtype(colors.dtype, np.floating) and colors.size:
                    if float(np.nanmax(colors)) <= 1.0:
                        colors = colors * 255.0
                colors = np.nan_to_num(colors).clip(0, 255).astype(np.uint8)
            server.scene.add_line_segments(
                str(spec.get("name", f"diagnostic/lines_{index}")),
                points=segments,
                colors=colors,
                line_width=float(spec.get("line_width", 2.5)),
            )

        if block:
            input("Viser is running; press Enter to close.\n")
    except BaseException:
        server.stop()
        raise
    if block:
        server.stop()
        return None
    return server


__all__ = ["show_pointcloud_scene"]
