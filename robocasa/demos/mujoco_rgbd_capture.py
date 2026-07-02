"""Render RGB and metric depth in an isolated process."""

from __future__ import annotations

import argparse

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("state")
    parser.add_argument("output")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_binary_path(args.model)
    data = mujoco.MjData(model)
    state = np.load(args.state)
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    if model.nmocap and "mocap_pos" in state:
        data.mocap_pos[:] = state["mocap_pos"]
        data.mocap_quat[:] = state["mocap_quat"]
    if model.neq and "eq_active" in state:
        data.eq_active[:] = state["eq_active"]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=int(args.height), width=int(args.width))
    renderer.update_scene(data, camera=args.camera)
    rgb = renderer.render().copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=args.camera)
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    renderer.close()

    np.savez_compressed(
        args.output,
        rgb=rgb,
        depth=depth,
    )


if __name__ == "__main__":
    main()
