"""Render the README example from archived learned position-IK configurations."""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from metaarm.scene import build_scene


def main():
    """Check the archived affine reconstruction and render its achieved tool path."""
    root = Path(__file__).resolve().parents[2]
    record = np.load(
        root / "results/expansion_20261031T123139402928Z/side40_position/test/records.npz",
        allow_pickle=False,
    )
    model = build_scene()
    data = mujoco.MjData(model)
    joints = model.jnt_qposadr[[model.joint(f"fr3_joint{index}").id for index in range(1, 8)]]

    # Verify that every displayed waypoint belongs to the saved three-coordinate chart.
    assert record["trajectory_learned_success"][0].all()
    np.testing.assert_allclose(
        record["trajectory_learned_q"][0],
        record["trajectory_reference"][0]
        + record["trajectory_learned_alpha"][0] @ record["trajectory_learned_basis"][0].T,
        atol=1e-6,
        rtol=0,
    )
    points = np.empty((33, 3))
    for index, q in enumerate(record["trajectory_learned_q"][0]):
        data.qpos[joints] = q
        mujoco.mj_forward(model, data)
        points[index] = data.site_xpos[model.site("attachment_site").id]
    np.testing.assert_array_less(
        np.linalg.norm(points - record["trajectory_target"][0], axis=1), 1e-3
    )

    # Use a fixed camera and the middle waypoint for a reproducible kinematic view.
    data.qpos[joints] = record["trajectory_learned_q"][0, 16]
    mujoco.mj_forward(model, data)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1600, 900
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.65
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.30, 0, 0.38]
    camera.distance, camera.azimuth, camera.elevation = 1.60, 40, -22
    renderer = mujoco.Renderer(model, height=900, width=1600)
    renderer.update_scene(data, camera=camera)

    # Overlay only achieved tool positions, retaining the original robot and scene geometry.
    for index in range(len(points) - 1):
        mujoco.mjv_initGeom(
            renderer.scene.geoms[renderer.scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array([0.03, 0.70, 0.85, 1.0]),
        )
        mujoco.mjv_connector(
            renderer.scene.geoms[renderer.scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.003,
            points[index],
            points[index + 1],
        )
        renderer.scene.ngeom += 1
    mujoco.mjv_initGeom(
        renderer.scene.geoms[renderer.scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.012, 0.012, 0.012]),
        points[16],
        np.eye(3).ravel(),
        np.array([0.95, 0.65, 0.12, 1.0]),
    )
    renderer.scene.ngeom += 1
    Image.fromarray(renderer.render()).save(root / "docs/figures/fr3_workspace.png")
    renderer.close()
    record.close()


if __name__ == "__main__":
    main()
