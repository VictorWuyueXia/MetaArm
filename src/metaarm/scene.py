"""Compile the FR3 experiment and establish its physical model contract."""

from pathlib import Path

import mujoco
import numpy as np


def build_scene() -> mujoco.MjModel:
    """Load the pinned FR3 and verify its named state and actuator mapping."""
    # Compile the project scene without modifying the upstream robot assets.
    root = Path(__file__).resolve().parents[2]
    scene = mujoco.MjSpec.from_file(str(root / "assets" / "franka_fr3" / "fr3.xml"))
    environment = mujoco.MjSpec.from_file(str(root / "assets" / "reach.xml"))
    scene.attach(environment, prefix="scene/", frame=scene.worldbody.add_frame())
    scene.option.timestep = 0.002
    model = scene.compile()
    names = [f"fr3_joint{i}" for i in range(1, 8)]
    joints = np.array([model.joint(name).id for name in names])
    actuators = np.array([model.actuator(name).id for name in names])
    q_indices = model.jnt_qposadr[joints]
    v_indices = model.jnt_dofadr[joints]

    # Contiguous named slices retain shared GPU views without assuming nq = 7.
    if not all(np.all(np.diff(indices) == 1) for indices in (q_indices, v_indices, actuators)):
        raise ValueError("FR3 arm coordinates and actuators must be contiguous and ordered.")
    if not np.all(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("The FR3 contract requires seven revolute arm joints.")
    if not np.array_equal(model.actuator_trnid[actuators, 0], joints):
        raise ValueError("FR3 actuator transmissions must follow the named joint order.")
    if not np.all(model.jnt_limited[joints]):
        raise ValueError("Every FR3 arm joint must have an explicit position limit.")

    # The base axes are the world axes, and the tool is the flange attachment site.
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    base = model.body("fr3_link0").id
    tool = model.site("attachment_site").id
    bounds = model.jnt_range[joints]
    home = data.qpos[q_indices]
    if not np.allclose(data.xpos[base], 0) or not np.allclose(data.xmat[base], np.eye(3).ravel()):
        raise ValueError("The FR3 base frame must coincide with the world frame.")
    if not np.all(np.isfinite(data.site_xpos[tool])):
        raise ValueError("The home tool position must be finite.")
    if not np.all((home > bounds[:, 0]) & (home < bounds[:, 1])):
        raise ValueError("The home configuration must be interior to the joint limits.")
    if model.opt.timestep != 0.002:
        raise ValueError("The physics interval is fixed at 0.002 seconds.")
    return model
