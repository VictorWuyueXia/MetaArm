"""Numerical contracts for FR3 geometry, reference IK, and batched CUDA dynamics."""

import tomllib
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from metaarm.ik import solve_ik
from metaarm.scene import build_scene
from metaarm.simulation import Simulation


def test_scene_contract():
    """Compare the declared base-aligned site Jacobian with central differences."""
    model = build_scene()
    data = mujoco.MjData(model)
    home = model.key("home").qpos.copy()
    data.qpos[:] = home
    mujoco.mj_forward(model, data)
    site = model.site("attachment_site").id
    assert model.nq == model.nv == model.nu == 7
    assert data.ncon == 0
    np.testing.assert_allclose(data.site_xpos[site], [0.55449948, 0, 0.62450243], atol=1e-7)
    jac_p, jac_r = np.zeros((3, 7)), np.zeros((3, 7))
    mujoco.mj_jacSite(model, data, jac_p, jac_r, site)

    # Differentiate translation and rotation at the same physical tool origin.
    numerical = np.zeros((6, 7))
    quat_plus, quat_minus, rotation_error = np.empty(4), np.empty(4), np.empty(3)
    for joint in range(7):
        data.qpos[:] = home
        data.qpos[joint] += 1e-6
        mujoco.mj_forward(model, data)
        plus = data.site_xpos[site].copy()
        mujoco.mju_mat2Quat(quat_plus, data.site_xmat[site])
        data.qpos[joint] -= 2e-6
        mujoco.mj_forward(model, data)
        numerical[:3, joint] = (plus - data.site_xpos[site]) / 2e-6
        mujoco.mju_mat2Quat(quat_minus, data.site_xmat[site])
        mujoco.mju_subQuat(rotation_error, quat_plus, quat_minus)
        numerical[3:, joint] = data.site_xmat[site].reshape(3, 3) @ rotation_error / 2e-6
    np.testing.assert_allclose(numerical, np.vstack((jac_p, jac_r)), atol=1e-6)


def test_ik_contract():
    """Verify target poses, seed preservation, bounds, and explicit solve failure."""
    model = build_scene()
    data = mujoco.MjData(model)
    seed = model.key("home").qpos.copy()
    experiment = Path(__file__).resolve().parents[1] / "experiments" / "reach.toml"
    targets = tomllib.loads(experiment.read_text())["target_poses"]
    for target in targets:
        solution, residual = solve_ik(model, data, np.asarray(target), seed)
        assert residual[0] < 1e-4
        assert residual[1] < 1e-3
        assert np.all(solution >= model.jnt_range[:, 0])
        assert np.all(solution <= model.jnt_range[:, 1])
    np.testing.assert_array_equal(seed, model.key("home").qpos)

    # Failure is observable and does not substitute another target or solver.
    pytest.raises(RuntimeError, solve_ik, model, data, np.array([5, 0, 5, 1, 0, 0, 0]), seed)
    pytest.raises(ValueError, solve_ik, model, data, np.array([0.5, 0, 0.5, 2, 0, 0, 0]), seed)
    pytest.raises(ValueError, solve_ik, model, data, np.asarray(targets[0]), np.ones(7) * 100)


def test_simulation_contract():
    """Check CPU/GPU agreement, shared memory, independent worlds, and complete reset."""
    model = build_scene()
    cpu = mujoco.MjData(model)
    sim = Simulation(model, 4)
    assert sim.q.device.type == "cuda"
    assert sim.ctrl.data_ptr() == wp.to_torch(sim.data.ctrl).data_ptr()
    contact_pose = np.array([0.55, 0.30, 0.08, 0, 0.923898314, 0.382638086, 0])
    contact_q, _ = solve_ik(model, sim.kinematics, contact_pose, model.key("home").qpos)

    # Both engines start from identical states in free motion and actual box contact.
    for initial in (model.key("home").qpos.copy(), contact_q):
        q_initial = torch.tensor(initial, dtype=torch.float32, device="cuda:0")
        sim.reset(q_initial)
        mujoco.mj_resetData(model, cpu)
        cpu.qpos[:] = initial
        cpu.ctrl[:] = initial
        mujoco.mj_forward(model, cpu)
        if np.array_equal(initial, contact_q):
            assert cpu.ncon > 0
        for _ in range(30):
            sim.step(q_initial.expand(4, 7))
            for _ in range(5):
                mujoco.mj_step(model, cpu)
        mujoco.mj_forward(model, cpu)
        np.testing.assert_allclose(sim.qpos[0].cpu().numpy(), cpu.qpos, atol=1e-3, rtol=0)
        np.testing.assert_allclose(sim.qvel[0].cpu().numpy(), cpu.qvel, atol=2e-2, rtol=0)

    # A command in one world cannot move the remaining independent robot copies.
    sim.reset(sim.home)
    references = sim.home.expand(4, 7).clone()
    references[1, 0] += 0.1
    for _ in range(60):
        sim.step(references)
    assert sim.q[1, 0].item() > 0.08
    assert abs(sim.q[0, 0].item()) < 1e-4
    first = sim.q.clone()
    sim.reset(sim.home)
    assert torch.count_nonzero(sim.qvel).item() == 0
    assert torch.count_nonzero(sim.time).item() == 0
    for _ in range(60):
        sim.step(references)
    torch.testing.assert_close(sim.q, first, atol=1e-6, rtol=0)

    # Bad references fail before changing the simulation; no clipping or CPU path exists.
    pytest.raises(ValueError, sim.step, torch.ones((4, 7), device="cuda:0") * 100)
    pytest.raises(ValueError, sim.reset, torch.ones(7) * 100)
