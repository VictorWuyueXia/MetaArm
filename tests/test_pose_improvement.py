"""Pose-chart algebra, derivative omission, constrained search, and singular rank checks."""

import json

import mujoco
import numpy as np
import torch

from metaarm.gpu_ik import CHECKPOINT, CoefficientIK
from metaarm.pose_model import PoseChart, pose_geometry
from metaarm.pose_solver import solve_pose
from metaarm.pose_study import singularity_report
from metaarm.scene import build_scene


def test_pose_chart(tmp_path):
    """Check the rank-preserving graph and differentiable, strictly feasible initialization."""
    model = build_scene()
    solver = CoefficientIK(model, 4, CHECKPOINT)
    q = torch.tensor(model.key("home").qpos, device="cuda:0").expand(4, -1).double()
    _, _, jacobian = solver.geometry(q)
    geometry = pose_geometry(jacobian)
    inverse, null = geometry
    np.testing.assert_allclose(
        (jacobian @ inverse).cpu(), np.tile(np.eye(6), (4, 1, 1)), atol=1e-12
    )
    assert (jacobian @ null[:, :, None]).abs().max() < 1e-12
    network = PoseChart(solver.bounds, 32, 0.48).cuda()
    displacement = torch.tensor([[0.03, -0.04, 0.02, 0.1, -0.1, 0.1]], device="cuda:0").expand(
        4, -1
    )
    basis, alpha, _ = network(q, geometry, displacement)
    np.testing.assert_allclose(
        (basis.mT @ basis).detach().cpu(), np.tile(np.eye(6), (4, 1, 1)), atol=1e-12
    )
    initial = q + (basis @ alpha[:, :, None]).squeeze(-1)
    assert torch.all(initial > solver.bounds[:, 0]) and torch.all(initial < solver.bounds[:, 1])
    loss, _ = network.loss(q, geometry, displacement, torch.zeros_like(q))
    loss.backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in network.parameters())
    # The standard GPU owner loads the new architecture with the same physical-asset checks.
    metadata = torch.load(CHECKPOINT, weights_only=True)["metadata"]
    metadata["specification"]["model"] = {"width": 32, "side": 0.48}
    checkpoint = tmp_path / "pose.pt"
    torch.save({"state_dict": network.state_dict(), "metadata": metadata}, checkpoint)
    loaded = CoefficientIK(model, 4, checkpoint)
    assert isinstance(loaded.network, PoseChart)
    restored = loaded.network(q, geometry, displacement)
    torch.testing.assert_close(restored[0], basis)
    torch.testing.assert_close(restored[1], alpha)


def test_pose_solver():
    """Compare reduced/full pose search and verify that residual-only FK preserves errors."""
    model = build_scene()
    solver = CoefficientIK(model, 3, CHECKPOINT)
    q = torch.tensor(model.key("home").qpos, device="cuda:0").expand(3, -1).double()
    position, rotation, jacobian = solver.geometry(q)
    target = {
        "position": position + torch.tensor([0.02, 0.01, -0.02], device="cuda:0"),
        "rotation": rotation,
    }
    error, derivative = solver.residual(q, target, "pose")
    fast, omitted = solver.residual(q, target, "pose", differentiate=False)
    torch.testing.assert_close(error, fast)
    assert derivative.shape == (3, 6, 7) and omitted is None
    network = PoseChart(solver.bounds, 32, 0.32).cuda()
    displacement = torch.cat((target["position"] - position, torch.zeros_like(position)), -1)
    basis, initial, _ = network(q, pose_geometry(jacobian), displacement)
    reduced = solve_pose(solver, q, basis.detach(), target, initial.detach())
    full = solve_pose(
        solver, q, torch.eye(7, device="cuda:0").expand(3, -1, -1), target, torch.zeros_like(q)
    )
    assert reduced["success"].all() and full["success"].all()
    assert reduced["joint_margin"].min() > 0
    # A reached branch supersedes its peers without labeling those peers successful.
    origins = q[:1].repeat(3, 1)
    origins[1:, 0] += torch.tensor([0.05, -0.05], device="cuda:0")
    grouped_target = {
        "position": position[:1].repeat(3, 1).reshape(1, 3, 3),
        "rotation": rotation[:1].repeat(3, 1, 1).reshape(1, 3, 3, 3),
    }
    grouped = solve_pose(
        solver,
        origins.reshape(1, 3, 7),
        basis.detach().reshape(1, 3, 7, 6),
        grouped_target,
        torch.zeros((1, 3, 6), device="cuda:0"),
    )
    assert grouped["success"].tolist() == [True, False, False]
    assert grouped["status"].tolist() == [0, 3, 3]
    assert grouped["iterations"].sum() == 0


def test_singularity_certificate(tmp_path):
    """Verify the exact elbow factor and rank loss inside the admitted pose volumes."""
    report = singularity_report(tmp_path / "singularity.json")
    assert abs(report["planar_factor"]) < 1e-16
    assert report["sigma"][-1] < 1e-14 and report["sigma"][-2] > 1e-3
    assert report["joint_margin"] > 0
    assert report["inside_cube"] == {
        "0.16": False,
        "0.24": True,
        "0.32": True,
        "0.4": True,
        "0.48": True,
    }
    assert json.loads((tmp_path / "singularity.json").read_text()) == report
    # The rank certificate persists along the free shoulder and wrist family coordinates.
    model = build_scene()
    data = mujoco.MjData(model)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for shoulder, wrist in ((-0.4, 0.7), (0.5, 1.6), (1.0, 2.8)):
        q = np.array(report["q"])
        q[0], q[1], q[5], q[6] = 0.3, shoulder, wrist, 0.4
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        mujoco.mj_jacSite(model, data, jp, jr, model.site("attachment_site").id)
        assert np.linalg.svd(np.vstack((jp, jr)), compute_uv=False)[-1] < 1e-14
