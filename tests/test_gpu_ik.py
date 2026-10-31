"""Independent geometry, constrained optimization, and frozen-chart IK checks."""

import mujoco
import numpy as np
import pytest
import torch
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial.transform import Rotation

from metaarm.basis import reference_geometry
from metaarm.gpu_ik import CHECKPOINT, CoefficientIK, constrained_step
from metaarm.scene import build_scene
from metaarm.simulation import Simulation


def test_geometry_and_residual():
    """Check GPU FK/Jacobians and the nonzero rotation-log derivative against CPU FK."""
    model = build_scene()
    solver = CoefficientIK(model, 4, CHECKPOINT)
    rng = np.random.default_rng(31)
    q = model.key("home").qpos + rng.uniform(-0.15, 0.15, (4, 7))
    q = torch.tensor(q, device="cuda:0", dtype=torch.float64)
    position, rotation, jacobian = solver.geometry(q)
    cpu = mujoco.MjData(model)
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    for row in range(4):
        cpu.qpos[:] = q[row].cpu().numpy()
        mujoco.mj_forward(model, cpu)
        mujoco.mj_jacSite(model, cpu, jp, jr, solver.site)
        np.testing.assert_allclose(position[row].cpu(), cpu.site_xpos[solver.site], atol=2e-7)
        np.testing.assert_allclose(
            rotation[row].cpu(), cpu.site_xmat[solver.site].reshape(3, 3), atol=5e-7
        )
        np.testing.assert_allclose(jacobian[row].cpu(), np.vstack((jp, jr)), atol=5e-7)

    target = {
        "position": position + 0.02,
        "rotation": rotation
        @ torch.tensor(Rotation.from_rotvec([0.35, -0.25, 0.15]).as_matrix(), device="cuda:0"),
    }
    error, derivative = solver.residual(q, target, "pose")
    numerical = np.empty((6, 7))
    for joint in range(7):
        residuals = []
        for sign in (1, -1):
            cpu.qpos[:] = q[0].cpu().numpy()
            cpu.qpos[joint] += sign * 1e-5
            mujoco.mj_forward(model, cpu)
            relative = target["rotation"][0].cpu().numpy().T @ cpu.site_xmat[solver.site].reshape(
                3, 3
            )
            residuals.append(
                np.r_[
                    (cpu.site_xpos[solver.site] - target["position"][0].cpu().numpy()) / 1e-3,
                    Rotation.from_matrix(relative).as_rotvec() / (np.pi / 180),
                ]
            )
        numerical[:, joint] = (residuals[0] - residuals[1]) / 2e-5
    np.testing.assert_allclose(derivative[0].cpu(), numerical, atol=7e-4, rtol=1e-4)
    zero, _ = solver.residual(q, {"position": position, "rotation": rotation}, "pose")
    assert error.isfinite().all() and zero.abs().max() < 1e-4


def test_constrained_step():
    """Compare the coupled CUDA quadratic solution with independent constrained SLSQP."""
    generator = torch.Generator(device="cuda:0").manual_seed(12)
    b = torch.linalg.qr(
        torch.randn(3, 7, 3, generator=generator, device="cuda:0", dtype=torch.float64)
    )[0]
    h = torch.eye(3, device="cuda:0", dtype=torch.float64).expand(3, -1, -1).clone()
    g = torch.tensor([[0.001, 0.002, -0.001], [3.0, -2.0, 1.0], [-5.0, 1.0, 4.0]], device="cuda:0")
    margin = torch.full((3, 14), 0.15, device="cuda:0", dtype=torch.float64)
    d = constrained_step(h, g.double(), b, margin)
    assert torch.all((torch.cat((b, -b), 1) @ d[..., None]).squeeze(-1) < margin)
    for row in range(3):
        a, gradient = b[row].cpu().numpy(), g[row].double().cpu().numpy()
        answer = minimize(
            lambda x: 0.5 * x @ x + gradient @ x,
            np.zeros(3),
            jac=lambda x: x + gradient,
            method="SLSQP",
            constraints=LinearConstraint(a, -0.15, 0.15),
            options={"ftol": 1e-12, "maxiter": 100},
        )
        assert answer.success
        np.testing.assert_allclose(d[row].cpu(), answer.x, atol=2e-6)


def test_fixed_chart_solve():
    """Reach known chart targets, preserve affine coordinates, and retain unreachable failures."""
    sim = Simulation(build_scene(), 4)
    solver = CoefficientIK(sim.model, 4, CHECKPOINT)
    physical = [v.clone() for v in (sim.qpos, sim.qvel, sim.ctrl, sim.time)]
    q = (
        torch.tensor(solver.model.key("home").qpos, device="cuda:0", dtype=torch.float32)
        .expand(4, -1)
        .clone()
    )
    normal = torch.tensor([0.0, 0.0, 1.0], device="cuda:0").expand(4, -1)
    for task in ("pose", "plane"):
        _, _, jacobian = solver.geometry(q)
        n = None if task == "pose" else normal
        b = solver.network(q, reference_geometry(jacobian.float(), task), task, n).detach().double()
        original = b.clone()
        alpha = torch.full((4, b.shape[-1]), 0.04, device="cuda:0", dtype=torch.float64)
        p, r, _ = solver.geometry(q.double() + (b @ alpha[..., None]).squeeze(-1))
        target = {"position": p, "rotation": r}
        for basis in (b, torch.eye(7, device="cuda:0").expand(4, -1, -1)):
            result = solver.solve(
                q, basis, target, task, torch.zeros((4, basis.shape[-1]), device="cuda:0")
            )
            assert result["success"].all() and (result["joint_margin"] > 0).all()
            torch.testing.assert_close(
                result["q"], q.double() + (basis.double() @ result["alpha"][..., None]).squeeze(-1)
            )
        torch.testing.assert_close(b, original)
        target["position"] = torch.full((4, 3), 10.0, device="cuda:0")
        failed = solver.solve(q, b, target, task, torch.zeros_like(alpha))
        assert not failed["success"].any()
        assert torch.isfinite(failed["q"]).all() and (failed["joint_margin"] > 0).all()
    pytest.raises(ValueError, solver.solve, q, b, target, "pose", torch.zeros_like(alpha))
    for original, current in zip(physical, (sim.qpos, sim.qvel, sim.ctrl, sim.time), strict=True):
        torch.testing.assert_close(original, current, atol=0, rtol=0)
