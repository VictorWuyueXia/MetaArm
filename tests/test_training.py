"""Algebra, exact covariance gradients, and checkpoint tests for motion subspaces."""

from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import torch

import metaarm.collection as collection
import metaarm.training as training
from metaarm.basis import MotionBasis, reference_geometry
from metaarm.scene import build_scene
from metaarm.simulation import Simulation


def test_basis_contract():
    """Check rank-preserving graphs, n/-n invariance, and base-yaw equivariance."""
    model = build_scene()
    state = mujoco.MjData(model)
    q = np.tile(model.key("home").qpos, (3, 1))
    q[:, 0] += [-0.2, 0.1, 0.3]
    jac = np.empty((3, 6, 7))
    for index, reference in enumerate(q):
        state.qpos[:] = reference
        mujoco.mj_forward(model, state)
        mujoco.mj_jacSite(
            model, state, jac[index, :3], jac[index, 3:], model.site("attachment_site").id
        )
    q = torch.tensor(q, device="cuda:0", dtype=torch.float64)
    jac = torch.tensor(jac, device="cuda:0", dtype=torch.float64)
    normals = torch.eye(3, device="cuda:0", dtype=torch.float64)
    specification = {"width": 32, "scale": 8.0, "normal_features": True, "equivariant": True}
    network = MotionBasis(model.jnt_range, specification).to(device="cuda:0", dtype=torch.float64)
    for task, dimension, normal in (("pose", 6, None), ("plane", 3, normals)):
        geometry = reference_geometry(jac, task)
        raw = network(q, geometry, task, normal, False)
        basis = network(q, geometry, task, normal)
        identity = torch.eye(dimension, device="cuda:0", dtype=torch.float64).expand(3, -1, -1)
        torch.testing.assert_close(jac[:, :dimension] @ raw, identity, atol=1e-10, rtol=0)
        torch.testing.assert_close(basis.mT @ basis, identity, atol=1e-10, rtol=0)
        torch.testing.assert_close(basis @ basis.mT @ raw, raw, atol=1e-10, rtol=0)
        if task == "plane":
            flipped = network(q, geometry, task, -normal)
            torch.testing.assert_close(basis @ basis.mT, flipped @ flipped.mT, atol=1e-12, rtol=0)

        # Rotate the world task axes and base joint together, keeping the joint subspace.
        angle = torch.tensor(0.25, device="cuda:0", dtype=torch.float64)
        rotation = torch.eye(3, device="cuda:0", dtype=torch.float64)
        rotation[:2, :2] = torch.stack(
            (torch.stack((angle.cos(), -angle.sin())), torch.stack((angle.sin(), angle.cos())))
        )
        transform = torch.block_diag(rotation, rotation)
        shifted = q.clone()
        shifted[:, 0] += angle
        rotated = None if normal is None else (rotation @ normal.unsqueeze(-1)).squeeze(-1)
        moved = network(shifted, reference_geometry(transform @ jac, task), task, rotated)
        torch.testing.assert_close(basis @ basis.mT, moved @ moved.mT, atol=1e-10, rtol=0)
    pytest.raises(ValueError, reference_geometry, torch.zeros_like(jac), "pose")
    pytest.raises(ValueError, network, q, reference_geometry(jac, "plane"), "plane", 2 * normals)


def test_basis_gradients():
    """Verify the sufficient-statistic projector loss and its gradients against QR samples."""
    torch.manual_seed(0)
    model = build_scene()
    spec = {"width": 16, "scale": 8.0, "normal_features": True, "equivariant": True}
    network = MotionBasis(model.jnt_range, spec).to(device="cuda:0", dtype=torch.float64)
    q = torch.tensor(
        model.key("home").qpos, device="cuda:0", dtype=torch.float64, requires_grad=True
    )
    jac = torch.eye(6, 7, device="cuda:0", dtype=torch.float64)
    jac[:, 6] = torch.arange(1, 7, device="cuda:0", dtype=torch.float64) / 10
    normal = torch.tensor([0.0, 0.0, 1.0], device="cuda:0", dtype=torch.float64)
    assert torch.autograd.gradcheck(
        network, (q, reference_geometry(jac, "plane"), "plane", normal), atol=1e-5, rtol=1e-3
    )
    samples = torch.randn(17, 7, device="cuda:0", dtype=torch.float64)
    covariance = samples.mT @ samples / len(samples)
    direct, sufficient = 0, 0
    for task, n in (("pose", None), ("plane", normal)):
        geometry = reference_geometry(jac, task)
        raw = network(q, geometry, task, n, False)
        basis = torch.linalg.qr(raw, mode="reduced")[0]
        direct = direct + (samples - samples @ basis @ basis.mT).square().mean()
        factor = torch.linalg.cholesky(raw.mT @ raw)
        residual = torch.eye(7, device="cuda:0") - raw @ torch.cholesky_solve(raw.mT, factor)
        sufficient = sufficient + (residual @ covariance @ residual.mT).trace() / 7
    torch.testing.assert_close(direct, sufficient, atol=1e-10, rtol=0)
    parameters = tuple(network.parameters())
    g1 = torch.autograd.grad(direct, parameters, retain_graph=True)
    g2 = torch.autograd.grad(sufficient, parameters)
    for first, second in zip(g1, g2, strict=True):
        torch.testing.assert_close(first, second, atol=1e-9, rtol=1e-5)
    assert torch.cat([g.flatten() for g in g2]).norm() > 1e-8


def test_training_checkpoint(tmp_path, monkeypatch):
    """Reject NaN failure labels and reproduce a validation-selected checkpoint after reload."""
    sim = Simulation(build_scene(), 4)
    monkeypatch.setattr(collection, "Simulation", Mock(return_value=sim))
    monkeypatch.setattr(collection, "REFERENCE_COUNTS", (2, 1, 1))
    source = tmp_path / "motions.npz"
    collection.collect(source)
    archive = np.load(source, allow_pickle=False)
    arrays = {name: archive[name].copy() for name in archive.files}
    archive.close()
    arrays["planar_included"][0] = False
    arrays["planar_success"][0] = False
    arrays["planar_label_mask"][0] = False
    arrays["planar_delta_q"][0] = np.nan
    source = tmp_path / "excluded.npz"
    np.savez_compressed(source, **arrays)
    records = training.load_motions(source)
    assert records["pose"]["covariance"].device.type == "cuda"
    assert torch.isfinite(records["plane"]["covariance"]).all()
    assert len(records["plane"]["covariance"]) == 7
    specification = {
        "model": {"width": 16, "scale": 8.0, "normal_features": True, "equivariant": True},
        "training": {"epochs": 2, "learning_rate": 1e-3, "weight_decay": 1e-4},
    }
    destination = tmp_path / "basis.pt"
    checkpoint = training.train(source, destination, specification)
    loaded = torch.load(destination, weights_only=True)
    assert loaded["metadata"]["specification"]["training"]["epochs"] == 2
    assert loaded["history"][-1]["epoch"] == 2 and "test" not in loaded["final"]
    assert all("test" not in epoch["metrics"] for epoch in loaded["history"])
    network = MotionBasis(records["joint_limits"], specification["model"]).to("cuda:0")
    network.load_state_dict(loaded["state_dict"])
    measured = training.evaluate(network, records, ("train", "validation"))
    assert measured == loaded["final"] == checkpoint["final"]
    assert measured["validation"]["sum"] <= loaded["initial"]["validation"]["sum"]
    assert torch.is_grad_enabled() and network.training
    pytest.raises(FileExistsError, training.train, source, destination, specification)
    arrays["planar_split"][1] = "test"
    invalid = tmp_path / "crossed.npz"
    np.savez_compressed(invalid, **arrays)
    pytest.raises(ValueError, training.load_motions, invalid)
