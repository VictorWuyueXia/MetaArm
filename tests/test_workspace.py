"""Regional data integrity, neural chart identities, and complete pipeline checks."""

import json

import mujoco
import numpy as np
import pytest
import torch

import metaarm.workspace_data as generation
from metaarm.basis import TaskBasis, reference_geometry
from metaarm.expansion_evaluation import validate_experiment
from metaarm.gpu_ik import CoefficientIK
from metaarm.scene import build_scene
from metaarm.workspace import train_workspace


def test_workspace_collection(tmp_path, monkeypatch):
    """Keep all records regional and retain a nonlinear full-joint teacher lifting."""
    monkeypatch.setattr(generation, "COUNTS", (2, 1, 1))
    path = tmp_path / "regional.npz"
    generation.collect_workspace(path)
    archive = np.load(path, allow_pickle=False)
    assert archive["q"].shape == (4, 33, 7) and archive["included"].all()
    assert len(np.unique(archive["q_ref"], axis=0)) == 4
    assert archive["split"].tolist() == ["train", "train", "validation", "test"]
    for key in ("target", "actual_position", "reference_position"):
        assert np.all((archive[key] >= generation.BOX[0]) & (archive[key] <= generation.BOX[1]))
    assert np.max(archive["teacher_error"]) < 1e-7
    np.testing.assert_array_equal(archive["delta_q"][:, 0], 0)
    np.testing.assert_array_equal(archive["delta_q"][:, -1], 0)
    assert np.linalg.matrix_rank(archive["delta_q"].reshape(-1, 7), tol=1e-6) > 3
    model, cpu = build_scene(), mujoco.MjData(build_scene())
    cpu.qpos[:] = archive["q"][0, 10]
    mujoco.mj_forward(model, cpu)
    jp = np.empty((3, model.nv))
    mujoco.mj_jacSite(model, cpu, jp, None, model.site("attachment_site").id)
    gradient = cpu.qpos - archive["q_center"]
    null_component = gradient - jp.T @ np.linalg.solve(jp @ jp.T, jp @ gradient)
    assert np.linalg.norm(null_component) < 1e-5
    archive.close()
    pytest.raises(FileExistsError, generation.collect_workspace, path)


def test_position_basis():
    """Verify position rank and exact covariance/sample loss and parameter derivatives."""
    model = build_scene()
    teacher = generation.lift_positions(model, generation.CENTER[None], model.key("home").qpos)
    q = torch.tensor(teacher["q"], device="cuda:0", dtype=torch.float64)
    jac = torch.tensor(teacher["jacobian"], device="cuda:0", dtype=torch.float64)
    geometry = reference_geometry(jac, "position")
    torch.manual_seed(0)
    network = TaskBasis(model.jnt_range, {"width": 16, "scale": 1.0}, "position", 2).to(
        device="cuda:0", dtype=torch.float64
    )
    raw, basis = network(q, geometry, False), network(q, geometry)
    torch.testing.assert_close(
        jac[:, :3] @ raw, torch.eye(3, device="cuda:0").double()[None], atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        basis.mT @ basis, torch.eye(3, device="cuda:0").double()[None], atol=1e-12, rtol=0
    )
    samples = torch.randn(1, 33, 7, device="cuda:0", dtype=torch.float64)
    covariance = samples.mT @ samples / 33
    direct = (samples - samples @ basis @ basis.mT).square().mean()
    compressed = network.reconstruction(q, geometry, covariance)
    torch.testing.assert_close(direct, compressed, atol=1e-12, rtol=0)
    parameters = tuple(network.parameters())
    g1, g2 = torch.autograd.grad(direct, parameters), torch.autograd.grad(compressed, parameters)
    for first, second in zip(g1, g2, strict=True):
        torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-8)


def test_workspace_checkpoint(tmp_path, monkeypatch):
    """Reload selected regional weights and exercise matched position-only GPU validation."""
    monkeypatch.setattr(generation, "COUNTS", (2, 1, 1))
    source, destination = tmp_path / "regional.npz", tmp_path / "regional.pt"
    generation.collect_workspace(source)
    training = {
        "depth": 2,
        "epochs": 500,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheme": "reconstruction",
    }
    result = train_workspace(source, destination, {"width": 16, "scale": 1.0}, training)
    loaded = torch.load(destination, weights_only=True)
    assert loaded["final"]["validation"] <= loaded["initial"]["validation"]
    assert len(loaded["history"]) == 50 and loaded["history"][-1]["epoch"] == 500
    assert "test" not in loaded["final"] and loaded["final"] == result["final"]
    solver = CoefficientIK(build_scene(), 1, destination)
    assert isinstance(solver.network, TaskBasis)
    report = validate_experiment(source, destination, "validation", tmp_path / "validation")
    assert report["trajectory"]["full"]["completed"] == 1
    assert report["trajectory"]["full"]["cpu_disagreements"] == 0
    records = np.load(tmp_path / "validation/records.npz", allow_pickle=False)
    archive = np.load(source, allow_pickle=False)
    row = np.flatnonzero(archive["split"] == "validation")
    q = torch.tensor(archive["q_ref"][row], device="cuda:0", dtype=torch.float32)
    j = torch.tensor(archive["J_ref"][row], device="cuda:0", dtype=torch.float32)
    delta = torch.tensor(archive["delta_q"][row], device="cuda:0", dtype=torch.float64)
    loss = solver.network.reconstruction(
        q, reference_geometry(j, "position"), delta.mT @ delta / 33
    ).item()
    assert loss == pytest.approx(loaded["final"]["validation"], abs=1e-16, rel=0)
    assert records["trajectory_learned_success"][0, 0]
    assert json.loads(archive["metadata"].item())["kind"] == "workspace_position_data"
    archive.close()
    records.close()
    pytest.raises(
        FileExistsError, train_workspace, source, destination, {"width": 16, "scale": 1.0}, training
    )
