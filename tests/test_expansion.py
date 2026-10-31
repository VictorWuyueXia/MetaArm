"""Full-pose teacher, generic chart, and paired experiment/timing contracts."""

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import metaarm.expansion as expansion
import metaarm.expansion_data as generation
from metaarm.basis import TaskBasis, reference_geometry
from metaarm.expansion_evaluation import validate_experiment
from metaarm.scene import build_scene
from metaarm.workspace import train_workspace
from metaarm.workspace_data import CENTER, lift_positions


def test_pose_teacher():
    """Check exact pose constraints and a full seven-joint minimum-motion stationary solution."""
    model = build_scene()
    center = lift_positions(model, CENTER[None], model.key("home").qpos)
    q = center["q"][0]
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = q
    mujoco.mj_forward(model, cpu)
    rotation = cpu.site_xmat[model.site("attachment_site").id].reshape(3, 3)
    targets = CENTER + np.array([[0.02, -0.01, 0.01], [-0.01, 0.02, -0.02]])
    rotations = (
        rotation @ Rotation.from_rotvec([[0.08, -0.04, 0.05], [-0.05, 0.06, 0.03]]).as_matrix()
    )
    labels = generation.lift_pose(model, targets, rotations, q)
    assert labels["success"].all() and labels["error"].max() < 1e-7
    assert labels["angular_error"].max() < 1e-7
    for row in range(2):
        gradient = labels["q"][row] - q
        jacobian = labels["jacobian"][row]
        projected = gradient - jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T, jacobian @ gradient
        )
        assert np.linalg.norm(projected) < 1e-5


def test_task_basis():
    """Verify three/six-column charts and exact gradients with both hidden-layer depths."""
    model = build_scene()
    teacher = lift_positions(model, CENTER[None], model.key("home").qpos)
    q = torch.tensor(teacher["q"], device="cuda:0", dtype=torch.float64)
    j = torch.tensor(teacher["jacobian"], device="cuda:0", dtype=torch.float64)
    for task in ("position", "pose"):
        m = 3 if task == "position" else 6
        for depth in (2, 3):
            network = (
                TaskBasis(model.jnt_range, {"width": 16, "scale": 1.0}, task, depth).double().cuda()
            )
            geometry = reference_geometry(j, task)
            raw, b = network(q, geometry, False), network(q, geometry)
            torch.testing.assert_close(
                j[:, :m] @ raw, torch.eye(m, device="cuda:0").double()[None], atol=1e-12, rtol=0
            )
            samples = torch.randn(1, 17, 7, device="cuda:0", dtype=torch.float64)
            direct = (samples - samples @ b @ b.mT).square().mean()
            compressed = network.reconstruction(q, geometry, samples.mT @ samples / 17)
            torch.testing.assert_close(direct, compressed, atol=1e-12, rtol=0)
            g1 = torch.autograd.grad(direct, tuple(network.parameters()))
            g2 = torch.autograd.grad(compressed, tuple(network.parameters()))
            for first, second in zip(g1, g2, strict=True):
                torch.testing.assert_close(first, second, atol=1e-9, rtol=1e-7)


def test_expansion_pipeline(tmp_path, monkeypatch):
    """Keep paired datasets identical and preserve dimension/timing identities after reload."""
    monkeypatch.setattr(generation, "COUNTS", (2, 1, 1))
    position, pose = tmp_path / "position.npz", tmp_path / "pose.npz"
    generation.collect_experiment(position, 0.16, "position")
    generation.collect_experiment(pose, 0.16, "pose")
    a, b = np.load(position, allow_pickle=False), np.load(pose, allow_pickle=False)
    np.testing.assert_array_equal(a["q_ref"], b["q_ref"])
    np.testing.assert_array_equal(a["target"], b["target"])
    a.close()
    b.close()
    training = {
        "depth": 3,
        "epochs": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "scheme": "mobility",
    }
    checkpoint = tmp_path / "pose.pt"
    fitted = train_workspace(pose, checkpoint, {"width": 16, "scale": 0.25}, training)
    assert fitted["metadata"]["task"] == "pose" and len(fitted["history"]) == 1
    report = validate_experiment(pose, checkpoint, "validation", tmp_path / "validation")
    assert report["trajectory"]["full"]["completed"] == 1
    for name, value in (
        ("BATCH_CASES", 4),
        ("LATENCY_CASES", 2),
        ("BATCH_REPEATS", 2),
        ("LATENCY_REPEATS", 2),
    ):
        monkeypatch.setattr(expansion, name, value)
    timing = expansion.benchmark_search(pose, checkpoint, tmp_path / "timing")
    assert timing["coordinates"] == 6 and timing["dense_system_entries"] == 36
    assert timing["coordinate_reduction_percent"] == 100 * (1 - 6 / 7)
    assert timing["modes"]["batch"]["cases"] == 4
    assert timing["modes"]["single"]["cases"] == 2
    assert timing["modes"]["single"]["cpu_disagreements"] == 0
    assert torch.is_grad_enabled()
