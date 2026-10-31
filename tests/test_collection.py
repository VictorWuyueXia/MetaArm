"""Numerical and archive checks for nominal general and planar motion records."""

import json
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

import metaarm.collection as collection
import metaarm.ik as ik
from metaarm.scene import build_scene
from metaarm.simulation import Simulation


def test_general_motion_contract():
    """Verify feasible smooth paths, seven joint directions, and GPU/CPU pose agreement."""
    sim = Simulation(build_scene(), 4)
    references = np.tile(sim.home.cpu().numpy(), (4, 1))
    references[:, 0] = [-0.45, -0.10, 0.30, 0.60]
    general = collection.generate_motions(sim, references, 0)
    q = general["q"]
    np.testing.assert_array_equal(q[:, 0], general["q_ref"])
    np.testing.assert_allclose(q[:, -1], general["q_ref"], atol=1e-7)
    np.testing.assert_allclose(general["qd"][:, [0, -1]], 0, atol=3e-6)
    np.testing.assert_array_equal(general["delta_q"], q - general["q_ref"][:, None])
    assert np.max(np.abs(general["delta_q"])) <= 0.30
    assert np.all(q > sim.bounds[:, 0].cpu().numpy())
    assert np.all(q < sim.bounds[:, 1].cpu().numpy())
    assert np.all(np.linalg.svd(general["delta_q"], compute_uv=False)[:, -1] > 1e-5)

    # Match a non-reference sample against independent CPU forward kinematics.
    sim.kinematics.qpos[sim.q_indices] = q[2, 19]
    mujoco.mj_forward(sim.model, sim.kinematics)
    np.testing.assert_allclose(general["p"][2, 19], sim.kinematics.site_xpos[sim.site], atol=5e-7)
    np.testing.assert_allclose(
        general["R"][2, 19], sim.kinematics.site_xmat[sim.site].reshape(3, 3), atol=5e-7
    )
    repeated = collection.generate_motions(sim, references, 0)
    np.testing.assert_array_equal(repeated["q"], q)
    changed = collection.generate_motions(sim, references, 1)
    assert not np.array_equal(changed["harmonics"], general["harmonics"])
    assert not np.array_equal(changed["q"], q)
    pytest.raises(ValueError, collection.generate_motions, sim, references[:3], 0)


def test_position_reference_contract(monkeypatch):
    """Check the SLSQP gradient, orientation-free labels, and trajectory failure masks."""
    sim = Simulation(build_scene(), 4)
    seed = sim.home.cpu().numpy().astype(np.float64)
    sim.kinematics.qpos[sim.q_indices] = seed + [0.1, 0.05, -0.1, 0.03, 0, 0.02, 0]
    mujoco.mj_forward(sim.model, sim.kinematics)
    target = sim.kinematics.site_xpos[sim.site].copy()
    spy = Mock(wraps=ik.minimize)
    monkeypatch.setattr(ik, "minimize", spy)
    result = ik.solve_position(sim.model, sim.kinematics, target, seed)
    assert result["feasible"] and result["errors"][0] <= 1e-3
    assert result["errors"].shape == (1,)
    objective = spy.call_args.args[0]
    _, gradient = objective(seed)
    numerical = np.empty(7)
    for joint in range(7):
        perturbation = np.eye(7)[joint] * 1e-6
        numerical[joint] = (
            objective(seed + perturbation)[0] - objective(seed - perturbation)[0]
        ) / 2e-6
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-4)

    # A failed teacher solve stops only its own path and excludes all its joint labels.
    failure = ik.solve_position(sim.model, sim.kinematics, np.array([5, 0, 5]), seed)
    assert failure["errors"][0] > 1e-3
    references = np.tile(seed, (4, 1))
    general = collection.generate_motions(sim, references, 0)
    failed_solver = Mock(return_value=failure)
    monkeypatch.setattr(collection, "solve_position", failed_solver)
    normals = np.tile([0.0, 0.0, 1.0], (4, 1, 1))
    planar = collection.project_motions(sim, general, normals)
    assert failed_solver.call_count == 4
    assert planar["attempted"][:, 0].all() and not planar["attempted"][:, 1:].any()
    assert not planar["label_mask"].any() and not planar["included"].any()
    assert np.isnan(planar["q"][:, 1:]).all()
    np.testing.assert_array_equal(planar["status"][:, 1:], -1)
    np.testing.assert_array_equal(planar["q"][:, 0], np.tile(failure["q"], (4, 1)))
    assert np.all(planar["message"][:, 0] != "unattempted")


def test_collection_archive(tmp_path, monkeypatch):
    """Round-trip split-aware records with real planar labels and inspect every label."""
    sim = Simulation(build_scene(), 4)
    monkeypatch.setattr(collection, "Simulation", Mock(return_value=sim))
    monkeypatch.setattr(collection, "REFERENCE_COUNTS", (2, 1, 1))
    monkeypatch.setattr(collection, "randbits", Mock(side_effect=[123, 456]))
    destination = tmp_path / "motions.npz"
    summary = collection.collect(destination)
    data = np.load(destination, allow_pickle=False)
    metadata = json.loads(data["metadata"].item())
    assert metadata["seed"] == 123
    assert metadata["kind"] == "nominal_kinematic_training_data"
    assert metadata["assets"] and metadata["git_revision"]
    assert summary["planar_planned_waypoints"] == 8 * 65
    np.testing.assert_array_equal(data["general_split"], ["train", "train", "validation", "test"])
    np.testing.assert_array_equal(
        data["planar_split"], data["general_split"][data["planar_reference_id"]]
    )

    # The projected targets lie on their declared planes, with orthonormal tangent axes.
    normals, origin, axes = data["planar_normal"], data["planar_origin"], data["planar_axes"]
    np.testing.assert_allclose(
        np.einsum("ptj,pj->pt", data["planar_target"] - origin[:, None], normals), 0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.einsum("pji,pjk->pik", axes, axes), np.tile(np.eye(2), (8, 1, 1)), atol=1e-12
    )
    np.testing.assert_allclose(np.einsum("pji,pj->pi", axes, normals), 0, atol=1e-12)
    assert data["planar_included"].any()
    assert np.array_equal(data["planar_label_mask"].any(axis=1), data["planar_included"])
    assert not any("angular" in key or "rotation_error" in key for key in data.files)

    # Re-evaluate every usable label on CPU rather than trusting stored solver errors.
    for episode, step in np.argwhere(data["planar_label_mask"]):
        sim.kinematics.qpos[sim.q_indices] = data["planar_q"][episode, step]
        mujoco.mj_forward(sim.model, sim.kinematics)
        assert (
            np.linalg.norm(
                sim.kinematics.site_xpos[sim.site] - data["planar_target"][episode, step]
            )
            <= 1e-3
        )
    # A second collection changes references, planes, and harmonic paths, not just filenames.
    second_path = tmp_path / "motions_second.npz"
    second_summary = collection.collect(second_path)
    second = np.load(second_path, allow_pickle=False)
    assert second_summary["seed"] == 456
    assert json.loads(second["metadata"].item())["seed"] == 456
    for field in ("general_q_ref", "general_harmonics", "general_q", "planar_normal"):
        assert not np.array_equal(data[field], second[field])
    np.testing.assert_array_equal(data["general_split"], second["general_split"])
    second.close()
    data.close()
    pytest.raises(FileExistsError, collection.collect, destination)
