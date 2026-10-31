"""Corpus completeness and cross-archive split isolation checks."""

from unittest.mock import Mock

import numpy as np
import pytest

import metaarm.collection as collection
from metaarm.motion_data import load_motions
from metaarm.scene import build_scene
from metaarm.simulation import Simulation


def test_all_archives(tmp_path, monkeypatch):
    """Every archive contributes its references, labels, and exact second moments."""
    sim = Simulation(build_scene(), 4)
    monkeypatch.setattr(collection, "Simulation", Mock(return_value=sim))
    monkeypatch.setattr(collection, "REFERENCE_COUNTS", (2, 1, 1))
    monkeypatch.setattr(collection, "randbits", Mock(side_effect=[17, 29]))
    collection.collect(tmp_path / "first.npz")
    collection.collect(tmp_path / "second.npz")
    records = load_motions(tmp_path)
    assert len(records["sources"]) == 2 and len(records["q_ref"]) == 8
    assert len(records["plane"]["reference"]) == 16
    assert set(records["pose"]["source_id"].cpu().tolist()) == {0, 1}
    assert records["pose"]["samples"].sum().item() == 8 * 65
    first = np.load(tmp_path / "first.npz", allow_pickle=False)
    delta = first["general_delta_q"].astype(np.float64)
    expected = np.swapaxes(delta, -1, -2) @ delta / delta.shape[1]
    np.testing.assert_allclose(
        records["pose"]["covariance"][:4].cpu().numpy(), expected, atol=1e-15
    )
    first.close()


def test_cross_archive_leakage(tmp_path, monkeypatch):
    """A copied reference cannot be relabelled as held out in another archive."""
    sim = Simulation(build_scene(), 4)
    monkeypatch.setattr(collection, "Simulation", Mock(return_value=sim))
    monkeypatch.setattr(collection, "REFERENCE_COUNTS", (2, 1, 1))
    collection.collect(tmp_path / "original.npz")
    archive = np.load(tmp_path / "original.npz", allow_pickle=False)
    arrays = {name: archive[name].copy() for name in archive.files}
    archive.close()
    arrays["general_split"][0] = "test"
    arrays["planar_split"][arrays["planar_reference_id"] == 0] = "test"
    np.savez_compressed(tmp_path / "crossed.npz", **arrays)
    pytest.raises(ValueError, load_motions, tmp_path)
