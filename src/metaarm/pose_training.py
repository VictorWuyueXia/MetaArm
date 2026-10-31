"""Independent regional pose-chart training with validation-only checkpoint selection."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from metaarm.pose_model import PoseChart, pose_geometry


def fit_pose(source, destination, width, epochs):
    """Fit every task-feasible waypoint while retaining the original reference-group split."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    archive = np.load(source, allow_pickle=False)
    metadata = json.loads(archive["metadata"].item())
    if metadata["kind"] != "workspace_pose_data":
        raise ValueError("Pose-chart fitting requires regional pose data.")
    root = Path(__file__).resolve().parents[2]
    for name, digest in metadata["assets"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"The pose-training asset differs: {name}.")
    count, steps = archive["q"].shape[:2]
    valid = (archive["teacher_error"] < 1e-7) & (archive["teacher_angular_error"] < 1e-7)
    valid &= archive["teacher_margin"] > 0
    q = np.repeat(archive["q_ref"], steps, axis=0)[valid.ravel()]
    jacobian = np.repeat(archive["J_ref"], steps, axis=0)[valid.ravel()]
    # Reference rotations come directly from each episode's zero-displacement waypoint.
    rotations = archive["target_rotation"][:, 0]
    relative = np.swapaxes(rotations[:, None], -1, -2) @ archive["target_rotation"]
    phi = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(count, steps, 3)
    phi = (rotations[:, None] @ phi[..., None]).squeeze(-1)
    translation = archive["target"] - archive["target"][:, :1]
    displacement = np.concatenate((translation, phi), -1).reshape(-1, 6)[valid.ravel()]
    label = archive["delta_q"].reshape(-1, 7)[valid.ravel()]
    split = np.repeat(archive["split"], steps)[valid.ravel()]
    tensors = [
        torch.tensor(value, device="cuda:0", dtype=torch.float64)
        for value in (q, jacobian, displacement, label)
    ]
    q, jacobian, displacement, label = tensors
    geometry = pose_geometry(jacobian)
    rows = {
        name: torch.tensor(np.flatnonzero(split == name), device="cuda:0")
        for name in ("train", "validation")
    }
    if any(len(row) == 0 for row in rows.values()):
        raise ValueError("Both training and validation labels are required.")
    torch.manual_seed(0)
    side = metadata["side"]
    network = PoseChart(metadata["joint_limits"], width, side).cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=2e-5)
    best, history = float("inf"), []
    started = time.perf_counter()
    # Full-batch supervision keeps each physically valid waypoint's original multiplicity.
    for epoch in range(epochs):
        r = rows["train"]
        optimizer.zero_grad(set_to_none=True)
        loss, _ = network.loss(q[r], tuple(v[r] for v in geometry), displacement[r], label[r])
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            r = rows["validation"]
            values = network.loss(q[r], tuple(v[r] for v in geometry), displacement[r], label[r])
            validation = values[1].item()
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_objective": loss.item(),
                    "validation_seed_mse": validation,
                }
            )
            if validation < best:
                best = validation
                state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
                best_epoch = epoch + 1
    torch.cuda.synchronize()
    report = {
        "width": width,
        "epochs": epochs,
        "side": side,
        "best_epoch": best_epoch,
        "validation_seed_mse": best,
        "seconds": time.perf_counter() - started,
        "labels": {name: len(r) for name, r in rows.items()},
        "history": history,
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "projection_weight": 4,
        "metadata": metadata,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "sources": [{"metadata": metadata}],
        "specification": {"model": {"width": width, "side": side}},
    }
    torch.save({"state_dict": state, "report": report, "metadata": envelope}, destination)
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2))
    archive.close()
    return report


if __name__ == "__main__":
    from metaarm.expansion_data import lift_pose
    from metaarm.scene import build_scene

    study = Path("results/pose_improvement_20261031T205242Z")
    model = build_scene()
    rng = np.random.default_rng(20261031)
    for side in (32, 40, 48):
        source = Path(f"results/expansion_20261031T123139402928Z/side{side}_pose/motions.npz")
        destination = study / f"side{side}" / "boundary_motions.npz"
        if destination.exists():
            raise FileExistsError(destination)
        archive = np.load(source, allow_pickle=False)
        values = {key: archive[key].copy() for key in archive.files}
        rows = np.flatnonzero(archive["split"] != "test")
        metadata = json.loads(archive["metadata"].item())
        targets = rng.uniform(-1, 1, (len(rows), 33, 3))
        # Alternating faces and edges expose extreme poses without using held-out lattice labels.
        targets[:, 1:, 0] = rng.choice([-1, 1], (len(rows), 32))
        targets[::2, 1:, 1] = rng.choice([-1, 1], (len(rows[::2]), 32))
        targets *= side / 200
        targets += np.array([0.5, 0, 0.5])
        angles = rng.uniform(-np.pi / 18, np.pi / 18, (len(rows), 33, 3))
        rotations = (
            archive["rotation_center"] @ Rotation.from_rotvec(angles.reshape(-1, 3)).as_matrix()
        )
        rotations = rotations.reshape(len(rows), 33, 3, 3)
        targets[:, 0] = archive["target"][rows, 0]
        rotations[:, 0] = archive["target_rotation"][rows, 0]
        labels = []
        for index, row in enumerate(rows):
            labels.append(lift_pose(model, targets[index], rotations[index], archive["q_ref"][row]))
            if index % 24 == 0:
                print("LABEL", side, index, flush=True)
        mapping = {
            "q": "q",
            "J": "jacobian",
            "teacher_error": "error",
            "teacher_angular_error": "angular_error",
            "teacher_margin": "margin",
            "teacher_status": "status",
            "teacher_iterations": "iterations",
            "teacher_seconds": "seconds",
            "success": "success",
        }
        extension = {
            key: np.stack([label[value] for label in labels]) for key, value in mapping.items()
        }
        extension.update(
            {
                "q_ref": archive["q_ref"][rows],
                "J_ref": archive["J_ref"][rows],
                "target": targets,
                "target_rotation": rotations,
                "split": archive["split"][rows],
                "episode_id": np.arange(168, 168 + len(rows)),
                "included": extension["success"].all(-1),
            }
        )
        extension["delta_q"] = extension["q"] - extension["q_ref"][:, None]
        for key, value in extension.items():
            values[key] = np.concatenate((values[key], value))
        metadata["augmentation"] = {
            "seed": 20261031,
            "kind": "independent face/edge poses",
            "episodes": len(rows),
            "teacher": "minimum-distance SLSQP",
        }
        values["metadata"] = json.dumps(metadata)
        destination = study / f"side{side}" / "boundary_motions.npz"
        np.savez_compressed(destination, **values)
        archive.close()
        print("DATA", side, flush=True)
