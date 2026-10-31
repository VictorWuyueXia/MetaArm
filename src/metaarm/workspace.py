"""One-command collection, fitting, and independent validation of a regional position basis."""

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from metaarm.basis import TaskBasis, reference_geometry
from metaarm.expansion_evaluation import validate_experiment
from metaarm.workspace_data import collect_workspace


def train_workspace(source, destination, specification, training):
    """Fit retained regional waypoints and select the declared validation objective."""
    if set(training) != {"depth", "epochs", "learning_rate", "weight_decay", "scheme"}:
        raise ValueError("Regional training has exactly five declared experiment fields.")
    if training["scheme"] not in ("reconstruction", "mobility"):
        raise ValueError("Unknown regional training objective.")
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    archive = np.load(source, allow_pickle=False)
    metadata = json.loads(archive["metadata"].item())
    if metadata["kind"] not in ("workspace_position_data", "workspace_pose_data"):
        raise ValueError("The regional trainer requires the declared position-workspace schema.")
    task = "position" if metadata["kind"] == "workspace_position_data" else "pose"
    dimension = 3 if task == "position" else 6
    root = Path(__file__).resolve().parents[2]
    for name, digest in metadata["assets"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"The regional physical asset differs: {name}.")
    included = archive["included"]
    q = torch.tensor(archive["q_ref"][included], device="cuda:0", dtype=torch.float32)
    j = torch.tensor(archive["J_ref"][included], device="cuda:0", dtype=torch.float32)
    delta = archive["delta_q"][included].astype(np.float64)
    covariance = torch.tensor(
        np.swapaxes(delta, -1, -2) @ delta / delta.shape[1], device="cuda:0", dtype=torch.float64
    )
    split = archive["split"][included]
    if training["scheme"] == "mobility":
        sample_jacobian = torch.tensor(
            archive["J"][included, :, :dimension], device="cuda:0", dtype=torch.float64
        )
        threshold = 0.1 if task == "position" else 0.03
    archive.close()
    rows = {
        name: torch.tensor(np.flatnonzero(split == name), device="cuda:0")
        for name in ("train", "validation")
    }
    if any(len(value) == 0 for value in rows.values()):
        raise ValueError("Regional fitting requires complete training and validation episodes.")
    geometry = reference_geometry(j, task)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    network = TaskBasis(metadata["joint_limits"], specification, task, training["depth"]).cuda()
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, training["epochs"], eta_min=training["learning_rate"] * 0.02
    )
    initial = {
        name: network.reconstruction(q[r], tuple(v[r] for v in geometry), covariance[r]).item()
        for name, r in rows.items()
    }
    initial_objective = dict(initial)
    if training["scheme"] == "mobility":
        for name, r in rows.items():
            b = network(q[r], tuple(v[r] for v in geometry)).double()
            singular = torch.linalg.svdvals(sample_jacobian[r] @ b[:, None])[..., -1]
            initial_objective[name] += (
                1e-3 * torch.relu(1 - singular / threshold).square().mean()
            ).item()
    best, epoch_best = initial_objective["validation"], 0
    best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
    history = []
    torch.cuda.synchronize()
    started = time.perf_counter()

    # One full covariance batch retains every waypoint and its original multiplicity.
    for epoch in range(training["epochs"]):
        r = rows["train"]
        optimizer.zero_grad(set_to_none=True)
        loss = network.reconstruction(q[r], tuple(v[r] for v in geometry), covariance[r])
        if training["scheme"] == "mobility":
            b = network(q[r], tuple(v[r] for v in geometry)).double()
            singular = torch.linalg.svdvals(sample_jacobian[r] @ b[:, None])[..., -1]
            loss = loss + 1e-3 * torch.relu(1 - singular / threshold).square().mean()
        loss.backward()
        if not torch.isfinite(torch.cat([v.grad.flatten() for v in network.parameters()])).all():
            raise FloatingPointError("A regional training gradient is nonfinite.")
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == training["epochs"]:
            metrics = {
                name: network.reconstruction(
                    q[r], tuple(v[r] for v in geometry), covariance[r]
                ).item()
                for name, r in rows.items()
            }
            objective = dict(metrics)
            if training["scheme"] == "mobility":
                for name, r in rows.items():
                    b = network(q[r], tuple(v[r] for v in geometry)).double()
                    singular = torch.linalg.svdvals(sample_jacobian[r] @ b[:, None])[..., -1]
                    objective[name] += (
                        1e-3 * torch.relu(1 - singular / threshold).square().mean()
                    ).item()
            history.append({"epoch": epoch + 1, "metrics": metrics, "objective": objective})
            if objective["validation"] < best:
                best, epoch_best = objective["validation"], epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    last_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
    network.load_state_dict(best_state)
    final = {
        name: network.reconstruction(q[r], tuple(v[r] for v in geometry), covariance[r]).item()
        for name, r in rows.items()
    }
    result = {
        "state_dict": best_state,
        "last_state_dict": last_state,
        "initial": initial,
        "history": history,
        "final": final,
        "metadata": {
            "task": task,
            "selected_epoch": epoch_best,
            "specification": {
                "model": specification,
                "training": training,
            },
            "sources": [
                {
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "metadata": metadata,
                }
            ],
            "fit_seconds": elapsed,
            "parameter_count": sum(v.numel() for v in network.parameters()),
            "gpu": torch.cuda.get_device_name(0),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "src/metaarm").glob("*.py"))
            },
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = destination.open("xb")
    torch.save(result, output)
    output.close()
    print(
        f"Fit {specification}: validation objective {best:.8g}, selected epoch {epoch_best}.",
        flush=True,
    )
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / "results" / f"workspace_{timestamp}"
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / "motions.npz"
    collect_workspace(source)
    trials = []
    for width, scale in ((32, 0.25), (32, 1.0), (32, 4.0), (64, 1.0)):
        name = f"width{width}_scale{scale:g}"
        path = destination / f"{name}.pt"
        checkpoint = train_workspace(
            source,
            path,
            {"width": width, "scale": scale},
            {
                "depth": 2,
                "epochs": 500,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "scheme": "reconstruction",
            },
        )
        validation = validate_experiment(source, path, "validation", destination / name)
        trials.append(
            {
                "name": name,
                "checkpoint": str(path),
                "specification": {"width": width, "scale": scale},
                "validation": validation,
                "selected_epoch": checkpoint["metadata"]["selected_epoch"],
            }
        )
        (destination / "trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
    winner = min(
        trials,
        key=lambda item: (
            -item["validation"]["trajectory"]["learned"]["completed"],
            item["validation"]["reconstruction"]["learned"],
        ),
    )
    selected = torch.load(winner["checkpoint"], weights_only=True)
    output = (destination / "selected.pt").open("xb")
    torch.save(selected, output)
    output.close()
    final = validate_experiment(source, destination / "selected.pt", "test", destination / "test")
    report = {
        "winner": winner,
        "test": final,
        "source": str(source),
        "checkpoint": str(destination / "selected.pt"),
        "trials": trials,
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed regional pipeline: {destination}", flush=True)
