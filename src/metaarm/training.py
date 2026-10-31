"""Corpus-wide covariance training with validation-selected neural motion bases."""

import hashlib
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from metaarm.basis import MotionBasis
from metaarm.motion_data import load_motions

BATCH_SIZE = 256


def evaluate(network, records, split_names):
    """Compute the exact sample-weighted reconstruction objective from second moments."""
    enabled, training = torch.is_grad_enabled(), network.training
    torch.set_grad_enabled(False)
    network.eval()
    metrics = {name: {} for name in split_names}
    for task in ("pose", "plane"):
        data = records[task]
        for name in split_names:
            rows = data["rows"][name]
            reference = data["reference"][rows]
            normal = None if task == "pose" else data["normal"][rows]
            geometry = tuple(value[reference] for value in records["geometry"][task])
            raw = network(records["q_ref"][reference], geometry, task, normal, False).double()
            factor, info = torch.linalg.cholesky_ex(raw.mT @ raw)
            if torch.any(info != 0):
                raise FloatingPointError("The raw basis Gram matrix is not positive definite.")
            residual = torch.eye(
                7, device=raw.device, dtype=raw.dtype
            ) - raw @ torch.cholesky_solve(raw.mT, factor)
            covariance = data["covariance"][rows]
            errors = (residual @ covariance @ residual.mT).diagonal(dim1=-2, dim2=-1).sum(-1) / 7
            weights = data["samples"][rows]
            loss = (errors * weights).sum() / weights.sum()
            energy = (covariance.diagonal(dim1=-2, dim2=-1).sum(-1) * weights).sum() / (
                7 * weights.sum()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite reconstruction metric.")
            metrics[name][task] = loss.item()
            metrics[name][f"{task}_unexplained_fraction"] = (loss / energy).item()
    for values in metrics.values():
        values["sum"] = values["pose"] + values["plane"]
    network.train(training)
    torch.set_grad_enabled(enabled)
    return metrics


def train(source, destination, specification):
    """Fit all retained episodes and preserve selected and final-weight evidence."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    model, settings = specification["model"], specification["training"]
    if set(settings) != {"epochs", "learning_rate", "weight_decay"}:
        raise ValueError("Training has exactly three experiment settings.")
    epochs = settings["epochs"]
    if type(epochs) is not int or epochs < 1:
        raise ValueError("The epoch count must be a positive integer.")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    records = load_motions(source)
    network = MotionBasis(records["joint_limits"], model).to("cuda:0")
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=settings["learning_rate"] * 0.02
    )
    initial = evaluate(network, records, ("train", "validation"))
    best, best_epoch = initial["validation"]["sum"], 0
    best_state = {
        name: value.detach().cpu().clone() for name, value in network.state_dict().items()
    }
    history = []
    steps = (
        max(len(records[t]["rows"]["train"]) for t in ("pose", "plane")) + BATCH_SIZE - 1
    ) // BATCH_SIZE
    torch.cuda.synchronize()
    started = time.perf_counter()

    # Covariance minibatches use every waypoint without repeating its identical basis.
    for epoch in range(epochs):
        batches = {
            task: torch.tensor_split(
                records[task]["rows"]["train"][
                    torch.randperm(len(records[task]["rows"]["train"]), device="cuda:0")
                ],
                steps,
            )
            for task in ("pose", "plane")
        }
        for batch in range(steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for task in ("pose", "plane"):
                data, rows = records[task], batches[task][batch]
                reference = data["reference"][rows]
                normal = None if task == "pose" else data["normal"][rows]
                geometry = tuple(value[reference] for value in records["geometry"][task])
                raw = network(records["q_ref"][reference], geometry, task, normal, False).double()
                factor, info = torch.linalg.cholesky_ex(raw.mT @ raw)
                if torch.any(info != 0):
                    raise FloatingPointError("The raw basis Gram matrix is not positive definite.")
                residual = torch.eye(
                    7, device=raw.device, dtype=raw.dtype
                ) - raw @ torch.cholesky_solve(raw.mT, factor)
                error = (residual @ data["covariance"][rows] @ residual.mT).diagonal(
                    dim1=-2, dim2=-1
                ).sum(-1) / 7
                weights = data["samples"][rows]
                losses.append((error * weights).sum() / weights.sum())
            loss = losses[0] + losses[1]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite reconstruction loss.")
            loss.backward()
            if not torch.isfinite(
                torch.cat([p.grad.flatten() for p in network.parameters()])
            ).all():
                raise FloatingPointError("Non-finite neural gradient.")
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            metrics = evaluate(network, records, ("train", "validation"))
            history.append({"epoch": epoch + 1, "metrics": metrics})
            if metrics["validation"]["sum"] < best:
                best, best_epoch = metrics["validation"]["sum"], epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}: validation={metrics['validation']['sum']:.7g}; "
                    f"best={best:.7g}",
                    flush=True,
                )

    # Select on validation only, retaining the final weights and original archive manifest.
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    last_state = {
        name: value.detach().cpu().clone() for name, value in network.state_dict().items()
    }
    last = evaluate(network, records, ("train", "validation"))
    network.load_state_dict(best_state)
    final = evaluate(network, records, ("train", "validation"))
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "specification": specification,
        "selected_epoch": best_epoch,
        "selection": "minimum validation task-MSE sum",
        "sources": records["sources"],
        "corpus_sha256": records["sha256"],
        "seed": 0,
        "fit_seconds": elapsed,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "batch_cap_episodes_per_task": BATCH_SIZE,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src" / "metaarm").glob("*.py"))
        },
        "git_revision": subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "task_samples": {
            task: {
                name: int(records[task]["samples"][rows].sum().item())
                for name, rows in records[task]["rows"].items()
            }
            for task in ("pose", "plane")
        },
    }
    result = {
        "state_dict": best_state,
        "last_state_dict": last_state,
        "metadata": metadata,
        "initial": initial,
        "history": history,
        "last": last,
        "final": final,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = destination.open("xb")
    torch.save(result, output)
    output.close()
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Supply one explicit archive or corpus directory.")
    specification = {
        "model": {"width": 64, "scale": 32.0, "normal_features": False, "equivariant": False},
        "training": {"epochs": 300, "learning_rate": 1e-3, "weight_decay": 1e-4},
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(__file__).resolve().parents[2] / "checkpoints" / f"basis_{timestamp}.pt"
    report = train(Path(sys.argv[1]), output, specification)
    print(f"Saved {output}; selected validation loss {report['final']['validation']['sum']:.7g}.")
