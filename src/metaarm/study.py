"""Measured corpus diagnostics and validation-only model/optimization comparisons."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from metaarm.basis import MotionBasis
from metaarm.motion_data import load_motions
from metaarm.training import evaluate, train


def diagnose(records, baseline):
    """Separate corpus scale, teacher drift, oracle bounds, and historical overfitting."""
    checkpoint = torch.load(baseline, weights_only=True)
    original = {"width": 64, "scale": 1.0, "normal_features": False, "equivariant": False}
    network = MotionBasis(records["joint_limits"], original).to("cuda:0")
    network.load_state_dict(checkpoint["state_dict"])
    result = {
        "archives": len(records["sources"]),
        "references": len(records["q_ref"]),
        "sources": [
            {
                k: source[k]
                for k in ("path", "sha256", "references", "planar_included", "planar_excluded")
            }
            for source in records["sources"]
        ],
        "historical_final": checkpoint["final"],
        "historical_best_epoch": 1
        + int(np.argmin([e["validation"]["sum"] for e in checkpoint["history"]])),
        "old_model_on_corpus": evaluate(network, records, ("train", "validation", "test")),
        "tasks": {},
    }
    for task in ("pose", "plane"):
        data = records[task]
        result["tasks"][task] = {}
        dimension = 6 if task == "pose" else 3
        for name, rows in data["rows"].items():
            covariance = data["covariance"][rows]
            weights = data["samples"][rows]
            values = torch.linalg.eigvalsh(covariance)
            oracle = (values[:, : 7 - dimension].sum(-1) * weights).sum() / (7 * weights.sum())
            mean = (covariance * weights[:, None, None]).sum(0) / weights.sum()
            constant = torch.linalg.eigvalsh(mean)[: 7 - dimension].sum() / 7
            result["tasks"][task][name] = {
                "episodes": len(rows),
                "samples": int(weights.sum()),
                "episode_oracle_mse": oracle.item(),
                "constant_subspace_mse": constant.item(),
                "zero_mse": (mean.trace() / 7).item(),
                "endpoint_norm_quantiles": torch.quantile(
                    data["endpoint"][rows],
                    torch.tensor([0.5, 0.9, 0.99, 1.0], device="cuda:0", dtype=torch.float64),
                ).tolist(),
            }
    return result


def study(source, destination):
    """Run controlled trials on one immutable corpus and test only the selected model."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records = load_motions(source)
    root = Path(__file__).resolve().parents[2]
    diagnosis = diagnose(records, root / "checkpoints" / "basis_20261031T095508626199Z.pt")
    (destination / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    base = {"width": 64, "scale": 1.0, "normal_features": False, "equivariant": False}
    settings = {"epochs": 300, "learning_rate": 1e-3, "weight_decay": 1e-4}
    trials = [
        ("corpus_baseline", base, settings),
        ("scale8", {**base, "scale": 8.0}, settings),
        ("scale32", {**base, "scale": 32.0}, settings),
        (
            "normal128",
            {**base, "width": 128, "scale": 32.0, "normal_features": True},
            {**settings, "epochs": 400},
        ),
        (
            "yaw128",
            {**base, "width": 128, "scale": 32.0, "normal_features": True, "equivariant": True},
            {**settings, "epochs": 400},
        ),
        (
            "regularized256",
            {**base, "width": 256, "scale": 32.0, "normal_features": True, "equivariant": True},
            {"epochs": 600, "learning_rate": 2e-3, "weight_decay": 1e-3},
        ),
    ]
    trials.extend(
        [
            ("scale128", {**base, "scale": 128.0}, settings),
            ("scale512", {**base, "scale": 512.0}, settings),
            (
                "slow64",
                {**base, "scale": 64.0},
                {"epochs": 800, "learning_rate": 3e-4, "weight_decay": 3e-3},
            ),
            (
                "small32",
                {**base, "width": 32, "scale": 32.0},
                {"epochs": 600, "learning_rate": 1e-3, "weight_decay": 1e-2},
            ),
        ]
    )
    results = []
    for name, model, training in trials:
        print(f"TRIAL {name}: {model}; {training}", flush=True)
        path = destination / f"{name}.pt"
        checkpoint = train(source, path, {"model": model, "training": training})
        if checkpoint["metadata"]["corpus_sha256"] != records["sha256"]:
            raise ValueError("The corpus changed during the comparison.")
        results.append(
            {
                "name": name,
                "checkpoint": str(path.resolve()),
                "specification": checkpoint["metadata"]["specification"],
                "selected_epoch": checkpoint["metadata"]["selected_epoch"],
                "fit_seconds": checkpoint["metadata"]["fit_seconds"],
                "metrics": checkpoint["final"],
            }
        )
        (destination / "trials.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"RESULT {name}: validation={checkpoint['final']['validation']['sum']:.8g}", flush=True
        )

    # Trial selection uses only validation; the common test groups remain outside fitting.
    winner = min(results, key=lambda item: item["metrics"]["validation"]["sum"])
    checkpoint = torch.load(winner["checkpoint"], weights_only=True)
    network = MotionBasis(records["joint_limits"], winner["specification"]["model"]).to("cuda:0")
    network.load_state_dict(checkpoint["state_dict"])
    checkpoint["final"] = evaluate(network, records, ("train", "validation", "test"))
    conditioning = {}
    for task in ("pose", "plane"):
        data = records[task]
        conditioning[task] = {}
        for name, rows in data["rows"].items():
            reference = data["reference"][rows]
            normal = None if task == "pose" else data["normal"][rows]
            geometry = tuple(value[reference] for value in records["geometry"][task])
            b = network(records["q_ref"][reference], geometry, task, normal).detach()
            dimension = b.shape[-1]
            conditioning[task][name] = (
                torch.linalg.svdvals(records["J_ref"][reference, :dimension] @ b)[..., -1]
                .min()
                .item()
            )
    checkpoint["metadata"]["study_selection"] = winner["name"]
    checkpoint["metadata"]["test_evaluated_after_selection"] = True
    checkpoint["conditioning"] = conditioning
    output = (destination / "selected.pt").open("xb")
    torch.save(checkpoint, output)
    output.close()
    report = {
        "winner": winner,
        "final": checkpoint["final"],
        "conditioning": conditioning,
        "archives": len(records["sources"]),
        "corpus_sha256": records["sha256"],
        "checkpoint": str((destination / "selected.pt").resolve()),
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Supply the explicit corpus directory.")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(__file__).resolve().parents[2] / "results" / f"study_{timestamp}"
    study(Path(sys.argv[1]), output)
