"""Independent adaptive workspace studies and repeated paired GPU search measurements."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from metaarm.basis import reference_geometry
from metaarm.expansion_data import collect_experiment
from metaarm.expansion_evaluation import validate_experiment
from metaarm.gpu_ik import ANGULAR_TOLERANCE, POSITION_TOLERANCE, CoefficientIK
from metaarm.scene import build_scene
from metaarm.workspace import train_workspace

BATCH_CASES, LATENCY_CASES, BATCH_REPEATS, LATENCY_REPEATS = 256, 16, 5, 3


def benchmark_search(source, checkpoint, destination):
    """Measure repeated matched searches, with basis preparation and failures kept visible."""
    enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    archive = np.load(source, allow_pickle=False)
    metadata = json.loads(archive["metadata"].item())
    task, k = metadata["task"], 3 if metadata["task"] == "position" else 6
    feasible = (archive["teacher_error"] < 1e-7) & (archive["teacher_margin"] > 0)
    if task == "pose":
        feasible &= archive["teacher_angular_error"] < 1e-7
    feasible &= archive["split"][:, None] == "test"
    feasible &= np.linalg.norm(archive["target"] - archive["target"][:, :1], axis=-1) > 0.005
    candidates = np.argwhere(feasible)
    indices = np.random.default_rng(958).choice(len(candidates), BATCH_CASES, replace=False)
    cases = candidates[indices]
    reference = archive["q_ref"][cases[:, 0]]
    positions = archive["target"][cases[:, 0], cases[:, 1]]
    rotations = archive["target_rotation"][cases[:, 0], cases[:, 1]]
    archive.close()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records = {"case_identity": cases}
    report = {
        "task": task,
        "coordinates": k,
        "coordinate_reduction_percent": 100 * (1 - k / 7),
        "coupled_inequalities": 14,
        "dense_system_entries": k * k,
        "dense_system_entry_reduction_percent": 100 * (1 - k * k / 49),
        "modes": {},
    }
    model = build_scene()
    for mode in ("batch", "single"):
        count = BATCH_CASES if mode == "batch" else LATENCY_CASES
        batch = count if mode == "batch" else 1
        repeats = BATCH_REPEATS if mode == "batch" else LATENCY_REPEATS
        solver = CoefficientIK(model, batch, checkpoint)
        q = torch.tensor(reference[:count], device="cuda:0", dtype=torch.float32)
        p = torch.tensor(positions[:count], device="cuda:0", dtype=torch.float64)
        r = torch.tensor(rotations[:count], device="cuda:0", dtype=torch.float64)
        blocks = 1 if mode == "batch" else count
        times = np.empty((repeats, 2, blocks))
        prep = np.empty_like(times)
        success = np.empty((2, count), bool)
        iterations = np.empty((2, count))
        solutions = np.empty((2, count, 7))
        for repeat in range(-1, repeats):
            for index in range(blocks):
                selection = slice(None) if mode == "batch" else slice(index, index + 1)
                target = {"position": p[selection], "rotation": r[selection]}
                for method in (0, 1) if repeat % 2 == 0 else (1, 0):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    if method == 0:
                        _, _, jac = solver.geometry(q[selection])
                        b = (
                            solver.network(q[selection], reference_geometry(jac.float(), task))
                            .detach()
                            .double()
                        )
                    else:
                        b = torch.eye(7, device="cuda:0").double().expand(batch, -1, -1)
                    torch.cuda.synchronize()
                    preparation = time.perf_counter() - started
                    result = solver.solve(
                        q[selection],
                        b,
                        target,
                        task,
                        torch.zeros((batch, b.shape[-1]), device="cuda:0"),
                    )
                    if repeat >= 0:
                        times[repeat, method, index] = result["seconds"]
                        prep[repeat, method, index] = preparation
                        success[method, selection] = result["success"].cpu().numpy()
                        iterations[method, selection] = result["iterations"].cpu().numpy()
                        solutions[method, selection] = result["q"].cpu().numpy()

        # Repeated single-case times support a genuine jointly-successful paired subset.
        cpu = mujoco.MjData(model)
        disagreements = 0
        for method in (0, 1):
            for index in range(count):
                cpu.qpos[:] = solutions[method, index]
                mujoco.mj_forward(model, cpu)
                ep = np.linalg.norm(cpu.site_xpos[solver.site] - positions[index])
                er = Rotation.from_matrix(
                    rotations[index].T @ cpu.site_xmat[solver.site].reshape(3, 3)
                ).magnitude()
                audited = ep <= POSITION_TOLERANCE and (
                    task == "position" or er <= ANGULAR_TOLERANCE
                )
                disagreements += int(audited != success[method, index])
        paired = success.all(axis=0)
        values = {
            "cases": count,
            "repeats": repeats,
            "successes": success.sum(axis=1).tolist(),
            "joint_success_cases": int(paired.sum()),
            "cpu_disagreements": disagreements,
            "search_median_seconds": np.median(times, axis=(0, 2)).tolist(),
            "search_quartiles_seconds": np.quantile(times, [0.25, 0.75], axis=(0, 2)).T.tolist(),
            "preparation_median_seconds": np.median(prep, axis=(0, 2)).tolist(),
            "inclusive_median_seconds": np.median(times + prep, axis=(0, 2)).tolist(),
            "mean_iterations": iterations.mean(axis=1).tolist(),
        }
        values["search_speedup"] = (
            values["search_median_seconds"][1] / values["search_median_seconds"][0]
        )
        values["inclusive_speedup"] = (
            values["inclusive_median_seconds"][1] / values["inclusive_median_seconds"][0]
        )
        values["search_time_reduction_percent"] = 100 * (1 - 1 / values["search_speedup"])
        if mode == "single" and paired.any():
            shared = np.median(times[:, :, paired], axis=(0, 2))
            values["shared_success_speedup"] = float(shared[1] / shared[0])
        report["modes"][mode] = values
        records.update(
            {
                f"{mode}_seconds": times,
                f"{mode}_preparation": prep,
                f"{mode}_success": success,
                f"{mode}_iterations": iterations,
                f"{mode}_q": solutions,
            }
        )
    np.savez_compressed(destination / "timings.npz", **records)
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.set_grad_enabled(enabled)
    return report


def run_expansion(destination):
    """Tune each size/task independently, freeze its winner, and evaluate its held-out cases."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    summary = []
    for side in (0.16, 0.24, 0.32, 0.40, 0.48):
        for task in ("position", "pose"):
            folder = destination / f"side{round(side * 100):02d}_{task}"
            folder.mkdir()
            source = folder / "motions.npz"
            metadata = collect_experiment(source, side, task)
            base = {
                "depth": 2,
                "epochs": 500,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "scheme": "reconstruction",
            }
            recipes = [
                ({"width": 32, "scale": 0.25}, base),
                ({"width": 64, "scale": 1.0}, base),
                ({"width": 128, "scale": 4.0}, base),
                ({"width": 64, "scale": 1.0}, {**base, "depth": 3}),
            ]
            trials = []
            for index in range(6):
                if index >= 4:
                    winner = min(
                        trials,
                        key=lambda t: (
                            -t["validation"]["trajectory"]["learned"]["completed"],
                            -t["validation"]["trajectory"]["learned"]["successful"],
                            t["validation"]["reconstruction"]["learned"],
                        ),
                    )
                    model = dict(winner["model"])
                    settings = {**winner["training"], "epochs": 1000, "learning_rate": 3e-4}
                    if index == 4:
                        settings["weight_decay"] = 1e-3
                    else:
                        settings.update({"depth": 3, "scheme": "mobility"})
                else:
                    model, settings = recipes[index]
                path = folder / f"trial{index + 1}.pt"
                fitted = train_workspace(source, path, model, settings)
                validation = validate_experiment(
                    source, path, "validation", folder / f"validation{index + 1}"
                )
                trials.append(
                    {
                        "checkpoint": str(path),
                        "model": model,
                        "training": settings,
                        "selected_epoch": fitted["metadata"]["selected_epoch"],
                        "validation": validation,
                    }
                )
                (folder / "trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")
            winner = min(
                trials,
                key=lambda t: (
                    -t["validation"]["trajectory"]["learned"]["completed"],
                    -t["validation"]["trajectory"]["learned"]["successful"],
                    t["validation"]["reconstruction"]["learned"],
                ),
            )
            selected = folder / "selected.pt"
            output = selected.open("xb")
            torch.save(torch.load(winner["checkpoint"], weights_only=True), output)
            output.close()
            final = validate_experiment(source, selected, "test", folder / "test")
            timing = benchmark_search(source, selected, folder / "timing")
            result = {
                "side": side,
                "task": task,
                "data": metadata,
                "winner": winner,
                "test": final,
                "timing": timing,
                "checkpoint": str(selected),
            }
            summary.append(result)
            (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            (destination / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(
                f"COMPLETE {side:.2f}m/{task}: "
                f"{final['trajectory']['learned']['completed']}/48 paths; "
                f"single-target speedup {timing['modes']['single']['search_speedup']:.3f}x.",
                flush=True,
            )
    return summary


if __name__ == "__main__":
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_expansion(Path(__file__).resolve().parents[2] / "results" / f"expansion_{timestamp}")
