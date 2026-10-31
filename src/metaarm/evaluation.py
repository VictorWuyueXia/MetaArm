"""Matched learned/full-joint GPU IK on every archived held-out trajectory."""

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from metaarm.basis import reference_geometry
from metaarm.gpu_ik import ANGULAR_TOLERANCE, CHECKPOINT, POSITION_TOLERANCE, CoefficientIK
from metaarm.motion_data import load_motions
from metaarm.scene import build_scene


def evaluate_ik(source, destination):
    """Compare identical fixed-chart tasks, retain failures, and independently audit candidates."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records = load_motions(Path(source))
    model = build_scene()
    groups = {
        task: {
            key: []
            for key in (
                "q_ref",
                "jacobian",
                "position",
                "rotation",
                "normal",
                "origin",
                "source_id",
                "episode_id",
            )
        }
        for task in ("pose", "plane")
    }
    for source_id, item in enumerate(records["sources"]):
        archive = np.load(item["path"], allow_pickle=False)
        for task, group in groups.items():
            prefix = "general" if task == "pose" else "planar"
            rows = np.flatnonzero(archive[f"{prefix}_split"] == "test")
            refs = rows if task == "pose" else archive["planar_reference_id"][rows]
            group["q_ref"].append(archive["general_q_ref"][refs])
            group["jacobian"].append(archive["general_J_ref"][refs])
            group["position"].append(
                archive["general_p"][rows] if task == "pose" else archive["planar_target"][rows]
            )
            group["rotation"].append(
                archive["general_R"][rows]
                if task == "pose"
                else np.repeat(
                    archive["general_R_ref"][refs, None], archive["planar_target"].shape[1], axis=1
                )
            )
            group["normal"].append(
                np.zeros((len(rows), 3)) if task == "pose" else archive["planar_normal"][rows]
            )
            group["origin"].append(
                archive["general_p_ref"][refs] if task == "pose" else archive["planar_origin"][rows]
            )
            group["source_id"].append(np.full(len(rows), source_id))
            group["episode_id"].append(archive[f"{prefix}_episode_id"][rows])
        archive.close()

    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "archives": len(records["sources"]),
        "corpus_sha256": records["sha256"],
        "sources": records["sources"],
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "position_tolerance": POSITION_TOLERANCE,
        "angular_tolerance": ANGULAR_TOLERANCE,
        "gpu": torch.cuda.get_device_name(0),
        "tasks": {},
        "policy": (
            "Stored test groups; fixed reference/basis; warm-start coefficients; "
            "stop each method at its first failed waypoint."
        ),
    }
    saved = {}
    for task, lists in groups.items():
        group = {key: np.concatenate(value) for key, value in lists.items()}
        count, steps = group["position"].shape[:2]
        solver = CoefficientIK(model, count, CHECKPOINT)
        q_ref = torch.tensor(group["q_ref"], device="cuda:0", dtype=torch.float32)
        jacobian = torch.tensor(group["jacobian"], device="cuda:0", dtype=torch.float32)
        normal = torch.tensor(group["normal"], device="cuda:0", dtype=torch.float32)
        positions = torch.tensor(group["position"], device="cuda:0", dtype=torch.float64)
        rotations = torch.tensor(group["rotation"], device="cuda:0", dtype=torch.float64)
        solver.geometry(q_ref)
        torch.cuda.synchronize()
        started = time.perf_counter()
        learned = (
            solver.network(
                q_ref, reference_geometry(jacobian, task), task, None if task == "pose" else normal
            )
            .detach()
            .double()
        )
        torch.cuda.synchronize()
        preparation = time.perf_counter() - started
        report["tasks"][task] = {
            "episodes": count,
            "waypoints": steps,
            "basis_seconds": preparation,
            "methods": {},
        }
        for key, value in group.items():
            saved[f"{task}_{key}"] = value

        # Each method receives the same episode references and a separate coefficient history.
        for method, basis in (
            ("learned", learned),
            ("full", torch.eye(7, device="cuda:0", dtype=torch.float64).expand(count, -1, -1)),
        ):
            dimension = basis.shape[-1]
            alpha = torch.zeros((count, dimension), device="cuda:0", dtype=torch.float64)
            solver.solve(
                q_ref,
                basis,
                {"position": positions[:, 1], "rotation": rotations[:, 1]},
                task,
                alpha,
            )
            alive = torch.ones(count, device="cuda:0", dtype=torch.bool)
            arrays = {
                "q": np.full((count, steps, 7), np.nan),
                "alpha": np.full((count, steps, dimension), np.nan),
                "attempted": np.zeros((count, steps), bool),
                "success": np.zeros((count, steps), bool),
                "status": np.full((count, steps), -1, np.int32),
                "iterations": np.full((count, steps), -1, np.int32),
            }
            for key in (
                "position_error",
                "angular_error",
                "joint_margin",
                "normal_error",
                "cpu_position_error",
                "cpu_angular_error",
            ):
                arrays[key] = np.full((count, steps), np.nan)
            batch_seconds = []
            for waypoint in range(steps):
                if not alive.any():
                    break
                attempted = alive.cpu().numpy()
                current = q_ref.double() + (basis @ alpha[..., None]).squeeze(-1)
                p, r, _ = solver.geometry(current)
                target = {
                    "position": torch.where(alive[:, None], positions[:, waypoint], p),
                    "rotation": torch.where(alive[:, None, None], rotations[:, waypoint], r),
                }
                result = solver.solve(q_ref, basis, target, task, alpha)
                batch_seconds.append(result["seconds"])
                arrays["attempted"][:, waypoint] = attempted
                for key in (
                    "q",
                    "alpha",
                    "success",
                    "status",
                    "iterations",
                    "position_error",
                    "angular_error",
                    "joint_margin",
                ):
                    arrays[key][attempted, waypoint] = result[key].cpu().numpy()[attempted]
                alpha = result["alpha"]
                alive &= result["success"]
                if waypoint % 16 == 0:
                    print(
                        f"{task}/{method}: waypoint {waypoint + 1}/{steps}, "
                        f"continuing {alive.sum().item()}/{count}",
                        flush=True,
                    )

            # CPU MuJoCo audits the actual attempted configurations rather than teacher labels.
            cpu = mujoco.MjData(model)
            for episode, waypoint in np.argwhere(arrays["attempted"]):
                cpu.qpos[solver.q_slice] = arrays["q"][episode, waypoint]
                mujoco.mj_forward(model, cpu)
                displacement = cpu.site_xpos[solver.site] - group["position"][episode, waypoint]
                arrays["cpu_position_error"][episode, waypoint] = np.linalg.norm(displacement)
                arrays["cpu_angular_error"][episode, waypoint] = (
                    0
                    if task == "plane"
                    else Rotation.from_matrix(
                        group["rotation"][episode, waypoint].T
                        @ cpu.site_xmat[solver.site].reshape(3, 3)
                    ).magnitude()
                )
                arrays["normal_error"][episode, waypoint] = (
                    0
                    if task == "pose"
                    else abs(
                        group["normal"][episode]
                        @ (cpu.site_xpos[solver.site] - group["origin"][episode])
                    )
                )
            attempted = arrays["attempted"]
            audited = (
                attempted
                & (arrays["cpu_position_error"] <= POSITION_TOLERANCE)
                & (arrays["cpu_angular_error"] <= ANGULAR_TOLERANCE)
                & (arrays["joint_margin"] >= 0)
            )
            differences = np.stack(
                [
                    arrays["cpu_position_error"] - arrays["position_error"],
                    arrays["cpu_angular_error"] - arrays["angular_error"],
                ]
            )
            difference = np.max(np.abs(differences[:, attempted]))
            if difference > 3e-6:
                raise FloatingPointError("Independent CPU and GPU task errors disagree.")
            summary = {
                "dimensions": dimension,
                "planned": count * steps,
                "attempted": int(attempted.sum()),
                "successful": int(arrays["success"].sum()),
                "completed_episodes": int(arrays["success"].all(axis=1).sum()),
                "audited_successful": int(audited.sum()),
                "cpu_success_disagreements": int(np.count_nonzero(audited != arrays["success"])),
                "solver_batch_seconds": float(sum(batch_seconds)),
                "mean_batch_seconds": float(np.mean(batch_seconds)),
                "amortized_seconds_per_attempt": float(sum(batch_seconds) / attempted.sum()),
                "cpu_gpu_error_max_difference": float(difference),
                "position_error_quantiles": np.quantile(
                    arrays["cpu_position_error"][attempted], [0.5, 0.95, 1]
                ).tolist(),
                "angular_error_quantiles": np.quantile(
                    arrays["cpu_angular_error"][attempted], [0.5, 0.95, 1]
                ).tolist(),
                "normal_error_max": float(np.max(arrays["normal_error"][attempted])),
                "minimum_joint_margin": float(np.min(arrays["joint_margin"][attempted])),
                "status_counts": {
                    str(k): int(np.count_nonzero(arrays["status"][attempted] == k))
                    for k in (0, 1, 2)
                },
            }
            report["tasks"][task]["methods"][method] = summary
            saved.update({f"{task}_{method}_{key}": value for key, value in arrays.items()})
            saved[f"{task}_{method}_batch_seconds"] = np.asarray(batch_seconds)
            print(json.dumps({task: {method: summary}}), flush=True)

    report["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("basis.py", "gpu_ik.py", "evaluation.py")
    }
    np.savez_compressed(destination / "trajectories.npz", **saved)
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {destination.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Supply one explicit corpus directory.")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(__file__).resolve().parents[2] / "results" / f"ik_{timestamp}"
    evaluate_ik(Path(sys.argv[1]), output)
