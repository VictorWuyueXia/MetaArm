"""Matched position/pose chart validation across regional trajectory and lattice targets."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from metaarm.basis import reference_geometry
from metaarm.gpu_ik import ANGULAR_TOLERANCE, POSITION_TOLERANCE, CoefficientIK
from metaarm.scene import build_scene


def validate_experiment(source, checkpoint, split, destination):
    """Compare learned, geometric, and full-joint searches on the same scoped task targets."""
    if split not in ("validation", "test"):
        raise ValueError("Evaluation requires an explicit validation or test split.")
    source, checkpoint, destination = Path(source), Path(checkpoint), Path(destination)
    saved = torch.load(checkpoint, weights_only=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != saved["metadata"]["sources"][0]["sha256"]:
        raise ValueError("The experiment checkpoint and dataset differ.")
    archive = np.load(source, allow_pickle=False)
    metadata = json.loads(archive["metadata"].item())
    if metadata["kind"] not in ("workspace_position_data", "workspace_pose_data"):
        raise ValueError("Unknown regional task schema.")
    task = "position" if metadata["kind"] == "workspace_position_data" else "pose"
    dimension = 3 if task == "position" else 6
    rows = np.flatnonzero(archive["split"] == split)
    reference, targets = archive["q_ref"][rows], archive["target"][rows]
    included = archive["included"][rows]
    delta = archive["delta_q"][rows][included].astype(np.float64)
    covariance = torch.tensor(
        np.swapaxes(delta, -1, -2) @ delta / delta.shape[1], device="cuda:0", dtype=torch.float64
    )
    teacher_q = archive["q"][rows]
    scopes = {"trajectory": (reference, targets)}
    rotation_scopes = {}
    if task == "pose":
        rotation_scopes["trajectory"] = archive["target_rotation"][rows]
    if split == "test":
        anchors = np.concatenate(
            (
                archive["q_center"][None],
                reference[np.linspace(0, len(reference) - 1, 8).astype(int)],
            )
        )
        box = np.asarray(metadata["box"])
        grid = (
            np.array(np.meshgrid(*(np.linspace(a, b, 9) for a, b in box.T), indexing="ij"))
            .reshape(3, -1)
            .T
        )
        scopes["grid"] = (
            np.repeat(anchors, len(grid), axis=0),
            np.tile(grid, (len(anchors), 1))[:, None],
        )
        if task == "pose":
            values = np.linspace(-np.pi / 18, np.pi / 18, 3)
            angles = np.array(np.meshgrid(values, values, values, indexing="ij")).reshape(3, -1).T
            order = np.random.default_rng(905).permutation(np.tile(np.arange(27), 27))
            rotations = archive["rotation_center"] @ Rotation.from_rotvec(angles[order]).as_matrix()
            rotation_scopes["grid"] = np.tile(rotations, (len(anchors), 1, 1))[:, None]
    archive.close()
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "task": task,
        "split": split,
        "box": metadata["box"],
        "reconstruction": {},
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    records, model = {}, build_scene()
    for scope, (reference, targets) in scopes.items():
        count, steps = targets.shape[:2]
        solver = CoefficientIK(model, count, checkpoint)
        q = torch.tensor(reference, device="cuda:0", dtype=torch.float32)
        desired = torch.tensor(targets, device="cuda:0", dtype=torch.float64)
        _, reference_rotation, jacobian = solver.geometry(q)
        geometry = reference_geometry(jacobian.float(), task)
        basis = solver.network(q, geometry).detach().double()
        geometric = torch.linalg.qr(geometry[1].double(), mode="reduced")[0]
        rotations = (
            np.repeat(reference_rotation.cpu().numpy()[:, None], steps, axis=1)
            if task == "position"
            else rotation_scopes[scope]
        )
        desired_rotation = torch.tensor(rotations, device="cuda:0", dtype=torch.float64)
        if scope == "trajectory":
            r = torch.tensor(np.flatnonzero(included), device="cuda:0")
            report["reconstruction"]["learned"] = solver.network.reconstruction(
                q[r], tuple(v[r] for v in geometry), covariance
            ).item()
            residual = torch.eye(7, device="cuda:0") - geometric[r] @ geometric[r].mT
            report["reconstruction"]["geometric"] = (
                (residual @ covariance @ residual.mT).diagonal(dim1=-2, dim2=-1).sum(-1).mean() / 7
            ).item()
        report[scope] = {}
        records[f"{scope}_reference"], records[f"{scope}_target"] = reference, targets
        for method, b in (
            ("learned", basis),
            ("geometric", geometric),
            ("full", torch.eye(7, device="cuda:0").double().expand(count, -1, -1)),
        ):
            k = b.shape[-1]
            alpha = torch.zeros((count, k), device="cuda:0", dtype=torch.float64)
            warm = 1 if scope == "trajectory" else 0
            solver.solve(
                q,
                b,
                {"position": desired[:, warm], "rotation": desired_rotation[:, warm]},
                task,
                alpha,
            )
            alive = torch.ones(count, device="cuda:0", dtype=torch.bool)
            trace = {
                key: np.full((count, steps, *tail), np.nan)
                for key, tail in (
                    ("q", (7,)),
                    ("alpha", (k,)),
                    ("position_error", ()),
                    ("angular_error", ()),
                    ("joint_margin", ()),
                    ("status", ()),
                    ("iterations", ()),
                )
            }
            trace["attempted"], trace["success"] = (
                np.zeros((count, steps), bool),
                np.zeros((count, steps), bool),
            )
            seconds = []
            for waypoint in range(steps):
                if not alive.any():
                    break
                attempted = alive.cpu().numpy()
                p, r, _ = solver.geometry(q.double() + (b @ alpha[..., None]).squeeze(-1))
                target = {
                    "position": torch.where(alive[:, None], desired[:, waypoint], p),
                    "rotation": torch.where(alive[:, None, None], desired_rotation[:, waypoint], r),
                }
                result = solver.solve(q, b, target, task, alpha)
                trace["attempted"][:, waypoint] = attempted
                for key in trace:
                    if key != "attempted":
                        trace[key][attempted, waypoint] = result[key].cpu().numpy()[attempted]
                seconds.append(result["seconds"])
                alpha, alive = result["alpha"], alive & result["success"]

            # CPU FK audits both task feasibility and the sampled frozen-chart mobility.
            cpu, jp, jr = mujoco.MjData(model), np.empty((3, model.nv)), np.empty((3, model.nv))
            for key in ("cpu_error", "rotation_discrepancy", "sigma_min"):
                trace[key] = np.full((count, steps), np.nan)
            fixed = b.cpu().numpy()
            for episode, waypoint in np.argwhere(trace["attempted"]):
                cpu.qpos[:] = trace["q"][episode, waypoint]
                mujoco.mj_forward(model, cpu)
                trace["cpu_error"][episode, waypoint] = np.linalg.norm(
                    cpu.site_xpos[solver.site] - targets[episode, waypoint]
                )
                relative = rotations[episode, waypoint].T @ cpu.site_xmat[solver.site].reshape(3, 3)
                trace["rotation_discrepancy"][episode, waypoint] = Rotation.from_matrix(
                    relative
                ).magnitude()
                mujoco.mj_jacSite(model, cpu, jp, jr, solver.site)
                trace["sigma_min"][episode, waypoint] = np.linalg.svd(
                    np.vstack((jp, jr))[:dimension] @ fixed[episode], compute_uv=False
                )[-1]
            attempted = trace["attempted"]
            difference = np.max(
                np.abs(trace["cpu_error"][attempted] - trace["position_error"][attempted])
            )
            audited = (
                attempted
                & (trace["cpu_error"] <= POSITION_TOLERANCE)
                & (trace["joint_margin"] >= 0)
            )
            if task == "pose":
                audited &= trace["rotation_discrepancy"] <= ANGULAR_TOLERANCE
                difference = max(
                    difference,
                    np.max(
                        np.abs(
                            trace["rotation_discrepancy"][attempted]
                            - trace["angular_error"][attempted]
                        )
                    ),
                )
            if difference > 3e-6:
                raise FloatingPointError("Independent task residual audits disagree.")
            metrics = {
                "episodes": count,
                "planned": count * steps,
                "attempted": int(attempted.sum()),
                "successful": int(trace["success"].sum()),
                "completed": int(trace["success"].all(axis=1).sum()),
                "cpu_disagreements": int(np.count_nonzero(audited != trace["success"])),
                "position_error_quantiles": np.quantile(
                    trace["cpu_error"][attempted], [0.5, 0.95, 1]
                ).tolist(),
                "rotation_error_quantiles": np.quantile(
                    trace["rotation_discrepancy"][attempted], [0.5, 0.95, 1]
                ).tolist(),
                "minimum_joint_margin": float(np.min(trace["joint_margin"][attempted])),
                "minimum_sampled_sigma": float(np.min(trace["sigma_min"][attempted])),
                "mean_iterations": float(np.mean(trace["iterations"][attempted])),
                "batch_seconds": float(sum(seconds)),
                "cpu_gpu_max_difference": float(difference),
            }
            if scope == "trajectory":
                label_rows = attempted & included[:, None]
                metrics["joint_solution_mse"] = float(
                    np.square(trace["q"][label_rows] - teacher_q[label_rows]).mean()
                )
            report[scope][method] = metrics
            records.update({f"{scope}_{method}_{key}": v for key, v in trace.items()})
            records[f"{scope}_{method}_basis"] = fixed
            records[f"{scope}_{method}_batch_seconds"] = np.asarray(seconds)
            print(
                f"{split}/{scope}/{task}/{method}: {metrics['completed']}/{count} completed.",
                flush=True,
            )
    np.savez_compressed(destination / "records.npz", **records)
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
