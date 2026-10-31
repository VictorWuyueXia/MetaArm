"""Reproducible regional pose-chart comparisons and analytical rank-loss witnesses."""

import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from metaarm.basis import reference_geometry
from metaarm.gpu_ik import CoefficientIK
from metaarm.pose_model import PoseChart, pose_geometry
from metaarm.pose_solver import solve_pose
from metaarm.scene import build_scene


@torch.no_grad()
def evaluate_pose(source, checkpoint, destination):
    """Compare identical independent pose requests, then time matched feasible queries."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    archive = np.load(source, allow_pickle=False)
    saved = torch.load(checkpoint, weights_only=True)
    report = saved["report"]
    network = PoseChart(report["metadata"]["joint_limits"], report["width"], report["side"]).cuda()
    network.load_state_dict(saved["state_dict"])
    network.eval()
    model = build_scene()
    test = archive["split"] == "test"
    refs = archive["q_ref"][test]
    trajectories = archive["target"][test]
    rotations = archive["target_rotation"][test]
    old = np.load(source.parent / "test" / "records.npz")
    # Reproduce the historical lattice rotations exactly, preserving its nine charts.
    angles = np.stack(
        np.meshgrid(*([np.linspace(-np.pi / 18, np.pi / 18, 3)] * 3), indexing="ij"), -1
    ).reshape(-1, 3)
    order = np.random.default_rng(905).permutation(np.tile(np.arange(27), 27))
    grid_rotation = archive["rotation_center"] @ Rotation.from_rotvec(angles[order]).as_matrix()
    pools = {
        "waypoints": (
            np.repeat(refs, trajectories.shape[1], axis=0),
            trajectories.reshape(-1, 3),
            rotations.reshape(-1, 3, 3),
        ),
        "grid": (
            old["grid_reference"],
            old["grid_target"][:, 0],
            np.tile(grid_rotation, (9, 1, 1)),
        ),
    }
    valid = test[:, None] & (archive["teacher_error"] < 1e-7)
    valid &= (archive["teacher_angular_error"] < 1e-7) & (archive["teacher_margin"] > 0)
    valid &= np.linalg.norm(archive["target"] - archive["target"][:, :1], axis=-1) > 0.005
    eligible = np.flatnonzero(valid.ravel())
    chosen = np.random.default_rng(958).choice(eligible, 256, replace=False)
    pools["timing"] = (
        np.repeat(archive["q_ref"], 33, axis=0)[chosen],
        archive["target"].reshape(-1, 3)[chosen],
        archive["target_rotation"].reshape(-1, 3, 3)[chosen],
    )
    destination.mkdir(parents=True)
    metrics, records = {}, {}
    for scope, (reference, position, rotation) in pools.items():
        solver = CoefficientIK(model, len(reference), source.parent / "selected.pt")
        q = torch.tensor(reference, device="cuda:0", dtype=torch.float64)
        target = {
            k: torch.tensor(v, device="cuda:0", dtype=torch.float64)
            for k, v in (("position", position), ("rotation", rotation))
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        p, r, j = solver.geometry(q)
        relative = r.mT.cpu().numpy() @ rotation
        phi = Rotation.from_matrix(relative).as_rotvec()
        displacement = torch.cat(
            (
                target["position"] - p,
                (r @ torch.tensor(phi, device="cuda:0")[:, :, None]).squeeze(-1),
            ),
            -1,
        )
        basis, alpha, graph = network(q, pose_geometry(j), displacement)
        fixed = solver.network(q.float(), reference_geometry(j.float(), "pose")).double()
        full = torch.eye(7, device="cuda:0", dtype=torch.float64).expand(len(q), -1, -1)
        joint_seed = (basis @ alpha[:, :, None]).squeeze(-1)
        _, _, seed_jacobian = solver.geometry(q + joint_seed)
        seed_inverse, seed_null = pose_geometry(seed_jacobian)
        seed_frame = seed_inverse + seed_null[:, :, None] * graph[:, None, :]
        rebased = torch.linalg.solve_triangular(
            torch.linalg.cholesky(seed_frame.mT @ seed_frame), seed_frame.mT, upper=False
        ).mT
        torch.cuda.synchronize()
        prep_seconds = time.perf_counter() - started
        methods = {
            "fixed_original": (fixed, torch.zeros_like(alpha), "original"),
            "fixed_gain": (fixed, torch.zeros_like(alpha), "gain"),
            "conditional_zero": (basis, torch.zeros_like(alpha), "gain"),
            "conditional_seed": (basis, alpha, "gain"),
            "full_original": (full, torch.zeros_like(joint_seed), "original"),
            "full_gain": (full, torch.zeros_like(joint_seed), "gain"),
            "full_seed": (full, joint_seed, "gain"),
            "conditional_rebase": (rebased, torch.zeros_like(alpha), "gain"),
        }
        metrics[scope] = {"count": len(q), "combined_preparation_seconds": prep_seconds}
        for name, (b, initial, policy) in methods.items():
            if policy == "original":
                result = solver.solve(q, b, target, "pose", initial)
            else:
                origin = q + joint_seed if name == "conditional_rebase" else q
                result = solve_pose(solver, origin, b, target, initial)
            records[f"{scope}_{name}_origin"] = (
                (q + joint_seed if name == "conditional_rebase" else q).cpu().numpy()
            )
            records[f"{scope}_{name}_basis"] = b.cpu().numpy()
            values = {
                key: value.detach().cpu().numpy()
                for key, value in result.items()
                if key != "seconds"
            }
            for key, value in values.items():
                records[f"{scope}_{name}_{key}"] = value
            metrics[scope][name] = {
                "success": int(values["success"].sum()),
                "seconds": result["seconds"],
                "mean_iterations": float(values["iterations"].mean()),
                "max_position_error": float(values["position_error"].max()),
                "max_angular_error": float(values["angular_error"].max()),
            }
            if scope == "timing":
                repeated = []
                for _ in range(5):
                    timed = (
                        solver.solve(q, b, target, "pose", initial)
                        if policy == "original"
                        else solve_pose(solver, origin, b, target, initial)
                    )
                    repeated.append(timed["seconds"])
                metrics[scope][name]["batch_seconds"] = repeated
                metrics[scope][name]["batch_median"] = float(np.median(repeated))
            print(scope, name, metrics[scope][name], flush=True)
        records[f"{scope}_reference"] = reference
        records[f"{scope}_target"] = position
        records[f"{scope}_rotation"] = rotation
        if scope != "timing":
            del solver
    # Independent double-precision FK audits every final joint candidate and strict classification.
    data = mujoco.MjData(model)
    site = model.site("attachment_site").id
    audit = {}
    for scope in pools:
        for name in methods:
            configurations = records[f"{scope}_{name}_q"]
            cpu_error, cpu_angle = [], []
            for index, configuration in enumerate(configurations):
                data.qpos[:7] = configuration
                mujoco.mj_forward(model, data)
                cpu_error.append(np.linalg.norm(data.site_xpos[site] - pools[scope][1][index]))
                relative = pools[scope][2][index].T @ data.site_xmat[site].reshape(3, 3)
                cpu_angle.append(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec()))
            cpu_success = (np.array(cpu_error) <= 0.001) & (np.array(cpu_angle) <= np.pi / 180)
            audit[f"{scope}_{name}"] = int(
                np.sum(cpu_success != records[f"{scope}_{name}_success"])
            )
            records[f"{scope}_{name}_cpu_error"] = np.array(cpu_error)
            records[f"{scope}_{name}_cpu_angle"] = np.array(cpu_angle)
    metrics["cpu_classification_disagreements"] = audit
    np.savez_compressed(destination / "records.npz", **records)
    (destination / "summary.json").write_text(json.dumps(metrics, indent=2))
    archive.close()
    old.close()
    return metrics


def singularity_report(destination):
    """Verify an exact planar elbow rank-loss family against the pinned FR3 model."""
    model = build_scene()
    data = mujoco.MjData(model)
    site = model.site("attachment_site").id
    a, d3, d5 = 0.0825, 0.316, 0.384
    elbow = -np.arctan2(a * (d3 + d5), d3 * d5 - a * a)
    q = np.array([0, 0.72, 0, elbow, 0, 0.53, -0.7853])
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, site)
    jacobian = np.vstack((jp[:, :7], jr[:, :7]))
    source = Path("results/expansion_20261031T123139402928Z/side24_pose/motions.npz")
    archive = np.load(source, allow_pickle=False)
    phi = Rotation.from_matrix(
        archive["rotation_center"].T @ data.site_xmat[site].reshape(3, 3)
    ).as_rotvec()
    position = data.site_xpos[site].copy()
    offset = float(np.max(np.abs(position - np.array([0.5, 0, 0.5]))))
    joint_margin = float(np.min(np.minimum(q - model.jnt_range[:7, 0], model.jnt_range[:7, 1] - q)))
    if joint_margin <= 0 or np.max(np.abs(phi)) >= np.pi / 18:
        raise ValueError("The analytical singular witness violates the declared pose domain.")
    result = {
        "q": q.tolist(),
        "position": position.tolist(),
        "relative_rotvec": phi.tolist(),
        "joint_margin": joint_margin,
        "sigma": np.linalg.svd(jacobian, compute_uv=False).tolist(),
        "planar_factor": float((d3 * d5 - a * a) * np.sin(elbow) + a * (d3 + d5) * np.cos(elbow)),
        "inside_cube": {str(side): offset < side / 2 for side in (0.16, 0.24, 0.32, 0.40, 0.48)},
        "rank_certificate": "q3=q5=0; parallel joints 2,4,6 have collinear anchors; rank <=5",
        "absence_claim": "No absence certificate for the 16 cm cube or all IK branches.",
    }
    Path(destination).write_text(json.dumps(result, indent=2))
    archive.close()
    return result
