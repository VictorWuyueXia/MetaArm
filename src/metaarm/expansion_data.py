"""Paired regional position/pose demonstrations with independent full-joint teachers."""

import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial.transform import Rotation

from metaarm.scene import build_scene
from metaarm.workspace_data import CENTER, lift_positions

COUNTS = (96, 24, 48)


def lift_pose(model, targets, rotations, q_ref):
    """Minimize joint displacement under exact position and rotation constraints."""
    cpu, site = mujoco.MjData(model), model.site("attachment_site").id
    bounds, target, desired = model.jnt_range, np.empty(3), np.empty((3, 3))
    records = {
        key: []
        for key in (
            "q",
            "position",
            "jacobian",
            "error",
            "angular_error",
            "margin",
            "success",
            "status",
            "iterations",
            "seconds",
        )
    }

    def joint_distance(q):
        displacement = q - q_ref
        return 0.5 * displacement @ displacement, displacement

    def pose_constraint(q):
        cpu.qpos[:] = q
        mujoco.mj_forward(model, cpu)
        rotation = cpu.site_xmat[site].reshape(3, 3)
        phi = Rotation.from_matrix(desired.T @ rotation).as_rotvec()
        return np.r_[cpu.site_xpos[site] - target, phi]

    def pose_jacobian(q):
        residual = pose_constraint(q)
        rotation, phi = cpu.site_xmat[site].reshape(3, 3), residual[3:]
        angle = np.linalg.norm(phi)
        if angle >= np.pi - 1e-6:
            raise ValueError("The pose teacher reaches the rotation logarithm cut.")
        skew = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
        coefficient = (
            1 / 12 + angle**2 / 720
            if angle < 1e-3
            else (1 - angle / (2 * np.tan(angle / 2))) / angle**2
        )
        inverse = np.eye(3) + skew / 2 + coefficient * (skew @ skew)
        jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
        mujoco.mj_jacSite(model, cpu, jp, jr, site)
        return np.vstack((jp, inverse @ rotation.T @ jr))

    for point, rotation in zip(targets, rotations, strict=True):
        target[:], desired[:] = point, rotation
        started = time.perf_counter()
        result = minimize(
            joint_distance,
            q_ref.copy(),
            jac=True,
            method="SLSQP",
            constraints=[
                LinearConstraint(np.eye(7), bounds[:, 0], bounds[:, 1]),
                {"type": "eq", "fun": pose_constraint, "jac": pose_jacobian},
            ],
            options={"ftol": 1e-12, "maxiter": 100},
        )
        seconds = time.perf_counter() - started
        residual = pose_constraint(result.x)
        error, angular = np.linalg.norm(residual[:3]), np.linalg.norm(residual[3:])
        margin = np.minimum(result.x - bounds[:, 0], bounds[:, 1] - result.x).min()
        jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
        mujoco.mj_jacSite(model, cpu, jp, jr, site)
        if not np.isfinite(result.x).all() or not np.isfinite(residual).all():
            raise FloatingPointError("The full-pose teacher produced nonfinite data.")
        values = (
            result.x.copy(),
            cpu.site_xpos[site].copy(),
            np.vstack((jp, jr)),
            error,
            angular,
            margin,
            bool(result.success and error < 1e-7 and angular < 1e-7 and margin > 0),
            result.status,
            result.nit,
            seconds,
        )
        for key, value in zip(records, values, strict=True):
            records[key].append(value)
    return {key: np.asarray(value) for key, value in records.items()}


def collect_experiment(destination, side, task):
    """Generate paired Cartesian paths and free-position or constrained-pose labels in one cube."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if task not in ("position", "pose") or side <= 0:
        raise ValueError("An experiment requires a positive cube side and an explicit task.")
    model = build_scene()
    center = lift_positions(model, CENTER[None], model.key("home").qpos)
    if not center["success"].all():
        raise RuntimeError("The common reference center failed.")
    q_center = center["q"][0]
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = q_center
    mujoco.mj_forward(model, cpu)
    rotation_center = cpu.site_xmat[model.site("attachment_site").id].reshape(3, 3).copy()

    # References remain identical across all sizes and both task constraints.
    rng, count = np.random.default_rng(60432), sum(COUNTS)
    positions = rng.uniform(CENTER - 0.04, CENTER + 0.04, (count, 3))
    angles = rng.uniform(-np.pi / 36, np.pi / 36, (count, 3))
    references = lift_pose(
        model, positions, rotation_center @ Rotation.from_rotvec(angles).as_matrix(), q_center
    )
    if (
        np.any(references["error"] > 1e-7)
        or np.any(references["angular_error"] > 1e-7)
        or np.any(references["margin"] <= 0)
    ):
        raise RuntimeError("A safely interior reference failed its declared pose.")
    box = np.stack((CENTER - side / 2, CENTER + side / 2))
    vertices = np.concatenate(
        (positions[:, None], rng.uniform(box[0], box[1], (count, 3, 3)), positions[:, None]), 1
    )
    angular_vertices = np.concatenate(
        (angles[:, None], rng.uniform(-np.pi / 18, np.pi / 18, (count, 3, 3)), angles[:, None]), 1
    )
    phase = np.arange(1, 9) / 8
    blend = (phase**3 * (10 - 15 * phase + 6 * phase**2))[None, None, :, None]
    targets = np.concatenate(
        (
            positions[:, None],
            ((1 - blend) * vertices[:, :-1, None] + blend * vertices[:, 1:, None]).reshape(
                count, 32, 3
            ),
        ),
        1,
    )
    angular_path = np.concatenate(
        (
            angles[:, None],
            (
                (1 - blend) * angular_vertices[:, :-1, None] + blend * angular_vertices[:, 1:, None]
            ).reshape(count, 32, 3),
        ),
        1,
    )
    rotations = rotation_center @ Rotation.from_rotvec(
        angular_path.reshape(-1, 3)
    ).as_matrix().reshape(count, 33, 3, 3)
    groups = []
    for episode in range(count):
        q_ref = references["q"][episode]
        labels = (
            lift_positions(model, targets[episode], q_ref)
            if task == "position"
            else lift_pose(model, targets[episode], rotations[episode], q_ref)
        )
        groups.append(labels)
        if (episode + 1) % 24 == 0:
            print(
                f"{side:.2f}m/{task}: labelled {episode + 1}/{count} reference paths.", flush=True
            )
    teacher = {key: np.stack([group[key] for group in groups]) for key in groups[0]}
    included = teacher["success"].all(axis=1)
    corners = np.array(np.meshgrid(*box.T, indexing="ij")).reshape(3, -1).T
    if task == "pose":
        orientation_corners = (
            np.array(np.meshgrid(*np.tile([-np.pi / 18, np.pi / 18], (3, 1)), indexing="ij"))
            .reshape(3, -1)
            .T
        )
        probe_target = np.repeat(corners, 8, axis=0)
        probe_rotation = (
            rotation_center @ Rotation.from_rotvec(np.tile(orientation_corners, (8, 1))).as_matrix()
        )
        probe = lift_pose(model, probe_target, probe_rotation, q_center)
    else:
        probe_target = corners
        probe = lift_positions(model, corners, q_center)

    # Expected teacher failures are retained and never change the declared region.
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "kind": f"workspace_{task}_data",
        "task": task,
        "side": side,
        "box": box.tolist(),
        "center": CENTER.tolist(),
        "orientation_limit_rad": float(np.pi / 18),
        "reference_counts": list(COUNTS),
        "seed": 60432,
        "samples": 33,
        "joint_limits": model.jnt_range.tolist(),
        "included": int(included.sum()),
        "excluded": int((~included).sum()),
        "label_seconds": float(teacher["seconds"].sum()),
        "assets": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "assets").rglob("*"))
            if p.is_file()
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src/metaarm").glob("*.py"))
        },
    }
    arrays = {
        "q_center": q_center,
        "rotation_center": rotation_center,
        "q_ref": references["q"],
        "J_ref": references["jacobian"],
        "target": targets,
        "target_rotation": rotations,
        "q": teacher["q"],
        "J": teacher["jacobian"],
        "delta_q": teacher["q"] - references["q"][:, None],
        "success": teacher["success"],
        "included": included,
        "split": np.repeat(["train", "validation", "test"], COUNTS),
        "episode_id": np.arange(count),
        "probe_target": probe_target,
        "probe_success": probe["success"],
        "probe_error": probe["error"],
        "metadata": np.array(json.dumps(metadata)),
    }
    for key in ("error", "margin", "status", "iterations", "seconds"):
        arrays[f"teacher_{key}"] = teacher[key]
    if task == "pose":
        arrays["teacher_angular_error"], arrays["probe_angular_error"] = (
            teacher["angular_error"],
            probe["angular_error"],
        )
    if not np.all((targets >= box[0]) & (targets <= box[1])):
        raise ValueError("A generated target left its declared cube.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = destination.open("xb")
    np.savez_compressed(output, **arrays)
    output.close()
    print(f"Saved {destination}: {included.sum()}/{count} converged teacher paths.", flush=True)
    return metadata
