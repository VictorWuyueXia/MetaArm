"""Bounded Cartesian demonstrations from an independent minimum-motion full-joint teacher."""

import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from metaarm.scene import build_scene

CENTER = np.array([0.50, 0.0, 0.50])
BOX = np.stack((CENTER - 0.08, CENTER + 0.08))
COUNTS = (96, 24, 48)


def lift_positions(model, targets, q_center):
    """Find the closest seven-joint configuration to one fixed center for each target."""
    targets, q_center = (
        np.asarray(targets, dtype=np.float64),
        np.asarray(q_center, dtype=np.float64),
    )
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
        raise ValueError("The teacher requires finite Cartesian target rows.")
    cpu, site = mujoco.MjData(model), model.site("attachment_site").id
    bounds = model.jnt_range
    records = {
        key: []
        for key in (
            "q",
            "position",
            "jacobian",
            "error",
            "margin",
            "success",
            "status",
            "iterations",
            "seconds",
        )
    }
    target = np.empty(3)

    # The objective fixes a seven-dimensional redundancy convention independently of learning.
    def joint_distance(q):
        displacement = q - q_center
        return 0.5 * displacement @ displacement, displacement

    def position_constraint(q):
        cpu.qpos[:] = q
        mujoco.mj_forward(model, cpu)
        return cpu.site_xpos[site] - target

    def position_jacobian(q):
        cpu.qpos[:] = q
        mujoco.mj_forward(model, cpu)
        jacobian = np.empty((3, model.nv))
        mujoco.mj_jacSite(model, cpu, jacobian, None, site)
        return jacobian

    for index, point in enumerate(targets):
        target[:] = point
        started = time.perf_counter()
        result = minimize(
            joint_distance,
            q_center.copy(),
            jac=True,
            method="SLSQP",
            constraints=[
                LinearConstraint(np.eye(7), bounds[:, 0], bounds[:, 1]),
                {"type": "eq", "fun": position_constraint, "jac": position_jacobian},
            ],
            options={"ftol": 1e-12, "maxiter": 100},
        )
        seconds = time.perf_counter() - started
        error = np.linalg.norm(position_constraint(result.x))
        margin = np.minimum(result.x - bounds[:, 0], bounds[:, 1] - result.x).min()
        jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
        mujoco.mj_jacSite(model, cpu, jp, jr, site)
        if not np.isfinite(result.x).all() or not np.isfinite(error):
            raise FloatingPointError("The full-joint teacher produced nonfinite data.")
        values = (
            result.x.copy(),
            cpu.site_xpos[site].copy(),
            np.vstack((jp, jr)),
            error,
            margin,
            bool(result.success and error < 1e-7 and margin > 0),
            result.status,
            result.nit,
            seconds,
        )
        for key, value in zip(records, values, strict=True):
            records[key].append(value)
        if (index + 1) % 1000 == 0:
            print(f"Labelled {index + 1}/{len(targets)} unique regional positions.", flush=True)
    return {key: np.asarray(value) for key, value in records.items()}


def collect_workspace(destination):
    """Generate disjoint regional reference groups and closed paths inside one frozen box."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    model = build_scene()
    center = lift_positions(model, CENTER[None], model.key("home").qpos)
    if not center["success"].all():
        raise RuntimeError(
            "The declared workspace center did not admit the full-joint initialization."
        )
    q_center = center["q"][0]
    corners = np.array(np.meshgrid(*BOX.T, indexing="ij")).reshape(3, -1).T
    probe = lift_positions(model, corners, q_center)
    if not probe["success"].all():
        raise RuntimeError("A declared workspace corner failed the full-joint probe.")

    # Convex quintic interpolation keeps every saved target inside the declared volume.
    rng = np.random.default_rng(60431)
    count = sum(COUNTS)
    reference = rng.uniform(BOX[0], BOX[1], (count, 3))
    vertices = np.concatenate(
        (reference[:, None], rng.uniform(BOX[0], BOX[1], (count, 3, 3)), reference[:, None]), axis=1
    )
    phase = np.arange(1, 9) / 8
    blend = (phase**3 * (10 - 15 * phase + 6 * phase**2))[None, None, :, None]
    paths = ((1 - blend) * vertices[:, :-1, None] + blend * vertices[:, 1:, None]).reshape(
        count, 32, 3
    )
    target = np.concatenate((reference[:, None], paths), axis=1)
    unique, indices = np.unique(target.reshape(-1, 3), axis=0, return_inverse=True)
    teacher = lift_positions(model, unique, q_center)
    indices = indices.reshape(count, 33)
    if not teacher["success"][indices[:, 0]].all():
        raise RuntimeError("A reference position failed the declared full-joint lifting.")
    q = teacher["q"][indices]
    success = teacher["success"][indices]
    included = success.all(axis=1)
    actual = teacher["position"][indices]
    if not np.all((target >= BOX[0]) & (target <= BOX[1])):
        raise ValueError("A generated target left the declared box.")
    if not np.all((actual[included] >= BOX[0]) & (actual[included] <= BOX[1])):
        raise ValueError("A successful teacher path left the declared box.")

    # Store every solver result and exact split before any neural fitting begins.
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "kind": "workspace_position_data",
        "box": BOX.tolist(),
        "center": CENTER.tolist(),
        "reference_counts": list(COUNTS),
        "seed": 60431,
        "samples": 33,
        "duration_seconds": 4.0,
        "joint_limits": model.jnt_range.tolist(),
        "teacher": "Seven-joint minimum distance to fixed q_center with exact position equality.",
        "assets": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "assets").rglob("*"))
            if p.is_file()
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src/metaarm").glob("*.py"))
        },
        "initialization_seconds": float(center["seconds"].sum()),
        "label_seconds": float(teacher["seconds"].sum()),
        "unique_positions": len(unique),
        "included": int(included.sum()),
        "excluded": int((~included).sum()),
    }
    arrays = {
        "q_center": q_center,
        "q_ref": q[:, 0],
        "J_ref": teacher["jacobian"][indices[:, 0]],
        "reference_position": reference,
        "target": target,
        "q": q,
        "delta_q": q - q[:, :1],
        "actual_position": actual,
        "success": success,
        "included": included,
        "split": np.repeat(["train", "validation", "test"], COUNTS),
        "episode_id": np.arange(count),
        "probe_target": corners,
        "probe_q": probe["q"],
        "probe_J": probe["jacobian"],
        "metadata": np.array(json.dumps(metadata)),
    }
    for key in ("error", "margin", "status", "iterations", "seconds"):
        arrays[f"teacher_{key}"] = teacher[key][indices]
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = destination.open("xb")
    np.savez_compressed(output, **arrays)
    output.close()
    print(f"Saved {destination}: {included.sum()}/{count} complete regional paths.", flush=True)
    return metadata
