"""Nominal motion records for learning general and plane-conditioned joint bases."""

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from secrets import randbits

import mujoco
import numpy as np
import torch

from metaarm.ik import solve_position
from metaarm.scene import build_scene
from metaarm.simulation import Simulation

REFERENCE_COUNTS = (48, 8, 8)
SAMPLES = 65
DURATION = 4.0


def generate_motions(sim: Simulation, q_ref: np.ndarray, seed: int) -> dict:
    """Generate smooth seven-direction paths and evaluate their poses in GPU batches."""
    # Normalize each harmonic column to bound every joint displacement by 0.30 rad.
    reference = torch.as_tensor(q_ref, dtype=torch.float32, device="cuda:0")
    if reference.ndim != 2 or reference.shape[1] != 7 or len(reference) % sim.num_envs:
        raise ValueError("Reference states must be (episodes, 7), divisible by the GPU batch.")
    if not torch.isfinite(reference).all() or not torch.all(
        (reference - 0.30 > sim.bounds[:, 0]) & (reference + 0.30 < sim.bounds[:, 1])
    ):
        raise ValueError("Every reference requires an interior 0.30 rad joint margin.")
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    harmonics = torch.randn((len(reference), 7, 7), generator=generator, device="cuda:0")
    harmonics *= 0.15 / harmonics.abs().sum(dim=1, keepdim=True)
    seconds = torch.linspace(0, DURATION, SAMPLES, device="cuda:0")
    frequency = 2 * torch.pi * torch.arange(1, 8, device="cuda:0") / DURATION
    angles = seconds[:, None] * frequency
    displacement = torch.einsum("th,ehj->etj", 1 - angles.cos(), harmonics)
    velocity = torch.einsum("th,ehj->etj", angles.sin() * frequency, harmonics)
    configurations = reference[:, None] + displacement

    # These resets evaluate nominal configurations; no dynamic transitions are inferred.
    states = torch.cat((reference, configurations.reshape(-1, 7)))
    positions = torch.empty((len(states), 3), device="cuda:0")
    rotations = torch.empty((len(states), 3, 3), device="cuda:0")
    for start in range(0, len(states), sim.num_envs):
        sim.reset(states[start : start + sim.num_envs])
        positions[start : start + sim.num_envs] = sim.site_xpos[:, sim.site]
        rotations[start : start + sim.num_envs] = sim.site_xmat[:, sim.site].reshape(-1, 3, 3)
    general = {
        "q_ref": reference.cpu().numpy(),
        "q": configurations.cpu().numpy(),
        "qd": velocity.cpu().numpy(),
        "delta_q": (configurations - reference[:, None]).cpu().numpy(),
        "p_ref": positions[: len(reference)].cpu().numpy(),
        "R_ref": rotations[: len(reference)].cpu().numpy(),
        "p": positions[len(reference) :].reshape(-1, SAMPLES, 3).cpu().numpy(),
        "R": rotations[len(reference) :].reshape(-1, SAMPLES, 3, 3).cpu().numpy(),
        "seconds": seconds.cpu().numpy(),
        "harmonics": harmonics.cpu().numpy(),
    }

    # Cache the precise reference Jacobian once; ordered CPU library calls are explicit.
    jac_p, jac_r = np.empty((3, sim.model.nv)), np.empty((3, sim.model.nv))
    general["J_ref"] = np.empty((len(reference), 6, 7))
    for episode, q in enumerate(general["q_ref"]):
        mujoco.mj_resetDataKeyframe(sim.model, sim.kinematics, sim.model.key("home").id)
        sim.kinematics.qpos[sim.q_indices] = q
        mujoco.mj_forward(sim.model, sim.kinematics)
        mujoco.mj_jacSite(sim.model, sim.kinematics, jac_p, jac_r, sim.site)
        general["J_ref"][episode] = np.vstack((jac_p[:, sim.v_indices], jac_r[:, sim.v_indices]))
    if not np.isfinite(general["J_ref"]).all() or np.any(
        np.linalg.svd(general["J_ref"], compute_uv=False)[:, -1] <= 1e-7
    ):
        raise ValueError("A sampled reference violates the full-row-rank assumption.")
    general["episode_id"] = np.arange(len(reference))
    return general


def project_motions(sim: Simulation, general: dict, normals: np.ndarray) -> dict:
    """Project tip displacements onto reference planes and construct ordered joint labels."""
    # Define an oriented tangent frame by the least-aligned coordinate axis.
    normals = np.asarray(normals, dtype=np.float64)
    if normals.ndim != 3 or normals.shape[0] != len(general["q_ref"]) or normals.shape[2] != 3:
        raise ValueError("Normals must have shape (references, planes, 3).")
    if (
        normals.shape[1] == 0
        or not np.isfinite(normals).all()
        or not np.allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-10, rtol=0)
    ):
        raise ValueError("Plane normals must be finite unit vectors.")
    reference_id = np.repeat(np.arange(len(normals)), normals.shape[1])
    normal = normals.reshape(-1, 3)
    axis = np.eye(3)[np.argmin(np.abs(normal), axis=1)]
    tangent = np.cross(normal, axis)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    origin = general["p_ref"][reference_id].astype(np.float64)
    delta = general["p"][reference_id] - origin[:, None]
    targets = (
        origin[:, None]
        + delta
        - np.einsum("ptj,pj->pt", delta, normal)[..., None] * normal[:, None]
    )
    count, steps = targets.shape[:2]
    planar = {
        "reference_id": reference_id,
        "episode_id": len(normals) + np.arange(count),
        "normal": normal,
        "origin": origin,
        "axes": np.stack((tangent, np.cross(normal, tangent)), -1),
        "target": targets,
        "q": np.full((count, steps, 7), np.nan),
        "delta_q": np.full((count, steps, 7), np.nan),
        "attempted": np.zeros((count, steps), bool),
        "success": np.zeros((count, steps), bool),
        "feasible": np.zeros((count, steps), bool),
        "optimizer_success": np.zeros((count, steps), bool),
        "message": np.full((count, steps), "unattempted", dtype="U256"),
        "gradient": np.full((count, steps, 7), np.nan),
        "multipliers": np.full((count, steps, 14), np.nan),
    }
    for key in ("position_error", "normal_error", "seconds", "objective"):
        planar[key] = np.full((count, steps), np.nan)
    for key in ("status", "iterations", "evaluations", "gradient_evaluations"):
        planar[key] = np.full((count, steps), -1, np.int32)

    # Stop each path at its first task failure and retain the actual failed candidate.
    for episode, ref in enumerate(reference_id):
        seed = general["q_ref"][ref].astype(np.float64)
        for waypoint, target in enumerate(targets[episode]):
            result = solve_position(sim.model, sim.kinematics, target, seed)
            optimizer = result["optimizer"]
            index = (episode, waypoint)
            planar["q"][index] = result["q"]
            planar["delta_q"][index] = result["q"] - general["q_ref"][ref]
            planar["attempted"][index], planar["feasible"][index] = True, result["feasible"]
            planar["position_error"][index] = result["errors"][0]
            planar["normal_error"][index] = abs(
                normal[episode] @ (sim.kinematics.site_xpos[sim.site] - origin[episode])
            )
            planar["seconds"][index], planar["objective"][index] = result["seconds"], optimizer.fun
            planar["optimizer_success"][index] = optimizer.success
            planar["status"][index], planar["message"][index] = optimizer.status, optimizer.message
            planar["iterations"][index], planar["evaluations"][index] = (
                optimizer.nit,
                optimizer.nfev,
            )
            planar["gradient_evaluations"][index] = optimizer.njev
            planar["gradient"][index], planar["multipliers"][index] = (
                optimizer.jac,
                optimizer.multipliers,
            )
            planar["success"][index] = result["feasible"] and result["errors"][0] <= 1e-3
            if not planar["success"][index]:
                break
            seed = result["q"]
    planar["included"] = np.all(planar["success"], axis=1)
    planar["label_mask"] = planar["included"][:, None] & planar["success"]
    return planar


def collect(destination: Path) -> dict:
    """Sample one split-aware dataset and save it without overwriting existing data."""
    # Hold out reference regions and plane tilts before creating any motion records.
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    started = time.perf_counter()
    sim = Simulation(build_scene(), 64)
    # One entropy-derived seed identifies and reproduces both numerical random streams.
    seed = randbits(63)
    rng = np.random.default_rng(seed)
    split_id = np.repeat(np.arange(3), REFERENCE_COUNTS)
    regions = np.array([[-0.60, 0.10], [0.25, 0.40], [0.55, 0.70]])
    tilts = np.deg2rad([[0, 45], [50, 65], [70, 85]])
    references = sim.home.cpu().numpy() + rng.uniform(-0.25, 0.25, (len(split_id), 7))
    references[:, 0] = rng.uniform(regions[split_id, 0], regions[split_id, 1])
    polar = rng.uniform(tilts[split_id, 0, None], tilts[split_id, 1, None], (len(split_id), 2))
    azimuth = rng.uniform(0, 2 * np.pi, polar.shape)
    normals = np.stack(
        (np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)), -1
    )
    generated = time.perf_counter()
    general = generate_motions(sim, references, seed)
    general_seconds = time.perf_counter() - generated
    general["split"] = np.array(["train", "validation", "test"])[split_id]
    generated = time.perf_counter()
    planar = project_motions(sim, general, normals)
    planar_seconds = time.perf_counter() - generated
    planar["split"] = general["split"][planar["reference_id"]]

    # Keep model/data provenance in the same pickle-free archive as the numerical arrays.
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "-c", "core.fsmonitor=false", "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    metadata = {
        "kind": "nominal_kinematic_training_data",
        "created_utc": datetime.now(UTC).isoformat(),
        "dynamics_integrated": False,
        "collision_checked": False,
        "seed": seed,
        "reference_counts": list(REFERENCE_COUNTS),
        "samples": SAMPLES,
        "duration_seconds": DURATION,
        "joint_displacement_bound_rad": 0.30,
        "reference_joint1_regions": regions.tolist(),
        "normal_tilt_ranges_rad": tilts.tolist(),
        "frame": "fr3_link0 = world; attachment_site tool; base-aligned [linear; angular] Jacobian",
        "joint_names": [f"fr3_joint{i}" for i in range(1, 8)],
        "joint_limits": sim.bounds.cpu().numpy().tolist(),
        "tool_translation": sim.model.site_pos[sim.site].tolist(),
        "git_revision": revision,
        "git_status": dirty,
        "packages": {
            name: version(name)
            for name in ("mujoco", "mujoco-warp", "warp-lang", "torch", "numpy", "scipy")
        },
        "gpu": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src" / "metaarm").glob("*.py"))
        },
        "assets": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "assets").rglob("*"))
            if p.is_file()
        },
        "general_generation_seconds": general_seconds,
        "planar_construction_seconds": planar_seconds,
        "collection_seconds": time.perf_counter() - started,
        "planar_included_episodes": int(planar["included"].sum()),
        "planar_excluded_episodes": int((~planar["included"]).sum()),
        "planar_planned_waypoints": int(planar["attempted"].size),
        "planar_attempted_waypoints": int(planar["attempted"].sum()),
        "planar_failed_waypoints": int((planar["attempted"] & ~planar["success"]).sum()),
        "label_rule": "Entire successful planar paths only; no orientation labels or rescue.",
        "missing_rule": "NaN/-1/unattempted are absent; use attempted and label_mask.",
    }
    arrays = {f"general_{key}": value for key, value in general.items()}
    arrays.update({f"planar_{key}": value for key, value in planar.items()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.open("xb")
    np.savez_compressed(archive, **arrays, metadata=np.array(json.dumps(metadata)))
    archive.close()
    return metadata


if __name__ == "__main__":
    # Give every invocation an explicit archive identity without replacing earlier data.
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(__file__).resolve().parents[2] / "data" / f"motions_{timestamp}.npz"
    report = collect(output)
    print(
        f"Saved {output} (seed {report['seed']}): "
        f"{report['planar_included_episodes']} complete planar episodes; "
        f"{report['planar_excluded_episodes']} excluded; "
        f"{report['planar_attempted_waypoints']}/{report['planar_planned_waypoints']} waypoints."
    )
