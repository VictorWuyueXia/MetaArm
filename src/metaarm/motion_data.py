"""Load every requested motion archive with immutable source and split identities."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from metaarm.basis import reference_geometry


def load_motions(source: Path) -> dict:
    """Pool complete archive groups and compute exact displacement second moments."""
    if not torch.cuda.is_available():
        raise RuntimeError("Motion-basis fitting requires CUDA PyTorch.")
    source = Path(source).resolve()
    paths = sorted(source.rglob("*.npz")) if source.is_dir() else [source]
    if not paths:
        raise ValueError("The requested corpus contains no NPZ archives.")
    root = Path(__file__).resolve().parents[2]
    references, jacobians, reference_splits, manifests = [], [], [], []
    grouped = {
        task: {
            key: []
            for key in (
                "reference",
                "covariance",
                "normal",
                "samples",
                "split",
                "source_id",
                "episode_id",
                "endpoint",
                "peak",
            )
        }
        for task in ("pose", "plane")
    }
    offset = 0
    for archive_id, path in enumerate(paths):
        archive = np.load(path, allow_pickle=False)
        metadata = json.loads(archive["metadata"].item())
        if metadata["kind"] != "nominal_kinematic_training_data":
            raise ValueError(f"Unexpected archive kind: {path}.")
        q, jac, split = archive["general_q_ref"], archive["general_J_ref"], archive["general_split"]
        if q.ndim != 2 or q.shape[1] != 7 or jac.shape != (len(q), 6, 7):
            raise ValueError(f"Invalid reference dimensions: {path}.")
        if split.shape != (len(q),) or not np.isin(split, ["train", "validation", "test"]).all():
            raise ValueError(f"Invalid reference splits: {path}.")
        if not np.isfinite(q).all() or len(np.unique(archive["general_episode_id"])) != len(q):
            raise ValueError(f"Non-finite or duplicate reference identities: {path}.")
        if archive_id == 0:
            assets, limits = metadata["assets"], metadata["joint_limits"]
            for name, digest in assets.items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Physical model differs from the corpus: {name}.")
        if metadata["assets"] != assets or metadata["joint_limits"] != limits:
            raise ValueError("All archives must use the same physical model and limits.")
        ref = archive["planar_reference_id"]
        included, success, mask = (
            archive[name] for name in ("planar_included", "planar_success", "planar_label_mask")
        )
        if ref.dtype.kind not in "iu" or np.any((ref < 0) | (ref >= len(q))):
            raise ValueError("A planar episode has an invalid reference.")
        if not np.array_equal(archive["planar_split"], split[ref]):
            raise ValueError("Related general and planar episodes cannot cross splits.")
        if (
            included.dtype != bool
            or not np.array_equal(included, success.all(axis=1))
            or not np.array_equal(mask, included[:, None] & success)
            or not archive["planar_attempted"][mask].all()
        ):
            raise ValueError("Labels must identify complete successful planar paths.")
        references.append(q.copy())
        jacobians.append(jac.copy())
        reference_splits.append(split.copy())

        # Each second moment retains every sample, including the nonzero path mean.
        for task in ("pose", "plane"):
            r = np.arange(len(q)) if task == "pose" else ref[included]
            d = (
                archive["general_delta_q"]
                if task == "pose"
                else archive["planar_delta_q"][included]
            )
            if d.ndim != 3 or d.shape[0] != len(r) or d.shape[2] != 7 or d.shape[1] == 0:
                raise ValueError("Displacements must be nonempty (episodes, samples, 7) arrays.")
            if not np.isfinite(d).all():
                raise ValueError("Included displacements must be finite.")
            d = d.astype(np.float64)
            g = grouped[task]
            g["reference"].append(r + offset)
            g["covariance"].append(np.swapaxes(d, -1, -2) @ d / d.shape[1])
            g["samples"].append(np.full(len(r), d.shape[1], np.int64))
            g["split"].append(split[r])
            g["source_id"].append(np.full(len(r), archive_id, np.int64))
            g["episode_id"].append(
                archive[f"{'general' if task == 'pose' else 'planar'}_episode_id"][
                    slice(None) if task == "pose" else included
                ]
            )
            g["endpoint"].append(np.linalg.norm(d[:, -1], axis=-1))
            g["peak"].append(np.max(np.abs(d), axis=(1, 2)))
            if task == "plane":
                g["normal"].append(archive["planar_normal"][included].copy())
        manifests.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "metadata": metadata,
                "references": len(q),
                "planar_included": int(included.sum()),
                "planar_excluded": int((~included).sum()),
            }
        )
        offset += len(q)
        archive.close()

    # Identical reference configurations cannot acquire different splits in other files.
    q = np.concatenate(references)
    split = np.concatenate(reference_splits)
    _, groups = np.unique(q, axis=0, return_inverse=True)
    identities = {name: set(groups[split == name]) for name in ("train", "validation", "test")}
    if any(
        identities[a] & identities[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("A reference configuration crosses corpus split boundaries.")
    records = {
        "q_ref": torch.tensor(q, device="cuda:0", dtype=torch.float32),
        "J_ref": torch.tensor(np.concatenate(jacobians), device="cuda:0", dtype=torch.float32),
        "joint_limits": torch.tensor(limits, device="cuda:0", dtype=torch.float32),
        "sources": manifests,
        "source": str(source),
        "sha256": hashlib.sha256(
            json.dumps([(m["path"], m["sha256"]) for m in manifests]).encode()
        ).hexdigest(),
    }
    records["geometry"] = {
        task: reference_geometry(records["J_ref"], task) for task in ("pose", "plane")
    }
    for task, data in grouped.items():
        splits = np.concatenate(data["split"])
        data["rows"] = {name: np.flatnonzero(splits == name) for name in identities}
        if any(len(rows) == 0 for rows in data["rows"].values()):
            raise ValueError("Both tasks require retained episodes in every split.")
        records[task] = {
            key: torch.tensor(
                np.concatenate(data[key]),
                device="cuda:0",
                dtype=torch.float64 if key in ("covariance", "endpoint", "peak") else torch.long,
            )
            for key in (
                "reference",
                "covariance",
                "samples",
                "source_id",
                "episode_id",
                "endpoint",
                "peak",
            )
        }
        records[task]["normal"] = (
            None
            if task == "pose"
            else torch.tensor(np.concatenate(data["normal"]), device="cuda:0", dtype=torch.float32)
        )
        records[task]["rows"] = {
            name: torch.tensor(rows, device="cuda:0", dtype=torch.long)
            for name, rows in data["rows"].items()
        }
    return records
