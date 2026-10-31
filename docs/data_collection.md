# Motion-data collection

The collector implements section 7 of the [mathematical formulation](1-mathematical_formulation_and_implementation_plan.md). These records train joint motion bases, not a joint-reference predictor.

## Run

From the project root:

~~~powershell
uv run --locked python -m metaarm.collection
~~~

Each invocation creates data/motions_<UTC timestamp>.npz and prints the full path on completion. Earlier archives, including the initial data/motions.npz, are preserved. An explicitly supplied existing path to collect(destination) still raises FileExistsError before computation. The standalone collector does not change the experiment UI or its reaching configuration. SciPy SLSQP is included in the locked project environment.

The archive contains nominal kinematic data. GPU forward kinematics evaluates prescribed joint configurations; there are no simulated dynamic transitions, torque labels, collision-free claims, or hardware demonstrations. The velocity field is the analytic derivative of the prescribed joint path.

## Sampling specification

There are 64 reference configurations: 48 training, eight validation, and eight test. Each has one general joint trajectory and two planar trajectories. Every path has 65 samples over four seconds.

Other joints vary within 0.25 rad of home. Joint 1 is sampled in disjoint reference regions: training [-0.60, 0.10], validation [0.25, 0.40], and test [0.55, 0.70] rad. Related general/planar paths always share their reference's split. Plane tilt from +z is sampled in [0,45], [50,65], and [70,85] degrees, respectively; azimuth spans a full turn. Thus both reference regions and plane orientations are held out. These are joint holdouts, not an isolated measurement of either generalization effect.

Each invocation draws a fresh 63-bit seed from operating-system entropy and records it in metadata and the completion message. That seed initializes both NumPy sampling and the dedicated CUDA PyTorch generator. Reusing it with the same implementation/environment reproduces the numerical samples; timestamps and timings still differ. Seven cosine harmonics produce smooth joint trajectories. For normalized phase s:

~~~
q(s) = q_ref + sum_h a_h [1 - cos(2*pi*h*s)], h = 1,...,7.
sum_h |a_h,j| = 0.15 rad for each joint j.
~~~

The displacement magnitude of each joint is therefore bounded by 0.30 rad. Reference margins are checked explicitly. The paths start/end at q_ref with zero endpoint velocity and provide seven joint-motion directions. A sampled reference failing the numerical full-row-rank check is an error; it is never resampled or regularized.

Repeated runs sample new references, joint harmonics, and planes within these same domains. The initial dataset establishes the numerical pipeline. Its size, local sampling domain, and random motion construction do not establish that the learned basis will generalize usefully.

## General records

Joint paths are generated with batched PyTorch matrices. MuJoCo Warp evaluates poses in batches of 64 worlds. Reference Jacobians are evaluated once on CPU at the stored reference configurations.

| Archive field | Meaning and shape |
| --- | --- |
| general_q_ref | Reference arm configurations, (64, 7). |
| general_q, general_qd | Prescribed joint positions and analytic velocities, (64, 65, 7). |
| general_delta_q | Exactly general_q minus its reference, (64, 65, 7). |
| general_p, general_R | Base-frame tool positions/rotations, (64, 65, 3) and (64, 65, 3, 3). |
| general_p_ref, general_R_ref | Reference tool poses, (64, 3) and (64, 3, 3). |
| general_J_ref | Base-aligned [linear; angular] Jacobians, (64, 6, 7). |
| general_seconds, general_harmonics | Prescribed sample times and harmonic coefficients. These coefficients are not learned-basis coefficients. |
| general_episode_id, general_split | Reference/episode identity and preassigned split. |

The GPU pose/joint arrays use float32. CPU Jacobians and planar optimization use float64. General displacements retain the stored joint subtraction identity rather than an independently rounded analytic displacement.

## Planar construction and failure semantics

For each reference, the origin is its tool position. A deterministic least-aligned coordinate axis defines an orthonormal tangent frame. General tip displacements are projected with I - n*n.T onto the plane through that origin.

The full-joint reference solver minimizes half the squared position residual normalized by 0.001 m. It uses exact position Jacobians, SLSQP, explicit linear joint constraints, 200 iterations maximum, and ftol = 1e-10. Orientation is absent. The first seed is q_ref and subsequent seeds are the previous successful joint result. No clipping, alternate solver, or rescue solve is used.

Task success requires joint feasibility and position error at most 0.001 m. Optimizer convergence is stored separately. A nonzero optimizer status is not mathematical unreachability; a zero status alone is not sufficient task success. At the first task failure, retain its actual candidate/result and mark the remaining planned waypoints unattempted. Other independent episodes continue.

Only entire successful planar trajectories provide learning labels. Successful prefixes of failed trajectories are preserved for analysis but excluded from training labels.

| Archive field | Meaning |
| --- | --- |
| planar_reference_id, planar_episode_id, planar_split | Links to general references, unique episode IDs, and split membership. |
| planar_normal, planar_origin, planar_axes | Explicit plane normal, location, and oriented tangent axes. |
| planar_target | Projected positions, (128, 65, 3); no orientation target. |
| planar_q, planar_delta_q | Actual candidate joints and displacements from q_ref, (128, 65, 7). |
| planar_attempted, planar_success, planar_feasible | Distinct attempt, task-success, and joint-feasibility masks. |
| planar_included, planar_label_mask | Complete-episode inclusion and the exact usable label mask. |
| planar_position_error, planar_normal_error | Raw errors in metres, independent of objective normalization. |
| planar_optimizer_success, planar_status, planar_message | Optimizer termination evidence, separate from task success. |
| planar_iterations, planar_evaluations, planar_gradient_evaluations | Original optimizer iteration/evaluation counts. |
| planar_objective, planar_gradient, planar_multipliers | Remaining numerical SLSQP result fields; candidate q stores its x. |
| planar_seconds | Measured minimization cost for each attempted waypoint. |

NaN denotes absent numerical values; -1 and "unattempted" denote absent status/count/message entries. Always use attempted/inclusion masks. No previous state is inserted as a missing label. There are no artificial zero angular errors for the free-orientation task.

## Provenance and reading

The metadata field is a JSON string stored with the arrays. It records UTC creation time, seeds and sampling domains, joint/tool/frame definitions, exact package versions, GPU/CUDA identity, Git revision and dirty paths, source-file and asset SHA-256 hashes, construction costs, inclusion/exclusion counts, and planned/attempted/failed waypoint counts.

Load without pickle and close the reader explicitly:

~~~python
import json
import numpy as np

# Use the path printed by the collector; this is the initial verification archive.
records = np.load("data/motions.npz", allow_pickle=False)
metadata = json.loads(records["metadata"].item())
usable = records["planar_label_mask"]
joint_labels = records["planar_q"][usable]
records.close()
~~~

The sample archive is generated only after all episodes finish. An interrupted standalone run is not a completed dataset. Construction time excludes final archive compression. Data and test output directories remain ignored by Git.

## Verification

The collection tests check smoothness, joint bounds, seven-dimensional motion span, reproducibility, GPU/CPU pose agreement, the analytic SLSQP gradient, explicit trajectory failure/unattempted masks, projected plane geometry, real reference labels, split linkage, archive round-trip without pickle, and overwrite rejection.

~~~powershell
uv run --locked pytest -q
uv run --locked ruff check src tests
~~~

On this machine, pytest's existing system temporary directory denied access. Validation used a new project-local directory via --basetemp under runs/. This is a test-run setting, not collector fallback behavior.
## Validated collection

The complete fixed collection on 2026-10-31 produced 64 general episodes (4,160 samples) and 128 complete planar episodes (8,320 labels). All 8,320 planned planar waypoints were attempted, all optimizer statuses were zero, and no planar episode was excluded in this sample. The planar split counts are 96 training, 16 validation, and 16 test episodes.

An independent CPU forward-kinematics audit of every usable planar joint label found a maximum target-position error of 3.15e-8 m. This is an observation on the generated targets, not a uniform solver or workspace guarantee. General generation took 1.22 s and planar construction 16.78 s on the workstation. The compressed archive is approximately 2.07 MiB.

All six project tests passed, including deliberately injected reference-solver failure. Ruff lint/format checks passed. The archive loads without pickle, stored source fingerprints match the implementation, and split/inclusion counts were independently checked.

When combining archives for training, preserve their stored train/validation/test memberships. Episode IDs are local to each archive: identify an episode by (archive, episode_id), rather than treating matching local IDs as the same episode. Do not randomly reassign samples across the recorded splits.
