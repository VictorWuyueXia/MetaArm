# Position basis for a local three-dimensional workspace

A learned three-column position basis was generated, trained, selected on validation, and tested entirely within the declared FR3 Cartesian volume. The selected model reached every sampled held-out trajectory and dense-grid target at the existing 1 mm position tolerance, with free tool orientation.

## Declared region and task

| Coordinate | Minimum | Maximum |
| --- | ---: | ---: |
| x | 0.42 m | 0.58 m |
| y | -0.08 m | 0.08 m |
| z | 0.42 m | 0.58 m |

This is a 16 cm cube centered at (0.50, 0, 0.50) m, expressed in fr3_link0/world axes at attachment_site. All reference positions and generated task targets belong to this box. Three coefficients control position; orientation is not constrained.

Independent full-joint center/corner probes were completed before data generation. All eight corners converged, with position error below 3.3e-13 m, joint margin above 0.84 rad, and position-Jacobian minimum singular value above 0.23. These samples motivated this comfortably reachable region; they do not certify every point in the cube.

## Reproduce the complete pipeline

~~~powershell
uv run --locked python -m metaarm.workspace
~~~

This command writes a new results/workspace_<UTC>/ directory containing the regional NPZ dataset, four trial checkpoints/histories, validation records, the selected checkpoint, and final trajectory/grid test records.

The existing broad data/ corpus and its models remain preserved. This regional fit uses only its scoped motions.npz; samples from other workspaces are not mixed into the fit. The cube, dataset sizes, seeds, optimizer recipe, and trial specifications are explicit source constants, with no new CLI/configuration options.

## Data and full-joint labeling

References are sampled inside the box using fixed seed 60431. There are 96 training, 24 validation, and 48 test references, with disjoint complete reference groups. Each episode follows a closed 33-point path through three independently sampled interior vertices. Quintic interpolation of segments inside the convex box keeps every path target inside the box.

The teacher solves seven joint variables independently for each unique target:

~~~
minimize_q  0.5 * ||q - q_center||²
subject to p(q) = p_target
           q_min <= q <= q_max
~~~

q_center is the full-joint solution at the declared Cartesian center. Every solve starts from that same configuration, establishing a consistent local minimum-motion convention. The teacher does not use a learned basis or restrict its search to three joint directions. Its demonstrations form a curved position-dependent section: their joint displacements have rank greater than three in the numerical test.

Exact repeated coordinates are deduplicated for labeling and restored with their original path multiplicities for training. There are 5,376 unique regional positions and 5,544 planned waypoint records. Full-joint labeling took 3.33 seconds, excluding center/corner setup and archive serialization.

Two teacher labels reached SLSQP's 100-iteration limit even though their Cartesian residuals were about 9e-13 m. Their minimum-distance optimization was therefore not declared converged. The two complete paths remain recorded and excluded from reconstruction labels. They were not resampled or replaced.

| Split | Planned episodes | Included label episodes | Included waypoint records |
| --- | ---: | ---: | ---: |
| Training | 96 | 95 | 3,135 |
| Validation | 24 | 24 | 792 |
| Test | 48 | 47 | 1,551 |

The excluded episodes are IDs 55 (training) and 150 (test). Coefficient-IK evaluation still uses all 48 test target trajectories, including ID 150; reconstruction MSE uses the 47 converged test label paths. All included teacher FK positions stay inside the declared box.

## Model and training

The regional TaskBasis uses a 7 → 32 → 32 → 21 tanh network, physical joint-limit input normalization, and 2,005 trainable parameters. It predicts a 7×3 matrix U. At each reference:

~~~
J_position.T = V H
G = solve_triangular(H, V.T).T
F = G + s * (U - V @ (V.T @ U))
B = qr(F).Q
q(alpha) = q_ref + B alpha
~~~

The selected correction scale is s = 0.25. As in the original formulation, J_position F = I in exact arithmetic at regular references. The basis and reference remain fixed throughout each trajectory or independent target solve.

Every included regional training waypoint contributes through its exact uncentered second moment. The objective is the per-joint displacement reconstruction MSE. There is no plane-normal conditioning, orientation head, coefficient network, or synthetic three-basis teacher.

Each trial uses seed 0, 500 full-batch AdamW epochs, learning rate 1e-3, weight decay 1e-4, and cosine decay to 2e-5. Weights are selected by validation reconstruction MSE. Models are then selected by validation trajectory completion first and reconstruction MSE second. The final test is evaluated after this selection.

| Width | Scale | Validation trajectory completion | Validation MSE (rad²/joint) |
| --- | ---: | ---: | ---: |
| **32** | **0.25** | **24/24** | **1.01855e-6** |
| 32 | 1 | 24/24 | 1.07174e-6 |
| 32 | 4 | 24/24 | 1.23157e-6 |
| 64 | 1 | 24/24 | 1.11753e-6 |

The selected epoch is 120. Its fitting time was 1.82 seconds. Initial validation MSE was 6.32850e-6; selected validation MSE was 1.01855e-6. Epoch 500 reached 1.09254e-6, so the earlier validation-selected weights were retained. All four histories and final-epoch weights are preserved.

## Held-out results and baselines

All three methods use the same CUDA constrained coefficient optimizer and 1 mm task tolerance. The learned and geometric charts have three coefficients; full joints use B = I_7. The geometric baseline is the orthonormalized position-Jacobian right inverse with no learned correction.

The trajectory test holds each reference/basis fixed over all 33 waypoints and warm-starts coefficients. The independent grid uses 9×9×9 target coordinates spanning the whole cube, including boundary faces/corners. It solves that grid from the center chart and eight predetermined held-out reference charts, with zero initial coefficients for every target: 6,561 independent problems per method.

| Method | Completed test trajectories | Successful trajectory waypoints | Successful independent grid solves |
| --- | ---: | ---: | ---: |
| Learned | 48/48 | 1,584/1,584 | 6,561/6,561 |
| Geometric | 48/48 | 1,584/1,584 | 6,561/6,561 |
| Full joints | 48/48 | 1,584/1,584 | 6,561/6,561 |

| Method / scope | Median / 95th / maximum position error | Minimum joint margin | Minimum sampled task-image singular value |
| --- | --- | ---: | ---: |
| Learned trajectories | 0.102 / 0.786 / 0.993 mm | 0.8237 rad | 0.2391 |
| Learned grid | 0.073 / 0.739 / 1.000 mm | 0.8123 rad | 0.2334 |
| Geometric grid | 0.073 / 0.739 / 0.999 mm | 0.8128 rad | 0.2333 |
| Full-joint grid | 0.038 / 0.810 / 1.000 mm | 0.4366 rad | 0.2175 |

The learned test reconstruction MSE is 1.00754e-6 rad²/joint, compared with 1.22881e-6 for the geometric basis: an 18.0% reduction. Both already achieve 100% sampled IK success, so this experiment does not establish a learned IK-success advantage over the geometric construction.

Synchronized GPU solve totals were 0.253/0.308/0.239 seconds for learned/geometric/full trajectories, and 0.0176/0.0137/0.0335 seconds for the 6,561-problem grid batches. These are batch throughput measurements after warm-up, excluding dataset generation, training, checkpoint loading, CPU audit, and serialization. They are not single-target latency measurements or repeated-run speedup estimates.

## Exported fixed center chart

Besides the reference-conditioned network, center_chart.npz and center_chart.csv provide one concrete fixed basis for the entire declared cube. The very same center chart achieved 729/729 grid targets. Its exact matrix and reference were regenerated in the original batch shape and reproduced every saved grid reconstruction with zero observed discrepancy.

| Joint | Reference q (rad) | B1 | B2 | B3 |
| --- | ---: | ---: | ---: | ---: |
| fr3_joint1 | 0.0000000 | -0.0002747 | 0.6949172 | -0.0002518 |
| fr3_joint2 | -0.0373896 | -0.5801173 | 0.0003850 | -0.7585804 |
| fr3_joint3 | -0.0000000 | 0.0001985 | 0.6999240 | -0.0000277 |
| fr3_joint4 | -1.8665931 | -0.6311279 | 0.0005432 | 0.6488725 |
| fr3_joint5 | -0.0000000 | 0.0003079 | 0.1649098 | 0.0012095 |
| fr3_joint6 | 1.4115694 | -0.5149188 | -0.0011013 | 0.0593194 |
| fr3_joint7 | -0.7853000 | -0.0001180 | -0.0000088 | 0.0002220 |

Use the full precision arrays in center_chart.npz for computation. The table is rounded. Columns are orthonormal joint-motion directions; alpha has joint-angle units.

~~~python
import numpy as np

chart = np.load("results/workspace_20261031T115748066551Z/center_chart.npz", allow_pickle=False)
q_ref, basis = chart["q_ref"], chart["basis"]
q = q_ref + basis @ alpha
~~~

Joint bounds remain coupled constraints on q_ref + B alpha. CoefficientIK accepts this frozen basis with task="position", or loads the selected neural checkpoint and constructs a new chart at an explicit reference. Its existing public signatures and pose/plane checkpoints remain supported.

## Interpretation and verification

Restricting the task to a regular 16 cm volume and using a consistent full-joint position lifting avoids the large path-dependent nullspace drift in the earlier demonstrations. A small nullspace correction preserves a well-conditioned sampled task image. This supports a useful local three-coefficient representation for the chosen region.

The geometric basis also succeeds everywhere sampled. Learning improves reconstruction modestly, while the region and reference convention account for much of the improvement in task reliability. The earlier plane benchmark used different targets, references, and labels, so its completion rates are not a controlled before/after comparison.

Successful finite grids and sampled singular values do not prove continuous volume coverage, global uniqueness, collision avoidance, or hardware feasibility. The result applies to the declared position task and reference family, with free orientation.

All 17 tests passed in 34.77 seconds; Ruff lint and formatting passed for 21 files. One first-use cuBLAS initialization warning appeared in the existing training gradient check. The tests check regional containment, complete split identities, teacher stationarity and curved displacement span, basis rank/gradient identities, checkpoint selection/reload, and matched position-only GPU IK while retaining the earlier simulation/collection/training/IK tests.

Independent float64 NumPy reconstruction over every included regional waypoint matched the GPU covariance objective within 1.54e-19 rad². Independent CPU MuJoCo checked every attempted final test candidate: no success-label disagreements or joint-limit violations occurred, and the maximum CPU/GPU position-error difference was 2.39e-7 m. Original broad-corpus archive hashes were verified unchanged.

## Saved artifacts

The run is results/workspace_20261031T115748066551Z/:

- motions.npz: every regional target, full-joint label, failure status, reference/split identity, inclusion mask, and physical/source provenance.
- selected.pt: selected 2,005-parameter position model with model specification, selected/final states, and training history.
- center_chart.npz and center_chart.csv: the tested single center reference/basis and its 729-point success/error arrays.
- width32_scale*/ and width64_scale1/: validation summaries and attempted/failed/unattempted records for each model.
- trials.json and summary.json: selection evidence and the final held-out report.
- test/records.npz and test/summary.json: complete trajectory and dense-grid results for learned, geometric, and full-joint methods.
- verification.json and source/: independent numerical checks, preserved-corpus verification, environment fingerprints, and the evaluated source snapshot.

The [expansion study](workspace_expansion.md) extends the implementation to full pose and independently tuned configurations. TaskBasis and the explicit training recipe replace the position-only class/trainer API; saved weights and center-chart arrays remain readable.
