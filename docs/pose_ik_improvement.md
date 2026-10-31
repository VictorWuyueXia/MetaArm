# Larger-workspace six-coordinate pose IK

## Result

Neural initialization substantially reduced pose-query latency. An explicit three-chart coefficient atlas recovered the nine 48 cm lattice cases missed by a single learned chart, matching the tested full-joint coverage. Increasing model capacity and adding boundary labels improved seed prediction but did not, by themselves, expand single-chart coverage.

The main speed benefit comes from the learned initial guess. Full-joint IK given the same guess is nearly as fast and remains a strong practical choice. The atlas is useful as a six-coordinate branch-search experiment; it is not evidence that six coordinates alone outperform seven.

## Scope and protocol

The independently fitted 32, 40, and 48 cm cubes retain center (0.5,0,0.5) m and the original componentwise +/-10 degree rotation-vector range. Both translation and orientation are constrained. Strict success remains <=1 mm translation, <=1 degree rotation, and joint-limit feasibility.

The original data have 96/24/48 training/validation/test reference groups with 33 waypoints each. All individually task-feasible labels are used, including labels from incomplete teacher paths. Teacher minimum-distance optimizer statuses remain recorded; a status failure is not a proof of physical unreachability. The new training uses the scoped pose archives, preserving every original training label; broad-region position/motion archives are not mixed into a pose-specific fit.

Boundary augmentation adds 120 independently sampled 33-pose episodes per size, using existing training and validation references only: alternating face and edge requests, random orientations, seed 20261031, and the existing exact full-joint SLSQP teacher. The first pose of each episode remains its reference. Raw failed labels are retained. Labels enter fitting only if translation and rotation error are each below 1e-7 and joint margin is positive. There are 288 episodes / 9,504 requested waypoint records per augmented archive. Original metadata's reference counts describe the parent dataset; the augmentation entry and actual split arrays describe the added groups.

There is no weight sharing between cube sizes. Each fit uses seed zero, full-batch AdamW (learning rate 0.001, weight decay 0.00001), cosine scheduling to 2% of the initial rate, and validation-only checkpoint selection. Width and training budget vary independently. Historical test cases informed mechanism changes and therefore are an adaptive benchmark, not an untouched final confirmation set.

A separate audit uses 32 new joint references per size, generated within +/-0.025 rad of the center configuration, and 512 fresh pose targets, half on an x face and half uniform in volume; orientations are uniform in the original rotation-vector box. Seed 20261101. Those references/targets never enter fitting or checkpoint selection. Methods within each size receive identical inputs; fresh targets are not paired between sizes.

## Model and solver

The old network predicted 42 correction entries for a six-column chart. The new two-hidden-layer tanh network takes seven normalized reference joints and six scaled pose-displacement coordinates. It predicts six null-graph entries and six tangent-seed corrections, with a fixed graph amplitude of four. A target-conditioned chart is a different policy from one chart frozen across an entire trajectory.

For a regular reference Jacobian, G=J^T(JJ^T)^-1 and z is a unit null vector obtained by projecting the third joint axis. F=G+z u^T satisfies JF=I. Cholesky orthonormalization gives B. The predicted seed belongs to q_ref + range(B); an explicitly defined feasible ray fraction constructs its strictly joint-feasible initial coefficients. A vanishing null gauge or failed factorization raises an error.

The seed loss is joint MSE plus a projection MSE and a small graph penalty. Initial trials use projection weight 0.25; boundary-focused trials increase it to four. Validation ranking uses seed MSE, never test coverage. These are joint-displacement prediction errors, not the previous study's trajectory covariance reconstruction errors.

The new solver retains the exact fourteen coupled joint inequalities, a 0.25 rad joint-step trust radius, and the 100-iteration budget. It uses gain-ratio LM damping and omits unused candidate Jacobians. Full-joint controls use the same solver. No tolerance relaxation, output clipping, or seven-joint rescue is used in coefficient search.

Three fixed-chart policies were checked: zero initialization, the learned seed, and a chart rebuilt around that seed. Rebuilding did not improve coverage and increased slow-case work. It is not the selected policy.

The atlas always launches three six-coordinate searches together. Their origins are the learned joint seed plus offsets (-0.9,0,+0.9) rad along the reference unit null vector, scaled by an explicit feasible ray fraction. Each branch retains the same learned B throughout. Once one branch reaches both strict tolerances, its peers stop with status 3 (superseded), without being marked successful. Selection minimizes the maximum normalized translation/rotation error. Stalled and exhausted branches retain distinct statuses. There is no failure-triggered retry.

## Eighteen independent training runs

The first three trials per cube use widths/budgets (32,600), (64,1200), and (128,1800). Boundary-supervised trials use (64,1600), (128,4000), and (256,6000). All six were trained from scratch. The following joint seed MSE values are rad^2 per joint; the validation distributions differ between the first and boundary series, so cross-series numbers are not a controlled data-only ablation.

| Cube | Original labels: width 32 / 64 / 128 | Boundary labels: width 64 / 128 / 256 |
| --- | --- | --- |
| 32 cm | 0.000249 / 0.000103 / 8.78e-05 | 0.000136 / 5.01e-05 / 3.2e-05 |
| 40 cm | 0.000432 / 0.000182 / 0.00012 | 0.000236 / 9.84e-05 / 7.21e-05 |
| 48 cm | 0.000754 / 0.000401 / 0.000223 | 0.000324 / 0.000157 / 9.87e-05 |

The selected boundary model has width 256 for every cube, selected independently. All parameter tensors, complete histories, best epochs, fitting times, and label counts are saved. This is a finite search over one model family and one training seed, not a global architecture optimum.

## Coverage

Each historical lattice contains 6,561 independent pose requests: 729 Cartesian points from nine reference charts. It assigns orientations rather than exhausting a six-dimensional grid. Independent waypoint evaluation has 1,584 requests per cube and must not be equated with the older sequential trajectory-completion metric.

| Cube | Old fixed six-coordinate chart | New single chart | Three-chart atlas | Full-joint IK |
| --- | ---: | ---: | ---: | ---: |
| 32 cm | 6138/6,561 | 6138/6,561 | 6138/6,561 | 6138/6,561 |
| 40 cm | 5652/6,561 | 5652/6,561 | 5652/6,561 | 5652/6,561 |
| 48 cm | 5166/6,561 | 5166/6,561 | 5175/6,561 | 5175/6,561 |

At 48 cm, all nine recovered full-joint-only cases use the positive null-offset branch. They share one difficult target pose across the nine reference charts. Lattice success increases from 78.738% to 78.875%, or 0.137 percentage points; the atlas reaches all targets solved by the full-joint baseline in this lattice. No previously successful fixed-chart lattice request is lost. The remaining 1,386 failures are not certified unreachable.

Fresh-audit successes out of 512:

| Cube | Old fixed chart | New single chart | Atlas | Full-joint from reference | Full-joint with neural seed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 cm | 460 | 460 | 460 | 460 | 460 |
| 40 cm | 406 | 406 | 406 | 405 | 406 |
| 48 cm | 355 | 355 | 355 | 355 | 355 |

No broad coverage increase appears in this fresh sample. At 40 cm, full-joint gain-ratio LM from the reference misses one case recovered by the other listed policies. This is evidence of solver/initialization sensitivity, not physical infeasibility.

## Query time and search cost

Timings use RTX 4070 CUDA, matched targets with independently feasible teacher labels, sixteen single queries with three interleaved repetitions, and 256-query batches with five repetitions. Each query has an untimed warm pass. All policies solve every timed request. Timed end-to-end queries include chart preparation, solver validation, dispatch, and synchronized search; search-only timers exclude preparation. Startup, checkpoint loading, data transfer from external applications, and physical execution are excluded. Raw case/repeat measurements and quartiles are retained.

Median single-query end-to-end time, milliseconds:

| Cube | Original full-joint | Improved full-joint | Full-joint + neural seed | Six coordinates + seed | Cooperative atlas |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 cm | 14.88 | 14.22 | 8.43 | 8.39 | 8.70 |
| 40 cm | 17.33 | 17.02 | 8.38 | 8.37 | 8.51 |
| 48 cm | 15.58 | 15.04 | 8.31 | 8.17 | 8.39 |

The atlas is about 1.6-2.0x faster than improved full-joint search from the reference on this feasible timing set, including preparation. Same-seed full-joint search is close to the single-chart and atlas results. Consequently the data support neural initialization as the principal latency improvement; they do not isolate a substantial advantage from deleting one continuous variable.

Median 256-query end-to-end batch time, milliseconds:

| Cube | Improved full-joint | Full-joint + neural seed | Six coordinates + seed | Cooperative atlas |
| --- | ---: | ---: | ---: | ---: |
| 32 cm | 25.27 | 8.21 | 8.28 | 9.22 |
| 40 cm | 27.92 | 13.43 | 12.10 | 13.59 |
| 48 cm | 34.94 | 8.36 | 8.02 | 8.70 |

These feasible-query gains do not apply uniformly to failure-heavy batches. The 48 cm lattice contains many unsolved requests; the final full-joint gain solver takes about 1.05 s, the single neural chart about 0.94 s, and the cooperative atlas about 2.8 s. The atlas pays for three difficult branches when none succeeds. Running branches to completion without cooperative stopping was also tested; its 48 cm feasible batch took 227 ms before stopping was introduced, versus roughly 6 ms search-only afterward. That implementation comparison is illustrative, not an interleaved controlled timing estimate.

A single six-coordinate chart reduces continuous variables 7 to 6 (14.29%) and dense normal-system entries 49 to 36 (26.53%), while retaining fourteen inequalities. The atlas adds three discrete choices and up to eighteen coefficient entries across its three parallel branches. Its feasible set is a union of three polytopes, not one globally six-dimensional representation of the whole robot workspace. No cross-dimensional feasible-volume percentage is claimed.

## Interpretation and singularities

The new seed reduces typical feasible-query iterations from about three to about one. Improving seed accuracy is more effective here than merely enlarging the old fixed reference chart. Boundary supervision reduces prediction error but cannot independently remove all affine-chart or physical task limitations. The positive null-offset branch recovers the remaining nine lattice cases, supporting the explanation that a different redundancy branch is needed near the joint-six boundary.

The [analytical singularity derivation](fr3_pose_singularities.md) proves a rank-five family of the full 6x7 FR3 Jacobian: q3=q5=0 and q4=-0.4670024236530116 rad. A strictly joint-feasible witness satisfies the specified orientation range and lies inside the 24 cm and all larger cubes. This establishes existence of a singular admissible joint preimage; it does not show that every inverse branch is singular or that all shared failures are caused by this family. The 16 cm cube has no absence certificate from this analysis.

At a regular reference, J_ref B is nonsingular by construction. Away from it, J(q)B can become singular even when the full J(q) remains regular. At a full-robot singularity every reduced chart loses rank. These are distinct statements.

For practical use on this evidence, the single neural chart offers fast six-coordinate search, the cooperative atlas trades additional branch work for matching historical full-joint coverage, and neural-seeded full-joint IK provides nearly the same feasible-query latency with fewer branch-management costs. Failure-heavy workloads favor the latter over the three-chart atlas. There is no single measured policy that wins every workload.

## Numerical audit and limits

Independent double-precision CPU MuJoCo agrees with every single-neural-chart, full-joint, fresh-audit, and cooperative-atlas classification in the final records. The fixed-chart gain-ratio baseline has one 32 cm lattice boundary disagreement: CPU translation error exceeds 1 mm by 45.6 nm. Its GPU flag is retained and disclosed; tolerances are not changed. CPU and GPU Jacobians differ at roundoff because GPU FK is float32.

The analytical rank proof uses exact geometry; sampled singular values alone do not certify continuous regularity. Results establish neither global reachability/infeasibility nor global optimality, collision avoidance, trajectory continuity across branch choices, dynamic feasibility, or hardware safety. Independent waypoint success is not a trajectory-tracking experiment. Timing repetitions describe this machine/workload, not independent model seeds or a guaranteed speedup.

## Implementation and reproducibility

The [implementation contract](pose_improvement_plan.md) lists the exact one-class/five-function/three-method source budget. Four compact modules implement the chart, training, solver, and evaluation; `CoefficientIK` directly loads the new twelve-output checkpoints using the same physical-asset checks as previous architectures. Its existing defaults and the simulation/UI startup remain available.

- `src/metaarm/pose_model.py`: rank-preserving target-conditioned chart and supervised objective.
- `src/metaarm/pose_training.py`: regional fit, complete training histories, and boundary data generation.
- `src/metaarm/pose_solver.py`: fixed-chart gain-ratio search, including explicit `(N,3,...)` cooperative atlas tensors.
- `src/metaarm/pose_study.py`: matched independent-query comparisons and singularity witness report.
- `tests/test_pose_improvement.py`: right-inverse/null identities, orthonormality, gradients, feasible initialization, direct checkpoint loading, full/reduced solver checks, cooperative statuses, and singular-family tests.

Completed evidence lives in `results/pose_improvement_20261031T205242Z/`. Each `sideXX` directory contains the original and boundary checkpoints/history, boundary dataset, validation selections, final comparison records, fresh audit, atlas candidates and selected branch identities, independent CPU audits, and raw latency repetitions. `training_trials.csv`, `coverage.csv`, and `latency.csv` are compact exports; `manifest.json` pins final source/checkpoint/data hashes. The source snapshots preserve the initial and final implementations.

From the project root, rerun the completed checkpoint comparisons with:

```powershell
uv run --locked python results/pose_improvement_20261031T205242Z/fresh_audit.py
uv run --locked python results/pose_improvement_20261031T205242Z/atlas_audit.py
uv run --locked python results/pose_improvement_20261031T205242Z/cpu_atlas_audit.py
uv run --locked python results/pose_improvement_20261031T205242Z/latency_audit.py
```

These scripts explicitly write their audit outputs. Training refuses to overwrite an existing checkpoint; boundary generation refuses an existing archive. `fit_pose(source, destination, width, epochs)` supports new independent fits with the fixed declared recipe. The saved initial source snapshot documents the first-series loss weight; the final trainer uses projection weight four.

All 23 project tests pass. Ruff lint and formatting checks pass.
