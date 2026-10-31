# Workspace expansion: results and inferred reasons

The independently tuned experiments support a useful three-coefficient position representation over a larger sampled region than the tested six-coefficient pose representation. Search-coordinate reduction translates into a substantial position batch benefit, but does not guarantee lower single-target or preparation-inclusive time.

This document distinguishes measured results from explanations supported by those results. The [complete study](workspace_expansion.md) contains the detailed protocol, selected models, timing tables, and artifact locations.

## Experiment scope

Ten experiments tested cube sides 16, 24, 32, 40, and 48 cm around (0.50, 0, 0.50) m. Each size/task pair independently generated data and searched six model/training configurations, giving 60 training runs.

Position tasks constrained only xyz and used three coefficients. Pose tasks constrained xyz and orientation and used six coefficients. Pose rotation-vector coordinates varied within ±10° per axis about the center rotation. Each task had its own seven-joint comparison; this is not an identical-task comparison of three versus six search variables.

All experiments shared the same reference configurations, with reference positions in a central 8 cm cube and reference rotations within ±5°. Target cubes expanded while this reference family remained fixed. There were 96/24/48 training/validation/test reference groups and 33 waypoints per trajectory.

Each learned basis stayed fixed throughout a trajectory. Final lattice evaluation used 729 Cartesian positions from nine reference charts, giving 6,561 target problems per configuration. Pose assigned orientations to those positions; it did not exhaust a six-dimensional pose grid.

## Measured results

| Cube side | Position lattice success | Pose lattice success | Position cached speedup | Pose cached speedup |
| --- | ---: | ---: | ---: | ---: |
| 16 cm | 100.00% | 100.00% | 1.00× | 0.94× |
| 24 cm | 100.00% | 99.45% | 1.23× | 1.01× |
| 32 cm | 100.00% | 93.55% | 1.20× | 1.05× |
| 40 cm | 100.00% | 86.15% | 1.41× | 0.96× |
| 48 cm | 99.45% | 78.74% | 1.16× | 0.89× |


Success counts concern the learned models. Speedup is full-joint median search time divided by reduced median search time, for matched warmed single-target cases with the basis already prepared. Values below one indicate a slowdown.

Position completed all 48 test trajectories at every size. Its lattice remained fully successful through 40 cm and reached 6,525/6,561 targets at 48 cm.

Pose trajectory completion was 48/48, 48/48, 47/48, 40/48, and 35/48 as side length increased. Its lattice counts were 6,561, 6,525, 6,138, 5,652, and 5,166.

Full-joint lattice success matched learned success except at 48 cm pose, where full-joint IK reached 5,175 targets, nine more than the learned chart. Thus most pose degradation also appeared in the unrestricted seven-joint comparison.

### Reconstruction discrepancy

The following values are mean squared joint reconstruction errors in rad² per joint, evaluated on converged held-out teacher paths:

| Cube side | Position joint MSE | Pose joint MSE |
| --- | ---: | ---: |
| 16 cm | 9.9584e-07 | 8.3293e-06 |
| 24 cm | 4.9745e-06 | 2.4474e-05 |
| 32 cm | 1.4363e-05 | 6.0234e-05 |
| 40 cm | 3.3519e-05 | 0.00010014 |
| 48 cm | 6.8685e-05 | 0.00012063 |


The best position models used width 32 and scale 0.25, with two hidden layers at 16–32 cm and three layers at 40–48 cm. The larger position winners used the tested mobility recipe. Pose models independently selected width 64, three hidden layers, scale 1, and the reconstruction objective. Larger networks and longer training were tested but did not consistently improve validation.

### Search-space and timing reduction

Position reduces independent variables from seven to three: 57.14%. Pose reduces them from seven to six: 14.29%. Dense system entries decrease from 49 to 9 or 36, respectively, while all fourteen coupled joint inequalities remain.

These are coordinate and matrix-size reductions. Volumes in different-dimensional spaces are not directly comparable, so no percentage reduction in physical feasible volume is inferred.

At 40 cm, position cached single-target time decreased from 9.524 to 6.744 ms, or 29.2%. Including reference geometry and basis preparation reduced the gain to 10.3%. The 256-target batch was 8.82× faster, corresponding to 88.7% less batch time.

Pose cached timing was near parity or slower. Including basis preparation made pose slower than full-joint search at every tested size.

## Inferred reasons

### 1. Orientation constraints reduce the usable target region

**Measured evidence:** pose success declines with cube size in both learned and full-joint IK. Independent full-joint SLSQP probes did not recover tolerance-feasible solutions for twelve sampled shared pose failures.

**Inference:** maintaining the specified orientation leaves less freedom to accommodate extreme Cartesian targets and joint limits. A significant part of the observed pose deterioration is therefore a task/kinematic or solver limitation shared with full-joint IK, rather than simply inadequate neural training.

**Limit:** failed local optimizers do not prove global unreachability. The study does not separate every physical infeasibility from every initialization or optimization failure.

### 2. Larger excursions make one fixed affine chart less accurate

**Measured evidence:** joint reconstruction MSE increases substantially with workspace size. At 48 cm pose, full-joint IK succeeds on nine lattice cases missed by the learned chart; reduced-space SLSQP refinement did not recover those cases from the failed candidates.

**Inference:** the nonlinear joint-motion set becomes harder to represent by one fixed affine subspace over larger excursions. Increasing network capacity changes the chosen chart but does not make that chart nonlinear in its coefficients. The extra nine failures identify a reduced-chart/optimizer limitation for targets known feasible through full-joint solutions.

**Limit:** these diagnostics do not prove that every possible learned chart would fail, or that no other initialization could recover a root inside the same chart.

### 3. Position benefits from a much larger coordinate reduction

**Measured evidence:** position removes four of seven optimization variables; pose removes only one. Position batch maxima change from three versus four iterations at 16 cm to five versus fifteen at 48 cm for reduced/full methods. Pose iteration counts remain similar between methods.

**Inference:** the position formulation combines smaller linear systems with shorter iteration tails. Six pose coefficients provide a smaller computational saving, while orientation residuals and their derivatives still need evaluation.

**Limit:** measured time is implementation- and workload-dependent. Coordinate counts alone do not predict runtime.

### 4. Fixed-size batching amplifies slow-case differences

**Measured evidence:** the position batch gain is much larger than its single-target gain, despite both timed methods successfully solving all 256 batch cases.

**Inference:** a batch runs until its slowest active case finishes. Fewer difficult-case iterations reduce work for the entire fixed-size batch, amplifying the effect of smaller systems.

**Limit:** the 8.82× batch result is not an 8.82× single-query latency improvement and cannot be attributed solely to factorization cost.

### 5. Basis preparation can consume the saved search time

**Measured evidence:** preparation-inclusive speedups are consistently lower than cached-search speedups. Pose becomes slower at every tested size once preparation is included.

**Inference:** reference FK/Jacobian evaluation, geometric factorization, neural inference, and QR have a fixed cost. That cost is easier to amortize when one chart serves a trajectory or many targets. For isolated pose queries, removing one search variable does not reliably compensate.

**Limit:** the measurements separate preparation from model/workspace construction. They do not include application startup or physical tracking.

### 6. Training capacity is not the only limiting factor

**Measured evidence:** independent searches varied width, depth, correction scale, learning rate, weight decay, duration, and objective. Larger or longer-trained models did not uniformly improve validation, and most task failures also occurred in full-joint IK.

**Inference:** simply enlarging the network is unlikely to resolve shared task limits or the fixed affine-chart restriction. The useful training objective must balance representation accuracy with actual coefficient-IK behavior.

**Limit:** six trials per configuration do not establish a global hyperparameter optimum. Only one sampling seed and a limited family of architectures and objectives were studied.

## Data and numerical qualifications

Teacher labels were generated by independent full-joint minimum-distance solves. Expected optimizer failures were retained, and complete unconverged paths were excluded from reconstruction training. Some teacher solves met task tolerances while failing the minimum-distance convergence test; those statuses are not labeled unreachable.

Reconstruction error consequently describes the retained label subset. Task evaluation includes every requested test trajectory and lattice target, preventing those exclusions from concealing task failures.

All learned success classifications agreed with independent CPU FK. Two supplementary baseline classifications differ at the strict 1 mm boundary by only 44.5 nm and 14.3 nm because the GPU FK is float32. These cases remain flagged in the audit.

Timing used matched targets with a known task-feasible teacher candidate. Both methods reached every timed case: sixteen single-target cases with three repeats and 256-case batches with five repeats. Reported variability describes cases and repeats, not independent model seeds or guaranteed hardware performance.

## Practical conclusion

For this reference family, fixed-chart policy, and sampled targets, 40 cm is the largest tested cube with complete learned position success. The 16 cm pose cube is the only tested pose volume with complete lattice success; 24 cm still completes all random test paths but misses some lattice boundary cases.

Three-coefficient position IK is the strongest observed reduction when charts can be reused. Six-coefficient pose IK provides motion compression but no consistent timing advantage over seven-joint IK in this implementation.

These results are sampled evidence. They do not establish continuous coverage, global optimality, collision avoidance, dynamic feasibility, or hardware performance.

## Evidence files

The study directory is results/expansion_20261031T123139402928Z/:

- comparison.csv: ten selected configurations, reconstruction errors, coverage, and timing ratios.
- trials.csv: all sixty independent training trials and validation selections.
- trajectory_metrics.csv and grid_metrics.csv: task errors, rotation discrepancies, joint margins, conditioning, and iteration counts.
- audit.json: paired-reference and numerical-boundary checks.
- failure_probes.json: independent full/reduced SLSQP diagnostics.
- Each configuration directory: dataset, checkpoints, validation/test records, and raw repeated timings.
- expansion.png, expansion.pdf, and expansion.svg: coverage/timing figure with source and QA records.

All 20 project tests passed; lint and formatting checks passed.
