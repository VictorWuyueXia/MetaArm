# Workspace expansion: independently tuned position and pose bases

Ten standalone experiments expanded the cube through sides 16, 24, 32, 40, and 48 cm. Each size/task pair generated its own data and independently selected a model from six training runs: 60 trials and 40,000 epoch updates, without weight sharing between configurations.

Position uses three coefficients with free orientation. Full pose uses six coefficients with position and orientation constraints. Each task is compared with its own seven-joint optimizer; this is not a pure coefficient-count ablation.

## Controlled setup

All cubes are centered at (0.50, 0, 0.50) m in the FR3 base/world frame. Their volumes grow by factors 1, 3.375, 8, 15.625, and 27.

The same 168 safe reference configurations are used across sizes and tasks. Reference positions occupy the central 8 cm cube; reference rotation-vector coordinates lie within ±5° per axis. Target position cubes expand while starts remain fixed. Results do not cover arbitrary starts throughout the larger cubes.

Cartesian paths are paired across tasks and use the same normalized control points across sizes. Pose additionally samples body rotation-vector coordinates within ±10° per axis about the center rotation. This fixed local orientation chart is not all SO(3).

Each experiment has 96/24/48 train/validation/test reference groups and 33-point closed quintic paths. Independent full-joint labels minimize distance to the episode reference under exact task constraints and joint bounds. Targets stay inside their declared cube; failed labels and corner probes are retained without shrinking regions or resampling failures.

This reruns the 16 cm baseline under a common episode-reference teacher. Its rates are not a controlled before/after comparison with the earlier fixed-center-label study.

## Independent model searches

The first four trials test width/depth/scale 32/2/0.25, 64/2/1, 128/2/4, and 64/3/1, with 500 AdamW epochs. Each pair then refines its own validation winner with 1,000 epochs, lower rate/stronger weight decay, followed by a three-layer mobility recipe.

The mobility objective adds a sampled teacher-state singular-value deficit penalty: thresholds 0.1 for position and 0.03 for pose, weight 1e-3. This is not a uniform chart-rank certificate. Width, depth, scale, duration, rate, regularization, and objective are varied independently per experiment.

Weights are selected on validation objective. Models are ranked by validation path completion, successful planned waypoints, and reconstruction MSE. Test results do not select models. All histories and final-epoch states remain saved.

| Side | Task | Width × depth | Scale | Scheme | Epoch budget / selected | Test joint MSE |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 16 cm | position | 32 × 2 | 0.25 | reconstruction | 1000 / 400 | 9.9584e-07 |
| 16 cm | pose | 64 × 3 | 1.0 | reconstruction | 500 / 430 | 8.3293e-06 |
| 24 cm | position | 32 × 2 | 0.25 | reconstruction | 1000 / 310 | 4.9745e-06 |
| 24 cm | pose | 64 × 3 | 1.0 | reconstruction | 500 / 180 | 2.4474e-05 |
| 32 cm | position | 32 × 2 | 0.25 | reconstruction | 500 / 100 | 1.4363e-05 |
| 32 cm | pose | 64 × 3 | 1.0 | reconstruction | 500 / 130 | 6.0234e-05 |
| 40 cm | position | 32 × 3 | 0.25 | mobility | 1000 / 40 | 3.3519e-05 |
| 40 cm | pose | 64 × 3 | 1.0 | reconstruction | 500 / 120 | 0.00010014 |
| 48 cm | position | 32 × 3 | 0.25 | mobility | 1000 / 40 | 6.8685e-05 |
| 48 cm | pose | 64 × 3 | 1.0 | reconstruction | 500 / 220 | 0.00012063 |


Position favors smaller corrections and smaller models; the 40/48 cm winners use the deeper mobility recipe. Pose experiments independently choose a deeper 64-unit reconstruction model. Repeated winning architectures are selection outcomes, not forced sharing.

## Coverage and motion discrepancies

All 48 held-out target paths are evaluated, including paths excluded from reconstruction labels. References/bases remain fixed within a path; coefficients warm-start, and each method stops its own path at the first failure.

The lattice spans 729 Cartesian targets from the center plus eight held-out reference charts. Pose assigns a reproducible set of 27 orientation tuples across those locations. This is 6,561 pose problems, not an exhaustive six-dimensional grid.

| Side | Position paths | Position lattice | Pose paths | Pose lattice: learned / full |
| --- | ---: | ---: | ---: | ---: |
| 16 cm | 48/48 | 6561/6561 | 48/48 | 6561 / 6561 of 6561 |
| 24 cm | 48/48 | 6561/6561 | 48/48 | 6525 / 6525 of 6561 |
| 32 cm | 48/48 | 6561/6561 | 47/48 | 6138 / 6138 of 6561 |
| 40 cm | 48/48 | 6561/6561 | 40/48 | 5652 / 5652 of 6561 |
| 48 cm | 48/48 | 6525/6561 | 35/48 | 5166 / 5175 of 6561 |


Every learned success classification agrees with CPU FK. Two supplementary baseline classifications differ at the strict 1 mm boundary: a geometric pose result at 32 cm exceeds it by 44.5 nm, and a full-joint position waypoint at 48 cm by 14.3 nm. Float32 GPU FK accounts for these differences; raw flags and independent audits are preserved. The table uses reported path counts and learned/full lattice counts.

Except for nine pose lattice cases at 48 cm, learned charts retain every lattice success of full-joint IK. The geometric position baseline misses two additional 48 cm lattice cases reached by the learned chart.

Independent full-joint SLSQP did not recover tolerance-feasible solutions for twelve sampled shared pose failures. Reduced-space SLSQP did not recover any of the nine reduced-only failures at 48 cm, although full-joint IK reached those targets. These local diagnostics support a chart/optimizer limitation; they do not prove global infeasibility.

## Search-space reduction

| Task | Coordinates | Reduction | Dense system entries | Entry reduction | Coupled inequalities |
| --- | ---: | ---: | ---: | ---: | ---: |
| Position | 7 → 3 | 57.14% | 49 → 9 | 81.63% | 14 |
| Pose | 7 → 6 | 14.29% | 49 → 36 | 26.53% | 14 |

The feasible set remains q_min - q_ref ≤ B alpha ≤ q_max - q_ref. Coefficient bounds are coupled. Different-dimensional spaces have different volume units; no fictitious physical-volume or exhaustive-grid reduction is claimed.

## Repeated paired search timing

Timing uses independent zero-initialized problems, separate from stopped trajectories. Reduced and full methods receive identical cases. The 256 batch cases and first sixteen single cases come from held-out targets with a task-feasible full-joint teacher candidate, selected before observing learned results.

Both methods reach all timed cases in every configuration. The jointly-successful subset therefore equals the complete sample. Single-target results use three repeats; batches use five. Raw timings, iteration counts, and interquartile ranges are retained.

Speedup = full-joint median time / reduced median time. Values below one are slowdowns. Cached search excludes basis preparation. Prepare + solve includes reference FK/Jacobian geometry, inference/QR, and the solve, excluding model/workspace construction.

| Side | Task | Cached reduced / full (ms) | Cached speedup | Prepare + solve speedup | Batch speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 cm | position | 5.701 / 5.723 | 1.004× | 0.813× | 1.302× |
| 16 cm | pose | 10.407 / 9.760 | 0.938× | 0.822× | 0.989× |
| 24 cm | position | 5.509 / 6.756 | 1.226× | 0.994× | 1.591× |
| 24 cm | pose | 13.385 / 13.565 | 1.013× | 0.918× | 1.059× |
| 32 cm | position | 7.630 / 9.166 | 1.201× | 1.032× | 5.389× |
| 32 cm | pose | 16.096 / 16.826 | 1.045× | 0.954× | 1.025× |
| 40 cm | position | 6.744 / 9.524 | 1.412× | 1.114× | 8.817× |
| 40 cm | pose | 17.919 / 17.291 | 0.965× | 0.900× | 1.030× |
| 48 cm | position | 7.574 / 8.749 | 1.155× | 0.993× | 8.425× |
| 48 cm | pose | 16.083 / 14.392 | 0.895× | 0.816× | 1.017× |


At 40 cm, cached position latency falls from 9.524 to 6.744 ms: a 29.2% reduction. Including preparation leaves a 10.3% reduction. Its measured batch speedup is 8.82×, or 88.7% less batch time.

Large position batch gains also reflect shorter iteration tails. Maximum iterations for reduced/full searches change from 3/4 at 16 cm to 5/15 at 48 cm. Fixed-size GPU batches wait for the slowest active case. Batch speedups must not be interpreted as single-target or solely factorization speedups.

Pose cached times are near parity or slower; preparation-inclusive pose timing is slower at every size. A 14.3% coordinate reduction does not reliably offset preparation and similar iteration counts.

IQR describes workload/repeat variability, not model-seed uncertainty or a hardware guarantee. All measurements use the same RTX 4070 and locked computation environment.

## Figure

![Coverage and repeated search costs](../results/expansion_20261031T123139402928Z/expansion.png)

All requested lattice targets are counted. Timing marks are medians and bars are interquartile ranges across case/repeat observations. Dashed/open-triangle controls use seven joints for their corresponding task. Cached-search timing is plotted; preparation-inclusive ratios appear above.

## Label availability and interpretation

| Side | Task | Included train / validation / test paths | Task-feasible teacher labels / planned |
| --- | --- | ---: | ---: |
| 16 cm | position | 78 / 22 / 41 | 5544 / 5544 |
| 16 cm | pose | 96 / 24 / 48 | 5544 / 5544 |
| 24 cm | position | 81 / 23 / 41 | 5544 / 5544 |
| 24 cm | pose | 93 / 24 / 48 | 5535 / 5544 |
| 32 cm | position | 92 / 22 / 45 | 5544 / 5544 |
| 32 cm | pose | 81 / 24 / 47 | 5470 / 5544 |
| 40 cm | position | 91 / 22 / 46 | 5544 / 5544 |
| 40 cm | pose | 60 / 21 / 40 | 5291 / 5544 |
| 48 cm | position | 93 / 23 / 48 | 5544 / 5544 |
| 48 cm | pose | 53 / 18 / 34 | 5127 / 5544 |


A teacher can reach task tolerances while hitting the minimum-distance optimizer's iteration limit. Complete converged label paths and individually task-feasible candidates are therefore reported separately. Failed statuses are not labeled mathematically unreachable.

Reconstruction MSE is per joint in rad² and uses all waypoints of included paths through uncentered second moments. It must be interpreted with label counts and task success, since wider cubes can exclude infeasible or unconverged paths.

Increasing error is consistent with approximating larger curved motion sets by fixed affine charts. Larger networks and longer fitting did not uniformly improve validation. The largest tested cube with complete sampled position success is 40 cm. For the declared pose orientation range, only 16 cm reaches every lattice target; 24 cm completes all random paths but misses some lattice boundary targets.

These results concern the central reference family and fixed-chart policy. They establish neither continuous coverage nor global optimality, collision avoidance, dynamic feasibility, or hardware performance. Winners are best among the tested configurations.

## Reproduce and inspect

~~~powershell
uv run --locked python -m metaarm.expansion
~~~

The command creates a new study with all ten data/training/test/timing experiments. Existing data and checkpoints remain preserved.

TaskBasis now has an explicit task and hidden depth; it produces three or six columns. Supported architecture dispatch retains historical 21-output regional and 84-output broad weights. The shared trainer takes explicit model and training dictionaries. Validation is validate_experiment in expansion_evaluation.py; existing position entrypoints are updated.

Results are in results/expansion_20261031T123139402928Z/. Each sideXX_position/pose directory contains scoped motions, six checkpoints and validation records, selected.pt, test records, and timing arrays. comparison.csv and trials.csv expose all ten selections and sixty trials. summary.json contains complete reports; audit.json and failure_probes.json preserve boundary/failure diagnostics; source/ preserves the evaluated implementation.

The figure has PDF/SVG/600-dpi PNG exports, plotting source, alignment and collision/text audits. PNG is the report raster; a missing submission TIFF is intentionally inapplicable.

All 20 tests passed in 28.71 seconds. Ruff lint/format checks passed for 24 files. Tests cover pose teaching, both chart dimensions/depths, covariance/gradient identities, explicit training recipes, paired timing dimensions, checkpoint reload, and earlier simulation/data/IK behavior.

A separate [results and inferred-reasons summary](workspace_expansion_interpretation.md) distinguishes observations, explanations, and limits.
