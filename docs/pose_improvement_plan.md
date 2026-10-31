# Pose IK improvement implementation contract

The experiment targets 32, 40, and 48 cm pose cubes independently. Existing datasets, splits, baseline solvers, joint inequalities, and 1 mm / 1 degree success criteria remain explicit. Target-conditioned charts are a changed policy, rather than a claim about the old trajectory-fixed basis. Full-joint IK receives the same learned seed as a control. No seven-joint rescue is hidden inside reduced search.

## Structure budget

Four new task-oriented modules, one class, five top-level functions, and three methods:

- `pose_model.py`: `pose_geometry(jacobian)` forms the right inverse G and a continuous unit null vector z; `PoseChart.__init__(limits, width, side)` defines a two-hidden-layer tanh network; `PoseChart.forward(q_ref, geometry, displacement)` returns an orthonormal six-column chart and its predicted initial coefficients; `PoseChart.loss(...)` fits joint displacement. Independent variables are the seven reference joints and six target displacement coordinates. The network predicts six null-graph entries and six tangent corrections. Width and cube side are the only architecture/experiment inputs.
- `pose_training.py`: `fit_pose(source, destination, width, epochs)` fits all individually task-feasible training waypoints, selects validation joint MSE, and saves checkpoint/history. Frozen AdamW learning rate 0.001, weight decay 0.00001, full-batch training, cosine schedule, seed zero. Trials vary width and epoch budget independently for each cube. No cross-split labels enter fitting.
- `pose_solver.py`: `solve_pose(solver, q_ref, basis, target, alpha_initial)` uses gain-ratio damped Gauss-Newton with the existing exact coupled-inequality QP. Trust radius and stopping constants stay fixed. Existing `CoefficientIK.geometry` and `residual` gain a derivative-computation flag to omit unused candidate Jacobians; existing defaults preserve behavior. Existing solver remains the baseline.
- `pose_study.py`: `evaluate_pose(source, checkpoint, destination)` evaluates matched held-out pose requests and records success, iterations, strict errors, search time, and preparation-inclusive time. Module entrypoint runs independent trials and selects only on validation reconstruction. `singularity_report(destination)` documents the exact elbow-family rank certificate and verifies workspace witnesses with CPU MuJoCo. These are the fourth and fifth top-level functions. Numerical optimizer callbacks, if necessary, are local mathematical objectives only.

Existing classes gain no new methods. The network is loaded directly by the study, so no checkpoint inference or fallback architecture is added. Tests cover chart identities, solver correctness against full-joint IK, and singular-family rank.

## Mathematical and evaluation contract

At a regular reference, G = J^T(JJ^T)^-1 and z is the normalized projection of the third joint axis into ker J. Reject a vanishing projection. F = G + z u^T satisfies JF = I. Cholesky orthonormalization gives B. A target-conditioned tangent prediction gives the initial coefficients. An explicit feasible ray fraction scales the seed into the coupled joint polytope; it is an initialization construction, never output clipping.

Training minimizes joint displacement MSE plus 0.25 times projection MSE and 0.00001 times graph magnitude. Pose displacement uses base-frame translation and base-frame reference-to-target rotation log. Rotation range is the existing +/-10 degrees per rotation-vector component.

Compare historical fixed-chart search, improved solver with the fixed chart, conditional chart with zero and learned initialization, and full-joint search from zero and the same learned joint seed. Timing includes fixed warmup, repeated matched queries, and separate chart preparation. Report target-conditioned policy, sampled nature of coverage, and any timing or success regressions without reinterpretation. Use existing test targets only for final evaluation, not hyperparameter choice.

Analytical singularity means rank loss of the complete 6x7 geometric Jacobian, distinct from rank loss of J(q)B. For the exact FR3 geometry, derive the planar q3=q5=0 elbow family and verify its joint and task bounds. Numerical absence of small singular values does not prove continuous-workspace regularity.

## Coding discipline

Use compact continuous mathematical workflows, professional one-line block comments, no progress/version labels, no unnecessary wrappers or configuration, no try/with/except, and no fallback values or behavior. Prefer batched matrix operations. Keep files approximately 50-250 lines and functions approximately 20-100 lines; the evaluation workflow may be split only by the declared task boundaries. Do not exceed this structure budget.

## Boundary-focused iteration

The first nine trials preserved coverage but missed the same nine full-joint-only 48 cm lattice cases. They approached joint six's lower bound near the elbow singular family. The second iteration adds independently sampled face/edge-biased requests using only the existing training and validation references (seed 20261031). Existing teacher labels and splits remain present. New task labels use the existing exact full-joint minimum-distance teacher; failed labels remain recorded and excluded only by explicit physical error/margin criteria. The module-level workflow in `pose_training.py` performs generation without adding classes, methods, helpers, or parameters. Three additional widths/budgets per workspace use the enlarged dataset; projection weight becomes 4 to emphasize chart containment. Old checkpoint inference stays unchanged. Select only by validation seed MSE, and report first versus second iteration results separately.

For latency, add a distinct module-level audit using the saved matched timing records: sixteen queries, three interleaved repeats, per-query basis preparation measured separately, plus the existing 256-query batch. Save the audit script alongside raw records; it introduces no reusable source API. Derivative omission and gain-ratio LM are tested separately from neural initialization.

## Seed-centered chart iteration

A final explicit policy centers the six-coordinate affine search at the learned feasible joint seed, and constructs its rank-preserving chart from the Jacobian at that seed. The same learned null-graph coefficients are retained. This tests whether reference-to-target chart curvature causes the remaining near-bound failures; it is a changed chart origin, never an unrestricted rescue or an extra optimization variable. Add a named `conditional_rebase` comparison to the existing evaluation workflow and fresh audit. The class/function/method/parameter budget is unchanged. Record its extra geometry/factorization time separately, and judge coverage on both the historical lattice and the fresh audit.

## Explicit parallel chart atlas

Boundary supervision and seed-centered charts do not remove the remaining lattice gap. Test a fixed three-chart atlas with origins at the learned feasible seed and at +/-0.9 rad along the reference unit null direction. All three branches always run together; no failure-triggered retry, full-joint rescue, or change to success tolerances is permitted. Each branch keeps the same learned six-column basis and exact coupled constraints. Feasible ray fractions construct shifted initial origins. Select the branch minimizing the maximum normalized translation/rotation error. Count three discrete charts and three sets of six coordinates; do not call this a global seven-to-six variable reduction. Measure branch-inclusive single-query latency and batch time. The study's artifact script implements the declared batched experiment without introducing source APIs or exceeding the structure budget.

## Cooperative atlas termination

`solve_pose` additionally accepts an explicit three-chart axis in its input tensors: origins `(N,3,7)`, bases `(N,3,7,6)`, coefficients `(N,3,6)`, and matching targets. This is a tensor-shape contract, not a new CLI/config parameter. Flatten only for GPU kinematics. Once any branch meets both strict task tolerances, stop the other branches of that query with status 3 (superseded); stalled and exhausted branches retain their original statuses. All three branches start together. Return per-branch candidates and actual success flags; the caller's min-max normalized-error selection chooses the query result. Existing two-dimensional inputs retain one-chart semantics. No new classes, methods, functions, helpers, or signature parameters are added.

## Runtime checkpoint loading

Use the existing `CoefficientIK` constructor to load an explicitly identified twelve-output `PoseChart` checkpoint, with the same physical-asset verification as existing models. The trainer includes the existing metadata envelope (`sources` and `specification.model`) alongside its detailed report; its two experiment fields remain width and cube side. Add only this architecture branch and its import to the existing constructor. Existing checkpoint behaviors and defaults remain unchanged. This adds no API, method, helper, or runtime parameter and lets the trained pose chart run without loading a historical baseline network.
