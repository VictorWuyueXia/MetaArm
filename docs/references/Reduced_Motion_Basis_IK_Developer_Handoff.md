---
title: "Reduced Motion Basis Inverse Kinematics"
subtitle: "Mathematical formulation, development specification, and experimental protocol"
---

**Status:** research and implementation specification, version 1.0  
**Target system:** seven revolute joint serial manipulator (Franka Research 3 as an optional reference platform)  
**Primary experiments:** six dimensional joint basis for general end effector pose motion; three dimensional joint basis for variable planar raster motion with unconstrained end effector attitude  
**Primary design choice:** a **constant joint space basis per evaluation episode**. Configuration dependent bases and frequent reanchoring are separate comparison methods, not silent implementation details.

---

# 1. Executive specification

We want to represent seven joint configurations by a smaller number of coordinated joint motion coefficients and solve nonlinear IK directly in those coordinates. Two primary cases are required:

- **General pose motion, `B6`:** six independent joint coordination vectors, `B6 ∈ R^(7×6)`, with a six dimensional Cartesian pose objective. The goal is to retain feasible local control of all three translations and all three rotations, and measure how much of the finite reachable pose set is preserved by a constant affine joint subspace.
- **Planar raster motion, `B3`:** three independent joint coordination vectors, `B3 ∈ R^(7×3)`, with **tip position only** as the IK objective. The end effector attitude is free. The goal is to retain three dimensional tip translation capability so a single basis can support raster planes with different orientations, and to examine efficient basis transfer between planes. Two vectors are the local theoretical minimum for one specified plane; `B3` is deliberately used for the primary experiment because it can also generate normal displacements and serves a wider family of planes.

**Explicit limits:** A full rank `B6` does **not** guarantee six Cartesian degrees of freedom, and a full rank `B3` does **not** guarantee three Cartesian translation degrees of freedom. The relevant quantity is `rank(J(q)B)`, not `rank(B)`. Full reduced Jacobian rank is a **local differential** result; it does not prove coverage of an entire pose set, compatibility with joint limits, collision avoidance, or finite raster reachability.

## Deliverables

1. A reusable kinematics and constrained IK package with identical interfaces for the full seven joint solver and reduced solvers.
2. Algorithms to construct, optimize, save, load, and evaluate constant `B6` and `B3` matrices.
3. An analytical basis transfer procedure for moving `B3` from one raster plane to another, with optional optimization of a small correction.
4. Reproducible experiments, datasets, plots, statistics, and machine readable metrics.
5. A theory note proving local rank criteria, testing sufficient regional conditions, and **clearly separating** verified claims from conjectures about global reachability.

# 2. Kinematic and task definitions

Let `n = 7` and `q ∈ R^7` denote joint angles, subject to `q_min ≤ q ≤ q_max`. Let `T(q) ∈ SE(3)` be the forward kinematics of the **actual tool tip**, including any fixed flange to tip transform, with tip position `p(q) ∈ R^3`. Use a consistent base frame and tip frame.

The geometric Jacobian is

$$
\begin{bmatrix}\dot p\\\omega\end{bmatrix}
=J_6(q)\dot q,
\qquad J_6(q)\in\mathbb R^{6\times7},
\qquad J_p(q)\in\mathbb R^{3\times7}.
$$

`J_p` is the translational block of `J6` when both are expressed in the same frame and use the same tip point. The angular block and twist ordering must be verified in the selected robotics library.

For either `k ∈ {6,3}`, define

$$
q(\alpha)=q_{\mathrm{ref}}+B_k\alpha,
\quad B_k\in\mathbb R^{7\times k},
\quad \alpha\in\mathbb R^k,
\quad B_k^T B_k=I_k.
$$

`q_ref` is fixed for an episode (a group of related targets or one raster). `B_k` is also fixed for that episode. This yields a `k` dimensional affine subspace in the seven dimensional joint space. Orthonormal columns remove arbitrary basis scaling and ease comparisons, although a change of basis within the same subspace does not change the feasible set if coefficient bounds do not interfere.

**Do not reset `q_ref` with a full dimensional IK solution at each waypoint.** That would conceal the search dimension and destroy the intended constant subspace experiment. A full seven dimensional IK is permitted to establish the initial starting configuration once per episode, with its runtime recorded separately.

## 2.1 General six dimensional pose task (`B6`)

Given a target `T_d ∈ SE(3)`, solve

$$
\begin{aligned}
\alpha^*=\underset{\alpha\in\mathbb R^6}{\operatorname{argmin}}
\quad \tfrac12\|e_6(q_{\mathrm{ref}}+B_6\alpha,T_d)\|_{W_6}^2
+\lambda_{\Delta}\|\alpha-\alpha_{\mathrm{prev}}\|^2,\\
&\text{subject to}\quad q_{\min}\le q_{\mathrm{ref}}+B_6\alpha\le q_{\max},
\quad g_{\mathrm{collision}}(q)\ge0\ \text{when collision geometry is enabled}.
\end{aligned}
$$

One valid local pose residual is `e6(q,Td) = Log(Td^{-1} T(q)) ∈ R^6`, computed with a **consistent** `SE(3)` convention. For reporting, always split it into a Euclidean tip position error in metres and a geodesic orientation error in radians; avoid reporting a raw mixed units norm. `W6` is positive definite and appropriately normalized, for example by dividing translation and rotation errors by their respective tolerances. `lambda_delta` regularizes changes from the preceding solution. It must be applied consistently across comparison solvers.

The reduced differential Jacobian is `J6(q)B6`, a `6×6` matrix. Local six DoF pose mobility requires

$$
\operatorname{rank}\big(J_6(q)B_6\big)=6.
$$

## 2.2 Planar raster task (`B3`)

Specify an orthonormal plane frame by origin `c ∈ R^3`, tangent vectors `e1,e2 ∈ R^3`, and normal `n=e1×e2`. The rectangle is

$$
\mathcal R(c,E,W,H)
=\{c+u e_1+v e_2:\ |u|\le W/2,\ |v|\le H/2\},
\qquad E=[e_1\ e_2].
$$

A raster generator specifies row spacing `h`, point spacing or sampling interval, and alternating sweep direction. This produces ordered target positions `p_d,0, …, p_d,N`. A complete path is a curve; the rectangle containing the curve has two intrinsic spatial coordinates. The experiment tests **arbitrary point targeting on the raster region and continuous waypoint tracking**, not just curve parameterization by a scalar path variable.

Use tip position only:

$$
\begin{aligned}
\alpha_i^*=\underset{\alpha\in\mathbb R^3}{\operatorname{argmin}}
\quad \tfrac12\|p(q_{\mathrm{ref}}+B_3\alpha)-p_{d,i}\|^2
+\lambda_{\Delta}\|\alpha-\alpha_{i-1}^*\|^2,\\
&\text{subject to joint and enabled environment constraints}.
\end{aligned}
$$

**Do not constrain end effector orientation**, except in an explicitly labeled safety subexperiment. A raster plane tilted by an arbitrary angle is still a two dimensional positional task. Three coefficients are chosen for the primary experiment because they can preserve all **three** tip translation directions locally:

$$
\operatorname{rank}\big(J_p(q)B_3\big)=3.
$$

A rank two basis would be sufficient for a **single known plane** at a configuration only if the image of its reduced position Jacobian equals the plane tangent subspace:

$$
\operatorname{Im}\big(J_p(q)B_2\big)
=\operatorname{span}\{e_1,e_2\}.
$$

Keep `B2` as an explanatory baseline / ablation, not as a replacement for the requested `B3` experiment.

# 3. Provable differential DoF and singularity criteria

## 3.1 Exact rank identity

Let `S_k = Im(B_k)` and let `J` be the task Jacobian (`J6` for general pose; `Jp` for position only). Because `B_k` is injective,

$$
\boxed{\operatorname{rank}(JB_k)
=k-\dim\big(S_k\cap\ker J\big).}
$$

**Proof sketch:** restrict the linear map `J` to `S_k`; its kernel is `S_k ∩ ker J`. Rank nullity on the `k` dimensional restricted domain yields the identity.

This means that a linearly independent joint motion direction can still be **Cartesian irrelevant**, if it belongs to the kernel of the task Jacobian.

At a nonsingular seven joint configuration with `rank(J6)=6`, the null space of `J6` has dimension one. Therefore:

- `rank(J6 B6)` is **5 or 6** for full rank `B6`.
- `rank(J6 B3)` is **2 or 3** for full rank `B3`, but its value is not the relevant performance criterion for position only tasks.
- If `rank(Jp)=3`, the null space of `Jp` has dimension four; `rank(Jp B3)` can be **0, 1, 2, or 3**. A rank three `B3` does not suffice on its own.

At singular configurations, smaller ranks are possible.

## 3.2 Geometric simplification for `B6`

With `rank(J6)=6`, let `n_J(q) ∈ R^7` be a unit vector spanning `ker J6(q)`. Because a six dimensional subspace of `R7` has a one dimensional orthogonal complement, let `c_B ∈ R^7` be the unit normal to `Im(B6)`, so `c_B^T B6=0`.

Then the following test is exact:

$$
\boxed{\operatorname{rank}(J_6(q)B_6)=6
\iff |c_B^T n_J(q)|>0.}
$$

If the full Jacobian has smallest nonzero singular value `sigma_6(J6)`, an additional conditioning bound is

$$
\sigma_{\min}(J_6(q)B_6)
\ge \sigma_6(J_6(q))\,|c_B^T n_J(q)|.
$$

This is useful for offline optimization of a **single excluded joint direction** `c_B`. The optimization need not search all 42 entries of `B6`; it can search a unit vector on `S6` and construct its orthogonal complement. The basis should be interpreted as an entire subspace; its particular six column coordinates are not unique.

## 3.3 Certification over a region

For a constant `B` and target Jacobian `J`, define the regional margin

$$
\gamma(B,\mathcal Q_B)
=\inf_{q\in\mathcal Q_B}\sigma_{\min}(J(q)B),
$$

where `Q_B` is the allowed section of `q_ref + Im(B)` inside the operational joint domain. For `B6`, use `J6`; for `B3`, use `Jp`.

If `gamma > 0`, the reduced differential task map retains full **local** rank at every configuration in that region. To certify this rather than merely sample it, one possible sufficient procedure is:

1. Bound the Jacobian Lipschitz constant `L_J` on a compact region (use derivative or interval bounds, not a fitted constant with no guarantee).
2. Cover the region with points `q_i` such that every configuration is within distance `rho` of some cover point.
3. Establish `min_i sigma_min(J(q_i)B) > L_J rho`.

By the singular value perturbation inequality, the reduced Jacobian then has positive smallest singular value throughout the region (because `B` is orthonormal).

**This still does not prove** that every desired finite target pose or tip position is contained in the reduced reachable set. A separate reachability coverage argument or certified inverse calculation is required. Numerical grid success must be described as empirical coverage only.

## 3.4 Finite target coverage definitions

Let `F6(q)=T(q)`, `F3(q)=p(q)` and `Q_B={q_ref+B alpha ∈ Q_feasible}`. Define

$$
\mathcal X_{B,6}=F_6(\mathcal Q_B),\qquad
\mathcal X_{B,3}=F_3(\mathcal Q_B).
$$

For a specified target set `X_target`, ask whether `X_target ⊆ X_B,6` (pose targets) or `X_target ⊆ X_B,3` (tip targets). Do **not** substitute a local rank test for this finite coverage test. One goal of the project is to determine when a constant reduced basis offers adequate finite target coverage, and when multiple local bases or a changing anchor become necessary.

# 4. Basis construction and optimization

## 4.1 Baselines for both cases

Implement at least five initializations or acquisition strategies:

1. **Full joint IK (`B7 = I7`)**: ground truth reference for reachability and solver cost, not a reduced method.
2. **Joint selection:** pick `k` of seven standard coordinate axes (use several subsets, including the strongest validation subset).
3. **Random orthonormal basis:** QR decomposition of a seeded Gaussian matrix; multiple seeds.
4. **Jacobian right singular vectors:** compute at the episode reference configuration. Use `J6` for `B6` and `Jp` for `B3`. Freeze those vectors for the **entire** episode; do not quietly update them during evaluation.
5. **Data informed PCA:** collect joint displacements from successful full IK trajectories and retain the leading `k` left singular vectors; ensure training and test task families are separated.

A **learned optimized basis** is the proposed method, evaluated against all five. Include a configuration dependent Jacobian pseudoinverse controller as an **upper capability / additional computation baseline**; report clearly that it is not a constant basis.

## 4.2 Stage one: geometry conditioned offline basis fitting

Optimize `B6` or `B3` over a training set of target configurations and full IK verified targets. Use a two level objective:

**(a) Differential conditioning:** maximize the worst case or a low percentile of `sigma_min(J(q)B)` across relevant sampled `q`, with orthonormality constraint `B^T B=I`. Normalize mixed translation and angular units for the `B6` Jacobian before assessing singular values. Avoid using raw `J6` singular values when their magnitude is dominated by unit choices.

**(b) Finite task reconstruction:** for each training target, **actually solve reduced nonlinear IK**, then penalize final pose or position error, infeasibility, path discontinuity, and solver cost. Local full rank alone often yields no informative task error on a single step; it is not sufficient as a training objective.

A generic outer objective is

$$
\begin{aligned}
\min_{B^TB=I_k}\quad
&\mathbb E_{\tau\sim\mathcal D_{\mathrm{train}}}
\big[L_{\mathrm{task}}(\alpha^*(B,\tau),\tau)
+\lambda_f L_{\mathrm{feasible}}+\lambda_t C_{\mathrm{online}}\big]\\
&+\lambda_c L_{\mathrm{conditioning}}(B),
\end{aligned}
$$

where `alpha*(B,tau)` is produced by the **same constrained reduced IK solver** used at test time. This is a bilevel design objective; optimize approximately if differentiating through the complete constrained solver is impractical.

Practical progression:

- Start with linear algebra baselines and rank/conditioning optimized bases (smooth surrogate, projected updates on the Stiefel manifold, or optimization of the excluded normal for `B6`).
- Next use target reconstruction loss over a modest batch of actual nonlinear IK solves. A derivative free outer optimizer or finite difference approximation is acceptable for the initial experiment; log the extra offline cost.
- Only if this demonstrates a measurable benefit, consider a neural basis generator and differentiable least squares meta learning, conceptually inspired by **Neural Fly** (DOI `10.1126/scirobotics.abm6597`). Unlike Neural Fly, our target is a kinematically constrained joint motion subspace rather than a disturbance force model; no direct transfer of its control stability proof is claimed.

## 4.3 `B6` special optimization variable

At a full pose regular configuration, a rank six subspace is represented by a unit excluded direction `c_B`. Construct `B6 = orthogonal_complement(c_B)` using a stable QR or Householder method. Optimize a robust angle objective such as

$$
\max_{\|c_B\|=1}
\min_{q\in\mathcal Q_{\mathrm{train}}}
|c_B^T n_J(q)|,
$$

with a penalty for poor full Jacobian conditioning and finite IK failure. This directly minimizes the chance that the **excluded** direction becomes orthogonal to the robot's instantaneous pose null vector. Do not confuse this with excluding the pose null vector itself: excluding the null vector locally is desirable.

## 4.4 `B3` acquisition for plane tasks

A constant rank three basis is trained to preserve **tip position** reachability. The primary conditioning criterion is `sigma_min(Jp(q)B3)`. Add finite raster coverage and continuity losses. Train on many plane orientations and origins if a **universal** `B3` is desired; train on one plane only for the **transfer** experiment. In both cases validate that `B3` is full column rank and monitor its reduced translational Jacobian.

The three basis directions should not be interpreted as "two in plane plus one global normal" at every robot pose. Their Cartesian images rotate and deform with `Jp(q)`. A fixed joint basis can preserve local full translational rank yet miss distant planar targets.

# 5. Transferring `B3` between raster planes

Transfer is part of the required `B3` experiment, not part of the `B6` general pose experiment.

Let the source raster have plane frame `EA=[e1_A,e2_A]`, reference `qA`, and learned orthonormal basis `BA ∈ R^(7×3)`. Let the target raster be obtained from the source by `pB=R pA+t`, with `R ∈ SO(3)`. Obtain one feasible target starting configuration `qB` using the full seven joint position IK, counting this initialization cost.

Define `JA=Jp(qA)` and `JB=Jp(qB)`. A transferred basis that preserves the source **Cartesian velocity mapping under rotation** at the target starting configuration is

$$
\boxed{
B_{\mathrm{tr}}
=B_A+J_B^\dagger\big(RJ_A B_A-J_BB_A\big).
}
$$

If `JB` has full row rank, then `JB B_tr = R JA BA`. This is the Frobenius norm nearest solution to `BA` under the linear equality constraint. The expression handles only **differential kinematics at the anchor**, not full raster coverage. Translation `t` acts through selecting `qB`, not through the tangent rotation.

Perform QR factorization `B_tr=QR_q` (if rank three) and use `Q` as the orthonormal target basis. QR changes the coefficient coordinate system, not the spanned joint subspace. If the source Cartesian image has rank three and the target Jacobian has full row rank, the resulting target basis has rank three. Reject or regularize near singular transfer anchors.

**Optional local correction:** improve the transferred basis using a small collection of feasible target plane configurations and/or raster targets, minimizing a combination of nonlinear raster IK loss, reduced Jacobian conditioning, and proximity of the **subspace projector** to the transferred subspace:

$$
L_{\mathrm{transfer}}
=L_{\mathrm{raster}}(B)
+\lambda_c L_{\mathrm{cond}}(B)
+\lambda_r\|BB^T-Q Q^T\|_F^2.
$$

The projector metric is preferable to `||B-Q||` because it does not penalize a harmless rotation of coordinates inside the same joint subspace.

Compare: direct source basis without transfer; analytical transfer; analytical transfer plus small correction; full relearning from scratch; and a source independent generic basis.

# 6. Implementation procedure

## Phase 0. Kinematic and solver validation

- Select a seven revolute joint URDF and explicitly set base frame, end effector link, and tool tip transform. If using FR3, verify the exact model, flange and tool frames against the installed URDF; do not assume a generic Panda configuration is identical.
- Implement `fk(q)`, `jacobian_pose(q)`, and `jacobian_position(q)` with consistent world/body frames and joint order.
- Validate `Jp` against symmetric finite differences in tip position and `J6` against a consistent local pose perturbation. Test random nonsingular states and several near singular states.
- Implement full `7D` position and pose IK with true joint limit constraints, using the same objective, stopping criteria, seed policy, and collision handling as the reduced solver.
- Record target feasibility with the full solver, but distinguish **solver found feasible** from **mathematically proven reachable**.

## Phase 1. Reproducible task generation

**General pose dataset:** sample feasible joint states, compute their forward poses, and create target sets distributed across a specified cubic tip position region. Include broad orientation variation but only claim reachability for poses verified by FK or IK. Split by trajectory or task region, not by randomly mixing neighboring waypoints. Specify joint reference poses independently from target generation to avoid easy zero motion cases.

**Plane dataset:** sample reference configurations and plane frames `(c,R)`. Define rectangle dimensions `(W,H)`, raster row spacing, waypoint spacing, and traversal direction. Reject tasks for which the full reference solver cannot track the path under the selected constraints. Evaluate variable plane tilt, plane translation, area, aspect ratio, and path length.

Suggested **configurable** initial bins rather than physical assumptions: plane tilt `{0°, 15°, 30°, 45°, 60°, 75°}`; raster dimensions `{small, medium, large}` derived from actual verified robot reachable space; at least two distinct starting postures per region; and separate unseen plane orientations and positions at test time.

Save joint reference, target poses or positions, plane frame, sequence of waypoints, full IK feasibility record, solver seed, and robot model checksum. Keep training, validation, and held out test tasks distinct.

## Phase 2. Baseline acquisition and reduced IK

- Generate `B6` and `B3` with joint selection, random QR, reference Jacobian SVD, and PCA from training trajectories.
- Validate `B^TB≈I`, rank, reduced Jacobian singular values, and FK/Jacobian agreement.
- Solve targets with `q=q_ref+B alpha`, warm starting `alpha` from the previous waypoint.
- Enforce coupled joint bounds `q_min ≤ q_ref+B alpha ≤ q_max`; do **not** replace these constraints with independent scalar bounds on `alpha` unless proven equivalent.
- Use the same numerical solver family for full IK and reduced IK wherever possible; measure objective/Jacobian evaluations and wall clock time, not only iteration counts.

## Phase 3. Optimize fixed bases

- Start with conditioning oriented offline optimization. For `B6`, use the excluded direction `c_B`; for `B3`, optimize an orthonormal `7×3` matrix.
- Optimize finite target coverage using actual reduced IK in the inner loop. Include failure penalties and preserve a record of failed targets; do not silently discard them from the objective.
- Freeze each optimized `B6` or `B3` for a complete test episode.
- Reevaluate on held out trajectories, disjoint plane poses, and independently sampled general pose targets.

## Phase 4. Transfer `B3` to another plane

- Take a basis acquired on a source plane, compute one target reference configuration, and apply the analytical transfer formula.
- Orthogonalize and verify the target projected Jacobian rank and condition number.
- Perform the full target raster without recomputing the basis at every waypoint.
- Repeat with a limited budget of correction examples or correction iterations, and compare with full relearning.
- Count source training, target initialization, analytic transfer, correction, and online IK separately.

## Phase 5. Regional verification and analysis

- Plot `sigma_min` and condition number along every trajectory and across a documented grid of feasible reduced configurations.
- Identify where reduced IK fails although full IK succeeds; classify failures as rank loss, poor conditioning, finite subspace coverage loss, joint limit violation, collision, or local solver convergence failure.
- Attempt certified lower bounds for `sigma_min` on small regions using interval or derivative bounds. Label larger numerical sweeps as empirical tests.
- If single fixed bases cannot support large regions, evaluate a **separately labeled** multi chart approach or occasional reanchoring and measure the number and cost of switches. Do not mix this with the constant basis main results.

# 7. Suggested software organization and interfaces

A Python package is adequate for the initial research benchmark. Suggested dependencies: NumPy/SciPy for numerical optimization, a reliable robot kinematics library such as Pinocchio for FK/Jacobians, PyTorch or JAX only if a learned basis model is later justified, and Matplotlib/Pandas for analysis. Lock exact library versions when the implementation environment is established.

```text
reduced_ik/
  config/
    robot.yaml
    ik.yaml
    experiments.yaml
  kinematics/
    robot_model.py             # FK, tip frames, geometric Jacobians
    jacobian_checks.py         # finite difference tests
  tasks/
    pose_tasks.py              # reachable 6D pose benchmarks
    plane_tasks.py             # plane parameters and raster waypoints
  basis/
    construction.py            # coordinate, random, SVD, PCA
    conditioning.py            # singular values, rank tests, rank margins
    optimize_b6.py             # excluded direction / robust basis fit
    optimize_b3.py             # positional task basis optimization
    transfer_plane.py          # analytical transfer and correction
    io.py                      # checkpoint metadata and validation
  ik/
    full_ik.py
    reduced_ik.py
    constraints.py
    trajectory.py
  experiments/
    eval_general_pose.py
    eval_raster.py
    eval_plane_transfer.py
    compare_baselines.py
  analysis/
    metrics.py
    plots.py
    regional_checks.py
  tests/
    test_jacobians.py
    test_basis_rank.py
    test_ik_constraints.py
    test_transfer_identity.py
    test_raster_generation.py
```

Minimum interfaces:

```python
robot.fk(q) -> SE3Pose
robot.jacobian_pose(q) -> ndarray[6, 7]
robot.jacobian_position(q) -> ndarray[3, 7]

make_basis(method, robot, task_data, k, seed) -> ndarray[7, k]
optimize_basis(initial_basis, task_data, k, settings) -> BasisResult

solve_ik(target, q_ref, basis, task_type, constraints, warm_start) -> IKResult
solve_trajectory(targets, q_ref, basis, task_type, constraints) -> TrajectoryResult

transfer_b3(B_A, q_A, q_B, R_BA, robot) -> BasisTransferResult

evaluate_pose_set(dataset, methods, settings) -> MetricsTable
evaluate_rasters(dataset, methods, settings) -> MetricsTable
```

Each `IKResult` must include final joint state, coefficients, raw position and orientation errors, objective, feasibility flags, stopping reason, iteration/evaluation counts, and measured solve time. Each `TrajectoryResult` must include all waypoint results, maximum joint step, estimated joint velocities under the chosen timestamps, collision flags when enabled, and the first failure index.

# 8. Experimental matrix

## Experiment G1: Six basis vectors for general pose motion

**Question:** How frequently does a fixed six dimensional joint subspace preserve full pose IK feasibility compared with full seven dimensional IK?

**Methods:** full `7D`, best coordinate `6D`, random `6D`, frozen SVD `6D`, PCA `6D`, optimized `B6`; optionally a dynamic pseudoinverse method as a nonconstant comparison.

**Tasks:** reachable, held out six DoF poses with diverse translations, orientations, starting configurations, and distances from the anchor. Include an interior subset and difficult joint limit / near singular subsets.

**Required outputs:** full pose success fraction; position and orientation errors; `sigma_min(J6 B6)`; distances to joint limits; solution continuity for pose sequences; solve time; reason for any full solver versus reduced solver disagreement.

**Key ablation:** vary the distance from the fixed anchor to the target. This distinguishes locally valid `B6` from finite workspace completeness.

## Experiment P1: Three basis vectors for one fixed plane

**Question:** Can a constant `B3` solve a full planar raster, including all turns, without joint space reanchoring?

**Methods:** full `7D` position IK; constant `B3` from all construction methods; position Jacobian SVD `B3`; optional task specific `B2`.

**Tasks:** multiple plane sizes, orientations, locations, start positions, and two raster directions. Orientation of the tool remains unconstrained. Verify full solver success first.

**Required outputs:** waypoint success fraction; whole raster success fraction; maximum and RMS tip error; normal distance to plane; trajectory smoothness; `sigma_min(JpB3)`; time per waypoint and per raster; reachable area coverage by independent position targets.

## Experiment P2: One `B3` shared across a family of planes

**Question:** Can one constant rank three joint basis preserve position reachability for different plane poses without retraining or transferring?

**Methods:** pooled PCA `B3`, pooled optimized `B3`, random `B3`, frozen SVD `B3`, full `7D` position IK.

**Tasks:** held out plane rotations and translations with varying sizes. Use a **fixed anchor per episode** and report anchor establishment separately.

**Required outputs:** error as a function of plane angle, translation, and area; successful plane fraction; percentage of full solver verified raster targets preserved; computational advantage.

## Experiment P3: Source to target plane transfer

**Question:** Is analytic basis transfer faster or more data efficient than acquiring a new target plane basis from scratch?

**Methods:** unmodified source `BA`; transferred `Btr`; transferred plus small corrected basis; target only optimized basis; pooled generic `B3`; full `7D`.

**Tasks:** paired source and target planes across tilt and origin change. Hold out transfer transformations from any training data.

**Required outputs:** target plane accuracy and success; number of target training samples / offline optimization iterations; analytic transfer and correction wall time; target start configuration cost; online per waypoint IK time. Include transfer from both nearby and distant robot postures.

## Experiment P4: Failure and certification study

**Question:** What precisely breaks reduced basis completeness?

**Tasks:** progressively larger rasters, proximity to joint limits, target configurations where projected Jacobian singular values approach zero, and planes whose regions are reachable only through distinct joint posture branches.

**Outputs:** visual classification of failure, empirical rank margin maps, optional small certified regions, and comparisons between constant, reanchored, and state dependent bases (the latter two clearly marked as different methods).

# 9. Measurement and reporting standard

Set position and orientation tolerances in the configuration file and state them in every plot/table. Proposed starting thresholds for controlled simulation studies are **1 mm** tip error and **1 degree** orientation error, subject to robot geometry, solver precision, and experimental feasibility; these are **test settings**, not promised capabilities. For planar tasks, orientation is not evaluated as a success constraint.

Report separately:

- **Target IK success:** a returned joint configuration satisfies all enabled constraints and the correct position/pose tolerance.
- **Whole trajectory success:** every waypoint succeeds and the joint trajectory meets configured continuity and velocity limits; no concealed restarts or full dimensional rescue steps.
- **Finite coverage:** percentage of a held out, full solver verified target set reached by the reduced solver. This is empirical unless specifically certified.
- **Differential mobility:** numerical rank, smallest singular value, and condition number of the correctly selected reduced Jacobian at sampled configurations.
- **Cost:** median and 95th percentile online solve time, FK/Jacobian evaluations, iteration counts, initialization cost, training cost, transfer/correction cost, and basis memory size.
- **Accuracy:** maximum/RMS position and orientation errors; normal deviation from raster plane; joint trajectory smoothness.

Use identical hardware, precision, stopping criteria, task targets, and initialization policy. Report at least five fixed random seeds for stochastic bases, with confidence intervals or bootstrap intervals for success rates and runtime. Explicitly separate failures of the subspace from failures of a particular nonlinear IK optimizer where possible (multiple seeded solves can help diagnose solver sensitivity).

# 10. Automated tests and acceptance criteria

**Mathematical tests**

1. Check `B^T B ≈ I` and `rank(B)=k` for every constructed and transferred basis.
2. At deliberately constructed configurations, verify `rank(JB)=k-dim(Im(B)∩ker(J))` numerically using controlled tolerances.
3. For `B6`, form an excluded normal `c_B`, compare `rank(J6B6)` with the absolute null vector dot product test, and verify the singular value lower bound numerically.
4. For transfer, whenever `JB` has full row rank, assert `||JB Btr - R JA BA||` is close to machine precision; then confirm QR preserves the joint subspace projector.
5. Verify one plane's raster generator produces points on that plane and obeys the specified dimensions and sweep spacing.

**Software acceptance**

- A single configuration switch runs full `7D`, `B6`, and `B3` IK with their respective task objectives.
- For any sampled output, joint limits are satisfied within configured tolerances; infeasible attempts are reported as failures, not silently clipped to limits.
- Fixed basis experiments do not call full joint IK after episode initialization, and do not update `q_ref` or `B` between waypoints.
- Evaluation reproduces all reported numerical results from saved seeds, configuration files, and basis checkpoints.
- Every plot and metrics table records whether the basis is constant, transferred, corrected, state dependent, or reanchored.
- End effector orientation is absent from the planar raster success criterion.

**Research acceptance (not guaranteed outcomes):** Demonstrate a meaningful improvement over at least the frozen SVD and PCA reduced baselines in held out finite reachability or runtime at a matched success tolerance. A result showing that rank conditions are often satisfied but fixed reduced subspaces fail finite coverage is also a valid research finding; do not suppress it.

# 11. Required outputs for developer handoff

Provide a runnable repository with documented installation and one command per experiment; generated datasets or deterministic dataset scripts; basis checkpoint files with robot/tool frame metadata; source and target plane transforms; raw per waypoint logs; summary CSV/Parquet metrics; figures; passing unit tests; and a short result report specifying confirmed observations, numerical limitations, and unproven claims.

**Initial implementation priority:** kinematic correctness → baseline solvers → `B6` local rank and pose IK evaluation → `B3` planar IK evaluation → offline optimization → interplane basis transfer → regional certification. Do not start neural basis learning until the analytical and fixed basis benchmarks are reproducible.

---

# Appendix A. Minimal experimental pseudocode

```python
for k, task_kind in [(6, 'pose'), (3, 'tip_position')]:
    train_tasks, test_tasks = make_disjoint_tasks(task_kind, robot, config)
    baselines = acquire_baseline_bases(k, train_tasks, robot, seed=config.seed)
    optimized = optimize_constant_basis(k, train_tasks, baselines, robot)

    for method, B in {**baselines, 'optimized': optimized}.items():
        assert B.shape == (7, k)
        assert orthonormal(B)
        for episode in test_tasks:
            q_ref = episode.start_q  # established once; no target-by-target reset
            result = solve_trajectory(
                targets=episode.targets,
                q_ref=q_ref,
                basis=B,
                task_type=task_kind,
                constraints=episode.constraints,
            )
            log_episode(method, k, episode, B, result)

# Additional 3D position / planar transfer benchmark:
for pair in plane_transfer_pairs:
    BA = load_source_plane_basis(pair.source)
    qA, qB = pair.source.start_q, pair.target.start_q
    Btr = analytic_transfer_and_qr(BA, qA, qB, pair.R_BA, robot)
    for variant in (BA, Btr, correct_basis(Btr, pair.target.training_subset)):
        log_episode('plane_transfer', 3, pair.target, variant,
                    solve_trajectory(pair.target.targets, qB, variant,
                                     'tip_position', pair.target.constraints))
```

# Appendix B. Boundaries of the claim

**Can prove directly:** linear algebra rank identity, maximum instantaneous Cartesian DoF, local rank test for `B6`, exact differential transfer when its Jacobian assumptions hold, and a regional rank guarantee given valid Lipschitz and covering bounds.

**Cannot claim without further work:** a constant rank six basis reaches every feasible six DoF pose; a constant rank three basis covers all accessible planar raster locations; successful sampled trajectories prove global reachability; reduced variable dimension guarantees runtime improvement; basis transfer preserves entire finite paths; or unconstrained tool attitude implies collision or machining feasibility.

**Related inspiration:** Neural Fly, *Science Robotics* (2022), DOI [10.1126/scirobotics.abm6597](https://doi.org/10.1126/scirobotics.abm6597), for the pattern of learning a reusable representation offline and adapting a small number of coefficients online. The present project addresses IK subspace coverage, not aerodynamic force estimation or adaptive flight control.
