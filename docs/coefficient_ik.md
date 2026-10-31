# GPU coefficient-space inverse kinematics

The learned coefficient solver, matched full-joint comparison, held-out trajectory evaluation, and experiment UI are implemented. Both optimizers run batched FK, Jacobian/residual evaluation, and coefficient updates on CUDA. The trained fixed chart is substantially less reliable than full-joint IK on the measured finite trajectories.

## Commands and UI

~~~powershell
uv run --locked python -m metaarm
uv run --locked python -m metaarm.evaluation data
~~~

The first command serves the experiment at http://127.0.0.1:8080. The second compares every stored test trajectory in every NPZ beneath data/ and creates a new results/ik_<UTC timestamp>/ directory.

The checkpoint is explicitly pinned to checkpoints/basis_20261031T105751490589Z_audited.pt. Its model specification, normalization, joint limits, and physical asset fingerprints are checked at construction. There is no latest-checkpoint search.

In the UI:

1. Select Pose or Plane and choose which method to drive with.
2. Edit a target or select a preset. Plane direction is normalized to a unit normal; zero/nonfinite directions are errors.
3. Apply target computes both learned and full-joint solutions from the same current robot reference. Their dimensions, errors, times, iterations, and stopping statuses are displayed.
4. Only a successful result from the selected method becomes the actuator goal. A failed selected result pauses without applying either candidate.
5. Run / Pause controls physical tracking. Reset restores home, zero time, and the initial controls. Quit stops the server.

Plane mode explicitly projects the requested point onto the displayed plane through the current tool. Tool orientation remains free. The target marker shows the applied point. Physical tracking error and plane-normal error are separate from the predicted IK result.

Each Apply starts a new independent reference chart. Its quintic joint reference interpolates within that chart; actual dynamics can deviate from it. Coefficient solving owns separate kinematics memory and leaves physical positions, velocities, controls, and time unchanged.

## Fixed-chart optimization

For one reference and frozen orthonormal basis:

~~~
q(alpha) = q_ref + B alpha
q_min - q_ref <= B alpha <= q_max - q_ref
~~~

The learned pose basis has six columns, the plane basis three, and the full-joint comparison uses B = I_7. Initial coefficients are zero; ordered trajectories use the preceding successful coefficient vector. Neither the neural input nor the basis changes during a trajectory.

Position residual is (p(q)-p_target)/1e-3. Pose adds phi/(pi/180), where phi = Log(R_target.T R(q)) is the principal body rotation vector. For the base-aligned geometric Jacobian:

~~~
D_alpha e_position = J_position B / 1e-3
D_alpha e_rotation = J_right_inverse(phi) R(q).T J_angular B / (pi/180)
J_right_inverse(phi) = I + hat(phi)/2 + a(theta) hat(phi)^2
a(theta) = (1 - theta/(2 tan(theta/2))) / theta^2
~~~

Analytic small-angle limits evaluate the logarithm and its derivative at zero. The rotation-log cut within 1e-4 rad of pi is rejected explicitly. The actual log derivative is tested at nonzero orientation error against independent CPU finite differences.

The optimizer uses a maximum of 100 damped Gauss-Newton iterations, an initial dimensionless damping of 1e-3, and a 0.25 rad joint-displacement trust radius. Strict cost decrease accepts a candidate and multiplies damping by 0.3; rejection multiplies it by 10. These are fixed numerical choices.

Each quadratic subproblem retains all fourteen coupled joint inequalities. A feasible unconstrained minimizer is already the constrained optimum. Otherwise, a feasible primal-dual interior-point iteration solves the constrained quadratic, with at most 24 inner iterations and explicit stationarity/complementarity tolerances. The barrier is a numerical QP mechanism, not an extra task penalty. Initial configurations must be strictly interior; invalid assumptions and nonfinite calculations raise errors.

FK uses the pinned [MuJoCo Warp kinematics implementation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html). Seven revolute world axes and anchors yield J_position[:,j] = axis_j × (p_tool-anchor_j) and J_angular[:,j] = axis_j. Neural inference/FK use float32; task algebra, coefficients, linear systems, and joint constraints use CUDA float64.

Stopping statuses are 0 for task tolerances reached, 1 for a small step or excessive damping, and 2 for the iteration limit. Task success is computed separately from final raw errors and joint feasibility. Status 0 is not a KKT or global-optimality certificate; statuses 1/2 do not prove a target unreachable.

## Exact implementation surface

The implementation uses one CoefficientIK class with four methods and two top-level functions:

| Definition | Role |
| --- | --- |
| CoefficientIK(model, num_envs, checkpoint) | Allocate independent GPU kinematics and verify/load the selected neural model. |
| geometry(q) | Return batched tool position, rotation, and base-aligned geometric Jacobian. |
| residual(q, target, task) | Return normalized position/pose residuals and joint-coordinate derivatives. |
| solve(q_ref, basis, target, task, alpha_initial) | Freeze the reference/basis and optimize only the supplied coefficient coordinates. |
| constrained_step(hessian, gradient, basis, margin) | Solve the feasible coupled joint-limit quadratic step. |
| evaluate_ik(source, destination) | Pool archived test tasks, run both methods, audit and save results. |

Target is a dictionary containing matching CUDA position (N,3) and proper rotation (N,3,3) tensors; plane tasks ignore orientation in their objective. Public inputs are explicit, with no runtime configuration or solver flags. The existing UI main/enqueue definitions remain the only UI workflow/callback functions. There are no new dataclasses, wrappers, optimizer frameworks, or dependencies.

## Held-out protocol

The final run used all 20 archives, 160 pose trajectories and 320 plane trajectories, each with 65 waypoints. Stored test reference regions and plane tilts were preserved. The trained model was fixed before these IK results were examined; these archive test groups were previously inspected during training work, so this is a matched held-out comparison rather than an untouched benchmark.

Both methods receive the same references, targets, tolerance scales, zero initial coefficients, and optimizer budget. A method stops its own trajectory at its first task failure; later entries retain NaN values, false attempted masks, and status -1. The other method continues its independent trajectory. Every failed candidate is retained.

Batch sizes remain 160/320. Completed/stopped rows receive their own current FK target as inactive padding while live trajectories continue. Synchronized solver timing includes those fixed-size CUDA batches and host control overhead; it excludes checkpoint loading, basis construction, warm-up, CPU auditing, and serialization. Batch time divided by attempted count is amortized throughput cost, not single-target latency. Different stopping histories prevent a simple speedup claim.

## Measured results

Final artifacts are in results/ik_20261031T112840039818Z/.

| Task / method | Completed trajectories | Successful / planned waypoints | Attempted waypoints | Solver batch total |
| --- | ---: | ---: | ---: | ---: |
| Pose / learned, 6 variables | 88 / 160 (55.0%) | 6,568 / 10,400 | 6,640 | 5.895 s |
| Pose / full joints, 7 variables | 160 / 160 (100%) | 10,400 / 10,400 | 10,400 | 1.172 s |
| Plane / learned, 3 variables | 19 / 320 (5.94%) | 3,107 / 20,800 | 3,408 | 10.431 s |
| Plane / full joints, 7 variables | 320 / 320 (100%) | 20,800 / 20,800 | 20,800 | 0.892 s |

Basis preparation took 0.0524 s for pose and 0.0509 s for plane, separately from solving. The learned method is slower here because difficult or failed targets require more iterations. Smaller search dimension alone did not improve measured efficiency.

Raw error quantiles below include failed attempted candidates:

| Task / method | Position median / 95th / maximum (m) | Rotation maximum (rad) | Plane-normal maximum (m) |
| --- | --- | ---: | ---: |
| Pose / learned | 0.000253 / 0.000883 / 0.003745 | 0.145757 | N/A |
| Pose / full | 0.000247 / 0.000846 / 0.001000 | 0.007495 | N/A |
| Plane / learned | 0.000196 / 0.005527 / 0.031142 | Orientation free | 0.016898 |
| Plane / full | 0.000185 / 0.000824 / 0.001000 | Orientation free | 0.000990 |

Independent float64 CPU MuJoCo evaluated every attempted candidate. Maximum CPU/GPU task-error difference was 4.31e-7, with zero success-label disagreements and zero joint-limit violations. Minimum recorded margins were positive, down to 1.79e-10 rad for the learned plane method.

## What the failures indicate

The first eight failed episodes per task were independently refined with CPU SLSQP, numerical derivatives, the same frozen basis/polytope, and 200 iterations. All sixteen optimizations reported success but remained outside task tolerances at nearly the same errors. This supports the GPU result as a local chart limitation; it is not a global infeasibility proof.

Only one of 72 failed pose candidates and 18 of 301 failed plane candidates were within 1e-6 rad of a joint limit. The failures are predominantly interior, rather than simply blocked by joint bounds.

CPU geometric-Jacobian audits found median minimum singular values of J_task B at failed candidates of 1.93e-5 for pose and 2.16e-5 for plane. There were 67 interior pose failures and 270 interior plane failures with minimum singular value below 1e-4. The trained basis had reference minima around 0.02–0.06, so finite motion can lose the differential mobility present at the reference. The data are consistent with chart folds or local stationary residuals.

The earlier reconstruction improvement therefore does not establish finite task coverage. The scale-32 model emphasizes displacement reconstruction, including teacher nullspace drift, without controlling conditioning away from the reference. A subsequent learning experiment should evaluate finite-chart conditioning and coefficient-IK success alongside reconstruction. Chart switching, reanchoring within a trajectory, full-joint rescue, new training, and collision avoidance are outside this implementation.

## Verification and artifacts

All 14 project tests passed in 29.77 seconds. One first-use cuBLAS context initialization warning occurred in the existing float64 training gradient check. Ruff lint and formatting passed for all 17 Python files.

The three added tests cover CPU/GPU FK and Jacobians, nonzero rotation-log finite differences, independent constrained-QP agreement, feasible pose/plane chart targets, full-joint comparison, unreachable targets, affine-coordinate preservation, and bitwise unchanged simulation state.

Live UI verification exercised learned pose application and physical run/pause, projected plane targets, free orientation, three-coefficient plane solving, explicit full-joint selection, unreachable-target pausing, reset to home/time zero, and clean shutdown. The final server is available paused at the local UI address.

The result directory contains summary.json with complete source/checkpoint/corpus fingerprints and counts, trajectories.npz with source-qualified identities and every attempted/failed/unattempted record, failure_audit.json with independent SLSQP checks, chart_audit.json with failed-candidate rank diagnostics, source/ with the evaluated implementation, and verification.json with final check evidence.
