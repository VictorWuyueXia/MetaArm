# MetaArm development foundation

The primary workspace is C:\Users\Victor\Documents\Projects\MetaArm. This plan defines the environment, physical simulator, experiment interface, and boundaries for subsequent data collection and learned-basis work.

The [research brief](research_brief.md) and [mathematical formulation](1-mathematical_formulation_and_implementation_plan.md) define the research method: learn joint motion bases and solve IK using six or three coefficients. GPU MuJoCo replaces the earlier CPU/Pinocchio infrastructure proposal without changing that objective. The former joint-reference prediction baseline is not the selected learning method.

## Infrastructure and isolated environment

| Component | Decision | Purpose |
| --- | --- | --- |
| Host | Native Windows x86-64, NVIDIA RTX 4070 | Use the existing workstation directly. |
| Environment | Python 3.12 in .venv, managed by uv | Isolate the project from Anaconda base and system packages. |
| Dependency records | pyproject.toml, uv.lock, .python-version | Freeze sources, exact resolution, and Python selection. |
| Physical model | Pinned Menagerie FR3 MJCF and meshes | One geometry, joint-limit, actuator, and tool-frame source. |
| Physics | MuJoCo Warp on cuda:0 | Simulate independent worlds together on the GPU. |
| Learning infrastructure | CUDA PyTorch | Shared GPU arrays and subsequent neural basis training. |
| UI | mjviser and Viser | Python-defined controls and visualization in one process. |
| ROS 2 | Add only for a specified integration | Keep middleware outside the GPU experiment loop. |

MuJoCo Warp targets batched throughput, without promising lower single-world latency than CPU MuJoCo. Use one explicit GPU backend and no automatic CPU substitution. [MuJoCo Warp documentation](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)

Warp publishes native Windows CUDA wheels. Compatibility must be established by execution, not inferred from GPU visibility. [Warp installation](https://github.com/NVIDIA/warp#installing)

## Packages and reproducibility

| Package | Role |
| --- | --- |
| mujoco | Model composition, compilation, reference kinematics, and analytic Jacobians. |
| mujoco-warp, warp-lang | Parallel CUDA dynamics and shared state arrays. |
| torch | GPU tensor operations and future learned motion bases. |
| numpy | CPU reference IK, numerical verification, and numerical datasets. |
| scipy | Analytic-gradient SLSQP position references for planar data. |
| mjviser, viser | Model visualization and experiment controls. |
| pytest, ruff | Numerical/integration checks, linting, and formatting. |

Use standard-library pathlib, tomllib, queue, and time for paths, experiment definitions, UI requests, and pacing. Add no configuration framework, generic robot-learning framework, separate web backend, or ROS dependency.

Use an explicit official CUDA PyTorch index; Windows PyPI packages can otherwise select CPU-only PyTorch. Freeze package sources and numerical/runtime choices rather than selecting them at startup. [uv PyTorch guidance](https://docs.astral.sh/uv/guides/integration/pytorch/)

Retain the upstream robot license and a source commit plus SHA-256 checksums in assets/franka_fr3/SOURCE.md. Project scene changes do not alter upstream files. [FR3 model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3)

## One-command launch

After installing uv once, create/recreate the environment and launch from the repository root:

~~~powershell
uv sync --locked
uv run --locked python -m metaarm
~~~

The second command launches the complete experiment. It reads experiments/reach.toml, initializes the scene/CUDA, warms and captures physics, validates targets, and serves http://127.0.0.1:8080. It starts paused. Cold launches compile GPU kernels; later launches reuse the cache.

One owning loop updates the physical state. UI callbacks enqueue requests only. Display world zero at a maximum of 30 Hz while all configured worlds advance together.

The interface provides presets, position in metres, orientation in roll/pitch/yaw degrees, Run/Pause, Reset, Apply target, and Quit. Show simulation time, world count, actual tool position, and separate position/angular errors. The pose frame represents the applied target; input edits are pending until Apply target.

Cartesian control is explicitly labelled **full-joint reference IK**. It is a simulator diagnostic, not a learned-basis solver or a substitute after reduced IK failure. [mjviser API](https://github.com/mujocolab/mjviser#python-api)

## File organization

~~~text
MetaArm/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── docs/
│   ├── DEVELOPMENT_PLAN.md
│   ├── simulation_setup.md
│   ├── research_brief.md
│   └── 1-mathematical_formulation_and_implementation_plan.md
├── assets/
│   ├── franka_fr3/          # Upstream MJCF, meshes, license, provenance.
│   └── reach.xml            # Fixed floor, table, contact box, and lighting.
├── experiments/
│   └── reach.toml           # Three experiment fields.
├── src/metaarm/
│   ├── __init__.py          # Package marker.
│   ├── scene.py             # Compose and verify the physical model.
│   ├── simulation.py        # Own and advance CUDA worlds.
│   ├── basis.py             # Rank-preserving neural motion bases.
│   ├── training.py          # CUDA reconstruction training and checkpoint evidence.
│   ├── collection.py        # General and projected planar motion records.
│   ├── ik.py                # Full-joint reference IK.
│   └── __main__.py          # Startup, UI, request dispatch, motion loop.
└── tests/
    └── test_foundation.py
~~~

Do not create empty collection/training modules or inactive UI controls. Generated datasets, metrics, and checkpoints belong in ignored output directories when those workflows exist.

## Exact current structural budget

The interactive simulation has four workflow modules, one custom class, three methods, four functions including the UI callback, and three test functions. There are zero dataclasses, worker classes, wrappers/helpers, generic backend layers, and CLI arguments. Revise this plan before expanding the structure.

| File | Callable and inputs | Role |
| --- | --- | --- |
| scene.py | build_scene() | Compose scene/robot and verify mappings, limits, home state, and frames. |
| simulation.py | Simulation.__init__(model, num_envs) | Allocate CPU/GPU state, tensor views, shared stream, and captured interval. |
| simulation.py | Simulation.reset(q_initial) | Reset complete dynamic state, controls, time, warm starts, and geometry. |
| simulation.py | Simulation.step(q_ref) | Check references, advance five physics steps, refresh geometry, check finiteness. |
| ik.py | solve_ik(model, kinematics, target_pose, q_seed) | Solve the seven-joint reference problem or raise explicitly. |
| __main__.py | main() | Read experiment, construct UI, dispatch requests, interpolate, render. |
| __main__.py | Nested enqueue(event) | Capture one browser request for the owning loop. |

| Object group | Responsibility |
| --- | --- |
| sim | One Simulation instance owning CPU/GPU model/data, stream/graph, indices, and tensor views. |
| server, scene_view | Viser server and ViserMujocoScene displaying one world. |
| requests | Standard-library queue for UI requests. |
| experiment, GUI handles | Plain dictionary and library handles; no custom command/config objects. |

The independent numerical groups are world count/time, joint state (q, qd), joint reference q_ref, and target pose/IK seed. Errors, Jacobians, interpolation phase, and solved targets are derived quantities, not new optimization variables.

## Physical and numerical contract

Use seven revolute FR3 joints with the upstream position actuators. Base/world frames coincide; the tool is attachment_site, 0.107 m along the final-link local z-axis. This is an arm/flange model. Select a gripper explicitly for manipulation.

Resolve named indices and validate contiguous ordering before constructing shared slices. Do not assume future scene coordinates all belong to the arm. Use SI units and radians internally, converting explicitly at the UI boundary.

Advance physics at 0.002 s and control at 0.01 s. Quintic joint-reference interpolation drives the actual actuators and never overwrites dynamic state. These choices do not establish hardware equivalence or collision-free planning.

Reference IK uses a separate CPU MjData, analytic site Jacobians, and a base-aligned SO(3) error. Limit damped least-squares increments and shorten them to retain joint feasibility. Keep damping, increment limits, tolerances, and iteration count beside the equations. Invalid, stalled, or non-converged solves raise errors.

Use float32 GPU arrays and float64 CPU reference IK. Warp/PyTorch share storage and an explicit non-default CUDA stream. Capture the warmed five-step control interval after measured launch overhead justified it, and refresh geometry after integration. [Warp/PyTorch interoperability](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html)

Batch across worlds and future learning samples; retain time/nonlinear-solver loops required by the mathematics. Copy only world zero for visualization. This plan does not assume differentiable MuJoCo Warp dynamics.

## Minimum experiment parameters

| Field | Meaning |
| --- | --- |
| num_envs | Number of independent worlds; the supplied experiment uses 64. |
| target_poses | Base-frame poses as (x, y, z, qw, qx, qy, qz). |
| motion_seconds | Duration of the smooth reference to an applied target. |

Freeze device, host/port, paths, types, numerical intervals, visualization rate, solver settings, and allocation capacities. An invalid batch or exhausted memory must raise an error rather than silently changing the experiment.

## Subsequent research workflows

Extend the physical workflow only after declaring the small structure budget for the next task.

1. **Collection:** record general joint motions, references, full states/controls, episode boundaries, geometry, and model/Jacobian provenance. Separate reset boundaries from physical transitions and retain solver failures.
2. **Training:** implement the learned motion-basis objective in the mathematical formulation. Split complete episodes/reference regions, train batched matrices, and save final weights and data/model provenance.
3. **Reduced IK:** optimize only the six- or three-dimensional coefficient vector with the selected within-solve basis policy and coupled joint constraints. Keep full-joint comparisons explicit.
4. **Evaluation:** compare held-out poses/planes, failures, raw position/angular errors, task-space rank samples, joint margins, and separated preparation/training/solve costs. Samples do not certify a uniform region.
5. **Integration:** add ROS 2 only for a specified interface, using a compatible Ubuntu/ROS/Franka deployment outside the GPU learning loop.

The foundation does not select a new learning architecture or replace neural basis training with joint-reference prediction.

## Coding discipline and verification

Use compact, continuous logic and minimum realization. Prefer removing code, direct library calls, vectorized arrays, and batched matrices. Keep ordinary files approximately 50–250 lines and substantive functions approximately 20–100 lines without padding short operations. Include concise professional one-line comments for key physical/numerical blocks and no development-stage narration.

Avoid unnecessary helpers, CLI/configuration, fallback values/behavior, and try/with/except constructs. Invalid assumptions and computation failures remain explicit. Algorithmic damping and physical constraints do not authorize hidden solver/backend substitution.

The three tests are test_scene_contract(), test_ik_contract(), and test_simulation_contract(). Check Jacobians by finite differences, feasible/unreachable IK, shared GPU storage, CPU/GPU free-motion/contact agreement, independent worlds, complete reset, and invalid references. Exercise run/pause, target application, reset, and shutdown in the browser.

Benchmark explicit batch sizes after warm-up. Distinguish physics throughput, browser rate, cold compilation, and GPU allocations. Installation and measured results are recorded in [simulation setup and validation](simulation_setup.md).
## Motion-data implementation budget

The collector follows section 7 of the mathematical formulation: nominal joint paths, forward-kinematic general demonstrations, and projected planar targets labelled by an explicit full-joint position solver. It does not use dynamic tracking states as kinematic reference labels.

Add one source file, src/metaarm/collection.py, with three workflow functions: generate_motions(sim, q_ref, seed) constructs feasible joint paths and obtains batched GPU poses plus CPU reference Jacobians; project_motions(sim, general, normals) constructs plane geometry and ordered position-reference labels; collect(destination) fixes the dataset specification, assigns disjoint splits, and saves arrays/provenance. Extend ik.py with solve_position(model, kinematics, target_position, q_seed) and nested objective(q), implementing full-joint SLSQP with an analytic position gradient and explicit joint constraints. The existing preview solver is unchanged. Add tests/test_collection.py with exactly three tests: test_general_motion_contract(), test_position_reference_contract(monkeypatch), and test_collection_archive(tmp_path, monkeypatch).

The addition budget is five callable definitions, zero new classes/methods/dataclasses/wrappers/helpers, and no CLI/configuration options. Reuse one Simulation/model/CPU workspace. Plain data have three groups: general records, planar records, and provenance. Independent numerical groups are reference joint states, general joint paths, plane geometry, and position-solver initial states/results. Function parameters are only those named above; destination is an explicit output path rather than an experiment setting.

The frozen dataset uses reference counts (48 training, 8 validation, 8 test), 65 samples over four seconds, and two planes per reference. Joint-1 reference regions are [-0.60, 0.10], [0.25, 0.40], and [0.55, 0.70] radians; other joints vary within 0.25 rad of home. Normal tilts are [0,45], [50,65], and [70,85] degrees. A fresh recorded run seed, seven cosine harmonics, and a 0.30 rad displacement bound define the recorded sampling specification. Split memberships are assigned before generation and shared by related general/planar episodes.

Retain compact, continuous logic, minimum realization, direct batched matrices, and professional one-line comments. No unnecessary helpers, fallback values/behavior, silent clipping, alternate solvers, or try/with/except constructs are permitted. Expected reference-solver failure is data: stop that ordered path, retain the failed candidate/status/cost, mark later samples unattempted, and exclude the whole failed planar episode from labels. Unexpected and non-finite failures raise errors. Undefined/unattempted array entries are explicitly marked NaN with masks, not replaced by a previous solution. Output is data/motions.npz and is never overwritten.

The collection addition is implemented and validated; see [motion-data collection](data_collection.md) for the exact command, schema, masks, and observed results.

## Repeatable collection command

The module entrypoint assigns each invocation a UTC timestamped data/motions_<timestamp>.npz path. The existing output variable is retained; one timestamp value is added, with no new functions, classes, helpers, parameters, configuration, fallback logic, or try/with/except constructs. collect(destination) still rejects an existing explicit destination. The original data/motions.npz archive remains unchanged. This changes only the entrypoint and its documentation, not sampling or solver behavior.

## Independent collection samples

Each collect(destination) invocation obtains seed = randbits(63) from OS entropy, records it in metadata, initializes the NumPy sampler with it, and passes it to generate_motions(sim, q_ref, seed) for the dedicated CUDA generator. This adds one required internal seed argument and a local seed value; no functions, classes, helpers, CLI/configuration fields, or fallback behavior are added. Existing tests verify equal-seed reproducibility, changed-seed motion variation, and distinct reference/plane/path samples across two controlled run seeds. Retain compact continuous logic and no try/with/except constructs. Timestamped archives and split domains remain unchanged.

## Original GPU neural-basis training budget

Add src/metaarm/basis.py with one MotionBasis class: __init__(joint_limits) constructs the fixed bounded neural matrices; forward(q_ref, J_ref, task, normal) returns the rank-preserving orthonormal pose/plane basis. Add src/metaarm/training.py with load_motions(source), evaluate(network, records, split_names), and train(source, destination). Loading validates the archive, masks, model provenance, and complete split groups; evaluation computes per-task reconstruction means and sampled reduced-Jacobian conditioning; fitting performs the balanced objective and saves final weights with metrics/provenance. Add tests/test_training.py with exactly test_basis_contract(), test_basis_gradients(), and test_training_checkpoint(tmp_path, monkeypatch).

The addition budget is two source files, one class, two methods, three top-level functions, and three tests. There are zero dataclasses, wrapper/helper layers, custom optimizers/datasets, or training frameworks. Owned objects are the GPU record dictionary, MotionBasis network, and Adam optimizer; other records/checkpoint contents are plain arrays and dictionaries. Independent data groups are reference configurations/Jacobians, motion displacements, plane normals, split/inclusion masks, and neural weights. Method/function inputs are exactly those named above.

The one required command argument is the explicit dataset path, necessary because collection archives have separate timestamped identities. There are no model, device, precision, seed, epoch, batch, or optimizer flags. The output receives a UTC timestamp under checkpoints and is never overwritten. A missing dataset is an error; no latest-file search or automatic collection is part of training.

Use the formulated 7-64-64-84 network with tanh after all affine layers, physical joint-limit input normalization, and 42/21/21 output partitions. Apply the right-inverse/nullspace construction with reduced QR and triangular solves, and normal projectors for the planar basis. Train the sum of the general and planar mean per-joint reconstruction errors. CUDA float32 is the explicit GPU implementation choice replacing the older CPU float64 training proposal; float64 CUDA tests verify the algebra and derivatives. There is no automatic precision/backend fallback or mixed precision.

Use seed 0, Adam learning rate 1e-3, and 500 complete passes. Each update pairs separate general/planar minibatches capped at 256, partitioned so each retained record is used once per pass. Validation does not select a checkpoint or stop training. Test records are evaluated only after fitting. Retain final weights, initial/final metrics, per-pass training/validation history, dataset identity and masks/counts, source/model hashes, package/GPU details, and sampled conditioning. Reconstruction performance and sampled rank are not IK success or workspace certificates.

Keep compact, continuous logic, direct batched matrices, minimum realization, and professional one-line block comments. Add no unnecessary helpers, configurations, fallback values/behavior, clipping, regularization of singular references, or try/with/except constructs. Reject invalid assumptions and nonfinite calculations explicitly. Exclude failed planar episodes before transferring their labels to the GPU rather than masking NaNs inside a loss.

The training addition is implemented and verified. Each optimizer update constructs the reference bases once for the training episodes and gathers the selected waypoint rows; repeated samples do not trigger repeated identical basis construction. The complete 500-pass run, final checkpoint, reconstruction metrics, late validation divergence, and numerical audits are documented in [GPU neural training](neural_training.md). Reduced coefficient IK remains separate work.

## Corpus-wide training improvement study

Use every NPZ archive under data/ while preserving its reference-linked train/validation/test groups. The current inventory has 18 archives and 1,152 distinct references. Do not discard existing archives or rewrite labels. Expected teacher failures remain excluded by their stored masks. Model selection uses validation only; the selected model receives the final test evaluation. The old test results are known historical observations, not a new untouched benchmark.

The revised structure budget is one MotionBasis class with __init__(joint_limits, specification) and forward(q_ref, geometry, task, normal, orthonormal=True); six top-level workflow/standard-operation functions: reference_geometry(J_ref, task) in basis.py, load_motions(source) in motion_data.py, evaluate(network, records, split_names) and train(source, destination, specification) in training.py, and diagnose(records, baseline) and study(source, destination) in study.py. There are zero new dataclasses, wrappers, custom dataset/optimizer classes, or automatic fallback paths. Add two corpus tests and retain/update the three training tests. The named numerical operation reference_geometry computes the reusable task QR/right inverse; it is not a wrapper layer.

Owned objects are the corpus tensor dictionary, one neural network, one AdamW optimizer/scheduler, and plain trial/checkpoint/result dictionaries. Independent data groups are reference configurations/Jacobians, motion second moments and sample counts, plane normals, archive/episode/split identities, and neural parameters. Model specification has four fields: width, scale, normal_features, equivariant. Training specification has three fields: epochs, learning_rate, weight_decay. Batch cap, device, numerical precision, and paths remain fixed implementation choices. The study command takes only the explicit corpus directory.

Covariance C = mean(delta_q delta_q.T) is a sufficient statistic for the fixed-reference reconstruction loss. Its loss is trace((I-P) C (I-P).T)/7, where P = F(F.T F)^{-1}F.T equals B B.T for the QR basis B. Float64 Cholesky solves compute the small projector matrices explicitly; the network remains CUDA float32. Validate loss and parameter gradients against the direct QR/sample formulation. Inference still returns reduced QR bases, and the right-inverse/nullspace construction retains the same exact-arithmetic local rank property. No pseudoinverse, rank truncation, or failed-solve replacement is permitted.

All samples contribute to second moments; task means are weighted by episode sample counts. Each pass visits all retained training episodes. Use a fixed cosine learning-rate schedule and explicit validation-selected weights for this improvement study, while retaining final-weight metrics/history. This selection policy intentionally replaces the older fixed-final-weight policy under the current request to vary training schemes. Test metrics do not select trials. Compare the original architecture on the full corpus, larger bounded nullspace corrections, nonlinear sign-invariant normal features, base-yaw equivariance, and capacity/regularization changes in several measured iterations.

Preserve compact continuous logic, batched matrices, minimum realization, and concise professional block comments. Keep source files around 50–250 lines and substantive functions around 20–100 lines. Add no unnecessary CLI/configuration, helpers, fallback values/behavior, or try/with/except constructs. State PCA bounds as per-episode unconstrained reference bounds, not achievable predictor accuracy or IK certificates. Preserve all archives and checkpoints and write separate study artifacts.

Ten controlled configurations were evaluated on the original 18-archive snapshot. Width 64, correction scale 32, the original projector conditioning, and no yaw canonicalization gave the lowest validation sum. Freeze that model and its 300-epoch cosine/AdamW recipe before generating two fresh archives. Refit on the resulting complete 20-file corpus and report both the full held-out groups and the fresh archive subset. Historical/interim test statistics were already examined; only the fresh subset is a new post-freeze audit. Do not describe the old test groups as previously unseen.

The completed study retained this exact four-file/six-function/one-class/two-method budget. The selected scale-32, width-64 model was refit on every one of 20 archives, selecting epoch 40 of 300. Same-corpus test loss changed from 0.08347872 to 0.00614023; the fresh two-archive test subset scored 0.00509447. Pose test error increased slightly, and reduced-Jacobian conditioning is lower, so reconstruction gain is not an IK-success claim. Full evidence, trial histories, manifest locations, and numerical verification are in [the improvement study](training_improvement_study.md). The original budget/policy above records the historical implementation; the corpus-study budget and selected policy are current.

## GPU coefficient IK and comparison budget

Implement the fixed-chart problem q = q_ref + B alpha with the trained audited checkpoint basis_20261031T105751490589Z_audited.pt. Preserve the coupled linear joint inequalities, the position residual, and the principal rotation-log residual with its actual derivative. The GPU request supersedes the original CPU SLSQP recipe: use batched damped Gauss-Newton with feasible constrained quadratic steps and explicit acceptance/damping updates. Compare full joints through B = I_7 using the identical implementation, tolerances, iteration budget, targets, and initialization. Never replace a failed reduced solve with a full-joint result.

The source budget is two new files, gpu_ik.py and evaluation.py, and changes to the existing __main__.py. Add one CoefficientIK class with exactly four methods: __init__(model, num_envs, checkpoint) owns a separate GPU kinematics workspace and verified trained network; geometry(q) evaluates batched tool FK and the geometric Jacobian; residual(q, target, task) evaluates normalized task residuals and their exact Jacobians; solve(q_ref, basis, target, task, alpha_initial) optimizes coefficients while freezing q_ref and basis. Add exactly two top-level workflow/standard-operation functions: constrained_step(hessian, gradient, basis, margin) solves the coupled joint-limit quadratic subproblem; evaluate_ik(source, destination) compares both methods on every stored held-out test trajectory. Existing main() and enqueue(event) handle UI startup and queued actions; add no callback/helper definitions. Add exactly three tests in test_gpu_ik.py: test_geometry_and_residual(), test_constrained_step(), and test_fixed_chart_solve(). Add zero dataclasses, wrappers, custom optimizer classes, or generic frameworks.

Owned objects are one CoefficientIK workspace/network per required batch size, the existing Simulation/UI/queue, and plain target/result/manifest dictionaries. Independent numerical groups are reference configurations, frozen bases, coefficient vectors, Cartesian target poses/plane geometry, and joint-limit constraints. Method/function parameters are exactly those named above; target contains position and rotation tensors, with orientation ignored for plane tasks. Model checkpoint, CUDA device, precision, iteration limits, damping schedule, step radius, and output locations are fixed source choices. The only solver tolerances are the formulated 1e-3 m and pi/180 rad. There are no new TOML fields or CLI flags; evaluation takes one explicit corpus directory. UI task/method selectors and plane direction are necessary experiment data.

The constrained quadratic operation first accepts the unconstrained minimizer only when it satisfies all joint inequalities; constrained rows use a primal-dual interior-point solve of the same quadratic program. The barrier belongs to the numerical QP algorithm, not an added task objective or relaxed joint constraint. Feasible step length and a fixed joint-space trust radius retain the affine polytope. Analytic small-angle limits in the SO(3) derivative are mathematical evaluations, not substitute behaviors. Reject the rotation-log cut, malformed tensors, nonfinite numerics, and incompatible assets explicitly.

Evaluate all test reference/plane trajectories from every NPZ archive, retain local identities qualified by source, freeze each basis for all waypoints, warm-start coefficients, stop each method's trajectory on its first task failure, and record failed candidates plus unattempted masks. Report episode completion, planned/attempted/successful waypoint counts, raw position/angular/plane errors, joint margins, statuses/iterations, and synchronized batch timing separately from model/basis preparation. Audit attempted candidates using independent CPU MuJoCo. Historical held-out groups were previously inspected; this is a matched comparison, not a previously untouched test claim.

The UI explicitly compares both methods at the same current reference and applies only the selected successful result. A failed selected result pauses the experiment and displays its residual/status; it never applies the other method's solution. Plane mode projects the requested point onto the displayed plane through the current tool, leaves orientation free, and reports normal error. Keep IK prediction and physical tracking telemetry separate. Solving uses independent GPU workspace and never resets simulation state.

Retain compact continuous logic, minimum realization, batched matrices, professional concise block comments, files near 50-250 lines, and substantive functions near 20-100 lines. Add no unnecessary helpers, configuration, fallback behavior, clipping of joint states, or try/with/except constructs. Test FK/Jacobians against CPU MuJoCo and orientation derivatives against finite differences, constrained steps against an independent convex solve, known feasible and unreachable reduced/full targets, fixed-chart affine reconstruction, and unchanged simulation state. Verify UI task/method controls, application/failure, run/pause/reset, and shutdown.

The coefficient-IK implementation retains the declared two-file/one-class/four-method/two-function structure and three added tests. The full suite has 14 passing tests. Every stored test trajectory in all 20 archives was compared; learned/full completion was 88/160 versus 160/160 pose trajectories and 19/320 versus 320/320 plane trajectories. Independent CPU task-error audits agreed on every success label; sixteen SLSQP refinements retained task failures, and failed-chart Jacobians were usually nearly singular in the joint interior. The live UI supports both task/method selections, explicit failure display, physical tracking, reset, and shutdown. Current usage, timings, and limitations are recorded in [GPU coefficient IK](coefficient_ik.md).

## Local Cartesian workspace pipeline budget

The task is three-dimensional tool position with free orientation and three learned coefficients. Select an interior axis-aligned cube near (0.50, 0, 0.50) m using independent full-joint corner/center probes before generating data. Freeze the chosen box; all saved reference positions, path targets, and successful label FK positions belong to it. The new scoped dataset and artifacts live together in results/workspace_<UTC>/, keeping the existing broad corpus and its commands unchanged. Earlier broad-region samples do not enter this local fit.

Add exactly three source files. workspace_data.py defines lift_positions(model, targets, q_center), the seven-variable minimum-distance position teacher, and collect_workspace(destination), the bounded dataset constructor. lift_positions uses exactly three necessary SciPy numerical callbacks: joint_distance(q), position_constraint(q), and position_jacobian(q). They express the objective and equality constraint/derivative directly, without helper layers. position_basis.py defines one PositionBasis class with exactly __init__(joint_limits, specification), forward(q_ref, geometry, orthonormal=True), and reconstruction(q_ref, geometry, covariance). They construct a 7-width-width-21 bounded network, the rank-preserving three-column basis, and the exact uncentered covariance reconstruction loss. workspace.py defines train_workspace(source, destination, specification) and validate_workspace(source, checkpoint, split, destination). Its module entrypoint directly executes collection, controlled training/validation, selection, and the final test; add no main/helper function. The public function budget is four, numerical callbacks three, new classes one, methods three, and dataclasses/wrappers/frameworks zero. Add exactly three tests in test_workspace.py: test_workspace_collection(), test_position_basis(), test_workspace_checkpoint().

Reuse CoefficientIK geometry/constraint optimization and reference_geometry. Extend their existing task selectors to explicit position-only volume tasks. The checkpoint constructor explicitly dispatches the two supported neural output dimensions (84 for the existing pose/plane network, 21 for PositionBasis) and rejects unsupported architectures; this is model-format dispatch, never runtime fallback. Preserve all existing method signatures, old checkpoints, broad collection/training behavior, and dependencies. UI changes are not needed for this data/training/validation request.

Owned objects are one CPU MuJoCo teacher workspace, a PositionBasis network and AdamW optimizer, existing GPU IK workspaces at the required batch sizes, and plain dataset/metric/checkpoint dictionaries. Independent data groups are the declared Cartesian box/targets, reference joint configurations/Jacobians, full-joint demonstrations, neural parameters, and coefficient vectors. All callable inputs are exactly those named above. The model specification has only width and scale; epochs, seed, precision, learning-rate schedule, tolerances, batch policy, paths, and dataset sizes remain source constants, not CLI/configuration options. The one-command regional workflow has no arguments.

The teacher minimizes half the squared distance to one center configuration subject to exact Cartesian position and joint limits, independently for each unique target. All seven joints are free. This produces a consistent local lifting without generating labels from a preselected three-dimensional basis. Probe the frozen center/corners and record initialization cost. Dataset reference counts are 96/24/48 train/validation/test with disjoint complete references, fixed seed, and 33-point closed quintic polylines inside the convex box. Deduplicate exact target coordinates before full-joint labeling, preserving all path/sample multiplicities afterward. Record solver status, residual, joint margin, inclusion masks, asset hashes, and source provenance. Exclude complete failed teacher paths explicitly and retain their records; do not resample failures or move targets outside the box.

Use every included regional training waypoint through the exact uncentered second moment, and preserve complete validation/test groups. Fit four declared models: width/scale (32, 0.25), (32, 1), (32, 4), and (64, 1), with 500 AdamW epochs and the fixed cosine recipe. Select weights using validation reconstruction error, then select the model by validation trajectory completion first and reconstruction error second. Keep all histories and last/selected states. Test the selected model once after this selection.

Compare learned, geometric right-inverse, and full-joint identity bases with the identical GPU coefficient optimizer. Freeze each chart through its trajectory, warm-start coefficients, stop at first failure, and preserve attempted/unattempted records. The final test additionally probes a 9x9x9 grid spanning the full declared cube from the center plus eight predetermined held-out references, using zero coefficients for each independent target. Report whole-trajectory completion, planned/attempted/successful counts, raw errors, reconstruction MSE, joint margins, sampled task-image singular values, and separated synchronized timings. Independently audit attempted candidates with CPU MuJoCo. A successful finite grid is sampled evidence, not a volume-coverage certificate.

Retain compact continuous logic, minimum realization, batched matrices, concise professional key-block comments, and approximately 50-250 line source files. Do not add unnecessary CLI/configuration, short wrapper/helper layers, fallback behavior, clipping, or try/with/except constructs. Tests verify regional containment, disjoint identities/teacher masks, minimum-motion full-joint labels, basis/gradient identities, checkpoint reload/selection, fixed-chart reconstruction, and a small regional IK comparison.

Keep the same three-new-file/four-public-function/one-new-class budget while allocating code for readable file lengths: place PositionBasis in the existing basis.py, keep data in workspace_data.py, fitting/orchestration in workspace.py, and validate_workspace in workspace_validation.py. The callable names, roles, inputs, and counts above are unchanged; position_basis.py is not retained as a separate file. The verified frozen box is x=[0.42,0.58], y=[-0.08,0.08], z=[0.42,0.58] metres. All eight full-joint corner probes succeeded with position errors below 3.3e-13 m, joint margins above 0.84 rad, and positional-Jacobian singular values above 0.23.

The regional run is complete within the declared three-new-file/four-public-function/three-numerical-callback/one-new-class/three-method/three-test budget. The selected width-32/scale-0.25 model has 2,005 parameters and selects epoch 120. Data contain 168 regional paths; two full-joint minimum-distance labels hit the iteration limit and their complete paths remain recorded/excluded from reconstruction. The fit uses all 3,135 included training waypoints. All 48 test trajectories and all 6,561 grid solves succeed at 1 mm, including the test target path whose teacher status was excluded. The geometric baseline also succeeds throughout. The single fixed center chart is exported and succeeds on all 729 cube-grid points. All 17 tests and lint/format checks pass. Full scope, provenance, baselines, and sampled-evidence limits are recorded in [the local workspace study](local_workspace.md).

## Workspace expansion and position/pose experiments

Treat each (cube side, task) pair as a separate data/training/validation/test experiment. Cube sides are 0.16, 0.24, 0.32, 0.40, and 0.48 m around (0.50,0,0.50) m. The user selected three-dimensional position with free orientation versus full six-dimensional pose, not different coefficient counts for an identical task. Full pose adds body rotation-vector coordinates in a fixed +/-10 degree per-axis range about the center rotation; position uses the identical Cartesian paths without orientation constraints. Each task is compared against its own seven-joint baseline. Cross-task time differences cannot be attributed solely to coordinate count.

Use common safely interior reference configurations across sizes and tasks: positions in the central 8 cm cube and rotations within +/-5 degrees, assigned before fitting with fixed seed. Generate references with full seven-joint pose IK. Construct 96/24/48 train/validation/test closed 33-point trajectories per experiment. For each trajectory, independently label each target with full seven-joint minimum distance to that trajectory's reference, ensuring zero endpoint displacement without sequential nullspace drift. All Cartesian targets stay in the selected cube. Retain failed targets/corner probes and complete-path inclusion masks; do not shrink cubes, move targets, or resample failures. This new protocol reruns the 16 cm baseline rather than comparing it directly with the earlier fixed-center teacher.

The new source-file budget is three: expansion_data.py, expansion_evaluation.py, and expansion.py. Add exactly five top-level functions: lift_pose(model, targets, rotations, q_ref) expresses the full-joint pose teacher; collect_experiment(destination, side, task) constructs one scoped experiment; validate_experiment(source, checkpoint, split, destination) compares fixed-chart trajectories and the test lattice; benchmark_search(source, checkpoint, destination) measures repeated matched search costs; run_expansion(destination) performs independent adaptive studies and final testing. lift_pose uses three required numerical callbacks, joint_distance(q), pose_constraint(q), and pose_jacobian(q). Add zero wrapper/helper layers, dataclasses, custom optimizer classes, or frameworks.

Generalize the existing PositionBasis class to TaskBasis with exactly three methods: __init__(joint_limits, specification, task, depth), forward(q_ref, geometry, orthonormal=True), and reconstruction(q_ref, geometry, covariance). Explicit task selects three or six columns, and depth selects two or three hidden tanh layers. Existing 21-output checkpoint weights remain readable; GPU checkpoint loading derives supported output dimension and hidden-layer count from actual weight matrices, and retains the existing 84-output broad model. Update all current internal callers/tests explicitly; add no compatibility wrapper or default task/depth. Extend the existing train_workspace(source, destination, specification, training) with a required experiment training dictionary. Existing position-only entrypoints pass their frozen recipe explicitly. Training remains batched CUDA covariance reconstruction, with an optional declared mobility scheme adding a sampled teacher-state task-image singular-value penalty; log both objective and raw MSE. TaskBasis retains two model fields, width and scale. Training has exactly five fields: depth, epochs, learning_rate, weight_decay, scheme. Paths, seeds, devices, precision, tolerances, sampling counts, timing repetitions, and schedules are fixed source choices, not CLI flags/configuration.

Each pair begins with four independent architecture/scale recipes, then performs two refinements around its own validation winner with different duration/rate/regularization or mobility objective. Select each trial's weights on its validation objective; rank models by validation whole-path completion, successful planned waypoints, and reconstruction error. Preserve every checkpoint/history and trial result. Final tests are evaluated only after the pair's search finishes. No trial selection or further training uses its test results. Record available converged training labels and full-joint failures per pair.

Owned objects are full-joint CPU teacher data, one TaskBasis/AdamW/scheduler per trial, existing GPU IK workspaces, and plain experiment/checkpoint/result dictionaries. Independent numerical groups are declared Cartesian/orientation targets, fixed reference states/Jacobians, full-joint demonstrations, learned weights, and coefficient vectors. The function and method inputs are exactly those named above. Add exactly three tests in test_expansion.py: test_pose_teacher(), test_task_basis(), and test_expansion_pipeline(); retain the existing suite. No new dependencies are needed for computation.

Measure search-space reduction as actual independent coordinates (7 to 3 or 6), the unchanged fourteen coupled joint inequalities, and reduced dense system size k squared. Do not compare volumes of spaces with different dimensions or present a hypothetical grid reduction as measured solver work. Save iteration counts and objective discrepancies. Search-time measurements use matched independent targets and zero initial coefficients, separately from early-stopped trajectories: warmed 256-target GPU batches and sixteen predetermined single-target cases, five batch and three single-case timing repeats, with identical cases for reduced/full methods. Record medians, spread, success/iteration counts, search-only cost, basis preparation, and preparation-inclusive cost. Report both all paired cases and the jointly successful subset; failures do not silently disappear from the primary timing result. The final lattice spans 9x9x9 Cartesian locations from the center plus eight held-out references; pose assigns declared rotations to those targets and is not called a full six-dimensional grid.

Preserve compact continuous mathematical logic, minimum realization, professional one-line key-block comments, source files near 50-250 lines, and a handful of substantive operations. Avoid unnecessary configuration/CLI, short wrapper helpers, fallback values/behavior, clipping, and try/with/except constructs. Independently check pose teacher residuals/geometry, three/six-column rank and loss/gradient identities, masked labels, checkpoint reload, timing dimensions/case matching, and CPU FK errors. A finite lattice is sampled evidence rather than a coverage proof. Report exact reduction percentages and measured speedups or slowdowns without conflating batch throughput and single-target latency.

Reuse the existing regional validator by moving it to expansion_evaluation.py and naming it validate_experiment; update internal imports directly and retain its original position-only entrypoint behavior. This avoids a duplicated validator or compatibility wrapper and stays within the three-added-file budget. Timing and orchestration share expansion.py. TaskBasis replaces PositionBasis directly; both historical 21-output weights and broad 84-output checkpoints remain supported by explicit weight-structure dispatch.

The expansion study completed ten independent configurations and sixty training trials within the declared callable/class budget. Position retained every sampled lattice target through 40 cm; pose declined earlier, largely matching full-joint IK. Cached position search reduced median single-case time by up to 29.2%; preparation reduced that gain, and pose preparation-inclusive search remained slower. All 20 tests and source checks pass. Model selections, discrepancies, exact space reductions, timing, and figure exports are in [the expansion study](workspace_expansion.md).
