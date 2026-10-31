# MetaArm

MetaArm studies inverse kinematics for a seven degree of freedom robot arm using a small vector of coefficients that combines learned joint motion patterns. The reference platform is the Franka Research 3 or a comparable seven joint manipulator.

A neural network learns a reusable motion basis from general robot motions. Online inverse kinematics searches the coefficient vector rather than all seven joint coordinates. The two primary experiments use six coefficients for general end effector pose motion and three coefficients for nominal planar tasks.

## GPU simulation

Run the experiment from this repository in PowerShell:

```powershell
uv run --locked python -m metaarm
```

Open [the local experiment UI](http://127.0.0.1:8080). The FR3 starts paused with 64 GPU worlds. Choose **Pose** or **Plane**, select **Learned coefficients** or **Full joints**, then press **Apply target** to compare both methods from the current reference. Only a successful result from the selected method is applied. **Run / Pause** controls tracking; Reset restores home/time/controls and Quit stops the server. Plane targets are projected onto the displayed plane with free orientation. Predicted IK and physical tracking errors are reported separately.

The project environment is already installed. To recreate it, install uv and run uv sync --locked. See [setup and validation](docs/simulation_setup.md) for the locked CUDA stack, model provenance, tests, measured performance, and physical-model limits.

## Motion-data collection

Generate the fixed general/planar kinematic dataset with:

```powershell
uv run --locked python -m metaarm.collection
```

Each invocation writes a new data/motions_<UTC timestamp>.npz archive with episode/reference splits, joint displacements, reference Jacobians, projected plane geometry, reference-solver failures, and provenance. Each run uses a fresh recorded random seed. Earlier archives are preserved; the new path and seed are printed on completion. See [data collection](docs/data_collection.md) for the sampling specification, schema, inclusion masks, and evidence boundaries.

## GPU neural motion-basis training

Train the selected motion-basis model on every archive under data/:

```powershell
uv run --locked python -m metaarm.training data
```

The trainer preserves archive/reference splits and uses every retained waypoint through the exact second-moment reconstruction loss. It runs 300 complete CUDA epochs with the selected 64-unit, scale-32 basis, then saves validation-selected and final-epoch weights with the full source manifest. See [neural training](docs/neural_training.md) for the equations and checkpoint API. The [ten-configuration study](docs/training_improvement_study.md) reduced common-corpus test loss from 0.08348 to 0.00614 using all 20 archives; the gain is planar reconstruction, while pose error increased slightly.

## GPU coefficient-space IK

Evaluate both methods on every stored held-out trajectory from all data archives:

~~~powershell
uv run --locked python -m metaarm.evaluation data
~~~

The same CUDA constrained optimizer uses six learned pose coefficients, three learned plane coefficients, or seven full-joint coordinates. References and bases remain fixed within each trajectory. On the 20-archive benchmark, learned/full-joint trajectory completion was 55%/100% for pose and 5.94%/100% for plane. The learned charts also took longer to solve. See [coefficient IK and comparison](docs/coefficient_ik.md) for equations, errors, timings, failure/rank audits, and saved records.

## Local 3D position workspace

Run a complete regional data → training → validation/test experiment:

~~~powershell
uv run --locked python -m metaarm.workspace
~~~

The fixed volume is x=0.42–0.58 m, y=-0.08–0.08 m, z=0.42–0.58 m, with free orientation and three coefficients. The selected model completed all 48 held-out paths and 6,561 independent grid solves at 1 mm tolerance. The geometric right-inverse baseline also completed all targets. See [the regional study](docs/local_workspace.md) for the dataset, model, exported fixed center chart, comparisons, and evidence limits. The existing UI remains the general pose/plane experiment.

## Workspace-size position/pose experiments

~~~powershell
uv run --locked python -m metaarm.expansion
~~~

This independently tunes 3D position and 6D pose models for 16–48 cm cubes and measures held-out success, motion discrepancies, and repeated search times against seven-joint IK. See [the expansion study](docs/workspace_expansion.md) for all ten configurations, sixty training runs, coordinate reductions, and cached/preparation-inclusive timing.

## Research formulation

A candidate affine reconstruction is

$$
q = q_{\mathrm{ref}} + \Phi_{\theta,k}(x)\alpha,
\qquad \Phi_{\theta,k}(x)\in\mathbb{R}^{7\times k},
\qquad \alpha\in\mathbb{R}^{k},
\qquad k\in\{6,3\}.
$$

Here, $q$ is the joint configuration, $q_{\mathrm{ref}}$ is the reference configuration, $x$ is the network input, and $\theta$ denotes the learned network weights. The implemented network uses the joint-limit-normalized reference configuration as input and a fixed basis within an episode. Alternative input and update policies remain research choices; affine reconstruction requires holding the basis input fixed during each solve.

| Experiment | Search dimension | Task |
| --- | --- | --- |
| General motion | 6 | End effector position and orientation |
| Nominal planar motion | 3 | Positions on a specified plane, with free orientation as the working assumption |
| Plane conditioning | 3 | Construct the planar basis using the plane normal in $\mathbb{R}^{3}$ |

Each reduced solve optimizes one coefficient vector. Joint limits apply to the reconstructed configuration. The proposed plane-normal multiplication requires a dimensionally explicit learned representation and a task-space rank analysis.

The theoretical objective is to establish conditions under which the learned motion directions preserve all degrees of freedom required by the task. The rank of the joint basis alone does not establish this property. Local differential mobility and coverage of a finite target region require separate arguments.

## Project documents

- [Development foundation](docs/DEVELOPMENT_PLAN.md): environment, file structure, and implementation budget.
- [Simulation setup and validation](docs/simulation_setup.md): one-command launch and verified GPU behavior.
- [GPU neural training](docs/neural_training.md): basis construction, all-corpus fitting, and checkpoint API.
- [Training improvement study](docs/training_improvement_study.md): diagnosis, ten measured configurations, and full/fresh test results.
- [GPU coefficient IK](docs/coefficient_ik.md): constrained solver, experiment controls, and matched held-out comparisons.
- [Local workspace basis](docs/local_workspace.md): scoped position data, selected neural model, and volume-grid validation.
- [Larger-workspace pose IK](docs/pose_ik_improvement.md): eighteen neural fits, cooperative coefficient charts, paired full-joint timing, and fresh-target audits.
- [Analytical FR3 singularities](docs/fr3_pose_singularities.md): exact Jacobian rank-loss family and admissible workspace witness.
- [Research brief](docs/research_brief.md): project scope, candidate formulation, proof objectives, and experiment questions.
- [Developer instructions](AGENTS.md): coding discipline and planning requirements.
- [Original handoff](docs/references/Reduced_Motion_Basis_IK_Developer_Handoff.md): supplied background reference, preserved unchanged.

The research brief follows the project owner's description. The original handoff is reference material. Its fixed-basis priority, delayed neural learning, and prescribed implementation hierarchy do not override the project's learned-basis objective or the developer instructions.

## Research status

The locked GPU environment, pinned FR3 model, MuJoCo Warp simulation, reference full-joint IK, and experiment UI are implemented and numerically checked. General/planar kinematic data collection is implemented. GPU neural motion-basis training is implemented. GPU coefficient IK and the matched full-joint comparison are implemented and integrated into the UI; the trained fixed chart has limited finite-trajectory success. Workspace studies report sampled coverage only; continuous workspace coverage and hardware validation remain unestablished.

## Inspiration

Michael O'Connell et al., *Neural-Fly enables rapid learning for agile flight in strong winds*, Science Robotics 7(66), eabm6597, 2022. [DOI](https://doi.org/10.1126/scirobotics.abm6597) · [Caltech record](https://authors.library.caltech.edu/records/q3grb-3vz72).

Neural-Fly motivates learning a shared representation offline and adapting a small set of linear coefficients online. MetaArm investigates this pattern for joint motion and inverse kinematics; its kinematic guarantees require an independent analysis.
