# MetaArm

MetaArm studies inverse kinematics for a seven degree of freedom robot arm using a small vector of coefficients that combines learned joint motion patterns. The reference platform is the Franka Research 3 or a comparable seven joint manipulator.

A neural network learns a reusable motion basis from general robot motions. Online inverse kinematics searches the coefficient vector rather than all seven joint coordinates. The two primary experiments use six coefficients for general end effector pose motion and three coefficients for nominal planar tasks.

## Research formulation

A candidate affine reconstruction is

$$
q = q_{\mathrm{ref}} + \Phi_{\theta,k}(x)\alpha,
\qquad \Phi_{\theta,k}(x)\in\mathbb{R}^{7\times k},
\qquad \alpha\in\mathbb{R}^{k},
\qquad k\in\{6,3\}.
$$

Here, $q$ is the joint configuration, $q_{\mathrm{ref}}$ is the reference configuration, $x$ is the network input, and $\theta$ denotes the learned network weights. The precise input and update policy remain research design decisions. For a reconstruction that is affine in $\alpha$, the basis input is held fixed during each inverse kinematics solve.

| Experiment | Search dimension | Task |
| --- | --- | --- |
| General motion | 6 | End effector position and orientation |
| Nominal planar motion | 3 | Positions on a specified plane, with free orientation as the working assumption |
| Plane conditioning | 3 | Construct the planar basis using the plane normal in $\mathbb{R}^{3}$ |

Each reduced solve optimizes one coefficient vector. Joint limits apply to the reconstructed configuration. The proposed plane-normal multiplication requires a dimensionally explicit learned representation and a task-space rank analysis.

The theoretical objective is to establish conditions under which the learned motion directions preserve all degrees of freedom required by the task. The rank of the joint basis alone does not establish this property. Local differential mobility and coverage of a finite target region require separate arguments.

## Project documents

- [Research brief](docs/research_brief.md): project scope, candidate formulation, proof objectives, and experiment questions.
- [Developer instructions](AGENTS.md): coding discipline and planning requirements.
- [Original handoff](docs/references/Reduced_Motion_Basis_IK_Developer_Handoff.md): supplied background reference, preserved unchanged.

The research brief follows the project owner's description. The original handoff is reference material. Its fixed-basis priority, delayed neural learning, and prescribed implementation hierarchy do not override the project's learned-basis objective or the developer instructions.

## Research status

This repository contains the project definition and development instructions. Robot models, training code, inverse kinematics solvers, experiments, and project-specific proofs have not been implemented. No physical validation or workspace coverage result is claimed.

## Inspiration

Michael O'Connell et al., *Neural-Fly enables rapid learning for agile flight in strong winds*, Science Robotics 7(66), eabm6597, 2022. [DOI](https://doi.org/10.1126/scirobotics.abm6597) · [Caltech record](https://authors.library.caltech.edu/records/q3grb-3vz72).

Neural-Fly motivates learning a shared representation offline and adapting a small set of linear coefficients online. MetaArm investigates this pattern for joint motion and inverse kinematics; its kinematic guarantees require an independent analysis.
