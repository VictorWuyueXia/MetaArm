# MetaArm research brief

## Project question

Can a neural representation of coordinated joint motion reduce inverse kinematics to six coefficients for general pose motion and three coefficients for nominal planar tasks while preserving the required task-space mobility?

The robot has seven revolute joints. Franka Research 3 is the intended reference platform, with a comparable seven joint arm acceptable. The exact robot model, joint ordering, base frame, and tool point must be established before implementation.

## Learned representation and coefficient search

Let $q\in\mathbb{R}^{7}$ denote joint configuration. A neural network with weights $\theta$ maps an input $x$ to a basis $\Phi_{\theta,k}(x)\in\mathbb{R}^{7\times k}$, with $k=6$ or $k=3$. General robot motions provide the training data. The primary method is a learned basis; analytical and data-linear bases serve as comparisons.

The working affine reconstruction is

$$
q(\alpha;x)=q_{\mathrm{ref}}+\Phi_{\theta,k}(x)\alpha,
\qquad \alpha\in\mathbb{R}^{k}.
$$

Online inverse kinematics optimizes one coefficient vector $\alpha$ for the target motion while enforcing the robot's joint constraints on $q(\alpha;x)$. The network weights are learned offline. The coefficient dimension is six or three; a vector itself does not have matrix rank six or three. Full column rank is a property to establish for the basis.

The input $x$, its relation to the reference state, and its update frequency remain open. Holding $x$ fixed within a solve preserves the affine reconstruction. A state-dependent basis may be evaluated between solves. Freezing one basis for a complete trajectory is a distinct experimental restriction, not an assumption inherited from the handoff. If $x$ depends on the optimized configuration within a solve, the decoder and its derivative must be reformulated explicitly.

## General pose motion

The six-coefficient experiment targets all three translations and all three rotations of the end effector. Let $J_6(q)\in\mathbb{R}^{6\times7}$ be the geometric pose Jacobian in a declared frame and twist convention. With the basis input fixed, the local task map has full differential rank when

$$
\operatorname{rank}\bigl(J_6(q)\Phi_{\theta,6}(x)\bigr)=6.
$$

Pose errors must report translation in metres and rotation in radians separately. A mixed pose objective requires an explicit normalization. Joint limits constrain the reconstructed seven joint configuration, not independent coefficient intervals unless equivalence is proved.

## Nominal planes and normal conditioning

The three-coefficient experiment targets positions on a specified nominal plane. End effector orientation is free as a working interpretation of a two-dimensional positional task; an orientation requirement changes the task dimension and must be stated explicitly.

A unit normal $n\in\mathbb{R}^{3}$ specifies the plane tangent directions. A plane origin is also needed to locate the plane, and in-plane axes define an oriented raster. The normal alone does not specify these quantities. Normals $n$ and $-n$ describe the same unoriented plane.

The proposed multiplication by the normal needs a representation with compatible dimensions. One candidate is a neural tensor contraction,

$$
\Phi_{\theta,3}(x,n)=\sum_{r=1}^{3}n_r A_{\theta,r}(x),
\qquad A_{\theta,r}(x)\in\mathbb{R}^{7\times3},
$$

where $n_r$ is the $r$th normal component and $A_{\theta,r}$ is the corresponding learned matrix slice. This candidate yields one $7\times3$ basis for one three-coefficient solve. It is a research hypothesis, not a selected architecture or a proved transfer rule. Direct multiplication of a $7\times3$ basis by a three-vector yields only one joint direction and does not provide three independent coefficient directions.

Let $J_p(q)\in\mathbb{R}^{3\times7}$ be the tool position Jacobian. Full rank of $J_p\Phi_{\theta,3}$ is sufficient for local three-dimensional positional mobility and hence local planar targeting. A nominal plane has two intrinsic positional degrees of freedom. A weaker plane-specific guarantee must establish that the reduced Cartesian image contains the plane tangent space while respecting the plane constraint. The third coefficient remains part of the requested experiment.

Normal conditioning must be tested across unseen plane orientations and origins. Linear contraction, nonlinear conditioning, and Jacobian-based transfer are different constructions. The handoff's Jacobian transfer formula does not resolve whether the requested normal-based construction works.

## Proof objectives and evidence

For a full column rank basis with a fixed input, the starting identity is

$$
\operatorname{rank}(J\Phi)=k-\dim\bigl(\operatorname{Im}\Phi\cap\ker J\bigr),
$$

where $J$ is the relevant task Jacobian, $\operatorname{Im}$ denotes image, and $\ker$ denotes kernel. The identity follows by applying rank-nullity to the restriction of $J$ to the basis image. It identifies the task-relevant property that learning and analysis must preserve.

The research aims to establish task-space mobility over a declared operating domain. The proof must state the robot configuration domain, basis input policy, task definition, and regularity assumptions. Local full rank gives a local result in task coordinates at an interior feasible point. Rank measurements on sampled configurations are numerical evidence. A uniform claim requires bounds over the complete declared domain.

Coverage of finite pose or planar target sets is a separate question. Full local rank alone does not establish global reachability, compatibility with joint limits, collision avoidance, or a computational speed advantage. A normal-linear construction must also be checked for structural rank obstructions before claiming applicability to every plane.

## Experimental questions

| Question | Comparison and evidence |
| --- | --- |
| Does the learned six-coefficient basis preserve general pose targeting? | Compare with full seven joint inverse kinematics and simple reduced baselines on held-out poses and trajectories. |
| Does the learned three-coefficient basis support nominal planar motion? | Evaluate independent point targets and continuous planar trajectories across plane sizes, positions, and orientations. |
| Does normal conditioning construct a useful basis for unseen planes? | Compare the normal-conditioned basis with an unconditioned learned basis and independently acquired plane bases. |
| Where do rank and finite coverage differ? | Inspect reduced Jacobians, joint-limit margins, target failures, and distance from the reference configuration. |

Useful reduced comparisons are a frozen Jacobian singular-vector basis and a motion-data principal-component basis. These comparisons support evaluation of the neural representation without replacing it as the primary research method. Record training, basis evaluation, initialization, and coefficient optimization costs separately. Use the same robot, targets, constraints, and task tolerances across methods.

The implementation sequence is robot kinematics and Jacobian verification, matched full and reduced inverse kinematics, general-motion data and neural basis training, six- and three-coefficient evaluations, normal conditioning, and regional analysis. No software architecture, dependency stack, solver, or numerical tolerance is selected by this document.

## Relationship to the supplied handoff

The original handoff is retained unchanged in [the reference directory](references/Reduced_Motion_Basis_IK_Developer_Handoff.md). Its rank distinctions and evaluation cautions inform this brief. Its fixed affine subspace requirement, deferred neural model, analytical transfer priority, extensive module tree, and configuration proposals are not adopted as project instructions. The project owner's description and root developer instructions govern those choices.

The conceptual inspiration is [Neural-Fly](https://authors.library.caltech.edu/records/q3grb-3vz72), DOI [10.1126/scirobotics.abm6597](https://doi.org/10.1126/scirobotics.abm6597): learn a reusable representation and adapt a small coefficient vector. MetaArm's representation concerns joint motion and inverse kinematics, so its task-space coverage analysis is independent of the flight-control results.
