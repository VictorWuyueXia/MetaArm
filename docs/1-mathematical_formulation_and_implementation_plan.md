# MetaArm: mathematical formulation and minimal implementation plan

Implementation note: this document records the original mathematical plan. The current GPU package layout, exact all-sample covariance loss, bounded correction scale 32, and validation-selected training policy are specified in [neural training](neural_training.md) and the [measured improvement study](training_improvement_study.md). Those measured implementation choices supersede the original CPU/fixed-final-weight recipe and flat file budget below; the reconstruction objective and reference-rank construction remain the same.

This document proposes a concrete formulation and an exact implementation budget. It specializes [the research brief](research_brief.md). The primary method learns coordinated joint motion from robot motion data and searches one coefficient vector for inverse kinematics. The supplied handoff remains background reference material.

## 1. Robot, task, and independent variables

Let the seven revolute joint coordinates be $q\in\mathcal Q\subset\mathbb R^7$, with $\mathcal Q=\{q:q_{\min}\le q\le q_{\max}\}$. The forward kinematics of the declared tool point are its position $p(q)\in\mathbb R^3$ and rotation $R(q)\in SO(3)$. The geometric Jacobian uses tool-point linear velocity and angular velocity, both expressed in base axes,

$$
\begin{bmatrix}\dot p\\\omega\end{bmatrix}
=J_6(q)\dot q,
\qquad J_6(q)=\begin{bmatrix}J_p(q)\\J_\omega(q)\end{bmatrix}\in\mathbb R^{6\times7}.
$$

The implementation is nominal kinematics with joint limits. It does not include dynamics, collision constraints, or hardware execution. The robot asset must identify the exact FR3 model revision, seven arm joints, base frame, tool frame, fixed tool transform, and joint limits. These are fixed model facts, not experiment settings. The model comes from the [official Franka description repository](https://github.com/frankarobotics/franka_description); the exact asset and tool frame must be verified before implementing kinematics.

| Independent quantity | Role |
| --- | --- |
| $q_0\in\operatorname{int}\mathcal Q$ | Reference configuration and neural input source; fixed for one motion episode. |
| $p_d,R_d$ | Desired tool position and, for the general task, desired rotation. |
| $c,n,E$ | Plane origin, unit normal, and in-plane axes; supplied task data. |
| $\theta$ | Neural weights; optimized offline and frozen during evaluation. |
| $\alpha\in\mathbb R^k$ | The only online decision variable; $k=6$ for pose and $k=3$ for planar position. |

All other quantities introduced here are derived from these inputs, the robot model, or recorded motion data. A coefficient vector has dimension $k$; matrix rank applies to the basis and its task-space image.

## 2. Affine motion reconstruction and basis policy

Define the normalized network input componentwise by

$$
x=2\,(q_0-q_{\min})\oslash(q_{\max}-q_{\min})-\mathbf1\in\mathbb R^7,
$$

where $\oslash$ denotes componentwise division. The reconstructed configuration is

$$
q(\alpha)=q_0+B_k\alpha,
\qquad B_6=\Phi_{\theta,6}(x),
\qquad B_3=\Phi_{\theta,3}(x,n),
\qquad B_k^TB_k=I_k.
$$

An episode is one independent target or an ordered target trajectory. Evaluate the network and construct $B_k$ once at its reference configuration. Freeze $q_0$, $x$, and $B_k$ throughout the episode; update only $\alpha$. For a sequence, warm-start the next solve with the previous coefficient vector. This fixed-chart benchmark is an explicit experimental choice. State-dependent updates, reanchoring, chart switching, and a basis depending on the current optimization iterate are outside the minimal implementation.

This policy gives $\partial q/\partial\alpha=B_k$ exactly and prevents a full joint search from being hidden inside repeated reference updates. A new episode may use a different reference and hence a different learned basis. Initialization on a translated plane may use one full joint solve before the episode; its cost is recorded separately.

## 3. A learned basis with an analytical rank constraint

An arbitrary neural matrix followed by orthogonalization guarantees joint-space independence only. The proposed construction constrains the neural output so that task-space rank is established at the reference configuration.

Use $m=6$ with $J_m=J_6$ for pose and $m=3$ with $J_m=J_p$ for position. For conditioning statements, express translation in units of $1\,\mathrm m$ and rotation in units of $1\,\mathrm{rad}$. Denote the resulting dimensionless task Jacobian by $\bar J_m$; these fixed coordinate scales do not depend on the experiment tolerance.

Assume $\bar J_m(q_0)$ has full row rank. Its reduced QR factorization defines

$$
\bar J_m(q_0)^T=V_mH_m,
\qquad V_m^TV_m=I_m,
\qquad G_m=V_mH_m^{-T},
\qquad N_m=I_7-V_mV_m^T.
$$

Here, $H_m\in\mathbb R^{m\times m}$ is nonsingular, $G_m\in\mathbb R^{7\times m}$ is a right inverse, and $N_m$ projects onto the task nullspace. Consequently,

$$
\bar J_m(q_0)G_m=I_m,
\qquad \bar J_m(q_0)N_m=0.
$$

Compute $H_m^{-T}$ by a triangular solve, not an explicit inverse. Do not regularize a singular reference, truncate its rank, or substitute another basis. Such a reference violates the declared model assumptions and must produce an explicit error.

One neural network produces three bounded matrices,

$$
\operatorname{reshape}(f_\theta(x))
=\bigl(U_6(x),U_t(x),U_n(x)\bigr),
\qquad U_6\in\mathbb R^{7\times6},
\qquad U_t,U_n\in\mathbb R^{7\times3}.
$$

The network has widths $7\to64\to64\to84$, with a hyperbolic tangent after each affine layer, including the output layer. The output partition is $42+21+21$. These are fixed architectural choices, not configuration options. The output tangent bounds each matrix entry in magnitude by one. The general-motion matrix is

$$
F_6=G_6+N_6U_6(x),
\qquad F_6=B_6S_6,
$$

where the second expression is reduced QR with nonsingular triangular factor $S_6$. The geometry fixes a task-relevant right-inverse component; learning selects the joint nullspace component and therefore the coordinated motion subspace. In computation, use $N_mU=U-V_m(V_m^TU)$ without forming a dense nullspace projector.

This construction is expressive over regular joint subspaces before bounding the neural output. For any full column rank matrix $B$ for which $\bar J_mB$ is invertible, $B(\bar J_mB)^{-1}$ is a right inverse and differs from $G_m$ only by a nullspace term. The bounded network deliberately restricts the size of that term. It does not reconstruct every possible seven-dimensional joint displacement exactly.

## 4. Plane-normal multiplication

A nominal plane is

$$
\mathcal P(c,n)=\{p:n^T(p-c)=0\},
\qquad \|n\|=1.
$$

For point generation, let $E\in\mathbb R^{3\times2}$ satisfy $E^TE=I_2$ and $E^Tn=0$. Then $p_d=c+Eu$ for an in-plane coordinate $u\in\mathbb R^2$. The normal determines tangent directions; $c$ locates the plane and $E$ specifies raster orientation. The local theorem applies to planar neighborhoods through $p(q_0)$, so require $n^T(p(q_0)-c)=0$ when invoking that theorem.

### Why a homogeneous normal contraction is insufficient for a universal rank-three claim

The direct tensor proposal would use $F(n)=\sum_{r=1}^3n_rA_r$, with $A_r\in\mathbb R^{7\times3}$. At a fixed reference configuration, define $M(n)=J_p(q_0)F(n)\in\mathbb R^{3\times3}$. Then $M(-n)=-M(n)$, so $\det M(-n)=-\det M(n)$. Continuity along a path on the unit sphere from $n$ to $-n$ implies that this determinant vanishes for at least one normal. Thus homogeneous linear contraction cannot preserve three-dimensional positional rank for every normal at a fixed reference.

This obstruction concerns rank three, not the two intrinsic degrees of freedom of one plane. It does not rule out a plane-specific rank-two construction, a restricted set of normals, or a different reference configuration for each normal. Moreover, instantaneous tangent rank alone does not prove finite planar coverage: a curved reachable surface can be tangent to a plane while intersecting it only at the reference point.

### Selected normal-conditioned construction

Use the normal and tangent projectors

$$
P_n=nn^T,
\qquad P_t=I_3-nn^T,
\qquad U_3(x,n)=U_t(x)P_t+U_n(x)P_n.
$$

Construct and orthogonalize the planar basis as

$$
F_3=G_3+N_3U_3(x,n),
\qquad F_3=B_3S_3.
$$

This is multiplication of the learned motion matrices by the plane normal through $nn^T$. It preserves three coefficient directions and gives the same raw matrix for $n$ and $-n$. Using two different learned matrices allows the joint subspace to depend on the normal. In comparison, multiplying one fixed $7\times3$ basis on the right by an invertible $3\times3$ matrix merely changes its coefficient coordinates and leaves its joint subspace unchanged.

The normal-conditioned nullspace term can change the joint subspace, while the analytical component preserves local three-dimensional position mobility. Keeping the third positional direction makes nearby planar targeting a consequence of an ordinary local inverse theorem. It does not add an orientation constraint to the planar task. Any advantage from normal conditioning must be measured against the same learned model evaluated with one fixed normal.

## 5. Exact local result and regional proof obligation

**Proposition.** Suppose the task map is continuously differentiable near an interior reference $q_0$, the reference task Jacobian has full row rank, and the construction in sections 3–4 uses exact arithmetic. For either task, $F_m$ has full column rank and

$$
\bar J_m(q_0)F_m=I_m,
\qquad \bar J_m(q_0)B_m=S_m^{-1},
\qquad \operatorname{rank}(\bar J_m(q_0)B_m)=m.
$$

**Proof.** Multiplication by $\bar J_m$ annihilates the nullspace term and maps $G_m$ to the identity. Therefore $F_ma=0$ implies $a=0$, so its reduced QR factor is nonsingular. Substitution of $F_m=B_mS_m$ gives the second identity and full rank. This argument holds for every unit normal in the position construction. The inverse function theorem applied to $\alpha\mapsto p(q_0+B_3\alpha)$, or to a local pose coordinate chart for $\alpha\mapsto(p,R)(q_0+B_6\alpha)$, establishes a task neighborhood with a local inverse. Since $q_0$ is interior, a sufficiently small coefficient neighborhood also satisfies the joint bounds.

The learned weights do not create this local rank guarantee; the construction supplies it. Learning is intended to improve finite motion reconstruction and the useful extent of the reduced chart. The all-normal statement concerns local mobility at the reference, not all plane origins or all reachable targets.

For a fixed episode, let $\gamma_0=\sigma_{\min}(S_m^{-1})=1/\|F_m\|_2$. If $\bar J_m$ is Lipschitz with a proved constant $L$ on the relevant joint region, then

$$
\sigma_{\min}\bigl(\bar J_m(q)B_m\bigr)
\ge\gamma_0-L\|q-q_0\|_2.
$$

Hence a radius $r$ with $Lr<\gamma_0$ preserves differential rank throughout the declared region. This uses $\|B_m\|_2=1$ and the singular-value perturbation bound. For a fixed regular position reference, the bounded network also gives the uniform-in-normal estimate

$$
\gamma_0(n)\ge\frac{1}{\|G_3\|_2+\|U_t\|_2+\|U_n\|_2}>0.
$$

This estimate is conservative. A sampled minimum singular value is not a certified Lipschitz bound or a coverage proof. Finite coverage additionally requires showing that the desired target set lies in the image of the feasible coefficient domain. Neither the local proposition nor regional rank alone establishes this inclusion.

## 6. Coefficient-space inverse kinematics

The feasible coefficient set is the coupled polytope

$$
\mathcal A_k(q_0,B_k)=\{\alpha:q_{\min}-q_0\le B_k\alpha\le q_{\max}-q_0\}.
$$

Do not replace this constraint with independent coefficient bounds. For a pose target, define normalized residuals using position tolerance $\varepsilon_p>0$ and angular tolerance $\varepsilon_R>0$,

$$
e_6(\alpha)=
\begin{bmatrix}
(p(q(\alpha))-p_d)/\varepsilon_p\\
\operatorname{Log}(R_d^TR(q(\alpha)))^\vee/\varepsilon_R
\end{bmatrix}.
$$

Here, $\operatorname{Log}(\cdot)^\vee$ is the principal rotation vector. Its derivative is used away from the rotation-logarithm cut at angle $\pi$. The position-only residual is $e_3(\alpha)=(p(q(\alpha))-p_d)/\varepsilon_p$. Both tasks solve

$$
\alpha^*\in\underset{\alpha\in\mathcal A_k}{\operatorname{argmin}}
\;\frac12\|e_m(\alpha)\|_2^2.
$$

No coefficient regularizer, penalty-based joint limits, or full joint rescue solve is included. The requested affine reconstruction already determines the search dimension. Initial coefficients are zero at the beginning of an episode and the previous solution at subsequent waypoints.

Use the actual residual Jacobian, not the geometric Jacobian as a substitute for it away from zero orientation error. With $Q=R_d^TR(q)$ and $J_{\log}(Q)$ denoting the right derivative of the rotation logarithm,

$$
D_\alpha e_6=
\begin{bmatrix}
J_p(q)/\varepsilon_p\\
J_{\log}(Q)R(q)^TJ_\omega(q)/\varepsilon_R
\end{bmatrix}B_6,
\qquad D_\alpha e_3=J_p(q)B_3/\varepsilon_p.
$$

The gradient is $(D_\alpha e_m)^Te_m$. The orientation expression uses a base-expressed angular Jacobian converted to body coordinates, consistent with [Pinocchio's right derivative convention](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/namespacepinocchio.html). Verify this expression against finite differences before using it in optimization.

Use [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html) with an objective callback returning value and gradient, and a [linear constraint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html) for $\mathcal A_k$. The same solver implements the explicitly requested full joint comparison by taking $B_7=I_7$. That comparison is never invoked as a replacement after a reduced solve fails.

Task success requires the declared joint bounds and raw position tolerance, plus the raw angular tolerance for pose. Preserve optimizer convergence status separately: a solver status is not a reachability certificate. Expected nonconvergence is an explicit failed result in the evaluation record. A trajectory stops at its first task failure and marks later waypoints unattempted; independent target episodes continue. Report attempted and planned waypoint counts separately. Invalid model assumptions, malformed inputs, and nonfinite calculations raise errors. No exception handling conceals them.

For planar targets, the three-dimensional position residual includes normal error. Report $|n^T(p(q)-c)|$ separately. A second plane equality is unnecessary when the desired point already lies on the plane. Tool orientation is absent from the planar objective and success criterion.

## 7. Motion data and neural training

Generate smooth, joint-limit-feasible joint trajectories and retain their reference configurations and joint displacements $\Delta q=q_{\mathrm{demo}}-q_0$. A numerical joint path is kinematic training data, not a demonstrated dynamically feasible hardware motion. Split complete episodes into disjoint training, validation, and test sets before fitting.

General trajectories supply pose demonstrations directly through forward kinematics. To obtain planar demonstrations, project general-motion tip displacements onto planes through their reference tip positions and solve the resulting targets with the explicit full joint position solver. Record its failures and construction cost. Only successful reference trajectories provide joint labels, and that inclusion criterion and its excluded counts are reported. A reference-solver failure is not labeled mathematically unreachable. Hold out plane orientations and reference regions at evaluation.

For an orthonormal basis, the least-squares coefficient label and reconstructed displacement are

$$
\hat\alpha=B_k^T\Delta q,
\qquad \widehat{\Delta q}=B_kB_k^T\Delta q.
$$

Let $\mathcal D_6$ and $\mathcal D_3$ denote the general and planar training records. Fit the same network with the sum of their mean joint reconstruction losses,

$$
\mathcal L(\theta)=
\frac1{|\mathcal D_6|}\sum_{i\in\mathcal D_6}
\frac{\|\Delta q_i-B_{6,i}B_{6,i}^T\Delta q_i\|_2^2}{7}
+\frac1{|\mathcal D_3|}\sum_{i\in\mathcal D_3}
\frac{\|\Delta q_i-B_{3,i}B_{3,i}^T\Delta q_i\|_2^2}{7}.
$$

Joint angles use radians. No learned encoder, coefficient network, rank penalty, differentiable optimization layer, or bilevel training is needed. Differentiate through the neural matrices and reduced QR; full column rank is supplied by the construction. [PyTorch reduced QR](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html) supports the required batched matrix operation and differentiation under its rank assumptions.

This reconstruction objective trains a motion representation; it does not directly optimize IK success. Evaluate finite target error with the actual constrained solver. Record reference Jacobians once during data generation and batch the neural evaluations, projector actions, reconstruction loss, and QR calculations. Serial robot-library calls and ordered waypoint solves remain explicit loops because their dependencies differ from batch learning.

## 8. Exact code structure and structural budget

The implementation adds exactly five Python source files. The layout is flat to keep the mathematical workflow visible. Existing documentation remains in place.

```text
MetaArm/
  robot.py                       # Fixed robot model and tool kinematics.
  basis.py                       # Neural motion matrices and rank-preserving bases.
  ik.py                          # Coefficient-space constrained inverse kinematics.
  train.py                       # Motion data construction and batched neural training.
  evaluate.py                    # Verification and matched pose/plane comparisons.
  requirements.txt               # Exact versions of four direct dependencies.
  assets/fr3.urdf                # Verified kinematic asset with source attribution.
  docs/mathematical_formulation_and_implementation_plan.md
```

The asset includes the fixed tool definition or an explicitly recorded fixed transform. Preserve upstream attribution and license notices with the asset. No custom URDF generator is planned. Pin NumPy, SciPy, PyTorch, and Pinocchio after checking their joint availability in one project environment. Package-manager name resolution and exact versions are environment preflight tasks, not additional runtime options. No package framework, configuration file, command-line parser, plotting module, or hardware interface is included.

| File | Exact custom definitions | Responsibility and approximate size |
| --- | --- | --- |
| `robot.py` | `kinematics(q)` | Load the fixed model once at module scope; return copied tool position, rotation, and base-aligned tool Jacobian for one configuration. About 50–90 lines. |
| `basis.py` | `MotionBasis.__init__(self)`; `MotionBasis.forward(self, q_ref, J_ref, task, normal)` | Construct the fixed neural layers; normalize references, partition outputs, apply the analytical geometry and normal projectors, and return the orthonormal basis. Support batch leading dimensions. About 90–160 lines. |
| `ik.py` | `solve_ik(q_ref, basis, target, alpha_initial)`; nested `objective(alpha)` | Reconstruct joints, evaluate residual and exact gradient, enforce coupled bounds, solve once, and return the explicit result. About 90–160 lines. |
| `train.py` | No custom function or class; ordered script body | Construct and split the motion records, obtain planar reference demonstrations, train the network, and save its state and data provenance. About 150–230 lines. |
| `evaluate.py` | No custom function or class; ordered script body | Run verification blocks, load the prescribed checkpoint, construct target episodes, compare methods, and save all outcomes and aggregate metrics. About 150–230 lines. |

**Definition budget:** one custom class; two methods; two top-level functions; one nested objective callback; five callable definitions in total. The callback is required by the optimizer interface and contains the actual objective mathematics. There are zero custom data classes, wrappers, utility helpers, task classes, solver classes, or additional entry-point functions. Standard neural layers and numerical-library internals do not create custom project abstractions. Keep ordinary substantive functions within approximately 20–100 lines; a short constructor is not padded to meet a line target.

**Persistent object budget:** one Pinocchio `model`; one mutable Pinocchio `data` workspace; one `MotionBasis` instance called `network`; and one Adam optimizer during training. The URDF frame identifier and joint-limit arrays are fixed model metadata. Do not add registries, factories, managers, caches, or alternate backends. Returned kinematic arrays must be copied because subsequent calls reuse `data`.

**Plain data budget:** motion records are NumPy arrays grouped by general and planar episode, with reference joint states, demonstration states, reference Jacobians, episode identifiers, and planar geometry where applicable. Saved records include robot checksum, frame definitions, and split membership. A target is one dictionary with exactly `position` and `rotation`; `rotation=None` explicitly denotes a position-only problem. A result is one dictionary with exactly `q`, `alpha`, `errors`, `feasible`, `optimizer`, and `seconds`. `errors` holds raw position and, when defined, angular error. The unmodified library optimizer result retains status, iteration count, and evaluation count. Undefined planar orientation is recorded as absent, not as a zero measurement.

**Independent variable budget in code:** the reference state, target data, plane geometry, neural weights, and coefficient vector are the five groups listed in section 1. `task` is the required categorical selector `pose` or `plane`; it determines $m$ and $k$, which are not separately configurable. The full comparison passes an identity basis directly to `solve_ik`. `alpha_initial` is explicit problem initialization. Intermediate Jacobians, QR factors, projectors, residuals, and gradients implement the equations and introduce no additional optimization variables.

**Experiment parameter budget:** only two adjustable numerical settings, $\varepsilon_p=10^{-3}\,\mathrm m$ and $\varepsilon_R=\pi/180\,\mathrm{rad}$, defined as literals in `ik.py`. These are proposed evaluation tolerances, not promised accuracy. Target locations, plane normals, raster dimensions, and reference configurations are stored task data, not additional CLI/configuration settings.

**Fixed implementation choices:** CPU computation, float64 arithmetic, random seed 0, the network architecture stated in section 3, Adam with learning rate $10^{-3}$, batch size 256, and 500 complete passes through the training records. Use SLSQP with the analytic gradient, at most 200 iterations, and `ftol=1e-10`. These are source literals in their owning workflow files. Use the last trained weights and report validation metrics; there is no automatic checkpoint substitution. Runtime settings, file locations, task modes, and coefficient dimension are not exposed through configuration or CLI options. The count and geometry of generated episodes are a fixed recorded dataset specification, established before training and retained with the data.

The generated local artifacts are exactly `data/motions.npz`, `checkpoints/basis.pt`, and `results/evaluation.npz`; these directories are already ignored by Git. The data archive holds the motion arrays and provenance, the checkpoint holds the network state, and the evaluation archive holds per-target outcomes and summary arrays. No serialization adapter or storage helper is planned. Release any archive reader explicitly; do not introduce a `with` block.

## 9. Implementation workflow and verification

Follow this continuous workflow without exceeding the structural budget.

1. **Establish robot geometry.** Verify the official asset, seven arm joints, tool point, joint bounds, and Jacobian frame convention. Implement `kinematics`. In `evaluate.py`, compare its translation and angular Jacobian blocks with central differences and verify the orientation-residual derivative. Use Pinocchio's `LOCAL_WORLD_ALIGNED` convention at the declared tool origin, not a spatial velocity expressed at the world origin.
2. **Establish matched optimization.** Implement the coefficient objective and joint-limit polytope once. Check a feasible target at the reference, nonzero reachable targets, coupled bound activation, and explicit failure records. Compare the analytic gradient with finite differences. The seven-joint reference uses the same objective and solver through an identity basis.
3. **Construct the learned representation.** Implement the fixed network and the two analytical basis constructions. Check orthonormality, $\bar J_mF_m=I_m$, and $\bar J_mB_m=S_m^{-1}$ at regular references. Check that normals $n$ and $-n$ give the same planar subspace projector. Check explicit failure at rank-deficient references. These checks verify implementation identities; they do not certify a finite workspace.
4. **Train on recorded motions.** Generate and save episode-separated motion data, including full solver failures and the exact inclusion mask. Train with the reconstruction loss and batched linear algebra. Save the final weights and report validation reconstruction error. Training requires no differentiation through Pinocchio or SLSQP.
5. **Evaluate the research questions.** Compare learned reduced IK, full seven-joint IK, and a frozen reference-Jacobian singular-vector basis. For planar episodes, also compare actual-normal conditioning with the same network evaluated at the fixed normal $(0,0,1)^T$. This is a deliberate ablation, not a substitute on failure. Evaluate independent target points and complete trajectories across held-out reference regions, plane rotations, translations, and task extents.

Report target and whole-episode success, raw position/angular errors, normal plane deviation, reduced-Jacobian singular values, joint-limit margins, and joint continuity. Record basis construction, initialization, online solving, and offline training times separately; report median and 95th-percentile solve times and retain every failed target. Plotting and additional baselines can be proposed later without entering this implementation budget. Joint continuity is a kinematic metric, not a dynamics or velocity-limit guarantee.

Implement compact, continuous logic with the minimum realization. Include a concise professional one-line comment for every key code block. Keep source files within 50–250 lines without padding. Do not add unnecessary helpers, CLI/configuration options, fallback values, fallback behavior, or `try`/`with`/`except` constructs. Do not clip joint solutions, change task objectives, switch solvers, or call full joint IK to replace a failed reduced result. Record expected research failures explicitly and let invalid assumptions and unexpected errors emerge directly.

## 10. Claims supported by this formulation

The analytical construction proves full local pose rank for six coefficients and full local position rank for three coefficients at a regular interior reference. The position result holds for every unit plane normal at that reference, and therefore supports nearby planar targeting through the reference tip point. A regional rank certificate additionally requires a valid Jacobian variation bound. Arbitrary finite workspace coverage, improved IK runtime, useful neural generalization, and benefits from plane-normal conditioning remain research questions to be evaluated.

The learned representation and low-dimensional coefficient search are conceptually inspired by [Neural-Fly](https://authors.library.caltech.edu/records/q3grb-3vz72). Its flight-control guarantees are not used in the kinematic proof here. This document authorizes no implementation by itself; it specifies the mathematical proposal and the code budget for the next implementation decision.
