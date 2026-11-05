# Task-Conditioned Neural Motion Bases for Reduced-Coordinate Inverse Kinematics

**Abstract—** Numerical inverse kinematics for redundant manipulators searches joint configurations that realize a desired end-effector task while satisfying joint limits. Repeated searches motivate learning a compact representation, but displacement reconstruction alone does not establish that the representation retains the task directions required by inverse kinematics. This paper studies neural joint-motion bases constructed from a task-Jacobian right inverse and a learned nullspace correction. The construction preserves full task rank at every regular reference, while coefficient optimization retains the original coupled joint limits. We derive conditions for preserving rank over a finite chart, an exact null-direction criterion for the codimension-one case, and a sufficient condition for continuous position-volume and planar coverage. These conditions distinguish reference-local guarantees from sampled workspace evidence. MuJoCo-based evaluations on a seven-joint Franka Research 3 compare broad motion distributions, region-specific position and pose tasks, and planar motion. Broad-distribution reconstruction improvements do not translate into reliable fixed-chart inverse kinematics. Region-specific position charts succeed throughout the sampled volumes over a wider range than pose charts. Target-conditioned neural initialization reduces feasible pose-query latency, and a cooperative chart atlas recovers the remaining single-chart cases reached by the full-joint baseline in the largest tested lattice. A matched neural-seed control shows that initialization accounts for most of the latency gain. The results identify the separate roles of task dimension, chart transversality, training distribution, and initialization in reduced-coordinate inverse kinematics.

**Index Terms—** Inverse kinematics, redundant manipulators, representation learning, dimensionality reduction, robot learning, numerical optimization.

## I. Introduction

Inverse kinematics (IK) converts a desired task-space state into a robot configuration. Manipulation systems repeatedly perform this conversion when placing an end effector, following a surface, or executing a sequence of poses. For a redundant arm, multiple joint configurations can realize the same task state. This freedom helps accommodate joint limits and secondary objectives, but introduces choices that a numerical solver must resolve. The computational question is whether repeated tasks admit a learned search representation that preserves the required task motion while reducing online work.

Jacobian methods express a local task correction as a joint update. Damped least-squares formulations improve numerical behavior when task directions are poorly conditioned [1], [2]. Their effectiveness depends on damping, initialization, and the treatment of constraints. TRAC-IK combines a modified Jacobian procedure and sequential quadratic programming to improve reliability in the presence of joint limits [3]. Analytical methods exploit robot geometry, and recent geometric decomposition extends the range of automatically constructed analytical solvers [4]. Analytical solution structure is tied to the supported kinematic decomposition, and redundant solutions still require a choice among admissible configurations. These approaches provide meaningful alternatives to learned IK; this paper does not assume that every analytical method requires manual derivation or that every numerical method searches inefficiently.

Learning can reduce online computation in several distinct ways. IKFlow learns a conditional distribution of joint solutions and represents the multiplicity of redundant IK [5]. Vu et al. learn redundancy parameters for an analytical inverse, combining learned decisions with geometric solution structure [6]. Probabilistic movement primitives provide compact representations for conditioning and combining trajectories [7]. Synergy-based robotic control also performs IK directly in reduced coordinates, with a task derivative obtained by multiplying the robot Jacobian by a synergy matrix [8]. Generative joint candidates require physical constraint verification and may undergo numerical refinement [5]. These methods establish that learned solution structure, compact motion representations, and reduced-coordinate IK are useful research directions. Their objectives do not, by themselves, imply that a learned affine joint chart preserves all task directions over a prescribed finite region.

The separation between an offline representation and online coefficients also appears in Neural-Fly, which learns a shared aerodynamic representation and adapts its coefficients during flight [9]. That work motivates the division of computation considered here. Its domain-adversarial training and adaptive-control guarantees do not transfer to kinematic optimization. Likewise, optimization-based meta-learning explicitly trains a model for its behavior after task adaptation [10]. The present implementation learns a shared motion representation using reconstruction and supervised initialization; online adaptation solves for task coefficients rather than updating neural weights. This distinction matters when interpreting both the algorithm and its guarantees.

The question addressed here is the relationship between motion compression and task solvability. A basis can reconstruct a demonstration accurately while its task image becomes singular away from the reference. Conversely, a geometric basis can preserve the required local task directions without being the best approximation to the recorded joint motion. GPU parallelization adds another distinction: throughput improvements from batching or multiple seeds do not necessarily reduce isolated-query latency. Parallel robot optimization already exploits these mechanisms [11]. A reduced-coordinate method must therefore be compared with full-joint search under matched constraints, initialization, and timing boundaries.

This paper makes three contributions. First, it formulates a task-conditioned neural chart in which an analytical right inverse preserves reference task rank and a learned nullspace graph adapts the joint-motion representation. Second, it derives finite-domain rank conditions, an exact codimension-one characterization of additional chart singularities, and a quantitative sufficient condition for position-volume and planar coverage. Third, it evaluates reconstruction, task success, conditioning, and GPU query time across broad, regional, and planar tasks, including geometric and same-neural-seed full-joint controls. The contribution is the combined construction, analysis, and controlled characterization; neither coefficient-space IK nor its chain-rule Jacobian is claimed as a new concept.

## II. Problem Formulation

### A. Robot, task, and singularities

The symbol \(I_d\) denotes the identity matrix of dimension \(d\); the subscript is omitted when the context fixes it. Superscript \(T\) denotes transpose. The norm \(\|\cdot\|_2\) is Euclidean for vectors and spectral for matrices, \(\sigma_{\min}\) denotes the smallest singular value, and \(\operatorname{tr}\) denotes matrix trace. All singular-value comparisons use fixed task-coordinate units.

Consider a serial arm with joint configuration \(q\in\mathbb R^n\), where \(n\) is the number of actuated joints. Its admissible joint set is

\[
\mathcal Q=\{q:q_{\min}\le q\le q_{\max}\},
\tag{1}
\]

where inequalities are componentwise and \(q_{\min},q_{\max}\in\mathbb R^n\) contain the physical joint limits. Forward kinematics provides tool position \(p(q)\in\mathbb R^3\) and orientation \(R(q)\in SO(3)\), where \(SO(3)\) is the set of proper three-dimensional rotation matrices. The geometric Jacobian is

\[
J_g(q)=\begin{bmatrix}J_p(q)\\J_\omega(q)\end{bmatrix},
\qquad
\begin{bmatrix}\dot p\\\omega\end{bmatrix}=J_g(q)\dot q.
\tag{2}
\]

Here \(J_p,J_\omega\in\mathbb R^{3\times n}\), and \(\omega\) is base-frame angular velocity. For a revolute joint with base-frame unit axis \(a_i\) and anchor \(r_i\), its column is \([a_i\times(p-r_i);a_i]\). The task Jacobian \(J_\tau\in\mathbb R^{m\times n}\) is \(J_p\) for position or \(J_g\) for full pose; \(\tau\) denotes task type and \(m\) its local constraint dimension. A task configuration is regular when \(\operatorname{rank}J_\tau=m\). Otherwise, the robot lacks at least one instantaneous task direction at that configuration.

For desired position \(p_d\), define \(e_p(q)=p(q)-p_d\). For desired orientation \(R_d\), define

\[
e_R(q)=\operatorname{Log}^{\vee}\!\left(R_d^TR(q)\right),
\qquad
 e_y(q)=\begin{cases}
 e_p(q)/\varepsilon_p,&\text{position},\\
 [e_p(q)/\varepsilon_p;\ e_R(q)/\varepsilon_R],&\text{pose},
 \end{cases}
\tag{3}
\]

where \(\operatorname{Log}^{\vee}\) is the principal rotation-vector logarithm, \(y\) denotes the desired task, and \(\varepsilon_p,\varepsilon_R>0\) are prescribed translation and rotation tolerances. Pose residuals are restricted away from the logarithm cut. Their derivative has the same rank as the geometric Jacobian because the orientation-coordinate transformation and tolerance scaling are nonsingular there. For pose, the implemented derivative is

\[
D_q e_y=\begin{bmatrix}
J_p/\varepsilon_p\\
\mathcal J_r^{-1}(e_R)R(q)^TJ_\omega/\varepsilon_R
\end{bmatrix},
\tag{3a}
\]

where \(\mathcal J_r^{-1}\) is the inverse right Jacobian of the rotation logarithm. For \(\phi\in\mathbb R^3\), \(\vartheta=\|\phi\|\), and the cross-product matrix \( [\phi]_\times v=\phi\times v\), it is

\[
\mathcal J_r^{-1}(\phi)=I+\tfrac12[\phi]_\times+
\frac{1-(\vartheta/2)\cot(\vartheta/2)}{\vartheta^2}[\phi]_\times^2.
\tag{3b}
\]

The scalar coefficient has limit \(1/12\) at zero; the implementation uses the analytic small-angle limit. Successful IK requires joint feasibility and each applicable physical error norm to meet its own tolerance; a small optimizer objective alone is not the success criterion.

### B. Planar targets and task dimension

Let a plane have origin \(o\), unit normal \(\nu\), and tangent frame \(E\in\mathbb R^{3\times2}\), with \(E^TE=I_2\) and \(E^T\nu=0\). A planar target is \(p_d=o+E\xi\), where \(\xi\in\mathbb R^2\) specifies its two intrinsic position coordinates. The implemented planar solver imposes the complete equality

\[
p(q)=o+E\xi,
\tag{4}
\]

using the three-component position residual. It therefore constrains tangent placement and normal closure, with free orientation. Its three coefficients do not make the plane intrinsically three-dimensional. Nor does this task constrain an in-plane heading. In this paper, **three-coefficient planar-position IK** always means (4).

### C. Reduced-coordinate search and computational objective

Let \(q_0\in\operatorname{int}\mathcal Q\) be a reference configuration and \(B\in\mathbb R^{n\times k}\) have orthonormal columns. The affine chart and feasible coefficient set are

\[
q(\alpha)=q_0+B\alpha,
\qquad
\mathcal A(q_0,B)=\{\alpha:q_{\min}-q_0\le B\alpha\le q_{\max}-q_0\},
\tag{5}
\]

where \(\alpha\in\mathbb R^k\) contains the online decision variables. The image of this set is a subset of the original joint space. All joint inequalities remain coupled through \(B\); independent coefficient bounds are not an equivalent replacement. The reduced IK problem is

\[
\min_{\alpha\in\mathcal A(q_0,B)}
\frac12\|e_y(q_0+B\alpha)\|_2^2.
\tag{6}
\]

Full-joint IK is the same formulation with \(B=I_n\). The reduced task derivative is \(D_qe_y(q)B\). Its rank cannot exceed \(k\), so retaining all \(m\) task directions requires \(k\ge m\). Regular-reference constructions assume \(m\le n\). The principal construction uses \(k=m\), the smallest number compatible with regular local task motion. For planar experiments, \(m=k=3\) refers to the ambient position constraint in (4).

The design objective is to learn a useful \(B\), obtain feasible solutions of (6), and reduce online time relative to matched full-joint IK. The relevant time is

\[
T_{\mathrm{query}}=T_{\mathrm{prepare}}+T_{\mathrm{search}}+T_{\mathrm{dispatch}},
\tag{7}
\]

where preparation includes reference kinematics, factorization, and neural inference. Reducing \(k\) changes small linear systems but does not remove forward kinematics, residual evaluation, joint inequalities, or difficult iterations. Variable reduction is therefore a design mechanism rather than a runtime guarantee.

## III. Learning Task-Conditioned Motion Charts

The offline computation collects full-joint demonstrations and learns shared neural weights. For a new task, online computation constructs a reference-conditioned basis, initializes its coefficients, solves the constrained task residual, and reconstructs the joint configuration. The basis remains fixed within each declared solve; changing the task context can select a different basis.

### A. Simulation and demonstration generation

The numerical system uses a pinned Franka Research 3 model in MuJoCo [12]. MuJoCo Warp supplies batched GPU kinematics [13]. Joint axes, tool transforms, joint limits, and reference Jacobians come from the same physical description. Independent double-precision CPU MuJoCo evaluations audit GPU poses, Jacobians, final task errors, and success labels. The collection is kinematic: it evaluates prescribed configurations and inverse-kinematic labels, rather than identifying dynamics or validating torque-level tracking.

The broad corpus samples reference configurations and smooth closed joint paths. A path can be written symbolically as

\[
q(t)=q_0+\sum_{h=1}^{H}c_h[1-\cos(2\pi h\,s(t))],
\tag{8}
\]

where \(H\) is the harmonic count, \(c_h\in\mathbb R^n\) are sampled joint amplitudes, and \(s(t)\) is normalized trajectory phase. Amplitude bounds keep the path inside joint limits. Tool poses obtained from (8) define general pose targets. Projecting the corresponding position displacements with \(I-\nu\nu^T\) produces planar targets on a plane through \(p(q_0)\). The broad planar teacher uses position-only full-joint IK with sequential warm starts. Because position constraints do not determine nullspace motion, a closed Cartesian path need not return the teacher to \(q_0\).

Regional datasets instead sample targets inside a declared Cartesian box. Closed paths interpolate between sampled interior vertices using quintic segments. Their full-joint teacher minimizes

\[
\min_{q\in\mathcal Q}\frac12\|q-q_a\|_2^2
\quad\text{subject to}\quad
p(q)=p_d,
\quad R(q)=R_d\ \text{when pose is constrained},
\tag{9}
\]

where \(q_a\) is a declared anchor. The initial position-volume study uses its center configuration; the paired expansion study uses each episode reference. Targets are labeled independently under that convention. This reduces uncontrolled path-dependent joint drift, but does not make local teacher convergence a proof of global feasibility or optimality.

References and related trajectories remain in the same training, validation, or test group. Every broad-corpus archive participates under its recorded membership; individual waypoints are not randomly repartitioned. Complete successful teacher paths supply the original reconstruction labels. Later pose-seed training additionally uses individually task-feasible waypoints from incomplete paths. Failed candidates, optimizer statuses, exclusion masks, and unattempted targets remain in the records. A failed minimum-distance optimization and an unreachable task are different events.

### B. A right inverse with a learned nullspace graph

At a regular reference, write \(J_0=J_\tau(q_0)\). Define its minimum-norm right inverse and orthogonal nullspace projector as

\[
G_0=J_0^T(J_0J_0^T)^{-1},
\qquad N_0=I_n-G_0J_0.
\tag{10}
\]

The matrix \(G_0\) converts an infinitesimal task displacement into a particular joint displacement. The matrix \(N_0\) removes the task-visible component of any joint vector. Given context \(c\), a neural network predicts \(U_\theta(c)\in\mathbb R^{n\times m}\), where \(\theta\) denotes its shared weights. The unnormalized chart is

\[
F_\theta(c)=G_0+N_0U_\theta(c).
\tag{11}
\]

The learned term changes the joint realization of each task direction without changing that direction's first-order task image at the reference. A correction amplitude, when used, is absorbed into \(U_\theta\) in (11). Reduced QR gives \(F_\theta=BS\), where \(S\in\mathbb R^{m\times m}\) is invertible, or Cholesky gives \(F_\theta^TF_\theta=LL^T\) and \(B=F_\theta L^{-T}\). Both produce an orthonormal basis for the same subspace. QR and Cholesky linear solves evaluate the right inverse and projectors without explicitly forming matrix inverses.

For broad planar tasks, the network uses two learned matrices,

\[
U_\theta(q_0,\nu)=U_t(q_0)(I-\nu\nu^T)+U_n(q_0)\nu\nu^T.
\tag{12}
\]

The subscripts distinguish tangent and normal task-coordinate components; the matrices are joint-motion corrections, not spatial directions themselves. Dependence on \(\nu\nu^T\) makes the chart identical for the two normals describing the same plane. Regional position and pose models condition on normalized reference joints and are fitted separately for their task region.

### C. Representation-Based Meta-Learning and Coefficient Adaptation

Let \(\mathcal D_\iota=\{\Delta q_{\iota t}\}_{t=1}^{T_\iota}\) contain an episode's joint displacements, with \(\Delta q_{\iota t}=q_{\iota t}-q_{0,\iota}\). The episode index also identifies its task context, and \(T_\iota\) is its number of retained waypoints. The representation objective has the nested form

\[
\min_\theta\sum_\iota w_\iota
\min_{\{\alpha_{\iota t}\}}
\frac{1}{nT_\iota}\sum_t
\|\Delta q_{\iota t}-B_\theta(c_\iota)\alpha_{\iota t}\|_2^2,
\tag{13}
\]

where \(w_\iota\ge0\) are prescribed episode weights. Task categories are balanced in the broad joint fit, and waypoint multiplicities are retained. With orthonormal \(B\), the inner unconstrained reconstruction problem has \(\alpha_{\iota t}^*=B^T\Delta q_{\iota t}\). It is not the constrained nonlinear IK problem (6).

Define the uncentered second moment \(C_\iota=T_\iota^{-1}\sum_t\Delta q_{\iota t}\Delta q_{\iota t}^T\) and projector \(P_\theta=B_\theta B_\theta^T\). Eliminating the reconstruction coefficients yields

\[
\mathcal L_{\mathrm{rec}}(\theta)=
\sum_\iota\frac{w_\iota}{n}
\operatorname{tr}[(I-P_\theta)C_\iota(I-P_\theta)^T].
\tag{14}
\]

The sufficient statistic removes repeated evaluations of an episode's fixed basis without subsampling its waypoints. The moments are not centered because the chart origin remains \(q_0\).

Equation (13) permits a representation-based meta-learning interpretation: shared weights encode a family of task-conditioned subspaces, while episode-specific coefficients adapt within that family. The implemented representation objective is reconstruction, followed by separate online IK. It neither differentiates through the constrained IK solver nor performs gradient-based adaptation of \(\theta\) on a new task. Consequently, it is not an implementation of MAML or Neural-Fly's learning algorithm. This paper uses **shared representation learning** for the implemented training procedure and **coefficient adaptation** for the online search.

### D. Target-conditioned pose charts and initialization

For a task with one redundant joint, \(n=m+1\), the nullspace has a unit basis vector \(z_0\). Equation (11) reduces to

\[
F=G_0+z_0u_\theta(c)^T,
\tag{15}
\]

where \(u_\theta\in\mathbb R^m\) specifies the nullspace graph. The compact implementation fixes a standard joint-coordinate vector \(e_j\) and sets \(z_0=N_0e_j/\|N_0e_j\|\). It is evaluated only where that declared projection is nonzero; a vanishing gauge raises an error even if \(J_0\) is regular. No alternate gauge is substituted. This implementation-domain restriction is separate from Proposition 1's general matrix construction. Rather than predicting every entry of a correction matrix, the refined pose network predicts \(u_\theta\) and a tangent-seed correction. Its context includes normalized \(q_0\) and the desired local pose displacement

\[
d_y=\begin{bmatrix}p_d-p(q_0)\\
R(q_0)\operatorname{Log}^{\vee}(R(q_0)^TR_d)\end{bmatrix}.
\tag{16}
\]

The rotational part is expressed in base coordinates to match \(J_g\). Two neural heads produce bounded graph entries and a tangent seed, \(u_\theta=\bar u\tanh h_\theta(c)\) and \(\beta_\theta=d_y+D_\beta\tanh v_\theta(c)\). Here \(\bar u>0\) bounds graph amplitude, \(D_\beta\) is a prescribed positive diagonal displacement scaling, and \(h_\theta,v_\theta\in\mathbb R^m\) are network outputs. Reference inputs are normalized componentwise as \(\widetilde q_{0,i}=2(q_{0,i}-q_{\min,i})/(q_{\max,i}-q_{\min,i})-1\). The predicted tangent vector gives \(\widehat{\Delta q}=F\beta_\theta\). A strictly feasible ray fraction \(\eta\in(0,1]\) is computed from physical joint margins, producing \(q_s=q_0+\eta\widehat{\Delta q}\) and \(\alpha_s=B^T(q_s-q_0)\). This constructs an initialization; it does not clip an optimized solution or alter the chart constraints.

Supervision combines seed and representation accuracy,

\[
\mathcal L_{\mathrm{seed}}=
\mathbb E\!\left[
\frac{\|q_s-q_0-\Delta q\|^2}{n}
+\lambda_{\mathrm{proj}}\frac{\|(I-BB^T)\Delta q\|^2}{n}
+\lambda_u\|u_\theta\|^2/m
\right],
\tag{17}
\]

with prescribed nonnegative weights \(\lambda_{\mathrm{proj}},\lambda_u\). This is supervised amortized initialization, not a new bilevel IK objective. The target-conditioned basis is fixed during each solve. A different target may select a different chart, so the union of query charts is not one global fixed subspace.

### E. Joint-constrained GPU optimization and chart branches

At the current coefficients, let \(e=e_y(q)\) and \(A=D_qe_y(q)B\). The search step solves a damped quadratic subproblem,

\[
\begin{aligned}
\min_{d\in\mathbb R^k}\;&
\tfrac12d^T(A^TA+\lambda D)d+(A^Te)^Td,\\
\text{subject to}\;&
q_{\min}-q\le Bd\le q_{\max}-q,
\end{aligned}
\tag{18}
\]

where \(\lambda>0\) is the damping and \(D\) is positive diagonal scaling derived from the normal matrix. A feasible primal-dual interior-point calculation treats active joint inequalities. Scaling a feasible step toward zero enforces a joint-displacement trust radius while preserving linear feasibility. The ratio of actual to predicted residual reduction controls damping; only cost-decreasing candidates are accepted. Candidate evaluation omits derivatives unused by the acceptance test. Stalling and iteration exhaustion remain explicit outcomes, not requests for a hidden rescue solve.

A fixed chart may exclude a useful redundancy branch. The tested atlas launches a prescribed finite set of charts in parallel,

\[
q^{(b)}(\alpha)=a_b+B\alpha,
\qquad a_b=q_s+\eta_b t_bz_0,
\tag{19}
\]

where \(b\) indexes branches, \(t_b\) are declared signed null offsets, and \(\eta_b\) gives each shifted origin a strictly feasible ray fraction. Every branch starts together and retains its own coupled polytope. Once a branch meets all applicable tolerances, its peers stop as superseded rather than being marked successful. Selection minimizes the maximum normalized physical error. The atlas uses multiple discrete choices and coefficient vectors; its total effort is not that of one \(m\)-variable solve. Full-joint controls receive both reference and identical neural-seed initializations to separate representation from initialization effects.

## IV. Rank Preservation and Workspace Verification

### A. Full task rank at the reference

**Proposition 1 (reference rank).** If \(J_0\in\mathbb R^{m\times n}\) has full row rank, then (11) has full column rank for every finite neural correction. For its orthonormal basis \(B\), \(J_0B\) is nonsingular.

**Proof.** Equations (10) give \(J_0G_0=I_m\) and \(J_0N_0=0\), hence \(J_0F=I_m\). A nonzero vector in \(\ker F\) would therefore belong to \(\ker I_m\), which is impossible. With \(F=BS\), multiplication by \(J_0\) yields \(J_0B=S^{-1}\). The Cholesky construction equivalently yields \(J_0B=L^{-T}\). ∎

All singular values and norm bounds use declared, fixed numerical task-coordinate units. Nonsingular task rescaling preserves rank, but changes conditioning margins and derivative bounds. The same calculation gives

\[
\sigma_{\min}(J_0B)=\|F\|_2^{-1}.
\tag{20}
\]

A large nullspace correction can preserve exact rank while reducing the numerical task margin. Rank preservation and conditioning are different properties. Because the reference is joint-interior and the task map is smooth, the inverse function theorem supplies a feasible task neighborhood. It does not identify that neighborhood with a specified finite cube or plane patch.

### B. When a chart creates an additional singularity

At any regular configuration, define an orthonormal frame for the Jacobian row space,

\[
V(q)=J_\tau(q)^T[J_\tau(q)J_\tau(q)^T]^{-1/2},
\qquad
\delta(q)=\|(I-BB^T)V(q)\|_2.
\tag{21}
\]

The scalar \(\delta\) is the sine of the largest principal angle between the learned search subspace and the task-visible joint subspace. It measures whether the chart has turned away from a required task direction.

**Proposition 2 (transversality).** For frozen orthonormal \(B\in\mathbb R^{n\times m}\) and regular \(J_\tau(q)\),

\[
\sigma_{\min}(J_\tau(q)B)
\ge\sigma_{\min}(J_\tau(q))\sqrt{1-\delta(q)^2}.
\tag{22}
\]

Moreover, \(J_\tau(q)B\) is nonsingular exactly when \(\delta(q)<1\).

**Proof.** Write \(J_\tau=(J_\tau J_\tau^T)^{1/2}V^T\). Since \(V\) and \(B\) have orthonormal columns, the smallest singular value of \(V^TB\) equals \(\sqrt{1-\|(I-BB^T)V\|_2^2}\). Applying the product singular-value bound proves (22). The first factor is invertible, so singularity is determined exactly by \(V^TB\). ∎

A rigorously established uniform bound \(\sup_{q\in\mathcal D}\delta(q)<1\) on a regular domain \(\mathcal D\) proves that the chart introduces no additional singularities there. At a full-robot singularity, \(\operatorname{rank}(J_\tau B)\le\operatorname{rank}J_\tau<m\) for every chart. A certificate on all regular points of a declared domain therefore makes the full and reduced singular sets coincide within that domain. Positive values at isolated grid solutions are not a uniform certificate.

For \(n=m+1\), let unit \(z(q)\) span \(\ker J_\tau(q)\) and let unit \(w\) span \(\ker B^T\). Cauchy–Binet and the maximal-minor null vector yield the exact identity

\[
\det(J_\tau(q)B)^2
=\det(J_\tau(q)J_\tau(q)^T)[z(q)^Tw]^2.
\tag{23}
\]

The squared maximal minors sum to \(\det(J_\tau J_\tau^T)\); their normalized vector is \(z\), and projection onto the chart normal gives the reduced determinant. Thus a regular robot acquires a chart singularity precisely when its null direction becomes orthogonal to the chart normal. For (15), that normal at the reference is

\[
w=\frac{z_0-J_0^Tu_\theta}{\sqrt{1+\|J_0^Tu_\theta\|_2^2}},
\qquad z_0^Tw>0.
\tag{24}
\]

Equation (24) explains reference transversality and the smaller angular margin associated with large graph corrections. It gives no automatic lower bound for \(z(q)^Tw\) away from the reference.

### C. A sufficient certificate for finite position-volume and plane coverage

Nonsingularity and target coverage are separate questions. A chart can remain regular over its coefficient domain while mapping to only part of a desired task region. The next result states a sufficient condition for both properties for position tasks.

**Proposition 3 (finite position coverage).** Fix an orthonormal three-column \(B\), and define \(g(\alpha)=p(q_0+B\alpha)-p(q_0)\). Let \(A_0=Dg(0)\) be nonsingular and \(\gamma=\sigma_{\min}(A_0)>0\). Assume that the closed coefficient ball \(\|\alpha\|_2\le r\) is joint-feasible and that a finite constant \(\ell\ge0\) satisfies

\[
\|Dg(\alpha)-Dg(\beta)\|_2\le \ell\|\alpha-\beta\|_2,
\qquad \ell r<\gamma,
\tag{25}
\]

for all points in that ball. Then the chart is regular throughout the ball, and every displacement satisfying

\[
\|y\|_2<\rho,
\qquad \rho=\gamma r-\tfrac12\ell r^2,
\tag{26}
\]

has a unique inverse within that coefficient ball.

**Proof.** Integrating the derivative variation gives \(\|g(\alpha)-A_0\alpha\|\le \ell\|\alpha\|^2/2\). For fixed \(y\), define \(\mathcal T_y(\alpha)=A_0^{-1}[y-(g(\alpha)-A_0\alpha)]\). Equations (25)–(26) make \(\mathcal T_y\) map the closed ball into itself, and its Lipschitz factor is at most \(\ell r/\gamma<1\). The contraction theorem gives a unique fixed point, which satisfies \(g(\alpha)=y\). Singular-value perturbation also gives \(\sigma_{\min}Dg(\alpha)\ge\gamma-\ell r>0\). ∎

A sufficient joint-feasibility test is \(r\|B_{i:}\|_2<\min(q_{0,i}-q_{\min,i},q_{\max,i}-q_{0,i})\) for every joint \(i\). A Cartesian cube contained in the task ball (26), or a plane disk through \(p(q_0)\) contained in that ball, is therefore continuously covered. For planar targets, this includes normal closure; it is stronger than rank two of \(E^TJ_pB\) alone.

A continuous certificate requires a valid bound \(\ell\), or an equivalent interval enclosure of the derivative over every cell of the declared domain. A covering grid with certified singular-value errors and covering radius \(h\) can instead use a lower bound of the form \(\min_{\mathrm{grid}}\sigma_{\min}-\ell h\). Successful IK endpoints need not cover the intervening coefficient domain and cannot justify that subtraction without a covering argument. For pose, the bound must also include the rotation-coordinate derivative. Section V reports sampled coverage and sampled rank; no continuous certificate for its large boxes or planar patch has been computed.

### D. Full-robot singularities in the FR3 workspace

The pinned FR3 geometry permits an analytical full-robot singularity. Let \(a\) be the elbow offset and \(d_3,d_5\) the adjacent link offsets. For \(q_3=q_5=0\), the planar elbow factor is

\[
A(q_4)=(d_3d_5-a^2)\sin q_4+a(d_3+d_5)\cos q_4.
\tag{27}
\]

At

\[
q_4^*=-\operatorname{atan2}\!\left(a(d_3+d_5),d_3d_5-a^2\right),
\tag{28}
\]

the anchors of the parallel joint axes two, four, and six are collinear. Their spatial twists span at most two coordinates. The remaining four joint twists occupy a complementary subspace of dimension at most three. Consequently, \(\operatorname{rank}J_g\le5\) for this family. Tool-point translation is an invertible twist transformation and does not change the argument. Appendix A derives the anchor geometry.

A joint-limit-interior member of this family lies within the specified orientation range and within the regional pose cubes beginning at the second tested size. This proves existence of a singular admissible joint preimage in those volumes. It does not prove that every inverse branch of that pose is singular, or that a failed numerical IK request is physically unreachable.

## V. Numerical Evaluation

### A. Protocol, baselines, and interpretation

All robot experiments use the seven-joint FR3 and the same attachment-site definition. Position tasks use three coefficients and free orientation; pose tasks use six coefficients and both translation and rotation constraints. Tolerances are 1 mm and 1 degree, respectively. Every candidate is checked against the original joint limits. Experiments distinguish (i) per-joint reconstruction mean squared error (MSE), (ii) completed trajectories under sequential stopping, (iii) independent target success, (iv) sampled task-image singular values, and (v) synchronized GPU time. Failure statuses remain part of the results.

Three baselines separate the main effects. Full-joint IK searches seven variables with the same residual and joint inequalities. The geometric reduced baseline orthonormalizes \(G_0\) without a learned correction. For refined pose experiments, neural-seeded full-joint IK starts at exactly the same joint seed as the learned reduced method. These comparisons do not benchmark TRAC-IK, analytical IK, IKFlow, or cuRobo, and no ranking against those external systems is inferred.

### B. Broad motion and planar-position tasks

The broad fit includes all 20 recorded archives: 1,280 references, 1,280 general pose episodes, and 2,560 projected planar episodes, totaling 249,600 waypoint records. Reference-joint regions and plane-normal tilts have preassigned held-out ranges. The test groups change both reference and plane orientation in the planar evaluation; they do not isolate these generalization factors. Ten training configurations vary correction amplitude, width, input symmetry, regularization, and duration. Selection uses validation reconstruction.

The selected broad model reduces the combined pose-plus-plane test reconstruction MSE from 0.0834787 to 0.00614023 rad² per joint. The reduction is not uniform across tasks: plane MSE decreases from 0.0824783 to 0.00500446, while pose MSE increases from 0.00100044 to 0.00113577. The planar teacher exhibits substantial joint drift on closed Cartesian loops, and general harmonic directions vary independently of the reference-only neural input. These observations explain why larger capacity need not resolve the learning problem.

**TABLE I. Broad held-out trajectory completion.** Both methods retain their own first-failure stopping histories. “Planar” denotes (4), not an intrinsic three-dimensional task.

| Task | Learned coefficients | Learned completed paths | Full-joint completed paths | Learned successful/planned waypoints |
| --- | ---: | ---: | ---: | ---: |
| General full pose | 6 | 88/160 | 160/160 | 6,568/10,400 |
| Projected planar position | 3 | 19/320 | 320/320 | 3,107/20,800 |

Table I shows that reconstruction improvement does not establish useful finite-chart IK. At failed interior candidates, 67 pose cases and 270 planar cases have minimum reduced task singular value below \(10^{-4}\). This numerical evidence is consistent with chart folds or stationary residuals; it is not an exact global classification of singularities. Independent CPU refinement of sampled failures remains outside task tolerances. Learned/full batched solve totals are 5.895/1.172 s for pose and 10.431/0.892 s for planar tasks, but different attempted target counts preclude interpreting those totals as matched query-speed ratios.

The broad three-coefficient experiment is planar-position IK. An unrestricted broad-distribution three-dimensional position-only comparison was not run. The general position-versus-pose comparison is therefore incomplete; regional position-versus-pose comparisons are available in Section V-D.

### C. Specific position volume and a verified planar slice

An independently collected local position study uses a 16 cm cube centered at \((0.50,0,0.50)\) m. Four model candidates are selected by validation trajectory completion and reconstruction error. The selected three-column chart achieves 48/48 test trajectories and 6,561/6,561 independent lattice targets. The geometric chart and full-joint baseline achieve the same sampled completion. Learned reconstruction MSE is 18.0% below the geometric chart's value, supporting a reconstruction improvement rather than a learned task-success advantage.

One exported center chart remains identical over 729 lattice targets. A post hoc audit of its central horizontal slice gives a specific plane patch using that same fixed basis. It is not a separately trained plane-specialized model. Independent CPU FK and Jacobians provide Table II.

**TABLE II. Sampled coverage and task span for one fixed regional chart.** The volume is \([0.42,0.58]\times[-0.08,0.08]\times[0.42,0.58]\) m. The plane patch sets \(z=0.50\) m. \(E=[e_x,e_y]\) selects horizontal tangent directions; singular values use position Jacobians in m/rad.

| Domain | Successful targets | Maximum position error (mm) | Minimum joint margin (rad) | Minimum \(\sigma_{\min}(J_pB)\) | Minimum \(\sigma_{\min}(E^TJ_pB)\) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Volume lattice, fixed center chart | 729/729 | 0.9843 | 0.8145 | 0.23350 | 0.24477 |
| Central plane slice | 81/81 | 0.9798 | 0.8171 | 0.24270 | 0.24618 |

Positive reduced singular values verify the three ambient position directions at the recorded configurations. Their tangent projections also have rank two. The maximum plane-normal error is 0.756 mm, so the solutions satisfy the position tolerance rather than exact plane invariance. Stored configurations reconstruct as \(q_0+B\alpha\) to within \(1.94\times10^{-16}\) rad. Table II verifies sampled span and sampled coverage. Proposition 3 supplies a continuous proof method, but its derivative bounds have not been certified for this complete cube or patch.

### D. Expanding three-dimensional position and pose regions

The expansion study considers cube sides 16, 24, 32, 40, and 48 cm about the same center. Pose targets use a componentwise local rotation-vector range of ±10 degrees; they do not range over all of \(SO(3)\). Common references lie in a central 8 cm position cube with ±5 degree reference rotations. Each size/task pair generates its own labels and independently searches six model/training configurations, giving 60 fits. Each dataset has 96/24/48 training/validation/test reference groups and 33-waypoint paths. The lattice has 729 positions evaluated from nine charts, giving 6,561 independent requests per method and configuration. Assigned orientation tuples do not constitute a full six-dimensional Cartesian product.

**TABLE III. Sampled coverage versus region size.** Path completion is out of 48; lattice success is out of 6,561. Position and pose change the physical task as well as coefficient count.

| Cube side (cm) | Position learned/full lattice | Pose learned lattice | Pose full-joint lattice | Learned position paths | Learned pose paths |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 6,561 / 6,561 | 6,561 | 6,561 | 48 | 48 |
| 24 | 6,561 / 6,561 | 6,525 | 6,525 | 48 | 48 |
| 32 | 6,561 / 6,561 | 6,138 | 6,138 | 48 | 47 |
| 40 | 6,561 / 6,561 | 5,652 | 5,652 | 48 | 40 |
| 48 | 6,525 / 6,525 | 5,166 | 5,175 | 48 | 35 |

Position reconstruction MSE increases from \(9.96\times10^{-7}\) to \(6.87\times10^{-5}\) rad² per joint as side length increases. Pose MSE increases from \(8.33\times10^{-6}\) to \(1.21\times10^{-4}\). Most pose coverage deterioration is shared with full-joint IK. The nine additional full-joint successes at the largest size identify a reduced-chart or reduced-optimizer limitation on targets known feasible through that baseline. Shared failures, including unsuccessful independent local refinements, do not establish global unreachability.

At 40 cm, median cached position search takes 6.744 ms per query versus 9.524 ms for full joints; including preparation reduces the speed ratio to 1.114. The 256-target cached batch ratio is 8.817, reflecting smaller systems and shorter iteration tails. Initial fixed-chart pose search has no consistent latency advantage, and preparation makes it slower than full-joint search at every tested size. These workload distinctions prevent using the strongest batch ratio as an isolated-query claim.

The continuous variable count decreases from seven to three for position and seven to six for pose. Dense normal-matrix entries decrease from 49 to 9 or 36, while all fourteen joint inequalities remain. These are coordinate and matrix-size reductions, not percentages of a comparable physical feasible volume.

### E. Pose initialization, chart branches, and matched query time

The refined pose study separately trains six models for each of the 32, 40, and 48 cm cubes. The first three vary hidden width and training budget using the original feasible labels. Three additional models use face/edge-biased targets from existing training and validation references, a larger projection penalty, and increased widths/budgets. Every fit starts from a fixed initialization procedure, and selection uses validation seed MSE. The augmented archives retain 9,504 requested waypoint records per size. Selected boundary models have width 256; their validation seed MSE values are \(3.20\times10^{-5}\), \(7.21\times10^{-5}\), and \(9.87\times10^{-5}\) rad² per joint. Initial and augmented validation distributions differ, so their MSE change is not a data-only ablation.

The single target-conditioned chart and a chart rebuilt around its seed retain the original lattice coverage. The three-branch atlas recovers all nine missing largest-cube cases, reaching 5,175/6,561 and matching full-joint IK. All recovered cases use the positive null-offset branch and share one target pose across nine references. No earlier fixed-chart lattice success is lost. The improvement is 0.137 percentage points in raw lattice success; it is not a broad conversion of failed targets into feasible ones.

A separate fresh-reference audit uses 32 new joint references and 512 new targets per cube, half sampled on a face and half in the volume. The atlas and single neural chart succeed on 460, 406, and 355 requests. Full-joint IK from the reference succeeds on 460, 405, and 355; neural-seeded full-joint IK reaches 460, 406, and 355. This sample establishes no broad atlas coverage advantage. Historical lattices were used adaptively for diagnosis, whereas these new requests never enter training or checkpoint selection.

Timing uses an RTX 4070, matched teacher-feasible requests, sixteen single-query cases with three interleaved repetitions, and 256-query batches with five repetitions. All methods succeed on every timed request. End-to-end measurements include per-query preparation, validation, dispatch, and synchronized search, but exclude model loading and application startup.

**TABLE IV. Median single-query end-to-end latency (ms).** Full from reference uses the improved gain-ratio solver. The same-seed full-joint control receives exactly the neural joint initialization used by the reduced method.

| Cube side (cm) | Full from reference | Full with neural seed | Single six-coordinate chart | Cooperative three-chart atlas |
| --- | ---: | ---: | ---: | ---: |
| 32 | 14.218 | 8.427 | 8.386 | 8.701 |
| 40 | 17.020 | 8.377 | 8.373 | 8.511 |
| 48 | 15.035 | 8.314 | 8.174 | 8.389 |

The atlas is approximately 1.6–2.0 times faster than full-joint search from the reference on these feasible requests. The same-neural-seed control is approximately as fast, identifying initialization as the principal demonstrated latency mechanism. For 256-query batches, atlas/full-from-reference end-to-end times are 9.217/25.268, 13.589/27.919, and 8.700/34.937 ms. The atlas contains three discrete branches and up to eighteen scalar coefficient entries across concurrent searches; it is not a single six-variable global problem.

Failure-heavy batches reverse part of this benefit. On the largest lattice, cooperative atlas search takes approximately 2.77 s versus 1.05 s for full-joint search from the reference and 0.94 s for a single neural chart. Queries with no successful branch continue paying for multiple difficult searches. Cooperative stopping removes unnecessary work after a branch succeeds but cannot remove the cost of three unsuccessful branches.

### F. Analytical and numerical singularity evidence

The FR3 family in (27)–(28) has \(q_4^*\approx-0.4670024\) rad. A witness is \((0,0.72,0,q_4^*,0,0.53,-0.7853)\) rad. Its tool position is approximately \((0.5998933,0,0.5979838)\) m, and its center-relative rotation vector is \((0.1692423,-0.1692755,0)\) rad. The minimum joint margin is 0.0902 rad. The orientation components satisfy the prescribed range, and the pose lies inside the 24 cm and larger cubes. The smallest CPU singular value is \(1.62\times10^{-17}\), corroborating the analytical rank-five construction. This witness lies outside the 16 cm cube; the analysis supplies no absence certificate for that smaller cube.

Independent CPU FK agrees with every final pose-study learned single-chart, full-joint, fresh-audit, and cooperative-atlas classification. One supplementary fixed-chart gain-policy case at 32 cm exceeds the translation tolerance by 45.6 nm under CPU evaluation; its GPU success flag is retained and disclosed. Earlier expansion audits also flag two supplementary geometric/full-joint classifications at numerical tolerance boundaries. No learned expansion success flag changes under those audits. Float32 GPU kinematics and float64 CPU kinematics are treated as separate numerical evaluations rather than silently reconciling their boundary cases.

## VI. Discussion

The experiments separate three requirements often combined under dimensionality reduction. Reconstruction asks whether observed joint displacements lie near a chart. Transversality asks whether that chart retains the task-visible directions at a configuration. Coverage asks whether the chart's feasible image contains the desired targets. Equations (20)–(26) show why none can be replaced by the rank of \(B\) alone. Broad-distribution failures provide a practical example: reconstruction improved while finite-motion task success remained poor.

Regional position tasks are more favorable because the task removes four joint search variables, and the selected reference and teacher conventions produce a more consistent local joint section. The geometric chart already succeeds throughout the smallest sampled volume, so these results do not establish that learning is necessary for that region. The learned construction improves approximation and conditions charts on task context; its IK benefit depends on whether that conditioning changes useful solver behavior.

Pose reduction removes only one continuous variable. Its linear algebra savings are limited, while orientation residuals, preparation cost, joint limits, and iteration tails remain. Neural initialization addresses a different source of cost by starting closer to a task solution. The same-seed control in Table IV supports that explanation. The atlas recovers a useful redundancy branch on the historical largest-cube cases, but its discrete cost and failure-heavy slowdown prevent a general superiority claim over full-joint numerical IK.

The analysis states a certifiable objective for learning and verification. A regular reference chart is guaranteed by construction. A finite region requires a bound on task-subspace angles or derivative variation over an explicitly defined joint/coefficient domain. Continuous target coverage additionally requires an image argument such as Proposition 3. Existing endpoint lattices provide sampled evidence for a particular position volume and plane slice; they do not complete that verification. A separate broad free-position benchmark and a plane-specialized regional study are also needed to complete the corresponding comparisons.

The evaluation is kinematic simulation on one robot and one hardware configuration. It does not establish collision avoidance, dynamic feasibility, closed-loop stability, continuous atlas-branch trajectories, or hardware performance. Published analytical and GPU solvers remain untested comparators. Models are validation-selected within finite searches using a fixed training seed; timing repetitions quantify cases and repetitions rather than independent trained-model uncertainty. The supported conclusion is that task-conditioned charts and learned seeds can reduce computation on declared workloads, while global rank, feasibility, and speed remain separate properties to establish.

## VII. Conclusion

This paper formulates neural motion bases for joint-constrained reduced-coordinate IK and proves full task rank at regular reference configurations. Transversality and quantitative inverse conditions identify the additional analysis required for finite-domain rank preservation and continuous position or planar coverage. Simulations show that broad displacement compression is insufficient for reliable fixed-chart IK, whereas regional position tasks support useful sampled coverage. For pose tasks, neural seeds provide most of the measured feasible-query latency reduction, and a cooperative atlas matches historical full-joint lattice coverage at additional branch-search cost. The next analytical step is a certified enclosure over an explicit chart domain; the next empirical step is a plane-specialized study and matched comparison with external IK solvers.

## Appendix A. Derivation of the FR3 Elbow Singularity

In planar coordinates with joint two as origin, a rigid frame change gives the relevant anchor coordinates

\[
r_2=(0,0),\quad r_4=(a,d_3),\quad
r_6=(a-a\cos q_4-d_5\sin q_4,\ d_3-a\sin q_4+d_5\cos q_4).
\tag{29}
\]

The determinant of \([r_4-r_2,r_6-r_2]\) is (27). At its root, joints two, four, and six have parallel axes with collinear anchors. Use spatial twists at a fixed origin, \(s_i=[-a_i\times r_i;a_i]\). Twists of those three joints occupy \((v_x,v_z,\omega_y)\) and have rank at most two. The other joint axes lie in the plane, so their twists occupy \((v_y,\omega_x,\omega_z)\) and have rank at most three. Summing the complementary block ranks gives at most five. Rotating the plane or translating the evaluation point applies an invertible transformation and preserves this bound. The result remains valid for admissible free shoulder and wrist coordinates within the stated family.

## Appendix B. Evidence Provenance and Reproducibility

These paths identify completed local artifacts, not an already-public data repository or software release.

| Evidence | Project-relative source |
| --- | --- |
| Broad reconstruction and training search | `docs/training_improvement_study.md`; `results/study_20261031T104533864495Z/final_audit.json` |
| Broad pose/plane success and conditioning | `results/ik_20261031T112840039818Z/summary.json`, `chart_audit.json`, `trajectories.npz` |
| Fixed position chart and central plane slice | `results/workspace_20261031T115748066551Z/center_chart.npz` and `test/records.npz` |
| Ten size/task configurations and sixty fits | `results/expansion_20261031T123139402928Z/comparison.csv`, `trials.csv`, `audit.json` |
| Refined pose fits, timing, branch audit, and hashes | `results/pose_improvement_20261031T205242Z/training_trials.csv`, `latency.csv`, `coverage.csv`, `manifest.json` |
| Analytical singular witness | `results/pose_improvement_20261031T205242Z/singularity.json`; `docs/fr3_pose_singularities.md` |

The central-plane audit selects grid rows \(4+9j\), \(j=0,\ldots,80\), from the first 729 center-chart requests. Their targets and success flags agree exactly with the exported chart arrays. Independent CPU forward kinematics and \(J_pB\) recomputation give Table II. The tangent frame selects the base x and y axes. Affine reconstruction uses the exported chart reference, which reproduces solver precision; the original reference recorded with the dataset differs by at most \(5.89\times10^{-8}\) rad.

The broad network has two hidden layers of width 64 and 10,132 parameters. The small position study selects a width-32, two-hidden-layer model with 2,005 parameters. Expansion models independently vary width, depth, correction amplitude, objective, learning rate, regularization, and duration; complete trial tables are retained in the artifacts. Refined pose models have two hidden layers and twelve outputs: six graph entries and six tangent corrections. Training uses AdamW, cosine learning-rate decay, validation selection, and data/asset fingerprints. Pose trials use widths 32–256 and budgets of 600–6,000 epochs, with no sharing between cube sizes. The atlas uses null offsets \((-0.9,0,+0.9)\) rad and feasible seed-ray fractions.

The final pose study uses Python 3.12.16, PyTorch 2.11.0 with CUDA 13.0, MuJoCo 3.17.0, SciPy 1.18.1, and an NVIDIA GeForce RTX 4070. Mixed-precision evaluations retain their original task flags. All 23 project tests pass, including chart identities, derivatives, checkpoint loading, joint constraints, cooperative branch statuses, and singular-family checks. Software tests are distinct from a continuous workspace certificate.

## References

[1] C. W. Wampler II, “Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods,” *IEEE Transactions on Systems, Man, and Cybernetics*, vol. SMC-16, no. 1, pp. 93–101, 1986, doi: [10.1109/TSMC.1986.289285](https://doi.org/10.1109/TSMC.1986.289285).

[2] S. R. Buss and J.-S. Kim, “Selectively damped least squares for inverse kinematics,” *Journal of Graphics Tools*, vol. 10, no. 3, pp. 37–49, 2005, doi: [10.1080/2151237X.2005.10129202](https://doi.org/10.1080/2151237X.2005.10129202). [Author manuscript](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/SdlsPaper.pdf).

[3] P. Beeson and B. Ames, “TRAC-IK: An open-source library for improved solving of generic inverse kinematics,” in *Proceedings of the IEEE-RAS International Conference on Humanoid Robots*, pp. 928–935, 2015, doi: [10.1109/HUMANOIDS.2015.7363472](https://doi.org/10.1109/HUMANOIDS.2015.7363472).

[4] D. Ostermeier, J. Külz, and M. Althoff, “Automatic geometric decomposition for analytical inverse kinematics,” *IEEE Robotics and Automation Letters*, vol. 10, no. 10, pp. 9964–9971, 2025, doi: [10.1109/LRA.2025.3597897](https://doi.org/10.1109/LRA.2025.3597897). [Author preprint](https://arxiv.org/abs/2409.14815).

[5] B. Ames, J. Morgan, and G. Konidaris, “IKFlow: Generating diverse inverse kinematics solutions,” *IEEE Robotics and Automation Letters*, vol. 7, no. 3, pp. 7177–7184, 2022, doi: [10.1109/LRA.2022.3181374](https://doi.org/10.1109/LRA.2022.3181374). [Author preprint](https://arxiv.org/abs/2111.08933).

[6] M. N. Vu, F. Beck, M. Schwegel, C. Hartl-Nesic, A. Nguyen, and A. Kugi, “Machine learning-based framework for optimally solving the analytical inverse kinematics for redundant manipulators,” *Mechatronics*, vol. 91, Art. no. 102970, 2023, doi: [10.1016/j.mechatronics.2023.102970](https://doi.org/10.1016/j.mechatronics.2023.102970). [Author preprint](https://arxiv.org/abs/2211.04275).

[7] A. Paraschos, C. Daniel, J. Peters, and G. Neumann, “Probabilistic movement primitives,” in *Advances in Neural Information Processing Systems 26*, pp. 2616–2624, 2013. [Official proceedings](https://papers.nips.cc/paper_files/paper/2013/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html).

[8] F. Stella, M. M. Achkar, C. Della Santina, and J. Hughes, “Synergy-based robotic quadruped leveraging passivity for natural intelligence and behavioural diversity,” *Nature Machine Intelligence*, vol. 7, no. 3, pp. 386–399, 2025, doi: [10.1038/s42256-025-00988-x](https://doi.org/10.1038/s42256-025-00988-x).

[9] M. O’Connell, G. Shi, X. Shi, K. Azizzadenesheli, A. Anandkumar, Y. Yue, and S.-J. Chung, “Neural-Fly enables rapid learning for agile flight in strong winds,” *Science Robotics*, vol. 7, no. 66, Art. no. eabm6597, 2022, doi: [10.1126/scirobotics.abm6597](https://doi.org/10.1126/scirobotics.abm6597). [Author manuscript](https://authors.library.caltech.edu/records/q3grb-3vz72).

[10] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in *Proceedings of the 34th International Conference on Machine Learning*, PMLR, vol. 70, pp. 1126–1135, 2017. [Official proceedings](https://proceedings.mlr.press/v70/finn17a.html).

[11] B. Sundaralingam et al., “cuRobo: Parallelized collision-free minimum-jerk robot motion generation,” *arXiv preprint arXiv:2310.17274*, 2023, doi: [10.48550/arXiv.2310.17274](https://doi.org/10.48550/arXiv.2310.17274).

[12] E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 5026–5033, 2012, doi: [10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109). [Author paper](https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf).

[13] Google DeepMind and NVIDIA, “MuJoCo Warp,” *MuJoCo Documentation*, release 3.17. [Official documentation](https://mujoco.readthedocs.io/en/3.17.0/mjwarp/index.html).

---

### Author notes outside the manuscript

**Section argument.** Motivation and prior mechanisms → constrained IK formulation → learned rank-preserving charts and coefficient adaptation → conditional finite-domain proofs → broad, regional, planar, and pose-query evidence → computation and coverage limits.

**Terminology ledger.**

| Canonical term | Definition | Distinction retained |
| --- | --- | --- |
| Task Jacobian, \(J_\tau\) | Joint-velocity to task-velocity map | Full-robot rank differs from reduced \(J_\tau B\) rank |
| Motion basis, \(B\) | Coefficient-to-joint-displacement map | Full column rank does not establish task span |
| Affine chart | \(q_0+\operatorname{range}B\), restricted by joint limits | One fixed chart differs from a target-conditioned family or atlas |
| Shared representation learning | Offline fitting of neural weights | Online coefficients adapt; neural weights remain fixed |
| Planar-position IK | Three ambient position constraints with a two-coordinate target family | Three coefficients do not imply a three-dimensional plane or a heading constraint |
| Sampled coverage | Success at the recorded target requests | Distinct from continuous rank and continuous image coverage |

**Claim–evidence map.** Reference rank is proved in Proposition 1 and implemented by construction. Finite-domain transversality and coverage are conditional results in Propositions 2–3; their hypotheses have not been certified over the reported large regions. Sampled volume/plane span is supported by the fixed-chart CPU audit in Table II. Regional coverage and timing are measured in Tables III–IV. Broad unrestricted position-only IK and a separately trained fixed-plane benchmark remain missing. No global coverage or global no-additional-singularity claim is made.

**Main-text allocation.** Reconstruction/task-success disagreement, regional task comparison, same-seed latency control, branch-search tradeoff, and the local-versus-regional proof boundary remain in the main text because they change the conclusions. Complete histories, hyperparameter trials, source hashes, package identities, and tolerance-boundary records are assigned to Appendix B and the source artifacts.

**Submission preparation.** This is an IEEE-style Markdown manuscript with verified reference links, not a compiled two-column submission. Author names, affiliations, funding, release/accession information, conference page limits, and author-reviewed disclosure statements must be supplied before submission. Continuous rank/coverage results must remain explicitly conditional unless the corresponding workspace certificates are completed. Venue-specific LaTeX conversion and page-length editing can preserve this mathematical and evidential structure.
