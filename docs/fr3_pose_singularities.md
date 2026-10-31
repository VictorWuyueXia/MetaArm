# Analytical FR3 pose-Jacobian singularities

## Jacobian under analysis

The current implementation uses the base-aligned geometric tool Jacobian

\[
J(q)=\begin{bmatrix}
\omega_1\times(p-r_1)&\cdots&\omega_7\times(p-r_7)\\
\omega_1&\cdots&\omega_7
\end{bmatrix}\in\mathbb R^{6\times7},
\]

where the axes and anchors come from the pinned FR3 MJCF. Full pose singularity means rank J < 6. The exact scalar criterion is

\[
\det(JJ^T)=\sum_{i=1}^7\det(J_{\setminus i})^2=0.
\]

This Cauchy–Binet identity requires all seven maximal minors to vanish, rather than only one square sub-Jacobian. The nonzero translation/rotation normalization constants and the inverse right Jacobian used by the principal rotation-log residual preserve rank away from the explicitly rejected logarithm cut. Floating-point singular values corroborate rank; they are not the analytical proof.

## Exact planar elbow family

Read the exact lengths from `assets/franka_fr3/fr3.xml`:

\[
a=0.0825,\qquad d_3=0.316,\qquad d_5=0.384\quad\text{m}.
\]

Set q3 = q5 = 0. The arbitrary q1 rotation merely rotates the plane. A rigid change of frame eliminates q2 when checking collinearity. In that plane, relative to joint two, the anchors of joints two, four, and six have xz coordinates

\[
r_2=(0,0),\quad r_4=(a,d_3),\quad
r_6=(a-a\cos q_4-d_5\sin q_4,\ d_3-a\sin q_4+d_5\cos q_4).
\]

Their signed two-dimensional area is

\[
A(q_4)=\det[r_4-r_2,\ r_6-r_2]
=(d_3d_5-a^2)\sin q_4+a(d_3+d_5)\cos q_4.
\]

Thus the joint-limit-admissible root is

\[
q_4^*=-\operatorname{atan2}\big(a(d_3+d_5),d_3d_5-a^2\big)
=-0.4670024236530116\text{ rad}.
\]

To prove rank loss, use spatial twists at the fixed origin, S_i = [-omega_i cross r_i; omega_i]. Translation of the evaluation point from the origin to the attachment site is an invertible twist transformation, so it preserves rank. In planar coordinates:

- Joints 2, 4, and 6 have parallel axes perpendicular to the plane. Their twists occupy the three coordinates (v_x, v_z, omega_y). When A = 0 their anchors are collinear, so this block has rank at most two.
- Joints 1, 3, 5, and 7 have axes in the plane. Their twists occupy the complementary three coordinates (v_y, omega_x, omega_z), so this block has rank at most three.

Consequently rank J <= 2 + 3 = 5 for q3 = q5 = 0 and q4 = q4*, for every admissible q1, q2, q6, and q7. The 0.088 m wrist offset and 0.107 m tool offset do not invalidate this argument: the former affects only the second block and the latter is an invertible tool-point translation. This is an analytical singular family of the complete seven-joint robot, independent of a learned basis.

## A witness inside the tested pose volumes

Choose, in radians,

\[
q^*=(0,\ 0.72,\ 0,\ -0.4670024236530116,\ 0,\ 0.53,\ -0.7853).
\]

Double-precision MuJoCo gives

\[
p(q^*)=(0.599893327820,\ 0,\ 0.597983832849)\text{ m},
\]

and the rotation vector relative to the study's center rotation is

\[
\phi=(0.169242258300,\ -0.169275488352,\ 0)\text{ rad}.
\]

Both nonzero components are strictly below pi/18 = 0.174532925199 rad. The orientation domain is a componentwise rotation-vector box; its Euclidean angle may exceed ten degrees. The minimum joint margin is 0.0902 rad. The largest Cartesian offset from (0.5,0,0.5) is 0.099893327820 m, leaving more than 20 mm inside the 24 cm cube. Orientation margin exceeds 0.00525 rad. These margins are much larger than numerical roundoff.

| Cube side | This singular pose lies inside the xyz and orientation domain |
| --- | --- |
| 16 cm | No: this witness exceeds the 8 cm Cartesian half-width |
| 24 cm | Yes |
| 32 cm | Yes |
| 40 cm | Yes |
| 48 cm | Yes |

The witness's six singular values are approximately (1.89869, 1.82412, 0.87760, 0.48697, 0.27911, 1.62e-17); its rank is five. Tests also verify the rank loss after changing the family's free shoulder and wrist coordinates. The lengths and area identity provide the proof; FK supplies the workspace-membership check with substantial interior margins.

**Scope of the conclusion:** the 24 cm and larger task volumes contain poses with a singular admissible joint preimage. This does not prove every IK branch of that pose is singular, and does not make the entire volume unreachable. This calculation does not certify either existence or absence of other singular families inside the 16 cm volume. No uniform nonsingularity claim is justified there by sampled results alone.

## Full and reduced rank are different questions

For the target-conditioned chart, G = J_ref^T (J_ref J_ref^T)^-1, z belongs to ker J_ref, and F = G + z u^T. Therefore J_ref F = I exactly. With F^T F = LL^T and B = F L^-T,

\[
J_{ref}B=L^{-T},
\]

which is nonsingular whenever the reference has full task rank and the declared null gauge exists. This is only a statement at the reference. At another configuration, J(q)B may lose rank even when J(q) has rank six. At a full-robot singularity, every six-column chart satisfies rank(J(q)B) <= 5. Cholesky factorization and the explicit gauge check reject unsupported reference charts instead of substituting another representation.

The workspace witness can belong to a different joint branch from those used by the experiments. Its existence explains why a global regularity guarantee is unavailable, but is not evidence that every observed optimization failure was caused by this family. The remaining nine single-chart lattice failures approach joint six's lower limit near the elbow value; that observation motivates the branch atlas without establishing a unique cause.

## Reproduction and evidence

`metaarm.pose_study.singularity_report(destination)` writes the witness, area factor, singular values, joint margin, orientation vector, and cube membership. The completed run saves these in `results/pose_improvement_20261031T205242Z/singularity.json`. `tests/test_pose_improvement.py` verifies the rank certificate and chart algebra. The separate [pose study](pose_ik_improvement.md) reports numerical performance and its limits.
