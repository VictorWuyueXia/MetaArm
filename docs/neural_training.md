# GPU neural motion-basis training

The trainer learns the six-/three-direction representation in the [mathematical formulation](1-mathematical_formulation_and_implementation_plan.md). The [measured improvement study](training_improvement_study.md) records the parameter search, historical comparison, and remaining limitations.

## Run on the complete corpus

~~~powershell
uv run --locked python -m metaarm.training data
~~~

A directory means every recursively discovered NPZ archive, sorted by path. One explicit NPZ file is also accepted. Existing archive identities and reference-linked train/validation/test memberships are preserved. Nothing selects only the latest file or repartitions individual waypoints.

Each invocation writes an exclusive UTC-timestamped checkpoints/basis_<timestamp>.pt file. The command has one required source argument and no tuning flags. Missing data, incompatible assets, crossed split identities, or nonfinite included labels raise errors.

## Basis construction

The selected model is 7 → 64 → 64 → 84, with tanh after each affine layer and 10,132 trainable parameters. Its input is the joint-limit-normalized reference configuration:

~~~
x = 2 * (q_ref - q_min) / (q_max - q_min) - 1
~~~

Outputs partition into U6 (7×6), Ut (7×3), and Un (7×3). For pose, J contains all six base-aligned Jacobian rows. For plane, J contains the three position rows and a unit normal n defines U = Ut + (Un - Ut) @ (n n.T). This expression has the same subspace for n and -n.

~~~
J.T = V H
G = solve_triangular(H, V.T).T
F = G + s * (U - V @ (V.T @ U))
B = qr(F).Q
~~~

The fixed selected scale is s = 32. The original model used s = 1; enlarging the bounded nullspace correction was the strongest measured improvement. G = V H^{-T} is evaluated by triangular solve. The task geometry is cached once per reference; the basis remains fixed within an episode or coefficient solve.

For regular references, JF = I in exact arithmetic for every scale. QR produces orthonormal columns, so the reference task image retains full rank. The numerical reference-Jacobian threshold is 1e-6. There is no singular-value truncation or regularized substitute. Large corrections can reduce the conditioning of JB; the study reports that tradeoff explicitly.

The Python model specification has exactly four research fields: width, scale, normal_features, equivariant. The last two enable the tested symmetric normal features and base-yaw canonicalization. Both are false in the selected model. These alternatives did not improve the selected validation objective. Device, precision, seed, batching, and output locations are fixed implementation choices.

## Exact all-sample objective

For each episode e, retain the uncentered second moment C_e = mean_t(delta_q_t delta_q_t.T) and its waypoint count T_e. Do not subtract the path mean: the affine origin remains q_ref.

For full-column-rank F:

~~~
P = F @ solve(F.T @ F, F.T) = B @ B.T
E = I - P
episode_mse = trace(E @ C_e @ E.T) / 7
task_mse = sum_e(T_e * episode_mse_e) / sum_e(T_e)
loss = pose_mse + plane_mse
~~~

This is exactly the original waypoint reconstruction objective, including every joint and waypoint. Covariance compression removes repeated evaluations of the same episode basis; it does not subsample, center, smooth, or relabel trajectories. The task sum keeps equal category weight regardless of their different sample counts.

Reference/neural computation uses CUDA float32. Second moments are computed from the stored labels in float64, and the small Gram matrices, Cholesky solves, and projection losses use CUDA float64. Positive-definiteness failures raise errors. Inference returns the reduced QR basis. Loss and parameter-gradient equivalence to direct QR/sample reconstruction are tested in float64.

All archives must share physical asset hashes and joint limits. Failed planar episodes are excluded before their labels reach CUDA, preserving the original whole-path inclusion masks. Local episode IDs are qualified by source ID; global reference offsets preserve plane-to-reference links. Identical reference configurations cannot cross split boundaries in different archives.

## Optimization and selection

The frozen default recipe is seed 0, AdamW learning rate 1e-3, weight decay 1e-4, and 300 complete epochs. A cosine schedule reduces the rate to 2% of its initial value. Each task is independently shuffled and partitioned into paired episode batches capped at 256; every retained training episode contributes once per epoch. On the 20-archive corpus, each epoch has eight paired updates of 120 pose and 240 plane episodes, each containing all 65 waypoints.

Training and validation losses are recorded every ten epochs and at the final epoch. The saved state_dict is the lowest validation task-MSE sum among initialization and these evaluations. The final-epoch weights are retained separately. Test labels do not contribute optimizer updates or checkpoint selection; ordinary training does not evaluate them.

The original fixed-final-weight policy is preserved as historical evidence in the study report. Validation selection, AdamW, the cosine schedule, episode batching, and the correction scale are explicit changes authorized by the request to improve the training scheme.

## Checkpoints and reload

The ordinary training checkpoint contains:

- state_dict: validation-selected CPU neural weights and joint-limit normalization buffer.
- last_state_dict: final-epoch CPU weights.
- initial, history, last, final: initialization, sampled training history, final-epoch metrics, and selected-state training/validation metrics.
- metadata: complete model/training specifications, selected epoch, every archive path/hash and source metadata, corpus hash, sample counts, source-code hashes, Git revision, seed, device/package identity, and fitting time.

The audited study checkpoint additionally contains final test metrics, fresh_audit metrics, sampled conditioning, and the post-selection audit description. Loading uses weights_only=True.

~~~python
import torch
from metaarm.basis import MotionBasis, reference_geometry

checkpoint = torch.load(
    "checkpoints/basis_20261031T105751490589Z_audited.pt",
    weights_only=True,
)
specification = checkpoint["metadata"]["specification"]["model"]
network = MotionBasis(checkpoint["state_dict"]["joint_limits"], specification).to("cuda:0")
network.load_state_dict(checkpoint["state_dict"])
network.eval()

# q_ref, J_ref, and normal are explicit matching CUDA reference tensors.
geometry = reference_geometry(J_ref, "plane")
basis = network(q_ref, geometry, "plane", normal).detach()
~~~

Future coefficient optimization must freeze this basis for the declared solve and impose the coupled joint limits on reconstructed configurations. Reconstruction error and sampled task rank do not establish IK success or finite workspace coverage.

## Measured result and verification

The final fit used every one of 20 archives: 1,280 references, 1,280 pose episodes, 2,560 plane episodes, and 249,600 joint-displacement samples across the three splits. It selected epoch 40 of 300 and took 19.07 seconds of fitting on the RTX 4070.

| Split | Pose MSE | Plane MSE | Sum |
| --- | ---: | ---: | ---: |
| Training | 0.000921104 | 0.004052877 | 0.004973981 |
| Validation | 0.000974160 | 0.004596158 | 0.005570318 |
| Test | 0.001135769 | 0.005004457 | 0.006140226 |

MSE is per joint in rad². The fresh two-archive test subset has sum 0.005094469. See the study report for the same-corpus baseline, pose tradeoff, and the distinction between previously inspected tests and the new audit.

Eleven tests cover physical simulation/collection, full-corpus loading, cross-archive split isolation, masked failed labels, basis identities and symmetries, CUDA gradients, covariance/sample loss-gradient equivalence, validation selection, exclusive output, and checkpoint reload. An independent NumPy QR reconstruction of every waypoint agrees with the final GPU metrics to within 5.59e-10 rad². Every source archive and training source-code fingerprint was verified.

Use a new project-local pytest temporary directory under runs/ on this workstation; its existing system pytest temp directory denies access. The training code has no backend or numerical fallback.

Finite-task follow-up: [GPU coefficient IK](coefficient_ik.md) evaluates the saved model in fixed-chart constrained solves. Its limited held-out trajectory completion confirms that reconstruction accuracy and reference rank do not establish finite-motion task coverage.
