# Neural motion-basis training improvement study

The selected model reduces the held-out task-MSE sum from 0.08347872 to 0.00614023 on the same complete 20-archive corpus: a 92.64% reduction, or 13.60 times lower error. Improvement comes from planar reconstruction. Pose test MSE increases from 0.00100044 to 0.00113577. This is the best validation configuration among ten measured configurations, not a global optimum or an IK-success result.

## Corpus and historical evidence

The initial inventory contained 18 independent timestamped NPZ archives, 1,152 references, and 224,640 waypoint records across the existing splits. The original checkpoint basis_20261031T095508626199Z.pt had trained on only motions_20261031T094447833506Z.npz. Its final single-archive losses were 0.03092284 training, 0.06722518 validation, and 0.08593707 test. Validation reached its historical minimum at epoch 73, but the old policy saved epoch 500.

Loading now pools every NPZ under data/ and preserves source-qualified episode identities, reference links, inclusion masks, and split assignments. No earlier archive or label was overwritten. No teacher labels were centered, shortened, or replaced.

After ten comparisons on the original 18-file snapshot, the selected architecture and training settings were frozen. Two more independent archives were generated with the unchanged collector:

- motions_20261031T105514357492Z.npz
- motions_20261031T105537858427Z.npz

The final fit includes all 20 archives, including the new training groups. Validation selection uses the combined validation groups. Their new test groups were first evaluated after the final fit; they contain 16 pose and 32 plane episodes. The old test groups had already been inspected during historical diagnosis and the first six comparisons. Follow-up configurations were chosen using validation, but the old tests must not be described as an untouched benchmark.

| Task | Training episodes / samples | Validation episodes / samples | Test episodes / samples |
| --- | ---: | ---: | ---: |
| Pose | 960 / 62,400 | 160 / 10,400 | 160 / 10,400 |
| Plane | 1,920 / 124,800 | 320 / 20,800 | 320 / 20,800 |

There are 1,280 distinct references and 249,600 retained waypoint records. All 2,560 planar episodes succeeded and were included. Recorded reference/plane distribution shifts are retained: joint 1 occupies different train/validation/test intervals and plane-normal tilts progress from 0–45° to 50–65° and 70–85°. Splitting individual waypoints would conceal this extrapolation requirement.

## Diagnosis

The most consequential measured bottleneck was the correction scale in F = G + s(I - VV.T)U. With tanh outputs and s = 1, the learned graph remained strongly biased toward the task right inverse G. On all 18 archives, the same-scale retraining achieved validation sum 0.04895153. Changing only scale to 8 reduced it to 0.00705691; scale 32 reduced it to 0.00541306. This comparison supports a scale/optimization bottleneck. It does not prove a universal expressivity limit for the original model.

The original training also used only one of the 18 available archives and retained heavily overfit final weights. Full-corpus loading and explicit validation selection address these observed failures. The full-corpus scale-1 experiment changes data and optimization together, so its gain over the historical checkpoint is not attributed to data quantity alone.

Plane demonstrations have substantial teacher-induced joint drift. On the original corpus, median endpoint distance from q_ref was 1.27–1.34 rad even though the projected tip paths return to their starting points. The largest endpoint norm was 2.56 rad. General joint paths return exactly to q_ref. Position-only SLSQP with sequential warm starts has no joint-return constraint or unique nullspace solution. This makes the plane labels harder to predict with a single three-dimensional subspace. The drift remains in the objective; removing it would change the dataset rather than improve the same task.

General motions also use independently sampled harmonic directions that are absent from the network input q_ref. A wider q_ref-only network cannot infer the realized random episode directions. This is a reasoned explanation for the persistent pose error and weak benefit from capacity; it is not a proved Bayes-risk value.

Uncentered second-moment eigenvalues distinguish representational error from prediction difficulty:

| Original 18-file split | Per-episode pose PCA oracle | Per-episode plane PCA oracle | Best constant pose subspace | Best constant plane subspace |
| --- | ---: | ---: | ---: | ---: |
| Training | 0.000007143 | 0.001097502 | 0.000955241 | 0.003958872 |
| Validation | 0.000010100 | 0.001171370 | 0.000759574 | 0.004160480 |
| Test | 0.000007841 | 0.001271213 | 0.000840296 | 0.004536419 |

Each episode oracle sees its full trajectory and ignores the task-right-inverse constraint. Each constant subspace is fitted separately to the stated split, including held-out labels. These are diagnostic references, not trained predictors or deployable baselines. Their gaps do not establish how much a q_ref-conditioned model can recover. The selected plane model approaches the constant-subspace scale; reaching the per-episode oracle would require resolving unobserved episode variability and geometric restrictions.

## Measured iterations

All ten configurations use the same original 18-archive snapshot, seed 0, all waypoints, the same loss, and validation-only weight selection. The base training recipe is AdamW, learning rate 1e-3, weight decay 1e-4, 300 epochs, and cosine decay to 2% of the initial rate. The sum is pose MSE plus plane MSE in rad².

The first comparisons isolate correction scale, then test richer normal conditioning and yaw symmetry with larger capacity. Follow-up comparisons test very large scales, longer low-rate training, and a smaller more strongly regularized network.

| Trial | Width | Scale | Extra normal features / yaw symmetry | Epochs | LR / weight decay | Selected epoch | Validation sum |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| corpus_baseline | 64 | 1 | No / no | 300 | 1e-3 / 1e-4 | 300 | 0.04895153 |
| scale8 | 64 | 8 | No / no | 300 | 1e-3 / 1e-4 | 60 | 0.00705691 |
| **scale32** | **64** | **32** | **No / no** | **300** | **1e-3 / 1e-4** | **50** | **0.00541306** |
| normal128 | 128 | 32 | Yes / no | 400 | 1e-3 / 1e-4 | 30 | 0.00560924 |
| yaw128 | 128 | 32 | Yes / yes | 400 | 1e-3 / 1e-4 | 120 | 0.00546323 |
| regularized256 | 256 | 32 | Yes / yes | 600 | 2e-3 / 1e-3 | 110 | 0.00544470 |
| scale128 | 64 | 128 | No / no | 300 | 1e-3 / 1e-4 | 140 | 0.00545578 |
| scale512 | 64 | 512 | No / no | 300 | 1e-3 / 1e-4 | 260 | 0.00542742 |
| slow64 | 64 | 64 | No / no | 800 | 3e-4 / 3e-3 | 150 | 0.00544174 |
| small32 | 32 | 32 | No / no | 600 | 1e-3 / 1e-2 | 330 | 0.00563653 |

Several leading results are close. No repeated-seed significance claim is made. Width 64 / scale 32 retained the lowest observed validation sum and the original 10,132-parameter size. Greater capacity, larger scales, and longer fitting did not improve that criterion. The total comparison fitting time was about 238 seconds; final retraining took 19.07 seconds.

The 20-archive refit selects epoch 40 of 300. Its validation sum is slightly higher than the 18-archive winner because both the evaluated corpus and minibatch sequence changed. The extra two files provide a fresh audit and more training examples; this experiment does not establish that increasing 18 to 20 files improves accuracy.

## Same-corpus before/after comparison

Both checkpoints below are evaluated on the exact same 20-file corpus. The old checkpoint's original training data remains unchanged; it is the historical trained baseline rather than a fresh scale-1 refit.

| Split | Old pose MSE | Selected pose MSE | Old plane MSE | Selected plane MSE | Old sum | Selected sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 0.000936185 | 0.000921104 | 0.043780368 | 0.004052877 | 0.044716552 | 0.004973981 |
| Validation | 0.000894606 | 0.000974160 | 0.070844902 | 0.004596158 | 0.071739509 | 0.005570318 |
| Test | 0.001000441 | 0.001135769 | 0.082478284 | 0.005004457 | 0.083478724 | 0.006140226 |

Plane test MSE falls by 93.93%, while pose test MSE rises by 13.53%. The selected test model leaves approximately 2.77% of plane displacement energy and 15.14% of pose displacement energy unexplained. The large aggregate gain must not be presented as improvement in both tasks.

| Fresh test subset | Old model | Selected model |
| --- | ---: | ---: |
| Pose MSE | 0.000779857 | 0.001155051 |
| Plane MSE | 0.093443440 | 0.003939418 |
| Sum | 0.094223296 | 0.005094469 |

Fresh-test sum is 94.59% lower, supporting the plane improvement beyond the previously inspected corpus. This audit is small and follows the same collector distribution. It is not a robot-control or task-success test.

## Geometry, numerical verification, and remaining work

The construction still has JF = I in exact arithmetic. Increasing s tilts the learned subspace toward nullspace motion and reduces the task-image conditioning relative to the historical model.

| Task | Minimum singular value of JB: training | Validation | Test |
| --- | ---: | ---: | ---: |
| Pose | 0.05263661 | 0.05606000 | 0.05945350 |
| Plane | 0.02050962 | 0.02231178 | 0.02364966 |

Historical single-archive test minima were approximately 0.1203 for pose and 0.1780 for plane. These different sampled reference populations do not form a controlled conditioning comparison, but the lower observed magnitudes require attention in coefficient IK. The objective remains reconstruction alone, so this conditioning tradeoff must be assessed in coefficient IK. These samples establish neither a uniform regional bound nor finite-target coverage.

Network and reference geometry use CUDA float32; the exact uncentered sufficient-statistic loss uses float64 Cholesky projection. It is algebraically the same all-waypoint loss. Independent NumPy float64 QR and direct reconstruction over every stored waypoint agree within 5.59e-10 rad². The maximum CUDA basis orthogonality error is 5.97e-7. Included paths, all original archive hashes, the complete current file set, corpus hash, and final training source fingerprints were verified.

Eleven tests verify the physical/collection contracts, corpus completeness, cross-file split isolation, basis identities/symmetries, loss and gradient equivalence, failed-label exclusion, validation-selected saving, and reloaded metrics. Ruff lint/format checks pass. The final suite completed in 24.43 seconds with one first-use cuBLAS context initialization warning during the float64 gradient check.

The remaining error calls for an explicitly specified next research target: predictable shared motion generators or repeated motions per reference, and a jointly meaningful plane-teacher nullspace policy, followed by constrained coefficient IK evaluation. Changing those labels or inputs changes the learning problem and must be evaluated alongside the present dataset. Adding unseen episode information solely to approach the PCA oracle would undermine the stated reference-conditioned representation.

## Artifacts and reproduction

Use the current selected default on the complete corpus:

~~~powershell
uv run --locked python -m metaarm.training data
~~~

The controlled study can be rerun on the current complete corpus:

~~~powershell
uv run --locked python -m metaarm.study data
~~~

The latter now runs all ten recipes and writes a new study directory. It uses the directory's current inventory; it does not recreate the earlier 18-file snapshot automatically.

Artifacts from this study:

- checkpoints/basis_20261031T105751490589Z_audited.pt: final 20-file selected model, full test/fresh metrics, and conditioning.
- checkpoints/basis_20261031T105751490589Z.pt: original final-fit checkpoint with selected and last-epoch states/history.
- results/study_20261031T104533864495Z/final_audit.json: all 20 exact source paths/hashes/metadata, full/fresh comparisons, independent numerical audit, and counts.
- results/study_20261031T104533864495Z/trials.json: all ten trial specifications, selected epochs, timings, and validation metrics.
- results/study_20261031T104533864495Z/diagnosis.json: original 18-file diagnosis, historical metrics, PCA references, and endpoint statistics.
- results/study_20261031T104533864495Z/round2_selection.json: validation winner after all ten comparisons.
- results/study_20261031T104533864495Z/summary.json and selected.pt: first-six comparison result on 18 files, retained as historical artifacts; the winning recipe remained unchanged after the follow-ups.
- Each named trial checkpoint retains its training history and selected/final weights. The source/ directory captures the original first-six implementation; final_source/ captures the final four training/study files, and final training source hashes are recorded in the final checkpoint. verification.json records the final checks and exact audited-checkpoint reload.

Final corpus SHA-256 is 0347795c5ed32b24afe6f53112dfd6995856ce96462c8c900f833347af8a78cb. All earlier checkpoints and data archives are preserved. Data/results/checkpoints remain local generated artifacts excluded by Git.
