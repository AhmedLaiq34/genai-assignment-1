# Task 1 training report (final model, trained on a Kaggle T4)

Written 2026-10-04 by the Task 1 lead. This replaces the earlier version of this file, which described the first (dense-latent) model.
All numbers come from `artifacts/logs/v2/t1_overnight_summary.json`, `studies/t1_universal_v2/`, the final run's `metrics.jsonl`,
and the evaluation folders `artifacts/eval/task1/20261004-1514_kaggle_t1_final_v2_{val,test}/`. Validation numbers were also re-computed on the
local laptop and match the Kaggle ones. The test set was opened **once** (after the model was final, with the student's approval), logged in `artifacts/test_access.log`.

## 1. Result in one paragraph

A convolutional autoencoder with a convolutional latent grid (16x16x8 = 2,048 values, 24:1 compression, no skip connections) was tuned with Optuna
(40 counted trials: 35 complete, 5 pruned, 0 failed), then trained for 100 epochs. The best checkpoint (epoch 98) has validation J 0.1237
(SSIM 0.791, PSNR 25.17 dB) and **test J 0.1330 (SSIM 0.775, PSNR 24.62 dB)** over 36,690 test cases. It beats the "do nothing" baseline strongly on
salt-and-pepper and on medium/high occlusion and high blur, but is slightly worse than the untouched input on mild damage (low blur, low occlusion).
The model is exported to ONNX (parity passed) and promoted as `t1_universal_ae`.

## 2. How we got here (short history)

1. **First model: flatten + dense vector latent** (local RTX 3050). A 40-trial study and a 100-epoch training gave validation J 0.3134, SSIM 0.454: outputs were blurry blobs.
   Diagnosis (details in `docs/LESSONS_FROM_TASK1.md`): the model had learned only an 8x8-resolution thumbnail (an 8x8 thumbnail of the clean image has J 0.3131);
   a 4x bigger bottleneck changed nothing (J 0.3145 vs 0.3151 after 30 epochs); the same model overfits 64 images to J 0.09, so the code is sound.
2. **Fix (decision D40):** a convolutional latent grid. Prototype results (30 epochs, validation only): 16x16x16 J 0.139, 8x8x64 J 0.166, 8x8x16 J 0.185, 16x16x4 J 0.190.
3. **Study `t1_universal_v2`** (this report) on the new architecture, run on Kaggle (decisions D16, D40, D41). The first dry run on Kaggle found two problems that the local dry run could not
   (a hard-coded local data path in the ONNX parity check, and stale code after a new dataset version); both were fixed before the real run (`docs/TASK1_PROBLEMS_AND_FIXES.md`).

## 3. Model

`UniversalAE(base_channels=64, depth=3, bottleneck_dim=2048, dropout=0.1477, latent="conv")`, 1,322,507 trainable parameters.
Encoder: three stride-2 4x4 convolutions with BatchNorm and ReLU (128 -> 64 -> 32 -> 16, channels 64, 128, 256). Latent: a 1x1 convolution to 8 channels (16x16x8 = 2,048 values; the
input has 49,152 values, so 24:1), with channel dropout before it. Decoder: a 1x1 convolution back to 256 channels, then three stride-2 transposed convolutions (128 -> 64 -> 3 channels), sigmoid.
No skip connections. Input and output are 128x128x3 in [0, 1]. Figure: `report/figures/t1_architecture.png`.

Loss: alpha*L1 + (1-alpha)*(1-SSIM) with alpha 0.5022 (chosen by Optuna). AdamW (lr 2.3e-3, weight decay 1e-4), cosine schedule, batch 32, no mixed precision, seed 42.
Selection by best validation J = 0.5*L1 + 0.5*(1-SSIM), independent of alpha.

## 4. Optuna study

- Study `t1_universal_v2`, TPE sampler (seed 42 + number of existing trials), MedianPruner (`n_startup_trials 8`, `n_warmup_steps 8`), 15 epochs per trial, objective J on the validation manifest (minimise).
- Run on Kaggle (T4), one session, **89.7 minutes**, 1 attempt, no failures. Counted trials: **35 complete, 5 pruned, 0 failed (40)**.
- Pruned trials: 8 (epoch 8, J 0.217, batch 128, 32 channels), 10 (epoch 8, J 0.322, batch 128, 16 channels), 16 (epoch 13, J 0.186, batch 32, 16 channels), 34 (epoch 8, J 0.174, batch 32, 32 channels), 38 (epoch 8, J 0.166, batch 32, 48 channels).
- Complete-trial J: min 0.1406, median 0.1542, max 0.3218.
- **Best trial: 33, J 0.1406** (15 epochs). Parameters: lr 2.304e-3, batch 32, bottleneck 2048 values, encoder channels 64, dropout 0.1477, alpha 0.5022.
- Top five trials (J): 33 (0.1406), 39 (0.1409), 32 (0.1412), 29 (0.1413), 31 (0.1414). All five share batch 32, 2048 latent values and 64 channels, with lr 2.0e-3 to 2.6e-3 and alpha 0.53 to 0.56.

Search space (provisional ranges, unchanged during the study):

- lr: float, log scale, 1e-4 to 3e-3
- batch: categorical {32, 64, 128}
- bottleneck_dim (total latent values): categorical {1024, 2048, 4096} (48:1, 24:1, 12:1)
- encoder_channels: categorical {16, 32, 48, 64}
- dropout: float, 0.0 to 0.3
- alpha: float, 0.5 to 0.95

Files: `studies/t1_universal_v2/` (`trials.csv`, three plots, DB copy); figures `report/figures/t1_optuna_history.png`, `t1_optuna_importance.png`, `t1_optuna_parallel.png`; `report/tables/t1_optuna_trials.csv`.
The first (dense) study is kept in `studies/t1_universal/` as evidence for the report's "alternatives investigated" section.

## 5. Final training

- Config `configs/task1_final_v2.yaml` (trial 33's parameters, 100 epochs), run `20261004-1514_kaggle_t1_final_v2`, one attempt, **17.0 minutes**, 9,016 optimiser steps at the best epoch (epoch 98), median 9.6 s/epoch.
- Best validation J 0.1237 at epoch 98 (J 0.1239 at epoch 100).

| Epoch | train loss | val J | val SSIM | val PSNR (dB) |
|---|---|---|---|---|
| 1 | 0.3062 | 0.2377 | 0.612 | 18.98 |
| 5 | 0.1708 | 0.1702 | 0.724 | 21.47 |
| 10 | 0.1547 | 0.1473 | 0.758 | 23.01 |
| 20 | 0.1457 | 0.1411 | 0.770 | 23.18 |
| 40 | 0.1394 | 0.1335 | 0.780 | 24.02 |
| 60 | 0.1339 | 0.1276 | 0.787 | 24.71 |
| 80 | 0.1296 | 0.1249 | 0.790 | 25.05 |
| 98 (best) | 0.1289 | 0.1237 | 0.791 | 25.17 |
| 100 | 0.1301 | 0.1239 | 0.791 | 25.13 |

Figure: `report/figures/t1_training_curves.png`.

## 6. Validation results (2,944 rows, 736 per condition)

Best checkpoint. "Input" columns score the corrupted input directly against the clean target (the do-nothing baseline). Severity labels here are tertiles of the sampled training-range
parameter, **not** the fixed test severities.

| Condition | MAE | SSIM | PSNR (dB) | J | J of input |
|---|---|---|---|---|---|
| clean | 0.0330 | 0.8302 | 26.68 | 0.1014 | 0.0000 |
| salt_pepper | 0.0338 | 0.8195 | 26.50 | 0.1072 | 0.3417 |
| gaussian_blur | 0.0329 | 0.8111 | 26.68 | 0.1109 | 0.0908 |
| occlusion | 0.0557 | 0.7051 | 20.82 | 0.1753 | 0.1909 |
| overall | 0.0388 | 0.7915 | 25.17 | 0.1237 | 0.1558 |

By severity (J, model vs input): salt low 0.1013 vs 0.2508, medium 0.1083 vs 0.3595, high 0.1119 vs 0.4157; blur low 0.1071 vs 0.0505, medium 0.1087 vs 0.0970, high 0.1159 vs 0.1173;
occlusion low 0.1491 vs 0.1304, medium 0.1793 vs 0.1978, high 0.2050 vs 0.2629. Files: `report/tables/t1_val_results.csv`, `report/figures/t1_val_representative_12.png`.

## 7. Test results (final evaluation, 3,669 images x 10 conditions = 36,690 cases)

Fixed severities from the assignment: salt-and-pepper p 0.03 / 0.08 / 0.15, blur (kernel, sigma) (3, 0.7) / (5, 1.5) / (7, 2.5), occlusion 1 / 2 / 3 rectangles at about 10 / 20 / 35 percent.
Overall: **J 0.1330, SSIM 0.7751, PSNR 24.62 dB, MAE 0.0412** (input baseline J 0.1898).

| Condition | Severity | MAE | SSIM | PSNR (dB) | J | J of input |
|---|---|---|---|---|---|---|
| clean | none | 0.0344 | 0.8252 | 26.33 | 0.1046 | 0.0000 |
| salt_pepper | low | 0.0342 | 0.8239 | 26.37 | 0.1052 | 0.2062 |
| salt_pepper | medium | 0.0349 | 0.8171 | 26.22 | 0.1089 | 0.3504 |
| salt_pepper | high | 0.0368 | 0.8009 | 25.82 | 0.1180 | 0.4354 |
| gaussian_blur | low | 0.0333 | 0.8247 | 26.63 | 0.1043 | 0.0361 |
| gaussian_blur | medium | 0.0337 | 0.8082 | 26.46 | 0.1127 | 0.1122 |
| gaussian_blur | high | 0.0392 | 0.7310 | 24.98 | 0.1541 | 0.1744 |
| occlusion | low | 0.0434 | 0.7707 | 23.11 | 0.1364 | 0.0913 |
| occlusion | medium | 0.0528 | 0.7165 | 21.19 | 0.1682 | 0.1817 |
| occlusion | high | 0.0690 | 0.6330 | 19.05 | 0.2180 | 0.3103 |

Per condition (all severities): salt-and-pepper J 0.1107 (input 0.3307), blur J 0.1237 (input 0.1076), occlusion J 0.1742 (input 0.1944).
Files: `report/tables/t1_test_results.csv`, `artifacts/eval/task1/.../per_image.csv`, figures `report/figures/t1_test_vs_input.png`, `t1_test_representative_12.png` (12 representative cases: target | input | output | error map x4),
`t1_test_worst_4.png` (4 failure cases).

## 8. ONNX export and parity

- `models/onnx/t1_universal_ae.onnx` (opset 17, dynamic batch, input `input`, output `output`), sidecar `smoke: false`, sha256 `f75e559538948c92bf7a9d743d53a170b6d1f9ab2cf4185c4dac9f28b26fdb96`.
- Parity on 16 validation inputs: on Kaggle max-abs 4.17e-7 / mean-abs 3.94e-8; re-checked on the local laptop: max-abs 8.34e-7. Tolerance 1e-4, passed. Row in `report/tables/onnx_parity.csv`.

## 9. Promoted checkpoint

`models/checkpoints/t1_universal_ae.pt` (15,922,419 bytes), sha256 `5052e4ffef7716e87bb9a2697b1ec66cdf2695111bd33cccc9342427ab591aa9`, recorded in `models/MANIFEST.json`
(source: the final run's `ckpt_best.pt`, same hash).

## 10. Tracking, time and retries

- Weights & Biases (online), project https://wandb.ai/ahmedlaiq34/genai-a1, group `task1`. **Final training run: https://wandb.ai/ahmedlaiq34/genai-a1/runs/4wuthgxm.**
  One run per trial plus the final run (41 run URLs are listed in `artifacts/logs/v2/t1_overnight_summary.json`). Runs from the earlier local attempts are in the same project and are not part of these results.
- Kaggle stage times (the runner polls every 30 s, so small stages round up): study 5,382 s (89.7 min), final config 30 s, training 1,020 s (17.0 min), evaluation 90 s, export 30 s, promotion 30 s. Every stage needed one attempt.
- Test evaluation: run locally once; opened the test manifest once (16:07 UTC in `artifacts/test_access.log`).

## 11. Things that look odd (numbers only)

- **Best trial on several range edges.** Batch 32 (lowest choice, used by 30 of 40 trials), 64 channels (highest, 26 of 40 trials), lr 2.3e-3 (upper bound 3e-3), alpha 0.502 (lower bound 0.5). The latent size 2048 is interior (32 of 40 trials used it).
- **The study kept improving late** (best trial 33 of 40; top five include trials 29, 31, 32, 33, 39).
- **Validation J still improving at epoch 98.** The curve is nearly flat after epoch 80 (0.1249 at 80, 0.1237 at 98).
- **Worse than the input on mild damage.** Test J: low blur 0.1043 vs input 0.0361; low occlusion 0.1364 vs 0.0913; medium blur 0.1127 vs 0.1122 (equal). Clean images also score J 0.1046, not 0 (the bottleneck loses detail).
- **Test slightly harder than validation** (overall J 0.1330 vs 0.1237), mostly because the fixed high-severity occlusions and blur are harsher than the average sampled ones.
- **Failure cases** (`t1_test_worst_4.png`): all four are large occlusion boxes (three rectangles) on photos with mostly black or dark backgrounds. The boxes merge with the background and the model fills them with blurry brown/grey.
- **Study J (15 epochs) 0.1406 vs final J (best of 100 epochs) 0.1237.**

## 12. Not done

- The Task 1 workspace was not re-tested end to end in the application and in Docker with this final ONNX file (the backend was tested with the earlier smoke model; `models/onnx/` is mounted into the backend, so it should pick the new file up, unverified).
- No limited-skip-connection ablation and no larger-latent follow-up for the mild-damage weakness.
- The LaTeX report, the Stitch design evidence and the demo video.
- Tasks 2 to 4 are separate. The Task 1 model and config are now final; no further changes will be made to them.
