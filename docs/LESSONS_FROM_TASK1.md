# Lessons from Task 1: read this before running any Task 2 study

Audience: the agent that writes and runs the Task 2 code (corruption classifier, three specialist autoencoders).
Written 2026-10-04 by the Task 1 lead. Everything below was measured in this repository unless it says "not verified".
Task 1 spent about a day on trial and error. This file exists so Task 2 does not repeat it.

## 1. The short version (what to do)

1. **Do not tune the dense-vector autoencoder.** The Task 1 model that flattens the encoder output into a dense vector latent
   (`latent="dense"`, the default of `UniversalAE`) learned only an 8x8-resolution thumbnail of the image, whatever the
   bottleneck size. The specialists share that class and config (`configs/task2_specialist.yaml`: `depth: 4`, dense, bottleneck 64 to 512),
   so they will have the same problem. Use the convolutional latent (`latent: conv`, `depth: 3`) described in section 3.
2. **Before any Optuna study, run the five-minute checks in section 6** (`tools/diag_quick_checks.py`). They would have saved Task 1 about 3 hours.
3. **Apply the four study fixes in section 4** to `classifier.py` and `specialist.py` (pruner warm-up, sampler seed, stale RUNNING trials,
   failed trials counted as budget). Task 1's `tune.py` is the reference implementation.
4. **Report every restoration number next to the "do nothing" baseline** (the corrupted input scored against the clean target).
   A specialist that is worse than its own input is not restoring anything, and the report must say so (section 3.5).
5. **Windows/laptop rules** (section 7): one heavy process at a time, `num_workers=0`, temp files on D:, W&B entity `ahmedlaiq34`.

## 2. Status of Task 1 (so you know what is final and what is not)

| Item | Status |
|---|---|
| First study `t1_universal` (dense latent): 36 complete, 4 pruned, 0 failed, best val J 0.3300 (trial 39) | Finished. Results in `studies/t1_universal/`, `docs/PHASE_T1_TRAINING_REPORT.md` |
| Final training of that dense model: best val J 0.3134 (epoch 41 of 100), SSIM 0.454, PSNR 19.28 dB | Finished. Blurry outputs; kept only as evidence |
| Second study `t1_universal_v2` (conv latent, config `configs/task1_universal.yaml`) | **Not run yet.** The student runs it on a Kaggle T4 (`docs/KAGGLE_T1_STEPS.md`). Its best parameters will appear in `studies/t1_universal_v2/` and `configs/task1_final_v2.yaml` |
| The conv latent itself | Implemented (`src/genai/models/autoencoder.py`, tests in `tests/test_autoencoder.py`), prototype-tested, ONNX export verified in a dry run |

Do not assume the v2 numbers below the prototype table in section 3.3; they do not exist yet. If `studies/t1_universal_v2/` exists when you read this,
use its best trial as your starting point for the specialist architecture.

## 3. Findings, with evidence

All runs below: validation manifest only, the test set was never touched. J = 0.5*L1 + 0.5*(1-SSIM), lower is better.

### 3.1 The dense latent plateaus at an 8x8 thumbnail

- The final dense Task 1 model has val J 0.3134, SSIM 0.4538, MAE 0.0805. A **clean image replaced by its 8x8 thumbnail** scores
  J 0.3131, SSIM 0.456, L1 0.0817. The model had learned a coarse colour layout and nothing else (the sample grids show blobs).
- Reference thumbnails (`tools/diag_quick_checks.py calibrate`): 8x8 J 0.3131, 16x16 J 0.2543, 32x32 J 0.1643, 64x64 J 0.0739.
  Use this table to read any J value as "equivalent resolution".
- Clean inputs were reconstructed no better than corrupted ones (clean 0.3075, blur 0.3075, salt 0.3092, occlusion 0.3294):
  the limit was the model, not the corruption.

### 3.2 A bigger bottleneck did not help; the code is not buggy

| Experiment (same settings, 30 epochs) | Best val J | SSIM |
|---|---|---|
| dense latent 512 | 0.3145 | 0.458 |
| dense latent 2048 (4x larger) | 0.3151 | 0.457 |
| dense latent 4096 | 0.3306 at epoch 13, behind the others, 41 s/epoch (stopped) | 0.433 |

- Training for 100 instead of 30 epochs gained only 0.001 (0.3134 vs 0.3145). More epochs and more latent values were not the lever.
- **Overfit test** (`diag_quick_checks.py overfit`, 64 clean images, 600 steps): J 0.09, SSIM 0.85. The model *can* learn detail, so
  the loss, the data pipeline, the evaluation and the export are fine. A flatten + dense layer simply generalises badly from
  2,944 images (it must relearn "which region goes where" for every position).
- Lesson: when J is flat across bottleneck size and epochs, stop sweeping those and test the architecture.

### 3.3 The convolutional latent fixes it

The conv latent keeps the encoder's spatial grid and only compresses channels: a 1x1 conv from the last encoder map to C latent
channels (and back), no dense layer, no skip connections. Total latent values = C x s x s, with s = 128 / 2^depth. Input and
output stay 128x128x3. Prototype results (30 epochs, base channels 64, lr 1.68e-4, batch 32, alpha 0.86, dropout 0.03, all 4 conditions):

| Latent grid | Values (compression) | Val J | SSIM | Clean | Salt | Blur | Occlusion |
|---|---|---|---|---|---|---|---|
| dense vector (100 epochs) | 512 (96:1) | 0.313 | 0.454 | 0.308 | 0.309 | 0.308 | 0.329 |
| conv 8x8x16 (depth 4) | 1,024 (48:1) | 0.185 | 0.679 | 0.163 | 0.187 | 0.167 | 0.224 |
| conv 16x16x4 (depth 3) | 1,024 (48:1) | 0.190 | 0.673 | 0.167 | 0.190 | 0.170 | 0.232 |
| conv 8x8x64 (depth 4) | 4,096 (12:1) | 0.166 | 0.711 | 0.140 | 0.172 | 0.146 | 0.207 |
| **conv 16x16x16 (depth 3)** | **4,096 (12:1)** | **0.139** | **0.761** | **0.099** | **0.157** | **0.116** | **0.186** |

- A finer latent grid (16x16 instead of 8x8) at the same number of values is clearly better, and it has 1.3M parameters instead of 22M and trains faster
  (about 7 to 8 s/epoch locally vs about 12 s for the dense model).
- Constructor: `UniversalAE(base_channels=64, depth=3, bottleneck_dim=4096, dropout=0.03, latent="conv")`. `bottleneck_dim` must be a multiple of s*s
  (256 for depth 3). Old configs without a `latent` key still build the dense model, so nothing existing breaks.
- This still satisfies the PDF's "genuine compressed latent" (12:1 to 48:1, no skips, strided encoder with growing channels, decoder to 128x128x3).
  Say in the report that the latent is a grid, not a flat vector. Decision D40 in `docs/DECISIONS.md` has the full record.

### 3.4 The "do nothing" baseline (the number every restoration result must be compared with)

`diag_quick_checks.py baseline` (val manifest, 736 rows per condition), corrupted input scored directly against the clean target:

| Condition | Input-as-output J | SSIM |
|---|---|---|
| clean | 0.0000 | 1.000 |
| salt_pepper | 0.3417 | 0.359 |
| gaussian_blur | 0.0908 | 0.844 |
| occlusion | 0.1909 | 0.712 |

By severity label (val severities are tertiles of the sampled parameter, not the fixed test levels): blur 0.051 / 0.097 / 0.117,
occlusion 0.130 / 0.198 / 0.263, salt 0.251 / 0.360 / 0.416 (low / medium / high).

### 3.5 Consequence for Task 2: blur is hard to beat

- The best prototype (conv 16x16x16, all conditions) scored J 0.116 on blur, **worse than the 0.0908 of the blurred input itself**. It beat the input on salt-and-pepper
  (0.157 vs 0.342) and on occlusion at the average (0.186 vs 0.191, only just). Any bottleneck autoencoder loses some detail, so mild blur is hard to improve on.
- This is exactly why the PDF has an identity bypass (Task 2) and an identity branch (Task 3). A blur specialist is trained on blur only, so it may do better than the universal
  model, but this is **not verified**. Check it with the baseline before running a study, and if the blur specialist cannot beat its input, report that honestly;
  do not hide it. A larger latent (more than 4096 values) for the blur specialist is a reasonable hypothesis to test with the quick `train --cond 2` check.

### 3.6 The first Optuna attempt: what went wrong and why

| Symptom | Cause | Fix (all in `src/genai/tasks/task1/tune.py` unless noted) |
|---|---|---|
| 5 of 11 trials pruned at epochs 1 to 3, 4 of them with batch 128 | Validation J sits on a plateau (about 0.46) for the first 4 to 6 epochs, then drops. The median pruner (warm-up 1 epoch) judged trials before they left the plateau, and large batches have fewer steps per epoch so they leave it later: the pruner was selecting for early speed | `MedianPruner(n_startup_trials=8, n_warmup_steps=8)` for 15-epoch trials. **Look at a real learning curve before choosing the warm-up** |
| After a restart, trial 2 repeated trial 0's parameters | TPE seed 42 re-drew the same first sample | `TPESampler(seed=42 + number_of_existing_trials)` |
| A killed trial stayed `RUNNING` in the DB forever | Process killed mid-trial | At start-up mark every `RUNNING` trial `FAIL` (`study.tell(number, state=FAIL)`) |
| A crashed trial used up one of the 40 | Budget was `n_trials - number of finished trials` | Count only `COMPLETE` and `PRUNED` as answered, so failed trials are retried |

Second-order finding: with the fixed pruner (warm-up 8, 15 epochs per trial) the dense study had 36 complete, 4 pruned, 0 failed, and its best trial was the *last* trial
(the study was still improving), with the best parameters on several range edges (batch 32, bottleneck 512, 64 channels). When the best value sits on a range edge, say so in the report.

Specialist-specific note: `specialist.py` reports the *running mean after each completed corruption* (steps 1, 2, 3), not epochs, so the epoch plateau problem does not apply
in the same way. The classifier reports per epoch, so it can: check its learning curve (macro-F1 near 0.25 is chance for 4 balanced classes) before choosing `n_warmup_steps`.
The seed, RUNNING and failed-trial fixes apply to both.

### 3.7 Infrastructure problems on this Windows laptop (RTX 3050 6 GB, 15.6 GB RAM)

- **DataLoader workers crash under memory pressure** (`WinError 1114` loading `torch\lib\shm.dll`), which hung a trial. Each Windows worker re-imports torch (about 0.5 GB).
  Use `num_workers=0`. Throughput was not a problem: an epoch of the conv model is about 8 s.
- **C: was 98 to 100 percent full** (the Windows pagefile lives there). Writing 3 GB of pytest temp folders to C: filled it and broke test runs ("No space left on device").
  Run tests with `--basetemp=artifacts/pytest_tmp/<name>` (create `artifacts/pytest_tmp` first) and `TMP`/`TEMP` pointing at `artifacts/tmp`.
- **Run one heavy process at a time.** Two training/pytest processes at once exhausted RAM twice.
- **CPU-only tests:** Task 2's tests assume the GPU is hidden (`CUDA_VISIBLE_DEVICES=-1`, decision D31); without it `tests/test_routing.py` fails with a device mismatch
  (2 tests). That is an environment issue, not a bug in the code.
- **W&B:** the entity is **`ahmedlaiq34`** (the username `ahmedlaiq` is rejected). Set `TRACKER=wandb WANDB_ENTITY=ahmedlaiq34 WANDB_PROJECT=genai-a1`.
- **`common.checkpoint.promote()`** crashed on the real `models/MANIFEST.json` because its layout is `{"version": 1, "models": []}`, not a list (fixed, D39).
  A dry run that uses a temp `models/` folder will not catch layout problems of the real file; check the real one once.
- **Background shell tools:** a single foreground command cannot wait more than 10 minutes; long jobs go in the background with a log file.

### 3.8 Timing reference (local RTX 3050)

Dense UniversalAE (32 to 64 channels): about 12 s/epoch, a 40-trial x 15-epoch study took 2 h, a 100-epoch final training 24 min.
Conv UniversalAE (depth 3, 64 channels): about 7 to 8 s/epoch at batch 32. The validation pass (2,944 rows) is about 2.5 to 3.5 s.
Specialist trials train three models each, so budget three times the single-model cost per trial (not measured for Task 2).

## 4. Changes to make in Task 2 before running a study

Find these by searching for the names; line numbers move. Task 1's `src/genai/tasks/task1/tune.py` (`run_study`) is the reference.

| Where | Change |
|---|---|
| `src/genai/tasks/task2/specialist.py`, `create_study(...)` (about line 378) and `classifier.py` (about line 515) | Sampler seed: `int(cfg["sampler"]["seed"]) + n_existing`, where `n_existing = len(optuna.load_study(...).trials)` (0 if the study does not exist yet, catch `KeyError`) |
| both, right after `create_study` | Mark any trial in state `RUNNING` as `FAIL` with `study.tell(t.number, state=optuna.trial.TrialState.FAIL)` |
| both, the line `already = len([t for t in study.trials if t.state.is_finished()])` (specialist about 379, classifier about 542) | Count only `COMPLETE` and `PRUNED`: `already = len([t for t in study.trials if t.state in (COMPLETE, PRUNED)])` |
| `configs/task2_classifier.yaml` pruner | Do not keep `n_warmup_steps: 1` blindly. Run one full classifier training, look at macro-F1 per epoch, and put the warm-up after the plateau |
| `configs/task2_specialist.yaml` `model:` | `latent: conv`, `depth: 3`, `bottleneck_dim: 4096`, `base_channels: 64` (provisional; the study tunes them) |
| same file, `tuned_params` | `bottleneck` choices become total latent values that are multiples of 256, e.g. `[1024, 2048, 4096]` (the old `[64, 128, 256, 512]` was for the dense vector and fails the multiple check). Add 8192 if the blur check in section 3.5 suggests it |
| same file, `depth` | Keep `depth` fixed (3) as in Task 1; it is not one of the PDF's required tuned parameters |
| the evaluation tables (`task2/evaluation.py` or wherever the routing tables are built) | Add the baseline columns `J_input` and `SSIM_input`, as `task1/evaluate.py` now does (look at `_per_image_metrics`) |

Everything else in `UniversalAE` is unchanged: `UniversalAE.from_config(cfg["model"])` already accepts the new `latent` key,
`routing.py` rebuilds specialists through it, and the ONNX export works for the conv latent (verified in the Task 1 dry run: max abs difference 6e-8).
Existing Task 2 checkpoints built with the dense default still load.

## 5. Decisions already taken (do not reopen without a reason)

| ID | Decision |
|---|---|
| D19 | Study robustness fixes (pruner warm-up, sampler seed, RUNNING to FAIL, budget counts answered trials), `num_workers=0`, TMP on D:, runner `tools/t1_overnight.py` |
| D39 | `promote()` accepts both MANIFEST layouts |
| D40 | Conv latent for Task 1 v2, evidence in section 3 |
| D41 | Task 1 training runs on Kaggle (code and data as private datasets) |
| D8 (plan) | Studies are scored on the fixed J = 0.5*L1 + 0.5*(1-SSIM), independent of alpha/lambda; classifier on macro-F1 |

Take the next free decision number in `docs/DECISIONS.md` for yours (append only, re-read the file right before appending).

## 6. Cheap validation protocol: do this before any study (about 15 minutes in total)

`tools/diag_quick_checks.py` (validation data only). Run one at a time, in the background with a log file if it takes more than a minute.

```
python tools/diag_quick_checks.py baseline                 # the number to beat, per corruption
python tools/diag_quick_checks.py calibrate                # what a given J means in resolution terms
python tools/diag_quick_checks.py overfit --latent conv --depth 3 --latent-values 4096   # can the model learn at all? measured for the conv model: J 0.142 / 0.112 / 0.095 at steps 200 / 400 / 600 (about 100 s)
python tools/diag_quick_checks.py train --latent conv --depth 3 --latent-values 4096 --epochs 30 --cond 1   # specialist-style: salt (1), blur (2), occlusion (3)
```

Procedure that worked for Task 1: (a) compare J with the calibration table, (b) run the overfit test, (c) change ONE thing per 30-epoch run
(latent size, then latent type, then grid size) and compare with an identical baseline run, (d) only then write the study ranges.
If J is identical across a parameter, that parameter is not your lever; stop sweeping it.

For the classifier there is no autoencoder, but the same logic holds: check the chance level (macro-F1 0.25), overfit a small batch, and look at one full
learning curve before choosing the pruner warm-up. Not verified for the classifier; this file's measurements are about the autoencoder.

## 7. Operating rules (carry over from Task 1)

- The official test set stays locked: no `--final-test` until the student approves, and nothing in training or tuning may read it.
- One heavy process at a time; long jobs in the background, output redirected to `artifacts/logs/`, polled every few minutes.
- Do not edit Task 1 files (`src/genai/tasks/task1/*`, `configs/task1_*`) or `docs/CONTRACTS.md` / the plan. Needed changes go in `docs/DECISIONS.md`.
- Resume must work: Optuna SQLite with `load_if_exists=True`, atomic checkpoints, `resume` from the same run folder.
- Do not commit to git unless the student asks.
- `tools/t1_overnight.py` is a supervisor for the Task 1 pipeline (restarts crashed stages, resumes, kills hung ones). It is Task 1 specific, but its structure
  (stages as child processes, state file, status log, stall detection) is a good template if Task 2 needs an unattended run.
