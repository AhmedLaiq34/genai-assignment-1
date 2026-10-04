# Phase report: Task 2 (corruption classifier, three specialists, hard-routed restoration)

Written by the lead session, 2026-10-04. Everything below was run unless marked otherwise. **No real training, no real Optuna study, no test-set evaluation, no GPU use was intended** (CPU-only mode, D20). Nothing was committed to git.

## 1. Status in one paragraph

All items of `docs/WORKER_PROMPT_T2.md` are built and verified on CPU, except the GPU benchmark (skipped on purpose, see section 8) and the Docker Compose check with the new endpoint. The full test suite gives **176 passed, 1 failed**; the one failure is a Task 1 test that broke because of the approved Task 1 budgets (D37), not because of Task 2. What is left is on your side: run the benchmark on the target GPU, approve budgets, then train (runbook in section 6).

## 2. Verification (all CPU, `CUDA_VISIBLE_DEVICES=-1`)

| What | Command (abridged) | Result |
|---|---|---|
| Full test suite | `pytest tests -q -p no:cacheprovider --basetemp=artifacts/pytest_tmp/run` | **176 passed, 1 failed** in 360 s. Baseline was 98; 79 tests are new: classifier 17, specialists 19, routing + ONNX 32, backend `/api/hard` 11 |
| The one failure | `tests/test_task1_pipeline.py::test_study_dry_run` | Not a Task 2 change: `configs/task1_universal.yaml` now holds the approved budgets (D17), so the test's "TBD must raise" check never fires and the test ran a 40-trial study (D37). It passed in the earlier phase when the config still said TBD |
| Classifier smoke | `scripts/train.py --task t2cls` with `max_steps=20`, 256 train images, 64 val rows, channels `[8,16,32]` | ran 2 epochs, wrote `artifacts/dryrun/runs/task2_classifier/20261004-0233_local_t2cls_smoke`. val acc 0.25, macro-F1 0.10: 20 steps, meaningless |
| Specialist smoke, three corruptions | `scripts/train.py --task t2spec --set specialist.corruption=salt` (then `blur`, `occlusion`), same limits | all three ran and wrote their own folders `task2_specialist_<corruption>/<run_id>_smoke`. val J about 0.460 each (20 steps, meaningless) |
| Classifier study dry run | `scripts/tune.py --task t2cls --dry-run --n-trials 2 --epochs 1` | `t2_classifier_dryrun`: 2 complete / 2 total; DB, `trials.csv` and 3 plots under `artifacts/dryrun/` |
| Specialist shared study dry run | `scripts/tune.py --task t2spec --dry-run --n-trials 2 --epochs 1` | `t2_specialist_shared_dryrun`: 2 complete / 2 total; `trials.csv` has `user_attrs_J_salt/blur/occlusion` for both trials; best mean J 0.46028 (meaningless: 1 epoch on a subset) |
| Real studies untouched | `git status studies artifacts/optuna` | no change in `studies/` or `artifacts/optuna/` |
| Four-checkpoint evaluation (smoke checkpoints, val, 256 rows) | `scripts/evaluate.py --task t2cls --set eval.checkpoints.<name>=<path> ... --set eval.max_rows=256` | oracle J 0.3450, predicted J 0.4599, classifier accuracy 0.250, 192 misroutes of 256. The classifier is untrained, so it sends nearly everything to one expert: this shows the failure analysis runs, not model quality |
| Classifier alone | `scripts/evaluate.py --task t2cls --ckpt <smoke ckpt>` | 2944 val rows, accuracy 0.2514, macro-F1 0.1054 |
| ONNX export + parity (smoke checkpoints) | `scripts/export_onnx.py --model t2_classifier\|t2_salt\|t2_blur\|t2_occlusion --ckpt ... --out-dir artifacts/dryrun/onnx --verify` | 16 real val inputs (4 per condition). Max-abs difference 1.2e-07 (classifier) and 5.96e-08 (each specialist), tolerance 1e-4, all passed. Routing parity: 16 of 16 agree (trivial: the random classifier predicted one class for all). Four rows tagged `smoke` appended to `report/tables/onnx_parity.csv` |
| Test set stayed locked | `artifacts/test_access.log` | does not exist |
| Smoke ONNX not in `models/onnx/` | `ls models/onnx` | only the Task 1 files |
| Frontend | `npm run build` | built, 37 modules, JS 152.85 kB |
| Backend | `tests/test_backend_hard.py` + `tests/test_backend.py` (inside the full suite) | pass. The mock server returned the right `/api/hard` shape for identity bypass and for experts (Agent S's curl run) |
| Benchmark script | `py_compile`; `scripts/benchmark.py --model t2cls --batch 64` with the GPU hidden | compiles; stops with "benchmark needs a CUDA GPU". The `t2cls` and `t2spec` measuring code itself has **not been run** |

Run folders and fixtures from these runs are in `artifacts/dryrun/` (1.5 GB) and `artifacts/fixtures/task2/` (gitignored).

## 3. Files created or changed

| Owner | Files |
|---|---|
| Lead | `src/genai/tasks/task2/{__init__,runs,train,tune,evaluate}.py`, `tests/conftest.py`, `scripts/benchmark.py` (t2cls, t2spec), `scripts/evaluate.py` (four-checkpoint mode), `scripts/export_onnx.py` (Task 2 models + routing parity), `notebooks/{kaggle,colab}_train.ipynb`, one-line edit of `tests/test_backend.py` (D25), `docs/DECISIONS.md` (D20 to D24, D35 to D38), `docs/AI_USE_LOG.md`, this report |
| P | `src/genai/models/classifier.py`, `src/genai/tasks/task2/classifier.py`, `configs/task2_classifier.yaml`, `tests/test_classifier.py` |
| Q | `src/genai/tasks/task2/specialist.py`, `configs/task2_specialist.yaml`, `tests/test_specialist.py` |
| R | `src/genai/tasks/task2/routing.py`, `evaluation.py`, `src/genai/export/task2_export.py`, `tests/test_routing.py`, `tests/test_onnx_t2.py`, `tests/t2_fixtures.py` |
| S | `app/backend/app/hard.py`, `app/backend/app/main.py` (shared input helper, `/api/hard`), `app/frontend/src/{App.jsx,api.js,useRun.js}`, `components/{InputControls,HardWorkspace}.jsx` (+ edits to `UniversalWorkspace.jsx`, `Stubs.jsx`), `app/frontend/mock/server.mjs`, `tests/test_backend_hard.py` |

Not edited (as required): Task 1 files, `src/genai/common/**`, `src/genai/pets/**`, `autoencoder.py`, `onnx_export.py`, `onnx_verify.py`, `CONTRACTS.md`, `IMPLEMENTATION_PLAN.md`. (`task1/tune.py` and `configs/task1_universal.yaml` show as modified: that is the Task 1 window.) `app/frontend/README.md` still says only Universal works.

## 4. What Task 3 can rely on

**Names.** Class order `0 clean, 1 salt_pepper, 2 gaussian_blur, 3 occlusion`. Specialist short names `salt` (cond 1), `blur` (2), `occlusion` (3). Promoted names `t2_classifier`, `t2_ae_salt`, `t2_ae_blur`, `t2_ae_occlusion`. Constants in `genai.tasks.task2` (`SPECIALIST_COND_ID`, `PROMOTED_NAMES`, `ONNX_KEYS`, `EXPERT_FOR_CLASS`).

**Checkpoint dict** (same contract as Task 1, plus `step_in_epoch`, `epoch_loss_sum`, `rng`; training-only keys). `ckpt["config"]` is the full config:

- classifier: `component: classifier`, `model: {channels, dropout, num_classes}`, `run_id`. Rebuild: `CorruptionClassifier.from_config(ckpt["config"]["model"])`. Forward gives **logits** (Task 3 divides by tau), input `[0,1]`, normalisation `x*2-1` is inside `forward`. `best_metric` = val macro-F1.
- specialist: `component: specialist`, `specialist: {corruption, cond_id}`, `model: {in_ch, base_channels, depth, bottleneck_dim, dropout}`, `run_id`. Rebuild: `UniversalAE.from_config(ckpt["config"]["model"])`. `best_metric` = val J on its own condition.

**Loaders.** `genai.tasks.task2.routing.load_task2_models(paths: dict) -> {"classifier", "salt", "blur", "occlusion"}` (eval-mode models; raises `ValueError` if a checkpoint is the wrong component or specialist). Also `load_component`, `check_checkpoint`, `describe_checkpoints`, `HardRoutedSystem(models, device)` with `classify`, `restore_by_route`, `route(x, true_cond, mode)`. `genai.tasks.task2.classifier.load_classifier(ckpt_path)`, `classification_metrics(y_true, y_pred)`. The sha256 check is `genai.common.checkpoint.sha256_file` against `models/MANIFEST.json`.

**Evaluation output** (`<output_root>/eval/task2/<timestamp>_<val|test>/`): `per_image.csv` (long format, `mode` = oracle or predicted, columns `image_id, cond, severity, true_class, predicted_class, oracle_route, predicted_route, mode, route_used, MAE, SSIM, PSNR, J, p_*`), `table_cond_severity_{oracle,predicted}.csv`, `classifier_report.json`, `confusion_normalised.{csv,png}`, `routing_failures.csv`, `misroute_confusion.csv`, `routing_failures_worst.png`, `summary.json`.

**ONNX** (opset 17, dynamic batch, legacy exporter): `t2_classifier.onnx` (`input` -> `logits`), `t2_ae_{salt,blur,occlusion}.onnx` (`input` -> `output`), each with a `.meta.json` sidecar (`smoke` flag, sha256). Export refuses to put a smoke checkpoint's ONNX under `models/onnx/` (D32).

## 5. Behaviour worth knowing

- **Classifier:** balanced batches (exact B/4 per class; batch size must be a multiple of 4, else `ValueError`), AdamW + cross-entropy, best checkpoint by val macro-F1. Study `t2_classifier` tunes lr, batch, conv channels (as names like `"16-32-64"`), dropout, weight decay.
- **Specialists:** `PetsTrainDataset(policy="fixed:k")`, loss `alpha*L1 + (1-alpha)*(1-SSIM)`, validation only on its own condition's val rows, selection by fixed J. The corruption must be given: `--set specialist.corruption=salt|blur|occlusion` (missing or invalid gives a clear error).
- **Shared specialist study:** one trial = **three** short trainings (salt, blur, occlusion) with the same sampled values; value = mean of the three best J; `J_salt/J_blur/J_occlusion` stored as trial attributes; pruning after each corruption (running mean, steps 1 to 3). Tunes lr, bottleneck, channels, batch, L1/SSIM weight.
- **Trials never feed a final run:** trial runs are written outside the real run folders and their checkpoints deleted (D27, D29), so `RESUME="auto"` cannot pick up a trial checkpoint.
- **Hard routing:** clean means identity bypass and a specialist is never called (tested). Oracle mode routes by the manifest's `cond_id`, predicted mode by the classifier argmax. Predicted-mode output reuses the oracle output where both routes agree.
- **Notebooks:** `MODE="benchmark"` now uses `TASK`; `MODE="final_config"` writes `configs/task2_specialist_final.yaml`; the override notes say `None` must be written `"null"`.

## 6. Runbook for Kaggle / Colab (nothing below was run by me)

**Data:** attach the existing `dist/cloud_data/pets_data.zip` as a private dataset (same upload as for Task 1). Push the code to GitHub first; fill `REPO_URL` / `BRANCH` in the notebook.

**Order of runs**
1. **Benchmark** (first, on the target GPU): notebook `TASK="t2cls"`, `MODE="benchmark"`; then `TASK="t2spec"`. Copy the new rows of `docs/BENCHMARKS.md` back.
2. **Classifier:** `TASK="t2cls"`, `CONFIG="configs/task2_classifier.yaml"`, `MODE="tune"` (study), then `MODE="train"` with the study's best values written into the config (edit `model`/`train` by hand or via `OVERRIDES`, and set `train.epochs`).
3. **Specialists:** `TASK="t2spec"`, `MODE="tune"` (shared study); then `MODE="final_config"` (writes `configs/task2_specialist_final.yaml`; set its `train.epochs`); then three finals with `CONFIG="configs/task2_specialist_final.yaml"` and `OVERRIDES={"specialist.corruption": "salt"}`, then `"blur"`, then `"occlusion"`.
4. Fill `n_trials`, `epochs_per_trial`, `train.epochs` (currently `TBD_AFTER_BENCHMARK`) before steps 2 and 3, or pass `--n-trials` / `--epochs` and `train.max_steps`.

**Parallelism:** the classifier and the specialists are independent and can run on different devices; the three specialist finals can run on three devices (each its own run folder). Only Task 3 needs all four.

**What to download and where to put it**
- `artifacts/runs/task2_classifier/<run_id>/` and `artifacts/runs/task2_specialist_<corruption>/<run_id>/` (use `ckpt_best.pt`, `metrics.jsonl`, `config.yaml`, `samples/`, the confusion files) into the same paths locally.
- `artifacts/optuna/t2_classifier.db` and `artifacts/optuna/t2_specialist_shared.db` locally under `artifacts/optuna/`; then copy finished studies' exports to `studies/t2_classifier/` and `studies/t2_specialist_shared/` (the study writes them to `study_dir`).
- `configs/task2_specialist_final.yaml` (commit it).

**Promote and record hashes** (locally, one line each):
```
python -c "from genai.common.checkpoint import promote; promote('artifacts/runs/task2_classifier/<run_id>/ckpt_best.pt', 't2_classifier')"
python -c "from genai.common.checkpoint import promote; promote('artifacts/runs/task2_specialist_salt/<run_id>/ckpt_best.pt', 't2_ae_salt')"
```
(same for `t2_ae_blur`, `t2_ae_occlusion`). `promote` copies to `models/checkpoints/<name>.pt` and writes the sha256 into `models/MANIFEST.json`. Task 2 final checkpoints are immutable after this.

**Then:** `python scripts/evaluate.py --task t2cls --set eval.checkpoints.classifier=models/checkpoints/t2_classifier.pt --set eval.checkpoints.salt=... --set eval.checkpoints.blur=... --set eval.checkpoints.occlusion=...` (val only), and `python scripts/export_onnx.py --model t2_classifier --ckpt models/checkpoints/t2_classifier.pt --verify` for each of the four models.

## 7. Budgets

No budget is proposed because **no GPU benchmark was possible**. Procedure: run the benchmark mode for `t2cls` and `t2spec` on the target GPU, take the measured seconds per step, work out seconds per epoch (steps per epoch = images / batch size; the classifier and specialists see 2,944 train images per epoch; add the measured val pass), then `trials ≈ allotted GPU time / measured seconds per trial`. Remember that **a specialist-study trial costs three trainings** (so its trials cost about 3 x epochs_per_trial epochs), while a classifier trial costs one. Write the approved numbers into `n_trials`, `epochs_per_trial` and `train.epochs` and record them as a DECISIONS row, as was done for Task 1 (D17).

## 8. Deviations, refactor candidates, open questions

- **CPU-only spelling (D31, D35):** `CUDA_VISIBLE_DEVICES=""` (written in the worker prompt) does not hide the GPU on this laptop; `-1` does. The first runs of agents P, Q, R and S used `""`, so they may have briefly used the GPU while Task 1 trains. My own runs all used `-1`.
- **C: drive full (D36):** C: hit 0 bytes free during integration (a 17 GB pagefile under memory pressure plus 3 GB of pytest temp folders). I deleted my own pytest temp folders and moved test temp to D:. Two Task 1 test runs failed with "No space left on device" before that. Worth checking, because the Task 1 training window shares this machine.
- **Task 1 test failure (D37):** see section 2; the fix belongs to the Task 1 window.
- **Trial-run folders (D27, D29):** differ from the prompt's "run directories" for trials, on purpose; accepted.
- **Duplication (D22, D38):** Task 2 loops copy `task1/train.py`; refactor candidate after Task 1 training. The Task 2 studies also lack Task 1's later robustness fixes (sampler seed + existing trials, marking RUNNING trials FAIL at restart).
- **Smoke ONNX guard (D32):** Task 2 export refuses to write smoke ONNX into `models/onnx/`; the app therefore cannot be demoed with a smoke Task 2 model there unless you relax one `if` in `task2_export.py`.
- **Routing parity check** is returned as a dict and printed, not written to the parity CSV (D34).
- **Specialist sample grids** use the first 12 val rows of its own corruption (D30).
- Resume is not bit-identical to an uninterrupted run (same as Task 1).

## 9. Not done

- No real training, no real Optuna study (only 2-trial 1-epoch dry runs under `artifacts/dryrun/`), no test-set evaluation (`final_test=True` implemented, only exercised on synthetic data in a test with a redirected access log).
- No GPU benchmark for `t2cls` / `t2spec` (the laptop was busy with Task 1 training); the benchmark code was compiled but never run, so expect to debug it on the first GPU run.
- No Docker Compose check with `/api/hard`; the page was only built and tested against the mock server and unit tests, never opened in a browser.
- Tasks 3 and 4 not started. The frontend is not derived from Stitch.
- Nothing pushed to GitHub or uploaded to Kaggle.

## 10. Follow-up after `docs/LESSONS_FROM_TASK1.md` (2026-10-04, GPU and CPU were free; no study, final training or test-set evaluation was run)

Logs of every number below: `artifacts/logs/t2_diag_*.log`, `t2_blur_real_pipeline.log`, `t2_salt_occ_pipeline.log`. Val manifest only. Decisions D42 and D43.

### 10.1 The quick checks (section 6 of the lessons file)

| Check | Result |
|---|---|
| `baseline` (corrupted input as output, J / SSIM) | clean 0.0000 / 1.000; salt 0.3417 / 0.359; blur 0.0908 / 0.844; occlusion 0.1909 / 0.712. By severity (I recomputed it): blur 0.0505 / 0.097 / 0.1173, occlusion 0.1304 / 0.1978 / 0.2629, salt 0.2508 / 0.3595 / 0.4157 (low / medium / high). Same as the lessons file |
| `calibrate` (clean image replaced by a thumbnail) | 8x8 J 0.3131, 16x16 0.2543, 32x32 0.1643, 64x64 0.0739 |
| `overfit` conv 4096 values, 64 images | J 0.1417 / 0.1125 / 0.0954 at steps 200 / 400 / 600 (SSIM 0.757 / 0.812 / 0.840). The model can learn; pipeline fine |
| `train --cond 1/2/3`, conv depth 3, 4096 values, 30 epochs, J on the specialist's OWN corruption | salt **0.1183** (input 0.3417), blur **0.1009** (input 0.0908), occlusion **0.1723** (input 0.1909). The real pipeline (`train.py --task t2spec`, 30 epochs) gave the same values: 0.1183, 0.1009, 0.1727 |

Note: `diag_quick_checks.py train` prints J for all four conditions; the number that counts for a specialist is the one on its own corruption (the other lines, e.g. the salt specialist scored on blur, 0.126, are not its job).

### 10.2 Does the blur specialist beat its own input? No (section 3.5 of the lessons file)

| Blur specialist (30 epochs) | Val J | Input J (do nothing) |
|---|---|---|
| all blur rows, 4096 latent values | 0.1009 | 0.0908 |
| 8192 latent values | 0.0983 | 0.0908 |
| 16384 latent values (3:1 compression) | 0.0962 | 0.0908 |
| by severity (4096, real pipeline) | high 0.1089, medium 0.0967, low 0.0956 | high 0.1173, medium 0.0970, low 0.0505 |

It helps only for strong blur, ties at medium blur and is much worse at low blur. A larger latent narrows the gap but does not close it. A likely reason (consistent with the numbers, not separately proven): even clean images come back at J about 0.099 from this no-skip autoencoder, which is already above the 0.0908 of the blurred input. Salt (0.118 vs 0.342) and occlusion (0.173 vs 0.191) do beat their inputs. This is recorded as D43 with the options (accept and report, or a residual / limited-skip blur specialist, which the PDF's "same basic architecture" wording may not allow); I applied neither and kept the PDF design (blur goes to the blur specialist).

### 10.3 Hard-routed system with these 30-epoch diagnostic models (smoke-tagged, `artifacts/dryrun/`, NOT final models)

Classifier: 20 epochs, default config (val accuracy 0.984, macro-F1 0.9837 in training; 0.985 / 0.985 in the evaluation). Evaluation on all 2944 val rows:

| | J | SSIM |
|---|---|---|
| input baseline (no restoration, clean routed to identity) | 0.1558 | 0.729 |
| oracle routing | 0.0980 | 0.833 |
| predicted routing | 0.0985 | 0.832 |

44 misroutes of 2944 (1.5 percent), mean J drop on a misroute 0.0385. Per condition oracle J vs input J: salt 0.118 vs 0.342, occlusion 0.173 vs 0.191, **blur 0.101 vs 0.091 (worse)**; clean is the identity bypass. In predicted mode clean images lose a little (J 0.004) when the classifier sends some of them to an expert. The baseline columns `SSIM_input` / `J_input` are in `per_image.csv`, both condition x severity tables, `routing_failures.csv` and `summary.json`.

### 10.4 Learning curve for the classifier pruner

macro-F1 per epoch: 0.69, 0.85, 0.93, 0.80, 0.96, 0.96, 0.97, 0.96, 0.97, 0.92, 0.96, 0.98, 0.98, 0.98 (epoch 14), 0.982 (15), 0.982, 0.982, 0.985, 0.984, 0.984 (20). There is no chance-level plateau, but the first epochs rise and wobble (dips at epochs 4 and 10), so the classifier pruner starts at epoch 5 (`n_warmup_steps: 5`, `n_startup_trials: 8`). The specialist study keeps corruptions as pruner steps (1, 2, 3); warm-up 2, so nothing is pruned on the salt training alone.

### 10.5 Code and config changes (D42)

- `configs/task2_specialist.yaml`: `latent: conv`, `depth: 3`, `base_channels: 64`, `bottleneck_dim: 4096`, `dropout: 0.03`, lr 1.7e-4, batch 32, alpha 0.86 (all PROVISIONAL), `bottleneck` choices `[1024, 2048, 4096]`, pruner as above. Depth stays fixed. 8192 was not added to the choices: it improved blur only slightly (0.0983) and does not change the finding.
- `configs/task2_classifier.yaml`: pruner as above.
- `src/genai/tasks/task2/runs.py`: `open_study` (sampler seed + existing trials, RUNNING trials marked FAIL) and `n_answered` (COMPLETE and PRUNED only); both studies use them (`classifier.py`, `specialist.py`).
- `src/genai/tasks/task2/evaluation.py`: input-baseline columns and the `oracle_vs_input_by_condition` block in `summary.json`.
- Tests: the specialist tests pin a tiny dense model like Task 1's, plus new tests for the conv-latent config and training, the study helpers, and the baseline columns. **Full suite: 186 passed** (CPU, `CUDA_VISIBLE_DEVICES=-1`), including the Task 1 test that failed earlier (the Task 1 window fixed it).

### 10.6 Still not done / waiting for the student

No Optuna study, no final training, no test-set evaluation, no budgets proposed (the benchmark on the target GPU is still to be run; a specialist trial still costs three trainings). A specialist-study dry run with the new conv config was not repeated (only the unit tests exercise it) because the instruction was to run no study. The 30-epoch runs above took about 8 s per epoch on the local RTX 3050 at batch 32 (about 4 min for one specialist), which is a measurement for budgeting: one specialist-study trial of E epochs per corruption would cost about 3 x E x 8 s plus validation.
