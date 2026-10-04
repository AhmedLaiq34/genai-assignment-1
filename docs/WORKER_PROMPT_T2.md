# Worker prompt: Task 2 (corruption classifier + three specialists + hard-routed restoration)

Open a NEW Claude Code session in this folder with Sonnet 5.5 selected and send:
"Read `docs/WORKER_PROMPT_T2.md` and execute everything below its `---` line as the lead. Read every file it tells you to read before starting."

---

You are the lead of a team of Sonnet coding agents implementing **Task 2** for a university Generative AI assignment. Task 2 consists of:
- the corruption classifier;
- three specialist autoencoders;
- hard-routed restoration with oracle and predicted modes;
- four ONNX models;
- `POST /api/hard`;
- the Hard-Routed Restoration workspace.

The student must understand every line, so write simple, well-commented code that reads like the existing Task 1 code. No money may be spent. **Spawn every worker agent with model `sonnet`.**

## 0. Before anything else: ask the student one question
"Is Task 1 training (another Claude window) still running on this laptop?"
- **If yes, or no answer:** CPU-only mode for the whole phase.
  - Prefix every Python command with `CUDA_VISIBLE_DEVICES=""` and use `num_workers=0`.
  - Do not start Docker.
  - Do not run anything that needs more than about 2 GB RAM.
  - Do not touch `artifacts/runs/task1*`, `artifacts/optuna/t1_universal.db`, `configs/task1_final.yaml` or `studies/t1_universal/`.
- **If no:** the GPU may be used for the smoke runs and the benchmark in §6, and Docker for the Compose check.

## 1. Read first (all agents)
1. **`docs/IMPLEMENTATION_PLAN.md`.** §3 (contracts) and §10 (workflow, notebooks, resilience) are binding. Especially §3.2, 3.5, 3.6, 3.8, 3.9, 3.10, 3.13, P3 (Task 2), P4 (what Task 3 will need), P5.
2. **`docs/WORKER_PROMPT_T0_T1.md`.** Its "Ground rules" and "Workflow, notebooks and resilience" sections apply unchanged:
   - the test set is locked;
   - no long training or Optuna studies;
   - no invented numbers;
   - no git commits;
   - report only what you ran;
   - append rows to `docs/AI_USE_LOG.md`.
3. **Status documents:** `docs/PHASE_T0_T1_REPORT.md`, `docs/DECISIONS.md` (D1 to D15), and `docs/CONTRACTS.md`.
4. **Assignment text:** `ASSIGNMENT_TRANSCRIPTION.md`, PDF pages 4 and 5 (Task 2) and page 8 (application).
5. **The existing code, which is the pattern to follow.** Read these before writing anything:
   - `src/genai/tasks/cli.py`. The task keys **`t2cls` and `t2spec` already exist**. Both resolve to `genai.tasks.task2.{train,tune,evaluate}`.
   - `scripts/train.py`, `tune.py`, `evaluate.py`, `benchmark.py`, `export_onnx.py`. These accept `--task t1|t2cls|t2spec|...` and `--set KEY=VALUE` overrides.
   - `src/genai/tasks/task1/config.py`, for its generic helpers: `load_config`, `make_run_id`, `runs_dir`, `find_latest_checkpoint`, `is_tbd`, `set_dotted`.
   - `src/genai/tasks/task1/train.py`, `tune.py`, `evaluate.py`. This is the loop, resume, checkpoint, tracker and study pattern.
   - `src/genai/models/autoencoder.py` (`UniversalAE`, `UniversalAE.from_config`).
   - `src/genai/pets/samplers.py`: policies `iid_uniform`, `balanced_batch`, `fixed:k`, plus `make_batch_sampler` and `BalancedBatchSampler`.
   - `src/genai/pets/dataset.py`: `PetsTrainDataset(split, policy, data_root, seed)`, `PetsManifestDataset(manifest_path, split, final_test, data_root)`, `resolve_data_paths`.
   - `src/genai/common/checkpoint.py` (`build_checkpoint`, `save_checkpoint`, `load_checkpoint`, `promote`), `metrics.py` (`evaluate_restoration`), `tracking.py` (`init_run(cfg, run_id, project, group)`), `constants.py` (`ONNX_FILES` keys `t2_classifier`, `t2_salt`, `t2_blur`, `t2_occlusion`).
   - `src/genai/export/onnx_export.py`, `onnx_verify.py`.
   - `configs/task1_universal.yaml` and the existing **structure-only** `configs/task2_classifier.yaml` and `configs/task2_specialist.yaml`.
   - `notebooks/kaggle_train.ipynb`, `notebooks/colab_train.ipynb`.
   - `app/backend/app/main.py`, `app/backend/app/preprocess.py`, `app/frontend/src/**`, `app/frontend/mock/**`.

## 2. Hard constraints
- **Do not modify Task 1 or shared-library files:**
  - `src/genai/models/autoencoder.py`, `src/genai/tasks/task1/**`, `configs/task1_*`;
  - `src/genai/common/**`, `src/genai/pets/**`;
  - `src/genai/export/onnx_export.py`, `onnx_verify.py`.

  Import from them. If a change there is genuinely needed, add a `docs/DECISIONS.md` row with status "question" and tell the lead. Do not work around it by editing.
- **Duplication is acceptable where Task 1 hard-codes its own choices.** For example, `task1/train.py` fixes policy `iid_uniform` and tracker group `task1`. Task 2 may write its own training loop modelled on `task1/train.py`, importing every helper it can (config helpers, `restoration_loss`, checkpoint and tracking functions, `build_val_loader` if suitable). Record the duplication as a DECISIONS row: "candidate refactor after Task 1 training finishes".
- **Shared log files:** the Task 1 training window may append to `docs/DECISIONS.md`, `docs/AI_USE_LOG.md` and `docs/BENCHMARKS.md` at the same time. Number your DECISIONS rows **D20 and up**. Re-read each of these files immediately before appending, and only ever append.
- **Each agent writes only the files it owns (§3).** Do not edit `docs/CONTRACTS.md` or `docs/IMPLEMENTATION_PLAN.md`.
- **The test set stays locked.** Implement `final_test=True`, but never run it.
- **No real training and no real Optuna study.** Allowed compute:
  - unit tests;
  - smoke runs of tens of steps on a tiny subset;
  - a 2-trial, 1-epoch dry run per study on a tiny subset, writing only under `artifacts/dryrun/`;
  - the §6 benchmark, only if the GPU is free.

  The real runs happen later on Kaggle/Colab or the laptop, after the student approves budgets.
- **The code must run unchanged on Kaggle/Colab.** The data root and `num_workers` come from the device profile, as in Task 1.
- **Checkpoints (§3.8):** promoted names are `t2_classifier`, `t2_ae_salt`, `t2_ae_blur`, `t2_ae_occlusion`. They are immutable once promoted. Task 3 will load them read-only and assert their sha256.
  - Every checkpoint's `config` must contain the full `model` section, so the model can be rebuilt with `<Class>.from_config(ckpt["config"]["model"])`.
  - The `config` must also contain the component name and, for specialists, the corruption name and `cond_id`.
- **Class order everywhere:** `0 clean, 1 salt_pepper, 2 gaussian_blur, 3 occlusion`. Specialist short names are `salt`, `blur`, `occlusion`, matching the ONNX file names.

## 3. How Task 2 plugs into the existing entry points (no new CLI flags)
- `--task t2cls` uses `configs/task2_classifier.yaml`, and `--task t2spec` uses `configs/task2_specialist.yaml`. Both call `genai.tasks.task2.train.run_training`, `tune.run_study` and `evaluate.run_evaluation`, as `cli.entry_point` already does.
- Each config gets a top-level `component: classifier` or `component: specialist`. The three files `task2/train.py`, `tune.py` and `evaluate.py` are **thin dispatchers** on `cfg["component"]` with the §10.1 signatures. The real code lives in the component modules below.
- The specialist corruption is chosen with an override: `--set specialist.corruption=salt|blur|occlusion` (notebooks use `OVERRIDES`). It is required for `run_training` with `t2spec`. Missing or invalid values raise a clear error. The shared study ignores it because it trains all three.
- Run directories:
  - `artifacts/runs/task2_classifier/<run_id>/`
  - `artifacts/runs/task2_specialist_<corruption>/<run_id>/`

  `run_id` comes from `make_run_id`, with desc `t2cls` or `t2spec_<corruption>`. Tracker group: `task2`.

## 4. Agents and ownership

### Agent P: classifier
**Owns:** `src/genai/models/classifier.py`, `src/genai/tasks/task2/classifier.py`, `configs/task2_classifier.yaml`, `tests/test_classifier.py`.

**Model:** `CorruptionClassifier(channels: list[int], dropout: float, num_classes=4)` with `from_config`.
- Architecture: conv-BatchNorm-ReLU blocks with stride-2 downsampling, global average pooling, dropout, then a linear layer to 4 logits.
- Input: `N×3×128×128` in `[0,1]`. Any normalisation happens inside `forward`.
- Output: logits only. Softmax lives in evaluation and the backend; Task 3 divides the logits by τ.

**Training (`train_classifier(cfg, resume, on_checkpoint)`):**
- Cross-entropy with AdamW and weight decay.
- **`PetsTrainDataset(policy="balanced_batch")` with `BalancedBatchSampler`**, so every batch has exactly B/4 of each class. Every batch-size choice must be a multiple of 4; assert it.
- The same resilience as Task 1: atomic `ckpt_last`/`ckpt_best`, full resume including RNG, `on_checkpoint`, `metrics.jsonl`, tracker.

**Validation (val manifest, all 4 conditions):**
- accuracy;
- macro precision, recall and F1;
- per-class precision, recall, F1 and support;
- the **row-normalised 4×4 confusion matrix**, saved as CSV and PNG in the run directory.

The best checkpoint is chosen by **macro-F1** (§3.6).

**Study (`study_classifier(cfg, on_checkpoint)`), name `t2_classifier`:**
- Tunes learning rate, batch size, channel configuration, dropout and weight decay (the PDF minimum).
- Median pruner, SQLite `artifacts/optuna/t2_classifier.db`, `load_if_exists=True`, a per-trial persistence hook, exports to `studies/t2_classifier/`.

**Config:** fill `configs/task2_classifier.yaml` in the same layout as `task1_universal.yaml` (task, seed, data_config, run, model, train, study keys, `tuned_params` with **PROVISIONAL** ranges).
- Keep `n_trials`, `epochs_per_trial` and `train.epochs` as `TBD_AFTER_BENCHMARK`.
- Keep the `TBD` guard behaviour of Task 1.

**Tests:**
- output shape and class order;
- every batch has exact class counts;
- a tiny-subset overfit check on CPU (loss decreases);
- resume continues `global_step`;
- the checkpoint config rebuilds the model;
- macro-F1 and the confusion-matrix code agree with sklearn on a hand-made example.

### Agent Q: specialists
**Owns:** `src/genai/tasks/task2/specialist.py`, `configs/task2_specialist.yaml`, `tests/test_specialist.py`.

**Model:** `UniversalAE` (imported; same class as Task 1, independently trained parameters).

**Training (`train_specialist(cfg, resume, on_checkpoint)`):**
- One specialist per call, using `PetsTrainDataset(policy=f"fixed:{cond_id}")`. It sees only its own corruption, freshly sampled on every load, with the clean image as target.
- Loss: `alpha*L1 + (1-alpha)*(1-SSIM)`, reusing `restoration_loss`.
- Validation uses only the val-manifest rows of its own `cond_id`, and selection is by the fixed J on those rows (§3.6). Also log PSNR, and use the same sample grids as Task 1, limited to its condition.
- The same resilience as Task 1. The checkpoint config records `corruption` and `cond_id`.

**Shared study (`study_specialists(cfg, on_checkpoint)`), name `t2_specialist_shared`:** PDF page 5 permits one shared search for a common architecture.
- Tunes learning rate, bottleneck dimension, channel configuration, batch size and the L1/SSIM weight.
- **Each trial trains a short specialist for each of the three corruptions with the same sampled hyperparameters, and is scored by the mean J of the three.**
- Report each corruption's J as Optuna trial user attributes. Pruning uses the running mean after each corruption finishes, or per-epoch if that's simpler; explain the choice in a comment.
- Config keys for cheaper trials: `epochs_per_trial`, `trial_train_subset`, `trial_val_subset`. This study costs three trainings per trial.
- SQLite `artifacts/optuna/t2_specialist_shared.db`, exports to `studies/t2_specialist_shared/`.

**Finalising:** `write_final_config(study_db, out_path="configs/task2_specialist_final.yaml")` copies the best trial's hyperparameters into a full training config, with `train.epochs` still `TBD` until the student sets it. Each specialist is then trained from it with `--set specialist.corruption=...`, with the same hyperparameters but its own corruption and run directory. The three runs can go on three separate devices.

**Tests:**
- `fixed:k` really yields only class k through the training loader;
- validation filters to the right condition;
- three runs land in three separate directories;
- the shared-study dry run (2 trials, 1 epoch, tiny subset, CPU) writes a DB, a `trials.csv` and per-corruption user attributes;
- `write_final_config` produces a loadable config.

### Agent R: routing, evaluation, ONNX
**Owns:** `src/genai/tasks/task2/routing.py`, `src/genai/tasks/task2/evaluation.py`, `src/genai/export/task2_export.py`, `tests/test_routing.py`, `tests/test_onnx_t2.py`.

**`HardRoutedSystem` (PyTorch, PDF page 5):**
- Steps: classifier, softmax, argmax.
- **Clean uses an identity bypass and never calls a specialist.** Every other class calls its specialist.
- Two modes: **oracle**, which routes by the manifest's true `cond_id`, and **predicted**, which routes by the classifier.
- `load_task2_models(paths: dict)` rebuilds all four models from their checkpoint configs and checks each specialist's stored `cond_id`.

**`evaluate_task2(cfg, checkpoints: dict, final_test=False)`:** the val manifest by default; the test manifest only with `final_test=True`, which is never run now. Outputs, all written to `artifacts/eval/task2/<timestamp>/`:
- a per-image CSV with `image_id, cond, severity, true_class, predicted_class, oracle_route, predicted_route, MAE, SSIM, PSNR, J` for both modes, computed on identical input tensors;
- condition × severity tables for both modes;
- the classifier report (accuracy, macro P/R/F1, per class, normalised confusion matrix);
- **routing-failure analysis:** every row where the predicted route differs from the oracle route, the J drop versus oracle, a confusion table of misroutes, and a grid of the worst cases (target | input | oracle output | predicted output | abs error).

The task-level `evaluate.run_evaluation(cfg, checkpoint, final_test)` gets four checkpoints from `cfg["eval"]["checkpoints"]` (a dict) when `checkpoint` isn't a single path. Document this in the config.

**ONNX (`task2_export.py`, §3.9, opset 17, dynamic batch, `dynamo=False` as in Task 1):**
- `t2_classifier.onnx`: input `input`, output `logits`.
- `t2_ae_salt.onnx`, `t2_ae_blur.onnx`, `t2_ae_occlusion.onnx`: input `input`, output `output`.
- Write a `.meta.json` sidecar like Task 1's, with `smoke` set correctly.
- **Parity:** use `onnx_verify.val_inputs` and the comparison approach of `verify_parity` (import them; don't edit that file). Check 16 val inputs that include all four conditions, with tolerance `ONNX_PARITY_TOL_MAX_ABS`, and append rows to `report/tables/onnx_parity.csv`.
- **Routing parity:** the ONNX classifier picks the same class as PyTorch on those inputs.

**Fixtures:** until real checkpoints exist, test against **randomly initialised fixture checkpoints** that a test helper writes under `artifacts/fixtures/task2/` (gitignored). Any ONNX built from them is marked `smoke: true` and is **never** written to `models/onnx/`. Write it to `artifacts/fixtures/task2/onnx/` instead.

### Agent S: application
**Owns:** the `/api/hard` additions in `app/backend/app/` (add a module such as `hard.py`, plus the minimal routing hookup in `main.py`), `app/frontend/src/**` for the Hard-Routed workspace, the mock-server addition in `app/frontend/mock/**`, and `tests/test_backend_hard.py`.

**Backend, `POST /api/hard` (§3.10):**
- Upload, sample, corruption and preprocessing handling must be the **same code path** as `/api/universal`. Refactor it into a shared helper inside `main.py` if needed; don't copy it.
- Pipeline: classifier ONNX, softmax, predicted class, then either identity bypass or the chosen specialist ONNX.
- The response adds:
  - `probs` (4, class order);
  - `predicted`;
  - `expert` (`salt`, `blur`, `occlusion` or `identity`);
  - `identity_bypass`;
  - `timing_ms` with `preprocess`, `classifier`, `expert` and `total`.
- `/api/health` already lists every `ONNX_FILES` key, so make sure the four Task 2 keys appear.
- A missing model file returns 503 for this endpoint only, and the server still starts.
- Tests: generate tiny random-weight ONNX files inside the test (in a temp dir, pointed at through the backend's model-dir setting). Check identity bypass, the expert choice, a missing-model 503, a bad upload 4xx, and that `/api/universal` still passes.

**Frontend:** make the **Hard-Routed Restoration** workspace functional:
- reuse the Universal Restoration input controls (refactor them into a shared component);
- four labelled probability bars;
- the predicted class and the selected expert, or "identity bypass";
- input and output images, latency breakdown, and a download button.

Plain design is fine. Add `/api/hard` to the mock server and verify with `npm run build` plus a mock-server run. Don't start Docker; no change to the Dockerfiles or Compose should be needed.

## 5. Lead tasks (you, not a sub-agent)
**Owns:** `src/genai/tasks/task2/{__init__,train,tune,evaluate}.py` (the dispatchers), `scripts/*.py`, `src/genai/tasks/cli.py` (only if needed), `notebooks/*.ipynb`, `.gitignore` (only if needed), `docs/DECISIONS.md` rows, `docs/PHASE_T2_REPORT.md`.

1. **Dispatchers (§3).** Write them first, so P, Q and R have the integration point from the start.
2. **Scripts.**
   - `scripts/export_onnx.py`: add the four Task 2 models through `task2_export`.
   - `scripts/benchmark.py`: add `t2cls` (classifier step with balanced batches) and `t2spec` (specialist step; same cost for all three corruptions apart from data loading). Follow `bench_t1`.
   - `scripts/evaluate.py`: support the Task 2 four-checkpoint evaluation.
3. **Notebooks (still thin drivers, §10.2).**
   - Replace the hard-coded `--model t1` in benchmark mode with the selected `TASK`.
   - Document the `t2spec` override `specialist.corruption=...`, and a `write_final_config` step for the specialist study.
   - Keep `RESUME="auto"`, and note that `None` must be written as `null` in `OVERRIDES`.
4. **Runbook in the report.**
   - Data: attach the existing `dist/cloud_data/pets_data.zip` dataset.
   - Order of runs:
     1. benchmark `t2cls` and `t2spec` on the cloud GPU;
     2. classifier study, then classifier final;
     3. specialist shared study, `write_final_config`, then three specialist finals.
   - Parallelism: classifier and specialists are independent and can run on separate devices; the three specialist finals can too. Only Task 3 needs all four.
   - Outputs: what to download, where to place it locally, and how to promote each final with `common.checkpoint.promote` and record the sha256s.
5. **Integration checks (CPU, or GPU where §0 allows):**
   - the **full** `pytest tests -q`, including the existing 98 tests, which must still pass;
   - both study dry runs;
   - one `t2spec` smoke run per corruption (tens of steps, tiny subset);
   - a classifier smoke run;
   - Task 2 evaluation and ONNX parity on fixtures;
   - backend tests;
   - `npm run build`.

## 6. Benchmark (only if §0 says the GPU is free)
Run `scripts/benchmark.py --model t2cls` and `--model t2spec` locally for a few batch sizes, with AMP on and off, and append the rows to `docs/BENCHMARKS.md`. If the GPU isn't free, skip this and say so.

## 7. Order
- **Lead first:** dispatchers and config skeleton keys (`component`, `specialist.corruption`).
- **Wave 1, in parallel:** P, Q, S.
- **Wave 2:** R. It may start immediately against fixtures and integrates when P and Q report their checkpoint layouts.
- **Last:** the remaining lead tasks and integration checks.

## 8. Final report: `docs/PHASE_T2_REPORT.md`, then stop
Include:
- files created or changed, per agent;
- every command run, with its result;
- the full test summary;
- dry-run and smoke outputs;
- the exact checkpoint dict layout and loader functions Task 3 can rely on;
- the Kaggle/Colab runbook;
- benchmark rows, if run.

**Budgets:** propose them **only from measured numbers**. If no GPU benchmark was possible, write the procedure instead: run the benchmark mode on the target GPU, then compute `trials ≈ allotted GPU time / measured time per trial`, remembering the shared specialist study costs three trainings per trial. Do not invent numbers.

**Deviations:** list deviations from the contracts, duplication recorded as refactor candidates, and open questions.

**Not done:** state plainly that there was no real training or study, no Docker Compose check with the new endpoint (unless §0 allowed it), no test-set evaluation, Tasks 3 and 4 not started, and the frontend not derived from Stitch.
