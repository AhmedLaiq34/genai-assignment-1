# Worker prompt: Task 4 (style-conditioned face-to-sketch cGAN)

Open a NEW Claude Code session in this folder with Sonnet 5.5 selected and send:
"Read `docs/WORKER_PROMPT_T4.md` and execute everything below its `---` line as the lead. Read every file it tells you to read before starting."

---

You are the lead of a team of Sonnet coding agents implementing **Task 4** for a university Generative AI assignment: the style-conditioned face-to-sketch conditional GAN. The team builds:
- FS2K data preparation;
- the U-Net generator and the PatchGAN discriminator, both with a style embedding;
- the training loop;
- the Optuna study script and the study rebuilt from the Colab console log;
- evaluation;
- the generator ONNX export;
- `POST /api/sketch`.

The design is fixed in **`docs/TASK4_PLAN.md`**. Implement it, and do not redesign it. If something in the plan turns out to be impossible, record a DECISIONS row with status "question" and choose the smallest deviation.

The assignment is individual, and the student must be able to explain every line. Write simple, well-commented code that reads like the existing Task 1 code. No money may be spent. **Spawn every worker agent with model `sonnet`.**

## 0. Before anything else: ask the student one question
"Is anything else using this laptop's GPU or Docker right now (Task 1 training, another Claude window, the Task 2 Docker stack from D50)?"
- **If yes, or no answer:** CPU-only mode for the whole phase.
  - Prefix every Python command with `CUDA_VISIBLE_DEVICES=""` and use `num_workers=0`.
  - Do not start or stop Docker.
  - Do not run anything that needs more than about 2 GB RAM.
  - Do not touch `artifacts/runs/task1*`, `artifacts/runs/task2*`, `artifacts/optuna/t1_*`, `artifacts/optuna/t2_*` or `models/` (except tmp copies in tests).
- **If no:** the GPU may be used for smoke runs and the benchmark (§6).

Also pass on the plan's questions Q2 to Q4 (`docs/TASK4_PLAN.md` section H) to the student once, and record the answers. If there is no answer, use the defaults stated there.

## 1. Read first (all agents)
1. **`docs/TASK4_PLAN.md`.** This is the authoritative design. Section numbers below refer to it.
2. **`docs/IMPLEMENTATION_PLAN.md`.** §3.7 to §3.13, P1b, P2, P3 (T4), P5 and §10 are binding.
3. **`ASSIGNMENT_TRANSCRIPTION.md`**, PDF pages 2, 7 and 8.
4. **`docs/WORKER_PROMPT_T0_T1.md`.** Its "Ground rules" and "Workflow, notebooks and resilience" sections apply unchanged:
   - the test set is locked;
   - no long training or Optuna studies;
   - no invented numbers;
   - no git commits;
   - report only what you ran;
   - append rows to `docs/AI_USE_LOG.md`.
5. **`docs/TASK1_PROBLEMS_AND_FIXES.md`** and the D46 row of `docs/DECISIONS.md`. These are the Kaggle lessons: nested input paths, no hidden `data_root` defaults, and a new dataset per code zip.
6. **Code to import and follow (do not edit):**
   - `src/genai/tasks/cli.py` (key `t4` exists);
   - `scripts/*.py`;
   - `src/genai/tasks/task1/{config,train,tune,evaluate}.py`;
   - `src/genai/common/{checkpoint,metrics,tracking,constants,paths,seed}.py`;
   - `src/genai/pets/dataset.py` (the test-access log pattern);
   - `src/genai/export/{onnx_export,onnx_verify,task2_export}.py`;
   - `tools/t2_pipeline.py`, `notebooks/kaggle_t2_pipeline.ipynb`, `docs/KAGGLE_T2_STEPS.md`;
   - `app/backend/app/{main,preprocess,hard}.py`, `tests/test_backend*.py`.
7. **`studies/task4_cgan/trials_from_console_log.csv`.**

## 2. Hard constraints
- **Ownership:** each agent writes only the files it owns (plan section I). The do-not-edit list in section I is absolute:
  - Task 1 and Task 2 files;
  - `src/genai/common/**`, `src/genai/pets/**`;
  - `app/frontend/**`, `app/frontend_v2/**`;
  - the plan and contract documents.

  Only you, the lead, touch `cli.py`, `scripts/*.py` and the notebooks, and only additively.
- **Locked test set:** implement `final_test=True` everywhere the plan asks for it, but **never run it**. Do not add lines to `artifacts/test_access.log`; tests use a tmp log path. Do not package test images.
- **No real training and no real study.** Allowed compute:
  - unit tests;
  - smoke runs of tens of steps on a tiny subset;
  - one 2-trial, 1-epoch study dry run on the fixture or a tiny subset, writing only under `artifacts/dryrun/`;
  - the benchmark of §6, only if §0 allows the GPU.

  Building the log study (plan B.2) **is** allowed: it only reads the CSV.
- **No invented numbers:**
  - `n_trials` and `train.epochs` stay `TBD_AFTER_BENCHMARK`;
  - the student's intended 200 epochs appears only as a comment;
  - budgets are proposed only from measured rows.
- **The code must run unchanged on Kaggle and Colab.** Paths come from the device profile and `resolve_fs2k_paths`, and there are no hidden "local" defaults (D46).
- **Fixture artifacts:** random-weight checkpoints and ONNX files go under `artifacts/fixtures/task4/`, never into `models/`.
- **Shared log files:** other windows append to `docs/DECISIONS.md`, `docs/AI_USE_LOG.md`, `docs/BENCHMARKS.md` and `docs/TRACEABILITY.md` at the same time.
  - Number your DECISIONS rows **D60 to D79**. Plan decision C*n* becomes D(59+*n*), and extra rows follow after D79's free slots.
  - Re-read each file immediately before appending, and only ever append.
- **No git commits, no pushes, no package installs** beyond what `requirements.txt` already lists. If something is missing (for example scikit-learn), ask the student.

## 3. Per-agent instructions
Each agent's files, functions, config keys and tests are in plan **section D** (specification) and **section E** (tests). Do not repeat them here. Pass each agent its sections and the do-not-edit list.

- **Agent A, data.** Plan C4, C5, C12 and D (`fs2k/*`, `data_fs2k.yaml`); tests in E.
  - Write the mini-FS2K fixture helper first, so B, D and E can use it.
  - The raw data is not on disk (Q4). Code against the fixture, make the real-data tests skip cleanly, and list in the report exactly what must be confirmed on the real download: the annotation field names, the style encoding and the photo→sketch naming rule.
  - Do not guess any of those silently. Parametrise them in `data_fs2k.yaml` with a comment saying "verify on the real download".
- **Agent B, model and training.** Plan C6 to C11, C14 to C18, C20, D (`cgan.py`, `task4/train.py`, `task4_final.yaml`) and E.
  - Return `_train`'s exact signature and the checkpoint dict layout to the lead as soon as they are fixed.
  - The best parameters in `task4_final.yaml` are copied **digit for digit** from the CSV row of trial 25.
- **Agent C, study.** Plan B.1, B.2 and D (`tune.py`, `log_study.py`, `task4_cgan.yaml`) and E.
  - The console output of B.1 is specified exactly; check it in tests with `capsys`/`caplog`.
  - Run `python -m genai.tasks.task4.log_study` once for real, so that `studies/task4_cgan/` holds the DB, `trials.csv`, three plots and `best_params.json`. Paste its printed output into the report.
- **Agent D, evaluation and ONNX.** Plan D (`task4/evaluate.py`, `export/task4_export.py`) and E.
  - Run the random-weight export early (plan E gate 5) and report any ONNX op problem to the lead immediately (plan H5).
- **Agent E, backend.** Plan D (backend) and E.
  - Replace only the `/api/sketch` 501 stub in `main.py` and reuse its upload and preprocess path; do not copy it.
  - Do not touch `/api/universal` or `/api/hard`.
  - Return the final API contract (request fields, response JSON, error codes) for the `app/frontend_v2` owner.

## 4. Lead tasks (you)
1. **First:** `src/genai/tasks/task4/__init__.py` and `config.py` (plan D). Also skeleton keys agreed with B and C.
2. **Scripts (additive):**
   - `scripts/prepare_fs2k.py` as a thin wrapper;
   - `bench_t4` in `scripts/benchmark.py`;
   - the `t4_generator` dispatch in `scripts/export_onnx.py`;
   - make sure `scripts/{train,tune,evaluate}.py --task t4`, including `tune.py --dry-run`, reach the Task 4 functions.
3. **Pipeline:** `tools/t4_pipeline.py` and `notebooks/kaggle_t4_pipeline.ipynb`, modelled on the Task 2 pipeline; steps in plan D. Write `docs/KAGGLE_T4_STEPS.md` following plan F, steps 5 to 7.
   - Rehearse the pipeline locally with `--data-root` pointing at a fake nested Kaggle folder that contains the packaged fixture data (plan E gate 8).
4. **`tools/t4_report_assets.py`** (plan G). Test it on smoke-run outputs only.
5. **Logs:**
   - DECISIONS rows D60 onwards for plan C1 to C20 and any deviation;
   - `docs/AI_USE_LOG.md` rows;
   - `docs/TRACEABILITY.md` Task 4 rows (plan A table → file → evidence → status).
6. **Integration checks:** the full `pytest tests -q`, where all existing tests must still pass and the new count is reported; the study dry run; one smoke training run with a resume; the smoke ONNX export with parity; the backend tests; `build_log_study`.

## 5. Order
- **Lead first:** step 4.1.
- **Wave 1, in parallel:** A, B, E. B uses random tensors until A's fixture exists.
- **Wave 2:** C and D, when B reports `_train` and the checkpoint layout. D may start on fixtures at once.
- **Last:** lead steps 4.2 to 4.6.

## 6. Benchmark (only if §0 says the GPU is free)
- Run `python scripts/benchmark.py --model t4 --batch 8 16 32 --amp both` with base channels 32 and 64, on the fixture or on the real cache if A's real-data step was possible, and append the rows to `docs/BENCHMARKS.md`.
- Laptop numbers are **not** Kaggle numbers. Label them by device.
- If the GPU is not free, skip the benchmark and say so.

## 7. Final report: `docs/PHASE_T4_REPORT.md`, then stop
Include:
- files created or changed, per agent;
- every command run, with its result;
- the full test summary;
- the printed output of `log_study` and of the dry-run study (showing the B.1 console format);
- smoke-run and resume output;
- the ONNX parity rows (fixture);
- the exact checkpoint layout and the `split_checkpoint` / `promote` commands;
- the API contract for `/api/sketch`;
- the student's answers to Q1 to Q4;
- the list of FS2K facts still to verify on the real download;
- the Kaggle runbook pointer (`docs/KAGGLE_T4_STEPS.md`);
- benchmark rows, if run.

**Budgets:** propose them only from measured numbers. If there is no GPU benchmark, write the procedure: run `bench_t4` on the target GPU, compute s/epoch = steps per epoch × s/step + val time, then
- final epochs: apply plan C17 (the student intends 200);
- study trials: `n_trials ≈ allotted GPU time / (15 × s/epoch)`.

**Deviations:** list any deviation from `docs/TASK4_PLAN.md` or the contracts, any duplication of Task 1 code (as a refactor candidate) and any open questions.

**Not done (state plainly):**
- no real FS2K training, no real Optuna study (B.3) and no confirmatory comparison;
- no test-set evaluation;
- no Docker Compose check with the new endpoint (unless §0 allowed it);
- no real-data split, cache or pair-audit grid unless FS2K was available;
- the Face-to-Sketch frontend workspace is not built by this team (`app/frontend_v2` owner);
- the final `t4_generator` / `t4_discriminator` are not promoted and `t4_generator.onnx` is not in `models/onnx/`.
