# Worker prompt: Task 3 (jointly trained soft mixture-of-experts restoration)

Open a NEW Claude Code session in this folder with Sonnet 5.5 selected and send:
"Read `docs/WORKER_PROMPT_T3.md` and execute everything below its `---` line as the lead. Read every file it tells you to read before starting."

---

You are the lead of a team of Sonnet coding agents implementing **Task 3** for a university Generative AI assignment: the soft mixture-of-experts restoration model built from the frozen Task 2 classifier and specialists. The team builds:
- `SoftMoE` (gate + identity branch + three experts) and the source-checkpoint finder with sha256 checks;
- the two-stage training loop (warm-up with frozen experts, then joint fine-tuning) and the Optuna study `t3_moe`;
- evaluation with routing analysis and the comparison with Task 1 and Task 2;
- the ONNX export of the complete soft pipeline (`output` and `weights`);
- `POST /api/soft`;
- the Kaggle runner, its notebook, the packaging of the Task 2 checkpoints and the student's Kaggle steps.

The design is fixed in **`docs/TASK3_PLAN.md`**. Implement it; do not redesign it. If something in the plan turns out to be impossible, record a DECISIONS row with status "question" and choose the smallest deviation.

The deadline is close and the student must be able to explain every line: write simple, well-commented code that reads like the existing Task 2 code (`src/genai/tasks/task2/classifier.py` is the closest model). No money may be spent. **Spawn every worker agent with model `sonnet`.**

## 0. Before anything else
1. Ask the student: "Is anything else using this laptop's GPU or Docker right now (Task 4 training or implementation, another Claude window, the Docker stack)?"
   - **If yes, or no answer:** CPU-only for the whole phase: prefix every Python command with `CUDA_VISIBLE_DEVICES=-1` (not an empty string: that does not hide the GPU on this laptop), `num_workers=0`, do not start or stop Docker, nothing above about 2 GB RAM. G2 then runs its CPU part only and G3 is skipped (say so in the report).
   - **If no:** the GPU may be used for G2, G3 and G6 (plan section E), one heavy process at a time.
2. Ask the student plan question **Q2** (budget approval, `docs/TASK3_PLAN.md` section H) once and record the answer as a DECISIONS row. Without an answer, implement with the plan's values but mark them "awaiting approval"; the real Kaggle run (gate G8) needs the approval.
3. Check running processes before any heavy command (`tasklist | findstr python`), and never start a second heavy process.

## 1. Read first
- **Lead, before anything else:** `docs/TASK1_PROBLEMS_AND_FIXES.md` (sections 2 and 4) and `docs/LESSONS_FROM_TASK1.md`. Every fix listed there applies to Task 3; the plan builds them in.
- **All agents:**
  1. `docs/TASK3_PLAN.md` (authoritative; section letters below refer to it);
  2. `docs/IMPLEMENTATION_PLAN.md` §3 (contracts 3.2, 3.5, 3.6, 3.8, 3.9, 3.10, 3.12, 3.13), P4, P5 and §10 (binding);
  3. `ASSIGNMENT_TRANSCRIPTION.md`, PDF pages 6 and 7 (Task 3) and page 2;
  4. `docs/PHASE_T2_TRAINING_REPORT.md` sections 3 and 4 (the numbers Task 3 is compared with).
- **Code to import and follow (do not edit):** `src/genai/tasks/task2/{__init__,runs,routing,classifier,specialist,evaluation}.py`, `src/genai/models/{classifier,autoencoder}.py`, `src/genai/tasks/task1/{config,train,tune}.py`, `src/genai/common/{checkpoint,metrics,tracking,constants,seed}.py`, `src/genai/pets/{dataset,samplers}.py`, `src/genai/export/{onnx_verify,task2_export}.py`, `tools/t2_pipeline.py`, `notebooks/kaggle_t2_pipeline.ipynb`, `docs/KAGGLE_T2_STEPS.md`, `app/backend/app/{main,hard}.py`, `tests/conftest.py`, `tests/t2_fixtures.py`, `tests/test_backend*.py`.

## 2. Hard constraints
- **Ownership:** each agent writes only its files (plan section I). The do-not-edit list there is absolute. Shared files (`scripts/*.py`, `app/backend/app/main.py`, `tests/test_backend.py`, `src/genai/tasks/cli.py`, `tools/package_cloud_code.py`, `docs/DECISIONS.md`, `docs/AI_USE_LOG.md`, `docs/BENCHMARKS.md`, `docs/TRACEABILITY.md`) are edited by you only, additively, after re-reading them immediately before each edit: another window (Task 4) may be editing them.
- **Task 2 checkpoints are read-only.** Never open `models/checkpoints/*` for writing, never copy over them, never promote under a Task 2 name. Tests use fixture copies in tmp folders. Their sha256 must be unchanged at the end of the phase: check it and put the four hashes in the report.
- **Locked test set:** implement `final_test=True` where the plan asks for it, but **never run it**; tests use a tmp test-access log. `artifacts/test_access.log` must not get a new line from this phase.
- **No real training and no real study** until the student has approved the budgets (Q2) **and** the gates G1 to G7 have passed. Allowed locally: unit tests, smoke runs of tens of steps on tiny subsets, the 2-trial dry run under `artifacts/dryrun*/`, the G2 quick check and the G3 benchmark (only if §0 allows the GPU), the G6 export of the untrained init model to `artifacts/fixtures/task3/`.
- **No invented numbers.** Budgets come from plan section C (labelled estimates) and from measured rows. Report only what was run; label estimates as estimates.
- **The code must run unchanged on Kaggle.** No `data_root="local"` default in anything reachable on Kaggle; pass the run's config (D46). Rehearse with `--data-root` against a fake nested input folder (gate G5).
- **Tests do not depend on the real configs:** pin tiny models and tiny budgets inside each test. Run the full suite after any config change. Run pytest with `CUDA_VISIBLE_DEVICES=-1`, `--basetemp=artifacts/pytest_tmp/<name>` and `TMP` / `TEMP` set to `D:\Generative AI\Assignment_01\artifacts\tmp` (C: is nearly full).
- **Windows care:** `num_workers=0` locally. Scripted edits: a literal `\n` inside a non-raw Python string is a real line break; apply multi-edits one at a time with an assertion on each target; after editing a runner or notebook, parse it (`python -m py_compile`, `json.load`) and dry-run it. Never leave a stray `python -` in a shell command (it waits on stdin). Read a run's own `metrics.jsonl` instead of piping background output through `grep`.
- **Notebook:** exactly 4 code cells, `DRY_RUN = True` committed, no debugging cell.
- **MANIFEST.json** is a bare list; `promote()` accepts both layouts. Never overwrite `models/MANIFEST.json` or `report/tables/onnx_parity.csv` with a Kaggle copy: merge.
- **Shared logs:** Task 3 DECISIONS rows are **D80 onwards** (plan B*n* -> D(79+*n*); D60 to D79 belong to Task 4). Re-read `docs/DECISIONS.md` right before appending; if D80 or later is already taken, continue after the highest number and list the mapping in the report. Append rows to `docs/AI_USE_LOG.md` (one per agent task) and the Task 3 rows to `docs/TRACEABILITY.md` (plan section A -> file -> evidence -> status).
- **No git commits, no pushes, no package installs** beyond `requirements.txt`. If something is missing, ask the student.
- **W&B:** entity `ahmedlaiq34`, project `genai-a1`, group `task3`; tests and dry runs use `TRACKER=none`.
- **Tone:** neutral and factual in every document and message; state counts and results plainly, without negative or exaggerated framing.

## 3. Per-agent instructions
Pass each agent: its row of plan section I, the plan sections named below, the do-not-edit list, §2 of this prompt, and the instruction to return what the section I table lists. Do not repeat the plan's specification in the agent prompt; point to it.

- **Agent M, model and sources.** Plan B1 to B5, B11, B12, D1, D2; tests in E (`test_moe.py`, `tests/t3_fixtures.py`). Write `tests/t3_fixtures.py` first and report its function signature at once, so T, E, A and K can use it. Reuse `task2.routing.load_component` for loading and its component checks.
- **Agent T, training and study.** Plan B2 to B10, B13 to B18, C, D3 to D5; tests in E. Return `_train`'s signature, the checkpoint layout and `measure_step_times` as soon as they are fixed. Run the study dry run (G4) and paste its output.
- **Agent E, evaluation, ONNX and report assets.** Plan B14, B16, B19, D6, D7, D11, G; tests in E. Start with the ONNX wrapper on fixture models and report any exporter problem immediately (plan H6). The comparison in `evaluate_task3` reuses `task2.routing.HardRoutedSystem`; do not copy its code.
- **Agent A, application.** Plan D9; tests in E. Write `soft.py` and `tests/test_backend_soft.py`; give the lead the exact `/api/soft` code for `main.py` (it replaces the 501 stub and reuses `check_fields`, `prepare_input`, `image_fields`; nothing else in `main.py` changes). Update only the Soft step of `app/frontend_v2/verification/real_backend.mjs` (plan E, after-the-run list); do not run it against a model that does not exist yet.
- **Agent K, Kaggle runner and packaging.** Plan C (stages, polling), D10; tests in E (`test_t3_pipeline.py`). The runner follows `tools/t2_pipeline.py` (copying its supervisor is accepted, decision D22 / D44; note it as a refactor candidate) with the plan's changes: 5 stages, 2-second exit poll, `--set` budget overrides, the `prepare` stage benchmark and gate, source hash checks at start and end. Write `docs/KAGGLE_T3_STEPS.md` from plan F.

## 4. Lead tasks (you)
1. **First:** `src/genai/tasks/task3/__init__.py` (plan D, "Shared names"), so every agent imports the same names. Then spawn Wave 1.
2. **Scripts (additive):** `bench_t3` in `scripts/benchmark.py` (plan D8); a `t3` branch for `--dry-run` in `scripts/tune.py` using `genai.tasks.task3.tune.dry_run_overrides`; a `t3_soft_moe` branch in `scripts/export_onnx.py` (`export_t3`, `verify_t3_parity`, data root from `--device-profile`); check that `scripts/{train,tune,evaluate}.py --task t3` reach the Task 3 functions.
3. **App hookup:** paste Agent A's `/api/soft` into `main.py`; in `tests/test_backend.py` remove `"soft"` from the 501 stub test (delete the test if its parametrisation becomes empty) and say so in the report.
4. **`tools/t3_quick_check.py`** (gate G2, plan E): read-only; prints the numbers listed there. Use the numbers to confirm or adjust the tau range (B18) and the pruner warm-up (B17); any change is a DECISIONS row and a change in `configs/task3_moe.yaml`, followed by the full test suite.
5. **Gates, in this order** (plan E): G1 tests -> G2 quick check -> G3 benchmark (if GPU allowed) -> G4 study dry run -> G5 full local pipeline dry run against a fake nested input folder that contains the unzipped pets data and the unzipped `t3_sources.zip` -> G6 real-size export, parity and `/api/soft` with the untrained init model. Stop at the first failing gate, fix it, and re-run from that gate. Package both zips (`tools/package_t3_sources.py`, `tools/package_cloud_code.py`) only after G6, and confirm that the code zip contains `tools/t3_pipeline.py` with `PIPELINE_VERSION = "t3-v1"`.
6. **Logs:** DECISIONS rows (B1 to B19 plus any deviation), AI_USE_LOG rows, TRACEABILITY Task 3 rows, benchmark rows.
7. Before finishing: re-hash the four Task 2 checkpoints and confirm `artifacts/test_access.log` has no new line.

## 5. Order
- **Lead first:** step 4.1 (minutes).
- **Wave 1, in parallel:** M, A, K. A uses a small stand-in ONNX with the same input/output names until M's model exists; K builds the supervisor and packaging against fixture files.
- **Wave 2, when M reports:** T and E in parallel.
- **Wave 3:** K wires its stages to T's and E's functions; you do steps 4.2 to 4.7.

## 6. Final report: `docs/PHASE_T3_REPORT.md`, then stop
Include:
- files created or changed, per agent;
- every command run, with its result;
- the full test summary (previous count and new count);
- the G2 quick-check numbers (init J vs Task 2 for tau 1 / 2 / 4, logit margin, weights per class; with the GPU also the 1 + 3 epoch curve and seconds per step) and what they changed (tau range, pruner warm-up);
- the G3 benchmark rows if run (laptop numbers, labelled);
- the dry-run outputs (study and full pipeline) and the G6 parity and `/api/soft` results;
- the exact checkpoint layout and the API contract of `/api/soft`;
- the budgets as approved (or "awaiting approval"), with the projection rule the `prepare` stage uses;
- the student's answers to Q1 and Q2; the D-number mapping;
- the Task 2 checkpoint hashes before and after (must be identical);
- deviations from `docs/TASK3_PLAN.md`, duplicated code recorded as refactor candidates, open questions.

**Not done (state plainly):** no real Optuna study and no real final training of the soft MoE; no Kaggle run (the student runs it, `docs/KAGGLE_T3_STEPS.md`); no test-set evaluation; `t3_soft_moe` not promoted and `models/onnx/t3_soft_moe.onnx` not written; no Docker Compose check and no `real_backend.mjs` run with a trained Task 3 model; report figures and tables for Task 3 not generated from real results.
