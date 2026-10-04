# Phase report: Task 3 implementation (soft mixture of experts)

Date: 2026-10-05. Lead: Claude (Sonnet 5.5); workers M, T, E, A, K (Sonnet). Design: `docs/TASK3_PLAN.md`. Nothing was trained for real; the real study and final training run on Kaggle (`docs/KAGGLE_T3_STEPS.md`).

## Student answers
- Q1: GPU and Docker were free (GPU used for G2, G5 and G6; Docker not started).
- Q2: budget of plan section C approved as planned (12 trials x (1+3) epochs, 16-minute cap, final 2+12 epochs, runner about 19 to 30 minutes, estimate). Recorded as D99.
- The student skipped the stand-alone laptop benchmark (G3 via `scripts/benchmark.py`) to save time. Laptop timings come from the `prepare` stage of the G5 run (see below). `bench_t3` is added to `scripts/benchmark.py` but has not been run.

## Decision numbers
B1 to B19 = D80 to D98 (`B<n>` -> `D(79+n)`); Q2 budget approval = D99. D80 was free (highest existing was D53).

## Files, per agent
- Lead: `src/genai/tasks/task3/__init__.py`, `tools/t3_quick_check.py`; additive edits in `scripts/benchmark.py` (`bench_t3`), `scripts/tune.py` (t3 dry-run branch), `scripts/export_onnx.py` (`t3_soft_moe` branch), `app/backend/app/main.py` (`/api/soft`, replaces the 501 stub), `tests/test_backend.py` (see below); logs `docs/DECISIONS.md`, `docs/AI_USE_LOG.md`, `docs/BENCHMARKS.md`, `docs/TRACEABILITY.md`. `scripts/{train,evaluate}.py --task t3` already reach the Task 3 functions through `cli.entry_point`; unchanged.
- M: `src/genai/models/moe.py`, `src/genai/tasks/task3/sources.py`, `tests/t3_fixtures.py`, `tests/test_moe.py`.
- T: `src/genai/tasks/task3/{train,tune}.py`, `configs/task3_moe.yaml`, `tests/test_task3_train.py`, `tests/test_task3_tune.py`.
- E: `src/genai/tasks/task3/evaluate.py`, `src/genai/export/task3_export.py`, `tools/t3_report_assets.py`, `tests/test_task3_eval.py`, `tests/test_onnx_t3.py`.
- A: `app/backend/app/soft.py`, `tests/test_backend_soft.py`, Soft step of `app/frontend_v2/verification/real_backend.mjs` (not run).
- K: `tools/t3_pipeline.py` (`PIPELINE_VERSION = "t3-v1"`), `tools/package_t3_sources.py`, `notebooks/kaggle_t3_pipeline.ipynb` (4 code cells, `DRY_RUN = True`), `docs/KAGGLE_T3_STEPS.md`, `tests/test_t3_pipeline.py`.
- `tests/test_backend.py`: `"soft"` was the only entry of the 501 stub test's parametrisation, so `test_stubs_are_501` was deleted.

## Commands and results
| Gate | Command | Result |
|---|---|---|
| G1 | `pytest tests -q` (CPU, `CUDA_VISIBLE_DEVICES=-1`) | **427 passed** (292 collected at the start of the phase; the count also includes tests of other windows). After a comment-only config edit, the 127 Task 3 and backend tests were run again: 127 passed. |
| G2 | `tools/t3_quick_check.py --rows 2944 --taus 1 2 4 5` (GPU), and `--rows 736 --train` | see below |
| G3 | not run as a stand-alone benchmark | laptop numbers from G5 `prepare` stage |
| G4 | `scripts/tune.py --task t3 --dry-run` with fixture sources (T) | 2 trials complete, trial 0 = PDF start (output in `artifacts/fixtures/task3_dry/dryrun_output.txt`) |
| G5 | `tools/t3_pipeline.py --dry-run --tag g5 --data-root artifacts/fake_input --num-workers 0` (real Task 2 sources, real-size models, nested fake input with unzipped `pets_data.zip` and `t3_sources.zip`) | `ALL STAGES DONE`; study 1 COMPLETE + 1 PRUNED; sources verified at start and end (unchanged); summary `artifacts/dryrun_t3/logs/t3_summary.json`; log `artifacts/logs/t3_g5.log` |
| G6 | `scripts/export_onnx.py --model t3_soft_moe --ckpt <quick-check model> --out-dir artifacts/fixtures/task3/onnx --verify`; `/api/soft` through the FastAPI test client | parity passed; 10 calls status 200 (see below) |

Packaging: `tools/package_t3_sources.py` -> `dist/cloud_models/t3_sources.zip` (57.0 MB, 5 checkpoints, hashes match); `tools/package_cloud_code.py` -> `dist/cloud_code/genai_code.zip` (255 KB, 98 files) contains `tools/t3_pipeline.py` with `PIPELINE_VERSION = "t3-v1"`.

## G2 quick-check numbers (validation manifest, 2,944 rows unless stated; `artifacts/eval/task3/quick_check.json`)
Soft copy of the Task 2 files at initialisation, same tensors:

| System | J overall |
|---|---|
| input (do nothing) | 0.1558 |
| Task 2 oracle | 0.1028 |
| Task 2 predicted | 0.1032 |
| soft, tau 1 | 0.1029 |
| soft, tau 2 | 0.1023 |
| soft, tau 4 | 0.1029 |
| soft, tau 5 | 0.1049 |

- Gate logit margin (top1 - top2): median 8.08, mean 9.73, p10 4.22, p90 17.04.
- Mean top weight: 0.988 (tau 1; 86.8% of rows above 0.99), 0.948 (tau 2), 0.821 (tau 4), 0.760 (tau 5; no row above 0.99).
- Mean weights per true class at tau 1 (identity, salt, blur, occlusion): clean [0.956, 0.000, 0.034, 0.010]; salt [0.000, 1.000, 0, 0]; blur [0.017, 0, 0.983, 0.000]; occlusion [0.006, 0.000, 0.000, 0.993]. Higher tau moves weight toward the identity branch on blur (tau 4: 0.18) and lowers blur J (0.1137 at tau 1, 0.1064 at tau 4) while raising clean J (0.0030 to 0.0161).
- 1 warm-up + 3 joint epochs at the PDF start (tau 1, 736 validation rows, GPU): val J 0.1040 (epoch 0), 0.1036, 0.1039, 0.1040, 0.1039; gate accuracy 0.9946 at epoch 0 and 0.991 to 0.992 afterwards; weights per class unchanged within 0.02; no collapse flag; 10.4 to 12.8 s per epoch including the validation pass. The curve is flat from epoch 1.
- Effect on the plan: tau range [0.5, 5.0] kept (tau 5 gives a mean top weight clearly below 1); pruner warm-up 2 kept (no plateau-then-drop in the curve). No change to `configs/task3_moe.yaml` values (D96, D97). At tau 1 the weights are nearly one-hot, so the PDF start values barely move the model; larger tau is where trials can differ.

## Laptop timings (RTX 3050, from the G5 `prepare` stage, batch 64, AMP on; laptop numbers, not T4)
Warm-up 0.127 s/step, joint 0.185 s/step, full validation 11.9 s, subset validation 7.0 s, peak memory 1,117 MB. The runner's projection for the real run on these numbers: one trial 64.5 s, study 12.9 min, final training 4.9 min, total 23.8 min (limit 35). On Kaggle the `prepare` stage measures the T4 and applies the same rule: it projects minutes per stage from the measured step times and stops the run (exit code 3) if the total exceeds `budget.max_minutes` = 35.

## G6 results
- Export of a real-size model (4,065,717 parameters, from the quick-check run, not promoted, `artifacts/fixtures/task3/onnx/t3_soft_moe.onnx`): parity on 16 real validation inputs (4 per class): `output` max abs diff 1.3e-6, `weights` 1.2e-7 (tolerance 1e-4), max |sum(weights) - 1| 1.2e-7, dominant branch agrees on 16 of 16. The two parity rows written to `report/tables/onnx_parity.csv` by this check were removed again, so the table only holds real models.
- `/api/soft` against that ONNX file: 10 calls (none / salt / blur / occlusion), all 200, weights sum to 1 within 1e-6, dominant branch identity / salt / blur / occlusion as expected, inference 14.6 to 18.6 ms; a bad upload gives 415 (`artifacts/eval/task3/api_check_init.json`). The model is the 1+3-epoch quick-check model, not an untrained one.

## Checkpoint layout (`ckpt_last.pt`, `ckpt_best.pt` in `<output_root>/runs/task3/<run_id>/`)
Standard `build_checkpoint` keys plus `stage` ("warmup" | "joint"), `step_in_epoch`, `rng`, `epoch_sums`, `n_joint_validated`. `config.component = "soft_moe"`, `config.model = SoftMoE.model_config()` (`gate`, `experts{salt,blur,occlusion}`, `tau`, `branches`), `config.source_checkpoints = {name: {file, sha256}}`, `config.run_id`, `run.smoke`. `SoftMoE.from_config(config["model"])` plus the state dict rebuilds the model without any Task 2 file. Best checkpoint = lowest J among non-collapsed joint-stage epochs.

## API contract: `POST /api/soft`
Form fields as `/api/hard`: `file` or `sample_id` (exactly one), `corruption` (none | salt_pepper | gaussian_blur | occlusion, default none), `severity` (low | medium | high, default medium), `params` (JSON), `seed` (default 42). Response: the image fields of `/api/hard` (`input_png_b64`, `output_png_b64`, `corruption_applied`, `params`, `seed`) plus `weights` (4 floats, order identity / salt / blur / occlusion, 6 decimals), `dominant`, `dominant_id`, `ranking`, `timing_ms {preprocess, inference, total}`. Errors: 503 (model file missing), 415 (undecodable upload), 400, 404, 422.

## Budgets (approved, D99) and the projection rule
Study: 12 trials x (1 warm-up + 3 joint) epochs, timeout 16 min (whichever comes first; actual counts are reported), MedianPruner n_startup 4 / n_warmup 2, PDF start values as trial 0. Final: 2 + 12 epochs, validation every epoch. If the Kaggle projection is slower than planned, cut in the order of plan section C through `BUDGET_OVERRIDES` (`--set`): `timeout_minutes=10`, `train.joint_epochs=8`, `epochs_per_trial.joint=2`, `train.val_every_epochs=2`.

## Task 2 checkpoints and the locked test set
Hashes before and after the phase are identical (read-only use):
- classifier `7d4a9a1f8a073e64e8cc21deec4db98d51ad48dc04eb9b5b4215f49102ece028`
- salt `4f8b2edb835c3b967fc8ef4ea1d43a4092841e3bfde4bd583dfb2c3a086eafee`
- blur `0d2aede1d0636d4f65115b2c0d064319cc7a0fd22b55a24fdf9d1af9a750d74c`
- occlusion `d8e82568179939796c1461265bed6af9412f5a727f6093bcc2b9370c2b64e1b4`

`artifacts/test_access.log` still has its 2 lines (no new entry). The T1 checkpoint hash is unchanged as well (`5052e4ff...`).

## Deviations, refactor candidates, open points
- `tools/t3_pipeline.py` copies the Task 2 supervisor structure (accepted, D22 / D44): refactor candidate. `evaluate.py` has a local `_collapse_flags` duplicating `train.collapse_flags`; `t3_report_assets.py` duplicates `to_tex` of `t2_report_assets.py`; `task3_export.py` imports three private helpers of `task2_export.py`.
- `measure_step_times` has an extra optional `sources` argument; `cfg["sources"]["expected_sha256"]` is an optional override of the Task 2 hashes for fixture tests only; `dry_run_overrides` also sets the dry-run budgets; the study folder gets `study_run.json`.
- The local pipeline dry run uses real sources and real-size models, so a Task 2 hash is never accepted from the config in a real run.
- The G5 dry-run folder `artifacts/dryrun_t3/` is emptied by the next pipeline dry run or by `tests/test_t3_pipeline.py`.
- Not verified: the Kaggle path (`/kaggle/input` discovery, secret cell, T4 timings, the budget gate with real T4 numbers) until the Kaggle dry run (G7).

## Not done
No real Optuna study and no real final training of the soft MoE; no Kaggle run (the student runs it, `docs/KAGGLE_T3_STEPS.md`); no test-set evaluation; `t3_soft_moe` not promoted and `models/onnx/t3_soft_moe.onnx` not written; no Docker Compose check and no `real_backend.mjs` run with a trained Task 3 model; report figures and tables for Task 3 not generated from real results; the stand-alone `scripts/benchmark.py --model t3` run was not executed.

## Next steps for the student
1. Upload `dist/cloud_models/t3_sources.zip` as a new private dataset `t3-sources` and `dist/cloud_code/genai_code.zip` as a new private dataset `genai-code-t3` (never a new version).
2. Import `notebooks/kaggle_t3_pipeline.ipynb`, add the inputs (pets data, `genai-code-t3`, `t3-sources`), GPU T4, Internet On, secret `WANDB_API_KEY`.
3. Dry run (`DRY_RUN = True`, Run All): check `code version: t3-v1`, the `prepare` benchmark and projection (at most 35 minutes), `ALL STAGES DONE`; then `DRY_RUN = False`, Save & Run All (Commit). Details: `docs/KAGGLE_T3_STEPS.md`.

## Addendum: Kaggle run (2026-10-05, D100)
The student capped training at 20 to 25 minutes and asked for the run to be started from here. Budgets were reduced in `configs/task3_moe.yaml` (10 trials, 9-minute cap, 1 + 2 epochs per trial, final 1 + 8 epochs, validation every 2nd epoch, limit 22 minutes). The run was started from the CLI without a Kaggle dry run (student decision) and ended with `ALL STAGES DONE` in about 9.5 minutes of runner time on a T4.

| System (validation, 2,944 rows) | J | SSIM |
|---|---|---|
| input (do nothing) | 0.1558 | 0.729 |
| Task 1 | 0.1237 | 0.791 |
| Task 2 oracle | 0.1028 | 0.828 |
| Task 2 predicted | 0.1032 | 0.828 |
| **Task 3 (soft MoE)** | **0.0932** | **0.846** |

- Task 3: MAE 0.0329, PSNR 28.89 dB; best epoch 6 of 9 (epoch 0 = Task 2 copy at tau 3.64: J 0.1025).
- Best trial 7: joint_lr 7.3e-5, tau 3.64, lambda_c 0.0198, lambda_b 0.0134, recon share r 0.403 (lambda_1 0.403, lambda_s 0.597); no parameter at a range edge. Study: 9 COMPLETE, 1 PRUNED (median), 2 FAIL (first attempts, missing W&B key; retried, then offline mode).
- Gate accuracy 0.946 vs 0.991 for the Task 2 classifier (blur 0.927, occlusion 0.860). A low lambda_c (0.02) leaves the gate less tied to the class label; it shifts weight to the identity branch on blur and occlusion (mean weights per class are in `weights_by_class_severity.csv`). No collapse, no inactive expert.
- Per condition against the input: occlusion 0.1578 vs 0.1909 and salt-and-pepper 0.1125 vs 0.3417 are better; clean 0.0056 vs 0 and blur 0.0970 vs 0.0908 are not.
- Local re-checks: promoted checkpoint sha256 equals the manifest; ONNX parity 8.9e-07 (`output`), 2.3e-07 (`weights`); validation evaluation re-run locally gives J 0.0932; `/api/soft` 10 calls, all 200 (`artifacts/eval/task3/api_check_final.json`); Task 2 hashes unchanged.
- Placed in the repository: `models/checkpoints/t3_soft_moe.pt`, `models/onnx/t3_soft_moe.onnx` (+ sidecar), manifest and parity rows merged, `artifacts/runs/task3/`, `artifacts/eval/task3/`, `studies/t3_moe/`, `artifacts/optuna/t3_moe.db`, `configs/task3_moe_final.yaml`, Kaggle logs in `artifacts/logs/t3_kaggle/`; raw download in `artifacts/kaggle_out/real/`.
- W&B: the secret was not attached, so the 11 runs were logged offline and synced at the end (`wandb sync` line in the status log); check the group `task3` in the project.
- Still not done: test-set evaluation, report figures from these results (`tools/t3_report_assets.py`), Docker Compose check and `real_backend.mjs` run with the promoted model, `docs/PHASE_T3_TRAINING_REPORT.md`.

## Addendum 2: report assets and app check (2026-10-05)
- `tools/t3_report_assets.py` was run on the validation evaluation folder, the final run, the study and the final config: 12 tables (CSV + LaTeX) in `report/tables/task3/` and 8 figures in `report/figures/task3/` (routing heatmap, weights distribution, dominant and distributed examples, J comparison, training curves, architecture, three Optuna plots). All are validation numbers.
- Docker: only the backend image was rebuilt (`docker compose up -d --build backend`; the frontend and the compose file are unchanged); `/api/soft` through nginx on :8080 answers with the promoted model.
- `app/frontend_v2/verification/real_backend.mjs` passed all checks, including Soft MoE for none / salt / blur / occlusion at medium (four weight bars summing to 1, dominant branch shown). Screenshots copied to `report/figures/app/app_soft-{none,salt,blur,occlusion}.png`.
- The test-set evaluation has not been run (needs the student's explicit approval; one `--final-test` session for Tasks 1, 2 and 3 together).

## Addendum 3: test-set evaluation (student approval, 2026-10-05, run once)
`scripts/evaluate.py --task t3 --ckpt models/checkpoints/t3_soft_moe.pt --final-test`: 36,690 test cases, results in `artifacts/eval/task3/20261005-012220_test/`; one new line in `artifacts/test_access.log` (2026-10-04T20:22:19 UTC, pets test manifest). Test assets: `report/tables/task3/test/`, `report/figures/task3/test/`.

| System (test) | J | SSIM |
|---|---|---|
| input (do nothing) | 0.1898 | 0.671 |
| Task 1 | 0.1330 | 0.775 |
| Task 2 oracle | 0.1277 | 0.787 |
| Task 2 predicted | 0.1276 | 0.787 |
| Task 3 (soft MoE) | 0.1142 | 0.811 |

Task 3 by condition (J): clean 0.0065, salt-and-pepper 0.1149, blur 0.1078, occlusion 0.1557. Gate accuracy on test 0.904 (Task 2 classifier 0.991). The ordering of the systems is the same as on validation.
