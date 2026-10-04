# Phase T4 report: style-conditioned face-to-sketch cGAN

Date 2026-10-05. Design: `docs/TASK4_PLAN.md`. Decisions: D59 to D79 in `docs/DECISIONS.md`. Runbook: `docs/KAGGLE_T4_STEPS.md`.

## 1. Result
Final training ran once on a Kaggle Tesla T4 (notebook `ahmedlaiq/t4-pipeline`, version 4, pipeline `t4-v2`), with the **logged best parameters of the Colab study (trial #25)**, **100 epochs** (student's time budget), no Optuna study.

| | |
|---|---|
| Training time | 23.7 min (pipeline total about 25.5 min), about 10 s/epoch |
| Best epoch (lowest val L1) | 57 of 100 |
| Validation, 159 pairs ([0,1] sketches) | L1 **0.0995**, SSIM **0.480**, PSNR **15.8** dB |
| By style (Style 1 / 2 / 3) | L1 0.078 / 0.139 / 0.083; SSIM 0.494 / 0.393 / 0.551 (Style 2 is the weakest) |
| ONNX parity (local, 16 val photos x 3 styles = 48 cases) | max-abs 1.2e-5 (tolerance 1e-4), passed; Kaggle check 5.4e-6 |
| `/api/sketch` with the real model | 200 for Styles 1, 2, 3 in 40 to 80 ms; `sample_id` works; style 4 -> 422 |
| Test set (1,046 pairs, opened once, 2026-10-05, `--final-test`, logged) | L1 **0.1102**, SSIM **0.460**, PSNR **15.2** dB; by style (Style 1 / 2 / 3, n = 619 / 381 / 46): L1 0.083 / 0.160 / 0.067, SSIM 0.504 / 0.371 / 0.606. Files: `artifacts/eval/task4/20261005-011017_test/`, `report/tables/task4/test_by_style.csv`, `report/figures/task4/*_test.png` |

The Colab study's 0.0928 is on an unknown L1 scale and from lost code: it is **not comparable** with 0.0995 and is quoted only as the source of the hyperparameters.

Training curves (`report/figures/task4/losses.png`) show no collapse: D real/fake loss fall slowly to about 0.13, mean sigmoid(D) ends near 0.9 (real) and 0.1 (fake), G L1 falls steadily. Validation L1 rises slightly after epoch 57 (mild over-fitting on 899 pairs), which is why the best checkpoint, not the last, was exported. `style_variations.png` shows three clearly different sketch styles for the same photo.

## 2. Files
- **Data (A):** `src/genai/fs2k/{pairs,split,dataset,prepare}.py`, `configs/data_fs2k.yaml`, `tests/test_fs2k.py`, `tests/fs2k_fixture.py`.
- **Model + training (B):** `src/genai/models/cgan.py`, `src/genai/tasks/task4/train.py`, `configs/task4_final.yaml`, `tests/test_cgan.py`, `tests/test_task4_train.py`.
- **Study (C):** `src/genai/tasks/task4/{tune,log_study}.py`, `configs/task4_cgan.yaml`, `tests/test_task4_tune.py`, `studies/task4_cgan/*` (rebuilt study).
- **Evaluation + ONNX (D):** `src/genai/tasks/task4/evaluate.py`, `src/genai/export/task4_export.py`, `tests/test_task4_eval.py`, `tests/test_onnx_t4.py`.
- **Backend (E):** `app/backend/app/sketch.py`, `/api/sketch` hookup in `app/backend/app/main.py`, `tests/test_backend_sketch.py`; six display-only test photos in `app/backend/samples_sketch/` (student agreed).
- **Lead:** `src/genai/tasks/task4/config.py`, `scripts/{prepare_fs2k,benchmark,export_onnx,tune}.py` (additive), `tools/{t4_pipeline,t4_report_assets}.py`, `notebooks/kaggle_t4_pipeline.ipynb`, `docs/KAGGLE_T4_STEPS.md`; one test line changed in `tests/test_backend.py` (the 501-stub test no longer lists `sketch`).
- **Artifacts (git-ignored):** `artifacts/runs/task4/20261004-1921_kaggle_t4_final_t4/`, `artifacts/eval/task4/20261004-194627_val/`, `models/checkpoints/t4_{generator,discriminator}.pt` (sha256 in `models/MANIFEST.json`, merged with Tasks 1 and 2), `models/onnx/t4_generator.onnx`.
- **Report assets:** `report/figures/task4/*.png`, `report/tables/task4/*.csv` (from `python tools/t4_report_assets.py`); one row appended to `report/tables/onnx_parity.csv`.

## 3. Commands run and results
| Command | Result |
|---|---|
| `python -m genai.fs2k.prepare` (real FS2K) | 1,058 train + 1,046 test pairs, all resolve; split 899/159; caches, audit grid |
| `pytest` on `test_fs2k, test_cgan, test_task4_train, test_task4_tune, test_onnx_t4, test_task4_eval, test_backend_sketch, test_backend, test_backend_hard` | **128 passed** |
| `python -m genai.tasks.task4.log_study` | studies/task4_cgan: 26 trials (14 COMPLETE, 6 PRUNED, 6 FAIL), best #25, 0.0928; printed the B.1 summary block |
| `scripts/tune.py --task t4 --dry-run --n-trials 2 --epochs 1` (real cache, CPU, tiny subset) | Optuna console lines + summary block as specified |
| Real-data smoke training, then resume (`--resume ckpt_last.pt`) | epoch 1 -> 2, `global_step` 8 -> 16, both optimisers restored |
| Smoke ONNX export + parity, smoke val evaluation | parity 4.8e-6; evaluation files written; `artifacts/test_access.log` unchanged (1 old Task 1 line) |
| `tools/t4_pipeline.py --dry-run --data-root <fake nested Kaggle folder>` | `ALL STAGES DONE` (data found by rglob) |
| Kaggle v3 (first real run) | **failed** before training: benchmark stage could not write `docs/BENCHMARKS.md` (docs/ not in the code zip). It did measure the T4: 0.087 s/step fp32, 0.037 s/step AMP (batch 8, 64 ch) |
| Kaggle v4 (real run) | all stages done; W&B offline (secret missing); synced locally as run `ugt4ngh7` |
| `scripts/export_onnx.py --model t4_generator --ckpt models/checkpoints/t4_generator.pt --verify` | parity 1.2e-5, passed |

Local benchmark (RTX 3050, `docs/BENCHMARKS.md`, `bench_t4`): batch 8, 64 channels about 8 to 13 s/epoch fp32. The laptop numbers are noisy (the same configuration measured 0.073 and 0.104 s/step in two runs) and the first row of each channel size in the first benchmark contains a cold-start validation pass (fixed afterwards with an untimed warm-up).

## 4. Checkpoint layout and promotion
`ckpt_*.pt` = CONTRACTS 3.8 dict (`model` = G, `optimizer` = Adam_G, ...) plus `discriminator`, `optimizer_d`, `scheduler_d`, `scaler_d`, `rng`, `step_in_epoch`, `fixed_sample_ids`, `epoch_sums`; `config.model = {base_channels: 64, style_dim: 32, dropout: 0.2934, num_styles: 3}`. `split_checkpoint(ckpt, out_dir)` writes `t4_generator.pt` and `t4_discriminator.pt` (weights + metadata, `role` key). Both were promoted with `common.checkpoint.promote`.

## 5. API contract `/api/sketch` (for the frontend)
`POST /api/sketch` multipart: `style` (1, 2 or 3, required), and `file` (image) **or** `sample_id` (from `GET /api/sketch/samples`). Response 200: `{photo_png_b64, sketch_png_b64, style_id (0..2), style_label, timing_ms:{preprocess, inference, total}}` (raw base64 PNGs, 128x128, sketch grayscale). Errors: 422 bad/missing style; 400 both or neither of file/sample_id or empty file; 404 unknown sample; 413 too large; 415 not an image; 503 model file missing. Samples: `GET /api/sketch/samples`, `GET /api/sketch/samples/{id}` (PNG).

## 6. Not done / open (stated plainly)
- **Test set was evaluated once** (section 1) and must not be used for any further tuning. Style 3 has only 46 test pairs, so its numbers are noisy; Style 2 (index 1) is again the weakest.
- **No Optuna study with the new code** (student skipped plan B.3); the repository study script was only dry-run tested. The Colab study is a rebuilt record (trials 0 to 5 are placeholders from the student's recollection).
- **Only 100 epochs** (intended 200; time budget); validation L1 was still flat to slightly rising after epoch 57, so more epochs would probably not help.
- **W&B** logged offline on Kaggle and was synced afterwards; the run page exists but live logging did not happen.
- **Docker Compose + frontend (2026-10-05):** the Face-to-Sketch workspace of `app/frontend_v2` was connected (sample-photo picker, `style_label` caption, mock updated, `app/backend/Dockerfile` now copies `samples_sketch/`). `docker compose up --build` and `BASE=http://localhost:8080 node verification/sketch_real.mjs` passed with the real `t4_generator.onnx`: six samples, Styles 1/2/3 give three different sketches, upload and PNG download work, no page errors. The stack is left running at http://localhost:8080 (`docker compose down` to stop).
- **Pair audit grid checked by the student (2026-10-05):** `report/figures/task4/fs2k_pair_audit.png` looks right (each sketch belongs to its photo, style labels consistent); the P1b gate is passed.
- **Full `pytest tests` not run in one go** (a first run had one flaky Task 2 test, `test_classifier.py::test_training_loss_decreases_on_tiny_subset`, that passes alone and is not Task 4); only the files listed in section 3 were run.
- UI **Style 2** (annotation index 1) is the weakest in L1/SSIM (0.139 vs 0.078 and 0.083) and, in the sample grids, has the heavier, darker sketches; report it as a limitation.
- FID/LPIPS not computed (optional).
- Duplication: the supervisor in `tools/t4_pipeline.py` repeats `tools/t2_pipeline.py` (refactor candidate); `tune.py` reuses `task1.tune.export_study`.
- Security note: the Kaggle token sits in the git-ignored `artifacts/kaggle_config/`; revoke it in Kaggle > Settings when finished.

## 7. Budget procedure (for a repeat)
`s/epoch = ceil(899 / batch) x s_per_step + val pass` from `bench_t4` on the target GPU; final epochs = what fits (T4 measured 10.8 s/epoch fp32, 5.1 s/epoch AMP); trials for a study `n_trials ~ allotted GPU time / (15 x s/epoch)`.
