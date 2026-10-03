# Phase report: Task 0 (data pipeline), Task 1 (universal AE), minimal app

Written by the lead session, 2026-10-04. Everything below was run unless marked otherwise. No Optuna study, final training or test-set evaluation was started. Nothing has been committed to git.

## 1. Status in one paragraph

All items in `docs/WORKER_PROMPT_T0_T1.md` are built and verified, except that nobody has opened the web page in a browser. The only steps left are on your side: upload the data to Kaggle, push the code to GitHub, run the T4 benchmark in the notebook, approve the budget, then train Task 1.

## 2. Verification

- **Tests:** `pytest tests -q` gives **98 passed** (lead's own re-run after all agents finished).
  - 23 corruptions, split and manifests (Agent A)
  - 18 shared utilities (Agent B)
  - 14 dataset and samplers (Agent C)
  - 15 backend (Agent G)
  - the rest: model, ONNX, Task 1 pipeline, constants
- **Data (Agent C, spot-checked by lead):**
  - 3680 trainval images, split into 2944 train and 736 val (seed 42)
  - 3669 test images, locked
  - val manifest: 2944 rows (736 x 4); test manifest: 36690 rows (3669 x 10)
  - val manifest sha256 matches its stored `.sha256` file (`f05f15d5...a5a9e`)
  - split sha256 `20e24ad9...dd34a`, test manifest sha256 `bd964041...97e5`
- **Cloud zip:** `dist/cloud_data/pets_data.zip`, 151.6 MB. Contains the trainval cache, splits, manifests and `CONTENTS.sha256`. No test cache.
- **Corruption examples grid:** `report/figures/corruption_examples.png`.
- **Smoke run (Agent D, local):** 50 steps on real cached data, checkpoint written, resumed. Step count and metrics continued after resume (global_step 25 -> 50). A real process kill was also tried; the checkpoint it caught was only at step 1, so a late-training kill was **not** tested.
- **ONNX parity (smoke model):** 16 val inputs, max-abs difference 2.086e-07, mean-abs 1.75e-08, tolerance 1e-4, passed. Row in `report/tables/onnx_parity.csv`.
- **Optuna dry run (Agent D):** 2 trials x 1 epoch on a tiny subset, study name `t1_universal_dryrun`, outputs under `artifacts/dryrun/` only. `studies/` and `artifacts/optuna/` untouched.
- **Val evaluation of the smoke checkpoint:** ran on 2944 val rows (J 0.4611, SSIM 0.297, PSNR 12.12). Meaningless as a quality number: 50 steps.
- **Docker (Agent G, not re-run by lead):** `docker compose up --build` came up. Through nginx on `http://localhost:8080`, `/api/health`, `/api/samples`, `/api/universal` (gaussian_blur/medium) and the 501 stub `/api/hard` responded; the index page was served. Containers were taken down afterwards. Image sizes: backend 1.66 GB, frontend 93.8 MB.
- **Backend local check:** uvicorn against the smoke ONNX; bad upload returned 415; total request about 44 to 47 ms.

## 3. Files created or changed

| Owner | Files |
|---|---|
| Lead | `.venv`, `docs/ENVIRONMENT.md`, `docs/DECISIONS.md` (D11 to D15), `docs/AI_USE_LOG.md`, `requirements.txt` (added fastapi, python-multipart, httpx, nbformat), renamed `scripts/*.py.py` to `scripts/*.py`, notebook `MODE="benchmark"` |
| A | `src/genai/pets/corruptions.py`, `split.py`, `manifests.py`; `tests/test_corruptions.py`, `test_split.py`, `test_manifests.py` |
| B | `src/genai/common/seed.py`, `checkpoint.py`, `tracking.py`, `metrics.py`, `timing.py`; `tests/test_common_*.py` |
| C | `src/genai/pets/dataset.py`, `samplers.py`; `scripts/prepare_pets.py`, `package_cloud_data.py`; `tests/test_dataset.py`, `test_samplers.py` |
| D | `src/genai/models/autoencoder.py`; `src/genai/tasks/task1/{config,train,tune,evaluate}.py`; `src/genai/tasks/cli.py`; `src/genai/export/{onnx_export,onnx_verify}.py`; `scripts/{train,tune,evaluate,export_onnx,benchmark}.py`; `configs/task1_universal.yaml`; `notebooks/{kaggle,colab}_train.ipynb`; `tests/test_autoencoder.py`, `test_onnx_t1.py`, `test_task1_pipeline.py` |
| F | `app/frontend/**` (React, Vite, Tailwind, Dockerfile, nginx.conf, mock server, README) |
| G | `app/backend/**` (main.py, preprocess.py, make_samples.py, Dockerfile, requirements), `docker-compose.yml`, `.dockerignore`, `tests/test_backend.py` |

## 4. Model

`UniversalAE(base_channels=32, depth=4, bottleneck_dim=256, dropout=0.1)`: conv encoder with stride-2 downsampling, flatten then dense to the latent, dense back, conv decoder, sigmoid, no skip connections. 9,785,987 trainable parameters. 49,152 input values compressed to 256 latent values (192:1). These values are provisional (smoke and benchmark only), not Optuna results.

## 5. Benchmark (local RTX 3050 6 GB, not the T4)

Config as above, 4 workers, one fixed real batch, 20 timed steps after 5 warmup. Train step includes the SSIM loss.

| batch | AMP | peak MB | s/step |
|---|---|---|---|
| 32 | off | 468 | 0.0324 |
| 32 | on | 384 | 0.0272 |
| 64 | off | 897 | 0.0572 |
| 64 | on | 582 | 0.0458 |
| 128 | off | 1175 | 0.1069 |
| 128 | on | 974 | 0.0825 |
| 256 | off | 3415 | 0.2078 |
| 256 | on | 1762 | 0.1569 |

- Loader throughput about 2,300 to 3,000 images/s with runtime corruption.
- A validation pass takes about 3.5 s for 2,944 rows.
- An earlier batch-256 run reported out-of-memory; a rerun measured 3415 MB, so the failure was probably another process using the GPU.

## 6. Proposed Task 1 budget (proposal for you to approve)

Derived from the local numbers only: one epoch at batch 128 is about 23 steps, roughly 6 s including validation.

- **Study:** 40 trials x 15 epochs with the median pruner, about 1 hour on the 3050 before pruning. The T4 should be similar or faster.
- **Final training:** about 100 epochs, roughly 10 to 15 minutes at the 3050 pace.
- **Before committing:** run the notebook's `benchmark` mode on the T4 and send the rows back. If the T4 is slower than estimated, cut the trial count.

`configs/task1_universal.yaml` still has `TBD_AFTER_BENCHMARK` for `train.epochs`, `n_trials` and `epochs_per_trial`. Using them while still TBD raises a clear error. The search ranges are filled in but marked PROVISIONAL.

## 7. Steps for you (I did none of these)

**Upload the data to Kaggle**
1. Kaggle, then Datasets, then New Dataset, set to private.
2. Upload `dist/cloud_data/pets_data.zip` and name it.
3. In a notebook, check the mount path under `/kaggle/input/<dataset>/`. `resolve_data_paths` looks for it.

**Push the code to GitHub**
1. Create a private repository.
2. `git add .`, `git commit`, `git push`.
3. `.gitignore` excludes `.venv`, `data/raw`, `data/cache`, checkpoints and ONNX files.

**Run on Kaggle**
1. Open `notebooks/kaggle_train.ipynb`. Fill in `REPO_URL` and `BRANCH` (and `CODE_DATASET` for the zip fallback).
2. Attach your data dataset, choose the T4, turn on internet.
3. Run with `MODE = "benchmark"` first, then `"tune"`, then `"train"`.
4. Download the run folder and the Optuna `.db` afterwards. The notebook's last cell lists them.

## 8. Deviations and open questions

- **D11:** the MLflow fallback uses SQLite (`artifacts/mlflow.db`) instead of the file store named in CONTRACTS 3.12, because the installed MLflow 3.16 rejects the plain file store. W&B remains the default.
- **D12:** pytorch_msssim SSIM uses an 11x11 Gaussian window by default, not 7x7.
- **D14:** Task 1 training runs on Kaggle by your choice, overriding the plan's local-first rule.
- **D15:** the notebooks have a `benchmark` mode.
- Resume restores all state but is not bit-identical to an uninterrupted run (the per-load RNG counter restarts).
- ONNX export uses the legacy exporter (`dynamo=False`); PyTorch flags it as deprecated.
- The backend calls the private helper `corruptions._sample_rects`; a public wrapper in Agent A's file would be cleaner.
- The backend image is 1.66 GB because it needs CPU torch for the shared corruption code.
- `configs/devices/kaggle.yaml` has `num_workers: 2 # TODO`. It needs the real CPU count from your Kaggle notebook.
- In the notebooks, a `None` value in `OVERRIDES` is parsed as the string "None"; write `null`.
- Local RAM (15.6 GB) is exhausted when several training processes run at once. Run one at a time. Smoke resumes needed `num_workers=0` or `train.val_workers=0` for this reason.

## 9. Not done

- No Optuna study, no final Task 1 training, no test-set evaluation (`--final-test` is implemented but never run).
- Tasks 2 to 4 not started. FS2K dataset access is unconfirmed.
- The frontend design was not derived from Google Stitch, and the page has not been opened in a browser (sample picker, upload, corruption controls and error display need a human check).
- No benchmark for models other than Task 1, and none on the T4.
- Nothing uploaded to Kaggle, nothing pushed to GitHub.
- The smoke checkpoints and `models/onnx/t1_universal_ae.onnx` (marked `smoke: true` in its sidecar) must not be used as final.
