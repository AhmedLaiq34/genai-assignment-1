# Worker prompt: Task 0 (shared data pipeline) + Task 1 + minimal app

Paste everything below the line into the orchestrating Sonnet session. It spawns the waves in order.

---

You are the lead of a team of Sonnet coding agents implementing **Task 0 (shared pet data pipeline), Task 1 (universal denoising autoencoder) and a minimal frontend/backend** for a university Generative AI assignment. The student must understand and explain every line, so write simple, readable, well-commented code. No money may be spent. Do not use paid services.

## Read first (all agents)
- `docs/IMPLEMENTATION_PLAN.md` (master plan). **§3 contracts and §10 workflow/notebook/resilience rules are binding.** For this phase, the per-agent file ownership in *this* prompt (agents E0, A to D, F, G) replaces the coarser §6 table. `.gitignore` already contains `dist/`. Also read `docs/CONTRACTS.md` and `docs/DECISIONS.md`.
- `ASSIGNMENT_TRANSCRIPTION.md` pages 2 to 4 and 8 (data, Task 1, app requirements). The PDF controls if anything conflicts.
- Existing scaffold under `src/genai/`, `scripts/`, `configs/`, `app/`, `notebooks/`. Stubs raise `NotImplementedError`; fill them in. Do not rename files or change function signatures given in CONTRACTS without recording it in `docs/DECISIONS.md` and telling the lead.

## Ground rules
- Windows 11, Git Bash/PowerShell. Use a venv at `.venv` with Python 3.13. Local GPU is an RTX 3050 6 GB, 15.6 GB RAM. Use `num_workers` of 4 or fewer, `persistent_workers=True`.
- **Do not edit** `docs/CONTRACTS.md`, `docs/IMPLEMENTATION_PLAN.md`, `ASSIGNMENT_TRANSCRIPTION.md`, `PROJECT_PHASES.md`. If a contract looks wrong, write a note in `docs/DECISIONS.md` (status "question") and tell the lead.
- Each agent writes **only** the files it owns (below). Never edit another agent's files.
- Do not invent numbers the assignment does not give (trial counts, epochs). Keep `TBD_AFTER_BENCHMARK` in configs until measured.
- **The official Oxford test set is locked.** No training/tuning code path may read it. Only a function called with `final_test=True` may, and it must append a line to `artifacts/test_access.log`.
- **Do not launch long training or the Optuna study.** Allowed compute: unit tests, a short smoke run (tens of steps), and the benchmark script. Full study and final training start only after the lead reports the benchmark and the student approves budgets.
- Never claim something works without running it. Report exact commands run and their output.
- Keep a short line per tool/purpose in `docs/AI_USE_LOG.md` (append only; each agent adds its own row).
- Do not commit to git unless told to.

## Workflow, notebooks and resilience
Follow `IMPLEMENTATION_PLAN.md` §10 exactly. In summary (the plan wins if anything differs):
- **Entry points (§10.1):**
  - `run_training(cfg, resume=None, on_checkpoint=None) -> Path`
  - `run_study(cfg, on_checkpoint=None) -> Path`
  - `run_evaluation(cfg, checkpoint, final_test=False) -> Path`

  `scripts/*.py` only parse args, merge `configs/devices/<profile>.yaml`, and call these.
- **Notebooks (§10.2):** four cells only: setup, parameters, call the function with an `on_checkpoint` hook, sync outputs. No model, loss, data or loop code.
- **Resilience (§10.3):**
  - atomic `ckpt_last.pt` every `checkpoint_every_minutes` and at each epoch end;
  - `ckpt_best.pt` on improvement;
  - full resume including all RNG states;
  - `metrics.jsonl` per epoch;
  - Optuna SQLite with `load_if_exists=True` and a per-trial persistence callback.
- **Code and data transport (§10.4):**
  - Code is edited only locally, pushed to GitHub, and cloned by the cloud notebooks. The fallback is a zipped private Kaggle Dataset.
  - Data is prepared locally. Only `data/cache/pets128/`, `data/splits/` and `data/manifests/` go to the cloud as a private Kaggle Dataset. The manifest sha256 is verified at load.
  - Never upload `.venv/`, `node_modules/`, `artifacts/`, `data/raw/` or `app/`.
- **Device choice:** Task 1 is attempted locally first; the cloud is used only if the benchmark shows the budget does not fit.

## Wave 0: environment (Agent E0, runs alone first, small)
Owns: `docs/ENVIRONMENT.md`, venv. Create `.venv` (`py -3.13 -m venv .venv`), install torch/torchvision from `https://download.pytorch.org/whl/cu128`, then `pip install -e . -r requirements.txt`. Verify `scripts/env_check.py`, CUDA tensor on GPU, `python -c "import optuna, onnx, onnxruntime, pytorch_msssim"`. Update the ENVIRONMENT doc with the real versions. If CUDA fails, stop and report to the lead.

## Wave 1: run concurrently once Wave 0 passes

### Agent A: Task 0 core (corruptions, split, manifests)
Owns: `src/genai/pets/corruptions.py`, `split.py`, `manifests.py`, `tests/test_corruptions.py`, `tests/test_split.py`, `tests/test_manifests.py`.
Implement exactly CONTRACTS section 3.3 and 3.4:
- Corruptions on `[0,1]` float tensors `3x128x128`: salt-and-pepper (per-pixel mask shared across channels, black/white 50/50), Gaussian blur (kernel 3/5/7, sigma range, reflect padding), black rectangular occlusion (1 to 3 rectangles, **union** coverage 10 to 35 percent, rejection sampling). `sample_params(rng)` and `apply(img, params, rng)` as in the contract; `params` is JSON-serialisable and is exactly what is stored in manifests.
- Fixed test severities from the contract table (non-overlapping rectangles for test occlusion with coverage about 10/20/35 percent within 1 percentage point).
- Severity label for training/validation: tertiles of the severity parameter.
- `split.py`: sorted official trainval IDs, `np.random.default_rng(42).permutation`, 80/20, test IDs stored separately with `locked: true`, sha256 of ID lists.
- `manifests.py`: build val manifest (4 rows per val image) and test manifest (10 rows per test image) with per-row seeds, schema from the contract, `manifest_version`, sha256 file. `render(row, clean_img)` regenerates the corrupted tensor from a row.
Tests (must pass): no ID overlap, 80/20 counts, reproducible split sha, 10 rows per test image, byte-identical regeneration of a row, union coverage correct with overlapping rectangles, salt fraction close to p, parameter ranges respected, different outputs across two random draws.
Return: API summary (function signatures), test output.

### Agent B: shared utilities
Owns: `src/genai/common/seed.py`, `checkpoint.py`, `tracking.py`, `metrics.py`, `timing.py`, `tests/test_common_*.py`.
- `seed_everything(seed)`, `worker_init_fn`.
- `checkpoint.py`: atomic save/load of the contract's checkpoint dict, sha256 helper, RNG-state capture/restore, `promote(ckpt, name)` that copies to `models/checkpoints/<name>.pt` and appends to `models/MANIFEST.json`.
- `tracking.py`: thin wrapper with `init_run(cfg, run_id)`, `log(metrics, step)`, `log_images(name, tensor_grid, step)`, `finish()`. Backend chosen by env var `TRACKER=wandb|mlflow|none` (default `none` so tests need no account). Implement W&B and MLflow behind the same interface.
- `metrics.py`: L1, SSIM (use `pytorch_msssim`, `data_range=1`), PSNR, and the **fixed study objective** `J = 0.5*L1 + 0.5*(1-SSIM)`; a helper `evaluate_restoration(model, loader, device)` returning per-condition aggregates.
- `timing.py`: CUDA-synchronised timer context manager.
Tests: checkpoint round-trip including RNG state, atomic save does not leave partial files, metrics on known tensors.

## Wave 2: after Agent A's tests pass (agents C and D may start from API signatures once A reports them)

### Agent C: Task 0 dataset layer
Owns: `src/genai/pets/dataset.py`, `samplers.py`, `scripts/prepare_pets.py`, `scripts/package_cloud_data.py`, `configs/data_pets.yaml` (values only if a contract value is missing; never change contract values), `tests/test_dataset.py`, `tests/test_samplers.py`.
- `scripts/prepare_pets.py`: download Oxford-IIIT Pet via torchvision (`trainval` and `test` splits) into `data/raw/`; load each image with `ImageOps.exif_transpose`, RGB, bicubic resize to 128x128; cache as uint8 numpy arrays plus an ID list in `data/cache/pets128/`; write `data/splits/pets_split.json` and the val/test manifests with sha256 files into `data/manifests/`; save a corruption example grid to `report/figures/corruption_examples.png` (clean, three corruptions at low/medium/high). Print counts.
- `dataset.py`: `PetsTrainDataset(split="train", policy=...)` (runtime corruption sampled every `__getitem__`, returns `(corrupted, clean, cond_id, severity_id)` float32 `[0,1]`), `PetsManifestDataset(manifest_path, split="val"|"test", final_test=False)` (deterministic via `render`). Test split access requires `final_test=True` and logs to `artifacts/test_access.log`. Cache loaded into RAM as uint8.
- Datasets must verify the manifest sha256 against the stored `.sha256` file on load and fail loudly on mismatch (plan §10.4).
- Data root comes from the device profile (`configs/devices/*.yaml`), so the same code reads `data/` locally and `/kaggle/input/<dataset>/` or `/content/...` in the cloud.
- `scripts/package_cloud_data.py`: builds `dist/cloud_data/pets_data.zip` containing only `data/cache/pets128/` (trainval only by default; `--include-test` adds the test cache for the final evaluation job), `data/splits/` and `data/manifests/`, plus a `CONTENTS.sha256`. Ready for upload as a private Kaggle Dataset or to Drive. Add `dist/` to `.gitignore` (tell the lead; A0 owns `.gitignore`).
- `samplers.py`: `iid_uniform` (Task 1), `balanced_batch` (exact B/4 per class, assert B%4==0), `fixed:k` (specialists). Used by Tasks 2 and 3 later, so keep them generic and tested.
Tests: two loads of one index differ in train; manifest dataset deterministic across loads and across workers; balanced batches have exact class counts over 100 batches; fixed:k yields only class k; test split refuses access without the flag; shapes `3x128x128`, dtype float32, range within `[0,1]`.
Return: counts printed by prepare script, the example grid path, test output.

### Agent D: Task 1 model, training, tuning, evaluation, export
Owns: `src/genai/models/autoencoder.py`, `src/genai/tasks/task1/*`, `configs/task1_universal.yaml`, `src/genai/export/*`, `scripts/benchmark.py`, `scripts/export_onnx.py`, `scripts/train.py`, `scripts/tune.py`, `scripts/evaluate.py`, `tests/test_autoencoder.py`, `tests/test_onnx_t1.py`, `notebooks/kaggle_train.ipynb`, `notebooks/colab_train.ipynb`.
Follow `IMPLEMENTATION_PLAN.md` sections 3.6, 3.8, 3.9, 3.13, 10 and P3 (Task 1).
- `autoencoder.py`: configurable `UniversalAE(in_ch=3, base_channels, depth, bottleneck_dim, dropout)`: conv encoder with stride-2 downsampling and growing channels, **flatten then dense layer to `bottleneck_dim`** (a genuine compressed latent), dense layer back, conv-transpose or upsample+conv decoder, `sigmoid` output, **no skip connections**. The same class will be reused for the Task 2 specialists, so keep all architecture choices constructor arguments. Add a `count_parameters` helper and a shape/compression-ratio print.
- `train.py`: `run_training` per the notebook policy; loss `alpha*L1 + (1-alpha)*(1-SSIM)`; AdamW; optional AMP flag; optional cosine scheduler; validation each epoch on the **val manifest** with the fixed objective J and per-condition metrics; sample grids (clean, corrupted, restored, |error|) for the same 12 fixed val images each N epochs; tracker logging.
- `tune.py`: `run_study` for study `t1_universal` tuning lr, batch size, bottleneck dim, encoder channels, dropout and alpha (the PDF minimum). Trial epochs and count come from the config; leave `TBD_AFTER_BENCHMARK`. Median pruner, SQLite, `load_if_exists`, per-trial persistence hook, export `trials.csv` and plots to `studies/t1_universal/`. **Write and test it with a 2-trial, 1-epoch dry run on a tiny subset only.**
- `evaluate.py`: `run_evaluation` on the val manifest by default; `final_test=True` runs the test manifest (condition x severity tables for clean, salt, blur, occlusion and low/medium/high; per-image CSV with MAE, SSIM, PSNR; grids of target | input | output | absolute error map; automatic selection of 12 representative and the 4 worst cases). Do not run `final_test` now.
- `benchmark.py --model t1 --batch ... [--amp]`: warmed-up seconds/step with CUDA sync, peak memory, dataloader throughput; appends a row to `docs/BENCHMARKS.md`. Run it locally for a few batch sizes with and without AMP.
- `export/onnx_export.py`, `onnx_verify.py`, `scripts/export_onnx.py`: export `t1_universal_ae.onnx` (opset 17, dynamic batch, input `input`, output `output`), check with `onnxruntime` on 16 val tensors, report max/mean abs difference.
- Notebooks: rewrite the two scaffold notebooks as thin drivers per plan §10.2.
  - Setup cell: `git clone --branch <BRANCH> <REPO_URL>`, with placeholders the student fills in; if the clone fails, fall back to the code zip at `/kaggle/input/<code-dataset>/`. Then `pip install -e .` and print the GPU.
  - Parameters cell: `TASK`, `CONFIG`, `DEVICE_PROFILE`, `RESUME` (auto-detect the latest `ckpt_last.pt` in the persistent folder if set to `"auto"`).
  - Run cell: imports and calls `run_training` or `run_study` with an `on_checkpoint` hook. On Colab the hook copies to Drive; on Kaggle it is a no-op because `/kaggle/working` persists.
  - Final cell: lists the outputs to download.
  - Both notebooks must be valid nbformat 4 and contain no logic beyond this.
- Add `run_id` creation and a `resume="auto"` helper in `train.py`, so a restarted cloud session continues the same run instead of starting a new one.
- Smoke run: train about 50 steps on the cached data, write a checkpoint, **kill and resume** from it (verify that the step count and metrics continue), export to ONNX, verify parity. Tag outputs as smoke (run id suffix `_smoke`) and do not promote them as final.
Return: benchmark table, smoke log, ONNX parity numbers, parameter count and compression ratio, list of config fields still `TBD_AFTER_BENCHMARK`.

## Wave 2 (parallel, independent): app

### Agent F: frontend (design is irrelevant now; function over style)
Owns: `app/frontend/**`. React + Vite + Tailwind CSS. A single page with a top navigation of four workspaces: **Universal Restoration**, **Hard-Routed Restoration**, **Soft Mixture-of-Experts Restoration**, **Face-to-Sketch Generator**. Only Universal Restoration is functional now; the other three show a "not implemented yet" panel with their final layout stubbed (probabilities bar placeholder, four-weight bar placeholder, side-by-side photo/sketch with style selector, webcam button placeholder). Universal Restoration: choose a clean sample (from `GET /api/samples`) or upload an image; pick a corruption (none, salt-and-pepper, blur, occlusion) and a severity (low/medium/high or custom parameters); button Run; show input, restored output, applied corruption settings, inference time, download button. Handle loading and error states. Use a single `src/api.js` for all fetch calls with base path `/api`. Add a Vite dev proxy to `http://localhost:8000`, a `Dockerfile` (build then nginx) and an `nginx.conf` proxying `/api`. Note in `app/frontend/README.md` that the layout must later be rebuilt from the Google Stitch design.
Verify with `npm install && npm run build`; test against a mock server or the real backend once Agent G is done.

### Agent G: backend and integration
Owns: `app/backend/**`, `docker-compose.yml`, `tests/test_backend.py`.
FastAPI per CONTRACTS section 3.10. Implement now: `GET /api/health` (which ONNX models loaded, with sha256), `GET /api/samples` and `GET /api/samples/{id}` (a few clean images from the official test cache, display only, small fixed list), `POST /api/universal` (multipart `file` or `sample_id`, plus `corruption`, `severity` or explicit `params`, `seed`). Preprocessing must be **identical** to training: `exif_transpose`, RGB, bicubic 128x128, `[0,1]`. Runtime corruption must import `genai.pets.corruptions` (install the package in the backend image from the repo; do not copy-paste the logic). If no corruption is selected the upload is not corrupted a second time. Run `t1_universal_ae.onnx` with ONNX Runtime on CPU, return base64 PNGs of input and output, applied params, and timing breakdown in ms. Validate file type and size; clear 4xx errors. The other endpoints return HTTP 501 with a JSON message. Missing model file: health reports it, endpoint returns 503, server still starts.
Backend `Dockerfile` must build with the repo root as context so the `genai` package can be installed. Write `docker-compose.yml` with `backend` and `frontend` services, `./models/onnx` mounted read-only, one command `docker compose up --build`. Test against the **smoke ONNX** from Agent D. Add pytest tests using FastAPI `TestClient` (health, samples, universal with a small smoke model or a tiny random-weight ONNX generated in the test).

## Order and gates (lead)
1. Wave 0 passes. 2. Wave 1 (A and B) pass their tests. 3. Wave 2: C starts when A passes; D starts when A, B and C APIs are reported (D may begin the model and export code earlier but must not train on real data before C passes); F starts immediately; G starts when D has a smoke ONNX (it may begin with a random-weight ONNX).
4. **Gate T0:** all data tests pass, `scripts/prepare_pets.py` has run, counts and example grid reported. 5. **Gate smoke:** checkpoint, resume, ONNX parity, backend response and browser path work, locally and with `docker compose up --build`.

## Final report from the lead (stop here, do not start the study or final training)
Report: files created or changed per agent; the path and size of `dist/cloud_data/pets_data.zip` and the exact steps for the student to upload it to Kaggle and push the code to GitHub (do not do either yourself); every command run with its result; test summary; dataset counts; benchmark table; ONNX parity; docker compose result; list of deviations from the contracts and open questions; and the concrete proposed trial/epoch budget for the Task 1 study derived from measured step times (labelled as a proposal for the student to approve). State plainly what is not done: no Optuna study, no final Task 1 training, no test-set evaluation, Tasks 2 to 4 not started, frontend design not derived from Stitch.
