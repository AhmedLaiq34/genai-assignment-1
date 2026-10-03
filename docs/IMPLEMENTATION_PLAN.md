# Implementation Plan — Generative AI Assignment #1

Status: **plan only. No model has been implemented, trained or evaluated.** Nothing in this file is evidence of completed work.

Sources: `ASSIGNMENT_TRANSCRIPTION.md` (PDF page refs as `[p.N]`; the original PDF controls), `PROJECT_PHASES.md` (proposal, reviewed below).
Labels used throughout:
- **[PDF]**: an obligation stated in the assignment.
- **[REC]**: my recommendation or a decision taken to remove ambiguity. It can be changed, but if it is, the change must be recorded in `docs/DECISIONS.md`.
- **[OPT]**: an optional enhancement. Do it only if time remains.

---

## 0. Verified environment (2026-10-03)

| Item | Finding | Consequence |
|---|---|---|
| GPU | NVIDIA RTX 3050 **6 GB** Laptop, driver 610.74 | Fits small 128×128 AEs, the classifier and a modest pix2pix. Batch sizes must be benchmarked, not guessed. |
| RAM | 15.6 GB | Use `num_workers` ≤ 4 on Windows. Preload resized 128×128 images (Oxford ≈ 7.4k imgs × 49 KB ≈ 0.36 GB uint8). |
| Disk | D: 126 GB free | Not a constraint. |
| Python | 3.14 (default), 3.13 available | **[REC]** Create the venv with **3.13** for wheel compatibility. Docker backend uses `python:3.12-slim`. |
| PyTorch | not installed | Install the CUDA wheel (`--index-url https://download.pytorch.org/whl/cu128`). Verify with `torch.cuda.is_available()`. |
| Docker / git / Node | Docker 29.1.5, git 2.52, Node 24 | Compose and the React build are possible locally. Docker GPU is **not** needed (ONNX Runtime CPU inference). |
| Datasets | none present | Oxford-IIIT Pet: free download (torchvision / robots.ox.ac.uk). FS2K: GitHub `DengPingFan/FS2K` → Google Drive link. **Access must be confirmed (open question).** |
| Kaggle / Colab | unverified | Treat as unavailable until the user confirms GPU type, weekly quota and session limits in their own account. |

---

## 1. Review of the handoff against the PDF (conflicts / clarifications)

1. **Deadline:** the PDF prints March 16, 2024. That date is a discrepancy. Use the real Classroom deadline (≈1.5 days of work remaining).
2. **Task 2 ordering:** the PDF says "After training the classifier, train three specialist…" [p.5]. The specialists do not technically depend on the classifier, so **[REC]** start the classifier first and run the specialists concurrently on other devices. The report keeps the PDF's narrative order. Low risk.
3. **Corruption assignment policy:** Task 1 requires i.i.d. equal-probability selection [p.3]. Task 2/3 require *balanced batches* [p.4, p.6]. **[REC]** Use one corruption module with two assignment policies: `iid_uniform` (Task 1) and `balanced_batch` (Task 2 classifier, Task 3), plus a `fixed:<k>` policy for the specialists.
4. **Validation objective vs. tuned α:** if the validation objective used the tuned loss weight α, trials would not be comparable. **[REC]** Every restoration study is scored on a **fixed** objective (§3.6), independent of α/λ.
5. **Hard routing ONNX:** the PDF requires the classifier and the 3 specialists as separate ONNX files [p.5]. The routing logic (argmax + identity bypass) lives in the backend. No combined hard-route graph is required.
6. **Smoke path in the handoff (Phase 2):** this includes React + Compose early. **[REC]** Keep it, but in a minimal form: one FastAPI endpoint, one bare React page, and a Compose file. The goal is to catch integration and ONNX-operator problems early, not to build the UI.
7. **Ambiguities resolved by [REC] (record in DECISIONS.md):** salt-and-pepper is per *pixel* (all 3 channels set together); occlusion coverage = **union** area; test occlusion rectangles are non-overlapping, so coverage is achievable within ±1 percentage point; resize = direct resize to 128×128 (no crop, aspect ratio not preserved), applied **before** corruption.

No conflict was found that requires overriding the handoff on substance.

---

## 2. Repository layout (scaffold)

```
.
├── README.md                     # execution instructions (skeleton now, completed in Phase 6)
├── pyproject.toml                # package `genai` (src layout), pinned deps
├── requirements.txt              # training env;  app/backend/requirements.txt = inference-only
├── .gitignore  .gitattributes    # LFS rules for *.onnx (only if LFS chosen)
├── configs/
│   ├── data_pets.yaml  data_fs2k.yaml
│   ├── task1_universal.yaml  task2_classifier.yaml  task2_specialist.yaml
│   ├── task3_moe.yaml  task4_cgan.yaml
│   └── devices/{local.yaml,kaggle.yaml,colab.yaml}   # paths only, no secrets
├── src/genai/
│   ├── common/      # seed.py, paths.py, checkpoint.py, tracking.py, metrics.py (L1/SSIM/PSNR), timing.py
│   ├── pets/        # split.py, corruptions.py, manifests.py, dataset.py, samplers.py
│   ├── fs2k/        # pairs.py, split.py, dataset.py (paired transforms)
│   ├── models/      # autoencoder.py, classifier.py, moe.py, cgan.py
│   ├── tasks/       # task1/, task2/, task3/, task4/  each: train.py tune.py evaluate.py
│   └── export/      # onnx_export.py, onnx_verify.py
├── scripts/         # thin CLI entry points: prepare_pets.py, prepare_fs2k.py, benchmark.py,
│                    #   train.py --task …, tune.py --task …, evaluate.py --task …, export_onnx.py
├── tests/           # pytest: corruptions, manifests, split, samplers, model shapes, onnx parity
├── data/            # GITIGNORED raw/ and cache/ ...
│   ├── splits/      # COMMITTED: pets_split.json, fs2k_split.json
│   └── manifests/   # COMMITTED: pets_val_manifest.jsonl, pets_test_manifest.jsonl (+ .sha256)
├── artifacts/       # GITIGNORED: runs/<task>/<run_id>/…, optuna working DBs, logs
├── studies/         # COMMITTED: final Optuna SQLite DBs + exported trials CSV/plots per task
├── models/
│   ├── MANIFEST.json   # COMMITTED: name, sha256, size, source run, download URL
│   ├── checkpoints/    # GITIGNORED (final .pt files; distributed via link)
│   └── onnx/           # GITIGNORED or LFS (final .onnx files)
├── notebooks/       # kaggle_*.ipynb / colab_*.ipynb: get code → install → call genai functions → sync outputs (§10)
├── app/
│   ├── backend/     # FastAPI, Dockerfile, requirements.txt (onnxruntime, fastapi, pillow, numpy)
│   └── frontend/    # React + Vite + Tailwind, Dockerfile (build → nginx)
├── docker-compose.yml
├── report/          # IEEEtran LaTeX, figures/, tables/ (generated by scripts, not hand-made)
└── docs/            # IMPLEMENTATION_PLAN.md, CONTRACTS.md, DECISIONS.md, TRACEABILITY.md, ENVIRONMENT.md, AI_USE_LOG.md
```

The notebooks contain **no model logic**. They are thin drivers that import and call repository functions directly. They do not shell out to scripts. The full policy and the cloud workflow are in §10.

---

## 3. Shared contracts (authoritative; mirrored in `docs/CONTRACTS.md`)

### 3.1 Pet image tensors [PDF p.2–3 + REC]
- Load with PIL, apply `ImageOps.exif_transpose`, convert to `RGB`, then `resize((128,128), BICUBIC)`. Cache the resized images as uint8 `.npy`/PNG under `data/cache/pets128/`.
- Model I/O: `float32`, shape `N×3×128×128`, range **[0,1]**. Decoder output passes through `sigmoid`. There is no ImageNet normalisation inside the restoration models. If the classifier/gate normalises, it does so *inside* its own `forward` so that every ONNX input is plain [0,1].
- Corruptions operate on the [0,1] tensor *after* resize.

### 3.2 Class / branch IDs (fixed order everywhere: code, ONNX outputs, API JSON, report)
`0 clean · 1 salt_pepper · 2 gaussian_blur · 3 occlusion`. MoE weights `w[0..3]` follow the same order, with `w0` as the identity branch [PDF p.6].

### 3.3 Corruptions [PDF p.3]
| Type | Train (sampled every load) | Test fixed severities L/M/H |
|---|---|---|
| salt_pepper | p ~ U(0.02, 0.15); a pixel mask is shared across channels; black/white with prob 0.5 | p = 0.03 / 0.08 / 0.15 |
| gaussian_blur | k ∈ {3,5,7} uniform, σ ~ U(0.5, 2.5); reflect padding | (3,0.7) / (5,1.5) / (7,2.5) |
| occlusion | n ∈ {1,2,3}, target union coverage ~ U(0.10,0.35), random locations, value 0; rejection-sample until the achieved union is within range | 1 rect ≈10%, 2 rects ≈20%, 3 rects ≈35% (non-overlapping, ±1 pp) |

- Every corruption function signature is `apply(img01: Tensor[3,H,W], params: dict, rng: np.random.Generator) -> Tensor`, together with a `sample_params(rng)` function. The params dict is exactly what the manifest stores.
- Severity label for training/validation samples [REC]: tertiles of the severity parameter (salt p; blur σ; occlusion coverage). This lets validation be reported per severity if needed.

### 3.4 Split and manifests [PDF p.2–3]
- `pets_split.json`: the official `trainval` IDs are sorted lexicographically, then `np.random.default_rng(42).permutation`. The first `floor(0.8·n)` go to train and the rest to val. The official `test` IDs are stored separately and flagged `"locked": true`. The file includes a sha256 of the ID lists.
- `pets_val_manifest.jsonl`: one row per (val image × 4 conditions), with params sampled from the **training** distributions using a per-row seed `= hash(42, "val", image_id, cond)`.
- `pets_test_manifest.jsonl`: one row per (test image × {clean + 3 types × 3 severities}) = 10 rows per image.
- Row schema: `{image_id, split, cond_id, cond_name, severity: "low|medium|high|null", params:{…}, rects:[[x0,y0,x1,y1],…], achieved_coverage, seed, manifest_version}`.
- **Determinism check:** regenerating any row twice gives byte-identical tensors (test in `tests/`).
- Tasks 1, 2 and 3 all read the **same** split and manifest files. No code path may read `test` unless invoked with `--final-test`, which writes a log line to `artifacts/test_access.log`.

### 3.5 Samplers
- `iid_uniform`: Task 1.
- `balanced_batch`: each batch contains exactly B/4 of each class (B must be divisible by 4; assert). Used by the Task 2 classifier and Task 3.
- `fixed:k`: Task 2 specialist k. It sees only its own corruption, freshly sampled each load.

### 3.6 Fixed validation objectives (study scoring, never tuned)
- Restoration (T1, T2 specialists, T3): minimise `J = 0.5·L1 + 0.5·(1−SSIM)`. This is computed on the val manifest (specialists: only their condition's rows). PSNR is also logged.
- Classifier: maximise macro-F1 on the val manifest.
- T3 additionally prunes on routing collapse [REC]: prune if any branch has mean weight < 0.02 on val, or if one branch is > 0.9 on rows of a different true class. These thresholds are a REC; record them in the report.
- T4: minimise val L1 on [0,1]-rescaled sketches. Also log SSIM. [OPT] FID/LPIPS on val.

### 3.7 FS2K [PDF p.7–8]
- Use the official `anno_train.json` / `anno_test.json` and take style from the annotations. Validate that each photo↔sketch pair exists. Produce a pair audit grid.
- Split: `sklearn.train_test_split(train, test_size=0.15, stratify=style, random_state=42)` → `data/splits/fs2k_split.json`.
- Style IDs `0,1,2` = FS2K annotation styles. UI labels "Style 1/2/3" map to 0/1/2. Verify the annotation encoding in Phase 1 and record it.
- Tensors: photo `N×3×128×128`, sketch `N×1×128×128` (grayscale) [REC], both in **[-1,1]**. The generator ends in `tanh`. The backend converts the output to [0,255].
- Paired augmentation [REC]: resize to 143 → identical random crop 128 → identical horizontal flip. The same RNG draw is used for both images (unit test).
- Style embedding: `nn.Embedding(3, d)` in **both** G (broadcast and concatenated at the bottleneck and/or input) and D (spatially tiled and concatenated with photo+sketch) [PDF p.7].

### 3.8 Checkpoints
- Run directory: `artifacts/runs/<task>/<run_id>/` containing `config.yaml`, `ckpt_last.pt` (overwritten every N min), `ckpt_best.pt`, `metrics.jsonl`, `samples/`.
  - `run_id = <YYYYmmdd-HHMM>_<device>_<shortdesc>`.
- Checkpoint dict: `{model, optimizer, scheduler, scaler, epoch, global_step, best_metric, config, seed, split_sha256, manifest_sha256, git_commit, torch_version, created_utc}`.
- Final models are *promoted* by script to `models/checkpoints/<name>.pt` and recorded in `models/MANIFEST.json` with their sha256. Names: `t1_universal_ae`, `t2_classifier`, `t2_ae_salt`, `t2_ae_blur`, `t2_ae_occlusion`, `t3_soft_moe`, `t4_generator` (+ `t4_discriminator`, used for training only).
- **Task 2 final checkpoints are immutable.** Task 3 loads them read-only and asserts their sha256 at startup.

### 3.9 ONNX (opset 17, dynamic batch axis, CPU ONNX Runtime in the backend)
| File (`models/onnx/`) | Inputs | Outputs |
|---|---|---|
| `t1_universal_ae.onnx` | `input` f32 N×3×128×128 [0,1] | `output` same |
| `t2_classifier.onnx` | `input` | `logits` N×4 (backend applies softmax) |
| `t2_ae_{salt,blur,occlusion}.onnx` | `input` | `output` |
| `t3_soft_moe.onnx` | `input` | `output` N×3×128×128, `weights` N×4 (softmax/τ included) |
| `t4_generator.onnx` | `photo` f32 N×3×128×128 [-1,1], `style` int64 N | `sketch` N×1×128×128 [-1,1] |

Parity check (`onnx_verify.py`): max-abs and mean-abs difference vs. PyTorch on ≥ 16 val inputs (all 4 classes; all 3 styles), with a REC tolerance of 1e-4 on max-abs. Results go to `report/tables/onnx_parity.csv`.

### 3.10 API (FastAPI) [PDF p.8]
`GET /api/health` → model load status, versions, sha256s.
`POST /api/universal`, `/api/hard`, `/api/soft` (multipart: `file`, optional `corruption`, `severity`, `seed`) → `{input_png_b64, output_png_b64, corruption_applied, params, timing_ms:{preprocess,inference,total}, …}`. `/api/hard` adds `probs[4]`, `predicted`, `expert` (and returns `identity_bypass:true` when the prediction is clean). `/api/soft` adds `weights[4]`, `dominant`.
`POST /api/sketch` (`file`, `style` ∈ {1,2,3}) → `{photo_png_b64, sketch_png_b64, style_id, timing_ms}`.
`GET /api/samples` → clean sample images (from the official test set, display only). Uploaded images are resized to 128 and, if no corruption is selected, are **not** corrupted again.

### 3.11 Reproducibility
`seed_everything(seed)` (python/numpy/torch/cuda). DataLoader `worker_init_fn` derives each worker's seed from the base seed. Seeds are recorded in every checkpoint and run. cuDNN benchmark may stay on for training, but this must be stated; evaluation is deterministic because the manifests are fixed.

### 3.12 Experiment tracking [PDF p.2]
**[REC] Weights & Biases (free personal account)** because runs come from up to 3 devices into one place. Use `WANDB_MODE=offline` on devices without internet, then `wandb sync`. Projects/groups: `genai-a1`, with group = task and run name = `run_id`. Log config, per-epoch losses, val metrics, sample grids, the best-checkpoint artifact *reference* (sha256, not the file), and the Optuna trial number.
Fallback if no account: MLflow file store per device (`artifacts/mlruns_<device>/`) with experiment names prefixed by device. This fallback is a decision point (see open questions). `common/tracking.py` is a thin wrapper so that the choice touches one file.

### 3.13 Optuna [PDF p.2, p.4, p.5, p.6, p.8]
- Each study is local SQLite at `artifacts/optuna/<study>.db`. The **writer is one process on one device.** It is never shared across devices and never placed on Drive or a network mount during a run.
- Each trial's callback copies the DB to persistent storage. After a study completes it is promoted to `studies/<study>.db` and exported (`trials.csv`, optimisation history, param importance, parallel-coordinate plots) to `studies/<study>/`.
- Studies and required search parameters (the PDF minimum; ranges are a REC to be refined after benchmarking):

| Study | Must tune [PDF] | Objective | Pruner |
|---|---|---|---|
| `t1_universal` | lr, batch, bottleneck dim, encoder channels, dropout, α | J ↓ | Median |
| `t2_classifier` | lr, batch, conv channels, dropout, weight decay | macro-F1 ↑ | Median |
| `t2_specialist_shared` | lr, bottleneck, channels, batch, L1/SSIM weight | mean J over 3 corruptions ↓ (PDF-permitted shared search) | Median |
| `t3_moe` | joint lr, τ, λ_c, λ_b, reconstruction weighting (λ_1 vs λ_s) | J ↓ + collapse pruning | Median + custom |
| `t4_cgan` | lr_G, lr_D, batch, base channels, dropout, embed dim, λ_L1 | val L1 ↓ | Median (reduced epochs per PDF p.8) |

- **Trial counts and epochs are NOT fixed here.** The rule: after the Phase 2 benchmark, `trials ≈ (GPU-hours allotted to the study) / (measured seconds per trial)`. Trials use reduced epochs and/or a training subset; this is declared in the report. The completed-trial count is whatever actually finished, and it is reported honestly [PDF p.4].

---

## 4. Phases, gates and parallelism

```
P0 env+contracts ─► P1a pet data ─┬─► T1 (smoke→study→final) ────────────┐
                  │               ├─► T2-cls  (study→final) ──┐            │
                  │               └─► T2-spec (study→3 finals)┴─► T3 ─────┤
                  ├─► P1b FS2K ─────► T4 (smoke→study→final) ─────────────┤
                  └─► Stitch (USER) ─► app skeleton ─► workspaces ────────┤
P2 smoke+benchmark gate sits between P1 and every study.                    ▼
                                         P5 final eval + ONNX parity ─► P6 Docker, report, demo
```

**Concurrency rules**
- These can run concurrently *on different devices*: T1, T2-cls, T2-spec, T4, app/frontend work, and report scaffolding.
- **Cannot** start: T3 real training before `t2_classifier` and all three `t2_ae_*` are promoted with hashes. T3 *code* can be written and tested against randomly initialised fixture checkpoints earlier.
- **Cannot** start: any test-set evaluation before configs are locked (P5).
- One heavy GPU job per GPU at a time unless the benchmark shows two fit (on the 6 GB 3050, assume one).

### P0 — Environment and contracts (local, CPU-only work)
Deliverables: venv (3.13) + CUDA torch, `docs/ENVIRONMENT.md` (nvidia-smi, torch/cuda versions, RAM, disk), `docs/CONTRACTS.md`, `docs/DECISIONS.md`, `docs/TRACEABILITY.md` (each PDF obligation → file/evidence → status), configs.
Gate: `python -c "import torch; torch.randn(1,device='cuda')"` succeeds; `docker compose version` works.

### P1a — Shared pet data (critical path; build first, owned by one agent)
Deliverables: `pets/split.py`, `corruptions.py`, `manifests.py`, `dataset.py`, `samplers.py`, `scripts/prepare_pets.py` (download → cache128 → split → manifests → sample grid `report/figures/corruption_examples.png`), and tests.
Verification (all must pass): zero ID overlap among train/val/test; 80/20 counts; reproducible split sha; every test image has exactly 10 rows; regenerating a manifest row is byte-identical; occlusion union coverage is within its range (accounting for overlap); salt fraction ≈ p; balanced batches have exact counts of 4 classes; specialists' samplers yield only their class; training corruption changes across two loads of the same index.

### P1b — FS2K (independent)
Deliverables: `fs2k/` modules, `scripts/prepare_fs2k.py`, split JSON, a pair-audit grid (photo|sketch|style for ~24 samples), and a test that paired transforms are identical.
Gate: all pairs resolve; style counts per split are reported; the audit grid has been checked visually by the **user**.

### P1c — Stitch design (USER action, free Google Stitch)
Four workspaces + a health/system panel. Export screenshots, prompt text and the share link to `report/figures/stitch/`. The frontend is implemented from this design.

### P2 — Smoke path and benchmark (gate before any study)
1. A tiny T1 AE trains 50 steps on the cached data → `ckpt_last.pt` → ONNX export → parity check → `/api/universal` → a bare React page in Compose shows the output.
2. Instantiate the classifier, full MoE (3 experts + gate) and the cGAN with random weights. Export each to ONNX and run it in ORT. This catches unsupported ops.
3. `scripts/benchmark.py --model {t1,t2cls,t2spec,t3,t4} --batch {…}` (AMP on/off). It records peak `torch.cuda.max_memory_allocated`, warmed-up s/step (with `cuda.synchronize`), val pass time and dataloader throughput, and appends to `docs/BENCHMARKS.md` keyed by device. Run it on local GPU and on each confirmed cloud device.
4. Derive per-study budgets from the measurements and write them into the configs. These are measured numbers, not guesses.
Gate: image → loader → step → checkpoint → ONNX → backend → browser works; every model's train step fits on its assigned device.

### P3 — Independent model workstreams
Each workstream follows the same sequence: study (reduced budget) → lock best params → **final training** of the selected config (full schedule) → promote checkpoint → val-only analysis → ONNX export + parity.

- **T1 Universal AE** [PDF p.3–4]: conv encoder (stride-2 downsampling, channels increasing) → **flattened dense bottleneck of dimension `z`** (tuned) → conv decoder. Default: **no skip connections** [REC]; this satisfies "genuine bottleneck" cleanly. [OPT] A limited-skip ablation (e.g. one low-res skip) if time remains, reported as an investigation. Loss: α·L1 + (1−α)(1−SSIM), with SSIM from `pytorch-msssim` or a tested local implementation.
- **T2 classifier** [p.4–5]: small CNN (conv-BN-ReLU blocks + GAP + dropout + FC4). CE loss, balanced batches.
- **T2 specialists** [p.5]: the *same* `autoencoder.py` class, using the shared-study architecture, trained 3× independently with `fixed:k` samplers. Each one is its own run, which allows 3 separate devices.
- **T4 cGAN** [p.7–8]: U-Net G (128→1 bottleneck or 128→4, dropout in the decoder as tuned), 70×70-style PatchGAN D, BCEWithLogits, + λ_L1·L1. Log D_real, D_fake, G_adv, G_L1 separately. Every K epochs, log a fixed grid of the **same** 8 val photos × 3 styles.

### P4 — T3 Soft MoE (dependent)
`moe.py`: gate = a copy of the classifier architecture, loaded from `t2_classifier.pt`; experts = copies loaded from `t2_ae_*.pt`; identity branch; `w = softmax(G(x)/τ)`; `x̂ = Σ w_k·branch_k`. Loss per [p.6]: λ1·L1 + λs(1−SSIM) + λc·CE(G logits, true label) + λb·Σ(w̄_k − ¼)². Two stages: warm-up (experts `requires_grad=False`, gate only) → joint fine-tuning (all parameters, smaller lr).
Verification tests: weights sum to 1; experts unchanged during warm-up (param hash); gradients are non-zero for gate and experts in the joint stage; Task 2 files' sha256 are unchanged after training. Every Optuna trial re-loads the *original* Task 2 checkpoints.
Analysis: mean weights per (true class × severity) heatmap; dominant vs. distributed examples; inactive/dominating expert check.

### P5 — Final evaluation (locked configs; test opened once)
`scripts/evaluate.py --final-test` over the **identical** test-manifest tensors for: T1, T2-oracle, T2-predicted, T3. Outputs: per-image CSV (`image_id, cond, severity, MAE, SSIM, PSNR, pred_class, weights…`), aggregated tables (by condition × severity), the classifier report (accuracy, macro P/R/F1, per-class, normalised confusion matrix [p.5]), grids (target | input | output | |error| map) with ≥12 representative + 4 failure cases for T1 [p.4], classifier-induced routing failures for T2 [p.5], and the routing heatmap for T3 [p.7]. T4: test metrics by style; same-photo/other-style samples shown qualitatively (no ground truth). ONNX parity on all 7 graphs.

### P6 — App, Docker, report, demo
- Backend: an ORT session per model loaded at startup, upload validation (type/size), shared preprocessing identical to training (unit test that compares backend preprocessing with `pets/dataset.py`), runtime corruption reusing `genai.pets.corruptions` (copied in or installed as a package; no reimplementation drift).
- Frontend: React + Vite + Tailwind from the Stitch design, with four workspaces, probability/weight bars, contribution highlight, webcam (`getUserMedia`) and download.
- `docker-compose.yml`: backend + frontend (nginx serving the build, with `/api` proxied). Models are mounted from `./models/onnx`. The one documented command is `docker compose up --build`, preceded by the documented model download (`scripts/fetch_models.py` checks sha256 against MANIFEST.json).
- Fresh-clone test on a clean directory.
- Report (IEEEtran): all figures and tables are produced by scripts into `report/figures|tables`. Appendices cover AI use and reused code.
- Demo video (USER records, 5–7 min): Compose startup, upload, runtime corruption, universal, hard (probabilities), soft (weights), sketch + download, W&B/MLflow records [p.2].

---

## 5. Compute allocation and persistence

| Device | Assigned work [REC] | Notes |
|---|---|---|
| Local RTX 3050 6 GB | All development, P1/P2, **T1 study + final**, ONNX, app, Docker | One GPU job at a time. Use laptop power mode at max and watch thermals. |
| Kaggle (if confirmed) | **T2 classifier → then T3**, or **T4** | Inputs: upload the repo + cached data as a *private Kaggle Dataset*. Outputs in `/kaggle/working`, persisted with "Save Version". Upload checkpoints as a dataset version for the next session. |
| Colab (if confirmed) | **T2 specialists** (3 sequential runs), or **T4** if Kaggle is busy | Mount Drive. Train on `/content` local disk. Copy `ckpt_last.pt` + Optuna DB to Drive every N minutes and at the end of each trial. |

If only one cloud GPU is available, the priority is the critical path: **T2 (classifier + specialists) → T3**. T4 then runs locally after T1, or interleaved.
Rules: each run passes an explicit `--device-profile` and a unique `run_id`. No two jobs share an output dir or study DB. Every training script supports `--resume artifacts/runs/.../ckpt_last.pt` and Optuna `load_if_exists=True`, so a reset loses at most N minutes. No quota circumvention: one account per platform, and published limits are respected.
Measure free-tier limits by reading them in the user's account (Kaggle shows the weekly GPU quota; Colab limits are unpublished, so record observed disconnects). Do not quote them here.

---

## 6. Agent ownership (if using multiple coding agents)

| Agent | Owns (exclusive write) | Consumes | Must return |
|---|---|---|---|
| A0 Orchestrator/User | `docs/CONTRACTS.md`, `configs/data_*.yaml`, `models/MANIFEST.json`, merges | everything | contract changes, merges |
| A1 Shared data | `src/genai/common/`, `src/genai/pets/`, `scripts/prepare_pets.py`, `tests/test_pets_*` | contracts | split + manifests + sha, passing tests, corruption grid |
| A2 Task 1 | `models/autoencoder.py`, `tasks/task1/`, `configs/task1_*.yaml` | A1 | study DB+exports, final ckpt+sha, val/test tables, figures, ONNX |
| A3 Task 2 | `models/classifier.py`, `tasks/task2/`, `configs/task2_*` | A1, `autoencoder.py` (read-only) | 2 study DBs, 4 final ckpts+sha, confusion matrix, oracle/predicted tables, 4 ONNX |
| A4 Task 3 | `models/moe.py`, `tasks/task3/`, `configs/task3_*` | A3 checkpoints (read-only) | study DB, final ckpt, heatmap, collapse analysis, MoE ONNX |
| A5 Task 4 | `src/genai/fs2k/`, `models/cgan.py`, `tasks/task4/`, `configs/*fs2k*, task4_*` | contracts | split, pair audit, study DB, G/D ckpts, loss curves, sample timeline, G ONNX |
| A6 Export | `src/genai/export/`, `scripts/export_onnx.py`, parity tests | all ckpts | parity CSV |
| A7 App | `app/`, `docker-compose.yml` | ONNX contract (works with smoke ONNX before the finals exist) | running Compose, screenshots |
| A8 Report | `report/` | generated figures/tables | LaTeX source + PDF |

Changes to a contract go through A0. Agents never edit another agent's files; they open a note in `docs/DECISIONS.md` instead. Shared `pyproject.toml`/`requirements.txt` are edited only by A0.

---

## 7. Report evidence checklist (per task) [PDF p.1–2, p.9]
Architecture diagram · loss definitions · complete corruption config table · Optuna search space, #completed trials, best trial, final config, plots · train/val curves · result tables (clean + 3 types × 3 severities; T4 by style) · visual grids + error maps · failure cases (T1 ≥4 discussed, ≥12 representative) · T2 confusion matrix + macro metrics + oracle vs. predicted + classifier-caused failures · T3 routing heatmap, dominance/distributed examples, inactive-expert analysis · T4 separate D/G loss curves, fixed-sample timeline · ONNX parity table · app architecture + Stitch evidence + screenshots · limitations · GitHub + YouTube links · citations · AI-use appendix (`docs/AI_USE_LOG.md` is maintained throughout).

---

## 8. Optional enhancements [OPT]
Limited-skip ablation for T1; LPIPS/FID for T4; mixed-corruption probe images for T3 (where hard routing fails); per-severity validation curves; public hosting; Git LFS for ONNX.

---

## 9. First implementation step after the scaffold
**A1 Shared data, step 1:** implement `src/genai/pets/corruptions.py` (three corruptions + `sample_params` + fixed test severities, per §3.3) and `split.py` (§3.4), with `tests/test_corruptions.py` and `tests/test_split.py`. These run on CPU with synthetic images, so the dataset does not need to be downloaded yet. Then `manifests.py` + `scripts/prepare_pets.py`. Task 1's `dataset.py` with the `iid_uniform` sampler is built directly on these, which keeps Task 1 independently runnable while Tasks 2/3 reuse the same code.

---

## 10. Development workflow, notebooks and cloud training [REC; binding on all agents]

### 10.1 Entry-point interface (every task)
Every task package `src/genai/tasks/<task>/` exposes plain functions:
- `train.py`: `run_training(cfg: dict, resume: str | None = None, on_checkpoint: Callable[[Path], None] | None = None) -> Path` (returns the run directory)
- `tune.py`: `run_study(cfg: dict, on_checkpoint=None) -> Path` (returns the study directory)
- `evaluate.py`: `run_evaluation(cfg: dict, checkpoint: str, final_test: bool = False) -> Path`

`scripts/train.py|tune.py|evaluate.py` only parse arguments (`--task`, `--config`, `--device-profile`, `--resume`, `--final-test`), load the YAML config merged with `configs/devices/<profile>.yaml`, and call these functions. Script and notebook use the **same** functions, so the code is tested once.

### 10.2 Notebook policy
- Notebooks (`notebooks/kaggle_train.ipynb`, `notebooks/colab_train.ipynb`) contain only these cells:
  1. setup (get code, `pip install -e .`, show GPU);
  2. parameters (`TASK`, config path, `RESUME`);
  3. `from genai.tasks.<task>.train import run_training` and call it with an `on_checkpoint` hook;
  4. sync outputs to persistent storage.
- They contain no model, loss, data or training-loop code. Changing the logic means editing `src/genai/`.
- Notebooks give no speed advantage. Throughput comes from: data on the runtime's local disk, a RAM-cached uint8 dataset, a suitable `num_workers`, and AMP if the benchmark shows a gain. Their value on Kaggle is unattended background execution ("Save & Run All") and persistence of `/kaggle/working`.

### 10.3 Resilience is Python code, not a notebook feature
`run_training` must:
- save `ckpt_last.pt` **atomically** (write to a temp file, then rename) every `checkpoint_every_minutes` (config) and at each epoch end;
- save `ckpt_best.pt` when the validation objective improves;
- **fully resume** from a checkpoint: model, optimizer, scheduler, AMP scaler, epoch, global step, best metric, Python/NumPy/Torch/CUDA RNG states;
- append `metrics.jsonl` every epoch and log to the tracker;
- call `on_checkpoint(path)` after each save. The notebook supplies this hook to copy the files to Drive (Colab). On Kaggle, `/kaggle/working` is already persisted.

`run_study` uses `artifacts/optuna/<study>.db` with `load_if_exists=True`, and calls `on_checkpoint` after each trial. A reset therefore loses at most `checkpoint_every_minutes` of work, or one unfinished trial.

### 10.4 Where work happens
1. **Local VS Code is the only place code is edited.**
   - Write the code, run the unit tests, the smoke run and the benchmark on the RTX 3050 before using any cloud GPU time.
   - A code failure must never be discovered on Kaggle or Colab.
2. **GitHub is the transport.** Push to the (private until submission) GitHub repository, which the PDF requires anyway [p.2].
   - Cloud notebooks `git clone` a specific commit or branch.
   - To change code: edit locally, push, re-clone. Never edit code on the cloud side; that is how the two copies drift.
3. **Getting code onto Kaggle:**
   - Preferred: enable notebook internet (this may require account verification; check your account) and `git clone`.
   - Fallback: zip `src/`, `configs/`, `scripts/`, `pyproject.toml`, `requirements.txt` and upload them as a private Kaggle Dataset (new version per code change).
4. **Getting data onto Kaggle/Colab:**
   - Run `scripts/prepare_pets.py` (and later `prepare_fs2k.py`) **locally**.
   - Upload only `data/cache/pets128/`, `data/splits/` and `data/manifests/` as a private Kaggle Dataset (Colab: copy them to Drive and then to `/content`).
   - This guarantees the cloud uses byte-identical splits and manifests, and it needs no internet. Verify the manifest sha256 at load time.
5. **Never upload:**
   - `.venv/`, `node_modules/`, `artifacts/`, `data/raw/`, `app/`;
   - `models/` and the official test cache, unless the job is the final evaluation.
6. **Bringing results back:**
   - Download the run directory (checkpoints, `metrics.jsonl`, samples) and `artifacts/optuna/<study>.db`.
   - Place them under `artifacts/runs/` and `artifacts/optuna/` locally.
   - Promote finals with `checkpoint.promote()`, which writes to `models/checkpoints/` and updates `MANIFEST.json`. Copy finished studies to `studies/`.
   - W&B runs sync by themselves when online; offline runs use `wandb sync`.
7. **Device choice per task:**
   - Task 1 is small, so attempt it locally first. Move it to the cloud only if the benchmark shows the budget does not fit.
   - Tasks 2 to 4 are the main users of Kaggle/Colab, following §5.

### 10.5 Agent instructions derived from this section
Coding agents must follow §3 (contracts), §6 (file ownership) and this section without further instruction. In particular, they must:
- implement §10.1 signatures and §10.3 resilience in every task;
- write notebooks per §10.2;
- keep the test split locked (§3.4);
- not launch Optuna studies or final training until the student approves budgets derived from the P2 benchmark.
