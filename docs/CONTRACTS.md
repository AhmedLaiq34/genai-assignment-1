# CONTRACTS (authoritative)

This file is the authoritative shared contract for all tasks and agents. **Only the orchestrator (A0) edits it.** Contract changes must also be recorded in `docs/DECISIONS.md`. Copied verbatim from `docs/IMPLEMENTATION_PLAN.md` section 3.

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
