# Task 4 plan: style-conditioned face-to-sketch cGAN

Planner output, 2026-10-04. Binding inputs: `docs/IMPLEMENTATION_PLAN.md` §3 (contracts) and §10 (workflow), PDF pages 2, 7 and 8 (`ASSIGNMENT_TRANSCRIPTION.md`), and `studies/task4_cgan/trials_from_console_log.csv`. Markers: **[PDF]** = assignment obligation, **[CON]** = repository contract (§3), **[REC]** = recommendation of this plan, **[OPT]** = optional.

All Task 4 code is new code in this repository. No Colab code exists. The Colab study supplies only (a) the best hyperparameters and (b) the study record rebuilt from the CSV.

**D numbers.** `docs/DECISIONS.md` already uses D1 to D50. D30 to D50 belong to Task 2. Task 4 therefore uses **D60 to D79**, which leaves D51 to D59 for the other windows. Decision C*n* below is written to the log as D(59+*n*).

---

## A. Specification checklist

| # | Obligation | Source | Planned file / function | Proving test or evidence |
|---|---|---|---|---|
| A1 | Official FS2K train/test definitions (`anno_train.json`, `anno_test.json`) | [PDF] p.7, [CON] §3.7 | `fs2k/pairs.py: load_annotations, resolve_pairs` | `test_fs2k.py::test_resolve_counts_real` (skipped without raw data): 1,058 train and 1,046 test pairs |
| A2 | Style taken from the annotations and encoded 0/1/2; UI "Style 1/2/3" maps to 0/1/2 | [CON] §3.7 | `pairs.py: style_of(entry)`, encoding recorded in `fs2k_split.json["style_encoding"]` | `test_style_values` (every style is in {0,1,2}); style counts printed by `prepare` and recorded as D60 |
| A3 | Every photo has a matching sketch (pairing validated) | [PDF] p.7, [CON] §3.7 | `pairs.py: resolve_pairs` fails and lists every unresolved entry | `test_resolve_pairs_fixture`, `test_missing_sketch_raises`; pair-audit grid looked at by the **student** |
| A4 | 15% stratified validation split from the official training portion, seed 42 | [PDF] p.7 | `fs2k/split.py: make_split` (`train_test_split(test_size=0.15, stratify=style, random_state=42)` on ids sorted first) | `test_split_counts` (899/159 on real data), `test_split_stratified`, `test_split_deterministic` (same sha256 twice), `test_split_disjoint` |
| A5 | The official test set stays locked during training and selection | [PDF] p.7 | `FS2KDataset(split="test")` raises unless `final_test=True`; that path appends a line to `artifacts/test_access.log` | `test_test_split_locked`; `tune.py` and `train.py` never pass `final_test` (grep check in the report) |
| A6 | Photos and sketches resized to 128×128 | [PDF] p.7 | `fs2k/prepare.py: load_photo, load_sketch` (EXIF transpose, RGB or L, bicubic) | `test_cache_shapes` |
| A7 | Tensor ranges: photo N×3×128×128 and sketch N×1×128×128, both [-1,1]; G ends in tanh | [CON] §3.7 | `fs2k/dataset.py: FS2KDataset.__getitem__`, `cgan.Generator` | `test_dataset_ranges`, `test_generator_shape_range` |
| A8 | Spatial augmentation identical for both images of a pair | [PDF] p.8 | `dataset.py: paired_augment` (photo and sketch are stacked into one 4-channel tensor, then cropped and flipped once) | `test_paired_transform_identical` (sketch = gray(photo) stays equal after augmentation over 50 draws) |
| A9 | U-Net generator G(x, s) | [PDF] p.7 | `models/cgan.py: Generator` | `test_generator_shape_range`, `test_skip_connections_used` |
| A10 | PatchGAN discriminator D(x, y, s) over photo–sketch pairs | [PDF] p.7 | `cgan.py: Discriminator` (70×70 receptive field, 14×14 logit map at 128) | `test_discriminator_patch_shape` |
| A11 | Learned categorical style embedding inside **both** G and D, not only a UI label | [PDF] p.7 | `Generator.style_embed`, `Discriminator.style_embed` (`nn.Embedding(3, style_dim)` each) | `test_style_changes_output` (G and D), `test_embedding_gradients` (non-zero grads in both tables after one step) |
| A12 | BCE with logits; L_G = L_adv + λ_L1·L1(y, G(x,s)) | [PDF] p.7–8 | `tasks/task4/train.py: d_step, g_step` | `test_losses_match_formula` (hand-computed on a tiny batch) |
| A13 | Separate logging of D real, D fake, G adversarial, G reconstruction and validation metrics | [PDF] p.8 | `train.py` writes `train/d_real, train/d_fake, train/g_adv, train/g_l1, val/l1, val/ssim, val/psnr, val/l1_style{0,1,2}` to `metrics.jsonl` and the tracker | `test_metrics_keys` (one epoch on CPU) |
| A14 | Samples logged at fixed intervals from the same validation photos | [PDF] p.8 | `train.py: fixed_sample_ids, save_sample_grid` (8 fixed val photos × 3 styles, every `sample_every_epochs`) | `test_fixed_samples_stable` (same ids before and after a resume) |
| A15 | Optuna tunes lr_G, lr_D, batch, base channels, dropout, embedding dim and λ_L1, with fewer epochs allowed | [PDF] p.8 | `task4/tune.py: run_study` (section B.1) | `test_study_dry_run` (2 trials × 1 epoch, tiny subset, CPU) |
| A16 | The selected configuration is retrained on the full schedule | [PDF] p.8 | `configs/task4_final.yaml` + `run_training` | the final run's `metrics.jsonl` (student run) |
| A17 | Only the generator is exported to ONNX (`photo`, `style` int64 → `sketch`) | [PDF] p.8, [CON] §3.9 | `export/task4_export.py: export_t4_generator` | `test_onnx_t4.py::test_parity_all_styles` (≤1e-4 max-abs, all 3 style ids) |
| A18 | Face-to-Sketch workspace: upload or webcam, Style 1/2/3, side by side, download | [PDF] p.8 | backend `/api/sketch` (this team); frontend in `app/frontend_v2` (other owner) | `tests/test_backend_sketch.py`; screenshot by the student |
| A19 | Experiment tracking of losses, hyperparameters, checkpoints and samples | [PDF] p.2, [CON] §3.12 | `common.tracking` (W&B, project `genai-a1`, group `task4`) | the W&B run page (student screenshot) |
| A20 | Checkpoints `t4_generator` and `t4_discriminator`, promoted with their sha256 | [CON] §3.8 | `train.py: split_checkpoint` + `common.checkpoint.promote` | `test_split_checkpoint_rebuilds` |

## B. Study records

### B.1 New Optuna study script (`src/genai/tasks/task4/tune.py`)
- **Signature:** `run_study(cfg, on_checkpoint=None) -> Path`. Resilience is the same as `task1/tune.py`:
  - SQLite storage with `load_if_exists=True`;
  - trials left RUNNING are marked FAIL at start-up;
  - FAIL trials do not use up the budget (D19);
  - after each trial the DB is copied to `persist_root` and `on_checkpoint(db)` is called;
  - `dry_run_overrides(cfg, tmp_root)`.
- **Config values:**
  - `study: task4_cgan` and `storage: artifacts/optuna/task4_cgan.db`;
  - `study_dir: studies/task4_cgan_repo` (C2: kept apart from the rebuilt log study in B.2);
  - `epochs_per_trial: 15`;
  - no `trial_train_subset` (full 899/159 split);
  - `n_trials: TBD_AFTER_BENCHMARK`;
  - `sampler: {name: TPE, seed: 42}`;
  - `pruner: {n_startup_trials: 5, n_warmup_steps: 5}` [REC]. The original pruner settings are unknown. With these values a trial can be pruned from epoch 5 of 15.
- **Search space (`tuned_params`, names exactly as in the log):**

| name | type | range | config target |
|---|---|---|---|
| `lr_g` | float, log | [1e-4, 4e-4] | `train.lr_g` |
| `lr_d` | float, log | [1e-4, 4e-4] | `train.lr_d` |
| `batch_size` | categorical | {8, 16, 32} | `train.batch_size` |
| `base_channels` | categorical | {32, 64} | `model.base_channels` (G and D) |
| `dropout` | float | [0.0, 0.5] | `model.dropout` (G decoder) |
| `style_dim` | categorical | {8, 16, 32} | `model.style_dim` (G and D) |
| `lambda_l1` | float, log | [10, 200] | `train.lambda_l1` |

  Checked against the CSV: every logged value lies inside these ranges (lr_g 1.05e-4 to 3.68e-4; lr_d 1.24e-4 to 3.67e-4; dropout 0.020 to 0.481; λ 11.4 to 199.7).
- **Objective:** the best (lowest) validation L1 over the trial's epochs. It is computed on sketches rescaled to **[0,1]** (`(x+1)/2`), as in §3.6. After each epoch the trial calls `trial.report(val_l1_epoch, step=epoch)` and then `trial.should_prune()`.
  - The scale of the Colab log value 0.0928 is unknown. New values and log values are therefore **not** compared as if they came from the same pipeline (see H3).
- **Console output (do not lower Optuna's logging).** Call `optuna.logging.set_verbosity(optuna.logging.INFO)` at the top of `run_study`. Create the study with exactly:
  ```python
  optuna.create_study(study_name=cfg["study"], storage=f"sqlite:///{db}", load_if_exists=True,
                      direction="minimize",
                      sampler=optuna.samplers.TPESampler(seed=seed + n_existing),
                      pruner=optuna.pruners.MedianPruner(n_startup_trials=..., n_warmup_steps=...))
  ```
  Optuna's own logger then prints:
  - "A new study created in RDB with name: task4_cgan", or on resume "Using an existing study with name 'task4_cgan' instead of creating a new one.";
  - one "Trial N finished with value: … and parameters: {…}. Best is trial M with value: …." per completed trial;
  - "Trial N pruned." per pruned trial.

  The trial runs' own prints go through `print` with a `[trial N]` prefix, so the Optuna lines stay readable.
- **End of study:** `print_study_summary(study, study_dir) -> Path` writes `best_params.json` and prints exactly:
  ```python
  print("=== Task 4 Optuna study complete ===")
  print(f"Trials run: {len(study.trials)}")
  print(f"Best trial: #{study.best_trial.number}  best val L1: {study.best_value:.4f}")
  print(f"Best params: {study.best_params}")
  print(f"Saved to {(study_dir / 'best_params.json').as_posix()}")
  ```
  - The `best_params.json` content is `{"study", "best_trial", "best_val_l1", "objective": "val L1 on [0,1] sketches", "params": {...}, "n_trials": {"COMPLETE": a, "PRUNED": b, "FAIL": c}, "source"}`.
  - `run_study` then calls `task1.tune.export_study(study, study_dir)` (imported, not copied), which writes `trials.csv` and three plots.
- **Trial runs** are written to `<output_root>/runs/task4_trials/<run_id>/`, so `resume="auto"` for the final run can never pick up a trial checkpoint. They save only at epoch ends and draw no sample grids. Each trial records the user attributes `run_id`, `best_epoch`, `val_ssim_at_best` and `seconds`.
- **Recommendation for `n_trials`:** 20, the same as the effective Colab study, but only if the benchmark shows it fits. The log suggests about 5 minutes per 15-epoch trial at batch 8 and 64 channels on the student's Colab GPU. That is about 100 minutes for 20 trials, an estimate that the benchmark must confirm.

### B.2 Study rebuilt from the console log (`src/genai/tasks/task4/log_study.py`)
- **Run with** `python -m genai.tasks.task4.log_study [--csv ...] [--out-db studies/task4_cgan/task4_cgan.db] [--overwrite]`.
- **Creates** study `task4_cgan` in its **own** file, `studies/task4_cgan/task4_cgan.db`.
  - It refuses to run if that file exists, unless `--overwrite` (which deletes and rebuilds the file). It can therefore never append duplicates.
  - No run of B.1 or B.3 ever uses this file as storage.
- **Trials are added in number order with `study.add_trial(optuna.trial.create_trial(...))`:**
  - **0 to 5: `FAIL`**, no params, user attribute `note = "crashed before training (filename bug) in the first Colab session; parameters not recorded"`. These keep Optuna's numbering equal to the log's (6 to 25) and reproduce "Trials run: 26". This rests on the student's recall and is labelled as such (C3).
  - **6 to 25 from the CSV:**
    - `COMPLETE` rows get `value=value_val_l1`, their params and distributions built from `configs/task4_cgan.yaml` `tuned_params`;
    - `PRUNED` rows get no params and no value;
    - every trial gets the user attributes `original_trial_number` and `source` (the CSV's `source` column).
  - Study user attributes: `source = "rebuilt from the Colab console log; original DB lost"` and `search_ranges = "repository ranges of B.1; the original Colab ranges are unknown"`.
- **Assertions before saving:**
  - 26 trials in total: 14 COMPLETE, 6 PRUNED, 6 FAIL;
  - the best trial is #25 with value 0.0928155106318572;
  - the best params equal the CSV row of trial 25.
- **Exports** go to `studies/task4_cgan/` through `task1.tune.export_study`: `trials.csv`, `optimization_history.png`, `param_importances.png` and `parallel_coordinate.png`.
- **Output:** the script prints the line `(study rebuilt from the Colab console log, not produced by tune.py)` and then calls `print_study_summary` from B.1. That writes `studies/task4_cgan/best_params.json`.
- No values are invented. Pruned trials have no intermediate values because the log has none.

### B.3 Confirmatory study with the new code [REC, optional if time runs out]
- **Command:** `scripts/tune.py --task t4` with `configs/task4_cgan.yaml`: 15 epochs per trial, full split, `n_trials` set from the measured benchmark (B.1 recommendation).
- **Outputs:** DB at `artifacts/optuna/task4_cgan.db`, promoted afterwards to `studies/task4_cgan_repo.db`; exports in `studies/task4_cgan_repo/`.
- **Rule:** the final training uses the **logged** best parameters (trial #25). The only exception is a "clear contradiction", defined in advance as **both** of these being true:
  - the confirmatory best differs in `batch_size`, `base_channels` or `style_dim`;
  - **and** retraining the logged #25 parameters for 15 epochs with the new code gives a val L1 worse than the confirmatory best by more than the spread between the confirmatory study's top 3 trials.

  If the rule triggers, `config.write_final_config` writes the new values and a DECISIONS row records why. Otherwise #25 stays.
- If B.3 is skipped, the report says that the repository study script was tested only in a dry run (H3).

### B.4 Technical notes (no report text)
- **The best values sit at the upper edges of the space:**
  - `lambda_l1`: 199.7 of 200 at #25, and the top three trials all have λ ≥ 173.8;
  - `style_dim`: 32 is the maximum, and the top 8 completed trials all use it;
  - `base_channels`: 64 is the maximum, used by 7 of the top 8;
  - `batch_size`: 8 is the minimum, and the top 8 all use it.

  The search may have been bounded too tightly on these parameters.
- **The objective favours a large λ_L1.** The objective is validation L1, and λ_L1 weights exactly that term in the generator loss, so a larger λ lowers validation L1 by construction. Validation L1 is therefore not a neutral judge of λ. Sketch realism (the adversarial part) is not measured by it.
- **The top trials are close.** Trials #25, #23 and #21 are 0.0928, 0.0934 and 0.0935: single runs of 15 epochs with no repeats, so the order of the top trials is not established.
- **Comparison plan:** if B.3 runs, compare its top 3 trials by val SSIM and by the fixed-photo grid (`[trial N]` sample grid at the last epoch, with `sample_every_epochs` forced to 15 for the top-3 retrain only). Otherwise only #25 is trained and no SSIM comparison is possible (the log has no SSIM).

## C. Decisions (the lead writes each one as D(59+n) in `docs/DECISIONS.md`)

1. **Code location.** All Task 4 code is written fresh under `src/genai/` and `scripts/`, and nothing is ported from Colab. Reason: the PDF (p.2) requires the code in the repository, and the Colab code no longer exists.
2. **Study names and files.** Study name `task4_cgan` everywhere, which replaces `t4_cgan` in §3.13. New runs are stored in `artifacts/optuna/task4_cgan.db` and promoted to `studies/task4_cgan_repo.db` + `studies/task4_cgan_repo/`. The rebuilt log study lives in `studies/task4_cgan/task4_cgan.db`. Reason: the study keeps the log's name, and the two records can never mix.
3. **Placeholder trials.** Trials 0 to 5 enter the rebuilt study as FAIL placeholders with no params. Reason: the trial numbers and "Trials run: 26" match the log; this rests on the student's recall and is labelled so.
4. **Data layout.**
   - Raw data in `data/raw/fs2k/` (the official download, unchanged, never uploaded).
   - Caches as 8-bit PNG named by `pair_id` (for example `photo1_image0001`):
     - `data/cache/fs2k128/{photo,sketch}/<pair_id>.png`, every pair resized directly to 128;
     - `data/cache/fs2k143/{photo,sketch}/<pair_id>.png`, official training portion only, resized to 143 for augmentation.
   - `data/splits/fs2k_split.json` contains `seed`, `val_fraction`, `train`/`val`/`test` id lists, `test_locked: true`, `style_encoding`, `style_counts` per split, a `pairs` map {id: {photo, sketch, style}} with raw relative paths, and `sha256` of the id lists.

   Reason: the training portion is resized only once, and validation, test and the app all use the same direct 128 resize.
5. **Preprocessing.** Photos: PIL → `exif_transpose` → `RGB` → `resize((128,128), BICUBIC)`. Sketches: the same but `L`. Reason: identical to §3.1 and to the backend's `preprocess_bytes`, so the app sees the same input as validation (unit test).
6. **Generator (U-Net, base c, style dim d, bottleneck at 4×4).**
   - **Input:** photo (3 channels) concatenated with the style embedding tiled to 128×128 (d channels).
   - **Encoder:** Conv4×4/s2 blocks giving 64:c (no norm), 32:2c, 16:4c, 8:8c, 4:8c, with InstanceNorm2d(affine) and LeakyReLU(0.2).
   - **Bottleneck:** the style embedding is tiled to 4×4 again and concatenated, followed by Conv3×3/s1 → 8c with InstanceNorm and ReLU.
   - **Decoder:** ConvTranspose4×4/s2 blocks, each concatenated with the matching encoder feature (skip connection):
     - 8:8c, 16:4c and 32:2c with InstanceNorm, Dropout(p = `dropout`) and ReLU;
     - 64:c with InstanceNorm and ReLU, without Dropout;
     - output layer: ConvTranspose 2c→1 to 128, then `tanh`.

   Reason: this is pix2pix shrunk to 128. Feeding the style at the input as well means the skip paths also carry it, so the decoder cannot ignore it. Stopping at 4×4 avoids InstanceNorm on 1×1 maps.
7. **Discriminator (PatchGAN).**
   - **Input:** photo (3) + sketch (1) + style embedding tiled to 128×128 (d), with its **own** `nn.Embedding(3, d)`.
   - **Layers:** Conv4×4/s2 c (no norm), then s2 2c, s2 4c, s1 8c (InstanceNorm, LeakyReLU 0.2), then Conv4×4/s1 → 1, which gives a 14×14 logit map and a 70×70 receptive field.
   - No dropout in D.

   Reason: the standard 70×70 PatchGAN, and D(x, y, s) as the PDF writes it.
8. **ONNX safety.**
   - Only Conv, ConvTranspose, InstanceNorm, LeakyReLU/ReLU, Tanh, Concat, Gather (embedding) and Expand are used. Tiling is `emb[:, :, None, None].expand(-1, -1, H, W)`.
   - Dropout is a no-op in `eval()`.
   - No BatchNorm, so train and eval behave the same apart from dropout.

   Reason: every one of these ops exists in opset 17. The P2 random-weight export test catches anything missed.
9. **Weights and losses.**
   - Weights are initialised from N(0, 0.02).
   - Losses:
     - `d_real = BCEWithLogits(D(x,y,s), 1)` and `d_fake = BCEWithLogits(D(x,G(x,s).detach(),s), 0)`, with `L_D = 0.5·(d_real + d_fake)`;
     - `g_adv = BCEWithLogits(D(x,G(x,s),s), 1)` and `g_l1 = mean|y − G(x,s)|` on [-1,1], with `L_G = g_adv + lambda_l1·g_l1`.
   - Optimisers: two Adam optimisers with betas (0.5, 0.999), learning rates `lr_g` and `lr_d`.
   - Order per batch: one D step, then one G step.

   Reason: the PDF formula exactly, plus the pix2pix conventions that the λ = 100 hint assumes.
10. **Learning-rate schedule.** Trials use a constant learning rate. The final run uses a constant rate for the first half of the epochs, then linear decay to 0 (`train.schedule: linear_decay_half`). Reason: the pix2pix schedule, and the only difference from the trials (stated in the report).
11. **AMP.** `train.amp: false` by default; it may be turned on only if the benchmark shows a gain and a 1-epoch AMP smoke run gives finite losses. With AMP on there are two GradScalers. Reason: GAN stability comes before speed.
12. **Augmentation.** Training uses resize 143 → the same random 128 crop → the same horizontal flip with p = 0.5, applied by stacking photo and sketch into one 4-channel tensor. Validation and test use no augmentation. Reason: identical by construction (PDF p.8), and easy to explain.
13. **Metrics.**
    - Per epoch on validation: L1, SSIM and PSNR on [0,1] (`common.metrics`, sketch with 1 channel), each also split by style. The trial and model-selection objective is val L1.
    - Training diagnostics: the mean of `sigmoid(D)` on real and on fake pairs.
    - FID and LPIPS are [OPT] and not planned.
14. **Checkpoints.**
    - `ckpt_last.pt` and `ckpt_best.pt` follow the §3.8 dict:
      - `model` = G state, `optimizer` = Adam_G, `scheduler` = sched_G, `scaler` = scaler_G or None;
      - plus `discriminator`, `optimizer_d`, `scheduler_d`, `scaler_d`, `rng`, `step_in_epoch` and `fixed_sample_ids`.
    - `config` contains the full config, including `model: {base_channels, style_dim, dropout, num_styles: 3}`.
    - `split_sha256` is the sha256 of `fs2k_split.json`; `manifest_sha256` is None (FS2K has no manifest).
    - `split_checkpoint(ckpt, out_dir)` writes `t4_generator.pt` (G only, with optimizer and similar entries set to None) and `t4_discriminator.pt` (`model` = D state, with the same config). Each is promoted with `promote(path, "t4_generator")` and `promote(path, "t4_discriminator")`.

    Reason: §3.8 names, and each file rebuilds with `Generator.from_config(ckpt["config"]["model"])`.
15. **Checkpoint cadence.** `ckpt_last.pt` is written atomically every `checkpoint_every_minutes: 10` [REC] and at each epoch end. `ckpt_best.pt` is written whenever val L1 improves. Optional `snapshot_every_epochs: 50` [REC] keeps `ckpt_epoch050.pt` and so on for a later visual comparison. Reason: a reset loses at most 10 minutes; snapshots allow a visual check of GAN checkpoints that L1 alone cannot judge.
16. **Best checkpoint.** The lowest val L1 on [0,1], the same objective as the study (§3.6). Val SSIM is reported next to it. Reason: one fixed rule, chosen before the run.
17. **Final schedule.** `train.epochs` stays `TBD_AFTER_BENCHMARK` in `configs/task4_final.yaml`; the student's intended value is **200**, on the same 899/159 split.
    - Rule: measure s/epoch (train + val) with `scripts/benchmark.py --model t4` on the target GPU, then set 200 only if 200 × s/epoch × 1.1 fits within one session. Otherwise run it across sessions with resume (C19).
    - The log suggests about 20 s/epoch on the student's Colab GPU (about 1 hour for 200 epochs). This is an estimate to replace with the measurement.
18. **Sample grids.**
    - Photos: 8 fixed val photos, chosen as the first 3, 3 and 2 pair ids of styles 0, 1 and 2 in sorted order, and stored in the checkpoint.
    - Grid row: photo | real sketch | G(x,0) | G(x,1) | G(x,2).
    - Cadence: at epoch 1 and then every `sample_every_epochs: 5` [REC], saved to `samples/epochNNN.png` and logged to the tracker.

    Reason: the PDF's "same validation photographs at fixed intervals", and the three styles side by side.
19. **Tracking and devices.**
    - Tracking: `TRACKER=wandb`, project `genai-a1`, group `task4`, run name = `run_id`, entity from `WANDB_ENTITY` (read by wandb itself). Each study trial is its own run in the same group, named `...task4_cgan-trialN`. Offline mode on devices without internet.
    - Devices:
      - the final training and the optional confirmatory study run on **Kaggle** (T4 GPU) through one unattended pipeline notebook, as for Task 2 (D44);
      - the laptop does data preparation, tests, smoke runs, the local benchmark, export, evaluation and the single test-set evaluation;
      - Colab is the fallback.
    - Resume across resets: `resume="auto"` searches `persist_root`, `output_root` and every folder in `cfg["resume_roots"]` (the previous Kaggle version's output, attached as an input). It continues `runs/task4/<run_id>/`.
20. **Ownership of the run folders.** `runs/task4/` holds only final and smoke runs, and `runs/task4_trials/` holds study trials. Reason: `resume="auto"` must never resume a trial.

## D. Module specification

Owners (section I): **L** = lead, **A** = data, **B** = model and training, **C** = study, **D** = evaluation and ONNX, **E** = backend.

**`src/genai/fs2k/pairs.py` (A)**
- `load_annotations(path) -> list[dict]` reads the JSON and returns the raw entries.
- `style_of(entry) -> int` reads the style field and maps it to 0/1/2. The field name and encoding are verified against the real file and recorded; any other value raises.
- `pair_id(entry) -> str`, for example `photo1/image0001` → `photo1_image0001`.
- `resolve_pairs(raw_root, anno_json) -> list[dict]`:
  - returns `{"id", "photo", "sketch", "style"}` with paths relative to `raw_root`;
  - finds the sketch by the photo→sketch naming rule of the download and tries the extensions `.jpg`, `.jpeg` and `.png`;
  - raises `FileNotFoundError` listing **every** unresolved entry.
- The naming rule is confirmed on the real download before coding against it.

**`src/genai/fs2k/split.py` (A)**
- `make_split(train_pairs, test_pairs, seed=42, val_fraction=0.15) -> dict`: sorts the ids, runs `train_test_split(stratify=styles, random_state=seed)` and builds the split dict of C4 (including `sha256`).
- `save_split(split, path)` and `load_split(path) -> dict`. `load_split` checks the sha256.

**`src/genai/fs2k/prepare.py` (A)**
- `load_photo(path, size) -> np.uint8 HxWx3` and `load_sketch(path, size) -> np.uint8 HxW` (C5).
- `build_cache(raw_root, split, cache_root)` writes `fs2k128` (all pairs) and `fs2k143` (official training pairs).
- `save_pair_audit_grid(cache_root, split, out_png, per_style=8)`: 24 rows of photo | sketch | "style k", written to `report/figures/task4/fs2k_pair_audit.png`.
- `package(data_root, out_zip, include_test=False)`: writes `dist/cloud_data/fs2k_data.zip` with `splits/fs2k_split.json` and the train/val images of both caches. Test images go in only with `include_test`.
- `export_app_samples(cache_root, split, out_dir, n=6)`: only if the student answers yes to Q3.
- `main(argv)`: the steps in order, printing pair counts and style counts per split.

**`src/genai/fs2k/dataset.py` (A)**
- `resolve_fs2k_paths(data_root) -> dict`: the profile name or a folder, then finds `splits/fs2k_split.json` with rglob (the Kaggle nesting of D46). Returns `{split_file, cache128, cache143}`.
- `paired_augment(photo, sketch, crop=128) -> tuple`: inputs are tensors at 143. Photo and sketch are concatenated into a 4×143×143 tensor, which gets one `torch.randint` crop and one flip with p = 0.5.
- `FS2KDataset(split: "train"|"val"|"test", data_root, augment: bool, final_test=False, subset: int|None=None)`:
  - `__getitem__` returns `(photo f32 3×128×128 [-1,1], sketch f32 1×128×128 [-1,1], style int64, pair_id)`;
  - the train split uses the 143 cache with `paired_augment` when `augment`, otherwise the 128 cache; val and test use the 128 cache;
  - `split="test"` without `final_test=True` raises; with it, a line is appended to `artifacts/test_access.log` (use the pets helper if it is public, otherwise the same line format).

**`configs/data_fs2k.yaml` (A):** the existing file, plus `cache143_dir`, `audit_grid` and `package_out`.

**`src/genai/models/cgan.py` (B)**
- `Generator(base_channels: int, style_dim: int, dropout: float, num_styles: int = 3)`:
  - `forward(photo: N×3×128×128 [-1,1], style: int64 N) -> N×1×128×128 [-1,1]` (C6);
  - `from_config(model_cfg)`.
- `Discriminator(base_channels: int, style_dim: int, num_styles: int = 3)`:
  - `forward(photo, sketch: N×1×128×128, style) -> logits N×1×14×14` (C7);
  - `from_config(model_cfg)`.
- `init_weights(module)` (C9).

**`src/genai/tasks/task4/config.py` (L, written first)**
- Re-exports `load_config`, `set_dotted`, `resolve`, `is_tbd`, `make_run_id` and `deep_copy` from `task1.config` (import only).
- `runs_dir(cfg, root_key="output_root", kind="task4")` returns `<root>/runs/task4` or `/runs/task4_trials`.
- `find_latest_checkpoint(cfg)` covers `persist_root`, `output_root` and `cfg.get("resume_roots", [])`, and skips `_smoke` runs unless `run.smoke`.
- `PARAM_TARGETS` holds the table of B.1.
- `write_final_config(best_params_json, base_config="configs/task4_cgan.yaml", out_path="configs/task4_final.yaml")` copies the params into a full training config and keeps `train.epochs` TBD.

**`src/genai/tasks/task4/train.py` (B)**
- `run_training(cfg, resume=None, on_checkpoint=None) -> Path` calls `_train(cfg, resume, on_checkpoint, report_fn=None) -> (run_dir, best_val_l1)`, the same split as Task 1, so `tune.py` reuses `_train`.
- Helpers:
  - `d_step(G, D, opt_d, batch, scaler) -> dict` and `g_step(G, D, opt_g, batch, lambda_l1, scaler) -> dict`;
  - `validate(G, loader, device) -> dict` with `l1, ssim, psnr, l1_style0..2`;
  - `fixed_sample_ids(val_ds) -> list` and `save_sample_grid(G, val_ds, ids, device, path) -> Tensor`;
  - `split_checkpoint(ckpt_path, out_dir) -> (g_path, d_path)`.
- Full resume (both models, both optimisers, schedulers and scalers, epoch, `step_in_epoch`, `global_step`, `best_metric`, RNG); `_MUST_MATCH = ("lr_g", "lr_d", "batch_size", "lambda_l1", "amp", "schedule")`.
- Smoke support: `run.smoke`, `train.max_steps`, `train.train_subset`, `train.val_subset`.
- Config keys:
  - `train.{epochs, batch_size, lr_g, lr_d, lambda_l1, betas, schedule, amp, checkpoint_every_minutes, sample_every_epochs, snapshot_every_epochs, log_every_steps, augment}`;
  - `model.{base_channels, style_dim, dropout, num_styles}`;
  - `run.{project, desc, smoke}`.
- Tracker group: `task4`.

**`configs/task4_final.yaml` (B)**
- `task: t4`, `seed: 42`, `data_config: configs/data_fs2k.yaml`.
- `model: {base_channels: 64, style_dim: 32, dropout: 0.29338252123610636, num_styles: 3}`.
- `train: {batch_size: 8, lr_g: 0.00023138114869747655, lr_d: 0.000349228952295573, lambda_l1: 199.71014173562637, betas: [0.5, 0.999], schedule: linear_decay_half, amp: false, epochs: TBD_AFTER_BENCHMARK, ...}`, with the comment `# student's intended value: 200`.
- `source: "Colab study task4_cgan, best trial #25, val L1 0.0928 (scale unknown)"`.

**`src/genai/tasks/task4/tune.py` (C):** `run_study`, `suggest_params`, `make_trial_config`, `print_study_summary` and `dry_run_overrides`, as in B.1.

**`configs/task4_cgan.yaml` (C):** replaces the structure-only file. It holds `task: t4`, `study`, `storage`, `study_dir`, `n_trials: TBD_AFTER_BENCHMARK`, `epochs_per_trial: 15`, `sampler`, `pruner`, `tuned_params` (B.1, with `type`/`low`/`high`/`log`/`choices` as in `task1_universal.yaml`), and the base `model`/`train` sections (`schedule: constant`, `augment: true`, `amp: false`).

**`src/genai/tasks/task4/log_study.py` (C):** `build_log_study(csv_path, out_db, config_path, overwrite=False) -> optuna.Study` and `main(argv)`, as in B.2.

**`src/genai/tasks/task4/evaluate.py` (D)**
- `run_evaluation(cfg, checkpoint, final_test=False) -> Path`:
  - evaluates val by default and test only with `final_test`;
  - writes to `<output_root>/eval/task4/<timestamp>_{val|test}/`;
  - outputs:
    - `per_image.csv` (`pair_id, style, l1, ssim, psnr`), `by_style.csv` and `summary.json`;
    - `results_grid.png` (8 photos per style: photo | real | generated);
    - `failures_grid.png` (the 8 worst L1);
    - `style_variations.png` (8 photos × the three styles, qualitative, no ground truth for the other styles).

**`src/genai/export/task4_export.py` (D)**
- `load_t4_generator(ckpt_path) -> (G, ckpt)` accepts a run checkpoint or a promoted `t4_generator.pt`.
- `export_t4_generator(name, ckpt_path, out_path)`: opset 17, `dynamo=False`, inputs `photo` (f32 N×3×128×128) and `style` (int64 N), output `sketch`, dynamic axis 0 on all three. It writes a `.meta.json` like Task 1's, with the `smoke` flag.
- `verify_t4_parity(name, ckpt_path, onnx_path, n=16, inputs=None, tag="", csv_path=None, data_root=None) -> dict`:
  - 16 val photos, each run with style ids 0, 1 and 2 (48 cases);
  - tolerance `ONNX_PARITY_TOL_MAX_ABS`;
  - one row appended to `report/tables/onnx_parity.csv` using `onnx_verify.FIELDS`;
  - without `inputs`, `data_root` is required (D46 lesson).

**`scripts/prepare_fs2k.py` (L):** a thin wrapper over `genai.fs2k.prepare.main` with `--config`, `--device-profile`, `--package`, `--include-test` and `--app-samples`.

**`scripts/benchmark.py`, `export_onnx.py`, `evaluate.py`, `tune.py` (L, additive)**
- `bench_t4`: one D+G step at the given batch sizes with base 32 and 64 and AMP on/off. It reports s/step, peak memory and val-pass time, and derives s/epoch as `ceil(899/B)·s/step + val`.
- `export_onnx.py` dispatches `t4_generator` to `task4_export`.
- `tune.py --dry-run` must reach `task4.tune.dry_run_overrides`.

**Notebook and tools (L)**
- `notebooks/kaggle_t4_pipeline.ipynb` and `tools/t4_pipeline.py` follow the pattern of `tools/t2_pipeline.py`. Steps: data check → benchmark → [optional confirmatory study] → final training (`resume="auto"`, `resume_roots`) → `split_checkpoint` → ONNX export + parity → val evaluation → `t4_results.zip`.
- `tools/t4_report_assets.py` produces section G.
- `docs/KAGGLE_T4_STEPS.md` holds the Kaggle steps.

**Backend (E): `app/backend/app/sketch.py`, plus a hookup in `main.py` that replaces the 501 stub**
- `POST /api/sketch`, multipart:
  - `file` (an image) **or** `sample_id`;
  - `style` ∈ {1,2,3} (form field; anything else gives 422).
- Path: the same upload validation and `preprocess_bytes` as `/api/universal`; `photo = img01*2-1`; the ONNX session `t4_generator` with `style = int64([style-1])`; `sketch01 = (out+1)/2`; then a grayscale PNG.
- Response: `{photo_png_b64, sketch_png_b64, style_id: 0..2, style_label: "Style k", timing_ms: {preprocess, inference, total}}`.
- Errors: a missing model gives 503 for this endpoint only, and a bad upload gives 4xx as now.
- `GET /api/sketch/samples` returns `[{id, url}]`, and `GET /api/sketch/samples/{id}` returns the PNG. The sample folder is the Q3 answer; without samples, the list is empty and the server still starts.
- `/api/health` already lists `t4_generator` through `ONNX_FILES`.

**Frontend needs (for the `app/frontend_v2` owner; nothing is built by this team)**
- The API contract above.
- Webcam frames are sent as `file` (PNG or JPEG blob).
- The page shows `photo_png_b64` (the 128 preprocessed input) and `sketch_png_b64` side by side, with `timing_ms`.
- Download: `sketch_png_b64` saved as `sketch_style<k>.png`.
- Optional: three calls with style 1/2/3 to show all styles of one photo.

## E. Verification plan

**Unit tests (CPU, `CUDA_VISIBLE_DEVICES=""`).** Synthetic data comes from a mini-FS2K fixture that a test helper writes, with 3 styles and about 40 pairs. Real-data tests are marked and skipped when `data/raw/fs2k` is missing.
- **`tests/test_fs2k.py` (A):**
  - resolve on the fixture, and a missing sketch raises;
  - style values are in {0,1,2};
  - **real data:** 1,058 train and 1,046 test pairs, a split of 899/159, and per-style validation counts within ±1 of 15% of each style's count;
  - the split is deterministic (equal sha256), with no overlap between train, val and test;
  - a test read without `final_test` raises; with it, exactly one log line is added (in a tmp log path);
  - `paired_augment` keeps sketch = gray(photo) over 50 draws;
  - ranges are [-1,1] and shapes are exact;
  - `load_photo` equals the backend's `preprocess_bytes` on the same file (within 1/255).
- **`tests/test_cgan.py` (B):**
  - shapes of G and D at batch 1 and 3, G output in [-1,1], and a D map of 14×14;
  - G output differs for styles 0, 1 and 2 at fixed weights; D logits differ by style;
  - after one `g_step` and one `d_step`, `G.style_embed.weight.grad` and `D.style_embed.weight.grad` are non-zero;
  - zeroing the skip features changes the output (the skips are wired);
  - `from_config` round-trip.
- **`tests/test_task4_train.py` (B):**
  - loss formula on a hand batch;
  - one-step CPU training produces finite values for all four losses;
  - 2-epoch tiny run → `metrics.jsonl` has all keys of A13, and `samples/` has the epoch-1 grid;
  - resume from the epoch-1 `ckpt_last.pt` continues `global_step` and keeps `fixed_sample_ids`, with both optimiser states restored;
  - `split_checkpoint` → both files rebuild their model;
  - `promote` into a tmp `models/` with a bare-list MANIFEST.
- **`tests/test_task4_tune.py` (C):**
  - dry run (2 trials × 1 epoch, tiny subset, tmp storage) writes the DB, `trials.csv` and `best_params.json`, and prints the block of B.1 (capsys);
  - a rerun prints "Using an existing study" (caplog);
  - a RUNNING trial is marked FAIL on restart;
  - `build_log_study`: 26/14/6/6, best #25 with 0.0928155106318572 and the exact params; it refuses an existing DB without `--overwrite`;
  - every CSV value lies inside the `tuned_params` ranges.
- **`tests/test_onnx_t4.py` (D):**
  - export of a random fixture G (under `artifacts/fixtures/task4/`, never `models/onnx/`, `smoke: true`);
  - ORT parity max-abs ≤ 1e-4 for each style id 0, 1 and 2, and at batch sizes 1 and 5 (dynamic axis).
- **`tests/test_task4_eval.py` (D):** val evaluation on the fixture writes all files of section D; a call with `final_test=True` logs the access.
- **`tests/test_backend_sketch.py` (E):**
  - a tiny random ONNX with the I/O names of §3.9 in a tmp model dir;
  - 200 with base64 PNGs, `style_id` = style − 1, a grayscale 128×128 sketch;
  - style 0 or 4 gives 422; a missing model gives 503 while `/api/health` still answers; a bad upload gives 4xx;
  - `/api/universal` and `/api/hard` tests still pass.
- **Full suite:** `pytest tests -q` (189 passed at D49; that number has to be checked again).

**The student looks at:** `report/figures/task4/fs2k_pair_audit.png` (24 rows: each sketch belongs to its photo, and the style labels look consistent). This is the P1b gate.

**Gates before the final training:**
1. All tests pass.
2. `prepare_fs2k` reports 1,058/1,046 pairs and 899/159, with style counts written to D60.
3. The student has approved the audit grid.
4. Smoke run: about 50 steps on a subset, then `ckpt_last` → resume → `split_checkpoint` → ONNX export + parity (fixture folder) → `/api/sketch` returns a sketch.
5. The random-weight ONNX export passes (P2.2).
6. `bench_t4` has been measured on the target GPU and written to `docs/BENCHMARKS.md`.
7. The student has set `train.epochs` (200 if C17's rule allows) and confirmed W&B login.
8. A pipeline dry run with a fake Kaggle folder layout (as D46) has passed.

## F. Execution runbook for the student

1. **Get FS2K.** Download the official FS2K release (link in the FS2K GitHub README) and extract it to `data/raw/fs2k/`, so that `anno_train.json`, `anno_test.json`, `photo/` and `sketch/` sit inside it. Never commit or upload it.
2. **Prepare.**
   - Run `python scripts/prepare_fs2k.py`. It prints the pair counts, the split and the style counts.
   - Open `report/figures/task4/fs2k_pair_audit.png` and confirm it.
   - Run `pytest tests -q`.
3. **Rebuild the log study.** Run `python -m genai.tasks.task4.log_study`, then check `studies/task4_cgan/` (DB, CSV, three plots, `best_params.json`).
4. **Smoke and local benchmark.**
   - Smoke run: `python scripts/train.py --task t4 --config configs/task4_final.yaml --set run.smoke=true --set train.max_steps=50 --set train.epochs=1`.
   - Benchmark: `python scripts/benchmark.py --model t4 --batch 8 16 32 --amp both`.
5. **Package for Kaggle.**
   - Run `python scripts/prepare_fs2k.py --package`, which writes `dist/cloud_data/fs2k_data.zip` without test images.
   - Push the code, or zip `src/ configs/ scripts/ tools/ pyproject.toml requirements.txt` as a **new** Kaggle dataset (D46).
   - **Never upload:** `data/raw/`, test images, `artifacts/`, `models/`, `app/`, `.venv/`, `node_modules/`.
6. **Kaggle run** (`notebooks/kaggle_t4_pipeline.ipynb`, steps in `docs/KAGGLE_T4_STEPS.md`; W&B key as a Kaggle secret, `TRACKER=wandb`):
   1. Benchmark step first.
   2. Set `train.epochs` from it (C17).
   3. Optionally run the confirmatory study (B.3).
   4. Final training with "Save & Run All (Commit)".
   - After a reset, start a new version, attach the previous version's output as an input, and keep `RESUME="auto"`. `resume_roots` then finds `ckpt_last.pt` and the run continues in the same `run_id` folder.
7. **Bring results back.**
   - Download `t4_results.zip`: the run folder (ckpt_best, ckpt_last, metrics.jsonl, samples, snapshots), the ONNX file and its meta, the val evaluation, the benchmark rows and the study DB if run.
   - Unpack it under `artifacts/runs/task4/`, `artifacts/optuna/` and `studies/`. Merge `onnx_parity.csv` and `MANIFEST.json`; do not overwrite them (D48).
8. **Promote and export.**
   - Promote: `python -c "from genai.tasks.task4.train import split_checkpoint; from genai.common.checkpoint import promote; g,d = split_checkpoint('artifacts/runs/task4/<run_id>/ckpt_best.pt','artifacts/promote_t4'); promote(g,'t4_generator'); promote(d,'t4_discriminator')"`.
   - Export: `python scripts/export_onnx.py --model t4_generator --ckpt models/checkpoints/t4_generator.pt --verify`.
9. **Evaluate on validation:** `python scripts/evaluate.py --task t4 --ckpt models/checkpoints/t4_generator.pt`. Run `tools/t4_report_assets.py` afterwards.
10. **Evaluate on test, once, at the very end with the config locked:** `python scripts/evaluate.py --task t4 --ckpt models/checkpoints/t4_generator.pt --final-test`. This needs the local test cache and is logged in `artifacts/test_access.log`.
11. **Application.** Run `docker compose up --build`, open the Face-to-Sketch workspace, try Style 1/2/3 and webcam, and take screenshots.

**Three styles for one photo:** shown qualitatively in `style_variations.png` from evaluation (val, and test at the end) and in the training sample grids, and live in the app by switching the style. There is no ground truth for the styles other than the photo's own, so no metric is computed for them.

## G. Report evidence checklist (`tools/t4_report_assets.py`, L)

| Evidence | File |
|---|---|
| Four loss curves (D real, D fake, G adv, G L1), plus mean sigmoid(D) real/fake | `report/figures/task4/losses.png` |
| Validation L1 / SSIM / PSNR per epoch, with the best epoch marked; per style | `report/figures/task4/val_metrics.png`, `report/tables/task4/val_by_style.csv` |
| Sample timeline: the fixed 8 photos at selected epochs (for example 1, 25%, 50%, 100% of the run) | `report/figures/task4/sample_timeline.png` |
| Final results per style (test, after `--final-test`; val until then) | `report/figures/task4/results_style{1,2,3}.png`, `report/tables/task4/test_by_style.csv` |
| One photo in all three styles | `report/figures/task4/style_variations.png` |
| Failure cases (worst L1) | `report/figures/task4/failures.png` |
| Pair-audit grid | `report/figures/task4/fs2k_pair_audit.png` |
| Split and style counts | `report/tables/task4/fs2k_split_counts.csv` |
| Colab study (rebuilt): trials table and three plots, plus the COMPLETE / PRUNED / FAIL counts | `studies/task4_cgan/*`, copied to `report/figures/task4/study_log_*.png`, `report/tables/task4/study_log_trials.csv` |
| Confirmatory study, if run | `studies/task4_cgan_repo/*` → `report/.../study_repo_*` |
| ONNX parity rows (3 styles) | `report/tables/onnx_parity.csv` |
| Benchmark rows | `docs/BENCHMARKS.md` |
| App screenshots (upload, webcam, 3 styles, download) and W&B run page | `report/figures/task4/app_*.png`, `wandb_*.png` (student) |

## H. Risks and open questions (only those that change what the student does)

1. **GAN instability or collapse.**
   - Signs: `d_fake` heading to 0 while `g_adv` keeps rising, sigmoid(D) on fake near 0 for many epochs, or identical sketches for all three styles in the grids.
   - Action: watch the W&B curves and the grids in the first hour. A resume does not fix a collapse, so restart from the last good snapshot with the same parameters. Report it.
2. **Style might be ignored.** With a large λ_L1 and skip connections, G can produce one average style.
   - Check: the `style_variations` grid and per-style val L1.
   - If all styles look the same, report it honestly; C6's input conditioning is the counter-measure already in the design.
3. **Comparability with the log.** The 0.0928 has an unknown L1 scale and comes from lost code.
   - Report the repository's own val L1 as the result.
   - Quote the log study only as the source of the hyperparameters.
   - Never present `studies/task4_cgan/task4_cgan.db` as output of `tune.py`.
4. **Small dataset.** 899 training pairs and 200 epochs mean over-fitting is likely, which is why best-by-val-L1 and snapshots exist. 159 validation pairs make per-style numbers noisy (one style may have few validation pairs), so report the counts.
5. **ONNX.** InstanceNorm, Gather and Expand are expected to export in opset 17, but this is not verified until the P2.2 random-weight export. If an op fails, replace that construction (for example the tiling with `repeat`) before any training.
6. **Free-tier limits.** The Kaggle session length and weekly GPU quota are shown in the account. The run fits only if the measured s/epoch allows it (C17), and resume (C19) covers resets.
7. **Not verifiable by the planner:**
   - the FS2K annotation field names, style encoding and file-naming rule (checked by A at step 1);
   - Kaggle T4 speed;
   - whether the second-half learning-rate decay helps (C10, a recommendation).

**Questions for the student (they change the plan):**
- **Q1.** Is Task 1 training, Docker (the D50 stack) or anything else using the laptop GPU right now? The worker asks this first.
- **Q2.** Should the confirmatory study (B.3) run? If yes, how much Kaggle GPU time may it use? Without an answer, the plan skips it and trains #25.
- **Q3.** May about 6 FS2K **test** photos (128 px) be stored in the repository as display-only app samples? Without an answer: no samples; upload and webcam only.
- **Q4.** Is the official FS2K download already somewhere on this laptop? `data/raw/fs2k` does not exist.

## I. Agents, ownership and order (Sonnet team)

| Agent | Owns | Returns |
|---|---|---|
| **L (lead)** | `src/genai/tasks/task4/{__init__,config}.py`; additive edits to `scripts/{prepare_fs2k,benchmark,export_onnx,evaluate,train,tune}.py` and `src/genai/tasks/cli.py` (only if needed); `notebooks/kaggle_t4_pipeline.ipynb` and additive cells in `notebooks/kaggle_train.ipynb`/`colab_train.ipynb`; `tools/t4_pipeline.py`, `tools/t4_report_assets.py`; `docs/KAGGLE_T4_STEPS.md`, `docs/PHASE_T4_REPORT.md`; appended rows in `docs/DECISIONS.md` (D60 to D79), `docs/AI_USE_LOG.md`, `docs/BENCHMARKS.md`, `docs/TRACEABILITY.md` (T4 rows) | integration and the final report |
| **A (data)** | `src/genai/fs2k/{pairs,split,dataset,prepare}.py`, `configs/data_fs2k.yaml`, `tests/test_fs2k.py`, the mini-FS2K fixture helper `tests/fs2k_fixture.py` | the dataset interface, the confirmed FS2K naming and style encoding, counts |
| **B (model + training)** | `src/genai/models/cgan.py`, `src/genai/tasks/task4/train.py`, `configs/task4_final.yaml`, `tests/test_cgan.py`, `tests/test_task4_train.py` | the `_train` signature, checkpoint layout, `split_checkpoint` |
| **C (study)** | `src/genai/tasks/task4/{tune,log_study}.py`, `configs/task4_cgan.yaml`, `tests/test_task4_tune.py`, generated `studies/task4_cgan/*` | dry-run output, the rebuilt study summary |
| **D (evaluation + ONNX)** | `src/genai/tasks/task4/evaluate.py`, `src/genai/export/task4_export.py`, `tests/test_task4_eval.py`, `tests/test_onnx_t4.py` | the parity result on fixtures and the evaluation folder layout |
| **E (backend)** | `app/backend/app/sketch.py`, the `/api/sketch` hookup lines in `app/backend/app/main.py` (replacing only the 501 stub), `tests/test_backend_sketch.py` | the final API contract for the frontend_v2 owner |

**Concurrency rules:**
- Do not edit:
  - Task 1 or Task 2 files (`src/genai/tasks/task{1,2}/**`, `src/genai/models/{autoencoder,classifier}.py`, `configs/task{1,2}_*`, `src/genai/export/{onnx_export,onnx_verify,task2_export}.py`, `app/backend/app/hard.py`, `tools/t1_*`, `tools/t2_*`);
  - `src/genai/common/**`, `src/genai/pets/**`;
  - `app/frontend/**`, `app/frontend_v2/**`;
  - `docs/IMPLEMENTATION_PLAN.md`, `docs/CONTRACTS.md`.

  Import from them instead. A genuinely needed change becomes a DECISIONS row with status "question".
- `src/genai/tasks/cli.py`, `scripts/*.py` and the notebooks are changed only by L, and only additively. The `t4` key already exists.

**Order:**
1. L writes `task4/config.py` and the `__init__`.
2. **Wave 1 in parallel:** A, B (against the dataset interface in D, using its own random tensors until A delivers) and E (against a random ONNX with the §3.9 names).
3. **Wave 2:** C (needs B's `_train`), and D (needs B's model and A's dataset; it can start on fixtures).
4. L's script, notebook and tool additions; the integration checks of section E's gates 1, 4, 5 and 8; the benchmark only if the GPU is free.
5. `docs/PHASE_T4_REPORT.md`.
