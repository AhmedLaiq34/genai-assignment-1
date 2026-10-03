# Generative AI Assignment #1 — execution map

This is a planning handoff, not evidence that any phase has been completed. Read `ASSIGNMENT_TRANSCRIPTION.md` for the assignment wording. If this map conflicts with the PDF, the PDF controls. The printed March 16, 2024 deadline is inconsistent with the current situation; the student reports about 1.5 days remaining. There is no money available for paid compute. Do not assign an hourly schedule or invent a marking rubric.

## Goal and ground rules

Build four trained and evaluated systems in **one** React/Tailwind + FastAPI browser app, export inference models to ONNX, start frontend/backend through Docker Compose, track experiments, and produce an IEEE LaTeX report and 5–7-minute demonstration. Research choices and failures must be explained. The student is completing this individually with AI assistance and must be able to explain, test, and modify every component.

Treat implementation, training, evaluation, report evidence, and packaging as deliverables of each task. Preserve the official test sets until final evaluation. Do not substitute a tiny smoke test for final results, claim unrun Optuna trials, or imply a performance threshold the PDF never sets.

## Dependency graph

```text
Environment and dataset access
  ├─ Shared pet split + corruption/manifests ─┬─ Task 1 universal AE ──────────┐
  │                                            └─ Task 2 classifier + 3 AEs ─┐ │
  │                                                                         └─ Task 3 soft MoE
  ├─ FS2K pairs + official split ──────────────── Task 4 conditional GAN      │
  └─ Stitch design + shared app contracts ────── frontend/backend/Compose     │
                                                                               ↓
                        Final exports + identical-input evaluation → report + demo
```

Task 1 and Task 4 training can proceed independently after their data checks. Task 2's classifier and specialists can be developed in parallel; three independently trained specialists may run on *separate* GPU devices after selecting their configuration. Task 3 code can be built against test fixtures early, but **real Task 3 training must begin from the completed Task 2 classifier and all three trained specialists**. Keep Task 2 checkpoints immutable. Documentation, figures/templates, app integration, and Docker can proceed alongside training. Agents working in parallel do not create additional GPU capacity.

## Phase 0 — verify feasibility and write the contracts

**Entry:** Empty repository, assignment text, local Victus 15 (13th-gen i5, RTX 3050); actual VRAM/RAM/storage, installed dependencies, free GPU allocation, and FS2K access are unconfirmed.

**Work:** Inspect environment, `nvidia-smi`, CUDA PyTorch, Docker, disk, Kaggle GPU/quota and Colab availability. Confirm official datasets can be accessed. Decide exact conventions: class IDs `[clean, salt, blur, occlusion]`, style IDs `[0,1,2]`, tensor shape/channels, ranges (`[0,1]` for restoration; an explicit range for GAN), image-size handling, color/EXIF, seed policy, config format, checkpoint schema, model output signatures, API responses, and filenames. Create a requirements traceability checklist with PDF pages and a compact architecture decision log. No paid services.

**Artifacts:** Environment record, `configs/`, repository layout, decision log, dependency lock or pinned versions, preliminary contracts, issue/risk register.

**Gate:** A sample GPU training step and Docker command work; dataset download paths and free cloud options are known. If a blocker remains, record it and implement unaffected work.

## Phase 1 — data foundations and Stitch design

**Pet data:** Use official development collection, split 80/20 seed 42, retain the official test untouched. RGB 128×128. Dynamic four-way equiprobable clean/salt-and-pepper/blur/occlusion training input; specialists receive their own corruption only. Classifier/MoE batches must actually balance four labels. Salt: probability U(0.02,0.15), black/white equally likely. Blur: kernel 3/5/7 and sigma U(0.5,2.5). Occlusion: 1–3 black rectangles with **union** coverage 10–35%. Store deterministic validation and test manifests, including seeds, settings, coordinates, achieved coverage, image IDs, and version/hashes. Final test has each image clean plus nine fixed corrupted conditions (three severities each).

**FS2K:** Use official train/test annotations, validate photograph–sketch pairing and style from annotations, split 15% of official training stratified by style with seed 42, keep official test out of tuning. Apply the *same* spatial transform to both members of a pair.

**Design:** Make original four-workspace interface in Google Stitch now. Save prompts, screenshots/exports for the report.

**Artifacts:** `src/data/`, preparation scripts, persisted split and corruption manifests, sample grids, pair audit, Stitch evidence.

**Gate:** Zero split overlap; sample pairs line up; every test condition is enumerated; same manifest regenerates same pixels; coverage accounts for rectangle overlap; balanced batches have exact class counts. Do not start expensive training until these checks pass.

## Phase 2 — end-to-end smoke path and measurement

Build a small universal AE, a minimal training script, a checkpoint, an ONNX export, a FastAPI endpoint, and a minimal React upload/result screen. Run the path through Docker Compose. Also instantiate/export-test the classifier, complete MoE wrapper, and style-conditioned generator with dummy weights to catch unsupported operators and input-signature problems early. Tag these files as **smoke artifacts**.

Measure realistic steps on local GPU and each available free GPU: batch sizes, peak memory, warmed-up seconds/step, validation time, and actual cost of the full three-expert MoE and two-optimizer GAN. Synchronize CUDA for timings. Check free platform session/quota in the user's own account. Do not quote speculative runtimes as measurements.

**Artifacts:** Smoke checkpoint/ONNX, working browser path, benchmark logs, revised trial/epoch budgets, resumable checkpoint code.

**Gate:** Image → loader → training step → checkpoint → ONNX → backend → browser works. A full training/validation step for each model fits on the assigned GPU. No expensive study before this gate.

## Phase 3 — independent model workstreams

### Workstream A: Task 1, universal restoration

One convolutional encoder, genuine compressed bottleneck, decoder, no unrestricted skip bypass. Train across clean and all three corruptions with L1 + SSIM. Optuna must investigate learning rate, batch size, bottleneck dimension, encoder channels, dropout, and L1/SSIM weight. Select by a fixed validation objective, report search space/completed trials/best configuration. Final train, evaluate clean and each severity, generate target/input/output/absolute-error images, discuss ≥12 representative examples and four failures. Export ONNX. Build **Universal Restoration** workspace with input, selected corruption settings, output, latency.

### Workstream B: Task 2, classifier + specialist models

Balanced four-class convolutional classifier trained with cross-entropy; Optuna learning rate, batch, channels, dropout, weight decay. Evaluate accuracy, macro precision/recall/F1, per-class measures, normalized confusion matrix. Train three independent AEs exclusively on salt, blur, and occlusion with clean targets. Use either specialist searches or the PDF-permitted shared architecture search; tune learning rate, bottleneck, channels, batch, L1/SSIM weighting. Final train each separately. Implement identity bypass for predicted-clean inputs, oracle and predicted routes, and routing-error failure analysis. Export classifier and all specialists to ONNX. Build **Hard-Routed Restoration** workspace with probabilities, predicted class, selected expert, output, latency.

### Workstream C: Task 4, face-to-sketch GAN

Paired FS2K data. U-Net generator + PatchGAN discriminator, learned categorical style embeddings incorporated in **both** networks. Paired L1 and adversarial objectives. Optuna both learning rates, batch, base channels, dropout, embedding dimension, reconstruction weight; short trials followed by full training of selected configuration. Record discriminator real/fake, generator adversarial/reconstruction, validation metrics, and same validation images at fixed intervals. Export **generator only** to ONNX. Build **Face-to-Sketch Generator** workspace with upload/webcam, Style 1/2/3, side-by-side display, download. Explain lack of paired targets for counterfactual style requests.

**Parallel rule:** A, B, C can proceed concurrently on independent resources. Do not launch multiple heavy jobs into one GPU without a benchmark. Independent cloud GPUs can run separate jobs; cloud availability and free quotas are not guaranteed. Keep training as normal Python scripts under the repository, invoked from notebooks if useful, so the submission is not notebook-only.

**Artifacts:** Source, configurations, Optuna studies, experiment tracking, final native checkpoints, images/curves, ONNX exports, task workspaces.

**Gate:** Every final model has a traceable validation selection, reproducible configuration, and checkpoint; Task 2 originals are saved and hashes recorded. Test data has not affected model selection.

## Phase 4 — dependent Task 3 workstream

Initialize a **new** gate from the Task 2 classifier and a **new** copy of the three specialists from their frozen Task 2 checkpoints; add identity branch. First warm up gate while experts stay frozen, then jointly fine-tune all components at a lower learning rate. Output differentiable weighted reconstruction and all four weights. Loss contains L1, (1−SSIM), corruption classification CE, and balance regularizer. Optuna joint LR, temperature, classification weight, balance weight, reconstruction weighting. Start each trial from the *same* original Task 2 checkpoints. Compare weights by true condition/severity, dominance/distribution and inactive branches. Export complete MoE inference graph. Build **Soft Mixture-of-Experts Restoration** workspace with four weights, contribution visualization, result, latency.

**Artifacts:** Source checkpoint hashes, warm-up/joint logs, study, final checkpoint, routing heatmap and cases, complete MoE ONNX, workspace.

**Gate:** Experts unchanged during warm-up, trainable in joint stage; weights sum to 1 and reconstruction gradients reach gate/experts; original Task 2 checkpoint files still hash identically.

## Phase 5 — fair final evaluation and export verification

Lock final configs/checkpoint choice *before* opening official tests. Evaluate universal, oracle-hard, predicted-hard, and soft MoE on **identical** pet test manifest tensors, including clean and all nine fixed conditions. Report class/severity results, MAE, SSIM, useful PSNR baseline, sample counts and inference timing; use correct image ranges. Explain classifier-caused failures, clean fidelity, four Task 1 failures, routing collapse/inactivity. Evaluate FS2K with annotated paired style, report by style; show alternate-style outputs qualitatively when no paired truth exists. Keep per-image outputs for audits. Numerically compare all required PyTorch/ONNX models on representative inputs and all style IDs. Verify hard-route behavior too.

**Artifacts:** Per-image metrics, plots, figures, tables, ONNX parity report, model hash/metadata/download manifest.

**Gate:** No test-driven retraining or HPO; every plotted number references input manifest, checkpoint, and metric code; ONNX inference works in the backend environment.

## Phase 6 — app completion, deployment and submission

Implement one responsive React/Tailwind app derived from saved Stitch design; FastAPI health, universal, hard, soft, and sketch operations; upload validation, shared preprocessing, ONNX sessions, errors, clean-sample/runtime-corruption controls, downloads and webcam option. Containerize **both** frontend and backend; provide model acquisition instructions or verified fetch and a single Docker Compose command. Test a fresh clone, unseen uploads, corrupted uploads without accidental second corruption, and application restart. Public hosting is optional.

Write IEEE LaTeX paper with research choices, alternatives, per-task methods/results, complete corruption configurations, Optuna study results, curves, confusion matrix, routing heatmap, grids/error maps, four failures, Stitch evidence, app screenshots, limitations, interpretation, GitHub link, YouTube link, citations, reuse attribution and AI-use appendix. Create 5–7-minute YouTube demo of Compose startup, four workspaces, image upload, runtime corruption, routing/weights, sketch, download, tracking. Submit according to verified Classroom deadline; PDF's March 16, 2024 date should be flagged.

**Artifacts:** GitHub repository, README, dependency files, Compose, Dockerfiles, model distribution links/hashes, report source/PDF, demo URL, AI-use appendix.

**Gate:** Someone can clone, obtain models, run one documented Compose command, access all workspaces in a browser, and reproduce the reported inference behavior without an IDE or separate scripts.

## Suggested free-compute allocation

| Location | Best use | Important constraint |
| --- | --- | --- |
| Local Victus RTX 3050 | Repository development, smoke tests, small runs, ONNX parity, final Docker application; additional training when available | Confirm VRAM and benchmark sustained steps first |
| Kaggle free GPU | Heavy GAN and MoE, searches/specialist runs when quota permits | Check actual account quota, allocated GPU(s), session length and persistent outputs; one agent per device is not implicit |
| Colab free GPU | Backup or separate independent run | GPU availability and limits vary; runtime can disconnect; save checkpoints outside ephemeral VM |

Keep datasets and intermediate train loops on the runtime disk when possible; save checkpoints and logs persistently at intervals. Do not assume simultaneous availability of Kaggle and Colab, or schedule training jobs before confirming real access. Assign device and output directory explicitly to each run; never let parallel jobs write the same study database/checkpoint path. If using many coding agents, divide ownership by module and interface contract, then review integration centrally. The student retains individual responsibility and documents AI assistance.
