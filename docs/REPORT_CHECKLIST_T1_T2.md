# Report checklist for Tasks 1 and 2 (one unified IEEE LaTeX paper)

Source of the requirements: `ASSIGNMENT_TRANSCRIPTION.md` PDF pages 1, 2, 4, 5, 9 (the PDF controls). Status as of 2026-10-04.
One paper for all four tasks; each task gets its own clearly identifiable methodology and results part.
All generated tables and figures come from scripts: `tools/t2_report_assets.py` (Task 2), `tools/report_common_assets.py` (shared),
and the Task 1 files already in `report/` (made by the Task 1 session). Rerun the two scripts after any change of results.

## 1. Shared sections (both tasks)

| Required content (PDF p. 9, 1-2) | Evidence in the repository | Status |
|---|---|---|
| Problem introduction | Assignment text; `docs/IMPLEMENTATION_PLAN.md` section 1 | text to write |
| Related research, alternatives investigated, reasons for choices | Alternatives actually investigated are documented: dense vs convolutional latent (`docs/LESSONS_FROM_TASK1.md` sec. 3, D40), "do nothing" baseline and thumbnail calibration (sec. 3.4), bottleneck-size sweep (sec. 3.2), blur-specialist latent sizes 4096/8192/16384 (`docs/PHASE_T2_REPORT.md` sec. 10, D43), pruner warm-up from learning curves (D19, D42), skip connections not used (D06). Candidate references: section 5 below | references to read and cite (student) |
| Dataset preparation (Oxford-IIIT Pet, 80/20 split seed 42, RGB 128x128, test untouched until the end) | `report/tables/dataset_split.{csv,tex}` (counts, manifest rows, sha256); `configs/data_pets.yaml`; `docs/DECISIONS.md` D01 to D04, D07 | ready |
| Complete corruption configuration (training distributions and fixed test severities) | `report/tables/corruption_config.{csv,tex}`; example grid `report/figures/corruption_examples.png` | ready |
| Runtime corruption in the loader, deterministic validation/test manifests | `src/genai/pets/dataset.py`, `manifests.py`; row schema in `docs/CONTRACTS.md` 3.4; manifests are byte-reproducible (tests) | text to write |
| Optuna search design (search space, sampler, pruner, budget) | `report/tables/search_spaces.{csv,tex}` (all three studies) | ready |
| Experimental setup (hardware, software, seed, optimiser, schedule, precision) | `report/tables/final_configs.{csv,tex}`; `docs/ENVIRONMENT.md` (laptop); Kaggle T4 for the final runs (D41, D44); torch version is in each checkpoint | ready |
| Experiment tracking (W&B or MLflow) | W&B project `ahmedlaiq34/genai-a1` (https://wandb.ai/ahmedlaiq34/genai-a1), groups `task1` and `task2`; metrics, configs and sample grids are logged. Checkpoints are recorded by sha256 in `models/MANIFEST.json` (they are not uploaded to W&B) | dashboard screenshots to take (student) |
| ONNX export and PyTorch/ONNX consistency | `report/tables/onnx_parity.csv` (5 rows, tag `final`, 16 val inputs each, tolerance 1e-4, all passed); routing parity 16/16 (`docs/PHASE_T2_TRAINING_REPORT.md` sec. 6) | ready |
| Application architecture and screenshots | `report/figures/app/*.png` (Hard-Routed for none/salt/blur/occlusion, Universal, System panel); `app/frontend_v2`, `app/backend`, `docker-compose.yml`; `docs/PHASE_T2_TRAINING_REPORT.md` sec. 8 | text to write; Universal screenshot with the real Task 1 model exists |
| Google Stitch design evidence | the Stitch export zips are in the student's Downloads | to copy into `report/figures/stitch/` (student) |
| GitHub link, YouTube link, model download link | none yet; `models/MANIFEST.json` has no download URLs | student |
| AI-use appendix (tools, tasks, verification) | `docs/AI_USE_LOG.md` (one row per agent task), `docs/DECISIONS.md` | text to write from the log |
| Limitations and conclusion | listed per task below | text to write |

## 2. Task 1 (universal denoising autoencoder)

| Required (PDF p. 3-4) | Evidence | Status |
|---|---|---|
| Architecture diagram; encoder/latent/decoder description; no skips (or justified skips) | `report/figures/t1_architecture.png`; `docs/PHASE_T1_TRAINING_REPORT.md` sec. 3 (UniversalAE, base 64, depth 3, conv latent 16x16x8 = 2,048 values, 24:1, 1,322,507 parameters); D06, D40 | ready |
| Loss L_UDA = alpha L1 + (1 - alpha)(1 - SSIM), alpha selected by Optuna | alpha 0.5022 (best trial); D08 (selection objective J = 0.5 L1 + 0.5 (1 - SSIM) independent of alpha); `final_configs` table | ready |
| Training procedure | Training report sec. 5; `report/figures/t1_training_curves.png` | ready |
| Optuna: complete search space, completed trials, best trial, final configuration | `search_spaces` table; training report sec. 4 (40 counted trials: 35 complete, 5 pruned, 0 failed; best trial 33, J 0.1406); `report/tables/t1_optuna_trials.csv`; `t1_optuna_history/importance/parallel.png` | ready |
| Results separately for clean, salt, blur, occlusion and for low/medium/high severity | `report/tables/t1_val_results.csv`, `t1_test_results.csv` (test: 3,669 images x 10 conditions, opened once with approval); `t1_test_vs_input.png` | ready |
| Visuals: target, input, output, absolute error map; at least 12 representative examples and 4 failure cases | `t1_val_representative_12.png`, `t1_test_representative_12.png`, `t1_test_worst_4.png` (rows of target / input / output / error x4); lists in `artifacts/eval/task1/*/selected.csv` | figures ready; the discussion of the 4 failure cases must be written (what failed and why) |
| ONNX + workspace "Universal Restoration" showing input, output, settings, inference time | `models/onnx/t1_universal_ae.onnx`; the real model was run through Docker and a Chrome-driven UI check on 2026-10-04 (D51); screenshot `report/figures/app/app_universal.png` | ready |
| Facts the text must state | Latent is a grid, not a flat vector (the first dense model learned only an 8x8 thumbnail, D40); mild damage (low blur, low occlusion) is worse than the untouched input; validation severities are tertiles of the sampled parameter, test severities are the fixed levels | text to write |

## 3. Task 2 (classifier and hard-routed specialists)

| Required (PDF p. 4-5) | Evidence | Status |
|---|---|---|
| Architecture diagram of the system, classifier and specialists | `report/figures/t2_pipeline.png` (new); `report/tables/task2/model_facts.{csv,tex}` (classifier 98,196 parameters; each specialist 1,322,507, same architecture as Task 1 with separate weights) | ready |
| Classifier: cross-entropy, balanced batches, tuned lr / batch / channels / dropout / weight decay | `configs/task2_classifier_final.yaml`; `src/genai/pets/samplers.py` (exact B/4 per class, tested); `search_spaces` table | ready |
| Specialists: own corruption only, clean target, shared Optuna study over lr / bottleneck / channels / batch / L1-SSIM weight, three independent trainings | `configs/task2_specialist_final.yaml`; `report/tables/task2/study_summary.*`; training report sec. 1 to 2 | ready |
| Optuna results: search space, completed trials, best trial, final configuration (both studies) | classifier 12 trials (11 complete, 1 pruned, 0 failed), best trial 10, macro-F1 0.9813; specialists 10 trials (6 complete, 4 pruned, 0 failed), best trial 5, mean J 0.1674; plots `report/figures/task2/cls_*`, `spec_*`; `studies/t2_*/trials.csv` | ready |
| Classifier results: accuracy, macro P/R/F1, per-class, normalised 4x4 confusion matrix | `report/tables/task2/classifier_report.*`, `confusion_normalised.*`; `report/figures/task2/confusion_normalised.png` (validation set) | ready for validation |
| Oracle vs predicted routing; classifier-caused failures identified and discussed | `report/tables/task2/results_oracle.*`, `results_predicted.*` (with J_input baseline), `misroute_confusion.*`, `routing_failures_worst.png`, `j_by_condition.png`; training report sec. 3 | ready on validation and test sets (test: `report/tables/task2/test/`, comparison with Task 1 in `comparison_t1_t2_test`) |
| Training/validation curves | `report/figures/task2/training_curves.png` | ready |
| Image grids and error maps | `report/figures/task2/restoration_examples_{salt,blur,occlusion}.png` | ready |
| Four ONNX files and the workspace "Hard-Routed Restoration" (4 probabilities, predicted class, expert, output, time) | `models/onnx/t2_*.onnx`; real-model API check and Docker check (D49, D50); Chrome-driven UI check (D51); screenshots `report/figures/app/app_hard-*.png` | ready |
| Facts the text must state | Specialists use the convolutional latent grid (2,048 values), D42; the blur specialist's J is higher than its input's at low and medium severity (D43, accepted); some best parameters are at range edges (classifier batch 32, dropout about 0.4; specialist base channels 64, alpha 0.59); budgets: classifier 12 trials x 8 epochs, final 30 epochs; specialists 10 trials x 8 epochs per corruption, finals 50 epochs (D47); predicted routing for blur is slightly better than oracle because mild blurs sent to the identity bypass lose less (training report sec. 3); validation severities are tertiles | text to write |

## 4. Open items that block a complete Task 1 + 2 report

1. **Task 2 test-set results: done** (2026-10-04, one run with the student's approval): `artifacts/eval/task2/20261004-234854_test/`, tables in `report/tables/task2/test/` (including `comparison_t1_t2_test`, same test tensors as Task 1), figures in `report/figures/task2/test/`, written up in `docs/PHASE_T2_TRAINING_REPORT.md` section 10. The test set is now opened for Tasks 1 and 2; do not open it again for them.
2. **Failure-case discussion text** for Task 1 (4 cases) and Task 2 (routing failures): the figures and lists exist, the explanation of why each failed must be written.
3. **Stitch evidence, W&B dashboard screenshots, GitHub / YouTube / model-download links** (student).
4. **References** (section 5): choose the ones actually consulted, read them, cite correctly.
5. Tasks 3 and 4 sections are not part of this checklist.

## 5. Candidate references (real publications; check each one before citing)

- Parkhi, Vedaldi, Zisserman, Jawahar, "Cats and dogs", CVPR 2012 (Oxford-IIIT Pet).
- Vincent, Larochelle, Bengio, Manzagol, "Extracting and composing robust features with denoising autoencoders", ICML 2008.
- Hinton and Salakhutdinov, "Reducing the dimensionality of data with neural networks", Science 2006.
- Wang, Bovik, Sheikh, Simoncelli, "Image quality assessment: from error visibility to structural similarity", IEEE TIP 2004 (SSIM).
- Zhao, Gallo, Frosio, Kautz, "Loss functions for image restoration with neural networks", IEEE TCI 2017 (L1 + SSIM losses).
- Akiba, Sano, Yanase, Ohta, Koyama, "Optuna: a next-generation hyperparameter optimization framework", KDD 2019.
- Loshchilov and Hutter, "Decoupled weight decay regularization", ICLR 2019 (AdamW); "SGDR: stochastic gradient descent with warm restarts", ICLR 2017 (cosine schedule).
- Ioffe and Szegedy, "Batch normalization", ICML 2015; Srivastava et al., "Dropout", JMLR 2014.
- Paszke et al., "PyTorch: an imperative style, high-performance deep learning library", NeurIPS 2019; pytorch-msssim (the SSIM implementation used); ONNX and ONNX Runtime documentation.
