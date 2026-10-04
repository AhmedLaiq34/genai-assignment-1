# Traceability: PDF obligation -> planned evidence

Source: `ASSIGNMENT_TRANSCRIPTION.md` (PDF page in the Page column). All status: not started.

| # | Obligation | PDF p. | Planned file / evidence | Status |
|---|---|---|---|---|
| G1 | Individual work; verify/understand all AI output; justify and modify live | 1 | `docs/AI_USE_LOG.md`, report appendix | not started |
| G2 | All 4 tasks in one browser app (input, run, inspect output, system info) | 1, 8 | `app/` | not started |
| G3 | Report in IEEE format via LaTeX; four tasks separately; figures/tables | 1-2, 9 | `report/` (IEEEtran) | not started |
| G4 | GitHub repo: source, configs, deps, data prep, train, eval, Optuna studies, ONNX export, app, Dockerfiles, Compose, README; no full datasets/large models in git | 2 | repo root, `studies/`, `README.md`, `models/MANIFEST.json` | not started |
| G5 | Model download link or Git LFS | 2 | `models/MANIFEST.json`, `scripts/fetch_models.py` | not started |
| G6 | Optuna in each of the 4 tasks | 2 | `studies/`, `configs/task*.yaml` | not started |
| G7 | MLflow or W&B records (hyperparams, losses, results, checkpoints, visuals) | 2 | `common/tracking.py`, W&B project `genai-a1` | not started |
| G8 | Google Stitch design + evidence in report | 2, 8 | `report/figures/stitch/` | not started |
| G9 | React + Tailwind frontend, FastAPI backend | 2 | `app/frontend`, `app/backend` | not started |
| G10 | ONNX export; PyTorch vs ONNX consistency; ONNX files or links | 2 | `export/`, `report/tables/onnx_parity.csv` | not started |
| G11 | Frontend + backend in Docker; one-command Compose | 2 | `docker-compose.yml`, Dockerfiles | not started |
| G12 | 5-7 min YouTube demo (startup, upload, runtime corruption, universal, hard, soft, sketch, download, tracking); link only | 2 | report link | not started |
| G13 | AI-use appendix (tools, tasks, verification) | 2 | report appendix | not started |
| G14 | Report sections: intro, related work, data prep, architecture, losses, training, Optuna design, setup, results, analysis, app architecture, limitations, conclusion; interpreted figures | 9 | `report/` | not started |
| G15 | Report evidence: arch diagrams, corruption config, curves, Optuna results, confusion matrices, tables, routing viz, grids, error maps, screenshots, failures | 9 | `report/figures`, `report/tables` | not started |
| D1 | Oxford-IIIT Pet, official trainval 80/20 seed 42, test untouched until final; RGB 128x128; same split T1-3 | 2-3 | `pets/split.py`, `data/splits/pets_split.json` | not started |
| D2 | Runtime corruption in loader; new type/severity each load; no saved corrupted copies | 3 | `pets/dataset.py`, `pets/corruptions.py` | not started |
| D3 | Equal-probability choice of clean/salt/blur/occlusion; label retained | 3 | `pets/samplers.py` (iid_uniform) | not started |
| D4 | Corruption definitions (salt 0.02-0.15; blur k in {3,5,7}, sigma 0.5-2.5; 1-3 black rects, 10-35%) | 3 | `pets/corruptions.py`, `configs/data_pets.yaml` | not started |
| D5 | Deterministic val and test manifests (type, severity, masks, blur, seed) | 3 | `data/manifests/` | not started |
| D6 | Test severities: salt 0.03/0.08/0.15; blur (3,0.7),(5,1.5),(7,2.5); occlusion 1/2/3 rects ~10/20/35% | 3 | `pets_test_manifest.jsonl` | not started |
| T1a | Universal DAE: conv encoder, genuine bottleneck, conv decoder, no unrestricted skips (limited skips justified) | 3-4 | `models/autoencoder.py` | not started |
| T1b | Loss alpha*L1 + (1-alpha)(1-SSIM); alpha=0.8 initial, selected by Optuna | 4 | `tasks/task1/train.py` | not started |
| T1c | Optuna: lr, batch, bottleneck, encoder channels, dropout, alpha; report space, completed trials, best, final | 4 | `studies/t1_universal*` | not started |
| T1d | Results: clean/salt/blur/occlusion x low/med/high; target/input/output/abs-error visuals; >=12 examples, 4 failure cases | 4 | `report/tables`, `report/figures` | not started |
| T1e | ONNX + "Universal Restoration" workspace (input, output, settings, inference time) | 4 | `t1_universal_ae.onnx`, `/api/universal` | not started |
| T2a | 4-class CNN classifier, balanced batches, CE loss | 4-5 | `models/classifier.py`, `tasks/task2/classifier.py`, `pets/samplers.py` (balanced_batch); tests `tests/test_classifier.py` | done (val macro-F1 0.9908; `docs/PHASE_T2_TRAINING_REPORT.md`) |
| T2b | Optuna for classifier: lr, batch, channels, dropout, weight decay | 5 | `studies/t2_classifier/` (trials.csv, DB, 3 plots), `report/tables/task2/study_summary.csv`; 12 trials: 11 complete, 1 pruned | done |
| T2c | Classifier results: accuracy, macro P/R/F1, per-class, normalised confusion matrix | 5 | `report/tables/task2/classifier_report.*`, `confusion_normalised.*`; `report/figures/task2/confusion_normalised.png`, `training_curves.png` | done (validation and test sets; `report/tables/task2/test/`) |
| T2d | 3 independently trained specialist AEs, each on own corruption; shared Optuna allowed (lr, bottleneck, channels, batch, L1/SSIM weight) | 5 | `tasks/task2/specialist.py`, `studies/t2_specialist_shared/` (10 trials: 6 complete, 4 pruned), `configs/task2_specialist_final.yaml`, runs in `artifacts/runs/task2_specialist_*` | done |
| T2e | Hard-routed inference with identity bypass for clean | 5 | `tasks/task2/routing.py` (HardRoutedSystem), `app/backend/app/hard.py`; real-model API check `artifacts/eval/task2/api_check.json` (10 calls OK) | done (identity bypass for clean verified) |
| T2f | Oracle vs predicted routing eval; classifier-caused failures discussed | 5 | `tasks/task2/evaluation.py`; `artifacts/eval/task2/20261004-213747_val/` (oracle J 0.1028, predicted J 0.1032, 27 misroutes); `report/tables/task2/results_*.csv`, `misroute_confusion.*`; `report/figures/task2/routing_failures_worst.png` | done on validation and test sets (`artifacts/eval/task2/20261004-234854_test/`: oracle J 0.1277, predicted J 0.1276, 335 misroutes) |
| T2g | "Hard-Routed Restoration" workspace (4 probs, predicted, expert, output, time); classifier + 3 specialists to ONNX | 5 | `app/frontend/src/components/HardWorkspace.jsx`, `/api/hard`; 4 ONNX files in `models/onnx/` (parity rows in `report/tables/onnx_parity.csv`) | backend verified with real models; page not yet looked at in a browser; Docker run not done |
| T3a | Soft MoE: gate + 3 experts + identity, w=softmax(G/tau), weighted sum | 5-6 | `src/genai/models/moe.py`; `tests/test_moe.py` (30 pass); 4,065,717 parameters; soft copy of Task 2 at init: J 0.1029 (tau 1) vs Task 2 predicted 0.1032 (`artifacts/eval/task3/quick_check.json`) | implemented, tested |
| T3b | Init from Task 2 classifier/specialists; gate-only warm-up then joint fine-tune at smaller lr | 6 | `src/genai/tasks/task3/train.py`; `tests/test_task3_train.py`; G2 1+3 epoch run `artifacts/quickcheck_t3/` | implemented, tested; Kaggle run done (D100): final J 0.0932, `models/checkpoints/t3_soft_moe.pt` |
| T3c | Loss lambda1 L1 + lambda_s(1-SSIM) + lambda_c CE + lambda_b balance (start 0.8/0.2/0.1/0.01) | 6 | `train.py` `moe_loss`; `test_loss_terms`, `test_balance_*` | implemented, tested |
| T3d | Optuna: joint lr, tau, classification weight, balance weight, reconstruction weighting; pruning optional | 6 | `src/genai/tasks/task3/tune.py`, `configs/task3_moe.yaml`; `tests/test_task3_tune.py`; dry run `artifacts/dryrun_t3/` | implemented; Kaggle study done (D100): 9 complete / 1 pruned / 2 failed (W&B key), `studies/t3_moe/` |
| T3e | Gate analysis: mean weights per true type x severity, dominant vs distributed examples, heatmap, inactive/dominant expert check | 7 | `src/genai/tasks/task3/evaluate.py`, `tools/t3_report_assets.py`; `tests/test_task3_eval.py` | implemented, tested; evaluation `artifacts/eval/task3/20261005-011630_val`; report figures not generated yet |
| T3f | "Soft Mixture-of-Experts Restoration" workspace (4 weights, output, time, contribution highlight); full MoE pipeline to ONNX | 7 | `src/genai/export/task3_export.py`, `app/backend/app/soft.py`, `/api/soft` in `main.py`; `tests/test_onnx_t3.py`, `tests/test_backend_soft.py`; real-size parity 1.3e-6 / 1.2e-7 (G6, `artifacts/eval/task3/api_check_init.json`) | implemented, tested; promoted `models/onnx/t3_soft_moe.onnx`, parity 8.9e-07 / 2.3e-07, `/api/soft` checked (`artifacts/eval/task3/api_check_final.json`) |
| T4a | FS2K official train/test; 15% val stratified by style seed 42; test untouched; 128x128; pairing preserved | 7 | `fs2k/split.py`, `data/splits/fs2k_split.json` (899/159/1046), `tests/test_fs2k.py`; audit grid `report/figures/task4/fs2k_pair_audit.png` (checked by the student 2026-10-05: pairing correct) | implemented, tested |
| T4b | U-Net generator G(x,s); PatchGAN discriminator D(x,y,s); style embedding in both G and D | 7 | `models/cgan.py`, `tests/test_cgan.py` (embedding grads in G and D) | implemented, tested |
| T4c | Loss L_adv + lambda_L1 L1 (BCE-logits; 100 initial, tuned) | 7-8 | `tasks/task4/train.py` (d_step/g_step), `tests/test_task4_train.py`; final run on a Kaggle T4, best val L1 0.0995 | trained |
| T4d | Optuna: lr_G, lr_D, batch, base channels, dropout, embedding dim, lambda_L1; reduced-epoch trials then full retrain | 8 | `tasks/task4/tune.py` (dry-run tested), `log_study.py`, `studies/task4_cgan/` (Colab study rebuilt from the console log: 14 complete, 6 pruned, 6 fail placeholders) | confirmatory study skipped |
| T4e | Identical spatial augmentation on both members of a pair | 8 | `fs2k/dataset.py: paired_augment`, `tests/test_fs2k.py` | tested |
| T4f | Log D real, D fake, G adv, G L1, val separately; fixed val photos sampled at intervals | 8 | `metrics.jsonl`, W&B run ugt4ngh7 (genai-a1, group task4), `report/figures/task4/losses.png`, `sample_timeline.png` | done |
| T4g | "Face-to-Sketch Generator" workspace (upload/webcam, Style 1-3, side-by-side, download); generator to ONNX | 8 | `/api/sketch` (`app/backend/app/sketch.py`), `models/onnx/t4_generator.onnx` (parity 1.2e-5) | backend done; workspace in app/frontend_v2 not checked |
| A1 | One coherent app, 4 workspaces; backend validates uploads, preprocesses, runs ONNX, returns routing and timing; health, universal, hard, soft, sketch endpoints | 8 | `app/backend/app/main.py` | not started |
| A2 | Runs locally via Docker Compose; evaluator can pick corruption/severity, upload corrupted image, inspect weights, restart from repo | 8 | `docker-compose.yml` | not started |
