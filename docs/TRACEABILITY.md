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
| T2a | 4-class CNN classifier, balanced batches, CE loss | 4-5 | `models/classifier.py`, `samplers.py` | not started |
| T2b | Optuna for classifier: lr, batch, channels, dropout, weight decay | 5 | `studies/t2_classifier*` | not started |
| T2c | Classifier results: accuracy, macro P/R/F1, per-class, normalised confusion matrix | 5 | `report/tables`, `report/figures` | not started |
| T2d | 3 independently trained specialist AEs, each on own corruption; shared Optuna allowed (lr, bottleneck, channels, batch, L1/SSIM weight) | 5 | `tasks/task2/`, `studies/t2_specialist_shared*` | not started |
| T2e | Hard-routed inference with identity bypass for clean | 5 | `/api/hard` | not started |
| T2f | Oracle vs predicted routing eval; classifier-caused failures discussed | 5 | `tasks/task2/evaluate.py` | not started |
| T2g | "Hard-Routed Restoration" workspace (4 probs, predicted, expert, output, time); classifier + 3 specialists to ONNX | 5 | `/api/hard`, 4 ONNX files | not started |
| T3a | Soft MoE: gate + 3 experts + identity, w=softmax(G/tau), weighted sum | 5-6 | `models/moe.py` | not started |
| T3b | Init from Task 2 classifier/specialists; gate-only warm-up then joint fine-tune at smaller lr | 6 | `tasks/task3/train.py` | not started |
| T3c | Loss lambda1 L1 + lambda_s(1-SSIM) + lambda_c CE + lambda_b balance (start 0.8/0.2/0.1/0.01) | 6 | `tasks/task3/train.py` | not started |
| T3d | Optuna: joint lr, tau, classification weight, balance weight, reconstruction weighting; pruning optional | 6 | `studies/t3_moe*` | not started |
| T3e | Gate analysis: mean weights per true type x severity, dominant vs distributed examples, heatmap, inactive/dominant expert check | 7 | `report/figures`, `tasks/task3/evaluate.py` | not started |
| T3f | "Soft Mixture-of-Experts Restoration" workspace (4 weights, output, time, contribution highlight); full MoE pipeline to ONNX | 7 | `t3_soft_moe.onnx`, `/api/soft` | not started |
| T4a | FS2K official train/test; 15% val stratified by style seed 42; test untouched; 128x128; pairing preserved | 7 | `fs2k/split.py`, `data/splits/fs2k_split.json` | not started |
| T4b | U-Net generator G(x,s); PatchGAN discriminator D(x,y,s); style embedding in both G and D | 7 | `models/cgan.py` | not started |
| T4c | Loss L_adv + lambda_L1 L1 (BCE-logits; 100 initial, tuned) | 7-8 | `tasks/task4/train.py` | not started |
| T4d | Optuna: lr_G, lr_D, batch, base channels, dropout, embedding dim, lambda_L1; reduced-epoch trials then full retrain | 8 | `studies/t4_cgan*` | not started |
| T4e | Identical spatial augmentation on both members of a pair | 8 | `fs2k/dataset.py`, test | not started |
| T4f | Log D real, D fake, G adv, G L1, val separately; fixed val photos sampled at intervals | 8 | `common/tracking.py`, `report/figures` | not started |
| T4g | "Face-to-Sketch Generator" workspace (upload/webcam, Style 1-3, side-by-side, download); generator to ONNX | 8 | `/api/sketch`, `t4_generator.onnx` | not started |
| A1 | One coherent app, 4 workspaces; backend validates uploads, preprocesses, runs ONNX, returns routing and timing; health, universal, hard, soft, sketch endpoints | 8 | `app/backend/app/main.py` | not started |
| A2 | Runs locally via Docker Compose; evaluator can pick corruption/severity, upload corrupted image, inspect weights, restart from repo | 8 | `docker-compose.yml` | not started |
