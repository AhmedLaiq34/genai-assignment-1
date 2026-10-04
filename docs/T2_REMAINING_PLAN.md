# Task 2: remaining work plan (for the Sonnet session)

State on 2026-10-04: models trained, promoted, ONNX exported and verified (`docs/PHASE_T2_TRAINING_REPORT.md`, D48).
What is left is listed below in priority order. Rules: no test-set run without the student's explicit approval;
no Task 1 file edits; one heavy process at a time; `CUDA_VISIBLE_DEVICES=-1` for pytest; temp files on D:
(`--basetemp=artifacts/pytest_tmp/<name>`); report only what was run; next free decision number is D49.

## Step 1. Backend check with the real ONNX files (no frontend needed) - PRIORITY
1. Confirm `models/onnx/` holds `t2_classifier.onnx`, `t2_ae_{salt,blur,occlusion}.onnx` (+ sidecars, `smoke: false`).
2. Start the backend in the background from `app/backend`: `..\..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000`
   (MODELS_DIR defaults to `models/onnx`; log to `artifacts/logs/t2_backend.log`).
3. `GET /api/health`: the four t2 keys show `loaded: true`, `smoke: false`, sha256 equal to the ONNX sidecars.
4. `GET /api/samples`, take one sample id. Then `POST /api/hard` (multipart `sample_id`, `seed=42`) for:
   corruption `none`, `salt_pepper`, `gaussian_blur`, `occlusion`, each at severity `low|medium|high` (10 calls).
   Acceptance per call: HTTP 200; `probs` has 4 values summing to ~1; `predicted` in class names;
   `expert` matches `predicted` (clean -> `identity`, `identity_bypass: true`, `timing_ms.expert == 0`);
   `timing_ms` has preprocess/classifier/expert/total. Record predicted class vs applied corruption (expect mostly correct;
   mild blur may be predicted clean, that is consistent with the val results).
5. Also one upload (`file=` a PNG/JPEG from the sample folder) and one bad upload (text file -> 4xx).
6. `POST /api/universal` once (Task 1 still works). Stop the server.
7. Save the responses (without the base64 images) to `artifacts/eval/task2/api_check.json`; decode one input/output pair per
   corruption to PNG in the same folder for a quick visual look.
Failure handling: a 503 means a file/name mismatch in `models/onnx`; a 500 means read the server log, fix in
`app/backend/app/hard.py` or `main.py` only, re-run `tests/test_backend_hard.py tests/test_backend.py`.

## Step 2. Docker Compose check (only if Docker Desktop is running; ~5-10 min build)
1. `docker compose up --build -d` at the repo root; `models/onnx` is mounted read-only.
2. Repeat 3 calls through nginx: `GET http://localhost:8080/api/health`, `POST /api/hard` (blur/medium and none), `GET /` (index).
3. `docker compose down`. Record image build success and the responses. If Docker is not running, skip and say so.

## Step 3. Frontend visual check (needs the student or a browser tool)
1. `cd app/frontend && npm run build` (must pass). Dev check: backend on :8000 + `npm run dev` (Vite proxy) or the Compose URL.
2. Student opens the Hard-Routed Restoration tab: pick a sample, choose blur/medium, Run. Check: 4 labelled probability bars,
   predicted class, expert or "identity bypass" label, input/output images, latency breakdown, download button.
   Then corruption `none` -> identity bypass shown. Screenshots to `report/figures/app/` (hard_routed_*.png).
3. Update the stale `app/frontend/README.md` line that says only Universal works (Hard-Routed now works).

## Step 4. Report figures and tables (from existing files, no training)
Write a small script `tools/t2_report_assets.py` (reads only; writes to `report/figures/task2/` and `report/tables/task2/`):
1. Tables (CSV + LaTeX `.tex` via pandas `to_latex`): from `artifacts/eval/task2/20261004-213747_val/`:
   `table_cond_severity_oracle.csv`, `_predicted.csv` (columns cond, severity, J, SSIM, PSNR, MAE, J_input, SSIM_input, count);
   classifier report (accuracy, macro P/R/F1, per-class P/R/F1/support) from `classifier_report.json`;
   misroute table from `misroute_confusion.csv`; study summary (counts COMPLETE/PRUNED/FAIL, best trial, best params)
   from `studies/t2_classifier/trials.csv`, `studies/t2_specialist_shared/trials.csv`.
2. Figures: copy `confusion_normalised.png`, `routing_failures_worst.png`; Optuna plots already in `studies/t2_*/`
   (optimization_history, param_importances, parallel_coordinate; check which exist, some may be `.skipped.txt`);
   training curves from each final run's `metrics.jsonl` (classifier macro-F1/loss per epoch; specialist val J per epoch,
   three lines); a grid of representative restorations per corruption (target | input | output | |error|) from the
   promoted checkpoints on fixed val rows (CPU is fine; reuse `HardRoutedSystem.restore_by_route`), and an
   "input J vs oracle J vs predicted J" bar chart per condition.
3. Architecture facts for the report text: classifier `channels 16-32-64-128, dropout 0.40` and specialist
   `UniversalAE(latent=conv, depth=3, base 64, bottleneck 2048)` parameter counts (`count_parameters`).
Acceptance: script runs in under ~2 min, every file listed above exists; numbers equal those in the training report.

## Step 5. Tests and docs
1. Full suite: `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --basetemp=artifacts/pytest_tmp/final` (expect 189+ passed).
2. Append D49 (what Steps 1-4 produced) to `docs/DECISIONS.md`; one row to `docs/AI_USE_LOG.md`.
3. Add a short "App check" and "Report assets" section to `docs/PHASE_T2_TRAINING_REPORT.md` (neutral, factual tone; the
   student asked not to frame the budgets negatively).
4. `docs/TRACEABILITY.md`: mark the Task 2 PDF obligations (classifier CE + balanced batches, Optuna params, metrics + confusion
   matrix, three specialists, shared study, hard routing with identity bypass, oracle vs predicted, routing failures,
   workspace, four ONNX) with their evidence file.

## Step 6. Commit (only if the student asks)
On branch `dev`. Check `git status` first: no `.pt`, `.onnx`, zips or `artifacts/` staged (all gitignored; `studies/t2_*` DBs and
CSVs ARE meant to be committed, `models/MANIFEST.json` too). Commit message ends with the attribution line from the system reminder.

## Deferred (not Task 2 completion, needs the student)
- Test-set evaluation (`scripts/evaluate.py --task t2cls --final-test` with the four promoted checkpoints) together with Tasks 1 and 3,
  after explicit approval. Note: `artifacts/test_access.log` already has one line (21:07 local) not written by Task 2.
- Stitch-based frontend redesign (zips in the student's Downloads).
- Task 3 can start now: it loads the four promoted checkpoints read-only and asserts the sha256 values in the training report.
