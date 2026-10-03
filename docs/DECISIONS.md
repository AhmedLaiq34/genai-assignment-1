# Decision log (ADR)

All entries are **proposed** until the student confirms them. Contract changes go through the orchestrator (A0).

| ID | Decision | Rationale | Status |
|---|---|---|---|
| D01 | Salt-and-pepper is per pixel (all 3 channels set together) | Removes PDF ambiguity (plan section 1.7) | proposed |
| D02 | Occlusion coverage = union area | Plan section 1.7 | proposed |
| D03 | Test occlusion rectangles are non-overlapping; coverage achievable within +/-1 pp | Plan section 1.7 | proposed |
| D04 | Resize is direct to 128x128 (no crop, aspect not preserved), applied before corruption | Plan section 1.7 | proposed |
| D05 | Experiment tracking with Weights & Biases (offline + sync on disconnected devices); MLflow file store is the fallback | One place for runs from up to 3 devices (CONTRACTS 3.12) | proposed |
| D06 | Task 1 default: no skip connections (limited-skip ablation optional) | Satisfies "genuine bottleneck" cleanly (PDF p.3-4) | proposed |
| D07 | Single corruption module with policies iid_uniform / balanced_batch / fixed:k | PDF p.3 vs p.4, p.6 (plan section 1.3) | proposed |
| D08 | Restoration studies scored on fixed J, independent of alpha/lambda | Comparable trials (plan section 1.4) | proposed |
| D09 | Start classifier first; run specialists concurrently on other devices | Specialists do not depend on classifier (plan section 1.2) | proposed |
| D10 | Python 3.13 venv; backend image python:3.12-slim | Wheel compatibility (plan section 0) | proposed |
| D11 | question (Agent B): MLflow fallback uses SQLite `artifacts/mlflow.db` instead of the file store in CONTRACTS 3.12, because the installed MLflow 3.16 rejects the plain file store | Installed MLflow behaviour; W&B remains the default (D05) | question |
| D12 | pytorch_msssim SSIM default window is 11x11 Gaussian (not 7x7); used as is at 128x128 | Library default, works at 128x128 | proposed |
| D13 | Scaffold scripts renamed from `scripts/*.py.py` to `scripts/*.py` | Scaffold naming error | proposed |
| D14 | Task 1 study and final training run on Kaggle T4 (student's choice), not the local RTX 3050; local GPU used only for tests, smoke runs, dry runs and benchmark | Student decision overrides the "local first" default in plan 10.4 (item 7) | proposed |
| D15 | Notebooks got a `MODE="benchmark"` that calls scripts/benchmark.py so the T4 budget can be measured before the study | Budgets must come from T4 measurements (plan 3.13) | proposed |
