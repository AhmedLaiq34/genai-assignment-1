# AI use log

Maintained throughout the project; feeds the report's AI-use appendix.

| date | tool | purpose | how verified |
|---|---|---|---|
| 2026-10-03 | Claude (Opus planner, Sonnet scaffold) | Implementation plan and repository scaffold | pending student review |
| 2026-10-03 | Claude Code (lead) | Environment setup (venv, torch cu128), scaffold script renames | verified by running scripts/env_check.py and CUDA tensor test |
| 2026-10-03 | Claude Code (Agent A, Sonnet) | Wrote pets corruptions/split/manifests + tests | pending student review; pytest passes |
| 2026-10-03 | Claude Code (Agent F) | React/Vite/Tailwind frontend scaffold in app/frontend (Universal Restoration working, 3 stubs, mock server) | pending student review; npm run build and mock API curl checks |
| 2026-10-03 | Claude Code (Agent B, Sonnet) | common/seed, checkpoint, tracking, metrics, timing + tests | verified by pytest tests/test_common_*.py; pending student review |
| 2026-10-03 | Claude Code (Agent D, Sonnet) | Wrote models/autoencoder.py, tasks/task1 (train/tune/evaluate/config), tasks/cli.py, export/*, scripts (train, tune, evaluate, export_onnx, benchmark), notebooks (kaggle/colab drivers), tests (autoencoder, onnx_t1, task1_pipeline) | verified by pytest (see agent report); smoke run / benchmark results recorded separately; pending student review |
| 2026-10-04 | Claude Code (Agent C, Sonnet) | pets dataset.py, samplers.py, prepare_pets.py, package_cloud_data.py + tests | verified by pytest tests/test_dataset.py tests/test_samplers.py (14 passed); pending student review |
| 2026-10-04 | Claude Code (lead) | Notebook benchmark mode; verification of agent outputs | notebooks validated with nbformat; test suite re-run (83 passed) |
| 2026-10-04 | Claude Code (Agent G, Sonnet) | FastAPI backend (health, samples, universal, 501 stubs), preprocess, Dockerfile, docker-compose.yml, .dockerignore, tests/test_backend.py | verified by pytest tests/test_backend.py (15 passed), local uvicorn curl, and docker compose up --build + curl through nginx; pending student review |
