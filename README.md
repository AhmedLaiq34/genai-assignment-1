# Generative AI Assignment #1

> All four tasks have saved training/evaluation records and local ONNX models. The submission PDF is in `report/main.pdf`. See `docs/SUBMISSION_AUDIT.md` for remaining submission checks.

## Overview of the 4 tasks
1. Universal multi-corruption denoising autoencoder (Oxford-IIIT Pet, 128x128).
2. Corruption classifier + hard-routed specialist autoencoders.
3. Jointly trained soft mixture-of-experts restoration.
4. Style-conditioned face-to-sketch cGAN (FS2K).

Authoritative references: `docs/IMPLEMENTATION_PLAN.md`, `docs/CONTRACTS.md`, `docs/TRACEABILITY.md`.

## Repository layout

`src/genai/` contains data pipelines, models, training, tuning, evaluation and export modules. `scripts/` provides CLI entry points; `configs/` contains data/device/task settings; `studies/` contains saved Optuna records. The active application is `app/frontend_v2/` plus `app/backend/`. `report/` contains the submission PDF and evidence. `docs/` contains decisions, phase reports and the AI-use log. Datasets, run folders and trained model binaries are obtained separately.

## Setup
```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -r requirements.txt
python scripts/env_check.py
```
Details: `docs/ENVIRONMENT.md`.

## Data preparation

Oxford preparation downloads the official partitions using torchvision. FS2K must be downloaded separately into the layout specified in `configs/data_fs2k.yaml`; see `docs/PHASE_T4_REPORT.md` for the verified pair/split counts.

```powershell
python scripts/prepare_pets.py --config configs/data_pets.yaml
python scripts/prepare_fs2k.py --config configs/data_fs2k.yaml
```

These preparation commands create caches and split metadata, including official test partitions. Use test data only for the final evaluation protocol. An evaluator running the application needs only the ONNX release assets and bundled samples, not either dataset.

## Training / Tuning

All training is already complete. For reproduction, these commands use the saved final configurations (not needed to run the app):

```powershell
python scripts/train.py --task t1 --config configs/task1_final_v2.yaml
python scripts/train.py --task t2cls --config configs/task2_classifier_final.yaml
python scripts/train.py --task t2spec --config configs/task2_specialist_final.yaml --set specialist.corruption=salt
python scripts/train.py --task t2spec --config configs/task2_specialist_final.yaml --set specialist.corruption=blur
python scripts/train.py --task t2spec --config configs/task2_specialist_final.yaml --set specialist.corruption=occlusion
python scripts/train.py --task t3 --config configs/task3_moe_final.yaml
python scripts/train.py --task t4 --config configs/task4_final.yaml
```

Task 3 initialization additionally requires the four Task 2 checkpoints, which are not included in the inference-only release. Tuning uses `scripts/tune.py --task TASK --config CONFIG`; task flags are `t1`, `t2cls`, `t2spec`, `t3`, `t4`. Study configurations are `task1_universal.yaml`, `task2_classifier.yaml`, `task2_specialist.yaml`, `task3_moe.yaml`, and `task4_cgan.yaml` under `configs/`. Read the corresponding phase reports and budgets before starting a new study; Task 4 retains its study source metadata and proposed repository ranges separately.

## Evaluation

`python scripts/evaluate.py --task TASK --config CONFIG` evaluates validation data using the selected task configuration; `--ckpt PATH` selects a checkpoint where applicable. Task 2 evaluates the classifier plus all three experts through its configuration. `--final-test` opens the locked official test set and logs access; use it only after model selection is complete. Existing final results are in `report/tables/` and the phase reports.

## ONNX export

`python scripts/export_onnx.py --model MODEL --ckpt PATH --verify` exports and checks parity on validation inputs. MODEL is one of `t1_universal`, `t2_classifier`, `t2_salt`, `t2_blur`, `t2_occlusion`, `t3_soft_moe`, or `t4_generator`. The inference release already contains all seven exported models. Existing parity results are in `report/tables/onnx_parity.csv`.

## Running the app
```
docker compose up --build
```
Download the model release first. Open http://localhost:8080; frontend port 8080 is public locally, backend port 8000 is internal. Stop with `docker compose down`. The backend provides health/universal/hard/soft/sketch operations and reports a missing model with HTTP 503. Compose uses the Stitch-based `app/frontend_v2`; `app/frontend` is the earlier implementation.

## Model downloads

[Download the seven trained ONNX models](https://github.com/AhmedLaiq34/genai-assignment-1/releases/tag/models-v1). Put the files in `models/onnx/`; SHA-256 hashes, sizes and individual URLs are in `models/MANIFEST.json`.

The repository and model release are public. Download and verify all seven models with:

```powershell
python scripts/fetch_models.py
docker compose up --build
```

Alternatively, use GitHub CLI:

```powershell
gh release download models-v1 --repo AhmedLaiq34/genai-assignment-1 --pattern '*.onnx' --dir models/onnx
python scripts/fetch_models.py --verify-only
docker compose up --build
```

Open http://localhost:8080. For a local transfer, use `python scripts/fetch_models.py --source PATH_TO_MODEL_FOLDER`. Existing files are verified and never replaced.

## Experiment tracking

Saved experiments use [Weights & Biases project genai-a1](https://wandb.ai/ahmedlaiq34/genai-a1). Configure your own account with `wandb login` for reproduction; never store credentials in Git. Phase reports describe online/offline runs and later synchronization. Tracking screenshots are included in `report/figures/wandb/`. MLflow is the supported fallback; see `docs/DECISIONS.md` and `src/genai/common/tracking.py`.

## Reproducibility
Seed 42, committed splits/manifests, recorded checkpoint hashes, ONNX hashes, final configs and Optuna records support reproduction. The manifest includes local verified model sizes and SHA-256 hashes; GitHub release asset digests were also checked. Historical runtime verification is recorded in phase reports; tests/builds were not repeated during the submission preparation at the student's request.

## Report
[IEEE-style PDF report](report/main.pdf), generated directly as a PDF at the student's request. [YouTube demonstration](https://youtu.be/4-InqI11ZQA). Regenerate with:

```powershell
python -m pip install reportlab
python tools/build_submission_pdf.py --youtube-url https://youtu.be/4-InqI11ZQA --model-url https://github.com/AhmedLaiq34/genai-assignment-1/releases/tag/models-v1
```

Submit the PDF to Google Classroom; include the YouTube link inside it, rather than uploading the video there.

## AI use
See `docs/AI_USE_LOG.md`.
