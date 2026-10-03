# Generative AI Assignment #1

> **STATUS: Scaffold only - no models implemented or trained.**

## Overview of the 4 tasks
1. Universal multi-corruption denoising autoencoder (Oxford-IIIT Pet, 128x128).
2. Corruption classifier + hard-routed specialist autoencoders.
3. Jointly trained soft mixture-of-experts restoration.
4. Style-conditioned face-to-sketch cGAN (FS2K).

Authoritative references: `docs/IMPLEMENTATION_PLAN.md`, `docs/CONTRACTS.md`, `docs/TRACEABILITY.md`.

## Repository layout
See `docs/IMPLEMENTATION_PLAN.md` section 2. TODO: short tree once stable.

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
TODO: `python scripts/prepare_pets.py`, `python scripts/prepare_fs2k.py` (not implemented).

## Training / Tuning
TODO: `python scripts/tune.py --task ...`, `python scripts/train.py --task ...` (not implemented).

## Evaluation
TODO: `python scripts/evaluate.py --task ... [--final-test]` (not implemented).

## ONNX export
TODO: `python scripts/export_onnx.py --verify` (not implemented).

## Running the app
```
docker compose up --build
```
TODO: only a health-check backend stub exists; frontend not yet built.

## Model downloads
TODO: `scripts/fetch_models.py` and links in `models/MANIFEST.json` (empty).

## Experiment tracking
TODO: Weights & Biases (proposed, see `docs/DECISIONS.md`), MLflow fallback.

## Reproducibility
Seed 42, committed splits and manifests, checkpoints record seeds and hashes (CONTRACTS 3.4, 3.8, 3.11). TODO: verify once implemented.

## Report
TODO: `report/` (IEEEtran LaTeX).

## AI use
See `docs/AI_USE_LOG.md`.
