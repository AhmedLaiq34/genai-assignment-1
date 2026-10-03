# Environment

## Local (recorded 2026-10-03)
| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3050 6 GB Laptop, driver 610.74 |
| RAM | 15.6 GB |
| Disk | D: 126 GB free |
| Python | 3.13.5 in `.venv` (3.14 is the system default; not used) |
| PyTorch | 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA available on RTX 3050 6 GB |
| Libraries | optuna 5.0.0, onnx 1.23.1, onnxruntime 1.30.0, wandb 0.30.0, mlflow 3.16.1, pytorch-msssim 1.0.0, numpy 2.5.2, pillow 12.3.0 |
| Docker | 29.1.5 |
| git | 2.52 |
| Node | 24.19 |

## Setup
```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install -r requirements.txt
python scripts/env_check.py
```

## Cloud devices (fill in from your own accounts; do not guess)
### Kaggle
- GPU type: TBD
- Weekly quota / session limit: TBD
- Python / torch versions: TBD

### Colab
- GPU type: TBD
- Observed disconnects / limits: TBD
- Python / torch versions: TBD

## Notes
- Scaffold scripts were named `scripts/*.py.py`; renamed to `scripts/*.py` (2026-10-03).
- Verified: `scripts/env_check.py`, CUDA tensor on GPU, import of optuna/onnx/onnxruntime/pytorch_msssim.
