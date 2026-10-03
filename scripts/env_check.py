"""Print environment information; tolerates a missing torch install."""
import platform
import shutil
import sys


def main() -> None:
    print("python      :", sys.version.split()[0], platform.platform())
    try:
        import psutil  # optional
        print("RAM (GB)    :", round(psutil.virtual_memory().total / 2**30, 1))
    except Exception:
        pass
    print("disk free GB:", round(shutil.disk_usage(".").free / 2**30, 1))
    try:
        import torch
    except ImportError:
        print("torch       : NOT INSTALLED")
        return
    print("torch       :", torch.__version__, "| cuda build:", torch.version.cuda)
    ok = torch.cuda.is_available()
    print("cuda avail  :", ok)
    if ok:
        p = torch.cuda.get_device_properties(0)
        print("GPU         :", p.name, "| VRAM GB:", round(p.total_memory / 2**30, 2))
    for mod in ("optuna", "wandb", "onnxruntime", "pytorch_msssim"):
        try:
            m = __import__(mod)
            print(f"{mod:12s}:", getattr(m, "__version__", "ok"))
        except ImportError:
            print(f"{mod:12s}: not installed")


if __name__ == "__main__":
    main()
