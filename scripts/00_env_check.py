import importlib
import platform
import sys

print(platform.platform(), sys.version)
for m in ["pandas", "numpy", "scipy", "sklearn", "torch", "optuna", "lightgbm", "ot"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:10s} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"{m:10s} MISSING ({e.__class__.__name__})")
try:
    import torch

    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception:  # noqa: BLE001
    pass
