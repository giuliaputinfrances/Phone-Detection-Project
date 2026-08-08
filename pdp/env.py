"""Environment check.

This exists because of a specific, silent failure: a CPU-only torch build
installs perfectly happily, and training then runs ~50x slower with no error
message. `pdp check-env` fails loudly instead.
"""

from __future__ import annotations

import platform
import sys


def collect() -> dict:
    info: dict = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = round(props.total_memory / 1024**3, 1)
            info["capability"] = f"{props.major}.{props.minor}"
    except ImportError:
        info["torch"] = None
        info["cuda_available"] = False

    for mod, key in (("ultralytics", "ultralytics"), ("cv2", "opencv"), ("numpy", "numpy")):
        try:
            info[key] = __import__(mod).__version__
        except ImportError:
            info[key] = None
    return info


def check(allow_cpu: bool = False) -> int:
    info = collect()
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"  {key:<{width}} : {value}")

    problems: list[str] = []
    if info.get("torch") is None:
        problems.append("torch is not installed")
    elif not info.get("cuda_available"):
        problems.append(
            "torch.cuda.is_available() is False — this is a CPU-only build.\n"
            "    Fix: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu129"
        )
    if info.get("ultralytics") is None:
        problems.append("ultralytics is not installed (pip install ultralytics)")
    if info.get("opencv") is None:
        problems.append("opencv-python is not installed")

    if not problems:
        print("\nOK: GPU training and inference are available.")
        return 0

    print("\nPROBLEMS:")
    for p in problems:
        print(f"  - {p}")
    if allow_cpu and all("cuda" in p.lower() for p in problems):
        print("\n--allow-cpu set: continuing anyway (expect ~50x slower training).")
        return 0
    return 1
