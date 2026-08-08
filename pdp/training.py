"""Train / validate / export / benchmark YOLO26.

Runs are driven entirely by a config file so they are reproducible and diffable
against each other. After training we snapshot the git SHA, the resolved args
and the dataset MANIFEST into the run directory — six weeks from now, "which
data produced the good model?" has to be answerable from the run dir alone.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

DEFAULT_TRAIN_ARGS: dict[str, Any] = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "workers": 4,
    "amp": True,
    "cache": "ram",
    "seed": 0,
    "deterministic": True,
    "patience": 20,
    "project": "runs/detect",
    # Fine-tuning defaults for a small, COCO-adjacent dataset. freeze=10 keeps
    # the backbone; AdamW + lr0=1e-3 is steadier than auto on a few hundred
    # images; mosaic is halved because heavy augmentation hurts small sets.
    "freeze": 10,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "mosaic": 0.5,
}


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return "unknown"
    # In a repo with no commits, `rev-parse HEAD` echoes "HEAD" on stdout and
    # reports the failure on stderr — don't record that as a revision.
    if len(out) == 40 and all(c in "0123456789abcdef" for c in out):
        return out
    return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return bool(out)
    except Exception:
        return False


def load_train_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"train config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    unknown = set(raw) - {"name", "model", "data", "args"}
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}")
    if "data" not in raw:
        raise ValueError(f"{path}: 'data' (path to dataset yaml) is required")

    args = dict(DEFAULT_TRAIN_ARGS)
    args.update(raw.get("args") or {})
    return {
        "name": raw.get("name", path.stem),
        "model": raw.get("model", "yolo26n.pt"),
        "data": raw["data"],
        "args": args,
    }


def _snapshot(run_dir: Path, cfg: dict, data_yaml: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": cfg,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass

    (run_dir / "pdp_run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    manifest = data_yaml.parent / "MANIFEST.json"
    if manifest.exists():
        shutil.copy2(manifest, run_dir / "dataset_MANIFEST.json")


def train(config_path: str | Path, **overrides: Any) -> dict:
    from ultralytics import YOLO

    cfg = load_train_config(config_path)
    cfg["args"].update({k: v for k, v in overrides.items() if v is not None})

    data_yaml = Path(cfg["data"])
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"dataset yaml not found: {data_yaml}. Run `pdp build-dataset` first."
        )

    args = dict(cfg["args"])
    args.setdefault("name", cfg["name"])
    run_dir = Path(args["project"]) / args["name"]

    log.info("training %s on %s -> %s", cfg["model"], data_yaml, run_dir)
    model = YOLO(cfg["model"])
    t0 = time.monotonic()
    results = model.train(data=str(data_yaml), **args)
    minutes = (time.monotonic() - t0) / 60.0

    # Ultralytics may add a suffix (name2, name3) if the dir existed.
    actual = Path(getattr(results, "save_dir", run_dir))
    _snapshot(actual, cfg, data_yaml)

    best = actual / "weights" / "best.pt"
    log.info("training finished in %.1f min; best weights: %s", minutes, best)
    return {"run_dir": str(actual), "best": str(best), "minutes": round(minutes, 1)}


def validate(weights: str | Path, data: str | Path, *, imgsz: int = 640,
             device: str | int = 0, split: str = "val") -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data), imgsz=imgsz, device=device, split=split, verbose=True)

    box = metrics.box
    per_class = {}
    names = getattr(metrics, "names", {}) or model.names
    try:
        for i, cls_idx in enumerate(box.ap_class_index):
            per_class[names[int(cls_idx)]] = {
                "AP50": round(float(box.ap50[i]), 4),
                "AP50-95": round(float(box.ap[i]), 4),
                "precision": round(float(box.p[i]), 4),
                "recall": round(float(box.r[i]), 4),
            }
    except Exception:  # pragma: no cover - metric layout varies by task
        log.warning("could not extract per-class metrics", exc_info=True)

    out = {
        "split": split,
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
    }
    log.info("val %s: mAP50=%.4f mAP50-95=%.4f", split, out["mAP50"], out["mAP50-95"])
    return out


def export(weights: str | Path, fmt: str = "onnx", *, imgsz: int = 640,
           quantize: int | None = 16, device: str | int = 0, **kwargs: Any) -> str:
    """Export for deployment.

    YOLO26's NMS-free head makes this much cleaner than earlier versions — there
    is no NMS plugin to wire up. Note that a TensorRT .engine is built for one
    specific GPU + driver + resolution and is not portable: the .pt stays the
    artifact of record, the .engine is a build product.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    path = model.export(format=fmt, imgsz=imgsz, quantize=quantize, device=device, **kwargs)
    log.info("exported %s -> %s", weights, path)
    return str(path)


def benchmark(weights: str | Path, *, imgsz: int = 640, device: str | int = 0,
              runs: int = 100, warmup: int = 10, quantize: int | None = 16) -> dict:
    """Latency at the real deployment resolution. p95 is what control cares about."""
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(weights))
    blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    kwargs = dict(imgsz=imgsz, device=device, quantize=quantize, verbose=False)

    for _ in range(warmup):
        model.predict(blank, **kwargs)

    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(blank, **kwargs)
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    out = {
        "weights": str(weights),
        "imgsz": imgsz,
        "runs": runs,
        "mean_ms": round(sum(samples) / len(samples), 2),
        "p50_ms": round(samples[len(samples) // 2], 2),
        "p95_ms": round(samples[int(0.95 * len(samples))], 2),
        "max_ms": round(samples[-1], 2),
    }
    out["fps"] = round(1000.0 / out["mean_ms"], 1)
    log.info("bench %s: %s", weights, json.dumps(out))
    return out
