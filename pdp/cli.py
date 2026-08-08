"""Single entry point: `pdp <command>`.

    pdp check-env                              verify CUDA/torch/ultralytics
    pdp cameras                                list capture devices (find Camo)
    pdp extract-frames VIDEO -o DIR            video -> deduped candidate frames
    pdp build-dataset -r RAW -n NAME           raw exports -> versioned dataset
    pdp validate-dataset DATASET               re-run the checks on a built set
    pdp train -c configs/train/x.yaml          fine-tune YOLO26
    pdp val -w best.pt -d data.yaml            evaluate
    pdp export -w best.pt -f engine            deployment artifact
    pdp bench -w best.pt                       latency at deployment resolution
    pdp predict -c configs/runtime/offline.yaml   run on a video file
    pdp live -c configs/runtime/live.yaml         run on the Camo feed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    # pygrabber's COM bindings generate type-library stubs on first use and
    # narrate all of it at INFO.
    logging.getLogger("comtypes").setLevel(logging.WARNING)


# --- commands -------------------------------------------------------------


def cmd_check_env(a: argparse.Namespace) -> int:
    from pdp.env import check

    return check(allow_cpu=a.allow_cpu)


def cmd_cameras(a: argparse.Namespace) -> int:
    from pdp.sources.webcam import list_devices

    devices = list_devices()
    if not devices:
        print("No video devices found (is pygrabber installed? is Camo Studio running?)")
        return 1
    print("Video capture devices:")
    for i, name in enumerate(devices):
        marker = "  <-- Camo" if "camo" in name.lower() else ""
        print(f"  [{i}] {name}{marker}")
    return 0


def cmd_extract_frames(a: argparse.Namespace) -> int:
    from pdp.data import extract_frames

    videos = [Path(a.video)] if Path(a.video).is_file() else sorted(
        p for p in Path(a.video).glob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
    )
    if not videos:
        print(f"no videos found at {a.video}", file=sys.stderr)
        return 1
    total = sum(
        extract_frames(v, a.out, fps=a.fps, min_hash_distance=a.min_distance,
                       max_frames=a.max_frames)
        for v in videos
    )
    print(f"\n{total} frames written to {a.out}")
    return 0


def cmd_build_dataset(a: argparse.Namespace) -> int:
    from pdp.config import load_classes
    from pdp.data import build_dataset

    classes = load_classes(a.classes)
    manifest = build_dataset(
        a.raw,
        Path(a.datasets_dir) / a.name,
        classes,
        val_sessions=a.val_sessions,
        test_sessions=a.test_sessions,
        val_ratio=a.val_ratio,
        test_ratio=a.test_ratio,
        seed=a.seed,
        copy=not a.link,
        overwrite=a.overwrite,
        check_duplicates=not a.skip_duplicate_check,
    )
    _print_dataset_summary(manifest)
    return 0 if manifest["valid"] else 1


def _print_dataset_summary(manifest: dict) -> None:
    print("\n" + "=" * 60)
    print(f"dataset: {manifest['name']}")
    totals = manifest["totals"]
    print(f"  images {totals['images']}  labeled {totals['labeled']}  "
          f"backgrounds {totals['backgrounds']}  boxes {totals['boxes']}")
    print("  class counts:")
    counts = manifest["class_counts"]
    if counts:
        widest = max(len(k) for k in counts)
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {name:<{widest}}  {n}")
        low = [n for n, c in counts.items() if c < 150]
        if low:
            print(f"  NOTE: under 150 instances for {low} — expect weak AP on these.")
        missing = [n for n in manifest["classes"].values() if n not in counts]
        if missing:
            print(f"  WARNING: zero instances for {missing}")
    else:
        print("    (none)")

    for w in manifest["warnings"]:
        print(f"  WARN  {w}")
    for e in manifest["errors"]:
        print(f"  ERROR {e}")
    print("=" * 60)
    print("VALID" if manifest["valid"] else "INVALID — fix the errors above before training")


def cmd_validate_dataset(a: argparse.Namespace) -> int:
    from pdp.config import load_classes
    from pdp.data import validate_dataset

    classes = load_classes(a.classes)
    report = validate_dataset(Path(a.dataset), len(classes),
                              check_duplicates=not a.skip_duplicate_check)
    print(f"images {report.images}  labeled {report.labels}  "
          f"backgrounds {report.backgrounds}  boxes {report.boxes}")
    for name, count in sorted(report.class_counts.items()):
        print(f"  {classes.get(name, name)}: {count}")
    for w in report.warnings:
        print(f"WARN  {w}")
    for e in report.errors:
        print(f"ERROR {e}")
    print("VALID" if report.ok else "INVALID")
    return 0 if report.ok else 1


def cmd_train(a: argparse.Namespace) -> int:
    from pdp.training import train

    out = train(a.config, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, device=a.device)
    print(json.dumps(out, indent=2))
    return 0


def cmd_val(a: argparse.Namespace) -> int:
    from pdp.training import validate

    out = validate(a.weights, a.data, imgsz=a.imgsz, device=a.device, split=a.split)
    print(json.dumps(out, indent=2))
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    from pdp.training import export

    print(export(a.weights, a.format, imgsz=a.imgsz, quantize=a.quantize, device=a.device))
    return 0


def cmd_bench(a: argparse.Namespace) -> int:
    from pdp.training import benchmark

    print(json.dumps(
        benchmark(a.weights, imgsz=a.imgsz, device=a.device, runs=a.runs,
                  quantize=a.quantize),
        indent=2,
    ))
    return 0


def _run_pipeline(a: argparse.Namespace, *, force_preview: bool | None = None) -> int:
    from pdp.config import load_runtime_config
    from pdp.pipeline import run

    cfg = load_runtime_config(a.config)
    if a.source:
        cfg.source.kind = "file"
        cfg.source.path = a.source
    if a.weights:
        cfg.detector.weights = a.weights
    if a.conf is not None:
        cfg.detector.conf = a.conf
    if a.video_out:
        cfg.sinks.video_out = a.video_out
    if a.events_out:
        cfg.sinks.events_out = a.events_out
    if force_preview is not None:
        cfg.sinks.preview = force_preview
    if a.no_preview:
        cfg.sinks.preview = False
    cfg.validate()

    summary = run(cfg)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_predict(a: argparse.Namespace) -> int:
    return _run_pipeline(a)


def cmd_live(a: argparse.Namespace) -> int:
    return _run_pipeline(a, force_preview=not a.no_preview)


# --- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdp", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("check-env", help="verify torch/CUDA/ultralytics")
    s.add_argument("--allow-cpu", action="store_true", help="don't fail when CUDA is missing")
    s.set_defaults(func=cmd_check_env)

    s = sub.add_parser("cameras", help="list capture devices")
    s.set_defaults(func=cmd_cameras)

    s = sub.add_parser("extract-frames", help="video(s) -> deduped candidate frames")
    s.add_argument("video", help="video file or a directory of videos")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--fps", type=float, default=2.0, help="sampling rate (default 2)")
    s.add_argument("--min-distance", type=int, default=6,
                   help="min aHash distance to keep a frame (default 6)")
    s.add_argument("--max-frames", type=int, default=None)
    s.set_defaults(func=cmd_extract_frames)

    s = sub.add_parser("build-dataset", help="raw session exports -> versioned dataset")
    s.add_argument("-r", "--raw", default="datasets/raw")
    s.add_argument("-n", "--name", required=True, help="e.g. obstacles_v1")
    s.add_argument("--datasets-dir", default="datasets")
    s.add_argument("--classes", default="configs/classes.yaml")
    s.add_argument("--val-sessions", nargs="*", default=None)
    s.add_argument("--test-sessions", nargs="*", default=None)
    s.add_argument("--val-ratio", type=float, default=0.2)
    s.add_argument("--test-ratio", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--link", action="store_true", help="hardlink instead of copying")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--skip-duplicate-check", action="store_true")
    s.set_defaults(func=cmd_build_dataset)

    s = sub.add_parser("validate-dataset", help="re-run checks on a built dataset")
    s.add_argument("dataset")
    s.add_argument("--classes", default="configs/classes.yaml")
    s.add_argument("--skip-duplicate-check", action="store_true")
    s.set_defaults(func=cmd_validate_dataset)

    s = sub.add_parser("train", help="fine-tune YOLO26")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--epochs", type=int, default=None)
    s.add_argument("--batch", type=int, default=None)
    s.add_argument("--imgsz", type=int, default=None)
    s.add_argument("--device", default=None)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("val", help="evaluate weights on a split")
    s.add_argument("-w", "--weights", required=True)
    s.add_argument("-d", "--data", required=True)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--device", default=0)
    s.add_argument("--split", default="val", choices=["train", "val", "test"])
    s.set_defaults(func=cmd_val)

    s = sub.add_parser("export", help="export for deployment")
    s.add_argument("-w", "--weights", required=True)
    s.add_argument("-f", "--format", default="onnx",
                   choices=["onnx", "engine", "openvino", "torchscript", "coreml"])
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--device", default=0)
    s.add_argument("--quantize", type=int, default=16, choices=[8, 16, 32],
                   help="16 = FP16 (default), 8 = INT8, 32 = FP32")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("bench", help="measure inference latency")
    s.add_argument("-w", "--weights", required=True)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--device", default=0)
    s.add_argument("--runs", type=int, default=100)
    s.add_argument("--quantize", type=int, default=16, choices=[8, 16, 32])
    s.set_defaults(func=cmd_bench)

    for cmd, fn, helptext in (
        ("predict", cmd_predict, "run the pipeline on a video file"),
        ("live", cmd_live, "run the pipeline on the Camo/webcam feed"),
    ):
        s = sub.add_parser(cmd, help=helptext)
        default_cfg = f"configs/runtime/{'offline' if cmd == 'predict' else 'live'}.yaml"
        s.add_argument("-c", "--config", default=default_cfg)
        s.add_argument("-s", "--source", default=None, help="override source.path")
        s.add_argument("-w", "--weights", default=None)
        s.add_argument("--conf", type=float, default=None)
        s.add_argument("--video-out", default=None)
        s.add_argument("--events-out", default=None)
        s.add_argument("--no-preview", action="store_true")
        s.set_defaults(func=fn)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger("pdp").error("%s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
