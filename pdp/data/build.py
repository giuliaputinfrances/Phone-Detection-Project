"""Build a versioned, trainable dataset out of raw annotation exports.

Input layout — one directory per *capture session* under `datasets/raw/`:

    datasets/raw/2026-08-08_hallway/
        images/*.jpg
        labels/*.txt        (or .txt files sitting next to the images)

Sessions are the unit of splitting, and that is the whole point. Consecutive
video frames are near-identical; a random frame-level split puts the same moment
in both train and val and your val mAP becomes fiction. Whole sessions go to
exactly one split.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pdp.data.validate import IMAGE_EXTS, validate_dataset

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass
class Session:
    name: str
    root: Path
    images: list[Path]

    @property
    def count(self) -> int:
        return len(self.images)


def discover_sessions(raw_root: Path) -> list[Session]:
    sessions: list[Session] = []
    for d in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        images_dir = d / "images" if (d / "images").is_dir() else d
        images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        if images:
            sessions.append(Session(d.name, d, images))
        else:
            log.warning("session %s has no images, skipping", d.name)
    return sessions


def _label_for(image: Path, session: Session) -> Path | None:
    """Find the .txt for an image: sibling labels/ dir, or next to the image."""
    candidates = [
        session.root / "labels" / f"{image.stem}.txt",
        image.with_suffix(".txt"),
    ]
    try:
        rel = image.relative_to(session.root / "images")
        candidates.insert(0, session.root / "labels" / rel.with_suffix(".txt"))
    except ValueError:
        pass
    for c in candidates:
        if c.exists():
            return c
    return None


def _assign(sessions: list[Session], val: set[str], test: set[str], seed: int,
            val_ratio: float, test_ratio: float) -> dict[str, str]:
    """Explicit assignment wins; the rest split deterministically by name hash."""
    out: dict[str, str] = {}
    for s in sessions:
        if s.name in val:
            out[s.name] = "val"
        elif s.name in test:
            out[s.name] = "test"

    unassigned = [s for s in sessions if s.name not in out]
    for s in unassigned:
        digest = hashlib.sha256(f"{seed}:{s.name}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        if bucket < test_ratio:
            out[s.name] = "test"
        elif bucket < test_ratio + val_ratio:
            out[s.name] = "val"
        else:
            out[s.name] = "train"
    return out


def build_dataset(
    raw_root: str | Path,
    out_root: str | Path,
    classes: dict[int, str],
    *,
    val_sessions: list[str] | None = None,
    test_sessions: list[str] | None = None,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 0,
    copy: bool = True,
    overwrite: bool = False,
    check_duplicates: bool = True,
) -> dict:
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw dataset root not found: {raw_root}")

    if out_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out_root} already exists. Datasets are versioned, not mutated: "
                f"bump the version (…_v2) or pass --overwrite."
            )
        shutil.rmtree(out_root)

    sessions = discover_sessions(raw_root)
    if not sessions:
        raise RuntimeError(f"no sessions with images found under {raw_root}")

    assignment = _assign(
        sessions, set(val_sessions or []), set(test_sessions or []),
        seed, val_ratio, test_ratio,
    )

    for split in SPLITS:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict] = {}
    unlabeled_total = 0
    for s in sessions:
        split = assignment[s.name]
        img_dst = out_root / "images" / split
        lbl_dst = out_root / "labels" / split
        copied = unlabeled = 0

        for img in s.images:
            # Prefix with the session so identically-named frames from different
            # sessions can't overwrite each other.
            stem = f"{s.name}__{img.stem}"
            target = img_dst / f"{stem}{img.suffix.lower()}"
            if copy:
                shutil.copy2(img, target)
            else:
                try:
                    target.hardlink_to(img)
                except OSError:
                    shutil.copy2(img, target)

            lbl = _label_for(img, s)
            if lbl is not None:
                shutil.copy2(lbl, lbl_dst / f"{stem}.txt")
            else:
                unlabeled += 1
            copied += 1

        unlabeled_total += unlabeled
        stats[s.name] = {"split": split, "images": copied, "unlabeled": unlabeled}
        log.info("session %-32s -> %-5s  %4d images (%d unlabeled)",
                 s.name, split, copied, unlabeled)

    if unlabeled_total:
        log.warning(
            "%d images have no label file. They will train as background negatives — "
            "intended, or an incomplete annotation export?", unlabeled_total,
        )

    # Remove empty test/ so ultralytics doesn't trip over a dangling path.
    for split in SPLITS:
        if not any((out_root / "images" / split).iterdir()):
            shutil.rmtree(out_root / "images" / split)
            shutil.rmtree(out_root / "labels" / split, ignore_errors=True)

    data_yaml = write_data_yaml(out_root, classes)
    report = validate_dataset(out_root, len(classes), check_duplicates=check_duplicates)

    manifest = {
        "name": out_root.name,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_root": str(raw_root),
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "classes": {str(k): v for k, v in sorted(classes.items())},
        "sessions": stats,
        "totals": {
            "images": report.images,
            "labeled": report.labels,
            "backgrounds": report.backgrounds,
            "boxes": report.boxes,
        },
        "class_counts": {classes[k]: v for k, v in sorted(report.class_counts.items())
                         if k in classes},
        "valid": report.ok,
        "errors": report.errors[:50],
        "warnings": report.warnings[:50],
    }
    (out_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    log.info("wrote %s and %s", data_yaml, out_root / "MANIFEST.json")
    return manifest


def write_data_yaml(root: Path, classes: dict[int, str]) -> Path:
    """Generated from configs/classes.yaml — never hand-edit the result."""
    doc: dict = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {int(k): v for k, v in sorted(classes.items())},
    }
    if (root / "images" / "test").exists():
        doc["test"] = "images/test"

    path = root / "data.yaml"
    header = (
        "# GENERATED by `pdp build-dataset` from configs/classes.yaml.\n"
        "# Do not edit by hand: rebuild the dataset instead.\n"
    )
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path
