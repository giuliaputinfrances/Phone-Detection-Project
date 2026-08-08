"""Dataset validation.

Every check here corresponds to a failure that is nearly invisible at training
time and obvious only weeks later: a stray class id shifts every label, an
out-of-range coordinate silently clips, an orphaned label file means an image
you thought was annotated never was, and duplicate images across splits inflate
val mAP into fiction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pdp.data.hashing import DUPLICATE_DISTANCE, dhash_file, hamming

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_BOX_NORM = 0.005  # ~3 px on a 640 image; smaller is almost always a mis-click


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)
    images: int = 0
    labels: int = 0
    backgrounds: int = 0
    boxes: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: Report) -> None:
        self.errors += other.errors
        self.warnings += other.warnings
        self.class_counts += other.class_counts
        self.images += other.images
        self.labels += other.labels
        self.backgrounds += other.backgrounds
        self.boxes += other.boxes


def label_path_for(image: Path) -> Path:
    """images/<split>/x.jpg -> labels/<split>/x.txt (Ultralytics convention)."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def validate_label_file(path: Path, num_classes: int) -> tuple[list[str], list[str], Counter, int]:
    errors: list[str] = []
    warnings: list[str] = []
    counts: Counter = Counter()
    boxes = 0

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls_id = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{path}:{lineno}: non-numeric field in {line!r}")
            continue

        if not 0 <= cls_id < num_classes:
            errors.append(
                f"{path}:{lineno}: class id {cls_id} outside 0..{num_classes - 1}"
            )
        for name, v in (("x", x), ("y", y), ("w", w), ("h", h)):
            if not 0.0 <= v <= 1.0:
                errors.append(f"{path}:{lineno}: {name}={v} outside [0,1] (not normalized?)")
        if w <= 0 or h <= 0:
            errors.append(f"{path}:{lineno}: zero/negative box {w}x{h}")
        elif w < MIN_BOX_NORM or h < MIN_BOX_NORM:
            warnings.append(f"{path}:{lineno}: very small box {w:.4f}x{h:.4f}")
        if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            warnings.append(f"{path}:{lineno}: box extends past the image edge")

        counts[cls_id] += 1
        boxes += 1

    return errors, warnings, counts, boxes


def validate_split(images_dir: Path, num_classes: int) -> Report:
    rep = Report()
    if not images_dir.exists():
        rep.errors.append(f"missing images dir: {images_dir}")
        return rep

    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    rep.images = len(images)
    if not images:
        rep.errors.append(f"no images found in {images_dir}")
        return rep

    labels_dir = Path(str(images_dir).replace("images", "labels", 1)) \
        if "images" in images_dir.parts else images_dir
    seen_labels: set[Path] = set()

    for img in images:
        lbl = label_path_for(img)
        if not lbl.exists():
            rep.backgrounds += 1  # legitimately a negative sample
            continue
        seen_labels.add(lbl.resolve())
        rep.labels += 1
        errs, warns, counts, boxes = validate_label_file(lbl, num_classes)
        rep.errors += errs
        rep.warnings += warns
        rep.class_counts += counts
        rep.boxes += boxes

    if labels_dir.exists():
        for lbl in labels_dir.rglob("*.txt"):
            if lbl.resolve() not in seen_labels:
                rep.errors.append(f"orphan label with no matching image: {lbl}")

    if rep.backgrounds and rep.images:
        pct = 100.0 * rep.backgrounds / rep.images
        if pct > 40:
            rep.warnings.append(
                f"{pct:.0f}% of images have no labels — intentional negatives, or "
                "did the annotation export go wrong?"
            )
    return rep


def find_cross_split_duplicates(
    root: Path, splits: list[str], max_distance: int = DUPLICATE_DISTANCE
) -> list[str]:
    """Near-duplicate images in two different splits = train/val leakage."""
    hashes: dict[str, list[tuple[int, Path]]] = {}
    for split in splits:
        d = root / "images" / split
        if not d.exists():
            continue
        entries = []
        for img in sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            h = dhash_file(img)
            if h is not None:
                entries.append((h, img))
        hashes[split] = entries

    issues: list[str] = []
    names = list(hashes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for ha, pa in hashes[a]:
                for hb, pb in hashes[b]:
                    if hamming(ha, hb) <= max_distance:
                        issues.append(f"near-duplicate across {a}/{b}: {pa.name} ~ {pb.name}")
    return issues


def validate_dataset(root: Path, num_classes: int, *, check_duplicates: bool = True) -> Report:
    rep = Report()
    splits = [s for s in ("train", "val", "test") if (root / "images" / s).exists()]
    if "train" not in splits or "val" not in splits:
        rep.errors.append(f"{root}: need at least images/train and images/val")
        return rep

    for split in splits:
        sub = validate_split(root / "images" / split, num_classes)
        sub.errors = [f"[{split}] {e}" for e in sub.errors]
        sub.warnings = [f"[{split}] {w}" for w in sub.warnings]
        rep.merge(sub)

    if check_duplicates:
        for issue in find_cross_split_duplicates(root, splits):
            rep.errors.append(f"[leakage] {issue}")

    return rep
