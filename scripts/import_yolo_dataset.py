"""Import a downloaded YOLO dataset into datasets/raw/ as train+val sessions.

Reports the class ids actually present, optionally remaps them, and splits the
files into two session folders so `pdp build-dataset` has both splits.

    python import_yolo_dataset.py --images D:/dl/balloons/images \
                                  --labels D:/dl/balloons/labels \
                                  --name balloons

    # a hand dataset that uses id 0 internally, but must become id 1 here:
    python import_yolo_dataset.py --images ... --labels ... --name hands --remap 0=1

Pass --dry-run first to see what it would do.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOWS_SAFE_PATH_CHARS = 240


def path_len(path: Path) -> int:
    try:
        return len(str(path.resolve(strict=False)))
    except OSError:
        return len(str(path.absolute()))


def imported_image_name(
    img: Path, img_dst: Path, index: int, *, short_names: bool
) -> str:
    original = img.name
    if short_names or path_len(img_dst / original) >= WINDOWS_SAFE_PATH_CHARS:
        return f"img_{index:06d}{img.suffix.lower()}"
    return original


def parse_remap(pairs: list[str]) -> dict[int, int]:
    out: dict[int, int] = {}
    for p in pairs:
        old, new = p.split("=")
        out[int(old)] = int(new)
    return out


def read_label(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append(parts)
        elif parts:
            print(f"  WARN {path.name}: skipping malformed line {line!r}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--name", required=True, help="session base name, e.g. balloons")
    ap.add_argument("--out", default="datasets/raw")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--remap", nargs="*", default=[], help="old=new, e.g. 0=1")
    ap.add_argument("--limit", type=int, default=None,
                    help="use at most N images (keeps classes balanced across datasets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--short-names", action="store_true",
                    help="rename imported files to img_000001.jpg style names")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing output sessions for this name")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    images_dir, labels_dir = Path(a.images), Path(a.labels)
    for d in (images_dir, labels_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}")
            return 1

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"ERROR: no images found in {images_dir}")
        return 1

    remap = parse_remap(a.remap)
    counts: Counter = Counter()
    missing = 0
    pairs: list[tuple[Path, Path | None]] = []
    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            lbl = None
            missing += 1
        else:
            for row in read_label(lbl):
                counts[int(row[0])] += 1
        pairs.append((img, lbl))

    print(f"{len(images)} images, {len(images) - missing} with labels, {missing} without")
    print("class ids present in the source labels:")
    for cid, n in sorted(counts.items()):
        arrow = f"  ->  {remap[cid]}" if cid in remap else ""
        print(f"  id {cid}: {n} boxes{arrow}")
    if remap:
        unmapped = sorted(set(counts) - set(remap))
        if unmapped:
            print(f"  (ids {unmapped} left unchanged)")

    rng = random.Random(a.seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    if a.limit and a.limit < len(shuffled):
        print(f"limiting to {a.limit} of {len(shuffled)} images")
        shuffled = shuffled[: a.limit]
    n_val = max(1, int(len(shuffled) * a.val_ratio))
    splits = {f"{a.name}_val": shuffled[:n_val], f"{a.name}_train": shuffled[n_val:]}

    out_root = Path(a.out)
    session_roots = [out_root / session for session in splits]
    if not a.dry_run:
        existing = [p for p in session_roots if p.exists()]
        if existing and not a.overwrite:
            print("ERROR: output session(s) already exist:")
            for p in existing:
                print(f"  {p}")
            print("Pass --overwrite to replace them, or use a different --name/--out.")
            return 1
        for p in existing:
            shutil.rmtree(p)

    shortened_total = 0
    for session, items in splits.items():
        print(f"{session}: {len(items)} images")
        img_dst = out_root / session / "images"
        lbl_dst = out_root / session / "labels"
        planned_names = [
            imported_image_name(img, img_dst, i, short_names=a.short_names)
            for i, (img, _) in enumerate(items)
        ]
        shortened_session = sum(
            1 for (img, _), name in zip(items, planned_names) if name != img.name
        )
        shortened_total += shortened_session
        if shortened_session:
            print(f"  shortened {shortened_session} filename(s)")
        if a.dry_run:
            continue
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for (img, lbl), image_name in zip(items, planned_names):
            target_img = img_dst / image_name
            try:
                shutil.copy2(img, target_img)
            except FileNotFoundError:
                print("ERROR: failed to copy image:")
                print(f"  source ({path_len(img)} chars): {img}")
                print(f"  dest   ({path_len(target_img)} chars): {target_img}")
                print("Try --short-names, a shorter --out path, or a shorter --name.")
                return 1
            if lbl is None:
                continue
            rows = read_label(lbl)
            for row in rows:
                row[0] = str(remap.get(int(row[0]), int(row[0])))
            (lbl_dst / f"{Path(image_name).stem}.txt").write_text(
                "".join(" ".join(r) + "\n" for r in rows), encoding="utf-8"
            )

    if a.dry_run:
        print("\n(dry run — nothing written)")
    else:
        print(f"\nwrote sessions under {out_root}")
    if shortened_total:
        print("short names keep the imported dataset under Windows path limits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
