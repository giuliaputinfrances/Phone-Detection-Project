"""Video -> candidate training frames.

Two things worth doing at extraction time rather than in the annotation tool:
sample at a low fps (adjacent frames carry almost no new information), and drop
near-duplicates. Annotating 400 nearly-identical frames costs real hours and
teaches the model almost nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from pdp.data.hashing import DEDUP_DISTANCE, dhash, hamming

log = logging.getLogger(__name__)


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    *,
    fps: float = 2.0,
    min_hash_distance: int = DEDUP_DISTANCE,
    jpeg_quality: int = 95,
    prefix: str | None = None,
    max_frames: int | None = None,
) -> int:
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or video.stem

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(fps, 0.01))))

    kept: list[int] = []
    idx = written = skipped = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx % step == 0:
            h = dhash(img)
            if all(hamming(h, prev) >= min_hash_distance for prev in kept):
                name = f"{prefix}_{idx:06d}.jpg"
                cv2.imwrite(str(out_dir / name), img,
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                kept.append(h)
                written += 1
                if max_frames and written >= max_frames:
                    break
            else:
                skipped += 1
        idx += 1
    cap.release()

    log.info(
        "%s: %d frames scanned, sampled every %d, wrote %d, skipped %d near-duplicates -> %s",
        video.name, idx, step, written, skipped, out_dir,
    )
    return written
