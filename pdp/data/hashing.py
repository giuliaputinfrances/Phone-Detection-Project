"""Perceptual hashing for near-duplicate detection.

dHash (horizontal gradient hash) rather than aHash: aHash compares every pixel
to the frame's mean, so two images that share an overall brightness — a flat
wall, a corridor, anything low-texture — collide even when their content
differs. dHash encodes *relationships between adjacent pixels*, which survives
exposure changes and still separates low-detail frames. At 16x16 it produces
256 bits, giving distances enough resolution to threshold meaningfully.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

HASH_SIZE = 16
HASH_BITS = HASH_SIZE * HASH_SIZE

# Defaults expressed as a fraction of HASH_BITS so they survive a size change.
DEDUP_DISTANCE = int(0.10 * HASH_BITS)      # 25/256 - "new enough to annotate"
DUPLICATE_DISTANCE = int(0.04 * HASH_BITS)  # 10/256 - "the same moment"


def dhash(image: np.ndarray, size: int = HASH_SIZE) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int.from_bytes(np.packbits(bits.flatten()).tobytes(), "big")


def dhash_file(path: str | Path, size: int = HASH_SIZE) -> int | None:
    img = cv2.imread(str(path))
    return None if img is None else dhash(img, size)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
