from __future__ import annotations

import time
from pathlib import Path

import cv2

from pdp.sources.base import FrameSource
from pdp.types import Frame


class FileSource(FrameSource):
    """Reads a video file. Used for every offline/debug run."""

    def __init__(self, path: str | Path, *, stride: int = 1, loop: bool = False,
                 max_frames: int | None = None) -> None:
        self.path = Path(path)
        self.stride = max(1, int(stride))
        self.loop = loop
        self.max_frames = max_frames
        self.source_id = self.path.name
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._raw_index = 0

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"video not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video: {self.path}")

    def read(self) -> Frame | None:
        if self._cap is None:
            raise RuntimeError("read() before open()")
        if self.max_frames is not None and self._frame_id >= self.max_frames:
            return None

        while True:
            ok, img = self._cap.read()
            if not ok:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._raw_index = 0
                    continue
                return None
            self._raw_index += 1
            if (self._raw_index - 1) % self.stride == 0:
                break

        frame = Frame(
            frame_id=self._frame_id,
            ts_mono=time.monotonic(),
            image_bgr=img,
            source_id=self.source_id,
        )
        self._frame_id += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def frame_count(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0
