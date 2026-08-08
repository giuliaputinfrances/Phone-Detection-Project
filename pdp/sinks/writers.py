from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from pdp.types import Command, DetectionResult

log = logging.getLogger(__name__)


class VideoWriterSink:
    """Annotated MP4. Opened lazily so the frame size comes from real data."""

    def __init__(self, path: str | Path, fps: float = 30.0) -> None:
        self.path = Path(path)
        self.fps = fps if fps and fps > 0 else 30.0
        self._writer: cv2.VideoWriter | None = None
        self.frames = 0

    def write(self, image: np.ndarray) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open video writer: {self.path}")
            log.info("writing annotated video -> %s (%dx%d @ %.1f)", self.path, w, h, self.fps)
        self._writer.write(image)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info("wrote %d frames -> %s", self.frames, self.path)


class JsonlSink:
    """One JSON object per frame: detections plus any commands they produced."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, result: DetectionResult, commands: list[Command] | None = None) -> None:
        event = result.to_event()
        if commands:
            event["commands"] = [c.to_dict() for c in commands]
        self._fh.write(json.dumps(event) + "\n")
        self.count += 1

    def close(self) -> None:
        self._fh.close()
        log.info("wrote %d events -> %s", self.count, self.path)


class Metrics:
    """Latency/FPS accounting. p95 matters more than the mean for control."""

    def __init__(self, window: int = 300) -> None:
        self.window = window
        self.infer_ms: list[float] = []
        self.loop_ms: list[float] = []
        self.frames = 0
        self.detections = 0
        self._t_start = time.monotonic()
        self._t_last = self._t_start

    def tick(self, result: DetectionResult) -> None:
        now = time.monotonic()
        self.frames += 1
        self.detections += len(result.detections)
        self.infer_ms.append(result.infer_ms)
        self.loop_ms.append((now - self._t_last) * 1000.0)
        self._t_last = now
        if len(self.infer_ms) > self.window:
            del self.infer_ms[: -self.window]
            del self.loop_ms[: -self.window]

    @staticmethod
    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        return s[min(len(s) - 1, int(0.95 * len(s)))]

    @property
    def fps(self) -> float:
        elapsed = time.monotonic() - self._t_start
        return self.frames / elapsed if elapsed > 0 else 0.0

    def hud(self) -> str:
        mean = statistics.fmean(self.infer_ms) if self.infer_ms else 0.0
        return (
            f"{self.fps:5.1f} FPS | infer {mean:5.1f} ms (p95 {self._p95(self.infer_ms):5.1f})"
        )

    def summary(self) -> dict[str, float]:
        return {
            "frames": self.frames,
            "detections": self.detections,
            "fps": round(self.fps, 2),
            "infer_ms_mean": round(statistics.fmean(self.infer_ms), 2) if self.infer_ms else 0.0,
            "infer_ms_p95": round(self._p95(self.infer_ms), 2),
            "loop_ms_mean": round(statistics.fmean(self.loop_ms), 2) if self.loop_ms else 0.0,
            "loop_ms_p95": round(self._p95(self.loop_ms), 2),
        }
