"""Data contracts shared by every stage of the pipeline.

These four types are the only things modules pass to each other. Keeping them
small and explicit is what lets a source, detector or control backend be
swapped without touching anything upstream or downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

XYXY = tuple[float, float, float, float]


@dataclass(frozen=True, eq=False)
class Frame:
    """A single captured image plus the metadata needed to trace its latency."""

    frame_id: int
    ts_mono: float  # time.monotonic() stamped at capture, never re-stamped
    image_bgr: np.ndarray
    source_id: str

    @property
    def height(self) -> int:
        return int(self.image_bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.image_bgr.shape[1])


@dataclass(frozen=True)
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: XYXY  # pixels, in the coordinate space of the source frame
    track_id: int | None = None

    @property
    def cx(self) -> float:
        return (self.xyxy[0] + self.xyxy[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.xyxy[1] + self.xyxy[3]) / 2.0

    @property
    def w(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def h(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cls_id": self.cls_id,
            "cls_name": self.cls_name,
            "conf": round(self.conf, 4),
            "xyxy": [round(v, 1) for v in self.xyxy],
            "track_id": self.track_id,
        }


@dataclass(frozen=True, eq=False)
class DetectionResult:
    frame: Frame
    detections: list[Detection]
    infer_ms: float
    model_id: str

    def to_event(self) -> dict[str, Any]:
        """Serializable record for the JSONL log (drops the image)."""
        return {
            "frame_id": self.frame.frame_id,
            "ts_mono": round(self.frame.ts_mono, 6),
            "source_id": self.frame.source_id,
            "width": self.frame.width,
            "height": self.frame.height,
            "infer_ms": round(self.infer_ms, 2),
            "model_id": self.model_id,
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass(frozen=True)
class Command:
    kind: Literal["servo"]
    channel: int
    target_deg: float
    speed_dps: float | None
    ts_mono: float
    reason: str  # human-readable trace, e.g. "track=14 cls=cone zone=left"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "channel": self.channel,
            "target_deg": round(self.target_deg, 2),
            "speed_dps": self.speed_dps,
            "ts_mono": round(self.ts_mono, 6),
            "reason": self.reason,
        }
