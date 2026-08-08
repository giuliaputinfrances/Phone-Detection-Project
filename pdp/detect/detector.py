"""Ultralytics YOLO26 wrapper.

Everything model-specific is contained here. Downstream code only ever sees
`DetectionResult`, so switching weights, switching to a TensorRT `.engine`, or
turning tracking on/off changes nothing outside this file.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from pdp.types import Detection, DetectionResult, Frame

log = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """'auto' -> cuda:0 when available, else cpu."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


class Detector:
    def __init__(
        self,
        weights: str | Path = "yolo26n.pt",
        *,
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        max_det: int = 100,
        quantize: int | str | None = 16,
        classes: list[int] | None = None,
        tracker: str | None = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLO

        self.weights = str(weights)
        self.device = resolve_device(device)
        self.imgsz = imgsz
        # YOLO26's default head is NMS-free (one-to-one), so there is no `iou`
        # knob to tune: `conf` alone controls the precision/recall tradeoff and
        # `max_det` caps the output directly.
        self.conf = conf
        self.max_det = max_det
        # ultralytics 8.4 replaced half/int8 with a single `quantize` scheme
        # (16 = FP16, 8 = INT8, None = FP32). FP16 is CUDA-only.
        self.quantize = quantize if self.device.startswith("cuda") else None
        self.classes = classes
        self.tracker = tracker

        self.model = YOLO(self.weights)
        self.names: dict[int, str] = dict(self.model.names)
        self.model_id = f"{Path(self.weights).stem}@{self.imgsz}"
        log.info(
            "detector ready: %s device=%s quantize=%s classes=%d tracker=%s",
            self.weights, self.device, self.quantize, len(self.names), self.tracker,
        )

    def warmup(self, height: int = 640, width: int = 640, runs: int = 2) -> None:
        """First inference includes CUDA/cuDNN init; do it before timing anything."""
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(runs):
            self._predict(blank)
        log.info("warmup complete")

    def _predict(self, image: np.ndarray):
        kwargs = dict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            max_det=self.max_det,
            device=self.device,
            quantize=self.quantize,
            classes=self.classes,
            verbose=False,
        )
        if self.tracker:
            # persist=True keeps track state across calls -> stable track_ids.
            return self.model.track(**kwargs, tracker=self.tracker, persist=True)
        return self.model.predict(**kwargs)

    def infer(self, frame: Frame) -> DetectionResult:
        t0 = time.perf_counter()
        results = self._predict(frame.image_bgr)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return DetectionResult(
            frame=frame,
            detections=self._to_detections(results[0]),
            infer_ms=infer_ms,
            model_id=self.model_id,
        )

    def _to_detections(self, result) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        out: list[Detection] = []
        for i in range(len(cls)):
            cid = int(cls[i])
            out.append(
                Detection(
                    cls_id=cid,
                    cls_name=self.names.get(cid, str(cid)),
                    conf=float(conf[i]),
                    xyxy=(float(xyxy[i][0]), float(xyxy[i][1]),
                          float(xyxy[i][2]), float(xyxy[i][3])),
                    track_id=int(ids[i]) if ids is not None else None,
                )
            )
        return out

    def reset_tracks(self) -> None:
        """Clear tracker state, e.g. between videos."""
        if hasattr(self.model, "predictor") and getattr(self.model.predictor, "trackers", None):
            for t in self.model.predictor.trackers:
                t.reset()
