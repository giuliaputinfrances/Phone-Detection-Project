from __future__ import annotations

import colorsys

import cv2
import numpy as np

from pdp.types import DetectionResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def class_color(cls_id: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR color per class id."""
    h = (cls_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def annotate(
    result: DetectionResult,
    *,
    labels: bool = True,
    zones: list[tuple[str, float, float]] | None = None,
    hud: str | None = None,
) -> np.ndarray:
    img = result.frame.image_bgr.copy()
    h, w = img.shape[:2]

    if zones:
        overlay = img.copy()
        for name, x0, x1 in zones:
            cv2.line(overlay, (int(x0 * w), 0), (int(x0 * w), h), (80, 80, 80), 1)
            cv2.putText(overlay, name, (int(x0 * w) + 4, h - 8), _FONT, 0.45,
                        (160, 160, 160), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    for det in result.detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        color = class_color(det.cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if not labels:
            continue
        tag = det.cls_name
        if det.track_id is not None:
            tag += f"#{det.track_id}"
        tag += f" {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, _FONT, 0.5, 1)
        ty = max(0, y1 - th - 6)
        cv2.rectangle(img, (x1, ty), (x1 + tw + 6, ty + th + 6), color, -1)
        cv2.putText(img, tag, (x1 + 3, ty + th + 1), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if hud:
        for i, line in enumerate(hud.split("\n")):
            y = 22 + i * 20
            cv2.putText(img, line, (10, y), _FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img
