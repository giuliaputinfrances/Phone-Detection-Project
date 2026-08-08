"""Webcam / virtual-camera source. This is how Camo Studio is consumed.

Camo installs a DirectShow virtual camera, so the iPhone shows up as an ordinary
capture device. The one non-obvious part is device *identification*: indices
shift whenever a camera is added, Camo restarts, or Windows wakes the IR camera.
This machine has an Intel UHD camera stack alongside the NVIDIA GPU, so it is a
real hazard. We therefore resolve by name and only fall back to a raw index.
"""

from __future__ import annotations

import logging
import time

import cv2

from pdp.sources.base import FrameSource
from pdp.types import Frame

log = logging.getLogger(__name__)

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


def list_devices() -> list[str]:
    """Enumerate DirectShow video devices by name. [] if unavailable."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        log.warning("pygrabber not installed; cannot resolve cameras by name")
        return []
    try:
        return list(FilterGraph().get_input_devices())
    except Exception as exc:  # pragma: no cover - driver dependent
        log.warning("device enumeration failed: %s", exc)
        return []


def resolve_device(device: int | str) -> int:
    """Turn a device name substring (e.g. 'Camo') into an index."""
    if isinstance(device, int) or str(device).isdigit():
        return int(device)

    names = list_devices()
    needle = str(device).lower()
    for idx, name in enumerate(names):
        if needle in name.lower():
            log.info("resolved camera %r -> index %d (%s)", device, idx, name)
            return idx

    listing = "\n".join(f"  [{i}] {n}" for i, n in enumerate(names)) or "  (none found)"
    raise RuntimeError(
        f"no video device matching {device!r}. Available devices:\n{listing}\n"
        "Is Camo Studio running and the iPhone connected?"
    )


class WebcamSource(FrameSource):
    def __init__(
        self,
        device: int | str = 0,
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 60.0,
        backend: str = "dshow",
        fourcc: str | None = "MJPG",
        warmup_frames: int = 5,
    ) -> None:
        self.device = device
        self.req_width = width
        self.req_height = height
        self.req_fps = fps
        self.backend = backend
        self.fourcc = fourcc
        self.warmup_frames = warmup_frames
        self.source_id = f"cam:{device}"
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0

    def open(self) -> None:
        index = resolve_device(self.device)
        api = _BACKENDS.get(self.backend, cv2.CAP_ANY)
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open camera index {index} with backend {self.backend!r}. "
                "Try backend: msmf."
            )

        # MJPG is what lets most virtual/UVC cameras actually deliver high fps at
        # 720p+; the default YUY2 mode silently caps out much lower.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # honored by DSHOW, ignored by MSMF

        self._cap = cap
        for _ in range(self.warmup_frames):
            cap.read()

        # Always report what the driver actually gave us. Camo substitutes the
        # nearest supported mode without complaining, and a config that claims
        # 60fps while delivering 30 sends you debugging the wrong layer.
        log.info(
            "camera open: index=%d requested=%dx%d@%.0f actual=%dx%d@%.0f",
            index, self.req_width, self.req_height, self.req_fps,
            self.width, self.height, self.fps,
        )

    def read(self) -> Frame | None:
        if self._cap is None:
            raise RuntimeError("read() before open()")
        ok, img = self._cap.read()
        ts = time.monotonic()
        if not ok:
            return None
        frame = Frame(self._frame_id, ts, img, self.source_id)
        self._frame_id += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0
