"""Drop-oldest capture thread.

The whole point: when inference is slower than the camera, we want *fewer,
current* frames rather than *all, stale* frames. A depth-1 slot that overwrites
gives exactly that, and keeps the OS capture buffer from accumulating seconds of
latency behind a real-time control loop.
"""

from __future__ import annotations

import threading

from pdp.sources.base import FrameSource
from pdp.types import Frame


class ThreadedSource(FrameSource):
    def __init__(self, inner: FrameSource, *, poll_timeout: float = 2.0) -> None:
        self.inner = inner
        self.poll_timeout = poll_timeout
        self.source_id = inner.source_id
        self._slot: Frame | None = None
        self._lock = threading.Lock()
        self._new = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._exhausted = False
        self.dropped = 0

    def open(self) -> None:
        self.inner.open()
        self.source_id = self.inner.source_id
        self._stop.clear()
        self._exhausted = False
        self._thread = threading.Thread(target=self._pump, name="capture", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while not self._stop.is_set():
            frame = self.inner.read()
            with self._new:
                if frame is None:
                    self._exhausted = True
                    self._new.notify_all()
                    return
                if self._slot is not None:
                    self.dropped += 1  # overwrite: the pending frame was stale
                self._slot = frame
                self._new.notify_all()

    def read(self) -> Frame | None:
        with self._new:
            while self._slot is None and not self._exhausted and not self._stop.is_set():
                if not self._new.wait(timeout=self.poll_timeout):
                    return None  # source stalled
            frame, self._slot = self._slot, None
            return frame

    def close(self) -> None:
        self._stop.set()
        with self._new:
            self._new.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.inner.close()

    @property
    def width(self) -> int:
        return self.inner.width

    @property
    def height(self) -> int:
        return self.inner.height

    @property
    def fps(self) -> float:
        return self.inner.fps
