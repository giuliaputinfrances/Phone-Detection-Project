import threading
import time

import numpy as np

from pdp.sources.base import FrameSource
from pdp.sources.threaded import ThreadedSource
from pdp.types import Frame


class FakeSource(FrameSource):
    """Emits frames on demand, optionally throttled, then ends."""

    def __init__(self, count=10, delay=0.0):
        self.count = count
        self.delay = delay
        self.source_id = "fake"
        self.i = 0
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def read(self):
        if self.i >= self.count:
            return None
        if self.delay:
            time.sleep(self.delay)
        f = Frame(self.i, time.monotonic(), np.zeros((4, 4, 3), np.uint8), self.source_id)
        self.i += 1
        return f

    def close(self):
        self.closed = True

    @property
    def width(self):
        return 4

    @property
    def height(self):
        return 4

    @property
    def fps(self):
        return 30.0


def test_threaded_source_yields_frames_and_ends():
    src = ThreadedSource(FakeSource(count=5, delay=0.005))
    src.open()
    try:
        got = []
        while (f := src.read()) is not None:
            got.append(f.frame_id)
    finally:
        src.close()
    assert got, "no frames delivered"
    assert got == sorted(got), "frames arrived out of order"


def test_threaded_source_drops_stale_frames_instead_of_queueing():
    # Producer is much faster than the consumer. The consumer must see recent
    # frames, not work through a backlog: that is the whole point of the
    # depth-1 drop-oldest slot.
    src = ThreadedSource(FakeSource(count=400, delay=0.001))
    src.open()
    try:
        seen = []
        for _ in range(5):
            f = src.read()
            if f is None:
                break
            seen.append(f.frame_id)
            time.sleep(0.03)  # slow consumer
    finally:
        src.close()

    assert src.dropped > 0, "nothing was dropped; the slot is queueing"
    gaps = [b - a for a, b in zip(seen, seen[1:])]
    assert any(g > 1 for g in gaps), f"consumer saw a contiguous backlog: {seen}"


def test_threaded_source_closes_inner_and_stops_thread():
    inner = FakeSource(count=10_000, delay=0.001)
    src = ThreadedSource(inner)
    src.open()
    src.read()
    src.close()
    assert inner.closed
    time.sleep(0.05)
    names = [t.name for t in threading.enumerate()]
    assert "capture" not in names


def test_read_after_exhaustion_returns_none():
    src = ThreadedSource(FakeSource(count=2))
    src.open()
    try:
        while src.read() is not None:
            pass
        assert src.read() is None
    finally:
        src.close()
