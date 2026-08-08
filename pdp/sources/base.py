from __future__ import annotations

import abc

from pdp.types import Frame


class FrameSource(abc.ABC):
    """Anything that yields Frames: a video file, a webcam, an RTSP stream."""

    source_id: str = "source"

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or None when the source is exhausted/closed."""

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def width(self) -> int: ...

    @property
    @abc.abstractmethod
    def height(self) -> int: ...

    @property
    @abc.abstractmethod
    def fps(self) -> float: ...

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self):
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame
