from pdp.sources.base import FrameSource
from pdp.sources.file import FileSource
from pdp.sources.threaded import ThreadedSource
from pdp.sources.webcam import WebcamSource, list_devices, resolve_device

__all__ = [
    "FrameSource",
    "FileSource",
    "ThreadedSource",
    "WebcamSource",
    "list_devices",
    "resolve_device",
]


def build_source(cfg) -> FrameSource:
    """Construct a source from a SourceConfig."""
    kind = cfg.kind
    if kind == "file":
        src: FrameSource = FileSource(
            cfg.path, stride=cfg.stride, loop=cfg.loop, max_frames=cfg.max_frames
        )
    elif kind in ("webcam", "camo"):
        src = WebcamSource(
            cfg.device,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            backend=cfg.backend,
            fourcc=cfg.fourcc,
        )
    else:
        raise ValueError(f"unknown source kind: {kind!r}")

    if cfg.threaded:
        src = ThreadedSource(src)
    return src
