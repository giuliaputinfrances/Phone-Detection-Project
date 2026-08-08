"""The single run loop, shared by offline (`predict`) and live (`live`) modes.

Only the config differs between them: a FileSource vs a threaded WebcamSource, a
video writer vs a preview window. Keeping one loop means the thing you debug on
a recorded video is literally the thing that runs on the iPhone feed.
"""

from __future__ import annotations

import json
import logging

import cv2

from pdp.config.schema import RuntimeConfig
from pdp.control import ControlLoop, build_backend
from pdp.detect import Detector
from pdp.logic import build_policy
from pdp.sinks import JsonlSink, Metrics, VideoWriterSink, annotate
from pdp.sources import build_source

log = logging.getLogger(__name__)


def run(cfg: RuntimeConfig, *, show_progress: bool = True) -> dict:
    source = build_source(cfg.source)
    detector = Detector(
        cfg.detector.weights,
        device=cfg.detector.device,
        imgsz=cfg.detector.imgsz,
        conf=cfg.detector.conf,
        max_det=cfg.detector.max_det,
        quantize=cfg.detector.quantize,
        classes=cfg.detector.classes,
        tracker=cfg.detector.tracker,
    )
    policy = build_policy(cfg.policy)
    control = ControlLoop(build_backend(cfg.control), cfg.control)
    metrics = Metrics()

    zones = [(z.name, z.x0, z.x1) for z in cfg.policy.zones] if cfg.sinks.draw_zones else None
    video: VideoWriterSink | None = None
    events: JsonlSink | None = None
    if cfg.sinks.events_out:
        events = JsonlSink(cfg.sinks.events_out)

    detector.warmup(cfg.detector.imgsz, cfg.detector.imgsz)

    interrupted = False
    try:
        source.open()
        control.start()
        if cfg.sinks.video_out:
            fps = source.fps if source.fps and source.fps > 1 else 30.0
            video = VideoWriterSink(cfg.sinks.video_out, fps)

        log.info(
            "pipeline start: source=%s %dx%d @ %.1f -> %s",
            source.source_id, source.width, source.height, source.fps, detector.model_id,
        )

        while True:
            frame = source.read()
            if frame is None:
                break

            result = detector.infer(frame)
            commands = policy.update(result)
            control.submit(commands)
            metrics.tick(result)

            if events is not None:
                events.write(result, commands)

            if video is not None or cfg.sinks.preview:
                hud = metrics.hud()
                if commands:
                    hud += f"\n{commands[0].reason}"
                img = annotate(result, labels=cfg.sinks.draw_labels, zones=zones, hud=hud)
                if video is not None:
                    video.write(img)
                if cfg.sinks.preview:
                    cv2.imshow(f"pdp: {cfg.name}", img)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        log.info("quit requested")
                        break

            if show_progress and metrics.frames % cfg.sinks.metrics_every == 0:
                log.info("%s | frames=%d dets=%d", metrics.hud(),
                         metrics.frames, metrics.detections)

    except KeyboardInterrupt:
        interrupted = True
        log.info("interrupted by user")
    finally:
        control.stop()
        source.close()
        if video is not None:
            video.close()
        if events is not None:
            events.close()
        if cfg.sinks.preview:
            cv2.destroyAllWindows()

    summary = metrics.summary()
    summary["interrupted"] = interrupted
    summary["commands_sent"] = control.sent_count
    if hasattr(source, "dropped"):
        summary["frames_dropped"] = source.dropped  # type: ignore[attr-defined]
    log.info("done: %s", json.dumps(summary))
    return summary
