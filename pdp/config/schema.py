"""Typed runtime configuration, loaded from YAML.

Plain dataclasses with explicit validation rather than a validation library:
the config surface is small, and this keeps the dependency list to things that
ultralytics already pulls in.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class SourceConfig:
    kind: str = "file"  # file | webcam | camo
    path: str = ""
    device: int | str = "Camo"
    width: int = 1280
    height: int = 720
    fps: float = 60.0
    backend: str = "dshow"  # dshow | msmf | any
    fourcc: str | None = "MJPG"
    stride: int = 1
    loop: bool = False
    max_frames: int | None = None
    threaded: bool = True

    def validate(self) -> None:
        if self.kind not in ("file", "webcam", "camo"):
            raise ConfigError(f"source.kind must be file|webcam|camo, got {self.kind!r}")
        if self.kind == "file" and not self.path:
            raise ConfigError("source.path is required when source.kind == 'file'")
        if self.backend not in ("dshow", "msmf", "any"):
            raise ConfigError(f"source.backend must be dshow|msmf|any, got {self.backend!r}")


@dataclass
class DetectorConfig:
    weights: str = "yolo26n.pt"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.25
    max_det: int = 100
    quantize: int | None = 16  # 16 = FP16 (CUDA only), 8 = INT8, null = FP32
    classes: list[int] | None = None
    tracker: str | None = "bytetrack.yaml"

    def validate(self) -> None:
        if not 0.0 < self.conf < 1.0:
            raise ConfigError(f"detector.conf must be in (0,1), got {self.conf}")
        if self.imgsz % 32 != 0:
            raise ConfigError(f"detector.imgsz must be a multiple of 32, got {self.imgsz}")
        if self.quantize not in (None, 8, 16, 32):
            raise ConfigError(
                f"detector.quantize must be null, 8, 16 or 32, got {self.quantize!r}"
            )


@dataclass
class ZoneConfig:
    name: str = "center"
    x0: float = 0.0  # normalized [0,1] horizontal bounds
    x1: float = 1.0

    def validate(self) -> None:
        if not 0.0 <= self.x0 < self.x1 <= 1.0:
            raise ConfigError(f"zone {self.name!r}: require 0 <= x0 < x1 <= 1")


@dataclass
class PolicyConfig:
    mode: str = "pan_tilt"  # pan_tilt | none
    priority: dict[str, int] = field(default_factory=dict)  # class name -> priority
    min_conf: float = 0.4
    min_hits: int = 3  # frames a track must persist before it can drive a servo
    max_misses: int = 5  # frames a track survives without a detection
    zones: list[ZoneConfig] = field(default_factory=list)
    ema_alpha: float = 0.35  # target smoothing; lower = smoother, more lag
    pan_channel: int = 0
    tilt_channel: int = 1
    pan_range_deg: tuple[float, float] = (-45.0, 45.0)
    tilt_range_deg: tuple[float, float] = (-30.0, 30.0)
    ref_box_height_px: float = 200.0  # bbox height at the reference distance
    ref_distance_m: float = 1.0

    def validate(self) -> None:
        if self.mode not in ("pan_tilt", "none"):
            raise ConfigError(f"policy.mode must be pan_tilt|none, got {self.mode!r}")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ConfigError("policy.ema_alpha must be in (0,1]")
        if self.min_hits < 1:
            raise ConfigError("policy.min_hits must be >= 1")
        for z in self.zones:
            z.validate()


@dataclass
class ControlConfig:
    # 'none' rather than 'null': bare `null` in YAML parses to None, not a string.
    backend: str = "none"  # none | serial
    port: str = "COM3"
    baud: int = 115200
    rate_hz: float = 50.0
    watchdog_ms: float = 500.0
    slew_dps: float = 180.0  # max servo speed; protects the gear train
    deadband_deg: float = 1.0  # ignore sub-threshold moves; stops buzzing
    neutral_deg: float = 0.0
    limits_deg: dict[int, tuple[float, float]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.backend is None:
            raise ConfigError(
                "control.backend is None — bare `null` in YAML parses as a null "
                "value. Write `backend: none` (quoted or unquoted) instead."
            )
        if self.backend not in ("none", "serial"):
            raise ConfigError(f"control.backend must be none|serial, got {self.backend!r}")
        if self.rate_hz <= 0:
            raise ConfigError("control.rate_hz must be > 0")


@dataclass
class SinksConfig:
    preview: bool = False
    video_out: str | None = None
    events_out: str | None = None
    draw_labels: bool = True
    draw_zones: bool = True
    metrics_every: int = 60  # log a latency/FPS summary every N frames


@dataclass
class RuntimeConfig:
    name: str = "default"
    source: SourceConfig = field(default_factory=SourceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    sinks: SinksConfig = field(default_factory=SinksConfig)

    def validate(self) -> RuntimeConfig:
        self.source.validate()
        self.detector.validate()
        self.policy.validate()
        self.control.validate()
        return self


def _build(cls, data: Any):
    """Recursively construct a dataclass from plain dicts, rejecting unknown keys."""
    if not is_dataclass(cls) or data is None:
        return data
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {cls.__name__}: {sorted(unknown)}; "
            f"valid keys are {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        # `zones` is the only nested-dataclass field; annotations are strings
        # here (PEP 563), so there is nothing to introspect generically.
        if key == "zones" and isinstance(value, list):
            kwargs[key] = [_build(ZoneConfig, v) for v in value]
        else:
            kwargs[key] = value
    return cls(**kwargs)


_SECTIONS = {
    "source": SourceConfig,
    "detector": DetectorConfig,
    "policy": PolicyConfig,
    "control": ControlConfig,
    "sinks": SinksConfig,
}


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    unknown = set(raw) - set(_SECTIONS) - {"name"}
    if unknown:
        raise ConfigError(f"unknown top-level key(s): {sorted(unknown)}")

    cfg = RuntimeConfig(name=raw.get("name", path.stem))
    for section, cls in _SECTIONS.items():
        if section in raw:
            setattr(cfg, section, _build(cls, raw[section]))
    # tuples survive YAML as lists; normalize the few tuple-typed fields
    cfg.policy.pan_range_deg = tuple(cfg.policy.pan_range_deg)  # type: ignore[assignment]
    cfg.policy.tilt_range_deg = tuple(cfg.policy.tilt_range_deg)  # type: ignore[assignment]
    cfg.control.limits_deg = {
        int(k): tuple(v) for k, v in (cfg.control.limits_deg or {}).items()
    }
    return cfg.validate()


def load_classes(path: str | Path = "configs/classes.yaml") -> dict[int, str]:
    """Load the append-only class taxonomy: {id: name}."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"classes file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    classes = raw.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ConfigError(f"{path}: expected a non-empty 'classes' mapping")

    out = {int(k): str(v) for k, v in classes.items()}
    expected = list(range(len(out)))
    if sorted(out) != expected:
        raise ConfigError(
            f"{path}: class ids must be contiguous from 0, got {sorted(out)}"
        )
    names = list(out.values())
    if len(set(names)) != len(names):
        raise ConfigError(f"{path}: duplicate class names: {names}")
    return out
