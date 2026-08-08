"""Detections -> intent.

The important idea here is that we act on *tracks*, not on raw per-frame
detections. A detector flickers: a box appears for one frame, jumps a few
pixels, briefly vanishes behind an occluder. Driving a servo from that produces
visible jitter and wear. Requiring a track to persist for `min_hits` frames, and
tolerating `max_misses` frames of absence, turns noisy detections into a stable
target worth moving a motor for.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field

from pdp.config.schema import PolicyConfig
from pdp.types import Command, Detection, DetectionResult

log = logging.getLogger(__name__)


def _lerp(t: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * t


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class TrackState:
    track_id: int
    cls_name: str
    hits: int = 0
    misses: int = 0
    last: Detection | None = None

    @property
    def confirmed_by(self) -> int:
        return self.hits


class Policy(abc.ABC):
    @abc.abstractmethod
    def update(self, result: DetectionResult) -> list[Command]: ...

    def reset(self) -> None:  # pragma: no cover - trivial
        pass


class NullPolicy(Policy):
    """Detect and log only; emits no commands. Used until the rig exists."""

    def update(self, result: DetectionResult) -> list[Command]:
        return []


class PanTiltPolicy(Policy):
    """Aim a 2-axis rig at the highest-priority confirmed target.

    Assumes a camera fixed relative to the pan/tilt base, so a target's position
    in the image maps directly to an absolute pointing angle. If the camera ends
    up mounted *on* the moving rig, this becomes a closed loop and should be
    changed to incremental (error-driven) control instead.
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        self.cfg = cfg
        self._tracks: dict[int, TrackState] = {}
        self._pan: float | None = None
        self._tilt: float | None = None
        self._active_id: int | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._pan = self._tilt = None
        self._active_id = None

    # -- track bookkeeping -------------------------------------------------

    def _age_tracks(self, dets: list[Detection]) -> None:
        seen: set[int] = set()
        for det in dets:
            if det.track_id is None or det.conf < self.cfg.min_conf:
                continue
            seen.add(det.track_id)
            st = self._tracks.get(det.track_id)
            if st is None:
                st = TrackState(det.track_id, det.cls_name)
                self._tracks[det.track_id] = st
            st.hits += 1
            st.misses = 0
            st.last = det
            st.cls_name = det.cls_name

        for tid, st in list(self._tracks.items()):
            if tid in seen:
                continue
            st.misses += 1
            if st.misses > self.cfg.max_misses:
                del self._tracks[tid]

    def _priority(self, cls_name: str) -> int:
        return self.cfg.priority.get(cls_name, 0)

    def _select(self) -> TrackState | None:
        """Highest priority class; ties broken by largest box (nearest)."""
        candidates = [
            st for st in self._tracks.values()
            if st.hits >= self.cfg.min_hits and st.misses == 0 and st.last is not None
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda st: (self._priority(st.cls_name), st.last.area),  # type: ignore[union-attr]
        )

    # -- helpers -----------------------------------------------------------

    def zone_of(self, cx_norm: float) -> str:
        for z in self.cfg.zones:
            if z.x0 <= cx_norm < z.x1:
                return z.name
        return "none"

    def distance_m(self, det: Detection) -> float | None:
        """Crude monocular range from apparent box height. Calibrate per class."""
        if det.h <= 1 or self.cfg.ref_box_height_px <= 0:
            return None
        return self.cfg.ref_distance_m * self.cfg.ref_box_height_px / det.h

    # -- main --------------------------------------------------------------

    def update(self, result: DetectionResult) -> list[Command]:
        self._age_tracks(result.detections)
        target = self._select()

        if target is None or target.last is None:
            # Emit nothing. The control watchdog returns the rig to neutral,
            # which keeps "what to do when we lose the target" in exactly one
            # place instead of two.
            self._active_id = None
            return []

        det = target.last
        w, h = result.frame.width, result.frame.height
        cx_norm = _clamp(det.cx / max(w, 1), 0.0, 1.0)
        cy_norm = _clamp(det.cy / max(h, 1), 0.0, 1.0)

        pan = _lerp(cx_norm, *self.cfg.pan_range_deg)
        tilt = _lerp(cy_norm, *self.cfg.tilt_range_deg)

        # Reset the smoother when we switch targets, so the rig doesn't sweep
        # through everything in between.
        if self._active_id != target.track_id:
            self._pan, self._tilt = pan, tilt
            self._active_id = target.track_id
        else:
            a = self.cfg.ema_alpha
            self._pan = pan if self._pan is None else a * pan + (1 - a) * self._pan
            self._tilt = tilt if self._tilt is None else a * tilt + (1 - a) * self._tilt

        dist = self.distance_m(det)
        reason = (
            f"track={target.track_id} cls={det.cls_name} conf={det.conf:.2f} "
            f"zone={self.zone_of(cx_norm)}"
            + (f" ~{dist:.2f}m" if dist is not None else "")
        )
        ts = result.frame.ts_mono
        return [
            Command("servo", self.cfg.pan_channel, float(self._pan), None, ts, reason),
            Command("servo", self.cfg.tilt_channel, float(self._tilt), None, ts, reason),
        ]


def build_policy(cfg: PolicyConfig) -> Policy:
    if cfg.mode == "none":
        return NullPolicy()
    if cfg.mode == "pan_tilt":
        return PanTiltPolicy(cfg)
    raise ValueError(f"unknown policy mode: {cfg.mode!r}")
