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

    The camera is mounted **on** the rig, so this is a closed loop: image
    position no longer tells you where the target is, it tells you how far off
    you are aimed. Mapping image position straight to an absolute angle — which
    is what this class used to do, and what works only for a camera watching
    from outside — chases its own tail. Centre the target, the computed angle
    falls back to zero, the rig swings away, and it oscillates forever.

    So we correct on error instead. The offset from frame centre is converted to
    real degrees through the camera's field of view, and a fraction of it is
    added to the current commanded angle each frame. Correcting the whole error
    would overshoot: the servo is still moving while the next frame is captured.

    Commands are emitted on every frame, including when nothing is confirmed, so
    the rig holds its aim rather than drifting back to neutral each time someone
    walks in front of the target.
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        self.cfg = cfg
        self._tracks: dict[int, TrackState] = {}
        # Dead reckoning: PWM servos report nothing, but they do obey absolute
        # positions, so the angle we last commanded is a good estimate of where
        # the rig actually is.
        self._pan: float = 0.0
        self._tilt: float = 0.0
        self._err_x: float | None = None
        self._err_y: float | None = None
        self._active_id: int | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._pan = self._tilt = 0.0
        self._err_x = self._err_y = None
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

    def _commands(self, ts: float, reason: str) -> list[Command]:
        return [
            Command("servo", self.cfg.pan_channel, float(self._pan), None, ts, reason),
            Command("servo", self.cfg.tilt_channel, float(self._tilt), None, ts, reason),
        ]

    def _step(self, current: float, err_deg: float, invert: bool,
              limits: tuple[float, float]) -> float:
        """Move a fraction of the way towards cancelling `err_deg`."""
        if abs(err_deg) < self.cfg.deadzone_deg:
            return current  # close enough; moving now would only buzz
        delta = self.cfg.gain * err_deg
        if invert:
            delta = -delta
        # Clamp the accumulator itself, not just the value sent downstream:
        # otherwise it winds past the limit while the target sits out of reach
        # and then owes that much travel before it responds again.
        return _clamp(current + delta, *limits)

    # -- main --------------------------------------------------------------

    def update(self, result: DetectionResult) -> list[Command]:
        self._age_tracks(result.detections)
        target = self._select()
        ts = result.frame.ts_mono

        if target is None or target.last is None:
            # Hold the current aim. A target that disappears behind a passer-by
            # almost always reappears where it was, so re-centring the rig would
            # throw away the framing we just worked to get.
            self._active_id = None
            self._err_x = self._err_y = None
            return self._commands(ts, "hold: no confirmed target")

        det = target.last
        w, h = result.frame.width, result.frame.height
        cx_norm = _clamp(det.cx / max(w, 1), 0.0, 1.0)
        cy_norm = _clamp(det.cy / max(h, 1), 0.0, 1.0)

        # How far off-aim we are, in real degrees. Routing through the field of
        # view is what makes `gain` a damping factor rather than a number tuned
        # by trial and error until the rig stops shaking.
        err_x = (cx_norm - 0.5) * self.cfg.fov_h_deg
        err_y = (cy_norm - 0.5) * self.cfg.fov_v_deg

        # Smooth the *error*, not the output: this filters detector jitter
        # before it becomes movement, without adding lag between the servo's
        # position and what we think it is. Reset on target change, or the rig
        # sweeps through everything between the old target and the new one.
        if self._active_id != target.track_id or self._err_x is None:
            self._active_id = target.track_id
            self._err_x, self._err_y = err_x, err_y
        else:
            a = self.cfg.ema_alpha
            self._err_x = a * err_x + (1 - a) * self._err_x
            self._err_y = a * err_y + (1 - a) * (self._err_y or 0.0)

        self._pan = self._step(self._pan, self._err_x, self.cfg.invert_pan,
                               self.cfg.pan_range_deg)
        self._tilt = self._step(self._tilt, self._err_y, self.cfg.invert_tilt,
                                self.cfg.tilt_range_deg)

        dist = self.distance_m(det)
        reason = (
            f"track={target.track_id} cls={det.cls_name} conf={det.conf:.2f} "
            f"zone={self.zone_of(cx_norm)} err={self._err_x:+.1f},{self._err_y:+.1f}deg"
            + (f" ~{dist:.2f}m" if dist is not None else "")
        )
        return self._commands(ts, reason)


def build_policy(cfg: PolicyConfig) -> Policy:
    if cfg.mode == "none":
        return NullPolicy()
    if cfg.mode == "pan_tilt":
        return PanTiltPolicy(cfg)
    raise ValueError(f"unknown policy mode: {cfg.mode!r}")
