"""Rate-limited control loop with the safety rules that keep servos alive.

Runs on its own thread so that serial I/O can never stall frame capture. Every
rule here (clamp, slew, deadband, watchdog) is duplicated in firmware in the
real rig — the PC can crash, hang, or lose USB, and the hardware has to be safe
on its own. This layer exists so the *software* never asks for something unsafe
in the first place.
"""

from __future__ import annotations

import logging
import threading
import time

from pdp.control.base import ControlBackend
from pdp.config.schema import ControlConfig
from pdp.types import Command

log = logging.getLogger(__name__)

DEFAULT_LIMITS = (-90.0, 90.0)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class ControlLoop:
    def __init__(self, backend: ControlBackend, cfg: ControlConfig) -> None:
        self.backend = backend
        self.cfg = cfg
        self._targets: dict[int, float] = {}
        self._current: dict[int, float] = {}
        self._sent: dict[int, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_cmd_ts = 0.0
        self._last_write_ts = 0.0
        self._watchdog_tripped = False
        self.sent_count = 0
        self.ping_count = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self.backend.connect()
        self._stop.clear()
        self._last_cmd_ts = self._last_write_ts = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.park()
        self.backend.close()

    def __enter__(self) -> ControlLoop:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- input -------------------------------------------------------------

    def submit(self, commands: list[Command]) -> None:
        """Thread-safe. Called from the inference thread once per frame."""
        if not commands:
            return
        with self._lock:
            for cmd in commands:
                lo, hi = self.cfg.limits_deg.get(cmd.channel, DEFAULT_LIMITS)
                self._targets[cmd.channel] = _clamp(cmd.target_deg, lo, hi)
                self._current.setdefault(cmd.channel, self.cfg.neutral_deg)
            self._last_cmd_ts = time.monotonic()
            if self._watchdog_tripped:
                log.info("control: watchdog cleared, target reacquired")
                self._watchdog_tripped = False

    def go_neutral(self) -> None:
        """Ask the loop to return to neutral (slew-limited, as normal)."""
        with self._lock:
            for ch in set(self._targets) | set(self._current):
                self._targets[ch] = self.cfg.neutral_deg

    def park(self) -> None:
        """Drive to neutral immediately, bypassing the loop.

        Used on shutdown, when the loop thread is already gone. This skips the
        slew limit on purpose — parking is the fail-safe, and the firmware
        rate-limits it anyway.
        """
        with self._lock:
            channels = sorted(set(self._targets) | set(self._current))
            self._current = {ch: self.cfg.neutral_deg for ch in channels}
            self._sent = {ch: self.cfg.neutral_deg for ch in channels}
        for ch in channels:
            try:
                self.backend.apply(ch, self.cfg.neutral_deg, "shutdown park")
            except Exception:
                log.exception("control: park failed on ch=%d", ch)

    # -- loop --------------------------------------------------------------

    def _run(self) -> None:
        period = 1.0 / self.cfg.rate_hz
        max_step = self.cfg.slew_dps * period
        next_tick = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic()

            with self._lock:
                stale_ms = (now - self._last_cmd_ts) * 1000.0
                if stale_ms > self.cfg.watchdog_ms and self._targets:
                    if not self._watchdog_tripped:
                        log.warning(
                            "control: watchdog tripped (%.0f ms without a command) "
                            "-> returning to neutral", stale_ms,
                        )
                        self._watchdog_tripped = True
                    for ch in self._targets:
                        self._targets[ch] = self.cfg.neutral_deg

                moves: list[tuple[int, float]] = []
                for ch, target in self._targets.items():
                    cur = self._current.get(ch, self.cfg.neutral_deg)
                    last = self._sent.get(ch)

                    # The deadband applies to the *request*, not to the output.
                    # A target less than deadband away from where the servo
                    # already sits is detector jitter, not a move: honoring it
                    # makes the servo buzz continuously. (Applying the deadband
                    # to the output instead would also block the final step of
                    # a legitimate slew, which is why it belongs here.)
                    if last is not None and abs(target - last) < self.cfg.deadband_deg:
                        target = last

                    delta = target - cur
                    if abs(delta) > max_step:  # slew limit
                        cur += max_step if delta > 0 else -max_step
                    else:
                        cur = target
                    self._current[ch] = cur

                    if last is None or abs(cur - last) > 1e-9:
                        self._sent[ch] = cur
                        moves.append((ch, cur))

            for ch, deg in moves:
                try:
                    self.backend.apply(ch, deg, "loop")
                    self.sent_count += 1
                except Exception:
                    log.exception("control: backend.apply failed on ch=%d", ch)

            # Keep-alive. A rig holding its aim produces no moves at all, and a
            # firmware watchdog can't tell that apart from a dead PC. Half the
            # watchdog period leaves room for one lost ping before it trips.
            if moves:
                self._last_write_ts = now
            elif (now - self._last_write_ts) * 1000.0 > self.cfg.watchdog_ms / 2.0:
                try:
                    self.backend.ping()
                    self.ping_count += 1
                except Exception:
                    log.exception("control: backend.ping failed")
                self._last_write_ts = now

            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()  # we fell behind; resync


def build_backend(cfg: ControlConfig) -> ControlBackend:
    if cfg.backend in ("none", "null"):
        from pdp.control.null import NullBackend

        return NullBackend()
    if cfg.backend == "serial":
        from pdp.control.serial_servo import SerialServoBackend

        return SerialServoBackend(cfg.port, cfg.baud)
    raise ValueError(f"unknown control backend: {cfg.backend!r}")
