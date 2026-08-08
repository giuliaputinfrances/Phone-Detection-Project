from __future__ import annotations

import logging

from pdp.control.base import ControlBackend

log = logging.getLogger(__name__)


class NullBackend(ControlBackend):
    """Logs what it would have sent.

    This is what makes the servo stage 'future' without being an afterthought:
    the entire pipeline runs end to end from day one, and Phase 7 swaps this for
    the serial backend with no change upstream.
    """

    def __init__(self, log_every: int = 1) -> None:
        self.log_every = max(1, log_every)
        self.count = 0
        self.last: dict[int, float] = {}

    def connect(self) -> None:
        log.info("control: NullBackend (no hardware, commands are logged only)")

    def apply(self, channel: int, target_deg: float, reason: str = "") -> None:
        self.count += 1
        self.last[channel] = target_deg
        if self.count % self.log_every == 0:
            log.debug("SERVO ch=%d -> %7.2f deg  (%s)", channel, target_deg, reason)

    def close(self) -> None:
        log.info("control: NullBackend closed after %d commands", self.count)
