"""Serial servo backend (Phase 7).

Line protocol, ASCII, ack'd — deliberately debuggable from a plain serial
monitor:

    ->  S <ch> <deg>\\n     set channel to an absolute angle
    ->  P\\n                ping
    <-  OK <ch> <deg>\\n  |  ERR <code>\\n

The firmware is the authority on limits: it must clamp, slew-limit and run its
own watchdog, because this process can die at any moment.

Untested against hardware — no rig exists yet. It is wired up so that Phase 7 is
a config change (`control.backend: serial`) plus firmware, not a rewrite.
"""

from __future__ import annotations

import logging

from pdp.control.base import ControlBackend

log = logging.getLogger(__name__)


class SerialServoBackend(ControlBackend):
    def __init__(self, port: str = "COM3", baud: int = 115200, *,
                 timeout: float = 0.05, require_ack: bool = True) -> None:
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.require_ack = require_ack
        self._ser = None

    def connect(self) -> None:
        try:
            import serial  # pyserial
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pyserial is required for the serial backend: pip install -e .[serial]"
            ) from exc

        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        # Most Arduino boards reset when the port opens; give the bootloader time.
        self._ser.reset_input_buffer()
        log.info("control: serial open on %s @ %d", self.port, self.baud)

    def apply(self, channel: int, target_deg: float, reason: str = "") -> None:
        if self._ser is None:
            raise RuntimeError("apply() before connect()")
        self._ser.write(f"S {channel} {target_deg:.2f}\n".encode("ascii"))
        if self.require_ack:
            resp = self._ser.readline().decode("ascii", "replace").strip()
            if not resp.startswith("OK"):
                log.warning("control: unexpected reply %r for ch=%d", resp, channel)

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.write(b"P\n")
            finally:
                self._ser.close()
                self._ser = None
                log.info("control: serial closed")
