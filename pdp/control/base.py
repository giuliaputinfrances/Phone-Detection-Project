from __future__ import annotations

import abc

from pdp.types import Command


class ControlBackend(abc.ABC):
    """Where commands actually go. NullBackend logs; SerialServoBackend moves metal."""

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def apply(self, channel: int, target_deg: float, reason: str = "") -> None:
        """Drive one channel to an absolute angle."""

    @abc.abstractmethod
    def close(self) -> None: ...

    def ping(self) -> None:
        """Prove the PC is still alive without commanding a move.

        ControlLoop only writes when an angle actually changes, so a rig that is
        holding its aim sends nothing at all — and a device with its own
        watchdog would read that silence as "the PC died" and recentre itself.
        Backends with no watchdog can ignore this, hence the no-op default.
        """

    def submit(self, commands: list[Command]) -> None:
        for cmd in commands:
            self.apply(cmd.channel, cmd.target_deg, cmd.reason)

    def __enter__(self) -> ControlBackend:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
