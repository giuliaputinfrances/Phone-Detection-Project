import time

from pdp.config import ControlConfig
from pdp.control.base import ControlBackend
from pdp.control.loop import ControlLoop
from pdp.types import Command


class RecordingBackend(ControlBackend):
    def __init__(self):
        self.applied: list[tuple[int, float]] = []
        self.connected = False
        self.pings = 0

    def connect(self):
        self.connected = True

    def apply(self, channel, target_deg, reason=""):
        self.applied.append((channel, round(target_deg, 3)))

    def ping(self):
        self.pings += 1

    def close(self):
        self.connected = False

    def angles(self, channel):
        return [deg for ch, deg in self.applied if ch == channel]


def cmd(ch=0, deg=30.0):
    return Command("servo", ch, deg, None, time.monotonic(), "test")


def cfg(**kw):
    base = dict(backend="none", rate_hz=200.0, watchdog_ms=100.0, slew_dps=180.0,
                deadband_deg=1.0, neutral_deg=0.0, limits_deg={0: (-45.0, 45.0)})
    base.update(kw)
    return ControlConfig(**base)


def test_clamps_to_channel_limits():
    be = RecordingBackend()
    loop = ControlLoop(be, cfg(slew_dps=100000.0))
    loop.start()
    try:
        loop.submit([cmd(0, 500.0)])
        time.sleep(0.1)
    finally:
        loop.stop()
    assert max(be.angles(0)) <= 45.0


def test_slew_limit_prevents_instant_jumps():
    be = RecordingBackend()
    # 20 deg/s at 200 Hz = 0.1 deg per tick; 45 deg cannot be reached quickly.
    loop = ControlLoop(be, cfg(slew_dps=20.0, deadband_deg=0.0))
    loop.start()
    try:
        loop.submit([cmd(0, 45.0)])
        time.sleep(0.15)
    finally:
        loop.stop()
    angles = be.angles(0)
    assert angles, "backend received nothing"
    assert max(angles) < 10.0, f"servo jumped straight to {max(angles)}"
    assert all(b - a <= 0.2 for a, b in zip(angles, angles[1:]) if b > a)


def test_watchdog_returns_to_neutral():
    be = RecordingBackend()
    loop = ControlLoop(be, cfg(slew_dps=100000.0, watchdog_ms=50.0))
    loop.start()
    try:
        loop.submit([cmd(0, 40.0)])
        time.sleep(0.03)
        assert be.angles(0)[-1] > 30.0
        time.sleep(0.25)  # stop feeding commands: watchdog must fire
        assert abs(be.angles(0)[-1]) < 1e-6
    finally:
        loop.stop()


def test_deadband_suppresses_micro_moves():
    be = RecordingBackend()
    loop = ControlLoop(be, cfg(slew_dps=100000.0, deadband_deg=5.0, watchdog_ms=10_000.0))
    loop.start()
    try:
        loop.submit([cmd(0, 10.0)])
        time.sleep(0.05)
        n = len(be.angles(0))
        for _ in range(10):  # jitter well under the deadband
            loop.submit([cmd(0, 10.4)])
            time.sleep(0.01)
        assert len(be.angles(0)) == n, "deadband did not suppress sub-threshold moves"
    finally:
        loop.stop()


def test_pings_while_holding_position():
    # A rig holding its aim sends no angles at all. Without a keep-alive the
    # firmware watchdog would read that silence as a dead PC and recentre.
    be = RecordingBackend()
    loop = ControlLoop(be, cfg(slew_dps=100000.0, watchdog_ms=60.0))
    loop.start()
    try:
        loop.submit([cmd(0, 20.0)])
        time.sleep(0.15)  # watchdog fires; the rig settles at neutral
        moves_before = len(be.applied)
        time.sleep(0.15)  # settled now: nothing to send but keep-alives
        assert len(be.applied) == moves_before, "backend moved while holding"
        assert be.pings > 0, "no keep-alive sent while holding position"
    finally:
        loop.stop()


def test_no_pings_while_actively_moving():
    # Every write already proves the PC is alive, so pinging on top of them
    # would be pure noise on the wire.
    be = RecordingBackend()
    loop = ControlLoop(be, cfg(slew_dps=20.0, deadband_deg=0.0, watchdog_ms=200.0))
    loop.start()
    try:
        for _ in range(10):  # keep feeding: it slews the whole time
            loop.submit([cmd(0, 45.0)])
            time.sleep(0.02)
        assert be.pings == 0, "pinged while the writes were already proving liveness"
    finally:
        loop.stop()


def test_stop_parks_at_neutral_and_closes():
    be = RecordingBackend()
    loop = ControlLoop(be, cfg(slew_dps=100000.0))
    loop.start()
    loop.submit([cmd(0, 40.0)])
    time.sleep(0.05)
    loop.stop()
    assert be.angles(0)[-1] == 0.0, "rig was left pointing at the last target"
    assert not be.connected
