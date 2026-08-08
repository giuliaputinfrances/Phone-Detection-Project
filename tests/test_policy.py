import numpy as np

from pdp.config import PolicyConfig, ZoneConfig
from pdp.logic import PanTiltPolicy
from pdp.types import Detection, DetectionResult, Frame

W, H = 640, 480


def frame(i=0):
    return Frame(i, float(i) / 30.0, np.zeros((H, W, 3), dtype=np.uint8), "test")


def det(cls_name="cone", cls_id=0, conf=0.9, track_id=1, cx=320.0, cy=240.0, size=100.0):
    return Detection(cls_id, cls_name, conf,
                     (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2), track_id)


def result(dets, i=0):
    return DetectionResult(frame(i), dets, 5.0, "test")


def cfg(**kw):
    base = dict(
        min_conf=0.4, min_hits=3, max_misses=2, ema_alpha=1.0,
        priority={"person": 30, "cone": 20},
        zones=[ZoneConfig("left", 0.0, 0.35), ZoneConfig("center", 0.35, 0.65),
               ZoneConfig("right", 0.65, 1.0)],
        pan_range_deg=(-45.0, 45.0), tilt_range_deg=(-30.0, 30.0),
    )
    base.update(kw)
    return PolicyConfig(**base)


def test_no_command_until_min_hits():
    p = PanTiltPolicy(cfg())
    assert p.update(result([det()], 0)) == []
    assert p.update(result([det()], 1)) == []
    assert p.update(result([det()], 2)) != []  # third hit confirms


def test_low_confidence_never_confirms():
    p = PanTiltPolicy(cfg())
    for i in range(10):
        assert p.update(result([det(conf=0.2)], i)) == []


def test_centered_target_maps_to_mid_range():
    p = PanTiltPolicy(cfg())
    for i in range(3):
        cmds = p.update(result([det(cx=W / 2, cy=H / 2)], i))
    pan, tilt = cmds[0], cmds[1]
    assert abs(pan.target_deg) < 0.5
    assert abs(tilt.target_deg) < 0.5


def test_left_target_maps_to_negative_pan():
    p = PanTiltPolicy(cfg())
    for i in range(3):
        cmds = p.update(result([det(cx=0.0)], i))
    assert cmds[0].target_deg == -45.0
    assert "zone=left" in cmds[0].reason


def test_priority_wins_over_size():
    p = PanTiltPolicy(cfg())
    small_person = det("person", 2, track_id=1, cx=100.0, size=40.0)
    big_cone = det("cone", 0, track_id=2, cx=500.0, size=300.0)
    for i in range(3):
        cmds = p.update(result([small_person, big_cone], i))
    assert "cls=person" in cmds[0].reason


def test_equal_priority_prefers_larger_box():
    p = PanTiltPolicy(cfg(priority={}))
    near = det("cone", 0, track_id=1, cx=100.0, size=250.0)
    far = det("cone", 0, track_id=2, cx=500.0, size=40.0)
    for i in range(3):
        cmds = p.update(result([near, far], i))
    assert "track=1" in cmds[0].reason


def test_track_survives_brief_miss_then_expires():
    p = PanTiltPolicy(cfg())
    for i in range(3):
        p.update(result([det()], i))
    assert p.update(result([], 3)) == []      # gap: no detection -> no command
    assert p.update(result([det()], 4)) != []  # still within max_misses

    for i in range(5, 9):                      # exceed max_misses -> track dropped
        p.update(result([], i))
    assert p.update(result([det()], 9)) == []  # must re-confirm from scratch


def test_switching_target_does_not_sweep_through_the_middle():
    # EMA is reset on target change, otherwise the rig pans across everything
    # between the old and new target.
    p = PanTiltPolicy(cfg(ema_alpha=0.2))
    for i in range(4):
        p.update(result([det(track_id=1, cx=0.0)], i))
    for i in range(4, 8):
        cmds = p.update(result([det(track_id=2, cx=float(W))], i))
    assert cmds[0].target_deg == 45.0


def test_distance_proxy_scales_inversely_with_box_height():
    p = PanTiltPolicy(cfg(ref_box_height_px=200.0, ref_distance_m=1.0))
    assert p.distance_m(det(size=200.0)) == 1.0
    assert p.distance_m(det(size=100.0)) == 2.0


def test_untracked_detections_are_ignored():
    p = PanTiltPolicy(cfg())
    for i in range(6):
        assert p.update(result([det(track_id=None)], i)) == []
