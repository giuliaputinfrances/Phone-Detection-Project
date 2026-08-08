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
        min_conf=0.4, min_hits=3, max_misses=2,
        ema_alpha=1.0,  # no error smoothing, so the arithmetic stays checkable
        fov_h_deg=60.0, fov_v_deg=45.0, gain=0.5, deadzone_deg=1.0,
        priority={"person": 30, "cone": 20},
        zones=[ZoneConfig("left", 0.0, 0.35), ZoneConfig("center", 0.35, 0.65),
               ZoneConfig("right", 0.65, 1.0)],
        pan_range_deg=(-45.0, 45.0), tilt_range_deg=(-30.0, 30.0),
    )
    base.update(kw)
    return PolicyConfig(**base)


def pan_of(cmds):
    return cmds[0].target_deg


def tilt_of(cmds):
    return cmds[1].target_deg


# -- confirmation ----------------------------------------------------------


def test_no_movement_until_min_hits():
    # Commands are always emitted (the rig holds its aim), but an unconfirmed
    # track must not move it.
    p = PanTiltPolicy(cfg())
    assert pan_of(p.update(result([det(cx=0.0)], 0))) == 0.0
    assert pan_of(p.update(result([det(cx=0.0)], 1))) == 0.0
    assert pan_of(p.update(result([det(cx=0.0)], 2))) != 0.0  # third hit confirms


def test_low_confidence_never_moves():
    p = PanTiltPolicy(cfg())
    for i in range(10):
        assert pan_of(p.update(result([det(conf=0.2, cx=0.0)], i))) == 0.0


def test_untracked_detections_are_ignored():
    p = PanTiltPolicy(cfg())
    for i in range(6):
        assert pan_of(p.update(result([det(track_id=None, cx=0.0)], i))) == 0.0


# -- closed-loop correction ------------------------------------------------


def test_centred_target_does_not_move_the_rig():
    p = PanTiltPolicy(cfg())
    for i in range(5):
        cmds = p.update(result([det(cx=W / 2, cy=H / 2)], i))
    assert pan_of(cmds) == 0.0
    assert tilt_of(cmds) == 0.0


def test_target_left_of_centre_drives_pan_negative():
    # cx=0 is half a frame off-centre: -0.5 * 60 deg fov = -30 deg of error,
    # of which gain=0.5 is applied -> -15 deg on the confirming frame.
    p = PanTiltPolicy(cfg())
    for i in range(3):
        cmds = p.update(result([det(cx=0.0)], i))
    assert pan_of(cmds) == -15.0
    assert "zone=left" in cmds[0].reason


def test_correction_converges_without_overshoot():
    # Each frame halves the remaining error and the rig approaches the target
    # monotonically: the property that a naive absolute mapping fails to have.
    p = PanTiltPolicy(cfg())
    angles = []
    for i in range(3):
        p.update(result([det(cx=0.0)], i))
    for i in range(3, 12):
        angles.append(pan_of(p.update(result([det(cx=0.0)], i))))

    assert all(b <= a for a, b in zip(angles, angles[1:]))  # monotonic
    assert min(angles) >= -45.0                             # never past the limit


def test_deadzone_suppresses_sub_threshold_error():
    # 0.5 deg of error, below the 1.0 deg deadzone: no movement at all.
    cx = W * (0.5 + 0.5 / 60.0)
    p = PanTiltPolicy(cfg())
    for i in range(5):
        cmds = p.update(result([det(cx=cx)], i))
    assert pan_of(cmds) == 0.0


def test_invert_pan_flips_the_correction():
    p = PanTiltPolicy(cfg(invert_pan=True))
    for i in range(3):
        cmds = p.update(result([det(cx=0.0)], i))
    assert pan_of(cmds) == 15.0


def test_accumulator_is_clamped_to_travel_limits():
    p = PanTiltPolicy(cfg())
    for i in range(40):
        cmds = p.update(result([det(cx=0.0, cy=0.0)], i))
    assert pan_of(cmds) == -45.0
    assert tilt_of(cmds) == -30.0


def test_target_below_centre_drives_tilt_positive():
    p = PanTiltPolicy(cfg())
    for i in range(3):
        cmds = p.update(result([det(cy=float(H))], i))
    assert tilt_of(cmds) > 0.0


# -- target selection ------------------------------------------------------


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


def test_switching_target_resets_the_error_smoother():
    # Otherwise the smoothed error carries across and the rig sweeps through
    # everything between the old target and the new one.
    p = PanTiltPolicy(cfg(ema_alpha=0.2))
    cone = det("cone", 0, track_id=1, cx=0.0)
    person = det("person", 2, track_id=2, cx=float(W))

    for i in range(3):                       # cone confirms and is selected
        p.update(result([cone], i))
    assert p._err_x == -30.0

    for i in range(3, 6):                    # person confirms and outranks it
        p.update(result([cone, person], i))

    # Averaged with the old target this would land near -18; snapped, at +30.
    assert p._err_x == 30.0


# -- holding ---------------------------------------------------------------


def test_holds_aim_when_target_is_lost():
    p = PanTiltPolicy(cfg())
    for i in range(3):
        cmds = p.update(result([det(cx=0.0)], i))
    aimed = pan_of(cmds)

    for i in range(3, 9):  # well past max_misses: the track is dropped
        cmds = p.update(result([], i))
    assert pan_of(cmds) == aimed
    assert "hold" in cmds[0].reason


def test_track_survives_brief_miss_then_expires():
    p = PanTiltPolicy(cfg())
    for i in range(3):
        p.update(result([det(cx=0.0)], i))
    moved = pan_of(p.update(result([], 3)))          # gap: holds, does not move
    assert pan_of(p.update(result([det(cx=0.0)], 4))) < moved  # still confirmed

    for i in range(5, 9):                            # exceed max_misses
        p.update(result([], i))
    held = pan_of(p.update(result([], 9)))
    assert pan_of(p.update(result([det(cx=0.0)], 10))) == held  # re-confirms first


# -- misc ------------------------------------------------------------------


def test_distance_proxy_scales_inversely_with_box_height():
    p = PanTiltPolicy(cfg(ref_box_height_px=200.0, ref_distance_m=1.0))
    assert p.distance_m(det(size=200.0)) == 1.0
    assert p.distance_m(det(size=100.0)) == 2.0
