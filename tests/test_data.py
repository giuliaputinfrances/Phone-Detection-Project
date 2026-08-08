import json

import cv2
import numpy as np
import pytest

from pdp.data import build_dataset, validate_dataset
from pdp.data.validate import validate_label_file

CLASSES = {0: "cone", 1: "box"}


def make_session(root, name, n=4, *, seed=0, label="0 0.5 0.5 0.2 0.3"):
    rng = np.random.default_rng(seed)
    (root / name / "images").mkdir(parents=True)
    (root / name / "labels").mkdir(parents=True)
    for i in range(n):
        img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(root / name / "images" / f"f{i}.jpg"), img)
        if label is not None:
            (root / name / "labels" / f"f{i}.txt").write_text(label + "\n")
    return root / name


# --- label validation -----------------------------------------------------


def test_rejects_unnormalized_coordinates(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 320 240 100 80\n")  # pixels, not normalized
    errors, _, _, _ = validate_label_file(p, 2)
    assert any("outside [0,1]" in e for e in errors)


def test_rejects_class_id_beyond_taxonomy(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("7 0.5 0.5 0.1 0.1\n")
    errors, _, _, _ = validate_label_file(p, 2)
    assert any("outside 0..1" in e for e in errors)


def test_rejects_wrong_field_count(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.1\n")
    errors, _, _, _ = validate_label_file(p, 2)
    assert any("expected 5 fields" in e for e in errors)


def test_counts_boxes_per_class(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n0 0.8 0.8 0.1 0.1\n")
    errors, _, counts, boxes = validate_label_file(p, 2)
    assert not errors
    assert counts == {0: 2, 1: 1}
    assert boxes == 3


# --- dataset build --------------------------------------------------------


def test_build_splits_by_session_not_by_frame(tmp_path):
    raw = tmp_path / "raw"
    for i, name in enumerate(["s_a", "s_b", "s_c", "s_d"]):
        make_session(raw, name, n=3, seed=i)

    manifest = build_dataset(raw, tmp_path / "out", CLASSES,
                             val_sessions=["s_b"], seed=0, check_duplicates=False)

    assert manifest["sessions"]["s_b"]["split"] == "val"
    # every frame of a session lands in exactly one split
    val_images = list((tmp_path / "out" / "images" / "val").glob("*.jpg"))
    assert {p.name.split("__")[0] for p in val_images} == {"s_b"}


def test_build_prefixes_names_so_sessions_cannot_collide(tmp_path):
    raw = tmp_path / "raw"
    make_session(raw, "s_a", n=2, seed=1)
    make_session(raw, "s_b", n=2, seed=2)
    build_dataset(raw, tmp_path / "out", CLASSES,
                  val_sessions=["s_b"], seed=0, check_duplicates=False)
    train = sorted(p.name for p in (tmp_path / "out" / "images" / "train").glob("*.jpg"))
    assert train == ["s_a__f0.jpg", "s_a__f1.jpg"]


def test_build_writes_generated_data_yaml_and_manifest(tmp_path):
    raw = tmp_path / "raw"
    make_session(raw, "s_a", n=2, seed=1)
    make_session(raw, "s_b", n=2, seed=2)
    build_dataset(raw, tmp_path / "out", CLASSES,
                  val_sessions=["s_b"], seed=0, check_duplicates=False)

    yaml_text = (tmp_path / "out" / "data.yaml").read_text()
    assert "GENERATED" in yaml_text
    assert "cone" in yaml_text
    manifest = json.loads((tmp_path / "out" / "MANIFEST.json").read_text())
    assert manifest["valid"]
    assert manifest["class_counts"]["cone"] == 4


def test_build_refuses_to_silently_overwrite(tmp_path):
    raw = tmp_path / "raw"
    make_session(raw, "s_a", n=2, seed=1)
    make_session(raw, "s_b", n=2, seed=2)
    build_dataset(raw, tmp_path / "out", CLASSES, val_sessions=["s_b"],
                  check_duplicates=False)
    with pytest.raises(FileExistsError, match="versioned"):
        build_dataset(raw, tmp_path / "out", CLASSES, val_sessions=["s_b"],
                      check_duplicates=False)


def test_duplicate_images_across_splits_are_flagged_as_leakage(tmp_path):
    raw = tmp_path / "raw"
    # identical content in two sessions -> the same moment in train and val
    make_session(raw, "s_a", n=2, seed=7)
    make_session(raw, "s_b", n=2, seed=7)
    manifest = build_dataset(raw, tmp_path / "out", CLASSES,
                             val_sessions=["s_b"], check_duplicates=True)
    assert not manifest["valid"]
    assert any("leakage" in e for e in manifest["errors"])


def test_images_without_labels_count_as_backgrounds(tmp_path):
    raw = tmp_path / "raw"
    make_session(raw, "s_a", n=3, seed=1)
    make_session(raw, "s_b", n=2, seed=2, label=None)
    manifest = build_dataset(raw, tmp_path / "out", CLASSES,
                             val_sessions=["s_b"], check_duplicates=False)
    assert manifest["totals"]["backgrounds"] == 2


def test_validate_catches_orphan_labels(tmp_path):
    raw = tmp_path / "raw"
    make_session(raw, "s_a", n=2, seed=1)
    make_session(raw, "s_b", n=2, seed=2)
    out = tmp_path / "out"
    build_dataset(raw, out, CLASSES, val_sessions=["s_b"], check_duplicates=False)

    (out / "labels" / "train" / "ghost.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    report = validate_dataset(out, len(CLASSES), check_duplicates=False)
    assert not report.ok
    assert any("orphan" in e for e in report.errors)
