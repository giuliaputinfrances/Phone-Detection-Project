import pytest

from pdp.config import ConfigError, load_classes, load_runtime_config


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_runtime_config(tmp_path):
    p = write(tmp_path, "r.yaml", """
name: t
source: {kind: file, path: a.mp4}
detector: {imgsz: 640, conf: 0.3}
control: {backend: none}
""")
    cfg = load_runtime_config(p)
    assert cfg.source.kind == "file"
    assert cfg.detector.conf == 0.3
    assert cfg.control.backend == "none"


def test_rejects_unknown_key(tmp_path):
    p = write(tmp_path, "r.yaml", "detector: {imgz: 640}\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_runtime_config(p)


def test_rejects_imgsz_not_multiple_of_32(tmp_path):
    p = write(tmp_path, "r.yaml", "source: {kind: file, path: a.mp4}\ndetector: {imgsz: 650}\n")
    with pytest.raises(ConfigError, match="multiple of 32"):
        load_runtime_config(p)


def test_file_source_requires_path(tmp_path):
    p = write(tmp_path, "r.yaml", "source: {kind: file}\n")
    with pytest.raises(ConfigError, match="source.path is required"):
        load_runtime_config(p)


def test_bare_yaml_null_backend_is_a_clear_error(tmp_path):
    # `backend: null` is a YAML null, not the string "null" - the message has to
    # say so, or this costs someone an hour.
    p = write(tmp_path, "r.yaml",
              "source: {kind: file, path: a.mp4}\ncontrol: {backend: null}\n")
    with pytest.raises(ConfigError, match="parses as a null value"):
        load_runtime_config(p)


def test_zones_become_dataclasses(tmp_path):
    p = write(tmp_path, "r.yaml", """
source: {kind: file, path: a.mp4}
policy:
  zones:
    - {name: left, x0: 0.0, x1: 0.5}
    - {name: right, x0: 0.5, x1: 1.0}
""")
    cfg = load_runtime_config(p)
    assert [z.name for z in cfg.policy.zones] == ["left", "right"]


def test_bad_zone_bounds_rejected(tmp_path):
    p = write(tmp_path, "r.yaml", """
source: {kind: file, path: a.mp4}
policy:
  zones: [{name: bad, x0: 0.8, x1: 0.2}]
""")
    with pytest.raises(ConfigError, match="x0 < x1"):
        load_runtime_config(p)


def test_classes_must_be_contiguous(tmp_path):
    p = write(tmp_path, "c.yaml", "classes:\n  0: a\n  2: b\n")
    with pytest.raises(ConfigError, match="contiguous"):
        load_classes(p)


def test_classes_reject_duplicate_names(tmp_path):
    p = write(tmp_path, "c.yaml", "classes:\n  0: a\n  1: a\n")
    with pytest.raises(ConfigError, match="duplicate"):
        load_classes(p)


def test_shipped_configs_are_valid():
    for path in ("configs/runtime/live.yaml", "configs/runtime/offline.yaml"):
        load_runtime_config(path)
    assert load_classes("configs/classes.yaml")
