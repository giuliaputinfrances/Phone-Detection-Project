# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pdp` — a YOLO26 obstacle-detection pipeline: iPhone camera (via Camo Studio virtual webcam) → detection → tracking → policy → servo commands over serial. Single installable package with one CLI entry point (`pdp`).

`docs/ARCHITECTURE.md` is the design document of record (dataflow, threading model, data contracts, phased roadmap, and the *reasoning* behind each choice). Read the relevant section before changing a subsystem — most non-obvious decisions are justified there, and `README.md` §State records what has actually been exercised on hardware versus what is only written.

## Environment

Work inside a Python 3.12 venv at `.venv/` (3.14 has patchy wheel coverage for `cv2`/`onnxruntime`/`tensorrt`/`pygrabber`; the global interpreter on this machine is 3.14 and cannot even import the test suite). **torch must be installed from the CUDA index before the project**, or pip resolves the CPU-only wheel and training silently runs ~50× slower.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -e ".[dev]"
pdp check-env          # must report cuda_available: True — hard-fails otherwise
```

## Commands

```powershell
pytest                              # full suite: needs the venv (cv2), but no GPU, camera, or hardware
pytest tests/test_policy.py         # one file
pytest tests/test_policy.py::test_x # one test
pytest -k "watchdog"                # by name
```

There is no linter or formatter configured; match surrounding style.

Pipeline commands (`pdp <command> --help` for the full argument list):

```powershell
pdp check-env                                       # first thing, always
pdp cameras                                         # DirectShow enumeration; find Camo
pdp extract-frames VIDEO -o DIR --fps 2             # video -> deduped candidate frames
pdp build-dataset -r datasets/raw -n obstacles_v1 --val-sessions <session>
pdp validate-dataset datasets/obstacles_v1
pdp train -c configs/train/obstacles_v1_n.yaml      # config-driven; CLI flags only override
pdp val -w runs/detect/<run>/weights/best.pt -d datasets/obstacles_v1/data.yaml
pdp predict -c configs/runtime/offline.yaml --video-out out.mp4
pdp live -c configs/runtime/live.yaml
pdp export -w best.pt -f engine
pdp bench -w best.engine
```

## Architecture

Linear dataflow, each stage talking to the next only through the frozen dataclasses in `pdp/types.py` (`Frame`, `Detection`, `DetectionResult`, `Command`). That contract is what makes stages swappable (file → Camo → RTSP; `.pt` → `.engine`; `NullBackend` → serial) without touching neighbours. Widening those types is a cross-cutting change — prefer not to.

```
Source → Detector → (tracker inside Detector) → Policy → ControlLoop → backend
   └──────────────────── Sinks: annotated video · JSONL · preview · metrics
```

Key invariants, each of which exists for a reason documented in ARCHITECTURE.md:

- **`pdp/pipeline.py` is the single run loop** shared by `predict` and `live`. Only the config differs. Do not fork it per mode — the point is that what you debug on a recorded video is literally what runs on the phone feed.
- **`ts_mono` is stamped once at capture** (`time.monotonic()`) and carried through untouched. It is the only way to measure true glass-to-servo latency; never re-stamp it downstream.
- **Threads:** capture (`pdp/sources/threaded.py`, depth-1 drop-oldest) → inference (main) → control (`pdp/control/loop.py`, fixed rate). Serial I/O must never run on the capture path.
- **`pdp/detect/detector.py` is the only model-aware code.** Ultralytics types stop there.
- **YOLO26 has an NMS-free one-to-one head**, so `iou`/`agnostic_nms` do nothing; `conf` is the sole precision/recall knob and `max_det` caps output directly. Ultralytics 8.4 also replaced `half`/`int8` with a single `quantize` argument (16/8/None).
- **Policy acts on *tracks*, not per-frame detections** (`min_hits` / `max_misses` / EMA). When the target is lost, `PanTiltPolicy` emits nothing and lets the control watchdog return to neutral — "what to do when we lose the target" lives in exactly one place.
- **`ControlLoop` enforces clamp, slew, deadband, watchdog** in Python; the real firmware duplicates all four, because the PC can crash or lose USB. The deadband applies to the *request*, not the output (see the comment in `loop.py`).
- **`NullBackend` (`control.backend: none`) is the default** and logs the commands it would send, so the whole pipeline runs end-to-end without hardware. `SerialServoBackend` is written against the line protocol in ARCHITECTURE.md §6 but has never talked to a board.

## Where things live

| Path | What it is |
|---|---|
| `docs/ARCHITECTURE.md` | Design doc of record: dataflow, threading, contracts, roadmap (§8), open questions (§10) |
| `pdp/types.py` | The four frozen dataclasses every stage passes around. Change with care — it is the cross-cutting contract |
| `pdp/pipeline.py` | The one run loop, shared by `predict` and `live` |
| `pdp/cli.py` | `pdp` entry point: argparse wiring only, with imports deferred per subcommand so startup stays fast |
| `pdp/config/schema.py` | Runtime config dataclasses + YAML loading and validation |
| `pdp/env.py` | `check-env`: fails loudly on a CPU-only torch build |
| `pdp/training.py` | `train` / `val` / `export` / `bench`, plus the git-SHA + manifest run snapshot |
| `pdp/sources/base.py` | `FrameSource` ABC — the seam for file / webcam / future RTSP |
| `pdp/sources/file.py` | Video-file source (stride, loop, max-frames) |
| `pdp/sources/webcam.py` | Camo/webcam via DirectShow; resolves devices **by name**, reads back the mode actually granted |
| `pdp/sources/threaded.py` | Depth-1 drop-oldest capture thread, so inference never consumes stale frames |
| `pdp/detect/detector.py` | The only model-aware file: ultralytics load, warmup, tracking, `Detection` mapping |
| `pdp/logic/policy.py` | Tracks → intent: priority, zones, persistence, EMA, distance proxy → `Command`s |
| `pdp/control/loop.py` | Rate-limited control thread enforcing clamp / slew / deadband / watchdog |
| `pdp/control/null.py` | Default backend: logs the commands it would have sent |
| `pdp/control/serial_servo.py` | Phase 7 serial backend — written, never run against hardware |
| `pdp/sinks/` | `draw.py` (annotation/HUD), `writers.py` (video writer, JSONL events, metrics) |
| `pdp/data/build.py` | Raw session exports → versioned dataset + `data.yaml` + `MANIFEST.json` |
| `pdp/data/validate.py` | Dataset checks: coord ranges, orphans, class ids, histograms |
| `pdp/data/frames.py` | Video → low-fps sampled frames, dropping near-duplicates so annotation hours aren't wasted |
| `pdp/data/hashing.py` | dHash (**not** aHash — see its docstring) perceptual hashing, used for both dedupe and cross-split leak detection |
| `configs/classes.yaml` | Class taxonomy, append-only. Currently still the placeholder list |
| `configs/train/*.yaml` | One file per training experiment |
| `configs/runtime/` | `live.yaml` (Camo) and `offline.yaml` (video file) |
| `scripts/` | Thin standalone shims over `pdp check-env` / `pdp cameras` |
| `datasets/`, `models/`, `runs/` | Git-ignored artifacts (only `MANIFEST.json` is tracked). All three are empty today |

## Config

Two independent config systems, both YAML, neither using a validation library:

- **Runtime** (`configs/runtime/{live,offline}.yaml`) → typed dataclasses in `pdp/config/schema.py`. Loading rejects unknown keys at every level and calls `validate()`; add a field by adding it to the dataclass. Note `control.backend: none` is the *string* "none" — bare `null` in YAML parses to `None` and is rejected with an explanatory error.
- **Training** (`configs/train/*.yaml`) → `pdp/training.py:load_train_config`, merged over `DEFAULT_TRAIN_ARGS` and passed to ultralytics. Hyperparameters belong in the config file, not in CLI flags; the CLI overrides exist for one-off experiments only.

`configs/classes.yaml` is the single source of truth for the class taxonomy — **append only, never reorder**, since reordering silently invalidates every existing label file. `data.yaml` is generated from it by `build-dataset` and must never be hand-edited.

## Datasets

`datasets/` and `runs/` are git-ignored except `MANIFEST.json` files. Datasets are versioned like code (`obstacles_v1` → `_v2`), never mutated — `build_dataset` refuses to overwrite without `--overwrite`.

**Splits are per capture session, never per frame.** Consecutive video frames are near-duplicates; a frame-level split leaks near-identical images into val and makes mAP fiction. `datasets/raw/<session>/` is the unit; whole sessions go to exactly one split, with a perceptual-hash duplicate check as a backstop.

Every training run snapshots the git SHA, resolved config, and the dataset `MANIFEST.json` into the run dir, so "which data produced this model?" is answerable from the run directory alone.

## Tests

`tests/` covers the config schema, policy state machine, control safety rules, dataset build/validation, and sources — all with fakes, no GPU or hardware. New logic in `logic/`, `control/`, `config/`, or `data/` should be testable the same way. Keep heavy imports lazy the way the existing modules do — `ultralytics` and `torch` are imported inside functions, never at module scope, so importing `pdp` stays cheap and the suite runs without a GPU stack.
