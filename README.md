# Phone-Detection-Project

Multi-class obstacle detection with **YOLO26**, fed by an iPhone camera through
**Camo Studio**, driving **servos** over serial.

The full design — dataflow, threading model, data contracts, phased roadmap and
the reasoning behind each choice — is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -e ".[dev]"
pdp check-env          # must report cuda_available: True
```

Install torch **before** the project, or pip resolves the CPU-only wheel from
PyPI and training silently runs ~50x slower.

## Workflow

```powershell
# 1. Capture. Record video through Camo, one folder per session, then:
pdp extract-frames datasets/raw/2026-08-08_hallway.mp4 -o datasets/interim/hallway --fps 2

# 2. Annotate (Roboflow or Label Studio), export YOLO format into
#    datasets/raw/<session>/{images,labels}/

# 3. Build a versioned dataset. Splits happen per session, never per frame.
pdp build-dataset -r datasets/raw -n obstacles_v1 --val-sessions 2026-08-08_hallway

# 4. Train
pdp train -c configs/train/obstacles_v1_n.yaml

# 5. Evaluate — val during development, test only at the very end
pdp val -w runs/detect/obstacles_v1_n_e100/weights/best.pt -d datasets/obstacles_v1/data.yaml

# 6. Watch it run on a recorded video before touching hardware
pdp predict -s some_clip.mp4 -w runs/detect/.../best.pt --video-out runs/offline/annotated.mp4

# 7. Live off the iPhone
pdp cameras                                   # confirm Camo is visible
pdp live -c configs/runtime/live.yaml

# 8. Deployment artifact + latency
pdp export -w .../best.pt -f engine
pdp bench  -w .../best.engine
```

`pdp <command> --help` for the full argument list.

## Layout

| Path | What |
|---|---|
| `configs/classes.yaml` | Class taxonomy — **append only**, never reorder |
| `configs/train/` | One YAML per training experiment |
| `configs/runtime/` | `live.yaml` (Camo) and `offline.yaml` (video file) |
| `pdp/sources/` | Frame sources: file, webcam/Camo, drop-oldest capture thread |
| `pdp/detect/` | YOLO26 wrapper — the only model-aware code |
| `pdp/logic/` | Tracks → intent (priority, zones, debounce, smoothing) |
| `pdp/control/` | Servo backends + the safety loop (clamp, slew, deadband, watchdog) |
| `pdp/data/` | Frame extraction, dataset build, validation |
| `pdp/pipeline.py` | The single run loop shared by offline and live |

## State

Verified working on this machine (RTX 3000 Ada, 8 GB, torch 2.8.0+cu129,
ultralytics 8.4.116):

- `check-env`, `cameras`, `build-dataset`, `validate-dataset`, `train`, `val`,
  `bench`, `export -f onnx`, and `predict` end to end on a real clip
- YOLO26n at 640, FP16: **25.9 ms mean / 33.5 ms p95 → 38.6 FPS**
- 40 unit tests, no GPU/camera/hardware required

Not yet exercised:

- **Camo / iPhone** — `pdp cameras` enumerates correctly but only found the
  built-in webcam, since Camo Studio wasn't running. The Camo path itself is
  untested against a real phone.
- **TensorRT export** — needs the `tensorrt` package installed; only ONNX has
  been run.
- **Servos (Phase 7)** — `pdp/control/serial_servo.py` is written against the
  line protocol in `docs/ARCHITECTURE.md` §6 but has never talked to a board.
  Everything upstream of it runs today against `NullBackend`, which logs the
  commands it would have sent.

The class list in `configs/classes.yaml` is a placeholder — replace it before
annotating anything.

## Tests

```powershell
pytest            # no GPU, no camera, no hardware required
```
