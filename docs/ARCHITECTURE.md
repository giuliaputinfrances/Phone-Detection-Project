# Phone-Detection-Project — Architecture Plan

Status: **implemented through Phase 6**; Phase 7 (servos) awaits hardware.
See the roadmap in §8 and `README.md` for usage.
Target: multi-class object/obstacle detection with YOLO26, fed by an iPhone camera via Camo Studio,
eventually driving servos over a serial link.

---

## 0. Environment baseline (measured 2026-08-08)

| Item | Value | Action |
|---|---|---|
| GPU | NVIDIA RTX 3000 Ada Laptop, 8 GB VRAM | Usable for training `yolo26n/s`, tight for `m`, no `l/x` at 640 |
| Driver / CUDA | 595.95 / CUDA 13.2 | Supports cu12x and cu13x torch wheels |
| Global Python | 3.14.4 | Do **not** use directly |
| Installed torch | 2.12.0+**cpu** (`torch.cuda.is_available() == False`) | Must be replaced with a CUDA build |
| ultralytics | not installed | Install 8.4.x (YOLO26 line) |

**Decision: create an isolated venv on Python 3.12.**
Python 3.14 is new enough that the long tail of CV wheels (`opencv-python`, `onnxruntime-gpu`,
`tensorrt`, `pygrabber`) has patchy `cp314` coverage. 3.12 avoids source builds. This is a
recommendation, not a measured blocker — if you prefer 3.14, the fallback is verifying each wheel
individually.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"  # must print True
pip install ultralytics
```
Install torch **before** ultralytics, otherwise pip resolves the CPU wheel from PyPI first.

---

## 1. System architecture

A single linear dataflow. Every stage talks to the next only through the data contracts in §2, so any
stage can be swapped (file → Camo → RTSP; PyTorch → TensorRT; log → servos) without touching the others.

```
┌──────────┐   Frame    ┌───────────┐  Detection[]  ┌─────────┐  Track[]  ┌────────┐  Command[]  ┌──────────┐
│  Source  │ ─────────► │ Detector  │ ────────────► │ Tracker │ ────────► │ Policy │ ──────────► │ Control  │
└──────────┘            └───────────┘               └─────────┘           └────────┘             └──────────┘
  Camo cam                YOLO26.pt                  ByteTrack             zones,                 serial →
  video file              /.engine                   stable IDs            hysteresis             servos
  RTSP                                                                     priority
      │                        │                          │                    │                      │
      └────────────────────────┴──────────► Sinks ◄───────┴────────────────────┴──────────────────────┘
                                    annotated video · JSONL events · live preview · metrics
```

### Threading model (decided now, because retrofitting it is painful)

Three threads, connected by **depth-1 drop-oldest queues**:

1. **Capture thread** — `cap.read()` in a tight loop, pushes the newest frame, discards the previous
   one if unconsumed. Prevents the OS camera buffer from accumulating seconds of stale video.
2. **Inference thread** (main) — pulls newest frame, runs detect → track → policy.
3. **Control thread** — consumes commands at its own fixed rate (e.g. 50 Hz), independent of FPS.

The hard rule: **serial I/O must never run on the capture path.** A blocked servo write must not stall
frame acquisition. Drop-oldest also means that when inference falls behind, you get *fewer, current*
frames rather than *all, stale* frames — the right tradeoff for real-time control.

### Module map

| Package | Responsibility | Swappable behind |
|---|---|---|
| `pdp/config/` | Load + validate YAML into typed objects (dataclasses) | — |
| `pdp/sources/` | `FrameSource` ABC → `FileSource`, `WebcamSource` (Camo), `RTSPSource` | `FrameSource` |
| `pdp/detect/` | Ultralytics wrapper: load, warmup, half/AMP, thresholds, `Detection` mapping | `Detector` |
| `pdp/track/` | ByteTrack / BoT-SORT, assigns persistent `track_id` | `Tracker` |
| `pdp/logic/` | ROI zones, per-class priority, distance proxy, debounce → `Command`s | `Policy` |
| `pdp/control/` | `ControlBackend` ABC → `NullBackend` (log-only), `SerialServoBackend`; `ControlLoop` applies the safety rules | `ControlBackend` |
| `pdp/sinks/` | Annotated writer, JSONL event log, FPS/latency metrics, preview window | `Sink` |
| `pdp/cli.py` | `train · val · predict · live · export · bench · ingest` | — |

`NullBackend` is what makes the servo work "future" without being an afterthought: the whole pipeline
runs end-to-end from day one, printing the commands it *would* send. Phase 7 swaps in the real backend
and nothing upstream changes.

---

## 2. Data contracts

Frozen dataclasses, defined once, used everywhere:

```python
Frame(frame_id: int, ts_mono: float, image_bgr: np.ndarray, source_id: str)

Detection(cls_id: int, cls_name: str, conf: float,
          xyxy: tuple[float, float, float, float],   # pixels, source resolution
          track_id: int | None = None)

DetectionResult(frame: Frame, detections: list[Detection],
                infer_ms: float, model_id: str)

Command(kind: Literal["servo"], channel: int, target_deg: float,
        speed_dps: float | None, ts_mono: float, reason: str)
```

`ts_mono` is `time.monotonic()`, stamped **at capture**, and carried through untouched. It is the only
way to measure true glass-to-servo latency later, and you cannot reconstruct it after the fact.
`reason` on `Command` is a human-readable trace string ("obstacle:cone track=14 zone=left") — invaluable
when a servo twitches and you need to know why.

---

## 3. Dataset architecture

### Directory layout

```
datasets/
├── raw/                     # untouched captures, never edited (git-ignored)
│   └── 2026-08-08_hallway/  # one dir per capture session
├── interim/                 # extracted frames awaiting annotation
└── obstacles_v1/            # a built, versioned, trainable dataset
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    ├── data.yaml            # generated, never hand-edited
    └── MANIFEST.json        # source sessions, counts, split seed, build date
```

`obstacles_v1` → `_v2` → … Datasets are versioned like code. A run's metrics are meaningless without
knowing exactly which dataset produced them.

### Class taxonomy — single source of truth

`configs/classes.yaml` defines the class list; `data.yaml` is **generated** from it by
`scripts/build_dataset.py`. Reordering classes silently invalidates every existing label file, so it
must live in exactly one place with a comment saying "append only, never reorder."

```yaml
# configs/classes.yaml — APPEND ONLY. Reordering invalidates all existing labels.
classes:
  0: cone
  1: box
  2: person
  3: chair
```

### Label format (Ultralytics YOLO)

One `.txt` per image, same basename, mirrored path (`images/train/a.jpg` ↔ `labels/train/a.txt`):

```
<cls_id> <x_center> <y_center> <width> <height>     # all normalized 0–1
```

Empty/absent `.txt` = a valid negative (background) image. Deliberately include ~10–15% negatives —
frames with no target objects — or the model learns "there is always a cone somewhere."

### Split strategy — the one thing most people get wrong

**Split by capture session, never by frame.** Consecutive video frames are near-duplicates; a random
frame-level split puts near-identical images in both train and val, and your val mAP becomes fiction —
90% on paper, useless in the hallway. Whole sessions go to exactly one split. `test/` should be a
session recorded on a *different day, different lighting*, and touched only at the very end.

### Annotation workflow

Roboflow (fastest, has YOLO26 export) or Label Studio (local, free, no upload). Either way, the export
lands in `datasets/raw/` and `scripts/build_dataset.py` normalizes it — nothing hand-copied into the
trainable tree. That script also **validates**, failing loudly on:

- coordinates outside `[0,1]`, zero-area or sub-4px boxes
- `cls_id >= len(classes)`
- label files with no matching image (orphans) and vice versa
- duplicate images across splits (perceptual hash — catches leakage the session rule missed)
- class-count histogram printed per split, so imbalance is visible before you train

### The single biggest accuracy lever

**Capture training data through Camo, from the iPhone, in the deployment environment.** A model trained
on internet images of cones and deployed on iPhone-through-Camo video suffers a domain gap — different
sensor, color pipeline, FOV, compression, mounting height. Same-camera data beats more data. Plan for
~150–300 images per class from the real rig, across varied lighting/angles/distances/occlusion, rather
than thousands of scraped ones.

---

## 4. Training pipeline

### Config, not arguments

`configs/train/obstacles_v1_n.yaml` holds every hyperparameter; the CLI takes only the config path.
Runs are then reproducible from one file, and diffable against each other.

### Baseline recipe (fine-tune from COCO weights)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")          # COCO-pretrained
model.train(
    data="datasets/obstacles_v1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,                       # 8 GB @ 640 with AMP; use batch=0.80 for autobatch
    freeze=10,                      # keep backbone; your classes are COCO-like
    optimizer="AdamW", lr0=0.001,   # stable for small datasets
    patience=20,
    mosaic=0.5,                     # heavy aug hurts small datasets
    amp=True, cache="ram",
    seed=0, deterministic=True,
    project="runs/detect", name="obstacles_v1_n_e100",
)
```

Rationale, per Ultralytics' fine-tuning guide: `freeze=10` preserves the backbone when your domain is
COCO-adjacent (people, chairs, boxes all exist in COCO); `freeze=23` (head only) if the dataset turns
out very small. `optimizer=auto` is the default and is fine, but explicit AdamW + `lr0=0.001` is more
predictable on a few-hundred-image set. Drop `mosaic` to `0.0` if val loss diverges early.

If small/distant obstacles are the priority, `imgsz=1280` is the highest-value change available —
at roughly 4× the memory, so halve the batch.

### Model size ladder

Start `yolo26n`. Only move up if val mAP is the bottleneck, and re-measure latency at each step —
`n` and `s` both clear real-time on this GPU; `m` at 640 will fit in 8 GB but eats headroom you'll want
for the display and control loop.

### Experiment tracking

Every run writes to `runs/detect/<name>/` and `scripts/train.py` additionally snapshots the git SHA,
the resolved config, and the dataset `MANIFEST.json` into the run dir. Six weeks from now, "which data
made the good model?" must be answerable from the run directory alone.

### Evaluation

1. `model.val()` on val → mAP50, mAP50-95, **per-class** AP, confusion matrix.
2. Final check on the held-out `test/` session — the number you actually believe.
3. `bench` command: mean/p95 latency and FPS at the real deployment resolution, `.pt` vs `.engine`.
4. Qualitative: run on a full unseen video, watch it. Metrics hide flicker and ID-switching; your eyes
   don't.

### Export path

`.pt` (train/debug) → ONNX → TensorRT `.engine` (deploy). YOLO26's NMS-free one-to-one head makes this
much cleaner than previous versions — no NMS plugin to wire up. YOLO26 exposes an `end2end` toggle to
export the traditional one-to-many head instead if a tool needs it. Note that a TensorRT engine is
built for one specific GPU + driver + resolution and is not portable; keep the `.pt` as the artifact of
record and treat `.engine` as a build product.

### NMS-free gotcha

With the one-to-one head there is no NMS, so `iou`/`agnostic_nms` no longer do anything, and `conf`
becomes the sole knob controlling the precision/recall tradeoff. `max_det` caps output directly. Expect
threshold tuning to behave differently than with YOLOv8/v11.

---

## 5. Camo Studio integration (Phase 4)

Camo Studio installs a **virtual webcam driver** on Windows, so the iPhone appears as an ordinary
capture device — no iOS-specific code needed. OpenCV opens it like any webcam.

**Enumerate by name, never by hardcoded index.** Device indices shift whenever a camera is plugged in,
Camo restarts, or Windows Hello wakes the IR camera — this machine already has an Intel UHD iGPU camera
stack alongside it. `WebcamSource` resolves `"Camo"` to an index at startup via DirectShow enumeration
(`pygrabber`), and fails with a listing of available devices if not found.

```python
cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # try CAP_MSMF as fallback
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,          60)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)          # honored by DSHOW, ignored by MSMF
```

Always **read back** what you set — Camo silently substitutes the nearest supported mode, and a config
that claims 60 fps while delivering 30 will send you debugging the wrong layer. Log the actual values.

Camo tuning that matters: lock exposure and white balance in Camo Studio. Auto-exposure changes the
image statistics between training capture and inference, and it will visibly move your confidence
scores. Disable Camo's beautify/filters entirely.

Latency budget to measure (`ts_mono` makes this possible): iPhone encode → USB/Wi-Fi → Camo driver →
OpenCV → inference → policy → serial. Use USB, not Wi-Fi; Wi-Fi adds tens of milliseconds of jitter,
which is worse than a constant delay for control.

Fallback if the Camo driver misbehaves under OpenCV: an RTSP/MJPEG iOS streaming app, consumed by
`RTSPSource`. Same `FrameSource` interface, one config line.

---

## 6. Control layer (Phase 7 — designed now, built later)

Kept behind `ControlBackend` so it can be developed against `NullBackend` and a recorded video, with no
hardware attached.

```
PC ──USB serial──► Arduino/ESP32 ──PWM──► servos   (add PCA9685 if >4 channels)
```

Line protocol, ASCII, ack'd — trivially debuggable in a serial monitor:

```
→  S <ch> <deg> <speed>\n     set channel target
→  P\n                        ping
←  OK <ch> <deg>\n  |  ERR <code>\n
```

Safety rules, all enforced on the **firmware** side too, never only in Python (the PC can crash, hang,
or lose USB):

- **Angle clamp** — per-channel min/max in firmware; commands outside are rejected, not clipped silently.
- **Slew limit** — cap deg/s so a detection glitch can't snap a servo and strip a gear.
- **Watchdog** — no valid command for 500 ms → return to neutral and hold.
- **Deadband + hysteresis** — ignore sub-threshold changes, or the servo will buzz continuously on
  detection jitter. Requires a track to persist N consecutive frames before acting.
- **EMA smoothing** on the target angle, since detector output is inherently noisy frame to frame.

This is where the tracker earns its place: acting on raw per-frame detections produces jitter and
flicker. Acting on a *track* with a stable ID and a few frames of persistence produces smooth motion.

---

## 7. Repository layout

```
Phone-Detection-Project/
├── configs/
│   ├── classes.yaml               # append-only taxonomy
│   ├── train/*.yaml               # one file per experiment
│   └── runtime/{live,offline}.yaml
├── datasets/                      # git-ignored except MANIFESTs
├── models/                        # .pt / .onnx / .engine (git-ignored, LFS if versioned)
├── runs/                          # ultralytics output (git-ignored)
├── docs/ARCHITECTURE.md
├── pdp/
│   ├── config/  sources/  detect/  track/  logic/  control/  sinks/
│   └── cli.py   types.py
├── scripts/                       # thin shims over `pdp <command>`
│   ├── check_env.py               # asserts CUDA, prints device — run first, always
│   └── list_cameras.py            # DirectShow enumeration for Camo
├── tests/                         # contract + policy tests (no GPU needed)
├── pyproject.toml
└── .gitignore                     # datasets/, runs/, models/, *.engine, .venv/
```

---

## 8. Phased roadmap

| Phase | Deliverable | Done when |
|---|---|---|
| 0 · Env | venv, CUDA torch, ultralytics, `check_env.py`, skeleton, `.gitignore` | `check_env.py` prints `True` + GPU name; `yolo26n.pt` predicts on a sample image |
| 1 · Data | `classes.yaml`, capture sessions, annotation, `build_dataset.py` + validator | `obstacles_v1` builds clean, session-split, class histogram reviewed |
| 2 · Train | `train.py`, baseline run, `val.py` | Baseline mAP50-95 recorded per class on val; run dir has git SHA + manifest |
| 3 · Offline | `FileSource` → detect → annotated video + JSONL | Full unseen video processed, output watched and judged acceptable |
| 4 · Camo | `list_cameras.py`, `WebcamSource`, capture thread | Live iPhone feed detecting at ≥25 FPS, actual resolution/FPS logged |
| 5 · Logic | Tracker, zones, debounce, `NullBackend` | Stable track IDs; command log looks correct with zero hardware attached |
| 6 · Optimize | ONNX/TensorRT export, `bench.py` | p95 latency measured `.pt` vs `.engine`; accuracy parity confirmed |
| 7 · Servos | Firmware, `SerialServoBackend`, safety | Watchdog, clamp, slew verified by unplugging USB mid-run |

Phases 3 and 5 exist specifically so the whole pipeline is debuggable on recorded video, with no phone
and no hardware in the loop. Every later phase then changes exactly one component.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Python 3.14 wheel gaps (`cv2`, `onnxruntime`, `tensorrt`) | Use a 3.12 venv (§0) |
| CPU torch installed globally → silent 50× slow training | `check_env.py` hard-fails on `cuda.is_available() == False` |
| 8 GB VRAM ceiling | `yolo26n/s` @ 640, AMP, autobatch; `imgsz=1280` only with reduced batch |
| Overfitting on a small dataset | `freeze`, reduced mosaic, `patience`, separate-day test session |
| Train/val leakage from adjacent video frames | Session-level splits + perceptual-hash dup check |
| Class imbalance | Histogram at build time; target ≥150 instances/class |
| Domain gap (web images vs Camo feed) | Train primarily on iPhone-through-Camo captures |
| Camo device index drift | Resolve by device name at startup |
| Servo jitter from per-frame noise | Track persistence + EMA + deadband + slew limit |
| TensorRT engine non-portability | `.pt` is the artifact of record; `.engine` is a build product |

---

## 10. Open questions

1. **Which classes?** Needed to write `classes.yaml` and size the annotation effort.
2. **Do datasets exist yet**, or is annotation starting from zero video?
3. **Annotation tool** — Roboflow (fast, cloud) vs Label Studio (local, private)?
4. **What do the servos do** — pan/tilt to track a target, or discrete avoidance moves? This determines
   whether `Policy` outputs continuous angles or discrete states, and it is worth settling before
   Phase 5.
5. **Frame budget** — target FPS and acceptable glass-to-servo latency.

---

## References

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/)
- [Fine-tune YOLO26 on a custom dataset](https://docs.ultralytics.com/guides/finetuning-guide/)
- [Detection dataset format](https://docs.ultralytics.com/datasets/detect/)
- [Train mode arguments](https://docs.ultralytics.com/modes/train/)
