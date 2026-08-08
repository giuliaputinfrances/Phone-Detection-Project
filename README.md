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

In one sentence: images come in from a file or the iPhone → the model finds
objects → the logic decides which one matters and where to point → the safety
layer turns that into servo movements → and along the way it saves a video, a
log, and speed numbers so you can see what happened.

### The folders at a glance

| Folder | In plain words |
|---|---|
| `configs/` | The settings files. All the knobs live here so you never edit code to change a number. |
| `pdp/` | The actual program. Everything the project *does* is in here. |
| `scripts/` | Two shortcuts for running common checks. |
| `tests/` | Automatic checks that prove the code works, without needing a GPU, camera, or servos. |
| `datasets/` | Your photos and labels. Deliberately not saved to git — too big. |
| `models/` | Where trained model files go. Also not saved to git. |
| `runs/` | Whatever training and evaluation spits out — charts, weights, logs. Created automatically. |
| `docs/` | The design document explaining why the project is built the way it is. |
| `.venv/`, `pdp.egg-info/`, `.pytest_cache/` | Tool-generated. Ignore them completely. |

### Root files

| File | What it's for |
|---|---|
| `README.md` | This file: setup, workflow, layout, and an honest list of what's tested and what isn't. |
| `pyproject.toml` | Tells Python what libraries the project needs, and creates the `pdp` command you type in the terminal. |
| `.gitignore` | List of things git should not save — videos, model weights, datasets. |
| `yolo26n.pt` | The pre-trained model that already knows COCO objects. Training starts from this instead of from nothing. |

### `configs/` — the settings

| File | What it's for |
|---|---|
| `classes.yaml` | The list of things you want to detect. **Only add to the bottom, never reorder** — the numbers are written into every label file you make, so shuffling them scrambles all your labelling work. |
| `train/obstacles_v1_n.yaml` | One training recipe: epochs, image size, learning rate, etc. Copy it for each new experiment instead of editing it, so runs stay comparable. |
| `runtime/offline.yaml` | Settings for running on a **recorded video**. The safe debugging mode: no phone, no hardware, same result every time. |
| `runtime/live.yaml` | Settings for running on the **live iPhone feed** through Camo. Stricter confidence threshold, because here a mistake actually moves a motor. |

### `pdp/` — top-level files

| File | What it's for |
|---|---|
| `types.py` | The four "shapes of data" every part of the program passes around: a Frame (one image), a Detection (one box), a DetectionResult (all boxes for one frame), and a Command (move servo X to angle Y). Because everything speaks this small common language, any part can be swapped without breaking the others. |
| `cli.py` | The menu. Every `pdp something` command is defined here and handed to the right code. |
| `pipeline.py` | The main loop: grab a frame → detect → decide → send commands → save/show the result. There's only **one** of these, used by both video files and the live camera, so what you debug on video is literally what runs on the phone. |
| `training.py` | Training, evaluating, exporting and speed-testing the model. Also saves a note in each run folder recording which code version and which dataset produced it. |
| `env.py` | Checks your setup is correct. Exists for one nasty problem: the CPU-only PyTorch installs fine and then trains ~50x slower with no error at all. |
| `__init__.py` | Plumbing. Makes the folder importable. |

### `pdp/config/` — reading the settings files

| File | What it's for |
|---|---|
| `schema.py` | Reads the YAML settings and checks them. Typo `imgz` instead of `imgsz` and it stops and lists the valid options instead of silently ignoring it. |
| `__init__.py` | Plumbing. |

### `pdp/sources/` — where images come from

| File | What it's for |
|---|---|
| `base.py` | The rulebook: any image source must open, give the next frame, and close. Nothing else cares whether that's a file or a camera. |
| `file.py` | Reads a video file. |
| `webcam.py` | Reads the iPhone through Camo. Finds the camera **by name**, not by number, because camera numbers shift whenever you plug something in. Reports what the camera actually gave you, since Camo quietly ignores requests it can't fulfil. |
| `threaded.py` | Grabs frames in the background and keeps only the newest. If detection can't keep up you get *fewer but current* frames instead of falling further behind. |
| `__init__.py` | Picks the right source based on your config file. |

### `pdp/detect/` — the AI model

| File | What it's for |
|---|---|
| `detector.py` | The only file that knows anything about YOLO. Loads the model, warms it up, runs it, keeps stable IDs on objects as they move, and converts the output into plain Detections. |
| `__init__.py` | Plumbing. |

### `pdp/logic/` — deciding what to do

| File | What it's for |
|---|---|
| `policy.py` | Turns detections into a decision. Acts on **tracked objects, not single frames**: a box must appear several frames in a row before it's taken seriously, and survives a few frames of disappearing behind something. Picks the most important object (person beats cone beats box) and smooths the aim so the servo glides instead of twitching. |
| `__init__.py` | Plumbing. |

### `pdp/control/` — moving the servos

| File | What it's for |
|---|---|
| `base.py` | The rulebook for anything that receives commands. |
| `null.py` | A fake servo that writes down what it *would* have done. This is why the whole system works today with no hardware attached. |
| `serial_servo.py` | The real servo driver, talking to an Arduino over USB in simple text commands. **Written but never tested** — no board exists yet. |
| `loop.py` | The safety layer, on its own timer. Never exceed the angle limits; never move faster than X degrees per second (a glitch can't strip a gear); ignore tiny movements (or the servo buzzes); return to neutral if no command arrives for half a second. Runs on a separate thread so a stuck USB cable can never freeze the camera. |
| `__init__.py` | Picks fake or real servos based on your config. |

### `pdp/sinks/` — outputs you can look at

| File | What it's for |
|---|---|
| `draw.py` | Draws the boxes, labels, zone lines, and the FPS counter onto the image. |
| `writers.py` | Saves the annotated video, writes a line-by-line log of everything detected and every command sent, and tracks speed. Reports the 95th-percentile time as well as the average, because for controlling a motor the *worst* frames matter more than the typical one. |
| `__init__.py` | Plumbing. |

### `pdp/data/` — preparing training data

| File | What it's for |
|---|---|
| `frames.py` | Turns a video into still images to label. Takes ~2 per second and throws away near-identical ones — labelling 400 near-identical frames costs hours and teaches the model nothing. |
| `hashing.py` | The "are these two images basically the same?" maths behind that. |
| `build.py` | Assembles labelled images into a training dataset. Splits by **recording session**, never by individual frame: frames a second apart look identical, so a random split puts the same moment in both the training and testing piles, and then your test score looks great and means nothing. Also writes a MANIFEST recording exactly what went in. |
| `validate.py` | Catches labelling mistakes before you waste hours training: wrong coordinate format, class numbers that don't exist, label files with no matching image, boxes too small to be real, duplicate images leaking between splits. |
| `__init__.py` | Plumbing. |

### `scripts/` and `tests/`

| File | What it's for |
|---|---|
| `scripts/check_env.py` | Shortcut for `pdp check-env`. Run this first, always. |
| `scripts/list_cameras.py` | Shortcut for `pdp cameras`. Confirms Camo is visible. |
| `tests/test_config.py` | Proves bad settings files get rejected with a useful message. |
| `tests/test_control.py` | Proves the servo safety rules work — limits, speed cap, watchdog, anti-buzz. |
| `tests/test_data.py` | Proves bad labels get caught and the session-split rule is enforced. |
| `tests/test_policy.py` | Proves the targeting logic waits for confirmation, prefers important objects, and doesn't sweep wildly when switching targets. |
| `tests/test_sources.py` | Proves the background frame grabber really does drop stale frames. |

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
