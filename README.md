# SecurIoT Edge API

Flask + Peewee + SQLite edge service. It buffers sensor readings locally the
moment a device posts them, then a background relay forwards them to the
Cloud API with retry/backoff so a connectivity interruption never loses data.

## Where this runs

This service runs on a machine local to the installation site, on the same
LAN as the ESP32-CAM, for example Juan's laptop during the demo, or a small
on-site PC/mini-PC in a real deployment. It does **not** run on the Cloud
API's VPS:

- YOLO inference is CPU-heavy (hundreds of ms per frame even on `yolov8n`),
  more than a small VPS can absorb alongside everything else running there.
- The ESP32-CAM needs LAN-level latency to `/frames`, since it posts a
  frame roughly every 2 seconds and reacts to the response in near
  real-time (pan/tilt tracking, door lock).

The only thing that needs internet access from this machine is the relay's
outbound call to the Cloud API (`CLOUD_API_URL`); the ESP32-CAM never talks
to the Cloud API directly, and the Cloud API never talks back to this
machine, everything from the Cloud API's side is a normal inbound HTTPS
request this service initiates.

## Prerequisites

- Python 3.12+
- Local disk space for the YOLO, PyTorch, ONNX Runtime, and ArcFace artifacts
- Face-specific Ultralytics YOLO `.pt` weights and ArcFace `.onnx` weights when
  facial recognition is enabled

## Setup

```bash
cd repos/securiot-edge-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

- `EDGE_DB_PATH`: path to the local SQLite buffer file (default `edge.db`)
- `DEVICE_SHARED_SECRET`: shared secret devices must send in `X-Device-Key`
- `CLOUD_API_URL`: base URL of the Cloud API this Edge instance relays to
- `CLOUD_DEVICE_API_KEY`: key the Edge API sends to the Cloud API
- `RELAY_INTERVAL_SECONDS`: how often the relay checks for unsynced readings

Camera detection config is listed in `.env.example` and can be overridden in
`.env`:

- `DETECTION_BACKEND`: `mock` (default, no ML dependencies) or `yolo` (real Ultralytics YOLOv8n inference)
- `YOLO_MODEL_PATH`: local path to YOLO weights, skips the automatic first-run download when set
- `ALLOWED_OBJECT_CLASS`: the non-person COCO class treated as a not-allowed object (default in `app/config.py`)
- `DETECTION_CONFIDENCE_THRESHOLD`: minimum confidence for a detection to count
- `DETECTION_DEBOUNCE_COUNT`: consecutive qualifying frames required before escalating
- `DOOR_ACTION_COOLDOWN_SECONDS`: minimum time between two door-lock actions for the same device
- `MAX_CONTENT_LENGTH`: maximum accepted size (bytes) for an uploaded frame

On Linux or Windows hosts where a CPU-only installation is required, install
the CPU PyTorch wheels before the project requirements so Ultralytics reuses
them instead of selecting a CUDA build:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## YOLO + ArcFace facial recognition

The optional vision module detects every face with a face-specific YOLO model,
extracts each face crop, produces a normalized ArcFace embedding, and compares
that embedding with locally enrolled identity centroids. It reports either the
best known identity or `unknown` in the existing `/frames` response. It does
not change the current object/person alert, debounce, pan/tilt, door-action, or
relay behavior.

The complete processing flow is:

```text
Camera / frame
  -> YOLO face detection
  -> face bounding box
  -> expanded square face crop
  -> ArcFace ONNX embedding
  -> cosine-similarity comparison
  -> identity / unknown
  -> Edge API response
```

### Architecture and responsibilities

The implementation is isolated under `app/vision`; the existing endpoint only
calls its public factory and serializes results.

```text
app/vision/
├── detector.py    # Face-specific Ultralytics YOLO inference
├── embedder.py    # ArcFace ONNX preprocessing and embedding inference
├── image.py       # Crop and embedding-normalization helpers
├── storage.py     # Local identity centroids and cosine matching
├── service.py     # YOLO -> crop -> ArcFace -> identity orchestration
├── enrollment.py  # Offline identity enrollment from local images
├── settings.py    # Validated configuration boundary
├── factory.py     # Lazy, cached model construction
└── errors.py      # Vision-specific operational errors
```

- **YOLO** is the only face detector. The configured model must be trained to
  detect faces; stock COCO checkpoints such as `yolov8n.pt` detect people but
  not face boxes. Inference receives the confidence, IoU, input-size,
  class-ID, maximum-detection, and device settings from configuration.
- **ArcFace** performs recognition only. The module runs a compatible NCHW
  ArcFace ONNX model directly through ONNX Runtime and never invokes a second
  detector. Each output embedding is L2-normalized.
- **Matching** uses vectorized cosine similarity against one normalized
  centroid per enrolled identity. A result is known only when its best score
  reaches `FACE_SIMILARITY_THRESHOLD`.

No model weights, captured face images, or enrolled `.npz` identities are
included in this repository.

### Technologies and runtime dependencies

| Component | Library | Purpose |
|---|---|---|
| Face detection | `ultralytics` + PyTorch | Load and run face-specific YOLO weights |
| Face recognition | `onnxruntime` | Run ArcFace ONNX inference, with CPU support by default |
| Image processing | `opencv-python` | Decode frames, crop/resize faces, and prepare NCHW tensors |
| Vector matching | `numpy` | Normalize embeddings and compute cosine similarities |
| Edge integration | Flask | Return recognition results from `POST /frames` |

`requirements.txt` installs the CPU ONNX Runtime package. For NVIDIA
acceleration, install a CUDA-compatible PyTorch build, replace `onnxruntime`
with the matching `onnxruntime-gpu` release, and set the device variables below.
Do not keep both ONNX Runtime packages installed in the same environment.

### Model files

Model files are external runtime artifacts and are ignored by Git. Create a
local models directory and place both compatible files there:

```text
models/
├── yolov8n-face.pt   # Face-specific Ultralytics detector
└── w600k_r50.onnx    # ArcFace recognition model
```

The filenames are defaults, not requirements; set `FACE_YOLO_MODEL_PATH` and
`ARCFACE_MODEL_PATH` to other relative or absolute runtime paths when needed.
The service does not silently download weights. Verify each model's origin,
checksum, input contract, and license before deployment. In particular,
pretrained InsightFace/ArcFace model licensing may restrict commercial use.

Docker builds also exclude models and identities. Mount both directories at
runtime and set container paths explicitly, for example:

```bash
docker run --rm -p 5000:5000 \
  --env-file .env \
  -e FACE_YOLO_MODEL_PATH=/models/yolov8n-face.pt \
  -e ARCFACE_MODEL_PATH=/models/w600k_r50.onnx \
  -e FACE_IDENTITY_DIR=/identities \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/data/faces:/identities:ro" \
  securiot-edge-api
```

### Facial-recognition configuration

| Variable | Default | Description |
|---|---:|---|
| `FACE_RECOGNITION_ENABLED` | `false` | Enable YOLO + ArcFace processing on `/frames` |
| `FACE_YOLO_MODEL_PATH` | `models/yolov8n-face.pt` | Face-specific YOLO `.pt` path |
| `FACE_YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Minimum face-detection confidence |
| `FACE_YOLO_IOU_THRESHOLD` | `0.45` | YOLO non-maximum-suppression IoU threshold |
| `FACE_YOLO_INPUT_SIZE` | `640` | YOLO inference image size |
| `FACE_YOLO_MAX_DETECTIONS` | `10` | Maximum face boxes per frame |
| `FACE_YOLO_CLASS_ID` | `0` | Face class ID in the custom YOLO model |
| `FACE_YOLO_DEVICE` | `cpu` | PyTorch device: `cpu`, `cuda:0`, `mps`, or `auto` |
| `ARCFACE_MODEL_PATH` | `models/w600k_r50.onnx` | ArcFace ONNX model path |
| `ARCFACE_DEVICE` | `cpu` | ONNX provider policy: `cpu`, `cuda`, `coreml`, or `auto` |
| `FACE_IDENTITY_DIR` | `data/faces` | Local directory containing enrolled identities |
| `FACE_SIMILARITY_THRESHOLD` | `0.45` | Minimum cosine score for a known identity |
| `FACE_CROP_MARGIN` | `0.2` | Proportional margin added around each YOLO box |
| `FACE_MINIMUM_ENROLLMENT_IMAGES` | `3` | Valid single-face images required for enrollment |

CPU is the default and minimum supported path. `auto` prefers CUDA, then MPS
for YOLO; ArcFace prefers CUDA, then CoreML, while retaining CPU fallback.
Calibrate the detection and similarity thresholds with the actual camera,
lighting, face detector, and enrolled population before production use.

### Enroll an identity

Use multiple local images with one clear face each. Enrollment applies the
same YOLO crop and ArcFace model used at runtime and writes only an identity
file under `FACE_IDENTITY_DIR`:

```bash
python scripts/enroll_face.py --name "Ada Lovelace" \
  samples/ada-front.jpg samples/ada-left.jpg samples/ada-right.jpg
```

Images with zero or multiple detected faces are skipped. Re-enrolling the same
normalized name atomically replaces its `identity.npz`. Restart a running
multi-process server after enrollment so every worker reloads the identity
database.

### Enable and test recognition

Set at least the following values in `.env`, then start the service normally:

```dotenv
FACE_RECOGNITION_ENABLED=true
FACE_YOLO_MODEL_PATH=models/yolov8n-face.pt
ARCFACE_MODEL_PATH=models/w600k_r50.onnx
FACE_IDENTITY_DIR=data/faces
FACE_YOLO_DEVICE=cpu
ARCFACE_DEVICE=cpu
```

Post a real JPEG frame:

```bash
curl -X POST localhost:5000/frames \
  -H "X-Device-Key: $DEVICE_SHARED_SECRET" \
  -F "device_id=dev-1" \
  -F "zone_id=zone-1" \
  -F "frame=@samples/test-face.jpg;type=image/jpeg"
```

When recognition is enabled, the otherwise backward-compatible response adds
a `faces` array:

```json
{
  "pan_delta": null,
  "tilt_delta": null,
  "door_action": null,
  "alert": false,
  "faces": [
    {
      "bbox": [118, 72, 286, 240],
      "detection_confidence": 0.94,
      "identity": "Ada Lovelace",
      "similarity": 0.81,
      "known": true
    }
  ]
}
```

With no enrolled identities, detected faces are returned as `unknown` with a
`null` similarity. A missing or incompatible model while recognition is
enabled returns `503 face recognition unavailable`; when disabled, neither
model is loaded and the existing response contract remains unchanged.

Face crops and embeddings are biometric data. Obtain explicit consent,
restrict filesystem access to `FACE_IDENTITY_DIR`, define retention/deletion
rules, and do not use this prototype as the sole basis for high-impact access
or safety decisions. The module does not provide liveness or anti-spoofing.
The configured YOLO model supplies boxes rather than facial landmarks, so
ArcFace receives square crops without landmark alignment; extreme poses can
therefore reduce recognition accuracy.

## Running

This service is meant to run on a machine local to the installation site
(the same LAN as the ESP32-CAM), not on the Cloud API's VPS: YOLO inference
is too heavy for a small VPS, and the device needs LAN-level latency to the
camera-detection endpoint anyway. See "LAN discovery (mDNS)" below for how
the ESP32-CAM finds this machine automatically.

```bash
flask --app wsgi run --host=0.0.0.0
```

`--host=0.0.0.0` is required, not optional: Flask's default (`127.0.0.1`)
only accepts connections from the same machine, and the ESP32-CAM reaches
this over the LAN.

The background relay worker starts automatically with the app (as an
APScheduler job running in a separate thread), so ingest requests never wait
on a Cloud API call.

## LAN discovery (mDNS)

The firmware (`securiot-embedded`) resolves this machine's address via mDNS
by default, querying the hostname `edge-api.local`, so nobody has to hardcode
an IP that changes every time this machine reconnects to WiFi. This uses
whatever mDNS responder already ships with the OS, no extra service to run:

- **macOS**: Bonjour is built in. Set this machine's hostname once:
  ```bash
  sudo scutil --set LocalHostName edge-api
  ```
  Confirm it took with `scutil --get LocalHostName` (should print `edge-api`)
  and, from another device on the same WiFi, `ping edge-api.local`.
- **Linux**: install and enable Avahi if it is not already running
  (`sudo apt install avahi-daemon`, most desktop distros have it by
  default), then set the hostname with `sudo hostnamectl set-hostname
  edge-api`.
- **Windows**: mDNS needs Bonjour installed (ships with iTunes, or install
  "Bonjour Print Services" standalone). Without it, skip mDNS and use the
  static fallback below.

If mDNS is unavailable or blocked on your router (some guest networks
isolate mDNS traffic on purpose), comment out `EDGE_API_MDNS_HOST` in the
firmware's `src/secrets.h` and set `EDGE_API_URL` to this machine's LAN IP
instead, the firmware falls back to it automatically.

## Sending a reading

```bash
curl -X POST localhost:5000/ingest \
  -H "X-Device-Key: $DEVICE_SHARED_SECRET" \
  -H 'Content-Type: application/json' \
  -d '{"reading_id":"r-1","device_id":"dev-1","zone_id":"zone-1","sensor_type":"door_contact","value":"open","recorded_at":"2026-09-07T00:00:00Z"}'
```

A valid request returns `201` and buffers the reading immediately; a missing
or wrong `X-Device-Key` returns `401`.

## Verifying the offline-buffering scenario manually

1. Stop the Cloud API (or point `CLOUD_API_URL` at an address nothing is
   listening on).
2. POST one or more readings to `/ingest`. Each returns `201`.
3. Inspect `edge.db` and confirm the rows have `synced=0` and a growing
   `sync_attempts` count (the relay logs a connection error each cycle
   instead of crashing).
4. Restart the Cloud API (or point `CLOUD_API_URL` back at the real one).
5. Wait for the next relay cycle (`RELAY_INTERVAL_SECONDS`) and confirm the
   rows flip to `synced=1` in `edge.db`, and that the Cloud API shows exactly
   one row per `reading_id` (no duplicates), since forwarding is idempotent
   on `reading_id`.

## Tests

```bash
python -m pytest
```

Automated tests use fakes and generated in-memory JPEGs, so they do not load
model weights, download artifacts, open a camera, or require a GPU. Real-model
accuracy must be tested separately with representative, consented images.

## Camera detection contract (Phase 4)

`POST /frames` is the single HTTP contract the ESP32-CAM firmware (Phase 4,
plan 04-03) implements against. One request, one response: upload a frame,
get back the servo/alert command for that frame.

### Request

- Method: `POST /frames`
- Header: `X-Device-Key: <shared secret>` — same check as `/ingest`, required
- Content-Type: `multipart/form-data`
- Form fields:
  - `device_id` (string, required)
  - `zone_id` (string, required)
  - `frame_width` / `frame_height` (integers, optional) — declared frame
    dimensions used for the pan/tilt calculation; default to 640x480 if
    omitted
  - `frame` (file, required) — the JPEG frame, as a multipart file field
    named exactly `frame`

### Response (200)

```json
{
  "pan_delta": -0.5625,
  "tilt_delta": 0.0,
  "door_action": "lock",
  "alert": true
}
```

| Field | Type | Nullable | Description |
|---|---|---|---|
| `pan_delta` | float in `[-1.0, 1.0]` | yes | Horizontal servo delta; negative = pan left. `null` unless a `person` is the confirmed detection in this frame. |
| `tilt_delta` | float in `[-1.0, 1.0]` | yes | Vertical servo delta; negative = tilt up. `null` under the same condition as `pan_delta`. |
| `door_action` | string (`"lock"`) | yes | `"lock"` when this frame fires the door-lock action; `null` when no confirmed detection this frame, or when a detection is confirmed but the device is still within its `DOOR_ACTION_COOLDOWN_SECONDS` window (pan/tilt still tracks in that case; the door just doesn't re-fire). |
| `alert` | boolean | no | `true` whenever a confirmed detection is active in this frame (drives a local buzzer/LED), `false` otherwise. |

Other responses: `401` (missing/wrong `X-Device-Key`), `400` (undecodable
image data), `413` (frame exceeds `MAX_CONTENT_LENGTH`).

### Manual test with curl

```bash
curl -X POST localhost:5000/frames \
  -H "X-Device-Key: $DEVICE_SHARED_SECRET" \
  -F "device_id=dev-1" \
  -F "zone_id=zone-1" \
  -F "frame=@tests/fixtures/sample_frames/person_left.jpg;type=image/jpeg"
```

### Mock vs real detection backend

`DETECTION_BACKEND=mock` (default) uses a filename-convention detector with
zero ML dependencies, so automated tests never need a camera, network
access, or model weights. To run real detection:

1. Set `DETECTION_BACKEND=yolo` in `.env`.
2. On first run, Ultralytics downloads `yolov8n.pt` automatically — this
   needs network access. It's a one-time manual step (Juan's environment),
   not part of the automated test suite.
3. Optionally point `YOLO_MODEL_PATH` at an existing local weights file to
   skip the download.

### Choosing what gets flagged (no custom training)

`yolov8n.pt` is Ultralytics' pretrained checkpoint on COCO, 80 general
object classes. `app/detection.py` only keeps two of them: `person` (always)
and whatever `ALLOWED_OBJECT_CLASS` is set to (default `backpack`).

This project does not fine-tune or train a custom model, that decision is
recorded in `.planning/PROJECT.md`: no dataset and no time for it within
the course schedule. If what you want to flag is already one of COCO's 80
classes (`knife`, `suitcase`, `handbag`, `cell phone`, `scissors`,
`baseball bat`, and so on, full list at
https://docs.ultralytics.com/datasets/detect/coco/), just set
`ALLOWED_OBJECT_CLASS` to that class name, no retraining needed. If it is
not one of those 80, detecting it would need a labeled dataset and a
fine-tuning pass, out of scope for now.

### End-to-end harness (no camera required)

```bash
python scripts/run_test_folder.py
```

Replays every image in `tests/fixtures/sample_frames/` (regenerate via
`python scripts/generate_test_images.py` if missing) through `/frames` for
one simulated device and prints a per-image summary — proving the full
frame → detection → debounce → telemetry → relay path end to end.

## Device simulator

`simulator/device_simulator.py` stands in for the physical ESP32 (built in
Phase 4). It posts synthetic door-contact readings to this API's `/ingest`
endpoint on a timer - see its own `simulator/requirements.txt` for its one
extra dependency (`requests`, already covered above).

For the full local demo (Cloud API + this Edge API + the simulator running
together, including how to reproduce the connectivity-interruption test),
see `docs/demo/phase-1-local-run.md` in the main planning repo.
