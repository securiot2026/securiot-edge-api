# SecurIoT Edge API

Flask + Peewee + SQLite edge service. It buffers sensor readings locally the
moment a device posts them, then a background relay forwards them to the
Cloud API with retry/backoff so a connectivity interruption never loses data.

## Prerequisites

- Python 3.12+

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

Phase 4 (camera detection) config, all optional with working defaults, add to `.env.example` and `.env` if you want to override them:

- `DETECTION_BACKEND`: `mock` (default, no ML dependencies) or `yolo` (real Ultralytics YOLOv8n inference)
- `YOLO_MODEL_PATH`: local path to YOLO weights, skips the automatic first-run download when set
- `ALLOWED_OBJECT_CLASS`: the non-person COCO class treated as a not-allowed object (default in `app/config.py`)
- `DETECTION_CONFIDENCE_THRESHOLD`: minimum confidence for a detection to count
- `DETECTION_DEBOUNCE_COUNT`: consecutive qualifying frames required before escalating
- `DOOR_ACTION_COOLDOWN_SECONDS`: minimum time between two door-lock actions for the same device
- `MAX_CONTENT_LENGTH`: maximum accepted size (bytes) for an uploaded frame

## Running

```bash
flask --app wsgi run
```

The background relay worker starts automatically with the app (as an
APScheduler job running in a separate thread), so ingest requests never wait
on a Cloud API call.

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
python -m pytest tests/test_relay.py
```

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
