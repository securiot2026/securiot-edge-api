# Run the ESP32 camera console on a Mac

Run the real Edge API, monitor the ESP32-CAM, and register a consenting
person locally at **http://127.0.0.1:5050/**. This is an opt-in development
console, not a production biometric or access-control system.

## Quick start

### 1. Prepare the camera

Use [securiot-embedded's preview mode](https://github.com/securiot2026/securiot-embedded/blob/feature/esp32-cam-yolo-detection/docs/CAMERA_PREVIEW.md)
with `CAMERA_PREVIEW_ONLY=1`. Join the Mac and camera to the same private
Wi-Fi network. Read the camera's current DHCP address from its serial log;
`172.20.10.2` below is an example, not a guaranteed address.

### 2. Install dependencies and supply model files

From this repository's `develop` branch, use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Provide these files locally; they are deliberately not in Git:

| Model | Purpose | Example location |
| --- | --- | --- |
| `yolov8n.pt` | General person/object detection | `models/yolov8n.pt` |
| `yolov8n-face.pt` | Face-specific YOLO detector | `models/yolov8n-face.pt` |
| `w600k_r50.onnx` | ArcFace embeddings | `models/w600k_r50.onnx` |

Use trusted model sources and check their provenance and licenses. COCO
weights such as `yolov8n.pt` do **not** replace the face-specific detector.
Pretrained InsightFace recognition weights may be restricted to
noncommercial research; check licensing before other uses.

Configure your own paths in the ignored `.env`:

```dotenv
EDGE_DB_PATH=edge.db
DETECTION_BACKEND=yolo
YOLO_MODEL_PATH=models/yolov8n.pt
FACE_RECOGNITION_ENABLED=true
FACE_YOLO_MODEL_PATH=models/yolov8n-face.pt
ARCFACE_MODEL_PATH=models/w600k_r50.onnx
FACE_IDENTITY_DIR=data/faces
FACE_YOLO_DEVICE=cpu
ARCFACE_DEVICE=cpu
FACE_MINIMUM_ENROLLMENT_IMAGES=3
```

Absolute paths to existing local models also work. CPU inference is the
verified baseline; accelerator providers are not required for this demo.

### 3. Start the console

```bash
python scripts/run_local_console.py --camera-url http://172.20.10.2 --port 5050
```

Open **http://127.0.0.1:5050/** on the same Mac. The launcher binds only to
loopback, starts one camera bridge, and disables Flask's reloader. Port
5050 avoids macOS AirPlay Receiver/Control Center, which can occupy 5000.
Use `Ctrl+C` in that terminal to stop it.

The normal `flask --app wsgi run --host=0.0.0.0` workflow is different:
it starts the regular LAN API and relay, but does not add this console.
Do not change the console to `0.0.0.0` or expose it through a public tunnel.

## What is connected

```text
ESP32 /capture ──> Mac snapshot bridge ──> authenticated Edge POST /frames
                                              │
                                      YOLO + ArcFace
                                              │
                                   status + recognition UI

Browser ──> ESP32 :81/stream       (optional full-rate video)
Browser ──> Edge /console/enroll  (explicit local registration)
```

- The bridge pulls JPEG snapshots, validates them, and sends device/zone
  metadata through the existing `/frames` contract.
- Its API key is generated at launch and stays server-side. It is not a
  Wi-Fi password and is not sent to the ESP32 or rendered in the page.
- Requests to the private camera and loopback API ignore environment proxies.
- Cloud relay is disabled. SQLite still buffers qualifying detection events
  locally; they are not forwarded to the cloud by this launcher.
- Returned pan/tilt, door, or alert fields are **not applied** to hardware.
  Preview mode deliberately skips servo initialization on the ESP32.

## Use the interface

| Area | Meaning |
| --- | --- |
| Live camera | Full-rate MJPEG or the latest frame analyzed by the bridge |
| Processed frames | Successful real `/frames` responses since this launch |
| Analysis latency | Time spent in the local `/frames` round trip, excluding camera fetch |
| Faces in frame | Face-specific YOLO detections in the latest result |
| Recognition results | Identity and similarity, or `Unknown face` |
| Register a face | Consent-gated local capture and enrollment |

Video FPS and analysis throughput are different. The bridge targets one
cycle every 0.5 seconds, but slow inference can reduce that rate. Live
MJPEG can remain faster. Bounding boxes come from the last analysis and can
lag the live video; the processed view is available for inspection.

The ESP32 serves one MJPEG viewer at a time. If another viewer occupies it,
the console can fall back to processed snapshots. Close other video tabs
before selecting **Use full-rate video**.

## Register a consenting person

1. Keep exactly one face in view of the ESP32, close enough and well lit.
2. Enter the display name in **Register a face**.
3. Confirm explicit consent. To overwrite a name that already exists,
   also confirm **Replace an existing identity with this name**.
4. Select **Capture & register face**. Slowly vary pose while keeping the
   face visible; avoid leaving the camera or adding another person.
5. Wait for the success message and updated identity list. Recognition
   resumes automatically after enrollment.

With the default minimum of three valid images, the console attempts five
fresh captures. In general, it attempts `FACE_MINIMUM_ENROLLMENT_IMAGES + 2`.
Images with zero or multiple faces are skipped by the enrollment pipeline;
too few valid images fail enrollment instead of creating a partial identity.
The bridge pauses during registration to avoid competing inference work.

Temporary photos are deleted after the request, including failed requests.
Successful enrollment stores local `.npz` embeddings in `data/faces/` and
invalidates the recognition cache. No identity is enrolled automatically.
The HTTP response does not expose model tensors or biometric embeddings.

## Local endpoints

| Method and path | Purpose |
| --- | --- |
| `GET /` | Console UI |
| `GET /console/status` | Connection, processing count, latency, latest recognition result |
| `GET /console/frame.jpg` | Most recent camera JPEG, or 503 before the first frame |
| `GET /console/identities` | Display names of locally enrolled identities |
| `POST /console/enroll` | Capture and enroll; requires page token and explicit consent |
| `POST /frames` | Existing device-key-authenticated detection API |

Console routes enforce loopback peer/host checks, same-origin enrollment
requests, and an anti-CSRF page token. Responses disable caching and apply
a Content Security Policy. These are local-demo safeguards, not a
multi-user authentication or authorization system. The enrollment CLI in
the main README remains a separate operator tool.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Waiting for ESP32 / camera unavailable | Keep the hotspot on, join the Mac to it, and verify the camera's latest IP. Test the camera's `/capture` URL. |
| Video works but processed count is stuck | Check Edge logs, model paths, `FACE_RECOGNITION_ENABLED`, and the actual `/frames` response. First model loading can be slower. |
| Blank MJPEG or second stream times out | Close other full-rate viewers or select processed view. |
| Enrollment reports too few valid images | Keep exactly one face visible, improve lighting, move closer, and try again. |
| Name already exists | Use another name, or explicitly confirm replacement. |
| Registration returns 403 after restarting Edge | Reload the page; the anti-CSRF token changes on every launch. |
| Console is inaccessible from a phone | Expected: it binds to the Mac's loopback interface. The ESP32 viewer is a separate LAN URL. |
| Port already in use | Stop the old instance, or choose another `--port` and open that port locally. |

## Verification and privacy checklist

```bash
python -m pytest -q
curl http://127.0.0.1:5050/console/status
```

The implementation was checked with **43 passing automated tests** and
real ESP32 JPEGs flowing through `/frames` with YOLO and ArcFace loaded.
The browser displayed a decoded 640 × 480 image and increasing real
processing counters. A real person's enrollment is an explicit operator
action, not an automated smoke test or a recognition-accuracy claim.

- Keep `.env`, model weights, `.npz` files, SQLite data, captured images,
  and local logs out of Git. Do not share screenshots containing identities.
- Embeddings are sensitive biometric data and are not encrypted at rest.
  Limit local access and define retention/deletion before collecting them.
- No liveness/anti-spoofing, production security, or high-stakes identity
  decisions are provided by this prototype.
