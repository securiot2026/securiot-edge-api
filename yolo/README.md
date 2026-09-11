# Standalone YOLO + ArcFace App

Local desktop application for registering people and recognizing them from a
camera. Run every command in this `yolo` directory.

## Requirements

- Python 3.10–3.12
- A connected camera
- A graphical desktop session
- Face-specific Ultralytics YOLO `.pt` weights

Stock Ultralytics COCO models such as `yolov8n.pt` detect people, not faces.
Provide a compatible face detector at `models/yolov8n-face.pt` or pass its path
with `--model`.

## Install

```bash
cd yolo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p models
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Place the face-specific YOLO weights at:

```text
yolo/models/yolov8n-face.pt
```

InsightFace downloads the ArcFace `buffalo_l` recognition model on its first
run. That first run requires internet access and permission to write to the
user model cache.

## Register a person

```bash
python main.py register --name "Ada Lovelace"
```

Look at the camera and follow the instructions in the window. The application
captures 30 valid photos by default and stores the local registration under
`data/users/`. Press `q` to cancel.

Use another camera or model when necessary:

```bash
python main.py --camera 1 --model /path/to/yolov8n-face.pt register --name "Ada Lovelace"
```

## Start recognition

Register at least one person, then run:

```bash
python main.py recognize
```

The window displays the recognized name and cosine similarity. Unknown faces
are labeled `DESCONOCIDO`. Press `q` to stop.

To select another camera or adjust the identity threshold:

```bash
python main.py --camera 1 recognize --threshold 0.50
```

## Configuration

Environment variables are optional:

| Variable | Default | Purpose |
|---|---:|---|
| `FACE_YOLO_MODEL` | `models/yolov8n-face.pt` | Face detector weights |
| `FACE_DATA_DIR` | `data` | Photos and identity database |
| `FACE_CAMERA_INDEX` | `0` | OpenCV camera index |
| `FACE_DETECTOR_CONFIDENCE` | `0.5` | Minimum YOLO confidence |
| `FACE_DETECTOR_INPUT_SIZE` | `640` | YOLO input size |
| `FACE_SIMILARITY_THRESHOLD` | `0.45` | Known-identity threshold |
| `FACE_PHOTO_COUNT` | `30` | Registration photos |
| `FACE_MINIMUM_EMBEDDINGS` | `10` | Embeddings required to save an identity |
| `FACE_BLUR_THRESHOLD` | `100.0` | Minimum image sharpness |
| `FACE_MINIMUM_FACE_SIZE` | `80` | Minimum face side in pixels |
| `FACE_CAPTURE_COOLDOWN` | `0.35` | Seconds between captures |
| `FACE_CROP_MARGIN` | `0.2` | Margin around each face |
| `FACE_FRAME_STRIDE` | `1` | Process one of every N frames |

Example:

```bash
FACE_FRAME_STRIDE=2 FACE_SIMILARITY_THRESHOLD=0.50 python main.py recognize
```

## Run tests

The automated tests do not open the camera or download model weights:

```bash
python -m unittest discover -s tests -v
```
