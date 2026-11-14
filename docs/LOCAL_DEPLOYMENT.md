# Running the Edge API on a laptop, with a public tunnel

This is how the Edge API runs today for the demo: on a Mac on the same WiFi as
the ESP32-CAM, started by launchd, and reachable from the Cloud API through a
Cloudflare named tunnel. The pieces are in `deploy/macos/`.

```
ESP32-CAM --LAN--> Edge API (laptop :5001) <--Cloudflare tunnel-- Cloud API (VPS) <-- Web App
```

Two directions matter:

- The board posts frames straight to the Edge over the LAN (`http://<laptop-ip>:5001`).
- The Cloud API calls the Edge for the live feed, face enrollment and manual
  commands. The VPS cannot reach a private IP, so it uses the tunnel hostname,
  saved as the device `edgeUrl` in the Web App.

## 1. Python environment and models

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt onnxruntime
mkdir -p models
```

The model files are not in git (`models/` is ignored). Download them yourself:

| File | Source | Notes |
| --- | --- | --- |
| `models/yolov8n.pt` | `github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt` | Person and object detection |
| `models/yolov8n-face.pt` | `github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt` | Face detector |
| `models/w600k_r50.onnx` | Inside `buffalo_l.zip` at `github.com/deepinsight/insightface/releases/download/v0.7/` | ArcFace. InsightFace weights are for non commercial research only |

## 2. Configuration outside the repo

Copy `deploy/macos/edge.env.example` to `~/.securiot/edge.env`, run
`chmod 600` on it and fill it in. It holds the Cloud API URL, the device API key
and the admin key. Keep real values out of git, chat and pull requests.

- `CLOUD_API_URL` and `CLOUD_DEVICE_API_KEY`: without them heartbeats go to
  `localhost:5000`, which is AirPlay on macOS and answers 403. The Web App then
  shows the device as offline. Get a fresh key with the authenticated request
  `POST /api/v1/devices/<id>/rotate-key` (the old key stops working).
- `EDGE_ADMIN_API_KEY`: must equal the Cloud API variable of the same name.
  Face enrollment, the face toggle and the live feed need it.

## 3. Start it at login with launchd

```bash
mkdir -p ~/.securiot
cp deploy/macos/run-edge.sh deploy/macos/run-tunnel.sh ~/.securiot/ && chmod 700 ~/.securiot/run-*.sh
cp deploy/macos/com.securiot.edge.plist.example ~/Library/LaunchAgents/com.securiot.edge.plist
# edit /Users/YOU in the plist, then:
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.securiot.edge.plist
launchctl kickstart -k gui/$(id -u)/com.securiot.edge    # restart after editing edge.env
tail -f ~/.securiot/edge.log
```

## 4. Cloudflare tunnel

1. In Cloudflare Zero Trust create a remotely managed tunnel (for example
   `securiot-edge`) with a public hostname such as `edge-local.<your-domain>`
   pointing to `http://localhost:5001`.
2. Save the tunnel token to `~/.securiot/tunnel.token` (`chmod 600`).
3. Install `deploy/macos/com.securiot.tunnel.plist.example` the same way as the
   Edge agent.
4. Set the device `edgeUrl` in the Web App to `https://edge-local.<your-domain>`.

If the zone has Bot Fight Mode, Cloudflare challenges the VPS and the Cloud API
gets a 403 with an empty body. Allow the VPS IP in IP Access Rules and add a
custom Skip rule for the tunnel hostname. The Edge still checks its own keys:
`X-Admin-Key` for the live feed and enrollment, `X-Device-Key` for frames.

## 5. Face enrollment and tracking

- Register a person from the Web App (at least 3 photos, with consent). The log
  then shows `faces count=N known=M` per frame.
- The camera follows the highest confidence `person` detection. The Edge returns
  `pan_delta` and `tilt_delta` in `[-1, 1]` and the firmware turns them into
  servo moves. If the camera moves away from you, flip the sign of the matching
  axis in the firmware servo code.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Web App says offline, log has `heartbeat: cloud API returned 403` | `CLOUD_API_URL` not set, or the device key was rotated |
| Live feed 403 with `{}` | Cloudflare bot protection challenging the VPS (section 4) |
| Live feed 502 "check EDGE_ADMIN_API_KEY" | Edge and Cloud API admin keys differ |
| No frames in the log | The board holds an old Edge IP. Re-provision it with the Setup QR |
| Slow frames | YOLO, face detection and ArcFace run per frame on the CPU |

## GitFlow and CI

Branch from `develop` (`feat/`, `fix/`, `docs/`, `chore/`), open a pull request
into `develop`, wait for the `test` check (`pytest` with the mock detector),
merge, and release with a pull request from `develop` into `main`.
`.github/workflows/ci.yml` needs no secrets.
