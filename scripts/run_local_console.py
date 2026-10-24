"""Run the real Edge API and opt-in registration console on loopback only."""

import argparse
import os
import secrets
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMP_NUM_THREADS", "4")

def main():
    from app import create_app
    from app.config import Config
    from app.local_console import CameraBridge, create_console_blueprint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-url", default="http://172.20.10.2")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()
    camera = urlsplit(args.camera_url)
    if (camera.scheme != "http" or not camera.hostname or camera.username
            or camera.password or camera.query or camera.fragment
            or camera.path not in {"", "/"}):
        parser.error("--camera-url must be an HTTP camera origin without credentials.")
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535.")

    class LocalConfig(Config):
        # Ephemeral key stays server-side; the bridge reuses the normal API.
        DEVICE_SHARED_SECRET = secrets.token_urlsafe(32)

    app = create_app(LocalConfig, start_relay=False)
    bridge = CameraBridge(
        args.camera_url.rstrip("/"), f"http://127.0.0.1:{args.port}",
        app.config["DEVICE_SHARED_SECRET"],
    )
    app.register_blueprint(create_console_blueprint(bridge, app.config))
    app.extensions["camera_bridge"] = bridge
    threading.Thread(target=bridge.run, name="camera-bridge", daemon=True).start()
    print(f"Local console: http://127.0.0.1:{args.port}/", flush=True)
    print("Cloud relay and physical actuator commands are not enabled.", flush=True)
    try:
        app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)
    finally:
        bridge.stopped.set()


if __name__ == "__main__":
    main()
