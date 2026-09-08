from app import create_app

app = create_app()

if __name__ == "__main__":
    # 0.0.0.0, not the 127.0.0.1 default: the ESP32-CAM reaches this over
    # the LAN, not from the same machine.
    app.run(host="0.0.0.0")
