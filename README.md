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

- `EDGE_DB_PATH` — path to the local SQLite buffer file (default `edge.db`)
- `DEVICE_SHARED_SECRET` — shared secret devices must send in `X-Device-Key`
- `CLOUD_API_URL` — base URL of the Cloud API this Edge instance relays to
- `CLOUD_DEVICE_API_KEY` — key the Edge API sends to the Cloud API
- `RELAY_INTERVAL_SECONDS` — how often the relay checks for unsynced readings

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

## Device simulator

`simulator/device_simulator.py` stands in for the physical ESP32 (built in
Phase 4). It posts synthetic door-contact readings to this API's `/ingest`
endpoint on a timer - see its own `simulator/requirements.txt` for its one
extra dependency (`requests`, already covered above).

For the full local demo (Cloud API + this Edge API + the simulator running
together, including how to reproduce the connectivity-interruption test),
see `docs/demo/phase-1-local-run.md` in the main planning repo.
