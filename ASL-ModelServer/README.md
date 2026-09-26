# ASL-ModelServer

FastAPI inference service. The phone sends only 21 hand landmarks (x, y, z) per frame; no video or images leave the device. The model recognizes only the 43 labels in `manifest.json`.

## Local development

Create a Python 3.11 environment, install `requirements.txt`, set `JWT_SECRET` and `MONGODB_URI` to the values used by ASL-Web/server, then run `python -m uvicorn server:app --host 127.0.0.1 --port 8000`. The shared Nginx gateway forwards `/api/asl/*` directly here. Point `EXPO_PUBLIC_API_URL` and `EXPO_PUBLIC_WS_URL` at the gateway, or set only `EXPO_PUBLIC_PUBLIC_BASE_URL`; pointing the API at port 3001 bypasses recognition.

`GET /health` checks model availability. `POST /api/asl/predict` requires `Authorization: Bearer <guest session token>` and `{ "landmarks": [[63 finite numbers], ...] }` with 15-60 frames. The guest stay must be active and not past checkout. Requests are limited to 256 KiB, 30/minute per stay by default, at most two simultaneous inferences, and a ten-second inference deadline. A 429 response means retry later. Do not publish port 8000 directly to the internet; expose only the gateway over HTTPS.

The root `docker-compose.nginx.yml` starts the model service with the gateway and reads the hotel's `.env` for the shared secret. The model container uses `host.docker.internal:27017` for development MongoDB; adapt this address for a production network. Never commit `.env` or expose `JWT_SECRET` to the mobile app.

## Live gateway verification

The test also checks hotel session validation through the gateway and rejects
bodies exceeding 256 KiB. `ASL_TEST_GATEWAY_URL` optionally overrides the local
gateway URL. Only set it to your trusted development gateway: the test sends a
temporary guest token signed with the configured secret to that address.
It prints the valid reference payload size to detect body-limit regressions.

From the repository root, with the gateway running on localhost:8080, configure `JWT_SECRET` in the test process to match the model service, then run:

```powershell
model_onnx/.venv/Scripts/python.exe ASL-ModelServer/tests/live_gateway_smoke.py
```

This opt-in test uses local MongoDB (`mongodb://127.0.0.1:27017/asl-hotel`, override with `ASL_TEST_MONGODB_URI`) and creates exactly one temporary test stay, deleted in `finally`. Use only a development database matching the model service. It checks authentication, 14/61-frame rejection, invalid frame width, expired stay rejection and HTTP/direct ONNX parity. The included reference sequence is resampled from 68 to 60 frames. Tokens and coordinates are not printed; output contains status, prediction, duration and cleanup confirmation. This verifies transport and inference, not recognition accuracy on phone gestures.

For phone testing, ngrok must target **8080**, not 3001. The mobile environment keeps the gateway's public URL. Perform a known model sign with the selected hand, then remove it for at least 0.5 seconds. Check the glosa/confidence under the camera and the `POST /api/asl/predict` response in model/gateway logs. Results below 0.60 appear as candidates rather than appended text. Do not press Send Request during inference-only testing.
