# ASL-ModelServer

FastAPI inference service. The phone sends only 21 hand landmarks (x, y, z) per frame; no video or images leave the device. The V5 Transformer from `model_onnx/V5_export_onnx` classifies the 139 labels in `manifest.json`, including `Ice` at index 58.

## Updating the model

Deploy `model.onnx` together with its matching `manifest.json` and `example_input.npy`. Replacing only the ONNX is insufficient. V5 uses input `landmarks` of shape `float32 [batch, 160, 63]`. Sample long gestures uniformly with `numpy.linspace(0, T-1, 160, dtype=int)`; append zero frames to short gestures **before** applying the new manifest's mean/std normalization. Output `logits` requires one softmax. The service verifies the model SHA-256 and retains one ONNX session per process. The installed V5 hash is `df3b1ee7265505b67811fa944e204c967ea23e2335c373b24fc0696bff089836`.

`inference.py` also supports V4's last-frame cropping, unnormalized padding and probability/index mapping for rollback. These rules are selected by the manifest; do not apply them to V5. Reverting the model also requires aligning the server's accepted sequence length and the mobile collector.

The Docker image embeds these files, so rebuild and recreate the service after an update. From the repository root, with Docker and the development database running:

```powershell
docker compose -f .\docker-compose.nginx.yml up -d --build asl-model-server
docker compose -f .\docker-compose.nginx.yml exec asl-model-server python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
docker compose -f .\docker-compose.nginx.yml exec asl-model-server python -c "from inference import Predictor; p=Predictor('.'); print(p.manifest['sha256']); print(p.manifest['input']); print(p.manifest['output'])"
```

Health should return `{"status":"ok","labels":139}`. V4 has the same label count, so also verify the hash and the `landmarks [batch,160,63]` / `logits` contract with the second command. If the gateway is already running, reload its upstream resolution after recreating the model container:

```powershell
docker compose -f .\docker-compose.nginx.yml exec nginx-gateway nginx -s reload
```

The gateway's `/health` route belongs to the hotel backend; check model health inside the model container as shown above.

## Local development

Create a Python 3.11 environment, install `requirements.txt`, set `JWT_SECRET` and `MONGODB_URI` to the values used by ASL-Web/server, then run `python -m uvicorn server:app --host 127.0.0.1 --port 8000`. The shared Nginx gateway forwards `/api/asl/*` directly here. Point `EXPO_PUBLIC_API_URL` and `EXPO_PUBLIC_WS_URL` at the gateway, or set only `EXPO_PUBLIC_PUBLIC_BASE_URL`; pointing the API at port 3001 bypasses recognition.

`GET /health` checks model availability. `POST /api/asl/predict` requires `Authorization: Bearer <guest session token>` and `{ "landmarks": [[63 finite numbers], ...] }` with 15-160 frames. Older 60/128-frame clients remain accepted; the server pads and normalizes them. The guest stay must be active and not past checkout. Requests are limited to 256 KiB, 30/minute per stay by default, at most two simultaneous inferences, and a ten-second inference deadline. A 429 response means retry later. Do not publish port 8000 directly to the internet; expose only the gateway over HTTPS.

The root `docker-compose.nginx.yml` starts the model service with the gateway and reads the hotel's `.env` for the shared secret. The model container uses `host.docker.internal:27017` for development MongoDB; adapt this address for a production network. Never commit `.env` or expose `JWT_SECRET` to the mobile app.

## Mobile hand mapping

Both native camera implementations emit the selected hand's image landmarks in the exact manifest order `l0_x,l0_y,l0_z,...,l20_z`. They send neither world landmarks nor both hands concatenated, and do not center, scale, normalize or flip the coordinates. The native handedness category is used directly for the left/right selector. V5 has the same feature columns and labels as V4, so no native coordinate remapping is required. Keep each selected hand as its own sequence.

The mobile collector waits until the selected hand is absent for 500 ms, preserving the complete gesture. Gestures over 160 detected frames are sampled uniformly to 160, preserving the beginning and end; shorter gestures need at least 15 frames and reach the server without padding or normalization. The server performs those steps. Capture is bounded at 600 detected frames: reaching that cap submits one sampled sequence and waits for hand exit before recording again, including while inference is busy. The frame indicator saturates at `/160` and prompts removal of the hand. Changes are in TypeScript; reload the updated JavaScript bundle or build a new release to distribute them.

Update the server at the mobile app's selected gateway before distributing the 160-frame client. This mobile development branch selects the local gateway using `EXPO_PUBLIC_LOCAL_GATEWAY_URL`, the private Metro host, or localhost; see `ASL-MobileAPP/LOCAL_DEVELOPMENT.md` for device networking. Updating local Docker does not update a remote backend. Earlier servers limited to 60 or 128 frames reject longer requests.

V5's export does not specify training FPS, camera mirroring, image orientation or handedness coverage. The code-level format is compatible, but verify real left/right gestures on the target phone before claiming recognition accuracy. The manifest reports 1,659 validation samples, ONNX accuracy about 97.53% and top-1 conversion parity 100%; these are exporter results, not a mobile evaluation performed here. V5 does not include `Water` or `TV`; available words and spelling follow this manifest exactly.

## Live gateway verification

The test also checks hotel session validation through the gateway and rejects
bodies exceeding 256 KiB. `ASL_TEST_GATEWAY_URL` optionally overrides the local
gateway URL. Only set it to your trusted development gateway: the test sends a
temporary guest token signed with the configured secret to that address.
It checks HTTP/direct inference parity for the reference gesture and a 160-frame payload derived by repeating that gesture, and prints both payload sizes to detect body-limit regressions.

From the repository root, with the gateway running on localhost:8080, configure `JWT_SECRET` in the test process to match the model service, then run:

```powershell
model_onnx/V1_export_onnx/.venv/Scripts/python.exe ASL-ModelServer/tests/live_gateway_smoke.py
```

This opt-in test uses local MongoDB (`mongodb://127.0.0.1:27017/asl-hotel`, override with `ASL_TEST_MONGODB_URI`) and creates exactly one temporary test stay, deleted in `finally`. Use only a development database matching the model service. It checks authentication, 14/161-frame rejection, invalid frame width, expired stay rejection and HTTP/direct ONNX parity. By default it uses V5's bundled 30-frame `example_input.npy` and marks the result `bundled_v5_example`. The exporter and integrated predictor both return `1` (index 1, confidence approximately 0.9054294) for this example. Pass `--input path/to/gesture.npy` (or `.csv`) for another sequence; files over 160 frames are sampled uniformly. Tokens and coordinates are not printed; output contains status, prediction, duration and cleanup confirmation. This verifies transport and inference, not recognition accuracy on phone gestures.

Run the local suite from `ASL-ModelServer` with a Python environment containing `requirements-test.txt`: `python -m pytest -q -p no:cacheprovider tests`. It covers V5 resampling, padding before normalization, logits/softmax and authenticated inference with the installed ONNX, plus regression coverage for V4's probability/index mapping.

If the hotel backend is stopped, pass `--model-only` to skip `/api/auth/validate` while still checking the model route's JWT, active stay and expired stay validation through Nginx. The summary records `hotel_validation_checked: false` in that case.

For phone testing, ngrok must target **8080**, not 3001. The mobile environment keeps the gateway's public URL. Perform a known model sign with the selected hand, then remove it for at least 0.5 seconds. Check the glosa/confidence under the camera and the `POST /api/asl/predict` response in model/gateway logs. Results below 0.60 appear as candidates rather than appended text. Do not press Send Request during inference-only testing.
