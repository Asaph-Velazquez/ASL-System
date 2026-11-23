# ASL Model Server

Authenticated FastAPI / ONNX inference for ASL System. The phone sends the
selected hand's 21 image landmarks per frame; recognition requests contain no
images or video. The installed V5 Transformer classifies 139 labels.

This service is versioned in the parent ASL-System repository. This guide
describes production `main`, whose single-hand contract differs from experimental
two-hand model branches.

## Model contract

| Property | Production value |
| --- | --- |
| Bundle | `model.onnx`, `manifest.json`, `example_input.npy` |
| Input | `landmarks`, `float32 [batch, 160, 63]` |
| Output | `logits`; apply one softmax |
| Labels | 139, ordered exactly as in the manifest |
| Model SHA-256 | `df3b1ee7265505b67811fa944e204c967ea23e2335c373b24fc0696bff089836` |

The predictor samples long sequences uniformly to 160 frames and appends zero
frames to shorter sequences **before** mean/std normalization. The manifest
selects preprocessing. V4 rollback support has different cropping, padding, and
probability rules; do not apply those rules to V5.

Labels preserve the manifest's spelling and index, including `Ice` at index 58.
The model does not include `Water` or `TV`. Replacing an ONNX file requires its
matching manifest and reference input. Startup verifies the model hash and keeps
one ONNX session per process.

## HTTP API

| Endpoint | Contract |
| --- | --- |
| `GET /health` | `{"status":"ok","labels":139}` when the model is ready |
| `POST /api/asl/predict` | Guest-authenticated landmark inference |

Prediction requires `Authorization: Bearer <guest-session-token>` and a JSON
object with `landmarks`: 15–160 frames, each containing exactly 63 finite numbers
in `l0_x,l0_y,l0_z,...,l20_z` order. The response includes `glosa`, `confidence`,
and `index`. The guest stay must be active and before checkout.

| Limit | Default |
| --- | --- |
| Request body | `256 KiB` |
| Requests per stay | 30 per minute |
| Concurrent inferences | 2 |
| Inference deadline | 10 seconds |

A `429` response indicates throttling. Keep port `8000` private; public clients
must use the HTTPS gateway's `/api/asl/*` route.

## Start with Docker

From the ASL-System root, with Docker Desktop and development MongoDB available:

```powershell
docker compose -f docker-compose.nginx.yml up -d --build asl-model-server
docker compose -f docker-compose.nginx.yml exec -T asl-model-server python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
docker compose -f docker-compose.nginx.yml exec -T asl-model-server python -c "from inference import Predictor; p=Predictor('.'); print(p.manifest['sha256']); print(p.manifest['input']); print(p.manifest['output'])"
```

If the gateway is not running, start the complete Compose file. After recreating
an upstream, validate and reload an existing gateway:

```powershell
docker compose -f docker-compose.nginx.yml exec -T nginx-gateway nginx -t
docker compose -f docker-compose.nginx.yml exec -T nginx-gateway nginx -s reload
```

The gateway's general `/health` belongs to the hotel backend; check model health
inside its container. Label count alone does not distinguish model revisions, so
also verify the hash and input/output contract.

Compose reads `JWT_SECRET` from `ASL-Web/server/.env` and uses
`mongodb://host.docker.internal:27017/asl-hotel` for development. Adapt the database
address to the deployment network. Never expose the shared secret to the phone.
The root PowerShell/Bash launchers start this stack by default.

## Run without Docker

From `ASL-ModelServer`, create and activate a Python 3.11 virtual environment:

```bash
python -m pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Before starting, set `JWT_SECRET` and `MONGODB_URI` to the hotel's values in the
process environment. The standalone command does not load the example file
automatically. Ensure the gateway can reach this host process if replacing the
Compose-managed model.

## Mobile compatibility

Production Mobile `master` sends one selected hand, with 15–60 captured frames.
The server accepts those sequences and pads them to its 160-frame input. Native
coordinates remain in image space; do not center, normalize, flip, or concatenate
hands on the client. Match client and server contracts before deployment.

Use the gateway origin for the mobile API and WebSocket configuration. Publishing
hotel port `3001` bypasses recognition; ngrok must target `8080`. Recognition
results below `0.60` remain candidates in the mobile draft. The guest decides
whether to submit a hotel request.

The exporter does not establish training FPS, phone mirroring, orientation, or
handedness coverage. Its validation metrics do not establish real-camera accuracy.
Verify representative gestures with each supported hand on the target device.

## Verification

From this service directory:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q -p no:cacheprovider tests
```

From the ASL-System root, run direct inference on the bundled reference:

```bash
python ASL-ModelServer/inference.py --model ASL-ModelServer --input ASL-ModelServer/example_input.npy
```

The opt-in gateway check requires local MongoDB, the running gateway/model, and a
process `JWT_SECRET` matching the service:

```bash
python ASL-ModelServer/tests/live_gateway_smoke.py
```

It creates one temporary development stay and deletes it in `finally`. It checks
authentication, active/expired stays, 14/161-frame rejection, invalid frame width,
oversized bodies, and direct/HTTP ONNX parity. Use `--model-only` to skip the hotel
backend's `/api/auth/validate` check while retaining model authorization checks.
`--input` accepts another `.npy` or `.csv` sequence.

`ASL_TEST_MONGODB_URI` and `ASL_TEST_GATEWAY_URL` can select trusted development
services. The check sends a signed temporary token to the selected gateway;
use a disposable development environment. It verifies transport and inference
parity, not accuracy on signs captured by a phone.

For updates or rollback, deploy a complete matching bundle, rebuild the image,
reload Nginx, and verify hash, preprocessing, authorization, and mobile compatibility.
