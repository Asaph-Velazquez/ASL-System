# ASL System

An accessible hotel communication platform for guests who use American Sign
Language (ASL) or text. The system combines guest requests, a staff dashboard,
remote interpreter calls, sign recognition, and private gloss-to-English
translation.

## Contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Services and gateway](#services-and-gateway)
- [Gloss translation](#gloss-translation)
- [Verification](#verification)
- [Repository maintenance](#repository-maintenance)

## Architecture

1. Guests authenticate with their stay QR code in the mobile app and choose ASL
   or text communication.
2. For recognition, the phone extracts hand landmarks and sends coordinates to
   `ASL-ModelServer` through the authenticated gateway. Recognition requests
   contain no images or video.
3. The guest reviews the recognized gloss and can edit the request before sending.
4. `ASL-Web` validates the request, optionally translates detected glosses into
   English, stores the result in MongoDB, and broadcasts it to authorized clients.
5. Staff manage requests, stays, transport, and interpreter follow-up in the web
   or desktop dashboard.
6. `ASL-CallAPP` handles interpreter availability, call signaling, and post-call
   reports. Live interpreter calls use camera and microphone media separately
   from the landmark-only recognition flow.

`ASL-IA` is a historical module. The active inference service is
[`ASL-ModelServer`](ASL-ModelServer/README.md).

## Repository layout

| Path | Responsibility | Ownership / production branch |
| --- | --- | --- |
| [`ASL-MobileAPP/`](ASL-MobileAPP/README.md) | Expo / React Native guest app | Git submodule, `master` |
| [`ASL-Web/`](ASL-Web/README.md) | React staff dashboard and hotel backend | Git submodule, `main` |
| [`ASL-CallAPP/`](ASL-CallAPP/README.md) | Interpreter console and call backend | Git submodule, `main` |
| [`ASL-ModelServer/`](ASL-ModelServer/README.md) | FastAPI / ONNX sign inference | Parent repository |
| [`ASL-GlossTranslator/`](ASL-GlossTranslator/README.md) | Private CPU gloss translation | Parent repository |
| [`infra/`](infra/) | Nginx, Android build tooling, and startup checks | Parent repository |
| [`.agents/`](.agents/) | Working conventions | Parent repository |
| [`run.ps1`](run.ps1) / [`run.sh`](run.sh) | Coordinated local startup | Parent repository |

The parent repository uses `main` for production. Submodules are pinned to exact
commits; cloning the parent reproduces those revisions.

## Getting started

### Prerequisites

- Git, Node.js 22, and npm.
- Docker Desktop with Linux containers and Docker Compose for the local database,
  gateway, and inference services.
- Native Android tooling, or macOS with Xcode for iOS. Sign capture and WebRTC
  require an Expo development build; Expo Go and the browser cannot load them.
- ngrok and a configured account for remote access to local services. Local-only
  startup can skip the tunnel.

### Clone and configure

```bash
git clone --recurse-submodules https://github.com/Asaph-Velazquez/ASL-System.git
cd ASL-System
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

Configure the environments in the [Web](ASL-Web/README.md),
[CallAPP](ASL-CallAPP/README.md), and [Mobile](ASL-MobileAPP/README.md) guides.
Replace example secrets. The hotel and model server share `JWT_SECRET`; the two
Node backends share `CALL_INTERNAL_TOKEN` and matching call-token configuration.
Store private values in local environment files.

### Start locally

From the repository root on Windows:

```powershell
.\run.ps1 -SkipNgrok
```

In a Bash environment with the required tools:

```bash
./run.sh --skip-ngrok
```

The launchers coordinate the two Node backends, both Vite interfaces, Expo,
MongoDB, and the Nginx/model stack. Omit the skip-tunnel option to start ngrok.
Use `-SkipMobile` / `--skip-mobile` when Expo is not needed.

PowerShell uses background processes by default; `-ShowWindows` shows their
terminals. Logs are written to the ignored `.dev-logs/` directory. Existing
listeners are reported and reused rather than restarted, so restart a backend
explicitly after changing its environment. Readiness checks establish service
availability, not complete application correctness.

## Services and gateway

| Service | Default address | Exposure |
| --- | --- | --- |
| Staff dashboard | `http://localhost:5173` | Development UI |
| Interpreter console | `http://localhost:5174` | Development UI |
| Hotel HTTP / WebSocket backend | `http://localhost:3001` | Internal upstream |
| Interpreter HTTP / WebSocket backend | `http://localhost:3101` | Internal upstream |
| Nginx gateway | `http://localhost:8080` | Shared entry point |
| Expo Metro | `http://localhost:8081` | Development bundler |
| MongoDB | `mongodb://localhost:27017` | Development database |
| Sign model | Port `8000` on the Docker network | Private service |
| Gloss translator | `http://127.0.0.1:8002` when enabled locally | Loopback only |

The [Nginx configuration](infra/nginx/nginx.conf) routes traffic as follows:

| Route | Destination |
| --- | --- |
| `/api/asl/*` | Private sign model, directly |
| `/calls`, `/api/interpreter/*`, `/api/calls/:callId/report` | Call backend |
| `/api/internal/*` | Blocked from public access |
| Other HTTP routes and `/ws/hotel` | Hotel backend |

For remote development, expose the gateway:

```bash
ngrok http 8080
```

Configure mobile API/WebSocket URLs with that gateway origin. Remote browser
camera and microphone access requires HTTPS. An HTTP `413` during recognition
can indicate a tunnel incorrectly pointed at `3001`: the hotel backend has a
smaller JSON limit than the model route's `256 KiB` limit. Correct the route
before changing payload limits.

Start just the gateway and model with:

```powershell
docker compose -f docker-compose.nginx.yml up -d --build
```

This configuration expects Node backends on the host and uses
`host.docker.internal`. Other Docker/Linux networks may need different upstream
addresses. The model reads the shared secret from `ASL-Web/server/.env`.

The hotel backend's proxy on `3001` remains a compatibility option for calls.
Use `-UseNginxGateway:$false` / `--no-nginx-gateway` for legacy startup; that mode
does not expose sign recognition. `-SkipDocker` / `--skip-docker` requires legacy
mode and externally available dependencies.

## Gloss translation

The FP32 translator requires the separately supplied model package at
`model_onnx/asl_translator_fp32/asl_translator_fp32`. Weights are excluded from Git
and are not downloaded at startup.

```powershell
.\run.ps1 -UseGlossTranslator -SkipNgrok
```

Or start the translator independently:

```powershell
docker compose -f docker-compose.gloss.yml up -d --build --wait --wait-timeout 180
Invoke-RestMethod http://127.0.0.1:8002/health
```

For a host-based hotel backend, set `GLOSS_TRANSLATOR_URL=http://127.0.0.1:8002`
and restart the backend. Natural English is preserved; detected glosses are
translated before persistence and broadcast. Original input remains in metadata.
When enabled, translation failures prevent partial requests from being stored.

The Bash launcher currently has no translation flag. Start the translator
independently and export `GLOSS_TRANSLATOR_URL` before starting Web.
For production, [`docker-compose.gloss-production.yml`](docker-compose.gloss-production.yml)
extends an existing deployment with a private FP32 service. INT8 remains an
evaluation variant because some translations changed meaning. See the
[translator guide](ASL-GlossTranslator/README.md) for resources, limits, and tests.

## Verification

| Component | Checks from its directory |
| --- | --- |
| Web frontend | `npm run lint`, `npm run build` |
| Web backend | `npm test` from `ASL-Web/server` |
| Interpreter frontend | `npm test`, `npm run build` from `ASL-CallAPP/app` |
| Call backend | `npm test` from `ASL-CallAPP/server` |
| Mobile | `npm run lint`, `npm run typecheck`, targeted `test:*` scripts |
| Sign model | `python -m pytest -q -p no:cacheprovider tests` from `ASL-ModelServer` |
| Translator | Contract tests described in its README |
| Startup launcher | `.\infra\tests\run-startup.tests.ps1` from the root |

Integration tests may require disposable local databases and running model
services. Device capture, permissions, and calls across real networks need
physical-device verification. See the [Android guide](infra/android/README.md)
for the containerized APK toolchain.

## Repository maintenance

The root `.gitignore` allows only the supplied top-level layout: the seven
directories above, `.gitignore`, `.gitmodules`, `README.md`, the workspace file,
four Compose files, and two startup scripts. Local tool configuration, release
worktrees, installers, model packages, exports, and temporary scripts remain
outside that layout. Nested rules exclude secrets, dependencies, and generated
artifacts. Each submodule maintains its own ignore rules.

Commit and push submodule changes to their production branches first. Then update
the parent's submodule pointers, commit the parent, and push `ASL-System` to
`main`. Review staged paths explicitly. Ignore rules do not remove already
tracked files.
