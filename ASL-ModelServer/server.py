"""Authenticated, bounded landmark inference; no image or video endpoint."""

import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Annotated

import jwt
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from inference import Predictor


MAX_BODY_BYTES = 256 * 1024
Frame = Annotated[list[FiniteFloat], Field(min_length=63, max_length=63)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    landmarks: Annotated[list[Frame], Field(min_length=15, max_length=60)]


class SlidingRateLimiter:
    def __init__(self, limit):
        self.limit = limit
        self.hits = defaultdict(deque)
        self.lock = Lock()

    def allow(self, key):
        now = time.monotonic()
        with self.lock:
            hits = self.hits[key]
            while hits and hits[0] <= now - 60:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


def active_stay(stays, stay_id):
    stay = stays.find_one({"stayId": stay_id}, {"active": 1, "status": 1, "checkOut": 1})
    if not stay or not stay.get("active") or stay.get("status") != "active":
        return False
    checkout = stay.get("checkOut")
    if not isinstance(checkout, datetime):
        return False
    if checkout.tzinfo is None:
        checkout = checkout.replace(tzinfo=timezone.utc)
    return checkout > datetime.now(timezone.utc)


def create_app(bundle_dir=None, *, predictor=None, stays=None, secret=None):
    bundle_dir = Path(bundle_dir or Path(__file__).resolve().parent)
    secret = secret or os.environ.get("JWT_SECRET")
    max_inflight = max(1, int(os.environ.get("ASL_MAX_INFLIGHT", "2")))
    limiter = SlidingRateLimiter(max(1, int(os.environ.get("ASL_RATE_LIMIT_PER_MINUTE", "30"))))

    @asynccontextmanager
    async def lifespan(app):
        if not secret or secret == "change-this-to-a-secure-64-character-hex-string":
            raise RuntimeError("JWT_SECRET must match the hotel server and be configured securely")
        client = None
        if stays is None:
            mongo_uri = os.environ.get("MONGODB_URI")
            if not mongo_uri:
                raise RuntimeError("MONGODB_URI is required")
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            client.admin.command("ping")
            app.state.stays = client.get_default_database().stays
        else:
            app.state.stays = stays
        app.state.predictor = predictor or Predictor(bundle_dir)
        app.state.slots = BoundedSemaphore(max_inflight)
        app.state.executor = ThreadPoolExecutor(max_workers=max_inflight)
        try:
            yield
        finally:
            app.state.executor.shutdown(wait=False, cancel_futures=True)
            if client:
                client.close()

    app = FastAPI(title="ASL Model Server", lifespan=lifespan)

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if request.url.path == "/api/asl/predict":
            try:
                declared_size = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
            if declared_size > MAX_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Payload too large"})
            body = await request.body()
            if len(body) > MAX_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Payload too large"})
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        return JSONResponse(status_code=422, content={"detail": [
            {key: error[key] for key in ("loc", "msg", "type")}
            for error in exc.errors()
        ]})

    @app.get("/health")
    def health():
        return {"status": "ok", "labels": len(app.state.predictor.manifest["labels"])}

    @app.post("/api/asl/predict")
    def predict(body: PredictionRequest, authorization: Annotated[str | None, Header()] = None):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Guest session required")
        try:
            claims = jwt.decode(authorization[7:], secret, algorithms=["HS256"], options={"require": ["exp"]})
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=401, detail="Invalid or expired session") from exc
        stay_id = claims.get("stayId")
        if not isinstance(stay_id, str) or not stay_id or claims.get("role") in ("staff", "admin"):
            raise HTTPException(status_code=403, detail="Guest session required")
        try:
            if not active_stay(app.state.stays, stay_id):
                raise HTTPException(status_code=403, detail="Active stay required")
        except PyMongoError as exc:
            raise HTTPException(status_code=503, detail="Stay verification unavailable") from exc
        if not limiter.allow(stay_id):
            raise HTTPException(status_code=429, detail="Inference rate limit reached")
        if not app.state.slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Inference busy")

        future = app.state.executor.submit(app.state.predictor.predict_sequence, body.landmarks)
        future.add_done_callback(lambda _future: app.state.slots.release())
        try:
            result = future.result(timeout=10)
        except FutureTimeout as exc:
            raise HTTPException(status_code=504, detail="Inference timed out") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"glosa": result["glosa"], "confidence": result["confidence"], "index": result["index"]}

    return app


app = create_app()
