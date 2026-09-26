from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient
from pymongo.errors import ServerSelectionTimeoutError

from server import create_app
from inference import Predictor


SECRET = "test-secret-not-for-production"
FRAMES = [[0.0] * 63 for _ in range(15)]


class FakeStays:
    def __init__(self, stay=None):
        self.stay = stay

    def find_one(self, query, projection):
        return self.stay if query["stayId"] == "stay-1" else None


class FakePredictor:
    manifest = {"labels": ["WATER"]}

    def predict_sequence(self, frames):
        assert len(frames) == 15
        return {"glosa": "WATER", "confidence": 0.9, "index": 0}


def token(**overrides):
    claims = {"stayId": "stay-1", "exp": datetime.now(timezone.utc) + timedelta(hours=1)}
    claims.update(overrides)
    return jwt.encode(claims, SECRET, algorithm="HS256")


def stay(**overrides):
    value = {"active": True, "status": "active", "checkOut": datetime.now(timezone.utc) + timedelta(hours=1)}
    value.update(overrides)
    return value


def request(client, session=None, body=None):
    return client.post("/api/asl/predict", json=body or {"landmarks": FRAMES}, headers={"Authorization": f"Bearer {session or token()}"})


def test_guest_inference_and_health():
    with TestClient(create_app(predictor=FakePredictor(), stays=FakeStays(stay()), secret=SECRET)) as client:
        assert client.get("/health").json() == {"status": "ok", "labels": 1}
        assert request(client).json() == {"glosa": "WATER", "confidence": 0.9, "index": 0}


def test_real_model_bundle_smoke():
    predictor = Predictor(".")
    with TestClient(create_app(predictor=predictor, stays=FakeStays(stay()), secret=SECRET)) as client:
        result = request(client)
        assert result.status_code == 200
        assert result.json()["glosa"] in predictor.manifest["labels"]
        assert 0 <= result.json()["confidence"] <= 1


def test_guest_auth_and_stay():
    with TestClient(create_app(predictor=FakePredictor(), stays=FakeStays(stay()), secret=SECRET)) as client:
        assert client.post("/api/asl/predict", json={"landmarks": FRAMES}).status_code == 401
        assert request(client, token(role="staff")).status_code == 403
        assert request(client, token(exp=datetime.now(timezone.utc) - timedelta(seconds=1))).status_code == 401
    for invalid in [None, stay(active=False), stay(status="ended"), stay(checkOut=datetime.now(timezone.utc) - timedelta(seconds=1))]:
        with TestClient(create_app(predictor=FakePredictor(), stays=FakeStays(invalid), secret=SECRET)) as client:
            assert request(client).status_code == 403


def test_shape_size_and_rate_limit(monkeypatch):
    monkeypatch.setenv("ASL_RATE_LIMIT_PER_MINUTE", "1")
    with TestClient(create_app(predictor=FakePredictor(), stays=FakeStays(stay()), secret=SECRET)) as client:
        assert request(client, body={"landmarks": FRAMES[:1]}).status_code == 422
        invalid_json = '{"landmarks":[' + ','.join(['[' + ','.join(['NaN'] * 63) + ']'] * 15) + ']}'
        assert client.post("/api/asl/predict", content=invalid_json,
                           headers={"Content-Type": "application/json"}).status_code == 422
        assert client.post("/api/asl/predict", content=b"x" * (256 * 1024 + 1)).status_code == 413
        assert request(client).status_code == 200
        assert request(client).status_code == 429


def test_stay_database_unavailable_is_not_treated_as_authorized():
    class BrokenStays:
        def find_one(self, query, projection):
            raise ServerSelectionTimeoutError("mongo unavailable")

    with TestClient(create_app(predictor=FakePredictor(), stays=BrokenStays(), secret=SECRET)) as client:
        assert request(client).status_code == 503


def test_missing_model_fails_startup():
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(bundle_dir=Path(__file__).parent / "no-model", stays=FakeStays(stay()), secret=SECRET)):
            pass
