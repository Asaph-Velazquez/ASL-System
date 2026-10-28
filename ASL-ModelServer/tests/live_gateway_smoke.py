"""Opt-in integration check against local Nginx, MongoDB and the real ONNX model."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

import jwt
import numpy as np
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import Predictor, load_sequence  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional CSV/NPY gesture; defaults to the bundled V5 example")
    parser.add_argument("--model-only", action="store_true", help="Skip hotel-backend validation; model authentication is still tested")
    args = parser.parse_args()
    secret = os.environ["JWT_SECRET"]
    mongo_uri = os.environ.get("ASL_TEST_MONGODB_URI", "mongodb://127.0.0.1:27017/asl-hotel")
    base_url = os.environ.get("ASL_TEST_GATEWAY_URL", "http://127.0.0.1:8080").rstrip("/")
    endpoint = base_url + "/api/asl/predict"
    opener = build_opener(ProxyHandler({}))
    predictor = Predictor(ROOT)
    sequence = load_sequence(args.input or ROOT / "example_input.npy", predictor.manifest["feature_columns"])
    if len(sequence) > 160:
        sequence = sequence[np.linspace(0, len(sequence) - 1, 160, dtype=int)]
    assert 15 <= len(sequence) <= 160 and sequence.shape[1] == 63
    frames = sequence.tolist()
    max_length_frames = np.resize(sequence, (160, 63)).tolist()
    expected = predictor.predict_sequence(sequence)
    now = datetime.now(timezone.utc)
    stay_id = "asl-inference-smoke-" + str(uuid4())
    token = jwt.encode({"stayId": stay_id, "exp": now + timedelta(minutes=5)}, secret, algorithm="HS256")

    def post(payload, session=None, url=endpoint):
        headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        if session:
            headers["Authorization"] = f"Bearer {session}"
        request = Request(url, json.dumps(payload).encode(), headers)
        started = time.perf_counter()
        try:
            response = opener.open(request, timeout=15)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read()
            try:
                body = json.loads(raw)
            except ValueError:
                body = {"detail": "Non-JSON response", "bytes": len(raw)}
            return response.status, body, round((time.perf_counter() - started) * 1000, 2)

    results = []
    with MongoClient(mongo_uri, serverSelectionTimeoutMS=2000) as client:
        stays = client.get_default_database().stays
        document_id = stays.insert_one({
            "stayId": stay_id, "roomNumber": "TEST-ASL-INFERENCE", "guestName": "Integration test",
            "active": True, "status": "active", "checkIn": now,
            "checkOut": now + timedelta(minutes=5), "createdAt": now,
        }).inserted_id
        try:
            if not args.model_only:
                status, body, elapsed = post({}, token, base_url + "/api/auth/validate")
                assert status == 200 and body.get("valid") is True, ("guest_validation", status, body)
                results.append({"case": "guest_validation", "status": status, "elapsed_ms": elapsed})
            for name, payload, session, expected_status in [
                ("no_session", {"landmarks": frames}, None, 401),
                ("short_sequence", {"landmarks": frames[:14]}, token, 422),
                ("long_sequence", {"landmarks": [frames[0]] * 161}, token, 422),
                ("invalid_frame", {"landmarks": [frame[:62] for frame in frames]}, token, 422),
                ("real_model_parity", {"landmarks": frames}, token, 200),
                ("max_length_model_parity", {"landmarks": max_length_frames}, token, 200),
                ("oversized_body", {"landmarks": frames, "padding": "x" * (256 * 1024)}, token, 413),
            ]:
                print(f"Checking {name}...", flush=True)
                status, body, elapsed = post(payload, session)
                assert status == expected_status, (name, status, body)
                result = {"case": name, "status": status, "elapsed_ms": elapsed}
                if status == 200:
                    reference = expected if name == "real_model_parity" else predictor.predict_sequence(payload["landmarks"])
                    assert body["glosa"] == reference["glosa"]
                    assert body["index"] == reference["index"]
                    assert 0 <= body["confidence"] <= 1
                    assert abs(body["confidence"] - reference["confidence"]) < 1e-5
                    result.update(body)
                results.append(result)
                print(json.dumps(result), flush=True)
            stays.update_one({"_id": document_id}, {"$set": {"checkOut": now - timedelta(seconds=1)}})
            status, _, elapsed = post({"landmarks": frames}, token)
            assert status == 403, ("expired_stay", status)
            results.append({"case": "expired_stay", "status": status, "elapsed_ms": elapsed})
        finally:
            deleted = stays.delete_one({"_id": document_id, "stayId": stay_id})
            assert deleted.deleted_count == 1, "Temporary test stay was not removed"
    print(json.dumps({"input_source": "gesture_file" if args.input else "bundled_v5_example",
                      "hotel_validation_checked": not args.model_only,
                      "frames": len(frames), "payload_bytes": len(json.dumps({"landmarks": frames}).encode()),
                      "max_length_payload_bytes": len(json.dumps({"landmarks": max_length_frames}).encode()),
                      "results": results, "temporary_stay_removed": True}, indent=2))


if __name__ == "__main__":
    main()
