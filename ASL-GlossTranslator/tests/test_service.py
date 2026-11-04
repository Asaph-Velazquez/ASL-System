"""Contratos HTTP sin cargar el modelo ni necesitar sus bibliotecas nativas."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service import InputTooLong, create_app
from verify_package import verify_package


def result(text, **options):
    return {"gloss": text, "english": "I need water.", "input_type": "gloss",
            "corrected": text, "preprocessed": "X-I NEED WATER", "changes": [], "noun_fixes": []}


class ServiceTests(unittest.TestCase):
    def test_ready_only_after_model_load(self):
        application = create_app(lambda: result)
        self.assertEqual(TestClient(application).get("/health").status_code, 503)
        with TestClient(application) as client:
            self.assertEqual(client.get("/health").json()["status"], "ok")
        self.assertFalse(application.state.ready)

    def test_full_result_and_options_preserved(self):
        calls = []

        def translate(text, **options):
            calls.append((text, options))
            return result(text)

        with TestClient(create_app(lambda: translate)) as client:
            response = client.post("/translate", json={"text": "I WATER NEED", "source": "recognizer",
                                                        "is_question": True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), result("I WATER NEED"))
        self.assertEqual(calls, [("I WATER NEED", {"source": "recognizer", "is_question": True})])

    def test_invalid_inputs_are_rejected_without_echo(self):
        with TestClient(create_app(lambda: result)) as client:
            for payload in [{}, {"text": " "}, {"text": 123}, {"text": []},
                            {"text": "secret", "source": "unknown"},
                            {"text": "secret", "is_question": "true"},
                            {"text": "secret", "extra": "private"},
                            {"text": "x" * 2001}, {"text": "\x00secret"}]:
                with self.subTest(payload_type=type(payload)):
                    response = client.post("/translate", json=payload)
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json(), {"detail": "invalid_translation_input"})
            response = client.post("/translate", content=b'{"text": "private",',
                                   headers={"Content-Type": "application/json"})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("private", response.text)

    def test_body_limit(self):
        with TestClient(create_app(lambda: result)) as client:
            response = client.post("/translate", content=b" " * 8193,
                                   headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 413)

    def test_token_limit_does_not_generate_partial_translation(self):
        def too_long(text, **options):
            raise InputTooLong("private source")

        with TestClient(create_app(lambda: too_long)) as client:
            response = client.post("/translate", json={"text": "I WATER NEED"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": "input_token_limit"})

    def test_failure_is_redacted_and_next_request_can_run(self):
        calls = []

        def flaky(text, **options):
            calls.append(text)
            if len(calls) == 1:
                raise RuntimeError("private text")
            return result(text)

        with TestClient(create_app(lambda: flaky)) as client:
            response = client.post("/translate", json={"text": "I NEED WATER"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": "translation_failed"})
            self.assertEqual(client.post("/translate", json={"text": "I NEED WATER"}).status_code, 200)

    def test_timeout_keeps_slot_until_native_work_finishes(self):
        started = threading.Event()
        release = threading.Event()

        def slow(text, **options):
            started.set()
            release.wait(timeout=2)
            return result(text)

        with TestClient(create_app(lambda: slow, timeout_seconds=0.03)) as client:
            response = client.post("/translate", json={"text": "I NEED WATER"})
            self.assertTrue(started.is_set())
            self.assertEqual(response.status_code, 504)
            self.assertEqual(response.json(), {"detail": "translation_timeout"})
            response = client.post("/translate", json={"text": "I NEED WATER"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": "translator_busy"})
            self.assertEqual(client.get("/health").status_code, 200)
            release.set()
            for _ in range(50):
                response = client.post("/translate", json={"text": "I NEED WATER"})
                if response.status_code == 200:
                    break
                time.sleep(0.01)
            self.assertEqual(response.status_code, 200)


class IntegrityTests(unittest.TestCase):
    def test_checksum_size_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "asset"
            artifact.write_bytes(b"original")
            entry = {"path": "asset", "bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}
            (root / "manifest.json").write_text(json.dumps({"files": [entry]}), encoding="utf-8")
            self.assertEqual(len(verify_package(root)["files"]), 1)
            artifact.write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                verify_package(root)
            artifact.write_bytes(b"short")
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                verify_package(root)
            artifact.unlink()
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                verify_package(root)


if __name__ == "__main__":
    unittest.main()
