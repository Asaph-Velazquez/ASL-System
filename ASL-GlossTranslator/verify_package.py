"""Verificación del paquete original antes de cargar pesos o ejecutar sus scripts."""

import hashlib
import json
from pathlib import Path


def verify_package(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise RuntimeError("Model package integrity check failed")
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError("Model package integrity check failed")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise RuntimeError("Model package integrity check failed")
    return manifest
