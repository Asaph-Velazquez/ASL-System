"""Genera un paquete independiente con pesos INT8 y procedencia verificable."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from verify_package import verify_package


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def count_operators(graph):
    counts = Counter(node.op_type for node in graph.node)
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                counts.update(count_operators(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    counts.update(count_operators(child))
    return counts


def quantize_package(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("Destination must be new and outside the original package")
    manifest = verify_package(source)
    shutil.copytree(source, destination)
    summaries = []
    for name in ("encoder_model.onnx", "decoder_model_merged.onnx"):
        original = source / "model" / name
        output = destination / "model" / name
        quantize_dynamic(
            original, output, per_channel=True, weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            extra_options={"EnableSubgraph": True, "MatMulConstBOnly": True},
        )
        graph = onnx.load(str(output))
        onnx.checker.check_model(graph)
        counts = count_operators(graph.graph)
        if not counts["MatMulInteger"]:
            raise RuntimeError(f"No quantized matrix operators in {name}")
        summaries.append({"path": f"model/{name}", "source_bytes": original.stat().st_size,
                          "bytes": output.stat().st_size, "operators": dict(counts)})
        del graph
    manifest["variant"] = "int8-dynamic"
    manifest["model"]["quantization"] = "dynamic QInt8 per-channel MatMul/Gemm"
    manifest["provenance"] = {
        "source_manifest_sha256": digest(source / "manifest.json"),
        "source_variant": "fp32", "onnxruntime": "1.22.0",
        "options": {"EnableSubgraph": True, "MatMulConstBOnly": True,
                    "per_channel": True, "weight_type": "QInt8"},
        "models": summaries,
    }
    for entry in manifest["files"]:
        path = destination / entry["path"]
        entry.update(bytes=path.stat().st_size, sha256=digest(path))
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verify_package(destination)
    # Verifica también la fuente para detectar cualquier alteración accidental.
    verify_package(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    arguments = parser.parse_args()
    quantize_package(arguments.source, arguments.destination)
