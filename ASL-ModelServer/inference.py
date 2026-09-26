"""CLI y utilidades de inferencia para el bundle ONNX."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load_manifest(bundle_dir):
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    model_path = bundle_dir / manifest.get("model_file", "model.onnx")
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontro el modelo: {model_path}")

    expected_hash = manifest.get("sha256")
    if expected_hash:
        actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("El checksum SHA-256 del modelo no coincide.")
    return manifest, model_path


def load_sequence(input_path, feature_columns):
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".npy":
        return np.load(input_path, allow_pickle=False)
    if input_path.suffix.lower() != ".csv":
        raise ValueError("La entrada debe ser un archivo .csv o .npy.")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("El CSV no contiene encabezado.")
        missing = [name for name in feature_columns if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Faltan columnas de features: {missing}")
        rows = [[row[name] for name in feature_columns] for row in reader]

    if not rows:
        raise ValueError("La secuencia esta vacia.")
    try:
        return np.asarray(rows, dtype=np.float32)
    except ValueError as exc:
        raise ValueError("El CSV contiene features no numericas o vacias.") from exc


def preprocess_sequence(sequence, manifest):
    sequence = np.asarray(sequence, dtype=np.float32)
    feature_dim = int(manifest["input"]["feature_dim"])
    max_seq_len = int(manifest["input"]["max_seq_len"])

    if sequence.ndim != 2 or sequence.shape[1] != feature_dim:
        raise ValueError(
            f"Se esperaba una matriz [T,{feature_dim}], se recibio {sequence.shape}."
        )
    if sequence.shape[0] == 0:
        raise ValueError("La secuencia esta vacia.")
    if not np.isfinite(sequence).all():
        raise ValueError("La secuencia contiene NaN o valores infinitos.")

    if sequence.shape[0] > max_seq_len:
        indices = np.linspace(
            0, sequence.shape[0] - 1, max_seq_len, dtype=int
        )
        sequence = sequence[indices]
    elif sequence.shape[0] < max_seq_len:
        padding = np.zeros(
            (max_seq_len - sequence.shape[0], feature_dim), dtype=np.float32
        )
        sequence = np.vstack((sequence, padding))

    mean = np.asarray(manifest["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(manifest["normalization"]["std"], dtype=np.float32)
    if mean.shape != (feature_dim,) or std.shape != (feature_dim,):
        raise ValueError("Las estadisticas del manifiesto no son validas.")
    if np.any(std <= 0):
        raise ValueError("La desviacion estandar del manifiesto no es valida.")
    return ((sequence - mean) / std).astype(np.float32, copy=False)


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class Predictor:
    """Carga y verifica el modelo una sola vez por proceso."""

    def __init__(self, bundle_dir):
        import onnxruntime as ort

        self.manifest, model_path = load_manifest(bundle_dir)
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def predict_sequence(self, sequence):
        prepared = preprocess_sequence(sequence, self.manifest)[None, ...]
        logits = self.session.run(
            [self.manifest["output"]["name"]],
            {self.manifest["input"]["name"]: prepared},
        )[0]
        probabilities = softmax(logits)[0]
        index = int(np.argmax(probabilities))
        labels = self.manifest["labels"]
        return {
            "index": index,
            "glosa": labels[index],
            "confidence": float(probabilities[index]),
            "probabilities": {
                label: float(probabilities[i]) for i, label in enumerate(labels)
            },
        }


def predict(bundle_dir, input_path):
    predictor = Predictor(bundle_dir)
    sequence = load_sequence(input_path, predictor.manifest["feature_columns"])
    return predictor.predict_sequence(sequence)


def main():
    parser = argparse.ArgumentParser(description="Inferencia del Transformer ONNX")
    parser.add_argument("--model", required=True, help="Directorio del bundle")
    parser.add_argument("--input", required=True, help="Secuencia .csv o .npy")
    args = parser.parse_args()
    print(json.dumps(predict(args.model, args.input), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
