from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference import Predictor, preprocess_sequence


BUNDLE_DIR = Path(__file__).resolve().parents[1]
V4_MANIFEST = {
    "input": {"name": "sequences", "shape": ["batch", 128, 63], "onnx_flattened": False},
    "output": {"name": "dense_2", "kind": "probabilities"},
    "labels": ["Ice", "HELP", "A"],
    "output_class_indices": [2, 0],
    "preprocessing": {"long_sequence": "keep last frames"},
}


@pytest.mark.parametrize("length", [15, 60, 128, 182])
def test_v4_keeps_last_frames_and_pads_without_normalizing(length):
    sequence = np.arange(length * 63, dtype=np.float32).reshape(length, 63) / 1000
    expected = np.zeros((128, 63), dtype=np.float32)
    count = min(length, 128)
    expected[:count] = sequence[-count:]
    np.testing.assert_array_equal(preprocess_sequence(sequence, V4_MANIFEST), expected)


@pytest.mark.parametrize("sequence", [[], np.zeros((15, 62)), np.full((15, 63), np.nan), np.full((15, 63), np.inf)])
def test_v4_rejects_invalid_sequences(sequence):
    with pytest.raises(ValueError):
        preprocess_sequence(sequence, V4_MANIFEST)


@pytest.mark.parametrize("flattened", [False, True])
def test_probabilities_are_not_softmaxed_and_output_indices_are_mapped(flattened):
    predictor = Predictor.__new__(Predictor)
    predictor.manifest = {**V4_MANIFEST, "input": {**V4_MANIFEST["input"], "onnx_flattened": flattened}}

    def run(outputs, inputs):
        assert outputs == ["dense_2"]
        assert inputs["sequences"].shape == ((1, 128 * 63) if flattened else (1, 128, 63))
        return [np.array([[0.1, 0.9]], dtype=np.float32)]

    predictor.session = SimpleNamespace(run=run)
    result = predictor.predict_sequence(np.zeros((15, 63)))
    assert result["index"] == 0
    assert result["glosa"] == "Ice"
    assert result["confidence"] == pytest.approx(0.9)
    assert result["probabilities"] == pytest.approx({"Ice": 0.9, "HELP": 0.0, "A": 0.1})


def test_legacy_bundle_still_resamples_and_normalizes():
    manifest = {
        "input": {"shape": ["batch", 2, 3], "feature_dim": 3, "max_seq_len": 2},
        "preprocessing": {"long_sequence": "uniform_linspace_resample"},
        "normalization": {"mean": [1, 2, 3], "std": [2, 2, 2]},
    }
    sequence = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]], dtype=np.float32)
    np.testing.assert_array_equal(preprocess_sequence(sequence, manifest), [[0, 0, 0], [2, 2, 2]])


@pytest.fixture(scope="module")
def installed_predictor():
    return Predictor(BUNDLE_DIR)


@pytest.mark.parametrize("length", [15, 60, 128, 160, 182, 600])
def test_installed_v5_matches_exporter_contract_with_real_onnx(installed_predictor, length):
    predictor = installed_predictor
    manifest = predictor.manifest
    assert manifest["input"]["shape"] == ["batch", 160, 63]
    assert manifest["input"]["name"] == "landmarks"
    assert manifest["output"]["name"] == "logits"
    assert len(manifest["labels"]) == 139
    assert manifest["labels"][58] == "Ice"
    assert manifest["feature_columns"] == [f"l{i}_{axis}" for i in range(21) for axis in "xyz"]
    sequence = np.random.default_rng(51).uniform(-0.1, 0.9, (length, 63)).astype(np.float32)
    raw = (sequence[np.linspace(0, length - 1, 160, dtype=int)] if length > 160
           else np.pad(sequence, ((0, 160 - length), (0, 0))))
    mean = np.asarray(manifest["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(manifest["normalization"]["std"], dtype=np.float32)
    prepared = (raw - mean) / std
    np.testing.assert_array_equal(preprocess_sequence(sequence, manifest), prepared)
    if length < 160:
        np.testing.assert_array_equal(prepared[length], -mean / std)
        assert np.any(prepared[length] != 0), "Padding must happen before normalization"
    logits = predictor.session.run(["logits"], {"landmarks": prepared[None]})[0][0]
    exp = np.exp(logits - logits.max())
    expected = exp / exp.sum()
    result = predictor.predict_sequence(sequence)
    np.testing.assert_allclose(list(result["probabilities"].values()), expected, rtol=1e-6, atol=1e-7)
    assert result["index"] == int(expected.argmax())
    assert result["confidence"] == pytest.approx(float(expected.max()))
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


def test_installed_example_is_a_valid_unmodified_gesture(installed_predictor):
    sequence = np.load(BUNDLE_DIR / "example_input.npy", allow_pickle=False)
    assert sequence.shape == (30, 63)
    assert np.isfinite(sequence).all()
    result = installed_predictor.predict_sequence(sequence)
    assert result["glosa"] == "1"
    assert result["index"] == 1
    assert result["confidence"] == pytest.approx(0.9054294228553772, abs=1e-6)
