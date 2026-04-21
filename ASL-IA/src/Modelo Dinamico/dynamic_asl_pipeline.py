import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

try:
    from mediapipe import solutions as mp_solutions
except Exception:
    from mediapipe.python import solutions as mp_solutions


DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_VALID_VIDEOS = 5
DEFAULT_TARGET_FRAMES = 32
DEFAULT_MAX_HANDS = 2
DEFAULT_HAND_LANDMARKS = 21
DEFAULT_FEATURE_SIZE = DEFAULT_MAX_HANDS * DEFAULT_HAND_LANDMARKS * 3


@dataclass
class SplitResult:
    train: List[str]
    val: List[str]
    test: List[str]
    excluded_classes: Dict[str, str]


class DynamicASLPipeline:
    def __init__(self, project_root: Optional[Path] = None):
        self.module_dir = Path(__file__).resolve().parent
        self.project_root = project_root or self._resolve_project_root()
        self.data_dir = self.project_root / "data"
        self.dynamic_dir = self.data_dir / "Dinamico"
        self.artifacts_dir = self.module_dir / "artifacts"
        self.models_dir = self.module_dir / "models"
        self.sequences_dir = self.artifacts_dir / "dynamic_sequences"

    def audit_dataset(
        self,
        output_csv: Optional[Path] = None,
        output_json: Optional[Path] = None,
        tiny_file_threshold_bytes: int = 1024,
        min_frames: int = 2,
        overwrite: bool = True,
    ) -> pd.DataFrame:
        output_csv = output_csv or self.artifacts_dir / "dynamic_audit.csv"
        output_json = output_json or self.artifacts_dir / "dynamic_audit_summary.json"

        records: List[Dict[str, object]] = []

        for class_dir in sorted(self.dynamic_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for video_path in sorted(class_dir.glob("*.mp4")):
                record = self._inspect_video(
                    video_path=video_path,
                    class_name=class_dir.name,
                    tiny_file_threshold_bytes=tiny_file_threshold_bytes,
                    min_frames=min_frames,
                )
                records.append(record)

        audit_df = pd.DataFrame(records)
        if audit_df.empty:
            raise RuntimeError(f"No se encontraron videos en {self.dynamic_dir}")

        summary = self._build_audit_summary(audit_df)
        self._ensure_parent(output_csv)
        self._ensure_parent(output_json)
        audit_df.to_csv(output_csv, index=False)
        output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        if not overwrite:
            return audit_df
        return audit_df

    def build_stable_subset(
        self,
        audit_df: Optional[pd.DataFrame] = None,
        audit_csv: Optional[Path] = None,
        output_manifest: Optional[Path] = None,
        min_valid_videos: int = DEFAULT_MIN_VALID_VIDEOS,
    ) -> pd.DataFrame:
        audit_csv = audit_csv or self.artifacts_dir / "dynamic_audit.csv"
        output_manifest = output_manifest or self.artifacts_dir / "dynamic_subset_manifest.csv"

        if audit_df is None:
            audit_df = pd.read_csv(audit_csv)

        valid_df = audit_df[audit_df["status"] == "valid"].copy()
        class_counts = valid_df["class"].value_counts()
        stable_classes = class_counts[class_counts >= min_valid_videos].index

        manifest_df = valid_df[valid_df["class"].isin(stable_classes)].copy()
        manifest_df = manifest_df.sort_values(["class", "video_name"]).drop_duplicates(subset=["video_path"])

        manifest_df["sample_id"] = manifest_df.apply(self._build_sample_id, axis=1)
        manifest_df["sequence_path"] = manifest_df["sample_id"].apply(
            lambda sample_id: str((self.sequences_dir / f"{sample_id}.npz").resolve())
        )

        self._ensure_parent(output_manifest)
        manifest_df.to_csv(output_manifest, index=False)
        return manifest_df

    def extract_sequences(
        self,
        manifest_df: Optional[pd.DataFrame] = None,
        manifest_csv: Optional[Path] = None,
        target_frames: int = DEFAULT_TARGET_FRAMES,
        overwrite: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        manifest_csv = manifest_csv or self.artifacts_dir / "dynamic_subset_manifest.csv"

        if manifest_df is None:
            manifest_df = pd.read_csv(manifest_csv)

        full_manifest_df = manifest_df.copy()
        if limit is not None:
            manifest_df = manifest_df.head(limit).copy()

        self.sequences_dir.mkdir(parents=True, exist_ok=True)
        enriched_rows: List[Dict[str, object]] = []

        with self._create_video_hands() as hands:
            for row in manifest_df.to_dict("records"):
                sequence_path = Path(row["sequence_path"])
                if sequence_path.exists() and not overwrite:
                    enriched_rows.append(row)
                    continue

                sequence, detection_frames, total_frames = self._extract_video_sequence(
                    video_path=Path(row["video_path"]),
                    hands=hands,
                )
                standardized_sequence = self.standardize_sequence(sequence, target_frames)

                np.savez_compressed(
                    sequence_path,
                    sequence=standardized_sequence.astype(np.float32),
                    raw_length=np.int32(len(sequence)),
                    total_frames=np.int32(total_frames),
                    frames_with_detection=np.int32(detection_frames),
                    label=np.array(row["class"]),
                    sample_id=np.array(row["sample_id"]),
                    source_path=np.array(row["video_path"]),
                )

                row["raw_sequence_length"] = len(sequence)
                row["frames_with_detection"] = detection_frames
                row["total_frames"] = total_frames
                row["target_frames"] = target_frames
                enriched_rows.append(row)

        enriched_df = pd.DataFrame(enriched_rows)
        if enriched_df.empty:
            return enriched_df

        if limit is None:
            enriched_df.to_csv(manifest_csv, index=False)
            return enriched_df

        merged_df = full_manifest_df.copy()
        update_columns = [
            column
            for column in ["raw_sequence_length", "frames_with_detection", "total_frames", "target_frames"]
            if column in enriched_df.columns
        ]
        if update_columns:
            updates = enriched_df[["sample_id"] + update_columns].drop_duplicates(subset=["sample_id"])
            merged_df = merged_df.drop(columns=update_columns, errors="ignore")
            merged_df = merged_df.merge(updates, on="sample_id", how="left")
            merged_df.to_csv(manifest_csv, index=False)

        return enriched_df

    def create_splits(
        self,
        manifest_df: Optional[pd.DataFrame] = None,
        manifest_csv: Optional[Path] = None,
        output_json: Optional[Path] = None,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> Dict[str, object]:
        manifest_csv = manifest_csv or self.artifacts_dir / "dynamic_subset_manifest.csv"
        output_json = output_json or self.artifacts_dir / "dynamic_splits.json"

        if manifest_df is None:
            manifest_df = pd.read_csv(manifest_csv)

        split_result = self._stratified_split(manifest_df, seed=seed)
        payload = {
            "seed": seed,
            "train": split_result.train,
            "val": split_result.val,
            "test": split_result.test,
            "excluded_classes": split_result.excluded_classes,
        }

        self._ensure_parent(output_json)
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def train_model(
        self,
        manifest_df: Optional[pd.DataFrame] = None,
        manifest_csv: Optional[Path] = None,
        splits_json: Optional[Path] = None,
        target_frames: int = DEFAULT_TARGET_FRAMES,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> Dict[str, object]:
        tf = self._require_tensorflow()
        manifest_csv = manifest_csv or self.artifacts_dir / "dynamic_subset_manifest.csv"
        splits_json = splits_json or self.artifacts_dir / "dynamic_splits.json"

        if manifest_df is None:
            manifest_df = pd.read_csv(manifest_csv)
        splits = json.loads(splits_json.read_text(encoding="utf-8"))

        train_df, val_df, test_df = self._frames_from_splits(manifest_df, splits)
        label_map = self._build_label_map(train_df, val_df, test_df)
        X_train, y_train = self._load_dataset_arrays(train_df, label_map, target_frames)
        X_val, y_val = self._load_dataset_arrays(val_df, label_map, target_frames)
        X_test, y_test = self._load_dataset_arrays(test_df, label_map, target_frames)

        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

        model = self._build_model(
            tf=tf,
            target_frames=target_frames,
            feature_size=DEFAULT_FEATURE_SIZE,
            num_classes=len(label_map),
            learning_rate=learning_rate,
        )

        class_weights = self._compute_class_weights(y_train)
        checkpoint_path = self.models_dir / "dynamic_baseline.keras"
        metadata_path = self.artifacts_dir / "dynamic_model_metadata.json"
        label_map_path = self.artifacts_dir / "dynamic_label_map.json"
        eval_path = self.artifacts_dir / "dynamic_evaluation.json"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        label_map_path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")

        callbacks = [self._create_validation_macro_f1_callback(tf, X_val, y_val, checkpoint_path)]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        if checkpoint_path.exists():
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            model.save(checkpoint_path)

        y_test_prob = model.predict(X_test, verbose=0)
        evaluation = self._build_evaluation_payload(y_test, y_test_prob, label_map)
        eval_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")

        metadata = {
            "random_seed": random_seed,
            "target_frames": target_frames,
            "feature_size": DEFAULT_FEATURE_SIZE,
            "num_classes": len(label_map),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "class_weights": {str(k): float(v) for k, v in class_weights.items()},
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "test_samples": int(len(test_df)),
            "model_path": str(checkpoint_path.resolve()),
            "label_map_path": str(label_map_path.resolve()),
            "evaluation_path": str(eval_path.resolve()),
            "history": {k: [float(item) for item in v] for k, v in history.history.items()},
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return metadata

    def evaluate_model(
        self,
        manifest_df: Optional[pd.DataFrame] = None,
        manifest_csv: Optional[Path] = None,
        splits_json: Optional[Path] = None,
        label_map_json: Optional[Path] = None,
        model_path: Optional[Path] = None,
        output_json: Optional[Path] = None,
        target_frames: int = DEFAULT_TARGET_FRAMES,
    ) -> Dict[str, object]:
        tf = self._require_tensorflow()
        manifest_csv = manifest_csv or self.artifacts_dir / "dynamic_subset_manifest.csv"
        splits_json = splits_json or self.artifacts_dir / "dynamic_splits.json"
        label_map_json = label_map_json or self.artifacts_dir / "dynamic_label_map.json"
        model_path = model_path or self.models_dir / "dynamic_baseline.keras"
        output_json = output_json or self.artifacts_dir / "dynamic_evaluation.json"

        if manifest_df is None:
            manifest_df = pd.read_csv(manifest_csv)

        splits = json.loads(splits_json.read_text(encoding="utf-8"))
        label_map = json.loads(label_map_json.read_text(encoding="utf-8"))
        _, _, test_df = self._frames_from_splits(manifest_df, splits)
        X_test, y_test = self._load_dataset_arrays(test_df, label_map, target_frames)

        model = tf.keras.models.load_model(model_path)
        y_prob = model.predict(X_test, verbose=0)
        evaluation = self._build_evaluation_payload(y_test, y_prob, label_map)
        output_json.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
        return evaluation

    def run_pipeline(
        self,
        target_frames: int = DEFAULT_TARGET_FRAMES,
        min_valid_videos: int = DEFAULT_MIN_VALID_VIDEOS,
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        audit_df = self.audit_dataset()
        manifest_df = self.build_stable_subset(audit_df=audit_df, min_valid_videos=min_valid_videos)
        manifest_df = self.extract_sequences(manifest_df=manifest_df, target_frames=target_frames)
        self.create_splits(manifest_df=manifest_df, seed=seed)
        self.train_model(
            manifest_df=manifest_df,
            target_frames=target_frames,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_seed=seed,
        )

    def standardize_sequence(self, sequence: np.ndarray, target_frames: int) -> np.ndarray:
        if sequence.size == 0 or len(sequence) == 0:
            return np.zeros((target_frames, DEFAULT_FEATURE_SIZE), dtype=np.float32)

        frame_indices = np.linspace(0, len(sequence) - 1, num=target_frames)
        frame_indices = np.round(frame_indices).astype(int)
        frame_indices = np.clip(frame_indices, 0, len(sequence) - 1)
        standardized = sequence[frame_indices]
        return standardized.astype(np.float32)

    def _inspect_video(
        self,
        video_path: Path,
        class_name: str,
        tiny_file_threshold_bytes: int,
        min_frames: int,
    ) -> Dict[str, object]:
        file_size = video_path.stat().st_size
        record: Dict[str, object] = {
            "class": class_name,
            "video_name": video_path.name,
            "video_path": str(video_path.resolve()),
            "bytes": file_size,
            "status": "valid",
            "reason": "ok",
            "frame_count": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration_seconds": 0.0,
        }

        if file_size < tiny_file_threshold_bytes:
            record["status"] = "invalid"
            record["reason"] = "tiny_file"
            return record

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            record["status"] = "invalid"
            record["reason"] = "cannot_open"
            return record

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frames_read = 0
        has_frame = False
        while frames_read < min_frames:
            ok, _ = capture.read()
            if not ok:
                break
            has_frame = True
            frames_read += 1

        capture.release()

        record["frame_count"] = frame_count
        record["fps"] = fps
        record["width"] = width
        record["height"] = height
        record["duration_seconds"] = float(frame_count / fps) if fps > 0 else 0.0

        if not has_frame or frame_count < min_frames:
            record["status"] = "invalid"
            record["reason"] = "insufficient_frames"
        elif width <= 0 or height <= 0:
            record["status"] = "invalid"
            record["reason"] = "invalid_resolution"

        return record

    def _build_audit_summary(self, audit_df: pd.DataFrame) -> Dict[str, object]:
        valid_df = audit_df[audit_df["status"] == "valid"]
        invalid_df = audit_df[audit_df["status"] != "valid"]
        valid_per_class = valid_df.groupby("class").size().to_dict()
        invalid_per_reason = invalid_df["reason"].value_counts().to_dict()

        stable_classes = {
            class_name: int(count)
            for class_name, count in valid_per_class.items()
            if count >= DEFAULT_MIN_VALID_VIDEOS
        }

        return {
            "dataset_root": str(self.dynamic_dir.resolve()),
            "total_videos": int(len(audit_df)),
            "valid_videos": int(len(valid_df)),
            "invalid_videos": int(len(invalid_df)),
            "invalid_by_reason": invalid_per_reason,
            "classes_total": int(audit_df["class"].nunique()),
            "classes_with_valid_videos": int(len(valid_per_class)),
            "classes_meeting_default_threshold": int(len(stable_classes)),
            "valid_videos_per_class": valid_per_class,
            "stable_classes": stable_classes,
        }

    def _extract_video_sequence(
        self,
        video_path: Path,
        hands,
    ) -> Tuple[np.ndarray, int, int]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return np.zeros((0, DEFAULT_FEATURE_SIZE), dtype=np.float32), 0, 0

        frames: List[np.ndarray] = []
        detections = 0
        total_frames = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            total_frames += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            frame_features = self._frame_features_from_results(results)
            if np.any(frame_features):
                detections += 1
            frames.append(frame_features)

        capture.release()

        if not frames:
            return np.zeros((0, DEFAULT_FEATURE_SIZE), dtype=np.float32), 0, total_frames
        return np.stack(frames).astype(np.float32), detections, total_frames

    def _frame_features_from_results(self, results) -> np.ndarray:
        frame_vector = np.zeros((DEFAULT_MAX_HANDS, DEFAULT_HAND_LANDMARKS, 3), dtype=np.float32)
        if not results.multi_hand_landmarks:
            return frame_vector.reshape(-1)

        ordered_hands: List[Tuple[str, object]] = []
        handedness = results.multi_handedness or []

        for index, hand_landmarks in enumerate(results.multi_hand_landmarks[:DEFAULT_MAX_HANDS]):
            hand_label = "unknown"
            if index < len(handedness):
                classification = handedness[index].classification
                if classification:
                    hand_label = classification[0].label.lower()
            ordered_hands.append((hand_label, hand_landmarks))

        ordered_hands.sort(key=lambda item: self._hand_sort_key(item[0]))

        for hand_index, (_, hand_landmarks) in enumerate(ordered_hands[:DEFAULT_MAX_HANDS]):
            normalized = self._normalize_hand_landmarks(hand_landmarks)
            frame_vector[hand_index] = normalized

        return frame_vector.reshape(-1)

    def _normalize_hand_landmarks(self, hand_landmarks) -> np.ndarray:
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        wrist = coords[0]
        centered = coords - wrist
        scale = np.max(np.linalg.norm(centered, axis=1))
        if scale > 0:
            centered = centered / scale
        return centered

    def _hand_sort_key(self, label: str) -> int:
        if label == "left":
            return 0
        if label == "right":
            return 1
        return 2

    def _stratified_split(self, manifest_df: pd.DataFrame, seed: int) -> SplitResult:
        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []
        excluded: Dict[str, str] = {}
        rng = np.random.default_rng(seed)

        for class_name, class_df in manifest_df.groupby("class"):
            sample_ids = class_df["sample_id"].tolist()
            rng.shuffle(sample_ids)
            n_samples = len(sample_ids)

            if n_samples < DEFAULT_MIN_VALID_VIDEOS:
                excluded[class_name] = "below_min_valid_videos"
                continue

            train_count = max(1, int(round(n_samples * 0.7)))
            val_count = max(1, int(round(n_samples * 0.15)))
            test_count = n_samples - train_count - val_count

            if test_count <= 0:
                test_count = 1
                if train_count > val_count:
                    train_count -= 1
                else:
                    val_count -= 1

            if train_count <= 0 or val_count <= 0 or test_count <= 0:
                excluded[class_name] = "cannot_allocate_split"
                continue

            if train_count + val_count + test_count != n_samples:
                train_count = n_samples - val_count - test_count

            if train_count <= 0:
                excluded[class_name] = "cannot_allocate_split"
                continue

            train_ids.extend(sample_ids[:train_count])
            val_ids.extend(sample_ids[train_count : train_count + val_count])
            test_ids.extend(sample_ids[train_count + val_count : train_count + val_count + test_count])

        return SplitResult(train=train_ids, val=val_ids, test=test_ids, excluded_classes=excluded)

    def _frames_from_splits(
        self,
        manifest_df: pd.DataFrame,
        splits: Dict[str, object],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        manifest_df = manifest_df.copy()
        sample_index = manifest_df.set_index("sample_id", drop=False)
        train_df = sample_index.loc[splits["train"]].reset_index(drop=True) if splits["train"] else manifest_df.iloc[0:0]
        val_df = sample_index.loc[splits["val"]].reset_index(drop=True) if splits["val"] else manifest_df.iloc[0:0]
        test_df = sample_index.loc[splits["test"]].reset_index(drop=True) if splits["test"] else manifest_df.iloc[0:0]
        return train_df, val_df, test_df

    def _build_label_map(self, *frames: pd.DataFrame) -> Dict[str, int]:
        labels = sorted({label for frame in frames for label in frame["class"].tolist()})
        return {label: index for index, label in enumerate(labels)}

    def _load_dataset_arrays(
        self,
        frame: pd.DataFrame,
        label_map: Dict[str, int],
        target_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        sequences: List[np.ndarray] = []
        labels: List[int] = []

        for row in frame.to_dict("records"):
            sequence_path = Path(row["sequence_path"])
            with np.load(sequence_path, allow_pickle=False) as payload:
                sequence = payload["sequence"].astype(np.float32)
            if sequence.shape != (target_frames, DEFAULT_FEATURE_SIZE):
                sequence = self.standardize_sequence(sequence, target_frames)
            sequences.append(sequence)
            labels.append(label_map[row["class"]])

        if not sequences:
            raise RuntimeError("No hay secuencias disponibles para construir el dataset.")

        return np.stack(sequences), np.array(labels, dtype=np.int32)

    def _build_model(self, tf, target_frames: int, feature_size: int, num_classes: int, learning_rate: float):
        inputs = tf.keras.Input(shape=(target_frames, feature_size), name="sequence")
        x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dynamic_asl_baseline")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        counts = Counter(labels.tolist())
        total = float(len(labels))
        num_classes = float(len(counts))
        return {int(label): total / (num_classes * count) for label, count in counts.items()}

    def _build_evaluation_payload(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        label_map: Dict[str, int],
    ) -> Dict[str, object]:
        id_to_label = {index: label for label, index in label_map.items()}
        y_pred = np.argmax(y_prob, axis=1)
        top3 = self._top_k_accuracy(y_true, y_prob, k=min(3, y_prob.shape[1]))
        confusion = self._confusion_matrix(y_true, y_pred, num_classes=len(label_map))
        report = self._classification_report(y_true, y_pred, id_to_label)

        return {
            "accuracy": self._accuracy(y_true, y_pred),
            "macro_f1": self._macro_f1(y_true, y_pred, num_classes=len(label_map)),
            "top_3_accuracy": top3,
            "labels": [id_to_label[i] for i in range(len(id_to_label))],
            "confusion_matrix": confusion.tolist(),
            "classification_report": report,
        }

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(y_true == y_pred))

    def _macro_f1(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
        f1_scores: List[float] = []
        for class_id in range(num_classes):
            true_positive = np.sum((y_true == class_id) & (y_pred == class_id))
            false_positive = np.sum((y_true != class_id) & (y_pred == class_id))
            false_negative = np.sum((y_true == class_id) & (y_pred != class_id))

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0

            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append((2 * precision * recall) / (precision + recall))
        return float(np.mean(f1_scores))

    def _confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
        matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        for true_label, pred_label in zip(y_true, y_pred):
            matrix[int(true_label), int(pred_label)] += 1
        return matrix

    def _classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        id_to_label: Dict[int, str],
    ) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for class_id, label in id_to_label.items():
            true_positive = np.sum((y_true == class_id) & (y_pred == class_id))
            false_positive = np.sum((y_true != class_id) & (y_pred == class_id))
            false_negative = np.sum((y_true == class_id) & (y_pred != class_id))
            support = int(np.sum(y_true == class_id))

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

            report[label] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        return report

    def _top_k_accuracy(self, y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
        top_k = np.argsort(y_prob, axis=1)[:, -k:]
        hits = [int(true_label in row) for true_label, row in zip(y_true, top_k)]
        return float(np.mean(hits))

    def _build_sample_id(self, row: pd.Series) -> str:
        normalized_name = row["video_name"].replace(".mp4", "").replace(" ", "_")
        return f"{row['class'].replace(' ', '_')}__{normalized_name}"

    def _create_video_hands(self):
        return mp_solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=DEFAULT_MAX_HANDS,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

    def _require_tensorflow(self):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow no está instalado. Instálalo antes de entrenar o evaluar el modelo dinámico."
            ) from exc
        return tf

    def _resolve_project_root(self) -> Path:
        current = self.module_dir
        for candidate in [current] + list(current.parents):
            if (candidate / "data" / "Dinamico").exists():
                return candidate
        raise RuntimeError(
            f"No se pudo resolver la raíz del proyecto desde {self.module_dir}. "
            "Se esperaba encontrar data/Dinamico en algún directorio padre."
        )

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _create_validation_macro_f1_callback(self, tf, X_val: np.ndarray, y_val: np.ndarray, checkpoint_path: Path):
        pipeline_cls = self

        class ValidationMacroF1Callback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.best_macro_f1 = -1.0

            def on_epoch_end(self, epoch, logs=None):
                predictions = self.model.predict(X_val, verbose=0)
                y_pred = np.argmax(predictions, axis=1)
                macro_f1 = pipeline_cls._static_macro_f1(y_val, y_pred)
                logs = logs or {}
                logs["val_macro_f1"] = macro_f1
                if macro_f1 > self.best_macro_f1:
                    self.best_macro_f1 = macro_f1
                    self.model.save(checkpoint_path)
                print(f"val_macro_f1: {macro_f1:.4f}")

        return ValidationMacroF1Callback()

    @staticmethod
    def _static_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        f1_scores = []
        for class_id in labels:
            tp = np.sum((y_true == class_id) & (y_pred == class_id))
            fp = np.sum((y_true != class_id) & (y_pred == class_id))
            fn = np.sum((y_true == class_id) & (y_pred != class_id))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1_scores.append((2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0)
        return float(np.mean(f1_scores)) if f1_scores else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline PoC para reconocimiento dinámico ASL.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audita la calidad del dataset dinámico.")
    audit.add_argument("--tiny-file-threshold", type=int, default=1024)
    audit.add_argument("--min-frames", type=int, default=2)

    subset = subparsers.add_parser("filter", help="Construye el subconjunto estable.")
    subset.add_argument("--min-valid-videos", type=int, default=DEFAULT_MIN_VALID_VIDEOS)

    extract = subparsers.add_parser("extract", help="Extrae y serializa secuencias de landmarks.")
    extract.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--limit", type=int, default=None)

    split = subparsers.add_parser("split", help="Genera splits train/val/test reproducibles.")
    split.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    train = subparsers.add_parser("train", help="Entrena el baseline secuencial.")
    train.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    evaluate = subparsers.add_parser("evaluate", help="Evalúa el modelo entrenado.")
    evaluate.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)

    pipeline = subparsers.add_parser("pipeline", help="Ejecuta el flujo completo.")
    pipeline.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    pipeline.add_argument("--min-valid-videos", type=int, default=DEFAULT_MIN_VALID_VIDEOS)
    pipeline.add_argument("--epochs", type=int, default=10)
    pipeline.add_argument("--batch-size", type=int, default=16)
    pipeline.add_argument("--learning-rate", type=float, default=1e-3)
    pipeline.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = DynamicASLPipeline()
    try:
        if args.command == "audit":
            df = pipeline.audit_dataset(
                tiny_file_threshold_bytes=args.tiny_file_threshold,
                min_frames=args.min_frames,
            )
            print(f"Auditoría completada: {len(df)} videos inspeccionados.")
        elif args.command == "filter":
            df = pipeline.build_stable_subset(min_valid_videos=args.min_valid_videos)
            print(f"Manifest generado: {len(df)} videos válidos en el subconjunto estable.")
        elif args.command == "extract":
            df = pipeline.extract_sequences(
                target_frames=args.target_frames,
                overwrite=args.overwrite,
                limit=args.limit,
            )
            print(f"Extracción completada: {len(df)} secuencias procesadas.")
        elif args.command == "split":
            payload = pipeline.create_splits(seed=args.seed)
            print(
                f"Splits generados. Train={len(payload['train'])}, "
                f"Val={len(payload['val'])}, Test={len(payload['test'])}"
            )
        elif args.command == "train":
            metadata = pipeline.train_model(
                target_frames=args.target_frames,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                random_seed=args.seed,
            )
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
        elif args.command == "evaluate":
            evaluation = pipeline.evaluate_model(target_frames=args.target_frames)
            print(json.dumps(evaluation, indent=2, ensure_ascii=False))
        elif args.command == "pipeline":
            pipeline.run_pipeline(
                target_frames=args.target_frames,
                min_valid_videos=args.min_valid_videos,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
            print("Pipeline completo ejecutado.")
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
