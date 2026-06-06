import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clean_dataset import ensure_cleaned_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize


VALID_CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]
DEFAULT_OUTPUT_DIR = "data/Metricas_Estaticas"
AMBIGUOUS_CLASS_GROUPS = [
    frozenset({"a", "e", "o", "s", "m", "n"}),
    frozenset({"r", "u", "v", "k"}),
    frozenset({"x", "y", "z"}),
    frozenset({"p", "q"}),
]


class StaticModelMetrics:
    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        test_size: float,
        random_state: int,
        augment_class: str | None = None,
        augment_factor: int = 0,
        augment_noise_std: float = 0.008,
        balance_to_max: bool = False,
        use_class_weights: bool = True,
        prototypes_per_class: int = 3,
    ):
        self.project_root = Path(__file__).resolve().parents[2]
        self.csv_path = self.resolve_project_path(csv_path)
        self.output_dir = self.resolve_project_path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.augment_class = augment_class.lower() if augment_class else None
        self.augment_factor = max(0, int(augment_factor))
        self.augment_noise_std = float(augment_noise_std)
        self.balance_to_max = balance_to_max
        self.use_class_weights = use_class_weights
        self.prototypes_per_class = max(1, int(prototypes_per_class))

    def resolve_project_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path

        if path.parts and path.parts[0] == self.project_root.name:
            return self.project_root.parent / path

        return self.project_root / path

    def load_dataset(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo CSV: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        df["class"] = df["class"].astype(str).str.lower()
        df = df[df["class"].isin(VALID_CLASSES)].copy()

        # Detectar e integrar dataset 'alphabet' si el caché existe
        alphabet_cache = self.project_root / "data" / "kagglehub" / "alphabet_landmarks_cache.pkl"
        if alphabet_cache.exists():
            print(f"📊 Caché de dataset 'alphabet' detectado")
            df_alphabet = self._load_alphabet_from_cache(alphabet_cache)
            if df_alphabet is not None and not df_alphabet.empty:
                initial_rows = len(df)
                df = pd.concat([df, df_alphabet], ignore_index=True)
                df["class"] = df["class"].astype(str).str.lower()
                df = df[df["class"].isin(VALID_CLASSES)].copy()
                print(f"   ✅ Dataset principal: {initial_rows} filas, 'alphabet' caché: {len(df_alphabet)} filas, total: {len(df)}")

        feature_columns = self.get_feature_columns(df)
        missing_values = df[feature_columns].isna().any(axis=1).sum()
        if missing_values:
            print(f"⚠️  Se descartarán {missing_values} filas con valores faltantes.")
            df = df.dropna(subset=feature_columns).copy()

        class_counts = df["class"].value_counts()
        invalid_for_split = class_counts[class_counts < 2]
        if not invalid_for_split.empty:
            print(
                "⚠️  Se descartarán clases con menos de 2 muestras para permitir un split estratificado: "
                + ", ".join(f"{label} ({count})" for label, count in invalid_for_split.items())
            )
            df = df[df["class"].isin(class_counts[class_counts >= 2].index)].copy()

        if df.empty:
            raise ValueError("El dataset no contiene suficientes datos válidos para evaluar.")

        return df

    def _load_alphabet_from_cache(self, cache_file: Path) -> pd.DataFrame | None:
        """Carga el caché de landmarks de 'alphabet' desde pickle."""
        try:
            with open(cache_file, "rb") as f:
                df_alphabet = pickle.load(f)
            return df_alphabet
        except Exception as e:
            print(f"⚠️  Error al cargar caché de 'alphabet': {e}")
            return None

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        landmark_columns = []
        for i in range(21):
            landmark_columns.extend(
                [f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"]
            )

        missing = [column for column in landmark_columns if column not in df.columns]
        if missing:
            raise ValueError(
                "Faltan columnas necesarias de landmarks en el CSV: "
                + ", ".join(missing[:10])
            )

        return landmark_columns

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        median = np.median(features)
        mad = np.median(np.abs(features - median))
        if mad == 0:
            mad = np.std(features)
            if mad == 0:
                return features - median

        normalized = (features - median) / mad
        return np.clip(normalized, -3, 3)

    def fast_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Correlación de Pearson optimizada para el bucle interno de scoring.
        Evita el costo elevado de np.corrcoef por comparación.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
        if denominator == 0:
            return 0.0

        correlation = float(np.dot(x_centered, y_centered) / denominator)
        if np.isnan(correlation):
            return 0.0
        return correlation

    def compute_similarity_components(
        self,
        sample_normalized: np.ndarray,
        prototype_normalized: np.ndarray,
    ) -> dict[str, float]:
        sample_norm = np.linalg.norm(sample_normalized)
        prototype_norm = np.linalg.norm(prototype_normalized)

        cosine_similarity = 0.0
        if sample_norm != 0 and prototype_norm != 0:
            cosine_similarity = float(
                np.dot(sample_normalized, prototype_normalized) / (sample_norm * prototype_norm)
            )

        correlation = self.fast_correlation(sample_normalized, prototype_normalized)
        euclidean_distance = float(np.linalg.norm(sample_normalized - prototype_normalized))

        geometry_sample = sample_normalized[63:] if len(sample_normalized) > 63 else np.array([], dtype=float)
        geometry_prototype = prototype_normalized[63:] if len(prototype_normalized) > 63 else np.array([], dtype=float)

        geometry_cosine = 0.0
        geometry_correlation = 0.0
        geometry_distance = euclidean_distance
        if len(geometry_sample) > 0 and len(geometry_sample) == len(geometry_prototype):
            geometry_sample_norm = np.linalg.norm(geometry_sample)
            geometry_prototype_norm = np.linalg.norm(geometry_prototype)
            if geometry_sample_norm != 0 and geometry_prototype_norm != 0:
                geometry_cosine = float(
                    np.dot(geometry_sample, geometry_prototype)
                    / (geometry_sample_norm * geometry_prototype_norm)
                )
            geometry_correlation = self.fast_correlation(geometry_sample, geometry_prototype)
            geometry_distance = float(np.linalg.norm(geometry_sample - geometry_prototype))

        return {
            "cosine_similarity": cosine_similarity,
            "correlation": correlation,
            "euclidean_distance": euclidean_distance,
            "geometry_cosine": geometry_cosine,
            "geometry_correlation": geometry_correlation,
            "geometry_distance": geometry_distance,
        }

    def combine_score(self, components: dict[str, float], emphasize_geometry: bool = False) -> float:
        if emphasize_geometry:
            return (
                components["cosine_similarity"] * 0.30
                + components["correlation"] * 0.20
                + components["geometry_cosine"] * 0.25
                + components["geometry_correlation"] * 0.15
                + (1 - components["geometry_distance"] / 5) * 0.10
            )

        return (
            components["cosine_similarity"] * 0.4
            + components["correlation"] * 0.4
            + (1 - components["euclidean_distance"] / 10) * 0.2
        )

    def find_ambiguous_group(self, ranked_labels: list[str]) -> frozenset[str] | None:
        top_labels = ranked_labels[: min(3, len(ranked_labels))]
        for group in AMBIGUOUS_CLASS_GROUPS:
            overlapping = [label for label in top_labels if label in group]
            if len(overlapping) >= 2:
                return group
        return None

    def add_distance_features(self, row: pd.Series) -> np.ndarray:
        landmarks = []
        for i in range(21):
            landmarks.append(
                [
                    row[f"landmark_{i}_x"],
                    row[f"landmark_{i}_y"],
                    row[f"landmark_{i}_z"],
                ]
            )

        landmarks = np.asarray(landmarks, dtype=float)
        features = landmarks.flatten().tolist()

        wrist = landmarks[0]
        fingertip_indices = [4, 8, 12, 16, 20]
        fingertip_distances = [np.linalg.norm(wrist - landmarks[idx]) for idx in fingertip_indices]

        consecutive_tip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
        consecutive_tip_distances = [
            np.linalg.norm(landmarks[a] - landmarks[b]) for a, b in consecutive_tip_pairs
        ]

        palm_span = np.linalg.norm(landmarks[5] - landmarks[17])

        features.extend(fingertip_distances)
        features.extend(consecutive_tip_distances)
        features.append(palm_span)

        return np.asarray(features, dtype=float)

    def build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.vstack([self.add_distance_features(row) for _, row in df.iterrows()])

    def augment_row(self, row: pd.Series) -> pd.Series:
        """
        Genera una versión ligeramente perturbada de una fila de landmarks.
        Mantiene la etiqueta original y simula variaciones pequeñas de captura.
        """
        augmented = row.copy()
        landmark_columns = [
            f"landmark_{i}_{axis}"
            for i in range(21)
            for axis in ["x", "y", "z"]
        ]

        noise = np.random.normal(0.0, self.augment_noise_std, size=len(landmark_columns))
        values = augmented[landmark_columns].astype(float).to_numpy() + noise

        # Pequeño sesgo específico para que la clase 'm' deje de parecerse tanto a 'e'/'o'
        if str(augmented.get("class", "")).lower() == "m":
            for idx in [8, 12, 16, 20]:
                values[idx * 3 + 1] *= 0.995  # y ligeramente más estable
                values[idx * 3 + 2] *= 1.005  # z ligeramente más abierto

        augmented[landmark_columns] = values
        return augmented

    def build_augmented_training_set(self, df: pd.DataFrame):
        """
        Construye el conjunto de entrenamiento con balanceo opcional y augmentación focalizada.
        """
        working_df = df.copy().reset_index(drop=True)

        if self.balance_to_max or (self.augment_class and self.augment_factor > 0):
            class_counts = working_df["class"].value_counts().to_dict()
            target_counts = class_counts.copy()

            if self.balance_to_max and class_counts:
                max_count = max(class_counts.values())
                for class_name in class_counts:
                    target_counts[class_name] = max_count

            if self.augment_class and self.augment_factor > 0:
                target_counts[self.augment_class] = max(
                    target_counts.get(self.augment_class, 0),
                    class_counts.get(self.augment_class, 0) * (self.augment_factor + 1),
                )

            augmented_rows = []
            for class_name, target_count in target_counts.items():
                class_rows = working_df[working_df["class"] == class_name]
                current_count = len(class_rows)
                if current_count == 0 or current_count >= target_count:
                    continue

                needed = target_count - current_count
                class_indices = class_rows.index.to_list()
                for i in range(needed):
                    base_row = working_df.loc[class_indices[i % len(class_indices)]]
                    new_row = self.augment_row(base_row)
                    new_row["class"] = class_name
                    augmented_rows.append(new_row)

            if augmented_rows:
                working_df = pd.concat([working_df, pd.DataFrame(augmented_rows)], ignore_index=True)

        X = self.build_feature_matrix(working_df)
        y = working_df["class"].astype(str).to_numpy()
        return working_df, X, y

    def split_dataset(self, X: np.ndarray, y: np.ndarray):
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        train_idx, test_idx = next(splitter.split(X, y))
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx

    def compute_weighted_mean(
        self,
        class_features: np.ndarray,
        class_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        if class_weights is not None and np.sum(class_weights) > 0:
            return np.average(class_features, axis=0, weights=class_weights)
        return np.mean(class_features, axis=0)

    def build_prototypes(self, X_train: np.ndarray, y_train: np.ndarray, sample_weights: np.ndarray | None = None):
        prototypes = {}
        for class_name in sorted(np.unique(y_train)):
            class_mask = y_train == class_name
            class_features = X_train[class_mask]
            class_w = np.asarray(sample_weights)[class_mask] if sample_weights is not None else None
            num_clusters = min(self.prototypes_per_class, len(class_features))

            if num_clusters <= 1 or len(class_features) < num_clusters * 4:
                prototypes[class_name] = [self.compute_weighted_mean(class_features, class_w)]
                continue

            kmeans = KMeans(
                n_clusters=num_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            cluster_ids = kmeans.fit_predict(class_features)

            class_prototypes = []
            for cluster_id in range(num_clusters):
                cluster_mask = cluster_ids == cluster_id
                cluster_features = class_features[cluster_mask]
                cluster_weights = class_w[cluster_mask] if class_w is not None else None
                class_prototypes.append(
                    self.compute_weighted_mean(cluster_features, cluster_weights)
                )

            prototypes[class_name] = class_prototypes
        return prototypes

    def compute_sample_weights(self, y_train: np.ndarray) -> np.ndarray | None:
        if not self.use_class_weights:
            return None

        counts = Counter(y_train.tolist())
        num_classes = len(counts)
        total = len(y_train)
        weights = {class_name: total / (num_classes * count) for class_name, count in counts.items() if count > 0}
        return np.asarray([weights[label] for label in y_train], dtype=float)

    def score_sample(self, sample: np.ndarray, prototypes: dict[str, list[np.ndarray]]):
        sample_normalized = self.normalize_features(sample)
        scores = {}

        for class_name, class_prototypes in prototypes.items():
            best_class_score = -np.inf

            for prototype in class_prototypes:
                prototype_normalized = self.normalize_features(prototype)
                components = self.compute_similarity_components(
                    sample_normalized,
                    prototype_normalized,
                )
                combined_score = self.combine_score(components)
                if combined_score > best_class_score:
                    best_class_score = combined_score

            scores[class_name] = best_class_score

        ranked_by_score = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ambiguous_group = self.find_ambiguous_group([label for label, _ in ranked_by_score])
        if ambiguous_group is not None:
            for class_name in ambiguous_group:
                class_prototypes = prototypes.get(class_name)
                if not class_prototypes:
                    continue

                best_refined_score = -np.inf
                for prototype in class_prototypes:
                    prototype_normalized = self.normalize_features(prototype)
                    components = self.compute_similarity_components(
                        sample_normalized,
                        prototype_normalized,
                    )
                    refined_score = self.combine_score(components, emphasize_geometry=True)
                    if refined_score > best_refined_score:
                        best_refined_score = refined_score

                if best_refined_score > -np.inf:
                    scores[class_name] = best_refined_score

        ordered_labels = sorted(scores)
        raw_scores = np.array([scores[label] for label in ordered_labels], dtype=float)
        probabilities = self.softmax(raw_scores)
        predicted_label = ordered_labels[int(np.argmax(raw_scores))]

        return predicted_label, ordered_labels, raw_scores, probabilities

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores)

    def predict_dataset(
        self,
        X_split: np.ndarray,
        y_split: np.ndarray,
        prototypes: dict[str, list[np.ndarray]],
    ):
        y_pred = []
        y_scores = []
        top3_hits = 0

        for sample, true_label in zip(X_split, y_split):
            predicted_label, ordered_labels, _, probabilities = self.score_sample(sample, prototypes)
            y_pred.append(predicted_label)
            y_scores.append(probabilities)

            ranked_labels = [
                ordered_labels[idx]
                for idx in np.argsort(probabilities)[::-1][: min(3, len(ordered_labels))]
            ]
            if true_label in ranked_labels:
                top3_hits += 1

        return np.asarray(y_pred), np.vstack(y_scores), float(top3_hits / len(y_split))

    def build_confusion_outputs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_labels: list[str],
    ):
        confusion = confusion_matrix(y_true, y_pred, labels=class_labels)
        confusion_df = pd.DataFrame(confusion, index=class_labels, columns=class_labels)

        confusion_normalized = confusion.astype(float)
        row_sums = confusion_normalized.sum(axis=1, keepdims=True)
        np.divide(
            confusion_normalized,
            row_sums,
            out=confusion_normalized,
            where=row_sums != 0,
        )
        confusion_normalized_df = pd.DataFrame(
            confusion_normalized,
            index=class_labels,
            columns=class_labels,
        )
        return confusion_df, confusion_normalized_df

    def evaluate(self) -> dict:
        df = self.load_dataset()
        working_df, X, y = self.build_augmented_training_set(df)

        X_train, X_test, y_train, y_test, train_idx, test_idx = self.split_dataset(X, y)
        sample_weights = self.compute_sample_weights(y_train)
        prototypes = self.build_prototypes(X_train, y_train, sample_weights=sample_weights)
        class_labels = sorted(prototypes.keys())

        y_train_pred, y_train_scores, train_top3_hits = self.predict_dataset(X_train, y_train, prototypes)
        y_pred, y_scores, top3_hits = self.predict_dataset(X_test, y_test, prototypes)

        y_test_bin = label_binarize(y_test, classes=class_labels)

        metrics_summary = {
            "num_samples_total": int(len(df)),
            "num_samples_total_augmented": int(len(working_df)),
            "num_samples_train": int(len(y_train)),
            "num_samples_test": int(len(y_test)),
            "num_classes": int(len(class_labels)),
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "train_balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision_macro": float(
                precision_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "precision_weighted": float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, y_pred, average="macro", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "top_3_accuracy": float(top3_hits),
            "train_top_3_accuracy": float(train_top3_hits),
            "augment_class": self.augment_class,
            "augment_factor": int(self.augment_factor),
            "augment_noise_std": float(self.augment_noise_std),
            "balance_to_max": bool(self.balance_to_max),
            "use_class_weights": bool(self.use_class_weights),
            "prototypes_per_class": int(self.prototypes_per_class),
        }

        try:
            metrics_summary["top_1_accuracy_sklearn"] = float(
                top_k_accuracy_score(y_test, y_scores, k=1, labels=class_labels)
            )
        except ValueError:
            metrics_summary["top_1_accuracy_sklearn"] = metrics_summary["accuracy"]

        auc_metrics = self.compute_auc_metrics(y_test, y_test_bin, y_scores, class_labels)
        metrics_summary.update(auc_metrics)

        report = classification_report(
            y_test,
            y_pred,
            labels=class_labels,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()

        train_confusion_df, train_confusion_normalized_df = self.build_confusion_outputs(
            y_train, y_train_pred, class_labels
        )
        confusion_df, confusion_normalized_df = self.build_confusion_outputs(
            y_test, y_pred, class_labels
        )

        prediction_df = pd.DataFrame(
            {
                "true_label": y_test,
                "predicted_label": y_pred,
                "is_correct": y_test == y_pred,
                "source_index": test_idx,
            }
        )

        source_columns = [col for col in ["image_path", "image_name"] if col in working_df.columns]
        if source_columns:
            source_df = working_df.iloc[test_idx][source_columns].reset_index(drop=True)
            prediction_df = pd.concat([prediction_df.reset_index(drop=True), source_df], axis=1)

        score_columns = pd.DataFrame(y_scores, columns=[f"score_{label}" for label in class_labels])
        prediction_df = pd.concat([prediction_df.reset_index(drop=True), score_columns], axis=1)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metricas_resumen.json").write_text(
            json.dumps(metrics_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report_df.to_csv(self.output_dir / "metricas_por_clase.csv", encoding="utf-8")
        train_confusion_df.to_csv(self.output_dir / "matriz_confusion_train.csv", encoding="utf-8")
        train_confusion_normalized_df.to_csv(
            self.output_dir / "matriz_confusion_train_normalizada.csv",
            encoding="utf-8",
        )
        confusion_df.to_csv(self.output_dir / "matriz_confusion.csv", encoding="utf-8")
        confusion_normalized_df.to_csv(
            self.output_dir / "matriz_confusion_normalizada.csv",
            encoding="utf-8",
        )
        prediction_df.to_csv(self.output_dir / "predicciones_detalladas.csv", index=False, encoding="utf-8")
        self.save_plots(
            metrics_summary,
            report_df,
            train_confusion_df,
            train_confusion_normalized_df,
            confusion_df,
            confusion_normalized_df,
        )

        return {
            "summary": metrics_summary,
            "report_df": report_df,
            "train_confusion_df": train_confusion_df,
            "train_confusion_normalized_df": train_confusion_normalized_df,
            "confusion_df": confusion_df,
            "confusion_normalized_df": confusion_normalized_df,
        }

    def compute_auc_metrics(
        self,
        y_test: np.ndarray,
        y_test_bin: np.ndarray,
        y_scores: np.ndarray,
        class_labels: list[str],
    ) -> dict:
        auc_metrics = {}

        if len(class_labels) == 2:
            try:
                auc_metrics["roc_auc"] = float(roc_auc_score(y_test_bin, y_scores[:, 1]))
            except ValueError:
                auc_metrics["roc_auc"] = None
            return auc_metrics

        for average_name in ["macro", "weighted"]:
            try:
                auc_metrics[f"roc_auc_ovr_{average_name}"] = float(
                    roc_auc_score(
                        y_test_bin,
                        y_scores,
                        average=average_name,
                        multi_class="ovr",
                    )
                )
            except ValueError:
                auc_metrics[f"roc_auc_ovr_{average_name}"] = None

            try:
                auc_metrics[f"roc_auc_ovo_{average_name}"] = float(
                    roc_auc_score(
                        y_test_bin,
                        y_scores,
                        average=average_name,
                        multi_class="ovo",
                    )
                )
            except ValueError:
                auc_metrics[f"roc_auc_ovo_{average_name}"] = None

        return auc_metrics

    def save_plots(
        self,
        metrics_summary: dict,
        report_df: pd.DataFrame,
        train_confusion_df: pd.DataFrame,
        train_confusion_normalized_df: pd.DataFrame,
        confusion_df: pd.DataFrame,
        confusion_normalized_df: pd.DataFrame,
    ):
        self.plot_confusion_matrix(
            train_confusion_df,
            self.output_dir / "matriz_confusion_train.png",
            "Matriz de Confusión - Entrenamiento",
            value_format="d",
        )
        self.plot_confusion_matrix(
            train_confusion_normalized_df,
            self.output_dir / "matriz_confusion_train_normalizada.png",
            "Matriz de Confusión Normalizada - Entrenamiento",
            value_format=".2f",
        )
        self.plot_confusion_matrix(
            confusion_df,
            self.output_dir / "matriz_confusion_validacion.png",
            "Matriz de Confusión - Validación",
            value_format="d",
        )
        self.plot_confusion_matrix(
            confusion_normalized_df,
            self.output_dir / "matriz_confusion_validacion_normalizada.png",
            "Matriz de Confusión Normalizada - Validación",
            value_format=".2f",
        )
        # Compatibilidad hacia atrás con nombres previos usando validación
        self.plot_confusion_matrix(
            confusion_df,
            self.output_dir / "matriz_confusion.png",
            "Matriz de Confusión - Validación",
            value_format="d",
        )
        self.plot_confusion_matrix(
            confusion_normalized_df,
            self.output_dir / "matriz_confusion_normalizada.png",
            "Matriz de Confusión Normalizada - Validación",
            value_format=".2f",
        )
        self.plot_summary_metrics(
            metrics_summary,
            self.output_dir / "metricas_resumen.png",
        )
        self.plot_class_metrics(
            report_df,
            self.output_dir / "metricas_por_clase.png",
        )

    def plot_confusion_matrix(
        self,
        matrix_df: pd.DataFrame,
        output_path: Path,
        title: str,
        value_format: str,
    ):
        labels = matrix_df.index.tolist()
        matrix = matrix_df.to_numpy()
        size = max(12, min(24, len(labels) * 0.45))

        fig, ax = plt.subplots(figsize=(size, size))
        image = ax.imshow(matrix, cmap="Blues", aspect="auto")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta real")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

        threshold = matrix.max() / 2 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=7,
                )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_summary_metrics(self, metrics_summary: dict, output_path: Path):
        metric_keys = [
            "accuracy",
            "balanced_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "top_3_accuracy",
        ]
        auc_keys = [key for key in metrics_summary if key.startswith("roc_auc") and metrics_summary[key] is not None]
        metric_keys.extend(sorted(auc_keys))

        labels = [key.replace("_", "\n") for key in metric_keys]
        values = [metrics_summary[key] for key in metric_keys]

        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(labels, values, color="#2A6F97")
        ax.set_ylim(0, 1.05)
        ax.set_title("Resumen de Métricas del Modelo Estático")
        ax.set_ylabel("Valor")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_class_metrics(self, report_df: pd.DataFrame, output_path: Path):
        excluded_rows = {"accuracy", "macro avg", "weighted avg"}
        class_df = report_df.loc[~report_df.index.isin(excluded_rows), ["precision", "recall", "f1-score"]].copy()
        class_df = class_df.sort_values("f1-score", ascending=True)

        height = max(8, min(22, len(class_df) * 0.35))
        fig, axes = plt.subplots(1, 3, figsize=(20, height), sharey=True)
        metrics = [
            ("precision", "#277DA1", "Precision por clase"),
            ("recall", "#90BE6D", "Recall por clase"),
            ("f1-score", "#F8961E", "F1-score por clase"),
        ]

        y_pos = np.arange(len(class_df))
        labels = class_df.index.tolist()

        for ax, (column, color, title) in zip(axes, metrics):
            values = class_df[column].to_numpy()
            ax.barh(y_pos, values, color=color)
            ax.set_title(title)
            ax.set_xlim(0, 1.05)
            ax.grid(axis="x", linestyle="--", alpha=0.35)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)

        fig.suptitle("Desempeño por Clase", fontsize=16)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def print_summary(self, results: dict):
        summary = results["summary"]
        print("\n📊 MÉTRICAS DEL MODELO ESTÁTICO")
        print("=" * 60)
        print(f"Muestras totales: {summary['num_samples_total']}")
        print(f"Muestras totales tras augmentación: {summary['num_samples_total_augmented']}")
        print(f"Muestras entrenamiento: {summary['num_samples_train']}")
        print(f"Muestras prueba: {summary['num_samples_test']}")
        print(f"Clases evaluadas: {summary['num_classes']}")
        if summary.get("augment_class"):
            print(f"Augmentación focalizada: {summary['augment_class']} x{summary['augment_factor']}")
        print(f"Balanceo a máximo: {summary['balance_to_max']}")
        print(f"Pesos por clase: {summary['use_class_weights']}")
        print(f"Protótipos por clase: {summary['prototypes_per_class']}")
        print("-" * 60)
        print(f"Train Accuracy:     {summary['train_accuracy']:.4f}")
        print(f"Train Bal. Acc.:    {summary['train_balanced_accuracy']:.4f}")
        print(f"Train Top-3 Acc.:   {summary['train_top_3_accuracy']:.4f}")
        print(f"Accuracy:           {summary['accuracy']:.4f}")
        print(f"Balanced Accuracy:  {summary['balanced_accuracy']:.4f}")
        print(f"Precision macro:    {summary['precision_macro']:.4f}")
        print(f"Precision weighted: {summary['precision_weighted']:.4f}")
        print(f"Recall macro:       {summary['recall_macro']:.4f}")
        print(f"Recall weighted:    {summary['recall_weighted']:.4f}")
        print(f"F1 macro:           {summary['f1_macro']:.4f}")
        print(f"F1 weighted:        {summary['f1_weighted']:.4f}")
        print(f"Top-3 Accuracy:     {summary['top_3_accuracy']:.4f}")

        for key, value in summary.items():
            if key.startswith("roc_auc"):
                value_str = "N/A" if value is None else f"{value:.4f}"
                print(f"{key}: {value_str}")

        print("-" * 60)
        print(f"Archivos generados en: {self.output_dir}")

    def ensure_output_dir_name(self):
        """
        Si el directorio de salida no fue personalizado, fuerza el nombre estándar.
        """
        default_path = self.resolve_project_path(DEFAULT_OUTPUT_DIR)
        if self.output_dir == self.resolve_project_path("data/metricas_modelo_estatico"):
            self.output_dir = default_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evalúa rigurosamente el modelo estático de ASL con métricas y matrices de confusión."
    )
    parser.add_argument(
        "--csv",
        default="data/hand_landmarks_dataset_cleaned.csv",
        help="Ruta al CSV de landmarks (limpio) del modelo estático.",
    )
    parser.add_argument(
        "--raw-csv",
        default="data/hand_landmarks_dataset_corrected.csv",
        help="CSV origen para generar el CSV limpio si no existe.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Mínimo de muestras por clase para mantener la clase.",
    )
    parser.add_argument(
        "--outlier-multiplier",
        type=float,
        default=1.5,
        help="Multiplicador IQR para detección de outliers.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Si se pasa, normaliza columnas de landmarks en el CSV limpio.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio donde se guardarán las métricas y matrices.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción del dataset usada para prueba.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para hacer reproducible la partición train/test.",
    )
    parser.add_argument(
        "--augment-class",
        default="m",
        help="Clase a reforzar con augmentación focalizada. Usa vacío para desactivar.",
    )
    parser.add_argument(
        "--augment-factor",
        type=int,
        default=4,
        help="Cantidad de muestras sintéticas extra por muestra original de la clase objetivo.",
    )
    parser.add_argument(
        "--augment-noise-std",
        type=float,
        default=0.008,
        help="Desviación estándar del ruido gaussiano aplicado a los landmarks durante augmentación.",
    )
    parser.add_argument(
        "--balance-to-max",
        action="store_true",
        help="Iguala todas las clases al tamaño de la clase mayor mediante augmentación.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Desactiva los pesos por clase al construir prototipos.",
    )
    parser.add_argument(
        "--prototypes-per-class",
        type=int,
        default=3,
        help="Número máximo de centroides por clase para capturar variaciones intra-clase.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    # Si el CSV limpio no existe, generarlo a partir del CSV corregido
    project_root = Path(__file__).resolve().parents[2]

    def resolve_main_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == project_root.name:
            return project_root.parent / path
        return project_root / path

    csv_path = resolve_main_path(args.csv)
    raw_csv_path = resolve_main_path(args.raw_csv)

    if not csv_path.exists():
        print(f"CSV limpio {csv_path} no encontrado. Generando desde {raw_csv_path}...")
        ensure_cleaned_dataset(
            raw_csv=str(raw_csv_path),
            output_csv=str(csv_path),
            min_count=args.min_count,
            outlier_multiplier=args.outlier_multiplier,
            normalize=args.normalize,
        )

    evaluator = StaticModelMetrics(
        csv_path=str(csv_path),
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        augment_class=args.augment_class.strip() or None,
        augment_factor=args.augment_factor,
        augment_noise_std=args.augment_noise_std,
        balance_to_max=args.balance_to_max,
        use_class_weights=not args.no_class_weights,
        prototypes_per_class=args.prototypes_per_class,
    )
    evaluator.ensure_output_dir_name()
    results = evaluator.evaluate()
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
