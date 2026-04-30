import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


class StaticModelMetrics:
    def __init__(self, csv_path: str, output_dir: str, test_size: float, random_state: int):
        self.project_root = Path(__file__).resolve().parents[2]
        self.csv_path = self.resolve_project_path(csv_path)
        self.output_dir = self.resolve_project_path(output_dir)
        self.test_size = test_size
        self.random_state = random_state

    def resolve_project_path(self, path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else self.project_root / path

    def load_dataset(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo CSV: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        df["class"] = df["class"].astype(str).str.lower()
        df = df[df["class"].isin(VALID_CLASSES)].copy()

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
        distances = [np.linalg.norm(wrist - landmarks[idx]) for idx in fingertip_indices]
        features.extend(distances)

        return np.asarray(features[:68], dtype=float)

    def build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.vstack([self.add_distance_features(row) for _, row in df.iterrows()])

    def split_dataset(self, X: np.ndarray, y: np.ndarray):
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        train_idx, test_idx = next(splitter.split(X, y))
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx

    def build_prototypes(self, X_train: np.ndarray, y_train: np.ndarray):
        prototypes = {}
        for class_name in sorted(np.unique(y_train)):
            class_features = X_train[y_train == class_name]
            prototypes[class_name] = np.mean(class_features, axis=0)
        return prototypes

    def score_sample(self, sample: np.ndarray, prototypes: dict[str, np.ndarray]):
        sample_normalized = self.normalize_features(sample)
        scores = {}

        for class_name, prototype in prototypes.items():
            prototype_normalized = self.normalize_features(prototype)

            sample_norm = np.linalg.norm(sample_normalized)
            prototype_norm = np.linalg.norm(prototype_normalized)
            cosine_similarity = 0.0
            if sample_norm != 0 and prototype_norm != 0:
                cosine_similarity = float(
                    np.dot(sample_normalized, prototype_normalized) / (sample_norm * prototype_norm)
                )

            correlation = np.corrcoef(sample_normalized, prototype_normalized)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            euclidean_distance = float(
                np.linalg.norm(sample_normalized - prototype_normalized)
            )
            combined_score = (
                cosine_similarity * 0.4
                + correlation * 0.4
                + (1 - euclidean_distance / 10) * 0.2
            )
            scores[class_name] = combined_score

        ordered_labels = sorted(scores)
        raw_scores = np.array([scores[label] for label in ordered_labels], dtype=float)
        probabilities = self.softmax(raw_scores)
        predicted_label = ordered_labels[int(np.argmax(raw_scores))]

        return predicted_label, ordered_labels, raw_scores, probabilities

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores)

    def evaluate(self) -> dict:
        df = self.load_dataset()
        X = self.build_feature_matrix(df)
        y = df["class"].to_numpy()

        X_train, X_test, y_train, y_test, train_idx, test_idx = self.split_dataset(X, y)
        prototypes = self.build_prototypes(X_train, y_train)
        class_labels = sorted(prototypes.keys())

        y_pred = []
        y_scores = []
        top3_hits = 0

        for sample, true_label in zip(X_test, y_test):
            predicted_label, ordered_labels, _, probabilities = self.score_sample(sample, prototypes)
            y_pred.append(predicted_label)
            y_scores.append(probabilities)

            ranked_labels = [
                ordered_labels[idx]
                for idx in np.argsort(probabilities)[::-1][: min(3, len(ordered_labels))]
            ]
            if true_label in ranked_labels:
                top3_hits += 1

        y_pred = np.asarray(y_pred)
        y_scores = np.vstack(y_scores)

        y_test_bin = label_binarize(y_test, classes=class_labels)

        metrics_summary = {
            "num_samples_total": int(len(df)),
            "num_samples_train": int(len(y_train)),
            "num_samples_test": int(len(y_test)),
            "num_classes": int(len(class_labels)),
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
            "top_3_accuracy": float(top3_hits / len(y_test)),
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

        confusion = confusion_matrix(y_test, y_pred, labels=class_labels)
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

        prediction_df = pd.DataFrame(
            {
                "true_label": y_test,
                "predicted_label": y_pred,
                "is_correct": y_test == y_pred,
                "source_index": test_idx,
            }
        )

        source_columns = [col for col in ["image_path", "image_name"] if col in df.columns]
        if source_columns:
            source_df = df.iloc[test_idx][source_columns].reset_index(drop=True)
            prediction_df = pd.concat([prediction_df.reset_index(drop=True), source_df], axis=1)

        score_columns = pd.DataFrame(y_scores, columns=[f"score_{label}" for label in class_labels])
        prediction_df = pd.concat([prediction_df.reset_index(drop=True), score_columns], axis=1)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metricas_resumen.json").write_text(
            json.dumps(metrics_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report_df.to_csv(self.output_dir / "metricas_por_clase.csv", encoding="utf-8")
        confusion_df.to_csv(self.output_dir / "matriz_confusion.csv", encoding="utf-8")
        confusion_normalized_df.to_csv(
            self.output_dir / "matriz_confusion_normalizada.csv",
            encoding="utf-8",
        )
        prediction_df.to_csv(self.output_dir / "predicciones_detalladas.csv", index=False, encoding="utf-8")
        self.save_plots(metrics_summary, report_df, confusion_df, confusion_normalized_df)

        return {
            "summary": metrics_summary,
            "report_df": report_df,
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
        confusion_df: pd.DataFrame,
        confusion_normalized_df: pd.DataFrame,
    ):
        self.plot_confusion_matrix(
            confusion_df,
            self.output_dir / "matriz_confusion.png",
            "Matriz de Confusión",
            value_format="d",
        )
        self.plot_confusion_matrix(
            confusion_normalized_df,
            self.output_dir / "matriz_confusion_normalizada.png",
            "Matriz de Confusión Normalizada",
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
        print(f"Muestras entrenamiento: {summary['num_samples_train']}")
        print(f"Muestras prueba: {summary['num_samples_test']}")
        print(f"Clases evaluadas: {summary['num_classes']}")
        print("-" * 60)
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evalúa rigurosamente el modelo estático de ASL con métricas y matrices de confusión."
    )
    parser.add_argument(
        "--csv",
        default="data/hand_landmarks_dataset_corrected.csv",
        help="Ruta al CSV de landmarks del modelo estático.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/metricas_modelo_estatico",
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
    return parser


def main():
    args = build_parser().parse_args()
    evaluator = StaticModelMetrics(
        csv_path=args.csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    results = evaluator.evaluate()
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()
