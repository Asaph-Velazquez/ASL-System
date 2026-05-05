from pathlib import Path
import json
import pandas as pd
import numpy as np


VALID_CLASSES = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]


def get_landmark_columns(num_landmarks: int = 21) -> list[str]:
    cols = []
    for i in range(num_landmarks):
        cols.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])
    return cols


def ensure_cleaned_dataset(
    raw_csv: str,
    output_csv: str,
    min_count: int = 2,
    outlier_multiplier: float = 1.5,
    normalize: bool = False,
):
    project_root = Path(__file__).resolve().parents[2]
    raw_path = Path(raw_csv)
    out_path = Path(output_csv)

    def resolve_path(path: Path) -> Path:
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == project_root.name:
            return project_root.parent / path
        return project_root / path

    raw_path = resolve_path(raw_path)
    out_path = resolve_path(out_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"CSV de origen no encontrado: {raw_path}")

    df = pd.read_csv(raw_path)
    # normalizar labels
    if "class" not in df.columns:
        raise ValueError("CSV no contiene la columna 'class'.")
    df["class"] = df["class"].astype(str).str.lower()
    df = df[df["class"].isin(VALID_CLASSES)].copy()

    landmark_cols = get_landmark_columns()
    missing_cols = [c for c in landmark_cols if c not in df.columns]
    if missing_cols:
        raise ValueError("Faltan columnas de landmarks en el CSV: " + ", ".join(missing_cols[:10]))

    # convertir a float y contar NaNs
    df[landmark_cols] = df[landmark_cols].astype(float)
    n_missing = int(df[landmark_cols].isna().any(axis=1).sum())
    if n_missing:
        print(f"⚠️  Eliminando {n_missing} filas con valores faltantes en landmarks.")
        df = df.dropna(subset=landmark_cols).copy()

    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        print(f"⚠️  Eliminando {n_dupes} filas duplicadas.")
        df = df.drop_duplicates().copy()

    # detección de outliers por IQR (por clase) — sólo eliminar si la clase mantiene >= min_count
    try:
        df_out = df.copy()
        to_drop_idx = []
        for cls, group in df.groupby("class"):
            q1 = group[landmark_cols].quantile(0.25)
            q3 = group[landmark_cols].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - outlier_multiplier * iqr
            upper = q3 + outlier_multiplier * iqr

            mask_outlier = ((group[landmark_cols] < lower) | (group[landmark_cols] > upper)).any(axis=1)
            n_outliers_cls = int(mask_outlier.sum())
            remaining = len(group) - n_outliers_cls

            # Sólo eliminar outliers si la clase mantendrá al menos min_count muestras
            if n_outliers_cls > 0 and remaining >= min_count:
                to_drop_idx.extend(group[mask_outlier].index.tolist())

        n_outliers = len(to_drop_idx)
        if n_outliers:
            print(f"⚠️  Eliminando {n_outliers} filas consideradas outliers (IQR multiplier={outlier_multiplier}) por clase.")
            df = df.drop(index=to_drop_idx).copy()
    except Exception:
        print("⚠️  Error durante detección de outliers: se omite paso de outliers.")

    # remover clases con pocas muestras (ahora después de eliminar outliers)
    counts = df["class"].value_counts()
    low = counts[counts < min_count]
    if not low.empty:
        print(
            "⚠️  Eliminando clases con menos de {0} muestras: {1}".format(
                min_count, ", ".join(f"{lab} ({c})" for lab, c in low.items())
            )
        )
        df = df[df["class"].isin(counts[counts >= min_count].index)].copy()

    if df.empty:
        raise ValueError("No quedan muestras válidas después de la limpieza.")

    # normalizar por columna (opcional) — median / MAD
    if normalize:
        med = df[landmark_cols].median()
        mad = (df[landmark_cols].subtract(med)).abs().median()
        mad_repl = mad.replace(0, mad.mean() if mad.mean() != 0 else 1.0)
        df[landmark_cols] = (df[landmark_cols] - med) / mad_repl
        df[landmark_cols] = df[landmark_cols].clip(-3, 3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    summary = {
        "num_samples": int(len(df)),
        "num_classes": int(df["class"].nunique()),
    }
    print(f"✔️  CSV limpio guardado en: {out_path} — muestras: {summary['num_samples']}, clases: {summary['num_classes']}")
    # also write a small json summary
    try:
        with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return out_path


if __name__ == "__main__":
    # CLI light
    import argparse

    parser = argparse.ArgumentParser(description="Limpia el dataset de landmarks y lo guarda.")
    parser.add_argument("--raw-csv", default="data/hand_landmarks_dataset_corrected.csv")
    parser.add_argument("--output-csv", default="data/hand_landmarks_dataset_cleaned.csv")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--outlier-multiplier", type=float, default=1.5)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    ensure_cleaned_dataset(
        raw_csv=args.raw_csv,
        output_csv=args.output_csv,
        min_count=args.min_count,
        outlier_multiplier=args.outlier_multiplier,
        normalize=args.normalize,
    )
