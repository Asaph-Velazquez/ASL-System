#!/usr/bin/env python
"""
Script para cachear los landmarks de 'alphabet' con procesamiento paralelo.

Ejecutar una sola vez:
    python "ASL-IA/src/Modelo Estatico/process_alphabet_cache.py"

Esto generará un archivo pickle en:
    data/kagglehub/alphabet_landmarks_cache.pkl

que luego será usado automáticamente por Metricas.py durante el entrenamiento.
"""

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd


def extract_landmarks_from_image(image_path: str) -> dict | None:
    """Extrae landmarks de una imagen individual."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            landmarks = results.multi_hand_landmarks[0]
            row = {}

            for i, lm in enumerate(landmarks.landmark):
                row[f"landmark_{i}_x"] = lm.x
                row[f"landmark_{i}_y"] = lm.y
                row[f"landmark_{i}_z"] = lm.z

            hands.close()
            return row
        
        hands.close()
        return None

    except Exception as e:
        return None


def process_alphabet_cache(alphabet_path: Path, num_workers: int = 4) -> pd.DataFrame:
    """Procesa todas las imágenes de 'alphabet' con paralelismo."""
    print(f"🔄 Procesando dataset 'alphabet' con {num_workers} workers...")

    rows = []
    letter_dirs = sorted(
        [d for d in alphabet_path.iterdir() if d.is_dir()],
        key=lambda x: x.name
    )

    total_images = sum(
        len(list(d.glob("*.jpg")) + list(d.glob("*.png")))
        for d in letter_dirs
    )
    print(f"📊 Total de imágenes a procesar: {total_images}")

    processed_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Enviar todas las tareas
        future_to_info = {}
        for letter_dir in letter_dirs:
            class_label = letter_dir.name.lower()
            img_files = list(letter_dir.glob("*.jpg")) + list(letter_dir.glob("*.png"))

            for img_file in img_files:
                future = executor.submit(extract_landmarks_from_image, str(img_file))
                future_to_info[future] = (class_label, img_file.name)

        # Procesar resultados conforme se completen
        for future in as_completed(future_to_info):
            class_label, img_name = future_to_info[future]
            try:
                landmarks = future.result()
                if landmarks is not None:
                    landmarks["class"] = class_label
                    rows.append(landmarks)
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1

            # Mostrar progreso cada 1000 imágenes
            if (processed_count + failed_count) % 1000 == 0:
                progress = processed_count + failed_count
                print(
                    f"   Progreso: {progress}/{total_images} "
                    f"({processed_count} exitosas, {failed_count} fallidas)"
                )

    print(
        f"✅ Procesamiento completado: {processed_count} imágenes exitosas, "
        f"{failed_count} fallidas"
    )

    if rows:
        df = pd.DataFrame(rows)
        return df
    else:
        print("⚠️  No se extrajeron landmarks válidos de 'alphabet'.")
        return None


def main():
    """Punto de entrada principal."""
    project_root = Path(__file__).resolve().parents[2]
    alphabet_path = project_root / "data" / "kagglehub" / "alphabet"
    cache_file = project_root / "data" / "kagglehub" / "alphabet_landmarks_cache.pkl"

    if not alphabet_path.exists():
        print(f"❌ Carpeta 'alphabet' no encontrada en: {alphabet_path}")
        sys.exit(1)

    print(f"📂 Procesando: {alphabet_path}")
    print(f"💾 Caché se guardará en: {cache_file}")
    print()

    # Procesar con 4 workers (ajusta según tu CPU)
    df_alphabet = process_alphabet_cache(alphabet_path, num_workers=4)

    if df_alphabet is not None and not df_alphabet.empty:
        # Guardar en caché
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(df_alphabet, f)
            print(f"\n✅ Caché guardado exitosamente: {cache_file}")
            print(f"   Total de filas: {len(df_alphabet)}")
            print(f"   Clases: {sorted(df_alphabet['class'].unique())}")
            print(f"\n📌 Ahora puedes usar `Metricas.py` normalmente, que cargará "
                  f"automáticamente este caché.")
        except Exception as e:
            print(f"❌ Error al guardar caché: {e}")
            sys.exit(1)
    else:
        print("❌ No se pudo procesar el dataset 'alphabet'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
