# Modelo Estático ASL

Este módulo contiene el flujo del modelo estático basado en landmarks de mano:

- `clean_dataset.py`: limpia y normaliza el CSV de landmarks.
- `Metricas.py`: evalúa el modelo estático y genera métricas, matrices de confusión y gráficos.
- `hand_vectorizer.py`: construye las features de 68 dimensiones a partir de landmarks.
- `hand_vectorizer_final.py`: script principal para ejecutar el reconocedor estático.
- `process_alphabet_cache.py`: cachea los landmarks del dataset 'alphabet' (195k imágenes) para usarlos en entrenamiento.

## Preparación

Desde la raíz del workspace, activa el entorno virtual y entra al proyecto:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
```

## Caché del dataset 'alphabet'

Si tienes disponible la carpeta `data/kagglehub/alphabet/` con +195k imágenes, puedes cachear los landmarks para acelerar entrenamientos posteriores:

```powershell
python "ASL-IA/src/Modelo Estatico/process_alphabet_cache.py"
```

Esto generará un archivo pickle (`data/kagglehub/alphabet_landmarks_cache.pkl`) que será usado automáticamente por `Metricas.py`. El caché se crea solo una vez y acelera significativamente el retrenamiento.

## Limpieza del dataset

Genera el CSV limpio a partir del CSV corregido original, si lo tienes disponible:

```powershell
python "ASL-IA/src/Modelo Estatico/clean_dataset.py" `
  --raw-csv "ASL-IA/data/hand_landmarks_dataset_corrected.csv" `
  --output-csv "ASL-IA/data/hand_landmarks_dataset_cleaned.csv"
```

Si en tu workspace solo existe la copia ya limpia generada durante las pruebas, puedes usarla como entrada temporal para reconstruir el archivo en la ruta correcta:

```powershell
python "ASL-IA/src/Modelo Estatico/clean_dataset.py" `
  --raw-csv "ASL-IA/ASL-IA/data/hand_landmarks_dataset_cleaned.csv" `
  --output-csv "ASL-IA/data/hand_landmarks_dataset_cleaned.csv"
```

## Métricas del modelo

Ejecuta la evaluación sobre el CSV limpio. Por defecto, las salidas se guardan en `ASL-IA/data/Metricas_Estaticas`. Si existe el caché de `alphabet`, se integrará automáticamente:

```powershell
python "ASL-IA/src/Modelo Estatico/Metricas.py" `
  --csv "ASL-IA/data/hand_landmarks_dataset_cleaned.csv" `
  --test-size 0.2 `
  --random-state 42
```

Si quieres reforzar la clase `m` durante el retrenamiento, puedes usar:

```powershell
python "ASL-IA/src/Modelo Estatico/Metricas.py" `
  --csv "ASL-IA/data/hand_landmarks_dataset_cleaned.csv" `
  --test-size 0.2 `
  --random-state 42 `
  --augment-class m `
  --augment-factor 2 `
  --augment-noise-std 0.008
```

Los artefactos se guardan en:

- `ASL-IA/data/Metricas_Estaticas/metricas_resumen.json`
- `ASL-IA/data/Metricas_Estaticas/metricas_por_clase.csv`
- `ASL-IA/data/Metricas_Estaticas/matriz_confusion.csv`
- `ASL-IA/data/Metricas_Estaticas/matriz_confusion_normalizada.csv`
- `ASL-IA/data/Metricas_Estaticas/predicciones_detalladas.csv`
- `ASL-IA/data/Metricas_Estaticas/*.png`

## Ejecución del modelo

Inicia el reconocedor estático con la versión final:

```powershell
python "ASL-IA/src/Modelo Estatico/hand_vectorizer_final.py"
```

## Notas

- El modelo usa features de 68 dimensiones por muestra: 63 coordenadas de 21 landmarks más 5 distancias desde la muñeca a las puntas de los dedos.
- Si cambias el dataset limpio, vuelve a ejecutar `Metricas.py` para regenerar resultados.
- Si eliminas `ASL-IA/data/reference_features_optimized.pkl`, el script principal volverá a construir las referencias al iniciar.
- El caché de `alphabet` (`alphabet_landmarks_cache.pkl`) se crea una sola vez. Si necesitas regenerarlo, elimina el archivo y ejecuta `process_alphabet_cache.py` nuevamente.
