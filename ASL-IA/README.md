# Guía del Proyecto: Reconocimiento de Señas ASL

Hola equipo,

Este documento sirve como guía central para nuestro proyecto de reconocimiento de señas en Lengua de Signos Americana (ASL). El objetivo es utilizar la visión por computadora para interpretar las letras y números del ASL en tiempo real. Para ello, nos apoyamos en la biblioteca `MediaPipe` para la detección de puntos de referencia de la mano, y en un algoritmo propio para la clasificación de los gestos.

## Funcionalidades Actuales

-   **Reconocimiento en Tiempo Real**: El sistema utiliza la cámara para capturar y reconocer señas de forma instantánea.
-   **Normalización Robusta (En desarrollo)**: El algoritmo de reconocimiento es invariante a la escala y a la distancia de la mano a la cámara, lo que mejora considerablemente la precisión.
-   **Arquitectura Modular**: El código está estructurado en módulos para facilitar su comprensión, mantenimiento y la colaboración en equipo.
-   **Extensible**: El sistema está diseñado para que sea sencillo añadir nuevas imágenes y así mejorar continuamente el modelo de reconocimiento.

## Estructura del Proyecto

A continuación, se detalla la función de cada componente principal:

-   `src/Modelo Estatico/hand_vectorizer_final.py`: **(Script Principal)**. Este es el script que se debe ejecutar para iniciar el sistema. La primera vez, procesa el conjunto de imágenes para crear un archivo de características (`reference_features_optimized.pkl`). En ejecuciones posteriores, carga este archivo para un inicio rápido.

-   `src/Modelo Estatico/add_new_dataset.py`: Herramienta para ampliar el conjunto de datos. Permite combinar un dataset existente con uno nuevo para mejorar la precisión del modelo.

-   `src/Modelo Estatico/hand_vectorizer.py`: Contiene la clase base para la extracción de características. Procesa las imágenes del dataset y guarda los puntos de referencia en un archivo CSV. Es la base sobre la que opera la versión final.

-   `hand_landmarks_dataset_corrected.csv`: La base de datos que contiene los puntos de referencia (landmarks) extraídos de cada imagen del dataset, junto a su clasificación.

-   `requirements.txt`: Define las dependencias de Python necesarias para ejecutar el proyecto.

-   `kagglehub/asl_dataset/`: Contiene el conjunto de datos de imágenes, organizado en subcarpetas por cada clase (letra o número).
-   `data/Dinamico/`: Dataset de videos por palabra usado como entrada del pipeline dinámico.
-   `src/Modelo Dinamico/`: Módulo aislado del modelo dinámico, con script, dependencias, documentación, artefactos y modelos.

## Pipeline Dinámico

El módulo dinámico es independiente del reconocedor estático actual. Vive dentro de `src/Modelo Dinamico/`, toma como entrada los videos `.mp4` de `data/Dinamico/` y guarda sus artefactos dentro de su propia carpeta.

### Artefactos generados

-   `src/Modelo Dinamico/artifacts/dynamic_audit.csv`: Resultado detallado de la auditoría por video.
-   `src/Modelo Dinamico/artifacts/dynamic_audit_summary.json`: Resumen agregado de calidad del dataset.
-   `src/Modelo Dinamico/artifacts/dynamic_subset_manifest.csv`: Videos válidos que sobreviven al filtro del PoC.
-   `src/Modelo Dinamico/artifacts/dynamic_sequences/`: Secuencias serializadas de landmarks por muestra.
-   `src/Modelo Dinamico/artifacts/dynamic_splits.json`: Partición reproducible `train/val/test`.
-   `src/Modelo Dinamico/artifacts/dynamic_label_map.json`: Mapeo clase-id usado en entrenamiento.
-   `src/Modelo Dinamico/artifacts/dynamic_model_metadata.json`: Configuración del modelo y metadatos de entrenamiento.
-   `src/Modelo Dinamico/artifacts/dynamic_evaluation.json`: Métricas y matriz de confusión del modelo dinámico.
-   `src/Modelo Dinamico/models/dynamic_baseline.keras`: Checkpoint del mejor baseline temporal.

### Flujo recomendado

Desde `ASL-IA`, ejecutar:

```bash
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" audit
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" filter --min-valid-videos 5
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" extract --target-frames 32
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" split --seed 42
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" train --target-frames 32 --epochs 10 --batch-size 16
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" evaluate --target-frames 32
```

También existe un comando integrado:

```bash
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" pipeline --target-frames 32 --min-valid-videos 5 --epochs 10
```

### Decisiones técnicas del PoC

-   Features por frame: landmarks 3D de hasta 2 manos (`126` features base).
-   Normalización por frame: centrado en la muñeca y reescalado por tamaño de mano.
-   Secuencia fija: remuestreo a `32` frames por muestra.
-   Criterio de estabilidad: clases con al menos `5` videos válidos tras la auditoría.
-   Baseline: `Masking -> BiLSTM(128) -> Dropout -> BiLSTM(64) -> Dense -> Softmax`.
-   Métrica principal: `macro F1`.

### Nota de entorno

El entrenamiento y la evaluación del pipeline dinámico requieren `TensorFlow`. Instálalo con el archivo `src/Modelo Dinamico/requirements.txt`. Si no está instalado, los comandos `train` y `evaluate` fallarán con un mensaje explícito, mientras que `audit`, `filter`, `extract` y `split` seguirán funcionando.

## Guía de Inicio

### 1. Instalación de Dependencias

Para comenzar, es necesario instalar las dependencias del proyecto. Ejecute el siguiente comando en la terminal desde la raíz del proyecto:

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el Sistema de Reconocimiento

Para iniciar la aplicación, ejecute el script principal:

```bash
python "src/Modelo Estatico/hand_vectorizer_final.py"
```

**Nota**: La primera ejecución tardará unos minutos mientras se procesa y se aprende del conjunto de datos. Tras este paso inicial, se activará la cámara. Al mostrar una seña, la clasificación aparecerá en la consola. Para detener la ejecución, presione la tecla `q`.

### 3. Ampliación del Conjunto de Datos (DESAROLLO)

Este paso es fundamental para mejorar progresivamente la precisión del modelo. Para añadir nuevas imágenes, el flujo de trabajo es el siguiente:

1.  **Preparar los datos**: Cree una carpeta con la misma estructura que `kagglehub/asl_dataset/`, utilizando subcarpetas para cada clase (`a`, `b`, `1`, `2`, etc.) y coloque ahí las nuevas imágenes.
2.  **Ejecutar el script de agregado**:

    ```bash
python "src/Modelo Estatico/add_new_dataset.py"
    ```

3.  **Seguir las instrucciones**: El script solicitará la ruta a la carpeta con los nuevos datos.
4.  **Forzar el re-entrenamiento**: Una vez finalizado el script, es necesario eliminar el archivo de características precalculadas para que el sistema lo regenere con el dataset ampliado.

    ```bash
    # En Windows
    del reference_features_optimized.pkl

    # En macOS / Linux
    rm reference_features_optimized.pkl
    ```

5.  **Verificar los cambios**: La próxima vez que se ejecute `src/Modelo Estatico/hand_vectorizer_final.py`, el sistema se re-entrenará con el conjunto de datos actualizado, lo que debería mejorar su rendimiento. # ASL-System
