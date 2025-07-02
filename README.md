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

-   `hand_vectorizer_final.py`: **(Script Principal)**. Este es el script que se debe ejecutar para iniciar el sistema. La primera vez, procesa el conjunto de imágenes para crear un archivo de características (`reference_features_optimized.pkl`). En ejecuciones posteriores, carga este archivo para un inicio rápido.

-   `add_new_dataset.py`: Herramienta para ampliar el conjunto de datos. Permite combinar un dataset existente con uno nuevo para mejorar la precisión del modelo.

-   `hand_vectorizer.py`: Contiene la clase base para la extracción de características. Procesa las imágenes del dataset y guarda los puntos de referencia en un archivo CSV. Es la base sobre la que opera la versión final.

-   `hand_landmarks_dataset_corrected.csv`: La base de datos que contiene los puntos de referencia (landmarks) extraídos de cada imagen del dataset, junto a su clasificación.

-   `requirements.txt`: Define las dependencias de Python necesarias para ejecutar el proyecto.

-   `kagglehub/asl_dataset/`: Contiene el conjunto de datos de imágenes, organizado en subcarpetas por cada clase (letra o número).

## Guía de Inicio

### 1. Instalación de Dependencias

Para comenzar, es necesario instalar las dependencias del proyecto. Ejecute el siguiente comando en la terminal desde la raíz del proyecto:

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el Sistema de Reconocimiento

Para iniciar la aplicación, ejecute el script principal:

```bash
python hand_vectorizer_final.py
```

**Nota**: La primera ejecución tardará unos minutos mientras se procesa y se aprende del conjunto de datos. Tras este paso inicial, se activará la cámara. Al mostrar una seña, la clasificación aparecerá en la consola. Para detener la ejecución, presione la tecla `q`.

### 3. Ampliación del Conjunto de Datos (DESAROLLO)

Este paso es fundamental para mejorar progresivamente la precisión del modelo. Para añadir nuevas imágenes, el flujo de trabajo es el siguiente:

1.  **Preparar los datos**: Cree una carpeta con la misma estructura que `kagglehub/asl_dataset/`, utilizando subcarpetas para cada clase (`a`, `b`, `1`, `2`, etc.) y coloque ahí las nuevas imágenes.
2.  **Ejecutar el script de agregado**:

    ```bash
    python add_new_dataset.py
    ```

3.  **Seguir las instrucciones**: El script solicitará la ruta a la carpeta con los nuevos datos.
4.  **Forzar el re-entrenamiento**: Una vez finalizado el script, es necesario eliminar el archivo de características precalculadas para que el sistema lo regenere con el dataset ampliado.

    ```bash
    # En Windows
    del reference_features_optimized.pkl

    # En macOS / Linux
    rm reference_features_optimized.pkl
    ```

5.  **Verificar los cambios**: La próxima vez que se ejecute `hand_vectorizer_final.py`, el sistema se re-entrenará con el conjunto de datos actualizado, lo que debería mejorar su rendimiento. # ASL-System
