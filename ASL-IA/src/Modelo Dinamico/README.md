# Modelo Dinamico

Este módulo contiene todo lo relacionado con el reconocimiento dinámico ASL dentro de `ASL-IA`.

## Contenido

- `dynamic_asl_pipeline.py`: CLI principal del pipeline.
- `artifacts/`: auditorías, manifiestos, splits, secuencias y reportes.
- `models/`: checkpoints entrenados del baseline.

## Uso

Desde la raíz `ASL-IA`:

```bash
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" audit
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" filter --min-valid-videos 5
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" extract --target-frames 32
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" split --seed 42
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" train --target-frames 32 --epochs 10 --batch-size 16
python "src/Modelo Dinamico/dynamic_asl_pipeline.py" evaluate --target-frames 32
```

## Nota

El dataset fuente sigue en `data/Dinamico/`. Los artefactos de procesamiento y los modelos generados se guardan en esta carpeta del módulo.