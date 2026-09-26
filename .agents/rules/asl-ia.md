# Reglas del modulo: ASL-IA

- Stack: Python con `opencv-python`, `mediapipe`, `numpy`, `pandas`.
- No rompas rutas relativas hacia `data/` o `kagglehub/asl_dataset/`.
- Si cambias procesamiento de landmarks o features, documenta impacto en:
  - `hand_landmarks_dataset_corrected.csv`
  - `reference_features_optimized.pkl`
- Valida al menos con `python -m py_compile <archivo>.py`.
- Si el flujo completo requiere camara o dataset pesado, deja claro que parte se
  pudo verificar localmente y cual no.
