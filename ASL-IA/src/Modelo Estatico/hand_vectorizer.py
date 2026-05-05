# =============================================================
# Archivo: hand_vectorizer.py
# Funcionalidad: Define la clase HandVectorizer para extraer y vectorizar características de las manos a partir de imágenes, utilizando MediaPipe y OpenCV. Permite procesar datasets de imágenes para reconocimiento de señas.
# =============================================================
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import mediapipe as mp
from typing import List, Dict, Tuple
import json
import copy

class HandVectorizer:
    def __init__(self, reference_dataset_path="data/kagglehub/asl_dataset", reference_csv_path="data/hand_landmarks_dataset_cleaned.csv"):
        """Inicializa el vectorizador de manos con MediaPipe"""
        self.project_root = Path(__file__).resolve().parents[2]
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1
        )
        self.reference_dataset_path = self.resolve_project_path(reference_dataset_path)
        self.reference_csv_path = self.resolve_project_path(reference_csv_path)
        # Cargar dataset de referencia al inicializar
        self.reference_data = None
        self.reference_landmarks, self.reference_labels = self.load_reference_dataset()

    def resolve_project_path(self, path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else self.project_root / path
        
    def load_reference_dataset(self, dataset_path="data/kagglehub/asl_dataset"):
        """Carga el dataset de referencia una sola vez"""
        dataset_path = self.resolve_project_path(dataset_path)
        # Soportar dataset adicional 'alphabet' dentro de data/kagglehub
        alphabet_path = self.project_root / "data" / "kagglehub" / "alphabet"
        cleaned_csv = self.resolve_project_path("data/hand_landmarks_dataset_cleaned.csv")
        corrected_csv = self.resolve_project_path("data/hand_landmarks_dataset_corrected.csv")
        fallback_csv = self.resolve_project_path("data/hand_landmarks_dataset.csv")
        # Intentar cargar el dataset corregido primero
        try:
            df = pd.read_csv(cleaned_csv)
            print("Dataset limpio cargado desde CSV")
        except FileNotFoundError:
            try:
                df = pd.read_csv(corrected_csv)
                print("Dataset corregido cargado desde CSV")
            except FileNotFoundError:
                try:
                    df = pd.read_csv(fallback_csv)
                    print("Dataset cargado desde CSV")
                except FileNotFoundError:
                    print("Dataset no encontrado. Procesando dataset...")
                    # Si existe la carpeta 'alphabet', procesarla también y concatenar
                    if alphabet_path.exists() and alphabet_path.is_dir():
                        print(f"Se detectó dataset adicional: {alphabet_path}")
                        df1 = self.process_dataset(dataset_path)
                        df2 = self.process_dataset(str(alphabet_path))
                        if df1 is None:
                            df = df2
                        elif df2 is None:
                            df = df1
                        else:
                            df = pd.concat([df1, df2], ignore_index=True)
                    else:
                        df = self.process_dataset(dataset_path, fallback_csv)
        
        if not df.empty:
            # Crear diccionario con características promedio por clase
            self.reference_data = {}
            valid_classes = 0
            
            for class_name in df['class'].unique():
                # Normalizar nombre de clase a minúscula (las carpetas externas pueden usar mayúsculas)
                class_name = str(class_name).lower()
                # Solo procesar clases válidas (letras a-z y números 0-9)
                if class_name in [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]:
                    class_data = df[df['class'].str.lower() == class_name]

                    # Generar características reales por fila y promediarlas
                    class_features = []
                    for _, row in class_data.iterrows():
                        row_features = self.get_generated_landmark_features(row)
                        if row_features is not None:
                            class_features.append(row_features)

                    if not class_features:
                        continue

                    avg_features = np.mean(class_features, axis=0)
                    self.reference_data[class_name] = np.asarray(avg_features, dtype=float)
                    valid_classes += 1
            
            print(f"Dataset de referencia cargado con {valid_classes} clases válidas")
            print(f"Clases disponibles: {sorted(self.reference_data.keys())}")
            
            # Verificar que todas las clases tengan 68 características
            for class_name, features in self.reference_data.items():
                if len(features) != 68:
                    print(f"⚠️  ADVERTENCIA: Clase '{class_name}' tiene {len(features)} características en lugar de 68")
            
            # Devolver los datos de referencia
            landmarks = list(self.reference_data.values())
            labels = list(self.reference_data.keys())
            return landmarks, labels
            
        else:
            print("No se pudo cargar el dataset de referencia")
            return None, None

    def get_generated_landmark_features(self, row: pd.Series) -> np.ndarray:
        """
        Construye un vector de 68 características a partir de una fila del CSV
        usando landmarks reales y distancias desde la muñeca a las puntas.
        """
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

        if len(features) < 68:
            features.extend([0.0] * (68 - len(features)))
        elif len(features) > 68:
            features = features[:68]

        return np.asarray(features, dtype=float)

    def preprocess_image(self, image):
        """
        Preprocesa la imagen para mejorar la detección de manos
        """
        # Convertir a RGB si es necesario
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Aumentar contraste
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def extract_hand_landmarks(self, image_path: str) -> Dict:
        """
        Extrae los puntos clave de la mano de una imagen
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Diccionario con los landmarks de la mano y metadatos
        """
        # Leer la imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            return None
        
        # Intentar diferentes preprocesamientos si no se detecta mano
        preprocessing_methods = [
            lambda img: img,  # Sin preprocesamiento
            lambda img: self.preprocess_image(img),  # Con contraste mejorado
            lambda img: cv2.resize(img, (img.shape[1]*2, img.shape[0]*2)),  # Escalar 2x
            lambda img: cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))  # Escalar 0.5x
        ]
        
        for i, preprocess_func in enumerate(preprocessing_methods):
            try:
                processed_image = preprocess_func(image)
                
                # Convertir BGR a RGB
                if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = processed_image
                
                # Procesar la imagen con MediaPipe
                results = self.hands.process(image_rgb)
                
                landmarks_data = {
                    'image_path': image_path,
                    'landmarks': [],
                    'hand_landmarks': None,
                    'hand_landmarks_3d': None,
                    'hand_world_landmarks': None,
                    'preprocessing_method': i
                }
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Extraer coordenadas 2D
                    landmarks_2d = []
                    for landmark in hand_landmarks.landmark:
                        landmarks_2d.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    landmarks_data['landmarks'] = landmarks_2d
                    landmarks_data['hand_landmarks'] = hand_landmarks
                    
                    if results.multi_hand_world_landmarks:
                        hand_world_landmarks = results.multi_hand_world_landmarks[0]
                        landmarks_data['hand_world_landmarks'] = hand_world_landmarks
                        
                        landmarks_3d = []
                        for landmark in hand_world_landmarks.landmark:
                            landmarks_3d.append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z
                            })
                        landmarks_data['hand_landmarks_3d'] = landmarks_3d
                    
                    return landmarks_data
                    
            except Exception as e:
                continue
        
        print(f"No se detectaron manos en: {image_path} (probados {len(preprocessing_methods)} métodos)")
        return None
    
    def process_dataset(self, dataset_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Procesa todo el dataset de imágenes y extrae los landmarks
        
        Args:
            dataset_path: Ruta al directorio del dataset
            output_path: Ruta para guardar el CSV con los resultados
            
        Returns:
            DataFrame con todos los landmarks extraídos
        """
        dataset_path = self.resolve_project_path(dataset_path)
        all_landmarks = []
        
        # Recorrer todas las carpetas de letras/números
        for class_folder in dataset_path.iterdir():
            if class_folder.is_dir():
                # Normalizar nombre de carpeta a minúsculas para unir A/B y a/b
                class_name = class_folder.name.lower()
                
                # Solo procesar clases válidas (letras a-z y números 0-9)
                if class_name in [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]:
                    print(f"Procesando clase: {class_name}")
                    
                    # Procesar todas las imágenes en la carpeta
                    for image_file in class_folder.glob("*.jpeg"):
                        print(f"  Procesando: {image_file.name}")
                        
                        landmarks_data = self.extract_hand_landmarks(str(image_file))
                        
                        if landmarks_data and landmarks_data['landmarks']:
                            # Crear fila para el DataFrame
                            row_data = {
                                'class': class_name,
                                'image_path': str(image_file),
                                'image_name': image_file.name
                            }
                            
                            # Agregar coordenadas de cada landmark
                            for i, landmark in enumerate(landmarks_data['landmarks']):
                                row_data[f'landmark_{i}_x'] = landmark['x']
                                row_data[f'landmark_{i}_y'] = landmark['y']
                                row_data[f'landmark_{i}_z'] = landmark['z']
                            
                            all_landmarks.append(row_data)
                        else:
                            print(f"    No se detectaron manos en: {image_file.name}")
                else:
                    print(f"Ignorando carpeta no válida: {class_name}")
        
        # Crear DataFrame
        df = pd.DataFrame(all_landmarks)
        
        # Guardar resultados
        if output_path:
            output_path = self.resolve_project_path(output_path)
            df.to_csv(output_path, index=False)
            print(f"Resultados guardados en: {output_path}")
        
        return df
    
    def visualize_landmarks(self, image_path: str, save_path: str = None):
        """
        Visualiza los landmarks detectados en una imagen
        
        Args:
            image_path: Ruta a la imagen
            save_path: Ruta para guardar la imagen con landmarks
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Dibujar landmarks
            annotated_image = image_rgb.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
            
            # Convertir de vuelta a BGR para guardar
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            if save_path:
                cv2.imwrite(save_path, annotated_image_bgr)
                print(f"Imagen con landmarks guardada en: {save_path}")
            else:
                # Mostrar imagen
                cv2.imshow('Hand Landmarks', annotated_image_bgr)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    def get_landmark_features(self, landmarks_data: Dict) -> np.ndarray:
        """
        Extrae características numéricas de los landmarks
        Siempre retorna exactamente 68 características
        """
        if not landmarks_data or not landmarks_data['landmarks']:
            return None
        
        landmarks = landmarks_data['landmarks']
        features = []
        
        # Coordenadas de todos los landmarks (21 × 3 = 63)
        for landmark in landmarks:
            features.extend([landmark['x'], landmark['y'], landmark['z']])
        
        # Características adicionales (5 distancias)
        if len(landmarks) >= 21:  # MediaPipe detecta 21 landmarks
            # Distancias entre puntos clave
            wrist = np.array([landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']])
            thumb_tip = np.array([landmarks[4]['x'], landmarks[4]['y'], landmarks[4]['z']])
            index_tip = np.array([landmarks[8]['x'], landmarks[8]['y'], landmarks[8]['z']])
            middle_tip = np.array([landmarks[12]['x'], landmarks[12]['y'], landmarks[12]['z']])
            ring_tip = np.array([landmarks[16]['x'], landmarks[16]['y'], landmarks[16]['z']])
            pinky_tip = np.array([landmarks[20]['x'], landmarks[20]['y'], landmarks[20]['z']])
            
            # Distancias desde la muñeca a las puntas de los dedos
            distances = [
                np.linalg.norm(wrist - thumb_tip),
                np.linalg.norm(wrist - index_tip),
                np.linalg.norm(wrist - middle_tip),
                np.linalg.norm(wrist - ring_tip),
                np.linalg.norm(wrist - pinky_tip)
            ]
            
            features.extend(distances)
        else:
            # Si no hay 21 landmarks, rellenar con ceros
            features.extend([0.0] * 5)
        
        # Asegurar que siempre tengamos exactamente 68 características
        while len(features) < 68:
            features.append(0.0)
        
        if len(features) > 68:
            features = features[:68]
        
        return np.array(features)

    def mirror_landmarks_data(self, landmarks_data: Dict) -> Dict:
        """
        Genera una versión espejo horizontal de los landmarks detectados.
        Esto permite reconocer la misma seña con mano izquierda o derecha.
        """
        if not landmarks_data or not landmarks_data.get('landmarks'):
            return None

        mirrored_data = copy.deepcopy(landmarks_data)
        mirrored_landmarks = []
        for landmark in landmarks_data['landmarks']:
            mirrored_landmarks.append({
                'x': 1.0 - landmark['x'],
                'y': landmark['y'],
                'z': landmark['z']
            })

        mirrored_data['landmarks'] = mirrored_landmarks
        mirrored_data['mirrored'] = True
        return mirrored_data

    def score_features_against_reference(self, current_features: np.ndarray):
        """
        Compara un vector de características contra las referencias y devuelve
        el mejor match junto con el ranking completo.
        """
        current_features_normalized = self.normalize_features(current_features)

        best_match = None
        best_score = -1
        best_distance = float('inf')
        top_matches = []

        for class_name, ref_features in self.reference_data.items():
            if len(ref_features) == len(current_features):
                ref_features_normalized = self.normalize_features(ref_features)

                cosine_denominator = (
                    np.linalg.norm(current_features_normalized)
                    * np.linalg.norm(ref_features_normalized)
                )
                if cosine_denominator == 0:
                    cosine_similarity = 0
                else:
                    cosine_similarity = np.dot(
                        current_features_normalized,
                        ref_features_normalized
                    ) / cosine_denominator

                correlation = np.corrcoef(
                    current_features_normalized,
                    ref_features_normalized
                )[0, 1]
                if np.isnan(correlation):
                    correlation = 0

                euclidean_distance = np.linalg.norm(
                    current_features_normalized - ref_features_normalized
                )
                combined_score = (
                    cosine_similarity * 0.4
                    + correlation * 0.4
                    + (1 - euclidean_distance / 10) * 0.2
                )

                top_matches.append({
                    'class': class_name,
                    'cosine_similarity': cosine_similarity,
                    'correlation': correlation,
                    'euclidean_distance': euclidean_distance,
                    'combined_score': combined_score
                })

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = class_name
                    best_distance = euclidean_distance

        top_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        return best_match, best_score, best_distance, top_matches

    def get_gesture_description(self, class_name):
        """
        Obtiene una descripción amigable de la seña
        """
        descriptions = {
            # Números
            '0': 'CERO - Mano cerrada',
            '1': 'UNO - Dedo índice extendido',
            '2': 'DOS - Dedo índice y medio extendidos',
            '3': 'TRES - Tres dedos extendidos',
            '4': 'CUATRO - Cuatro dedos extendidos',
            '5': 'CINCO - Mano abierta',
            '6': 'SEIS - Pulgar y meñique extendidos',
            '7': 'SIETE - Pulgar, índice y medio extendidos',
            '8': 'OCHO - Pulgar e índice formando L',
            '9': 'NUEVE - Pulgar e índice formando círculo',
            
            # Letras
            'a': 'LETRA A - Puño cerrado',
            'b': 'LETRA B - Mano plana, dedos juntos',
            'c': 'LETRA C - Mano curvada como C',
            'd': 'LETRA D - Dedo índice y pulgar formando D',
            'e': 'LETRA E - Dedo índice y medio formando E',
            'f': 'LETRA F - Pulgar e índice formando F',
            'g': 'LETRA G - Dedo índice apuntando',
            'h': 'LETRA H - Dedo índice y medio extendidos',
            'i': 'LETRA I - Dedo meñique extendido',
            'j': 'LETRA J - Dedo índice haciendo J',
            'k': 'LETRA K - Dedo índice y medio formando V',
            'l': 'LETRA L - Pulgar e índice formando L',
            'm': 'LETRA M - Tres dedos doblados',
            'n': 'LETRA N - Dos dedos doblados',
            'o': 'LETRA O - Mano formando círculo',
            'p': 'LETRA P - Dedo índice apuntando hacia abajo',
            'q': 'LETRA Q - Dedo índice apuntando hacia abajo',
            'r': 'LETRA R - Dedo índice y medio cruzados',
            's': 'LETRA S - Puño cerrado',
            't': 'LETRA T - Dedo índice y pulgar formando T',
            'u': 'LETRA U - Dedo índice y medio juntos',
            'v': 'LETRA V - Dedo índice y medio formando V',
            'w': 'LETRA W - Tres dedos extendidos',
            'x': 'LETRA X - Dedo índice doblado',
            'y': 'LETRA Y - Pulgar y meñique extendidos',
            'z': 'LETRA Z - Dedo índice haciendo Z'
        }
        return descriptions.get(class_name, f'Seña: {class_name}')

    def recognize_gesture_fast(self, landmarks_data):
        """
        Reconoce el gesto de forma rápida usando el dataset pre-cargado
        con normalización robusta y múltiples métricas
        """
        if not landmarks_data or not landmarks_data['landmarks']:
            return "❌ No se detectó mano"
        
        if self.reference_data is None:
            return "❌ Dataset no cargado"
        
        feature_variants = []

        current_features = self.get_landmark_features(landmarks_data)
        if current_features is not None:
            feature_variants.append(("original", current_features))

        mirrored_landmarks_data = self.mirror_landmarks_data(landmarks_data)
        mirrored_features = self.get_landmark_features(mirrored_landmarks_data)
        if mirrored_features is not None:
            feature_variants.append(("mirrored", mirrored_features))

        if not feature_variants:
            return "❌ No se pudieron extraer características"

        best_variant = "original"
        best_match = None
        best_score = -1
        best_distance = float('inf')
        top_matches = []

        for variant_name, variant_features in feature_variants:
            variant_match, variant_score, variant_distance, variant_top_matches = (
                self.score_features_against_reference(variant_features)
            )
            if variant_score > best_score:
                best_variant = variant_name
                best_match = variant_match
                best_score = variant_score
                best_distance = variant_distance
                top_matches = variant_top_matches
        
        # Ordenar por score combinado
        # Mostrar top 5 coincidencias
        print(f"\n🔍 Top 5 coincidencias (normalizadas, variante: {best_variant}):")
        for i, match in enumerate(top_matches[:5]):
            confidence = max(0, match['combined_score'])
            description = self.get_gesture_description(match['class'])
            print(f"      {i+1}. {description}: {confidence:.1%} (cos: {match['cosine_similarity']:.3f}, corr: {match['correlation']:.3f}, dist: {match['euclidean_distance']:.3f})")
        
        # Determinar confianza y respuesta
        if best_match:
            confidence = max(0, best_score)
            description = self.get_gesture_description(best_match)
            
            if confidence > 0.6:  # Umbral alto para alta confianza
                return f"✅ {description} ({confidence:.1%})"
            elif confidence > 0.4:  # Umbral medio para confianza media
                return f"🤔 Posible {description} ({confidence:.1%})"
            elif confidence > 0.2:  # Umbral bajo pero aún mostrar
                return f"❓ Muy posible {description} ({confidence:.1%})"
            else:  # Muy baja confianza
                return f"❓ Posible {description} ({confidence:.1%}) - Baja confianza"
        else:
            return "❌ Seña no reconocida"
    
    def normalize_features(self, features):
        """
        Normaliza las características usando z-score robusto
        """
        features = np.array(features)
        
        # Calcular estadísticas robustas
        median = np.median(features)
        mad = np.median(np.abs(features - median))  # Median Absolute Deviation
        
        # Evitar división por cero
        if mad == 0:
            mad = np.std(features)
            if mad == 0:
                return features - median  # Solo centrar
        
        # Normalización robusta
        normalized = (features - median) / mad
        
        # Recortar valores extremos
        normalized = np.clip(normalized, -3, 3)
        
        return normalized

    def run_camera(self):
        """
        Activa la cámara y muestra los landmarks de la mano en tiempo real.
        Pulsa 'q' para salir.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo abrir la cámara.")
            return
        print("Cámara activada. Pulsa 'q' para salir.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo leer el frame de la cámara.")
                break
            # Convertir BGR a RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            # Dibujar landmarks si se detectan
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            cv2.imshow('Hand Landmarks - Presiona q para salir', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def run_camera_with_recognition(self):
        """
        Activa la cámara y muestra los landmarks de la mano en tiempo real,
        además de reconocer y mostrar en consola qué seña se está haciendo.
        Pulsa 'q' para salir.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara.")
            return
        
        # Configurar resolución para mejor rendimiento
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Crear instancia de MediaPipe optimizada para video
        video_hands = self.create_video_hands()
        
        print("🎥 Cámara activada. Pulsa 'q' para salir.")
        print("🖐️  Mostrando las señas detectadas en consola...")
        print("=" * 60)
        
        last_recognition = ""
        frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 3  # Solo mostrar "no reconocida" después de 3 fallos
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ No se pudo leer el frame de la cámara.")
                break
            
            # Convertir BGR a RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = video_hands.process(image_rgb)
            
            # Dibujar landmarks si se detectan
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Reconocer gesto cada 15 frames (más frecuente)
                if frame_count % 15 == 0:
                    landmarks_data = {
                        'landmarks': []
                    }
                    for landmark in results.multi_hand_landmarks[0].landmark:
                        landmarks_data['landmarks'].append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    recognition = self.recognize_gesture_fast(landmarks_data)
                    
                    # Solo mostrar si es diferente o si es un fallo después de varios intentos
                    if (recognition != last_recognition or 
                        (recognition == "❌ Seña no reconocida" and consecutive_failures >= max_consecutive_failures)):
                        
                        if recognition == "❌ Seña no reconocida":
                            consecutive_failures += 1
                            if consecutive_failures >= max_consecutive_failures:
                                print(f"🖐️  {recognition}")
                        else:
                            consecutive_failures = 0  # Resetear contador de fallos
                            print(f"🖐️  {recognition}")
                        
                        last_recognition = recognition
            else:
                # Si no se detectan manos, resetear contador de fallos
                consecutive_failures = 0
            
            # Mostrar información en la imagen
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                cv2.putText(frame, f"Manos: {len(results.multi_hand_landmarks)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No se detectan manos", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Hand Landmarks - Presiona q para salir', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        video_hands.close()
        print("🖐️  Cámara cerrada.")

    def create_video_hands(self):
        """Crea una instancia de MediaPipe optimizada para video"""
        return self.mp_hands.Hands(
            static_image_mode=False,  # False para video en tiempo real
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1
        )

def main():
    """Función principal para ejecutar la vectorización de manos.
    """
    dataset_path = "data/kagglehub/asl_dataset"
    csv_path = "data/hand_landmarks_dataset.csv"
    vectorizer = HandVectorizer()
    
    # Procesar dataset
    print("Iniciando procesamiento del dataset...")
    # Soportar dataset adicional 'alphabet' dentro de data/kagglehub
    alphabet_path = vectorizer.project_root / "data" / "kagglehub" / "alphabet"
    if alphabet_path.exists() and alphabet_path.is_dir():
        print(f"Se detectó dataset adicional: {alphabet_path}")
        df1 = vectorizer.process_dataset(dataset_path)
        df2 = vectorizer.process_dataset(str(alphabet_path))
        if df1 is None:
            df = df2
        elif df2 is None:
            df = df1
        else:
            df = pd.concat([df1, df2], ignore_index=True)
            # Guardar CSV combinado
            df.to_csv(vectorizer.resolve_project_path(csv_path), index=False)
    else:
        df = vectorizer.process_dataset(dataset_path, csv_path)
    
    print(f"Dataset procesado. Total de imágenes procesadas: {len(df)}")
    print(f"Clases encontradas: {df['class'].unique()}")
    
    # Mostrar estadísticas
    if not df.empty:
        print("\nEstadísticas del dataset:")
        print(df['class'].value_counts())
        
        # Ejemplo de visualización
        sample_image = df.iloc[0]['image_path']
        print(f"\nVisualizando ejemplo: {sample_image}")
        vectorizer.visualize_landmarks(sample_image, "sample_landmarks.jpg")

if __name__ == "__main__":
    vectorizer = HandVectorizer()
    vectorizer.run_camera_with_recognition()
    
