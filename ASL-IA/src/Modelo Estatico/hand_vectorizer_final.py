# =============================================================
# Archivo: hand_vectorizer_final.py
# Funcionalidad: Ejecuta el procesamiento final de los datasets de imágenes de manos, aplicando la vectorización y generando los archivos necesarios para el reconocimiento de señas.
# =============================================================

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from pathlib import Path
import pickle
from hand_vectorizer import HandVectorizer
from sklearn.cluster import KMeans

class OptimizedHandVectorizer(HandVectorizer):
    """
    Vectorizador de manos optimizado con normalización robusta
    para ser invariante a escala y distancia de la cámara
    """
    
    def __init__(self):
        # Replicar inicialización de MediaPipe sin cargar el dataset base
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.project_root = Path(__file__).resolve().parents[2]
        self.prototypes_per_class = 3
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1
        )
        self.reference_data = None
        self.feature_scaler = None
        self.reference_features = None
        self.reference_labels = None
        
    def normalize_features_robust(self, features):
        """
        Normalización robusta que hace las características invariantes a escala
        """
        features = np.array(features)
        
        # 1. Separar coordenadas de landmarks (63) y distancias (5)
        landmark_coords = features[:63]  # x,y,z de 21 landmarks
        distances = features[63:68] if len(features) >= 68 else features[63:]
        
        # 2. Normalizar coordenadas de landmarks por escala
        # Reshape a (21, 3) para procesar cada landmark
        landmarks_3d = landmark_coords.reshape(-1, 3)
        
        # Calcular el centro de masa de la mano
        center = np.mean(landmarks_3d, axis=0)
        
        # Centrar los landmarks
        centered_landmarks = landmarks_3d - center
        
        # Calcular la escala basada en la distancia máxima desde el centro
        max_distance = np.max(np.linalg.norm(centered_landmarks, axis=1))
        
        # Normalizar por escala (hacer la mano de tamaño unitario)
        if max_distance > 0:
            normalized_landmarks = centered_landmarks / max_distance
        else:
            normalized_landmarks = centered_landmarks
        
        # 3. Normalizar distancias relativas
        if len(distances) > 0:
            # Calcular distancias relativas al tamaño de la mano
            if max_distance > 0:
                normalized_distances = distances / max_distance
            else:
                normalized_distances = distances
        else:
            normalized_distances = np.array([])
        
        # 4. Reconstruir vector de características
        normalized_features = np.concatenate([
            normalized_landmarks.flatten(),
            normalized_distances
        ])
        
        # 5. Asegurar que siempre tengamos 68 características
        while len(normalized_features) < 68:
            normalized_features = np.append(normalized_features, 0.0)
        
        if len(normalized_features) > 68:
            normalized_features = normalized_features[:68]
        
        return normalized_features
    
    def extract_scale_invariant_features(self, landmarks_data):
        """
        Extrae características invariantes a escala de los landmarks
        """
        if not landmarks_data or not landmarks_data['landmarks']:
            return None
        
        landmarks = landmarks_data['landmarks']
        
        # Obtener características básicas
        basic_features = self.get_landmark_features(landmarks_data)
        if basic_features is None:
            return None
        
        # Aplicar normalización robusta
        normalized_features = self.normalize_features_robust(basic_features)
        
        return normalized_features
    
    def load_reference_dataset_optimized(self, dataset_path=None):
        """
        Carga el dataset de referencia desde un archivo pkl optimizado.
        o desde el dataset de imágenes si no existe.
        """
        processed_file = self.project_root / "data" / "reference_features_optimized.pkl"
        if dataset_path is None:
            dataset_path = self.project_root / "data" / "kagglehub" / "asl_dataset"
        # Soportar dataset adicional 'alphabet' dentro de data/kagglehub
        alphabet_path = self.project_root / "data" / "kagglehub" / "alphabet"
        
        try:
            with open(processed_file, "rb") as f:
                self.reference_data = pickle.load(f)
            self.reference_data = self.normalize_reference_data_structure(self.reference_data)
            self.reference_features = [prototype for prototypes in self.reference_data.values() for prototype in prototypes]
            self.reference_labels = list(self.reference_data.keys())
            print(f"✅ Dataset cargado con {len(self.reference_features)} clases")
            return
        except FileNotFoundError:
            print("🔄 Procesando dataset para extraer características optimizadas...")
            # Construir lista de paths a procesar
            dataset_paths = [Path(dataset_path)]
            if alphabet_path.exists() and alphabet_path.is_dir():
                dataset_paths.append(alphabet_path)
            
            self.reference_features, self.reference_labels = self.process_and_save_dataset(
                dataset_paths, processed_file
            )
    
    def process_and_save_dataset(self, dataset_paths, output_pkl_path):
        """
        Procesa uno o varios paths de dataset completos, extrae características y lo guarda en un archivo pkl.
        """
        self.reference_data = {}

        # Aceptar una lista de paths o un solo path
        if not isinstance(dataset_paths, (list, tuple)):
            dataset_paths = [Path(dataset_paths)]
        else:
            dataset_paths = [Path(p) for p in dataset_paths]

        # Mapear clase -> lista de features (para promediar si aparece en varios datasets)
        class_features_map = {}

        for dataset_path in dataset_paths:
            if not dataset_path.exists():
                print(f"Advertencia: dataset no encontrado: {dataset_path}")
                continue

            # Procesar cada carpeta de clase dentro del dataset
            for class_folder in dataset_path.iterdir():
                if not class_folder.is_dir():
                    continue

                class_name = class_folder.name.lower()
                if class_name not in [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]:
                    print(f"Ignorando carpeta no válida: {class_folder.name}")
                    continue

                print(f"  Procesando clase: {class_name} (desde {dataset_path.name})")

                collected = []
                for image_file in class_folder.glob("*.jpeg"):
                    landmarks_data = self.extract_hand_landmarks(str(image_file))
                    if landmarks_data and landmarks_data['landmarks']:
                        features = self.extract_scale_invariant_features(landmarks_data)
                        if features is not None:
                            collected.append(features)

                if collected:
                    class_features_map.setdefault(class_name, []).extend(collected)
                    print(f"    ✅ {len(collected)} imágenes procesadas")
                else:
                    print(f"    ❌ No se pudieron extraer características en {class_folder}")

        # Construir múltiples prototipos por clase y guardar
        for class_name, feats in class_features_map.items():
            self.reference_data[class_name] = self.build_class_prototypes(feats)

        print("💾 Guardando características optimizadas...")
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(self.reference_data, f)

        print(f"✅ Dataset optimizado cargado con {len(self.reference_data)} clases")
        print(f"📁 Características guardadas en: {output_pkl_path}")

        if self.reference_data:
            self.reference_features = [prototype for prototypes in self.reference_data.values() for prototype in prototypes]
            self.reference_labels = list(self.reference_data.keys())
        else:
            self.reference_features = []
            self.reference_labels = []
        return self.reference_features, self.reference_labels

    def normalize_reference_data_structure(self, reference_data):
        """
        Mantiene compatibilidad con PKL antiguos de un solo prototipo por clase.
        """
        normalized = {}
        for class_name, value in reference_data.items():
            if isinstance(value, list):
                normalized[class_name] = [np.asarray(prototype, dtype=float) for prototype in value]
            else:
                normalized[class_name] = [np.asarray(value, dtype=float)]
        return normalized

    def build_class_prototypes(self, class_features):
        """
        Genera varios centroides por clase para capturar variación intra-clase.
        """
        class_features = np.asarray(class_features, dtype=float)
        num_clusters = min(self.prototypes_per_class, len(class_features))

        if num_clusters <= 1 or len(class_features) < num_clusters * 4:
            return [np.mean(class_features, axis=0)]

        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=10,
        )
        cluster_ids = kmeans.fit_predict(class_features)

        prototypes = []
        for cluster_id in range(num_clusters):
            cluster_features = class_features[cluster_ids == cluster_id]
            if len(cluster_features) > 0:
                prototypes.append(np.mean(cluster_features, axis=0))

        return prototypes if prototypes else [np.mean(class_features, axis=0)]
    
    def recognize_gesture_optimized(self, landmarks_data):
        """
        Reconoce gestos con características optimizadas y normalizadas
        """
        if not landmarks_data or not landmarks_data['landmarks']:
            return "❌ No se detectó mano"
        
        if self.reference_data is None:
            return "❌ Dataset no cargado"
        
        feature_variants = []

        current_features = self.extract_scale_invariant_features(landmarks_data)
        if current_features is not None:
            feature_variants.append(("original", current_features))

        mirrored_landmarks_data = self.mirror_landmarks_data(landmarks_data)
        mirrored_features = self.extract_scale_invariant_features(mirrored_landmarks_data)
        if mirrored_features is not None:
            feature_variants.append(("mirrored", mirrored_features))

        if not feature_variants:
            return "❌ No se pudieron extraer características"

        best_variant = "original"
        best_match = None
        best_score = -1
        top_matches = []

        for variant_name, variant_features in feature_variants:
            variant_top_matches = []
            variant_best_match = None
            variant_best_score = -1

            for class_name, class_prototypes in self.reference_data.items():
                best_class_match = None

                for ref_features in class_prototypes:
                    if len(ref_features) != len(variant_features):
                        continue

                    denominator = np.linalg.norm(variant_features) * np.linalg.norm(ref_features)
                    if denominator == 0:
                        cosine_similarity = 0
                    else:
                        cosine_similarity = np.dot(variant_features, ref_features) / denominator

                    correlation = np.corrcoef(variant_features, ref_features)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0

                    combined_score = (cosine_similarity * 0.7 + correlation * 0.3)

                    if best_class_match is None or combined_score > best_class_match['combined_score']:
                        best_class_match = {
                            'class': class_name,
                            'cosine_similarity': cosine_similarity,
                            'correlation': correlation,
                            'combined_score': combined_score
                        }

                if best_class_match is not None:
                    variant_top_matches.append(best_class_match)

                    if best_class_match['combined_score'] > variant_best_score:
                        variant_best_score = best_class_match['combined_score']
                        variant_best_match = class_name

            variant_top_matches.sort(key=lambda x: x['combined_score'], reverse=True)

            if variant_best_score > best_score:
                best_variant = variant_name
                best_match = variant_best_match
                best_score = variant_best_score
                top_matches = variant_top_matches
        
        # Mostrar top 3 coincidencias (menos verbose)
        print(f"\n🔍 Top 3 coincidencias (variante: {best_variant}):")
        for i, match in enumerate(top_matches[:3]):
            confidence = max(0, match['combined_score'])
            description = self.get_gesture_description(match['class'])
            print(f"   {i+1}. {description}: {confidence:.1%}")
        
        # Determinar confianza y respuesta
        if best_match:
            confidence = max(0, best_score)
            description = self.get_gesture_description(best_match)
            
            if confidence > 0.75:  # Umbral alto para alta confianza
                return f"✅ {description} ({confidence:.1%})"
            elif confidence > 0.60:  # Umbral medio
                return f"🤔 {description} ({confidence:.1%})"
            elif confidence > 0.45:  # Umbral bajo
                return f"❓ {description} ({confidence:.1%})"
            else:
                return f"❓ Posible {description} ({confidence:.1%})"
        else:
            return "❌ Seña no reconocida"
    
    def run_camera_optimized(self):
        """
        Cámara optimizada con reconocimiento robusto
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara.")
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Crear instancia optimizada para video
        video_hands = self.create_video_hands()
        
        print("🎥 Cámara optimizada activada. Pulsa 'q' para salir.")
        print("🖐️  Sistema optimizado para funcionar a cualquier distancia")
        print("=" * 60)
        
        last_recognition = ""
        frame_count = 0
        recognition_interval = 20  # Reconocer cada 20 frames
        
        cv2.namedWindow('Sistema Optimizado - Presiona q para salir', cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = video_hands.process(image_rgb)
            
            # Dibujar landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Reconocer gesto
                if frame_count % recognition_interval == 0:
                    landmarks_data = {
                        'landmarks': []
                    }
                    for landmark in results.multi_hand_landmarks[0].landmark:
                        landmarks_data['landmarks'].append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    recognition = self.recognize_gesture_optimized(landmarks_data)
                    
                    if recognition != last_recognition:
                        print(f"🖐️  {recognition}")
                        last_recognition = recognition
            else:
                # Resetear reconocimiento si no hay manos
                if last_recognition != "":
                    last_recognition = ""
            
            # Mostrar información en pantalla
            cv2.putText(frame, "Sistema Optimizado - Cualquier distancia", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if results.multi_hand_landmarks:
                cv2.putText(frame, f"Manos: {len(results.multi_hand_landmarks)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No se detectan manos", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Sistema Optimizado - Presiona q para salir', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        video_hands.close()
        print("🖐️  Cámara cerrada.")

def main():
    """
    Función principal del sistema optimizado
    """
    print("🚀 SISTEMA DE RECONOCIMIENTO DE SEÑAS OPTIMIZADO")
    print("=" * 60)
    print("✨ Características:")
    print("   • Normalización invariante a escala")
    print("   • Funciona a cualquier distancia de la cámara")
    print("   • Reconocimiento más estable")
    print("   • Menos sensible a variaciones de posición")
    print("=" * 60)
    
    # Crear vectorizador
    vectorizer = OptimizedHandVectorizer()
    
    # Cargar dataset
    vectorizer.load_reference_dataset_optimized()
    
    if vectorizer.reference_data:
        classes = sorted(vectorizer.reference_data.keys())
        print(f"\n📚 Clases disponibles ({len(classes)}):")
        print("   Números: " + ", ".join([c for c in classes if c.isdigit()]))
        print("   Letras: " + ", ".join([c for c in classes if c.isalpha()]))
    
    print("\n🎯 Iniciando cámara optimizada...")
    print("💡 Consejos:")
    print("   • Funciona a cualquier distancia de la cámara")
    print("   • Mantén la mano bien iluminada")
    print("   • Haz las señas de forma clara y pausada")
    print("   • El sistema es más estable que antes")
    
    # Iniciar cámara
    vectorizer.run_camera_optimized()

if __name__ == "__main__":
    main() 
