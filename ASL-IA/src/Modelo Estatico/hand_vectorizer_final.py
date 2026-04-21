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
            
        try:
            with open(processed_file, "rb") as f:
                self.reference_data = pickle.load(f)
            self.reference_features = list(self.reference_data.values())
            self.reference_labels = list(self.reference_data.keys())
            print(f"✅ Dataset cargado con {len(self.reference_features)} clases")
            return
        except FileNotFoundError:
            print("🔄 Procesando dataset para extraer características optimizadas...")
            
            self.reference_features, self.reference_labels = self.process_and_save_dataset(
                dataset_path, processed_file
            )
    
    def process_and_save_dataset(self, dataset_path, output_pkl_path):
        """
        Procesa el dataset completo, extrae características y lo guarda en un archivo pkl.
        """
        self.reference_data = {}
        dataset_path = Path(dataset_path)
        
        # Procesar cada clase
        for class_folder in dataset_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                
                # Solo procesar clases válidas
                if class_name in [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]:
                    print(f"  Procesando clase: {class_name}")
                    
                    class_features = []
                    
                    # Procesar imágenes de la clase
                    for image_file in class_folder.glob("*.jpeg"):
                        landmarks_data = self.extract_hand_landmarks(str(image_file))
                        
                        if landmarks_data and landmarks_data['landmarks']:
                            features = self.extract_scale_invariant_features(landmarks_data)
                            if features is not None:
                                class_features.append(features)
                    
                    if class_features:
                        # Calcular características promedio de la clase
                        avg_features = np.mean(class_features, axis=0)
                        self.reference_data[class_name] = avg_features
                        print(f"    ✅ {len(class_features)} imágenes procesadas")
                    else:
                        print(f"    ❌ No se pudieron extraer características")
        
        # Guardar características procesadas
        print("💾 Guardando características optimizadas...")
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(self.reference_data, f)
        
        print(f"✅ Dataset optimizado cargado con {len(self.reference_data)} clases")
        print(f"📁 Características guardadas en: {output_pkl_path}")
        
        if self.reference_data:
            self.reference_features = list(self.reference_data.values())
            self.reference_labels = list(self.reference_data.keys())
        else:
            self.reference_features = []
            self.reference_labels = []
        return self.reference_features, self.reference_labels
    
    def recognize_gesture_optimized(self, landmarks_data):
        """
        Reconoce gestos con características optimizadas y normalizadas
        """
        if not landmarks_data or not landmarks_data['landmarks']:
            return "❌ No se detectó mano"
        
        if self.reference_data is None:
            return "❌ Dataset no cargado"
        
        # Extraer características optimizadas
        current_features = self.extract_scale_invariant_features(landmarks_data)
        if current_features is None:
            return "❌ No se pudieron extraer características"
        
        # Comparar con cada clase de referencia
        best_match = None
        best_score = -1
        top_matches = []
        
        for class_name, ref_features in self.reference_data.items():
            if len(ref_features) == len(current_features):
                # Calcular similitud de coseno
                cosine_similarity = np.dot(current_features, ref_features) / (
                    np.linalg.norm(current_features) * np.linalg.norm(ref_features)
                )
                
                # Calcular correlación
                correlation = np.corrcoef(current_features, ref_features)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
                
                # Score combinado
                combined_score = (cosine_similarity * 0.7 + correlation * 0.3)
                
                top_matches.append({
                    'class': class_name,
                    'cosine_similarity': cosine_similarity,
                    'correlation': correlation,
                    'combined_score': combined_score
                })
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = class_name
        
        # Ordenar por score
        top_matches.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Mostrar top 3 coincidencias (menos verbose)
        print("\n🔍 Top 3 coincidencias:")
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
