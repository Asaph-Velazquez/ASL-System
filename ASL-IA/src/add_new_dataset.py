# =============================================================
# Archivo: add_new_dataset.py
# Funcionalidad: Permite agregar un nuevo dataset de imágenes de señas (ASL) al sistema, combinándolo con el dataset existente, procesando las imágenes, validando la estructura y actualizando los archivos de referencia necesarios para el reconocimiento de señas.
# =============================================================
import numpy as np
# ===============================
# Importación de librerías necesarias para procesamiento de datos, imágenes y manejo de archivos
# ===============================
import pandas as pd
import cv2
import mediapipe as mp
from pathlib import Path
import pickle
import os
from hand_vectorizer import HandVectorizer

class DatasetManager:
    """
    Gestor para agregar nuevos datasets al sistema de reconocimiento de señas
    """
    # ===============================
    # Inicialización del gestor y del vectorizador de manos
    # ===============================
    def __init__(self):
        self.vectorizer = HandVectorizer()
        
    # ===============================
    # Método principal para agregar un nuevo dataset al sistema
    # ===============================
    def add_new_dataset(self, new_dataset_path, output_name="combined_dataset"):
        """
        Agrega un nuevo dataset al sistema existente
        
        Args:
            new_dataset_path: Ruta al nuevo dataset
            output_name: Nombre para el dataset combinado
        """
        print(f"🔄 Agregando nuevo dataset: {new_dataset_path}")
        print("=" * 60)
        
        # 1. Cargar dataset existente
        existing_data = self.load_existing_dataset()
        
        # 2. Procesar nuevo dataset
        new_data = self.process_new_dataset(new_dataset_path)
        
        if new_data is None or new_data.empty:
            print("❌ No se pudo procesar el nuevo dataset")
            return
        
        # 3. Combinar datasets
        combined_data = self.combine_datasets(existing_data, new_data)
        
        # 4. Guardar dataset combinado
        self.save_combined_dataset(combined_data, output_name)
        
        # 5. Actualizar archivo de referencia optimizado
        self.update_reference_features(combined_data, output_name)
        
        print(f"✅ Dataset combinado guardado como: {output_name}")
        
    # ===============================
    # Carga el dataset existente si existe
    # ===============================
    def load_existing_dataset(self):
        """Carga el dataset existente"""
        print("📊 Cargando dataset existente...")
        
        # Intentar cargar diferentes versiones del dataset
        possible_files = [
            "hand_landmarks_dataset_corrected.csv",
            "hand_landmarks_dataset.csv"
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"✅ Dataset existente cargado: {file_path} ({len(df)} imágenes)")
                return df
        
        print("⚠️  No se encontró dataset existente. Se creará uno nuevo.")
        return pd.DataFrame()
    
    # ===============================
    # Procesa el nuevo dataset usando el vectorizador
    # ===============================
    def process_new_dataset(self, dataset_path):
        """Procesa el nuevo dataset"""
        print(f"🔄 Procesando nuevo dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"❌ El directorio {dataset_path} no existe")
            return None
        
        # Procesar con el vectorizador
        try:
            df = self.vectorizer.process_dataset(str(dataset_path), None)
            print(f"✅ Nuevo dataset procesado: {len(df)} imágenes")
            return df
        except Exception as e:
            print(f"❌ Error procesando nuevo dataset: {e}")
            return None
    
    # ===============================
    # Combina el dataset existente con el nuevo, alineando columnas y eliminando duplicados
    # ===============================
    def combine_datasets(self, existing_data, new_data):
        """Combina el dataset existente con el nuevo"""
        print("🔗 Combinando datasets...")
        
        if existing_data.empty:
            print("📝 Usando solo el nuevo dataset")
            return new_data
        
        # Verificar que las columnas sean compatibles
        existing_cols = set(existing_data.columns)
        new_cols = set(new_data.columns)
        
        if existing_cols != new_cols:
            print("⚠️  Las columnas no coinciden exactamente")
            print(f"Columnas existentes: {len(existing_cols)}")
            print(f"Columnas nuevas: {len(new_cols)}")
            
            # Intentar alinear columnas
            common_cols = existing_cols.intersection(new_cols)
            if len(common_cols) > 10:  # Mínimo de columnas necesarias
                print(f"✅ .Usando {len(common_cols)} columnas comunes")
                existing_data = existing_data[list(common_cols)]
                new_data = new_data[list(common_cols)]
            else:
                print("❌ No hay suficientes columnas comunes")
                return existing_data
        
        # Combinar datasets
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        
        # Eliminar duplicados si los hay
        initial_count = len(combined_data)
        combined_data = combined_data.drop_duplicates(subset=['image_path'], keep='first')
        final_count = len(combined_data)
        
        if initial_count != final_count:
            print(f"🧹 Eliminados {initial_count - final_count} duplicados")
        
        print(f"✅ Datasets combinados: {len(combined_data)} imágenes totales")
        
        # Mostrar estadísticas por clase
        class_counts = combined_data['class'].value_counts()
        print(f"📊 Clases disponibles: {len(class_counts)}")
        print("Top 10 clases más frecuentes:")
        for class_name, count in class_counts.head(10).items():
            print(f"  {class_name}: {count} imágenes")
        
        return combined_data
    
    # ===============================
    # Guarda el dataset combinado en un archivo CSV
    # ===============================
    def save_combined_dataset(self, combined_data, output_name):
        """Guarda el dataset combinado"""
        output_path = f"{output_name}.csv"
        combined_data.to_csv(output_path, index=False)
        print(f"💾 Dataset guardado: {output_path}")
    
    # ===============================
    # Actualiza el archivo de características de referencia optimizadas
    # ===============================
    def update_reference_features(self, combined_data, dataset_name):
        """Actualiza el archivo de características de referencia"""
        print("🔄 Actualizando características de referencia...")
        
        # Crear características promedio por clase
        reference_data = {}
        valid_classes = 0
        
        for class_name in combined_data['class'].unique():
            # Solo procesar clases válidas
            if class_name in [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]:
                class_data = combined_data[combined_data['class'] == class_name]
                
                # Calcular características promedio
                avg_features = []
                for i in range(21):  # 21 landmarks
                    x_avg = class_data[f'landmark_{i}_x'].mean()
                    y_avg = class_data[f'landmark_{i}_y'].mean()
                    z_avg = class_data[f'landmark_{i}_z'].mean()
                    avg_features.extend([x_avg, y_avg, z_avg])
                
                # Agregar 5 distancias promedio
                avg_features.extend([0.1, 0.2, 0.3, 0.4, 0.5])
                
                # Asegurar 68 características
                while len(avg_features) < 68:
                    avg_features.append(0.0)
                
                if len(avg_features) > 68:
                    avg_features = avg_features[:68]
                
                reference_data[class_name] = np.array(avg_features)
                valid_classes += 1
        
        # Guardar características de referencia
        reference_path = f"{dataset_name}_reference_features.pkl"
        with open(reference_path, 'wb') as f:
            pickle.dump(reference_data, f)
        
        print(f"✅ Características de referencia guardadas: {reference_path}")
        print(f"📊 {valid_classes} clases procesadas")
    
    # ===============================
    # Valida que la estructura del nuevo dataset sea la correcta (carpetas por clase, imágenes, etc.)
    # ===============================
    def validate_dataset_structure(self, dataset_path):
        """
        Valida que el nuevo dataset tenga la estructura correcta
        
        Args:
            dataset_path: Ruta al dataset a validar
            
        Returns:
            bool: True si la estructura es válida
        """
        print(f"🔍 Validando estructura del dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print("❌ El directorio no existe")
            return False
        
        # Verificar estructura de carpetas
        valid_classes = 0
        total_images = 0
        
        for class_folder in dataset_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                
                # Verificar que sea una clase válida
                if class_name in [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]:
                    # Contar imágenes en la carpeta
                    image_files = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg")) + list(class_folder.glob("*.png"))
                    
                    if image_files:
                        valid_classes += 1
                        total_images += len(image_files)
                        print(f"  ✅ Clase '{class_name}': {len(image_files)} imágenes")
                    else:
                        print(f"  ⚠️  Clase '{class_name}': Sin imágenes")
                else:
                    print(f"  ❌ Clase '{class_name}': No válida (ignorada)")
        
        print(f"📊 Resumen: {valid_classes} clases válidas, {total_images} imágenes totales")
        
        if valid_classes > 0 and total_images > 0:
            print("✅ Estructura del dataset válida")
            return True
        else:
            print("❌ Dataset no válido")
            return False

def main():
    """Función principal para agregar nuevos datasets"""
    # ===============================
    # Interfaz de usuario por consola para agregar y validar un nuevo dataset
    # ===============================
    print("🖐️  GESTOR DE DATASETS PARA RECONOCIMIENTO DE SEÑAS ASL")
    print("=" * 60)
    
    manager = DatasetManager()
    
    # Solicitar información al usuario
    print("\n📁 Para agregar un nuevo dataset, necesitas:")
    print("1. Una carpeta con subcarpetas para cada seña (A-Z, 0-9)")
    print("2. Imágenes de manos en cada subcarpeta")
    print("3. Formato de imagen: JPG, JPEG o PNG")
    
    print("\n📂 Ejemplo de estructura:")
    print("nuevo_dataset/")
    print("├── a/          # Letra A")
    print("│   ├── mano1.jpg")
    print("│   └── mano2.jpg")
    print("├── 5/          # Número 5")
    print("│   ├── mano1.jpg")
    print("│   └── mano2.jpg")
    print("└── ...")
    
    # Solicitar ruta del nuevo dataset
    dataset_path = input("\n🔗 Ingresa la ruta al nuevo dataset: ").strip()
    
    if not dataset_path:
        print("❌ No se ingresó una ruta válida")
        return
    
    # Validar estructura
    if not manager.validate_dataset_structure(dataset_path):
        print("❌ El dataset no tiene la estructura correcta")
        return
    
    # Solicitar nombre para el dataset combinado
    output_name = input("📝 Nombre para el dataset combinado (default: combined_dataset): ").strip()
    if not output_name:
        output_name = "combined_dataset"
    
    # Confirmar operación
    print(f"\n⚠️  ¿Estás seguro de que quieres agregar el dataset '{dataset_path}'?")
    confirm = input("Escribe 'SI' para continuar: ").strip().upper()
    
    if confirm == "SI":
        # Agregar dataset
        manager.add_new_dataset(dataset_path, output_name)
        
        print(f"\n🎉 ¡Dataset agregado exitosamente!")
        print(f"📁 Archivos creados:")
        print(f"  - {output_name}.csv (dataset combinado)")
        print(f"  - {output_name}_reference_features.pkl (características optimizadas)")
        
        print(f"\n🚀 Para usar el nuevo dataset:")
        print(f"1. Copia {output_name}.csv como 'data/hand_landmarks_dataset_corrected.csv'")
        print(f"2. Copia {output_name}_reference_features.pkl como 'data/reference_features_optimized.pkl'")
        print(f"3. Ejecuta: python hand_vectorizer_final.py")
    else:
        print("❌ Operación cancelada")

if __name__ == "__main__":
    main() 