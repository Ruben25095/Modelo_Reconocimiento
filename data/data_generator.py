import numpy as np
import os
import csv
from sklearn.preprocessing import LabelEncoder
from .preprocessor import Preprocessor

class CelebADataGenerator:
    def __init__(self, dataset_path, img_size=(160, 160)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.preprocessor = Preprocessor(img_size)
        self.images = []
        self.labels = []
        self.encoded_labels = []
        self.label_encoder = LabelEncoder()
        
    def load_dataset_simple(self, max_samples=None, num_classes=100):
        """Carga el dataset agrupando en un número manejable de clases"""
        img_dir = os.path.join(self.dataset_path, 'img_align_celeba')
        if not os.path.exists(img_dir):
            raise ValueError(f"Directorio de imágenes no encontrado: {img_dir}")
        
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        self.images = []
        self.labels = []
        
        for img_name in image_files:
            img_path = os.path.join(img_dir, img_name)
            self.images.append(img_path)
            # Crear un número limitado de clases usando hash
            label = hash(img_name) % num_classes  # Limitar a num_classes clases
            self.labels.append(label)
        
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        # Verificar distribución de clases
        class_dist = self.get_class_distribution()
        classes_with_multiple = sum(1 for count in class_dist.values() if count >= 2)
        
        print(f"Dataset cargado: {len(self.images)} imágenes, {len(class_dist)} clases")
        print(f"Clases con múltiples muestras: {classes_with_multiple}")
        
        if classes_with_multiple < 2:
            print("⚠️  ADVERTENCIA: Muy pocas clases con múltiples muestras para tripletes")
            print("💡 Considera aumentar max_samples o reducir num_classes")
        
        return self
    
    def load_dataset_with_attributes(self, max_samples=None, num_classes=100):
        """Intenta cargar usando el archivo de atributos"""
        attr_file = os.path.join(self.dataset_path, 'list_attr_celeba.csv')
        
        if not os.path.exists(attr_file):
            print("Archivo de atributos no encontrado, usando método simple")
            return self.load_dataset_simple(max_samples, num_classes)
        
        print("Cargando atributos desde CSV...")
        attributes_dict = self._load_attributes_csv(attr_file)
        
        img_dir = os.path.join(self.dataset_path, 'img_align_celeba')
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_samples:
            image_files = image_files[:max_samples]
        
        self.images = []
        self.labels = []
        
        for img_name in image_files:
            if img_name in attributes_dict:
                img_path = os.path.join(img_dir, img_name)
                self.images.append(img_path)
                # Crear identidad limitada usando atributos
                identity_code = hash(attributes_dict[img_name]) % num_classes
                self.labels.append(identity_code)
        
        if len(self.images) == 0:
            print("No se pudieron cargar imágenes con atributos, usando método simple")
            return self.load_dataset_simple(max_samples, num_classes)
        
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        # Verificar distribución
        class_dist = self.get_class_distribution()
        classes_with_multiple = sum(1 for count in class_dist.values() if count >= 2)
        
        print(f"Dataset con atributos cargado: {len(self.images)} imágenes, {len(class_dist)} clases")
        print(f"Clases con múltiples muestras: {classes_with_multiple}")
        
        return self

    def _load_attributes_csv(self, csv_file):
        """Carga el archivo de atributos sin pandas"""
        attributes_dict = {}
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Leer header
                
                for row in reader:
                    if len(row) > 0:
                        image_id = row[0]
                        # Usar más atributos para mejor diferenciación
                        attrs = []
                        for val in row[1:16]:  # Usar primeros 15 atributos
                            if val.strip() == '1':
                                attrs.append('1')
                            else:
                                attrs.append('0')
                        attributes_dict[image_id] = ''.join(attrs)
        except Exception as e:
            print(f"Error cargando atributos {csv_file}: {e}")
        return attributes_dict
    
    def get_triplet_batch(self, batch_size=32):
        """Genera un batch de tripletes con mejor manejo de clases"""
        if len(self.images) == 0:
            raise ValueError("Primero debe cargar el dataset")
        
        n_classes = len(np.unique(self.encoded_labels))
        
        # Obtener distribución de clases
        class_counts = {}
        for label in self.encoded_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Filtrar clases con al menos 2 muestras
        valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
        
        if len(valid_classes) < 2:
            raise ValueError(f"Se necesitan al menos 2 clases con múltiples muestras. Clases válidas: {len(valid_classes)}")
        
        # Crear índice por clase
        class_indices = {}
        for idx, label in enumerate(self.encoded_labels):
            if label in valid_classes:
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
        
        anchors, positives, negatives = [], [], []
        attempts = 0
        max_attempts = batch_size * 20  # Más intentos
        
        while len(anchors) < batch_size and attempts < max_attempts:
            attempts += 1
            
            try:
                # Seleccionar clase para anchor/positive
                class_idx = np.random.choice(valid_classes)
                class_samples = class_indices[class_idx]
                
                if len(class_samples) < 2:
                    continue
                    
                # Seleccionar anchor y positive de la misma clase
                anchor_idx, positive_idx = np.random.choice(class_samples, 2, replace=False)
                
                # Seleccionar clase diferente para negative
                negative_classes = [cls for cls in valid_classes if cls != class_idx]
                if not negative_classes:
                    continue
                    
                negative_class = np.random.choice(negative_classes)
                negative_samples = class_indices[negative_class]
                
                if len(negative_samples) == 0:
                    continue
                    
                negative_idx = np.random.choice(negative_samples)
                
                # Cargar y preprocesar imágenes
                anchor_img = self.preprocessor.load_and_preprocess(self.images[anchor_idx])
                positive_img = self.preprocessor.load_and_preprocess(self.images[positive_idx])
                negative_img = self.preprocessor.load_and_preprocess(self.images[negative_idx])
                
                if (anchor_img is not None and positive_img is not None and 
                    negative_img is not None):
                    anchors.append(anchor_img)
                    positives.append(positive_img)
                    negatives.append(negative_img)
                    
            except Exception as e:
                continue  # Continuar en caso de error
        
        if len(anchors) == 0:
            raise ValueError(f"No se pudieron generar tripletes después de {max_attempts} intentos")
        
        if len(anchors) < batch_size:
            print(f"⚠️  Solo se generaron {len(anchors)} tripletes de {batch_size}")
        
        return [np.array(anchors), np.array(positives), np.array(negatives)]
    
    def data_generator(self, batch_size=32):
        """Generador infinito de datos para entrenamiento con mejor manejo de errores"""
        while True:
            try:
                batch = self.get_triplet_batch(batch_size)
                if len(batch[0]) > 0:
                    yield batch, np.zeros(len(batch[0]))
                else:
                    # Si no se pudo generar batch, esperar un poco y reintentar
                    import time
                    time.sleep(0.1)
            except Exception as e:
                print(f"❌ Error en generador: {e}")
                # Esperar antes de reintentar
                import time
                time.sleep(1)
                continue
    
    def get_class_distribution(self):
        """Retorna la distribución de clases"""
        unique, counts = np.unique(self.encoded_labels, return_counts=True)
        return dict(zip(unique, counts))
    
    def analyze_dataset(self):
        """Analiza el dataset y muestra estadísticas útiles"""
        if len(self.images) == 0:
            print("❌ Dataset no cargado")
            return
        
        class_dist = self.get_class_distribution()
        total_images = len(self.images)
        total_classes = len(class_dist)
        
        classes_with_multiple = sum(1 for count in class_dist.values() if count >= 2)
        classes_with_many = sum(1 for count in class_dist.values() if count >= 5)
        
        print(f"📊 ANÁLISIS DEL DATASET:")
        print(f"   - Total imágenes: {total_images}")
        print(f"   - Total clases: {total_classes}")
        print(f"   - Clases con ≥2 muestras: {classes_with_multiple}")
        print(f"   - Clases con ≥5 muestras: {classes_with_many}")
        print(f"   - Promedio de imágenes por clase: {total_images/total_classes:.1f}")
        
        # Mostrar las clases más grandes
        largest_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"   - Clases más grandes: {dict(largest_classes)}")