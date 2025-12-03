import cv2
import numpy as np

class Preprocessor:
    def __init__(self, target_size=(160, 160)):
        self.target_size = target_size
    
    def load_and_preprocess(self, image_path):
        """Carga y preprocesa una imagen desde archivo"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.target_size)
            image = image.astype('float32')
            image = (image - 127.5) / 127.5  # Normalizar a [-1, 1]
            return image
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            return None

    def load_and_preprocess_from_array(self, image_array):
        """Preprocesa una imagen desde un array numpy"""
        try:
            if len(image_array.shape) == 3:
                image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image = image_array
                
            image = cv2.resize(image, self.target_size)
            image = image.astype('float32')
            image = (image - 127.5) / 127.5  # Normalizar a [-1, 1]
            return image
        except Exception as e:
            print(f"Error procesando array: {e}")
            return None
    
    def preprocess_batch(self, image_paths):
        """Preprocesa un batch de imágenes"""
        processed_images = []
        valid_paths = []
        
        for path in image_paths:
            img = self.load_and_preprocess(path)
            if img is not None:
                processed_images.append(img)
                valid_paths.append(path)
        
        return np.array(processed_images), valid_paths
    
    def augment_image(self, image):
        """Aumenta datos de una imagen"""
        # Flip horizontal aleatorio
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Ajuste de brillo
        brightness = np.random.uniform(0.8, 1.2)
        image = image * brightness
        image = np.clip(image, -1.0, 1.0)
        
        return image