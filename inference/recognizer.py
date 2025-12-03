import numpy as np
import cv2
from utils.face_detector import FaceDetector

class FaceRecognizer:
    def __init__(self, model_path=None, threshold=0.6):
        self.threshold = threshold
        self.face_detector = FaceDetector()
        self.known_embeddings = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Carga modelo pre-entrenado"""
        from models.face_embedding_model import FaceEmbeddingModel
        self.model = FaceEmbeddingModel()
        self.model.load_model(model_path)
        self.embedding_model = self.model.get_embedding_model()
    
    def extract_embedding(self, face_image):
        """Extrae embedding de un rostro"""
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        embedding = self.embedding_model.predict(face_image)
        return embedding[0]
    
    def add_known_face(self, name, embedding):
        """Agrega un rostro conocido a la base de datos"""
        self.known_embeddings[name] = embedding
    
    def recognize_face(self, face_image):
        """Reconoce un rostro comparando con la base de datos"""
        query_embedding = self.extract_embedding(face_image)
        
        best_match = None
        min_distance = float('inf')
        
        for name, known_embedding in self.known_embeddings.items():
            distance = np.linalg.norm(query_embedding - known_embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        if min_distance > self.threshold:
            return "Unknown", min_distance
        
        return best_match, min_distance
    
    def process_image(self, image_path):
        """Procesa una imagen completa: detecta y reconoce rostros"""
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        faces_coords = self.face_detector.detect_faces(image)
        results = []
        
        for coords in faces_coords:
            face, (x, y, w, h) = self.face_detector.extract_face(rgb_image, coords)
            name, confidence = self.recognize_face(face)
            
            results.append({
                'name': name,
                'confidence': 1 - confidence,  # Convertir a confianza
                'bbox': (x, y, w, h)
            })
        
        return results