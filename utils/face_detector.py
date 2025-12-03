import cv2
import numpy as np

class FaceDetector:
    def __init__(self, method='haar'):
        self.method = method
        
        if method == 'haar':
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif method == 'dnn':
            # Configuración para DNN face detector
            pass
    
    def detect_faces(self, image):
        """Detecta rostros en una imagen"""
        if self.method == 'haar':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            return faces
        else:
            # Implementar otros métodos de detección
            return []
    
    def extract_face(self, image, face_coords, target_size=(160, 160)):
        """Extrae y redimensiona un rostro"""
        x, y, w, h = face_coords
        # Expandir ligeramente el área del rostro
        x = max(0, x - int(0.1 * w))
        y = max(0, y - int(0.1 * h))
        w = min(image.shape[1] - x, int(1.2 * w))
        h = min(image.shape[0] - y, int(1.2 * h))
        
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        return face, (x, y, w, h)