import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
import time
import sys

class CompleteSingleWindowRecognizer:
    def __init__(self):
        print("🚀 INICIANDO - VERSIÓN COMPLETA CON UNA VENTANA")
        self.window_name = "Reconocimiento Facial"
        self.cap = None
        self.is_running = False
        
        # Base de datos de rostros
        self.known_faces = {}
        self.face_database_file = 'face_database.pkl'
        
        # Forzar limpieza inicial
        self.force_clean_windows()
        
    def force_clean_windows(self):
        """Limpieza agresiva de ventanas previas"""
        print("🧹 Limpiando ventanas previas...")
        try:
            cv2.destroyAllWindows()
            for i in range(5):
                cv2.waitKey(1)
            time.sleep(1)
        except:
            pass
    
    def initialize_components(self):
        """Inicializa todos los componentes necesarios"""
        # 1. Cargar modelo
        print("🧠 Cargando modelo...")
        try:
            model_path = '../saved_models/best_model.h5'
            if not os.path.exists(model_path):
                print("❌ Modelo no encontrado")
                return None, None, None
            
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Modelo cargado")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return None, None, None
        
        # 2. Cargar detector de rostros
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("✅ Detector de rostros cargado")
        except Exception as e:
            print(f"❌ Error cargando detector: {e}")
            return None, None, None
        
        # 3. Cargar base de datos
        self.load_face_database()
        
        # 4. Inicializar cámara
        if not self.initialize_camera():
            return None, None, None
        
        return self.model, self.face_cascade, self.cap
    
    def initialize_camera(self):
        """Inicializa la cámara"""
        print("🎥 Inicializando cámara...")
        
        if self.cap:
            self.cap.release()
            time.sleep(0.5)
        
        try:
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 15)
                print("✅ Cámara inicializada")
                return True
        except Exception as e:
            print(f"❌ Error con cámara: {e}")
        
        return False
    
    def load_face_database(self):
        """Carga la base de datos de rostros"""
        try:
            if os.path.exists(self.face_database_file):
                with open(self.face_database_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"✅ Base de datos cargada: {len(self.known_faces)} rostros")
        except Exception as e:
            print(f"⚠️  Error cargando base de datos: {e}")
            self.known_faces = {}
    
    def save_face_database(self):
        """Guarda la base de datos"""
        try:
            with open(self.face_database_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
        except Exception as e:
            print(f"⚠️  Error guardando base de datos: {e}")
    
    def extract_embedding(self, face_image):
        """Extrae embedding de un rostro"""
        try:
            if len(face_image.shape) == 3:
                face_image = np.expand_dims(face_image, axis=0)
            embedding = self.model.predict(face_image, verbose=0)[0]
            return embedding
        except Exception as e:
            print(f"⚠️  Error extrayendo embedding: {e}")
            return np.zeros(128)
    
    def preprocess_face(self, face_image):
        """Preprocesa un rostro para el modelo"""
        try:
            face = cv2.resize(face_image, (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype('float32')
            face = (face - 127.5) / 127.5
            return face
        except Exception as e:
            print(f"⚠️  Error preprocesando rostro: {e}")
            return np.zeros((160, 160, 3), dtype=np.float32)
    
    def recognize_face(self, face_image):
        """Reconoce un rostro"""
        try:
            preprocessed_face = self.preprocess_face(face_image)
            query_embedding = self.extract_embedding(preprocessed_face)
            
            best_match = "Desconocido"
            best_confidence = 0
            
            for name, data in self.known_faces.items():
                known_embedding = data['embedding']
                similarity = np.dot(query_embedding, known_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(known_embedding)
                )
                
                if similarity > best_confidence:
                    best_confidence = similarity
                    if similarity > 0.6:  # Threshold
                        best_match = name
            
            return best_match, best_confidence
        except Exception as e:
            print(f"⚠️  Error reconociendo rostro: {e}")
            return "Error", 0.0
    
    def register_face(self, name, face_image):
        """Registra un nuevo rostro"""
        try:
            embedding = self.extract_embedding(self.preprocess_face(face_image))
            self.known_faces[name] = {
                'embedding': embedding,
                'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'sample_image': face_image
            }
            self.save_face_database()
            print(f"✅ Rostro '{name}' registrado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error registrando rostro: {e}")
            return False
    
    def draw_face_info(self, frame, x, y, w, h, name, confidence):
        """Dibuja información del rostro"""
        try:
            if confidence > 0.8:
                color = (0, 255, 0)  # Verde
            elif confidence > 0.6:
                color = (0, 255, 255)  # Amarillo
            else:
                color = (0, 0, 255)  # Rojo
            
            # Rectángulo
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Etiqueta
            label = f"{name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Fondo para texto
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Texto
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            print(f"⚠️  Error dibujando información: {e}")
    
    def run_complete(self):
        """Ejecuta la aplicación completa"""
        print("🎯 INICIANDO APLICACIÓN COMPLETA")
        print("=" * 50)
        
        # Inicializar componentes
        model, face_cascade, cap = self.initialize_components()
        if not all([model, face_cascade, cap]):
            print("❌ Error inicializando componentes")
            return
        
        self.is_running = True
        window_created = False
        fps_counter = 0
        fps_time = time.time()
        
        print("\n🎮 CONTROLES COMPLETOS:")
        print("   Q = Salir")
        print("   R = Registrar rostro actual")
        print("   L = Listar rostros registrados")
        print("   C = Limpiar pantalla")
        print("=" * 50)
        
        try:
            while self.is_running:
                # Leer frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Error de frame")
                    time.sleep(0.1)
                    continue
                
                # Procesamiento
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detectar rostros
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
                
                # Procesar rostros
                for (x, y, w, h) in faces[:3]:  # Máximo 3 rostros
                    try:
                        face_roi = frame[y:y+h, x:x+w]
                        name, confidence = self.recognize_face(face_roi)
                        self.draw_face_info(frame, x, y, w, h, name, confidence)
                    except Exception as e:
                        continue
                
                # Calcular FPS
                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()
                else:
                    fps = fps_counter
                
                # Información en pantalla
                cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Rostros: {len(faces)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Registrados: {len(self.known_faces)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "R=Registrar, Q=Salir, L=Listar", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Mostrar UNA ventana
                if not window_created:
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_name, 800, 600)
                    window_created = True
                    print("✅ Ventana única creada")
                
                cv2.imshow(self.window_name, frame)
                
                # Manejo de teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("👋 Saliendo...")
                    break
                elif key == ord('r') and len(faces) > 0:
                    self.handle_face_registration(frame, faces)
                elif key == ord('l'):
                    self.list_registered_faces()
                elif key == ord('c'):
                    os.system('clear' if os.name == 'posix' else 'cls')
                
        except KeyboardInterrupt:
            print("\n🛑 Interrumpido por usuario")
        except Exception as e:
            print(f"❌ Error crítico: {e}")
        finally:
            self.cleanup()
    
    def handle_face_registration(self, frame, faces):
        """Maneja el registro de rostros"""
        try:
            if len(faces) == 0:
                print("❌ No hay rostros para registrar")
                return
            
            # Usar el rostro más grande
            areas = [w * h for (x, y, w, h) in faces]
            largest_idx = np.argmax(areas)
            x, y, w, h = faces[largest_idx]
            
            face_roi = frame[y:y+h, x:x+w]
            
            print("\n📝 REGISTRO DE ROSTRO:")
            print("   Rostro detectado - Ingresa el nombre")
            name = input("   Nombre: ").strip()
            
            if name:
                if self.register_face(name, face_roi):
                    print(f"   ✅ '{name}' registrado exitosamente")
                else:
                    print(f"   ❌ Error registrando '{name}'")
            else:
                print("   ❌ Registro cancelado")
                
        except Exception as e:
            print(f"❌ Error en registro: {e}")
    
    def list_registered_faces(self):
        """Lista los rostros registrados"""
        print("\n👥 ROSTROS REGISTRADOS:")
        print("=" * 30)
        if not self.known_faces:
            print("   No hay rostros registrados")
            return
        
        for i, (name, data) in enumerate(self.known_faces.items(), 1):
            print(f"   {i}. {name} - {data['registration_date']}")
        print(f"   Total: {len(self.known_faces)} rostros")
    
    def cleanup(self):
        """Limpieza garantizada"""
        print("\n🧹 Realizando limpieza final...")
        self.is_running = False
        
        # Cerrar cámara
        if self.cap:
            self.cap.release()
            print("✅ Cámara liberada")
        
        # Cerrar ventanas
        try:
            cv2.destroyWindow(self.window_name)
            cv2.destroyAllWindows()
        except:
            pass
        
        # Pequeñas pausas para asegurar cierre
        for i in range(10):
            cv2.waitKey(1)
            time.sleep(0.01)
        
        print("✅ Limpieza completada - Cero ventanas")

def main():
    print("👤 RECONOCIMIENTO FACIAL COMPLETO")
    print("=" * 60)
    print("🛡️  Una ventana + Funciones completas + Registro activado")
    print("=" * 60)
    
    app = CompleteSingleWindowRecognizer()
    app.run_complete()

if __name__ == "__main__":
    main()