import os

class Config:
    # Rutas
    DATASET_PATH = 'data'
    IMAGE_DIR = os.path.join(DATASET_PATH, 'img_align_celeba')
    MODEL_SAVE_PATH = 'saved_models'
    
    # Archivos CSV
    ATTRIBUTES_FILE = 'list_attr_celeba.csv'
    IDENTITY_FILE = 'identity_CelebA.csv'  # Si existe
    BBOX_FILE = 'list_bbox_celeba.csv'
    PARTITION_FILE = 'list_eval_partition.csv'
    LANDMARKS_FILE = 'list_landmarks_align_celeba.csv'
    
    # Parámetros del modelo
    IMAGE_SIZE = (160, 160)
    EMBEDDING_DIM = 128
    BATCH_SIZE = 32
    MARGIN = 0.2
    
    # Atributos para identidades sintéticas (si no hay identidades reales)
    SELECTED_ATTRIBUTES = [
        'Male', 'Young', 'Eyeglasses', 'Smiling', 
        'Bald', 'Mustache', 'Wearing_Lipstick'
    ]
    
    @classmethod
    def create_directories(cls):
        """Crea los directorios necesarios"""
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)