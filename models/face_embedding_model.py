import tensorflow as tf
from keras import Model
from .base_network import BaseNetwork

class FaceEmbeddingModel:
    def __init__(self, input_shape=(160, 160, 3), embedding_dim=128):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.base_network = BaseNetwork(input_shape, embedding_dim)
        self.model = self._build_triplet_model()
    
    def _build_triplet_model(self):
        """Construye modelo para entrenamiento con triplet loss"""
        # Crear red base
        base_network = self.base_network.build_model()
        
        # Definir inputs para tripletes
        anchor_input = tf.keras.Input(shape=self.input_shape, name='anchor_input')
        positive_input = tf.keras.Input(shape=self.input_shape, name='positive_input')  
        negative_input = tf.keras.Input(shape=self.input_shape, name='negative_input')
        
        # Generar embeddings - usar la misma base_network para todos
        anchor_embedding = base_network(anchor_input)
        positive_embedding = base_network(positive_input)
        negative_embedding = base_network(negative_input)
        
        # Modelo de entrenamiento - SOLO las embeddings como salida
        training_model = Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=[anchor_embedding, positive_embedding, negative_embedding]
        )
        
        return training_model
    
    def get_embedding_model(self):
        """Retorna modelo para extracción de embeddings"""
        return self.base_network.build_model()
    
    def save_model(self, filepath):
        """Guarda el modelo completo"""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Carga modelo pre-entrenado"""
        self.model = tf.keras.models.load_model(
            filepath, 
            custom_objects={'TripletLoss': TripletLoss}
        )