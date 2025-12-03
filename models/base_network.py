import tensorflow as tf
from keras import layers, Model

class BaseNetwork:
    def __init__(self, input_shape=(160, 160, 3), embedding_dim=128):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
    
    def build_inception_module(self, x, filters):
        """Módulo Inception para extracción de características"""
        # Rama 1x1
        branch1 = layers.Conv2D(filters[0], 1, activation='relu')(x)
        
        # Rama 1x1 -> 3x3
        branch2 = layers.Conv2D(filters[1], 1, activation='relu')(x)
        branch2 = layers.Conv2D(filters[2], 3, padding='same', activation='relu')(branch2)
        
        # Rama 1x1 -> 5x5
        branch3 = layers.Conv2D(filters[3], 1, activation='relu')(x)
        branch3 = layers.Conv2D(filters[4], 5, padding='same', activation='relu')(branch3)
        
        # Rama pooling
        branch4 = layers.MaxPooling2D(3, strides=1, padding='same')(x)
        branch4 = layers.Conv2D(filters[5], 1, activation='relu')(branch4)
        
        return layers.concatenate([branch1, branch2, branch3, branch4])
    
    def build_model(self):
        """Construye la red base para extracción de embeddings"""
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Capa inicial
        x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2)(x)
        
        # Capas convolucionales
        x = layers.Conv2D(64, 1, activation='relu')(x)
        x = layers.Conv2D(192, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2)(x)
        
        # Módulos Inception
        x = self.build_inception_module(x, [64, 96, 128, 16, 32, 32])
        x = self.build_inception_module(x, [128, 128, 192, 32, 96, 64])
        
        # Capas finales
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Capa de embedding - SIN normalización L2 aquí
        embeddings = layers.Dense(self.embedding_dim, name='embeddings')(x)
        
        # Modelo simple sin capas lambda complicadas
        return Model(inputs, embeddings, name='base_network')