import tensorflow as tf
from tensorflow.keras import losses

class TripletLoss:
    def __init__(self, margin=0.2):
        self.margin = margin
        
    def __call__(self, y_true, y_pred):
        """
        Versión compatible con Keras
        y_pred debería ser una lista de 3 tensores: [anchor, positive, negative]
        """
        # Verificar que y_pred tenga 3 elementos
        if len(y_pred) != 3:
            raise ValueError(f"Se esperaban 3 salidas, pero se obtuvieron {len(y_pred)}")
        
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        
        # Normalizar embeddings
        anchor = tf.math.l2_normalize(anchor, axis=1)
        positive = tf.math.l2_normalize(positive, axis=1)
        negative = tf.math.l2_normalize(negative, axis=1)
        
        # Calcular distancias
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Calcular triplet loss
        basic_loss = pos_dist - neg_dist + self.margin
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        
        return loss

# Versión alternativa que funciona mejor con train_on_batch
class TripletLossWrapper:
    def __init__(self, margin=0.2):
        self.margin = margin
        
    def compute_loss(self, y_true, y_pred):
        """Versión para usar directamente con train_on_batch"""
        return TripletLoss(self.margin)(y_true, y_pred)