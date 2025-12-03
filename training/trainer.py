import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
import numpy as np

class FaceModelTrainer:
    def __init__(self, model, data_generator):
        self.model = model
        self.data_generator = data_generator
    
    def compile_model(self, learning_rate=0.0001):
        """Compila el modelo para entrenamiento"""
        from models.triplet_loss import TripletLoss
        
        self.model.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=TripletLoss(margin=0.2)
        )
    
    def _create_tf_dataset(self, batch_size=32):
        """Crea un tf.data.Dataset desde el generador"""
        def generator_wrapper():
            for batch_data, _ in self.data_generator.data_generator(batch_size):
                # batch_data es [anchors, positives, negatives]
                inputs = {
                    'anchor_input': batch_data[0],
                    'positive_input': batch_data[1], 
                    'negative_input': batch_data[2]
                }
                # Para triplet loss, las salidas son las mismas embeddings
                # pero la pérdida se calcula internamente
                yield inputs, [batch_data[0], batch_data[1], batch_data[2]]  # Dummy outputs
        
        # Definir la estructura de output_signature correctamente
        output_signature = (
            {
                'anchor_input': tf.TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32),
                'positive_input': tf.TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32),
                'negative_input': tf.TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32)
            },
            (
                tf.TensorSpec(shape=(None, 128), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 128), dtype=tf.float32), 
                tf.TensorSpec(shape=(None, 128), dtype=tf.float32)
            )
        )
        
        dataset = tf.data.Dataset.from_generator(
            generator_wrapper,
            output_signature=output_signature
        )
        
        return dataset
    
    def train_simple_generator(self, epochs=50, batch_size=32, model_save_path='saved_models/'):
        """Entrenamiento usando generador simple"""
        os.makedirs(model_save_path, exist_ok=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(model_save_path, 'best_face_model.h5'),
                monitor='loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='loss',
                patience=5,
                factor=0.5,
                verbose=1
            ),
            EarlyStopping(
                monitor='loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Calcular steps por epoch
        steps_per_epoch = max(1, len(self.data_generator.images) // batch_size)
        
        print(f"Steps por epoch: {steps_per_epoch}")
        print(f"Batch size: {batch_size}")
        print(f"Total imágenes: {len(self.data_generator.images)}")
        
        # Entrenar con generador simple
        history = self.model.model.fit(
            self.data_generator.data_generator(batch_size),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar modelo final
        self.model.model.save(os.path.join(model_save_path, 'final_face_model.h5'))
        
        return history
    
    def train(self, epochs=50, batch_size=32, model_save_path='saved_models/'):
        """Método principal de entrenamiento"""
        return self.train_simple_generator(epochs, batch_size, model_save_path)