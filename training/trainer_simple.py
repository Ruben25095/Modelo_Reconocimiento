import tensorflow as tf
import os
import numpy as np

class RobustFaceTrainer:
    def __init__(self, model, data_generator):
        self.model = model
        self.data_generator = data_generator
        self.loss_history = []
    
    def compile_model(self, learning_rate=0.0001):
        """Compila el modelo de forma compatible"""
        from models.triplet_loss import TripletLoss
        
        # Compilar con pérdida personalizada
        self.model.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=TripletLoss(margin=0.2)
        )
        print("✅ Modelo compilado con TripletLoss")
    
    def train_robust(self, epochs=10, batch_size=16, model_save_path='saved_models/'):
        """Entrenamiento robusto con manejo mejorado de errores"""
        os.makedirs(model_save_path, exist_ok=True)
        
        print(f"🔥 Iniciando entrenamiento robusto...")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Imágenes: {len(self.data_generator.images)}")
        
        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch + 1}/{epochs}")
            epoch_losses = []
            successful_steps = 0
            
            # Calcular steps por epoch
            steps_per_epoch = max(1, len(self.data_generator.images) // batch_size)
            
            for step in range(steps_per_epoch):
                try:
                    # Obtener batch
                    batch_data = self.data_generator.get_triplet_batch(batch_size)
                    
                    if len(batch_data[0]) == 0:
                        continue
                    
                    # Crear datos de entrenamiento compatibles
                    # Para triplet loss, las "etiquetas" no se usan realmente
                    dummy_labels = [
                        np.zeros(len(batch_data[0])),  # Para anchor
                        np.zeros(len(batch_data[0])),  # Para positive  
                        np.zeros(len(batch_data[0]))   # Para negative
                    ]
                    
                    # Entrenar con el batch
                    loss = self.model.model.train_on_batch(batch_data, dummy_labels)
                    epoch_losses.append(loss)
                    successful_steps += 1
                    
                    if step % 10 == 0:
                        current_loss = np.mean(epoch_losses) if epoch_losses else 0
                        print(f"   Step {step:3d}/{steps_per_epoch} - Loss: {loss:.4f} (Avg: {current_loss:.4f})")
                        
                except Exception as e:
                    print(f"   ⚠️  Error en step {step}: {str(e)[:100]}...")
                    continue
            
            # Estadísticas de la epoch
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                self.loss_history.append(avg_loss)
                print(f"   ✅ Epoch completada - Steps exitosos: {successful_steps}/{steps_per_epoch}")
                print(f"   📉 Loss promedio: {avg_loss:.4f}")
            else:
                print(f"   ❌ Epoch fallida - No se pudo entrenar")
                continue
            
            # Guardar checkpoint
            if (epoch + 1) % 2 == 0 or (epoch + 1) == epochs:
                checkpoint_path = f'{model_save_path}/checkpoint_epoch_{epoch+1}.h5'
                self.model.model.save(checkpoint_path)
                print(f"   💾 Checkpoint guardado: {checkpoint_path}")
        
        # Guardar modelo final
        final_path = f'{model_save_path}/final_model.h5'
        self.model.model.save(final_path)
        print(f"\n💾 Modelo final guardado: {final_path}")
        
        return {"loss": self.loss_history}