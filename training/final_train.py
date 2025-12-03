import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from data.data_generator import CelebADataGenerator
from models.face_embedding_model import FaceEmbeddingModel

def manual_triplet_loss(anchor, positive, negative, margin=0.2):
    """Calcula triplet loss manualmente"""
    # Normalizar embeddings
    anchor = tf.math.l2_normalize(anchor, axis=1)
    positive = tf.math.l2_normalize(positive, axis=1)
    negative = tf.math.l2_normalize(negative, axis=1)
    
    # Calcular distancias
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    # Calcular pérdida
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

def final_train():
    print("🚀 ENTRENAMIENTO DEFINITIVO")
    print("=" * 50)
    
    # Configuración
    MAX_SAMPLES = 2000
    NUM_CLASSES = 20
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 0.0001
    
    # 1. Cargar datos
    print("📥 Cargando datos...")
    data_gen = CelebADataGenerator('data')
    data_gen.load_dataset_simple(
        max_samples=MAX_SAMPLES,
        num_classes=NUM_CLASSES
    )
    data_gen.analyze_dataset()
    
    # 2. Crear modelo
    print("🧠 Creando modelo...")
    model = FaceEmbeddingModel()
    
    # 3. Obtener el modelo base para embeddings
    base_model = model.get_embedding_model()
    
    # 4. Optimizador
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    
    # 5. Función de entrenamiento con @tf.function para mejor rendimiento
    @tf.function
    def train_step(anchor_batch, positive_batch, negative_batch):
        with tf.GradientTape() as tape:
            # Forward pass a través del mismo modelo base
            anchor_emb = base_model(anchor_batch, training=True)
            positive_emb = base_model(positive_batch, training=True)
            negative_emb = base_model(negative_batch, training=True)
            
            # Calcular pérdida
            loss = manual_triplet_loss(anchor_emb, positive_emb, negative_emb)
        
        # Calcular gradientes
        gradients = tape.gradient(loss, base_model.trainable_variables)
        
        # Aplicar gradientes
        optimizer.apply_gradients(zip(gradients, base_model.trainable_variables))
        
        return loss
    
    # 6. Entrenamiento
    print("🔥 Iniciando entrenamiento...")
    os.makedirs('saved_models', exist_ok=True)
    
    # Historial de pérdidas
    loss_history = []
    
    for epoch in range(EPOCHS):
        print(f"\n📍 Epoch {epoch + 1}/{EPOCHS}")
        epoch_losses = []
        successful_steps = 0
        
        # Calcular steps por epoch
        steps_per_epoch = len(data_gen.images) // BATCH_SIZE
        
        for step in range(steps_per_epoch):
            try:
                # Obtener batch
                batch_data = data_gen.get_triplet_batch(BATCH_SIZE)
                anchor_batch, positive_batch, negative_batch = batch_data
                
                # Entrenar
                loss = train_step(anchor_batch, positive_batch, negative_batch)
                loss_value = loss.numpy()
                epoch_losses.append(loss_value)
                successful_steps += 1
                
                # Mostrar progreso
                if step % 10 == 0:
                    avg_loss = np.mean(epoch_losses) if epoch_losses else 0
                    print(f"   Step {step:3d}/{steps_per_epoch} - Loss: {loss_value:.4f} (Avg: {avg_loss:.4f})")
                    
            except Exception as e:
                print(f"   ⚠️  Error en step {step}: {str(e)[:80]}...")
                continue
        
        # Estadísticas de la epoch
        if epoch_losses:
            avg_epoch_loss = np.mean(epoch_losses)
            loss_history.append(avg_epoch_loss)
            print(f"   ✅ Epoch completada - Steps exitosos: {successful_steps}/{steps_per_epoch}")
            print(f"   📉 Loss promedio: {avg_epoch_loss:.4f}")
        else:
            print(f"   ❌ Epoch fallida - No se pudo entrenar")
            continue
        
        # Guardar checkpoint
        checkpoint_path = f'saved_models/epoch_{epoch+1}.h5'
        base_model.save(checkpoint_path)
        print(f"   💾 Modelo guardado: {checkpoint_path}")
    
    # 7. Guardar modelo final
    final_path = 'saved_models/final_face_model.h5'
    base_model.save(final_path)
    print(f"\n💾 Modelo final guardado: {final_path}")
    
    # 8. Resumen
    print("\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
    print("=" * 50)
    print(f"📈 Pérdida final: {loss_history[-1]:.4f}" if loss_history else "N/A")
    print(f"📊 Historial de pérdidas: {[f'{loss:.4f}' for loss in loss_history]}")
    print(f"🔍 Para usar el modelo: python test_model.py")

if __name__ == "__main__":
    final_train()