import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from data.data_generator import CelebADataGenerator
from models.face_embedding_model import FaceEmbeddingModel

def manual_triplet_loss(anchor, positive, negative, margin=0.2):
    """Calcula triplet loss manualmente"""
    anchor = tf.math.l2_normalize(anchor, axis=1)
    positive = tf.math.l2_normalize(positive, axis=1)
    negative = tf.math.l2_normalize(negative, axis=1)
    
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

def retrain_model():
    print("🔄 REENTRENANDO MODELO MEJORADO")
    print("=" * 50)
    
    # Configuraciones disponibles
    configs = {
        '1': {'name': 'RÁPIDO', 'samples': 2000, 'classes': 20, 'epochs': 5, 'batch_size': 16},
        '2': {'name': 'BALANCEADO', 'samples': 5000, 'classes': 50, 'epochs': 10, 'batch_size': 32},
        '3': {'name': 'COMPLETO', 'samples': 10000, 'classes': 100, 'epochs': 15, 'batch_size': 32},
        '4': {'name': 'AVANZADO', 'samples': 20000, 'classes': 200, 'epochs': 20, 'batch_size': 64}
    }
    
    print("🎯 CONFIGURACIONES DISPONIBLES:")
    for key, config in configs.items():
        print(f"   {key}. {config['name']}:")
        print(f"      - Muestras: {config['samples']:,}")
        print(f"      - Clases: {config['classes']}")
        print(f"      - Epochs: {config['epochs']}")
        print(f"      - Batch: {config['batch_size']}")
    
    choice = input("\nSelecciona configuración (1-4): ").strip()
    
    if choice not in configs:
        print("❌ Opción inválida, usando configuración BALANCEADO")
        choice = '2'
    
    config = configs[choice]
    
    print(f"\n🚀 INICIANDO REENTRENAMIENTO - {config['name']}")
    print("=" * 50)
    
    # 1. Cargar datos
    print("📥 Cargando datos...")
    data_gen = CelebADataGenerator('data')
    data_gen.load_dataset_simple(
        max_samples=config['samples'],
        num_classes=config['classes']
    )
    data_gen.analyze_dataset()
    
    # 2. Crear modelo (nuevo o cargar existente)
    print("🧠 Configurando modelo...")
    
    load_existing = input("¿Cargar modelo existente? (s/n): ").strip().lower()
    if load_existing == 's' and os.path.exists('saved_models/final_face_model.h5'):
        print("📥 Cargando modelo existente...")
        base_model = tf.keras.models.load_model('saved_models/final_face_model.h5')
        print("✅ Modelo existente cargado")
    else:
        print("🆕 Creando nuevo modelo...")
        model = FaceEmbeddingModel()
        base_model = model.get_embedding_model()
        print("✅ Nuevo modelo creado")
    
    # 3. Optimizador con learning rate ajustable
    lr = float(input("Learning rate (default 0.0001): ") or "0.0001")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # 4. Función de entrenamiento mejorada
    @tf.function
    def train_step(anchor_batch, positive_batch, negative_batch):
        with tf.GradientTape() as tape:
            anchor_emb = base_model(anchor_batch, training=True)
            positive_emb = base_model(positive_batch, training=True)
            negative_emb = base_model(negative_batch, training=True)
            loss = manual_triplet_loss(anchor_emb, positive_emb, negative_emb)
        
        gradients = tape.gradient(loss, base_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, base_model.trainable_variables))
        return loss
    
    # 5. Entrenamiento con métricas
    print("🔥 Iniciando entrenamiento...")
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)
    
    loss_history = []
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        print(f"\n📍 Epoch {epoch + 1}/{config['epochs']}")
        epoch_losses = []
        successful_steps = 0
        
        steps_per_epoch = len(data_gen.images) // config['batch_size']
        
        for step in range(steps_per_epoch):
            try:
                batch_data = data_gen.get_triplet_batch(config['batch_size'])
                if len(batch_data[0]) == 0:
                    continue
                
                loss = train_step(batch_data[0], batch_data[1], batch_data[2])
                loss_value = loss.numpy()
                epoch_losses.append(loss_value)
                successful_steps += 1
                
                # Progreso cada 10% de los steps
                if step % max(1, steps_per_epoch // 10) == 0:
                    progress = (step / steps_per_epoch) * 100
                    current_avg = np.mean(epoch_losses) if epoch_losses else 0
                    print(f"   █ {progress:3.0f}% - Loss: {loss_value:.4f} (Avg: {current_avg:.4f})")
                    
            except Exception as e:
                if step % 50 == 0:  # No spammear errores
                    print(f"   ⚠️  Error en step {step}: {str(e)[:50]}...")
                continue
        
        # Estadísticas de la epoch
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            loss_history.append(avg_loss)
            
            print(f"   ✅ Epoch completada - Steps: {successful_steps}/{steps_per_epoch}")
            print(f"   📉 Loss promedio: {avg_loss:.4f}")
            
            # Guardar el mejor modelo
            if avg_loss < best_loss:
                best_loss = avg_loss
                base_model.save('saved_models/best_model.h5')
                print(f"   🏆 ¡Nuevo mejor modelo! Loss: {avg_loss:.4f}")
            
            # Guardar checkpoint cada 5 epochs o al final
            if (epoch + 1) % 5 == 0 or (epoch + 1) == config['epochs']:
                base_model.save(f'saved_models/checkpoint_epoch_{epoch+1}.h5')
                print(f"   💾 Checkpoint guardado: epoch_{epoch+1}")
        else:
            print(f"   ❌ Epoch fallida")
    
    # 6. Guardar modelo final y resultados
    base_model.save('saved_models/retrained_final_model.h5')
    
    # Guardar historial de entrenamiento
    np.save('training_logs/loss_history.npy', loss_history)
    
    print(f"\n🎉 ¡REENTRENAMIENTO COMPLETADO!")
    print("=" * 50)
    print(f"📊 CONFIGURACIÓN: {config['name']}")
    print(f"📈 Pérdida final: {loss_history[-1]:.4f}" if loss_history else "N/A")
    print(f"📉 Mejor pérdida: {best_loss:.4f}")
    print(f"📈 Historial: {[f'{loss:.4f}' for loss in loss_history]}")
    
    # Mostrar gráfica de progreso si matplotlib está disponible
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, 'b-', linewidth=2, label='Pérdida')
        plt.title('Progreso del Entrenamiento')
        plt.xlabel('Epoch')
        plt.ylabel('Triplet Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('training_logs/training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Gráfica guardada en: training_logs/training_progress.png")
    except:
        print("💡 Instala matplotlib para ver gráficas: pip install matplotlib")

if __name__ == "__main__":
    retrain_model()