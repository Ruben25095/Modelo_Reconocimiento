import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data.data_generator import CelebADataGenerator

def visualize_embeddings():
    print("📊 VISUALIZANDO EMBEDDINGS")
    print("=" * 50)
    
    # 1. Cargar modelo
    model_path = 'saved_models/final_face_model.h5'
    if not os.path.exists(model_path):
        print("❌ Modelo no encontrado")
        return
    
    model = tf.keras.models.load_model(model_path)
    
    # 2. Cargar datos
    data_gen = CelebADataGenerator('data')
    data_gen.load_dataset_simple(max_samples=200, num_classes=5)
    
    # 3. Extraer embeddings
    print("🎯 Extrayendo embeddings...")
    embeddings = []
    labels = []
    
    for i in range(min(50, len(data_gen.images))):
        try:
            # Cargar y preprocesar imagen
            img = data_gen.preprocessor.load_and_preprocess(data_gen.images[i])
            if img is not None:
                # Extraer embedding
                emb = model.predict(np.expand_dims(img, axis=0), verbose=0)
                embeddings.append(emb[0])
                labels.append(data_gen.labels[i])
        except:
            continue
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"✅ {len(embeddings)} embeddings extraídos")
    
    # 4. Reducir dimensionalidad con t-SNE
    print("🔍 Aplicando t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 5. Visualizar
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Clases')
    plt.title('Visualización de Embeddings (t-SNE)')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.grid(True, alpha=0.3)
    
    # Guardar figura
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/embeddings_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Visualización guardada en: results/embeddings_visualization.png")

if __name__ == "__main__":
    visualize_embeddings()