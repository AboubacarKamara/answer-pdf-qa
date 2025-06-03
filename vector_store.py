import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim=1536):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, metadata):
        self.index.add(np.array(embeddings).astype('float32'))
        self.texts.extend(metadata)
    
    def search(self, query_embedding, top_k=5):
        # 1. Transformer l'embedding en tableau NumPy (1, d)
        q_emb = np.array(query_embedding, dtype='float32').reshape(1, -1)

        # 2. Préparer les buffers pour les résultats
        distances = np.empty((1, top_k), dtype='float32')
        labels = np.empty((1, top_k), dtype='int64')

        # 3. Appel bas niveau avec TOUS les arguments requis (5)
        self.index.search(
            1,                              # n
            faiss.swig_ptr(q_emb),         # x
            top_k,                         # k
            faiss.swig_ptr(distances),     # distances
            faiss.swig_ptr(labels)         # labels
        )

        # 4. Résultats lisibles
        results = [self.texts[i] for i in labels[0] if i >= 0 and i < len(self.texts)]
        return results





