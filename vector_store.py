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
        # Conversion propre du vecteur renvoyé par OpenAI (liste) en np.ndarray
        q_emb = np.array(query_embedding, dtype='float32').reshape(1, -1)

        distances = np.empty((1, top_k), dtype='float32')
        labels = np.empty((1, top_k), dtype='int64')

        # Appel à l'API SWIG avec les bons types et dans le bon ordre
        self.index.search(
            faiss.swig_ptr(q_emb),   # const float*
            top_k,                   # idx_t k
            faiss.swig_ptr(distances),
            faiss.swig_ptr(labels)
        )

        results = [self.texts[i] for i in labels[0] if i >= 0 and i < len(self.texts)]
        return results




