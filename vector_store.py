import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, dim=1536):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, metadata):
        self.index.add(np.array(embeddings).astype('float32'))
        self.texts.extend(metadata)
    
    def search(self, query_embedding, top_k=5):
        q_emb = np.array([query_embedding]).astype('float32')
        try:
            D, I = self.index.search(q_emb, top_k)
        except Exception as e:
            print(f"Erreur faiss search: {e}")
            raise
        results = []
        for i in I[0]:
            if i < len(self.texts):
                results.append(self.texts[i])
        return results

