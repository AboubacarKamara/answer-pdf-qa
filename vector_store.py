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
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        return [self.texts[i] for i in I[0]]
