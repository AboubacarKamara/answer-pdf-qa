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
        q_emb = np.array([query_embedding], dtype='float32')
        n, d = q_emb.shape

        n = np.int64(n)
        top_k = np.int64(top_k)

        distances = np.empty((n, top_k), dtype='float32')
        labels = np.empty((n, top_k), dtype='int64')

        self.index.search(n, faiss.swig_ptr(q_emb), top_k,
                        faiss.swig_ptr(distances), faiss.swig_ptr(labels))

        results = []
        for i in labels[0]:
            if i < len(self.texts):
                results.append(self.texts[i])
        return results



