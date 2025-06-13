from sentence_transformers import CrossEncoder
import numpy as np
import faiss

class VectorStore:
    def __init__(self, dim=1536):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.embeddings = []
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def add(self, embeddings, metadata):
        self.index.add(np.array(embeddings).astype('float32'))
        self.embeddings.extend(embeddings)
        self.texts.extend(metadata)

    def search_with_context(
        self, query, query_embedding, 
        faiss_top_k=20, final_top_k=5, context_size=10, window_size=2
    ):
        q_emb = np.array([query_embedding]).astype('float32')
        D, I = self.index.search(q_emb, faiss_top_k)

        # Préparer les candidats avec concaténation contextuelle
        candidates = []
        seen_indices = set()
        for i in I[0]:
            if i < len(self.texts):
                chunk = ""
                for j in range(max(0, i - window_size), min(i + window_size + 1, len(self.texts))):
                    if j not in seen_indices:
                        chunk += self.texts[j]["content"] + " "
                        seen_indices.add(j)
                candidates.append(chunk.strip())

        # Reranking
        pairs = [[query, passage] for passage in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        # Résultats principaux + contexte complémentaire
        top_hits = [x[1] for x in ranked[:final_top_k]]
        context_chunks = [x[1] for x in ranked[final_top_k:final_top_k + context_size]]

        return {
            "top_hits": top_hits,
            "context_chunks": context_chunks,
            "all_used_chunks": top_hits + context_chunks
        }
