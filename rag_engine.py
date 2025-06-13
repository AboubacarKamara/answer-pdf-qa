from embedder import get_embedding
from vector_store import VectorStore
from openai import OpenAI

client = OpenAI()

def answer_query(query, vectorstore: VectorStore):
    # Obtenir l'embedding de la requête
    q_embedding = get_embedding(query)

    # Recherche avec reranking + contexte élargi
    results = vectorstore.search_with_context(
        query=query,
        query_embedding=q_embedding,
        faiss_top_k=30,        # Nombre initial de résultats FAISS
        final_top_k=5,         # Nombre de top passages pertinents
        context_size=10,       # Nombre de passages de contexte supplémentaires
        window_size=2          # Fenêtre contextuelle autour des passages
    )

    # Construire le contexte pour le prompt
    context_chunks = results["all_used_chunks"]
    context = "\n\n".join(context_chunks)

    # Créer le prompt
    prompt = f"""Réponds uniquement à partir du texte suivant. Si la réponse n’est pas clairement indiquée dans le texte, dis simplement : "Je ne sais pas."

{context}

Question : {query}
Réponse :"""

    # Appel à OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content

