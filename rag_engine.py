from embedder import get_embedding
from vector_store import VectorStore
from openai import OpenAI

client = OpenAI()

def answer_query(query, vectorstore: VectorStore):
    q_embedding = get_embedding(query)
    top_chunks = vectorstore.search(q_embedding, top_k=5)
    context = "\n\n".join([c["content"] for c in top_chunks])

    prompt = f"""Réponds uniquement à partir du texte suivant, si la réponse n'est pas dans le texte, réponds "Je ne sais pas":

{context}

Question : {query}
Réponse :"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content
