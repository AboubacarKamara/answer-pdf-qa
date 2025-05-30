from openai import OpenAI
import faiss
import pickle
import numpy as np
from embedder import get_embedding
from vector_store import VectorStore
import os

# Initialisation clé API OpenAI (remplace par ta clé)
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 1. Charger l'index FAISS
index = faiss.read_index("data/faiss.index")

# 2. Charger VectorStore (avec les chunks)
with open("data/vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

# 3. Question utilisateur
question = "Quel est le devoir de toute personne responsable ?"

# 4. Calculer l'embedding de la question
q_embedding = get_embedding(question)

# 5. Recherche des top 3 chunks les plus proches
D, I = index.search(np.array([q_embedding]).astype("float32"), 3)

# 6. Construire un contexte à partir des chunks trouvés
context = "\n\n---\n\n".join([vectorstore.texts[idx]['content'] for idx in I[0]])

# 7. Préparer le prompt pour GPT (on inclut la question + contexte)
prompt = f"Voici des extraits de documents :\n{context}\n\nEn te basant uniquement sur ces extraits, réponds à la question suivante :\n{question}"

# 8. Appeler l'API ChatCompletion OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Tu es un assistant qui répond uniquement avec les infos données."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2,
    max_tokens=500
)

# 9. Afficher la réponse
print(response.choices[0].message.content)
