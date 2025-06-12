import os
from pdf_loader import load_and_split_pdf
from embedder import get_embedding
from vector_store import VectorStore
import pickle
import faiss
import numpy as np

def init_index():
    """
    Initialise l'index FAISS pour la recherche de similarité.
    Cette fonction charge un PDF, génère des embeddings et crée un index FAISS.
    """
    print("Initialisation de l'index...")
    # 1. Charger et découper le PDF
    chunks = load_and_split_pdf("./Cours.pdf")

    # chunks = chunks[:50]

    # 2. Générer les embeddings
    embeddings = [get_embedding(chunk["content"]) for chunk in chunks]

    # Convertir en numpy array
    embeddings = np.array(embeddings).astype("float32")

    # 3. Créer l'index
    vectorstore = VectorStore()
    vectorstore.add(embeddings, chunks)

    # 4. Créer l’index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Index FAISS créé avec {index.ntotal} vecteurs")

    # 5. Sauvegarder l’index et les chunks
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/faiss.index")

    # 6. Sauvegarder le vecteur et le texte associé
    vectorstore.index = None
    with open("data/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    print("Index construit avec succès.")
