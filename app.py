from fastapi import FastAPI, Request, HTTPException
from rag_engine import answer_query
from init_index import init_index
import pickle
import faiss

app = FastAPI()

# Charger l’index sauvegardé
with open("data/vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)


@app.get("/")
def root():
    return {"message": "API en ligne ✅"}

@app.get("/init_index")
def run_index_init():
    init_index()
    return {"status": "index created"}

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
    except Exception:
        return {"error": "Corps JSON invalide ou absent."}

    question = body.get("question")
    if not question:
        return {"error": "Aucune question fournie."}

    print(f"❓ Question reçue : {question}")
    answer = answer_query(question, vectorstore)
    print(f"✅ Réponse générée")
    return {"answer": answer}

@app.get("/debug/faiss_version")
async def faiss_version():
    return {
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "faiss_file": faiss.__file__,
        "faiss_type": str(type(faiss.IndexFlatL2(1)))
    }