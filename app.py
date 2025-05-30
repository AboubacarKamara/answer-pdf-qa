from fastapi import FastAPI, Request
from rag_engine import answer_query
import pickle

app = FastAPI()

# Charger l’index sauvegardé
with open("data/vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)


@app.get("/")
def root():
    return {"message": "API en ligne ✅"}


@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = body.get("question")
    if not question:
        return {"error": "Aucune question fournie."}
    print(f"❓ Question reçue : {question}")
    answer = answer_query(question, vectorstore)
    print(f"✅ Réponse générée")
    return {"answer": answer}
