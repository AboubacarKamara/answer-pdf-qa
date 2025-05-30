from fastapi import FastAPI, Request
from rag_engine import answer_query
import pickle

app = FastAPI()

# Charger l’index sauvegardé
with open("data/vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = body.get("question")
    answer = answer_query(question, vectorstore)
    return {"answer": answer}
