from flask import Flask, request, jsonify
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import OpenAI
import pdfplumber
import logging
import os

# Réduction du bruit PDF
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# --- CONFIGURATION ---
PDF_PATH = "./Cours.pdf"
OPENAI_API_KEY = "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"  # Remplace ici
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Extraction texte PDF ---
with pdfplumber.open(PDF_PATH) as pdf:
    pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
full_text = "\n".join(pages)
print(f"[INFO] Texte extrait : {len(full_text)} caractères")

# --- Split en chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_text(full_text)

# Encapsule chaque chunk dans un objet Document (meilleure gestion du contexte)
docs = [Document(page_content=t, metadata={"source": f"page_{i}"}) for i, t in enumerate(texts)]

# --- Embedding + Vectorstore ---
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# --- Setup LLM + QA ---
llm = OpenAI(temperature=0)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --- API Flask ---
app = Flask(__name__)

@app.route("/ask", methods=["GET"])
def ask():
    question = request.args.get("question", "")
    if not question:
        return jsonify({"error": "Paramètre 'question' requis"}), 400

    prompt = (
        f"Réponds uniquement à partir du contenu du document fourni. "
        f"Si aucune information ne permet de répondre, indique : 'Non trouvé dans le document.'\n\n"
        f"Question : {question}"
    )

    result = qa_chain(prompt)

    return jsonify({
        "question": question,
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]],
    })

if __name__ == "__main__":
    app.run(port=5000)
