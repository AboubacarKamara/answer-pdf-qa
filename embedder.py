import openai
import os
from dotenv import load_dotenv

load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding
