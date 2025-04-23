import smtplib
import random
import openai
import json
import logging
from pinecone import Pinecone
from config import (
    PINECONE_API_KEY,
    PRODUCT_INDEX_NAME,
    REVIEW_INDEX_NAME,
    OPENAI_API_KEY
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
openai.api_key = OPENAI_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)

def generate_embeddings(text: str, model: str = "text-embedding-ada-002", dimensions: int = None):
    try:
        logging.info(f"Generating embedding for: {text[:50]}...")
        if dimensions:
            response = openai.Embedding.create(input=text, model=model, dimensions=dimensions)
        else:
            response = openai.Embedding.create(input=text, model=model)

        logging.info(f"Embedding successful.")
        return response["data"][0]["embedding"]

    except openai.error.OpenAIError as e:
        logging.error(f"Embedding API Error: {e}")
        return None

def upsert_documents(index, documents: list, batch_size=50):
    try:
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            index.upsert(vectors=batch)
            logging.info(f"Uploaded batch {i // batch_size + 1}")
    except Exception as e:
        logging.error(f"Error during upsert: {e}")
