import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

PRODUCT_INDEX_NAME = os.getenv("PRODUCT_INDEX_NAME")
PRODUCT_DIMENSION = int(os.getenv("PRODUCT_DIMENSION", 1536))

REVIEW_INDEX_NAME = os.getenv("REVIEW_INDEX_NAME")
REVIEW_DIMENSION = int(os.getenv("REVIEW_DIMENSION", 768))

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PRODUCT_INDEX_NAME, REVIEW_INDEX_NAME]):
    raise ValueError("Missing required environment variables.")