import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_KEY") or os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "test2")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "rag_docs")

# OpenRouter / LLM settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "openai/gpt-4o-mini")

# Embedding models mapping by vector dimension
DIMENSION_EMBEDDING_MODELS = {
    1536: "openai/text-embedding-3-small",
    3072: "openai/text-embedding-3-large",
}
