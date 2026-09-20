"""Centralized config, loaded from environment variables (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class Settings:
    # Pinecone
    PINECONE_API_KEY = _get("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = _get("PINECONE_INDEX_NAME", "agentic-ai-ebook")
    PINECONE_CLOUD = _get("PINECONE_CLOUD", "aws")
    PINECONE_REGION = _get("PINECONE_REGION", "us-east-1")

    # Embeddings (Pinecone hosted inference)
    EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "multilingual-e5-large")
    EMBEDDING_DIM = 1024

    # LLM
    GROQ_API_KEY = _get("GROQ_API_KEY")
    GROQ_MODEL = _get("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Source doc
    SOURCE_PDF_URL = _get("SOURCE_PDF_URL", "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf")
    LOCAL_PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "source.pdf")

    # Chunking
    CHUNK_SIZE = int(_get("CHUNK_SIZE", "900"))
    CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "150"))

    # Retrieval / guardrails
    TOP_K = int(_get("TOP_K", "5"))
    RELEVANCE_SCORE_THRESHOLD = float(_get("RELEVANCE_SCORE_THRESHOLD", "0.72"))
    GROUNDEDNESS_THRESHOLD = float(_get("GROUNDEDNESS_THRESHOLD", "0.6"))
    MAX_QUERY_CHARS = int(_get("MAX_QUERY_CHARS", "800"))
    MIN_QUERY_CHARS = 2

    # App
    LOG_LEVEL = _get("LOG_LEVEL", "INFO")


settings = Settings()
