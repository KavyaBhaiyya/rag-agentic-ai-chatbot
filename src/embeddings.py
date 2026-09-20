"""Embedding generation via Pinecone's hosted Inference API.

Using Pinecone's own inference endpoint (instead of a separate embedding
provider) keeps the project to one fewer API key/dependency while still
satisfying the "text embeddings" requirement.
"""
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    def __init__(self, pinecone_client):
        self.pc = pinecone_client
        self.model = settings.EMBEDDING_MODEL

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def _embed_batch(self, texts: List[str], input_type: str) -> List[List[float]]:
        try:
            result = self.pc.inference.embed(
                model=self.model,
                inputs=texts,
                parameters={"input_type": input_type, "truncate": "END"},
            )
        except Exception as e:
            log.error(f"Embedding call failed: {e}")
            raise
        return [item["values"] for item in result]

    def embed_documents(self, texts: List[str], batch_size: int = 96) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                vectors.extend(self._embed_batch(batch, "passage"))
            except Exception as e:
                raise EmbeddingError(f"Failed to embed document batch starting at {i}: {e}") from e
        return vectors

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed an empty query")
        try:
            return self._embed_batch([text], "query")[0]
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}") from e
