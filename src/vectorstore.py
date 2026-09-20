"""Thin wrapper around Pinecone: index lifecycle, upsert, query."""
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)


class VectorStoreError(RuntimeError):
    pass


def get_pinecone_client() -> Pinecone:
    if not settings.PINECONE_API_KEY:
        raise VectorStoreError(
            "PINECONE_API_KEY is missing. Set it in your .env file (see .env.example)."
        )
    return Pinecone(api_key=settings.PINECONE_API_KEY)


class VectorStore:
    def __init__(self, pc: Pinecone, index_name: str = None):
        self.pc = pc
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self._index = None

    def ensure_index(self, dim: int = settings.EMBEDDING_DIM):
        existing = [i["name"] for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            log.info(f"Creating Pinecone index '{self.index_name}' (dim={dim})")
            self.pc.create_index(
                name=self.index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION,
                ),
            )
        self._index = self.pc.Index(self.index_name)
        return self._index

    @property
    def index(self):
        if self._index is None:
            self._index = self.pc.Index(self.index_name)
        return self._index

    def upsert(self, records: List[Dict], vectors: List[List[float]], batch_size: int = 100):
        """records: chunk dicts with id/text/page_number. vectors: matching embeddings."""
        if len(records) != len(vectors):
            raise VectorStoreError("records and vectors length mismatch")

        items = []
        for rec, vec in zip(records, vectors):
            items.append({
                "id": rec["id"],
                "values": vec,
                "metadata": {
                    "text": rec["text"],
                    "page_number": rec["page_number"],
                },
            })

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
            except Exception as e:
                raise VectorStoreError(f"Upsert failed for batch starting at {i}: {e}") from e
        log.info(f"Upserted {len(items)} vectors into '{self.index_name}'")

    def query(self, vector: List[float], top_k: int = None) -> List[Dict]:
        top_k = top_k or settings.TOP_K
        try:
            result = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        except Exception as e:
            raise VectorStoreError(f"Pinecone query failed: {e}") from e

        matches = []
        for m in result.get("matches", []):
            meta = m.get("metadata", {})
            matches.append({
                "id": m.get("id"),
                "score": m.get("score", 0.0),
                "text": meta.get("text", ""),
                "page_number": meta.get("page_number"),
            })
        return matches
