"""CLI: run the full ingestion pipeline once to (re)populate Pinecone.

Usage:
    python -m src.ingest
    python -m src.ingest --pdf path/to/local.pdf
"""
import argparse
import sys

from src.config import settings
from src.logger import get_logger
from src.pdf_loader import download_pdf, extract_pages
from src.chunking import chunk_pages
from src.embeddings import Embedder, EmbeddingError
from src.vectorstore import get_pinecone_client, VectorStore, VectorStoreError

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest the source PDF into Pinecone.")
    parser.add_argument("--pdf", help="Path to a local PDF (skips download)", default=None)
    args = parser.parse_args()

    try:
        pdf_path = args.pdf or download_pdf()
        pages = extract_pages(pdf_path)
        records = chunk_pages(pages)
        log.info(f"Built {len(records)} chunks from {len(pages)} pages")

        pc = get_pinecone_client()
        store = VectorStore(pc)
        store.ensure_index()

        embedder = Embedder(pc)
        texts = [r["text"] for r in records]
        vectors = embedder.embed_documents(texts)

        store.upsert(records, vectors)
        log.info("Ingestion complete.")
    except (EmbeddingError, VectorStoreError, RuntimeError) as e:
        log.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
