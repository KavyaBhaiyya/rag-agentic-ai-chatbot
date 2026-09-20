"""Turn page-level text into overlapping chunks suitable for embedding."""
from typing import List, Dict
from src.config import settings


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Character-based sliding-window chunking with sentence-friendly breaks.

    Simple on purpose: split on whitespace into words, pack words into
    windows of ~chunk_size characters, overlap consecutive windows.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    current: List[str] = []
    current_len = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # keep the tail of this chunk as overlap for the next one
            overlap_words = []
            overlap_len = 0
            for w in reversed(current):
                overlap_len += len(w) + 1
                overlap_words.insert(0, w)
                if overlap_len >= overlap:
                    break
            current = overlap_words
            current_len = sum(len(w) + 1 for w in current)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """Chunk every page and attach metadata (page number, chunk index, id)."""
    records = []
    chunk_id = 0
    for page in pages:
        page_chunks = chunk_text(page["text"])
        for idx, chunk in enumerate(page_chunks):
            records.append({
                "id": f"chunk-{chunk_id}",
                "text": chunk,
                "page_number": page["page_number"],
                "chunk_index_on_page": idx,
            })
            chunk_id += 1
    return records
