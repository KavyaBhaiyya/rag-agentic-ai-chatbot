from src.chunking import chunk_text, chunk_pages
from src.pdf_loader import clean_text


def test_clean_text_fixes_hyphenation():
    raw = "This is agen-\ntic AI in practice."
    assert "agentic" in clean_text(raw)


def test_clean_text_removes_stray_page_numbers():
    raw = "Some content here.\n\n42\n\nMore content follows."
    cleaned = clean_text(raw)
    assert "\n42\n" not in cleaned


def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_chunk_text_respects_overlap_smaller_than_size():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # every chunk should be non-empty
    assert all(c.strip() for c in chunks)


def test_chunk_text_rejects_bad_overlap():
    try:
        chunk_text("hello world", chunk_size=100, overlap=200)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_pages_assigns_metadata():
    pages = [
        {"page_number": 1, "text": "word " * 400},
        {"page_number": 2, "text": "another " * 300},
    ]
    records = chunk_pages(pages)
    assert len(records) > 0
    for r in records:
        assert "id" in r and "text" in r and "page_number" in r
    # ids must be unique
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids))
    # page numbers preserved
    assert {r["page_number"] for r in records} == {1, 2}
