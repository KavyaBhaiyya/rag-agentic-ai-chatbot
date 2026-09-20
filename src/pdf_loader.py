"""Download the source PDF (if needed) and extract clean text per page."""
import os
import re
from typing import List, Dict

import requests
from pypdf import PdfReader

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)


def download_pdf(url: str = None, dest: str = None) -> str:
    """Download the PDF to disk if not already present. Returns local path."""
    url = url or settings.SOURCE_PDF_URL
    dest = dest or settings.LOCAL_PDF_PATH
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        log.info(f"Using cached PDF at {dest}")
        return dest

    log.info(f"Downloading PDF from {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download source PDF from {url}: {e}") from e

    with open(dest, "wb") as f:
        f.write(resp.content)
    log.info(f"Saved PDF to {dest} ({len(resp.content)} bytes)")
    return dest


def clean_text(text: str) -> str:
    """Basic whitespace / artifact cleanup for extracted PDF text."""
    if not text:
        return ""
    # collapse hyphenated line-breaks: "agen-\ntic" -> "agentic"
    text = re.sub(r"-\n(\w)", r"\1", text)
    # normalize newlines/whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # drop stray page-number-only lines
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    return text.strip()


def extract_pages(pdf_path: str) -> List[Dict]:
    """Return a list of {page_number, text} for every non-empty page."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:
            pages.append({"page_number": i, "text": cleaned})
    if not pages:
        raise RuntimeError("No extractable text found in PDF (it may be scanned/image-only).")
    log.info(f"Extracted text from {len(pages)} pages")
    return pages
