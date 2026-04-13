"""PDF text extraction using PyMuPDF with page-level structure."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from raw_to_embedding.models import PdfPage, RawDocument

logger = logging.getLogger(__name__)


def extract_pdf(path: Path, source_label: str | None = None) -> RawDocument:
    """Extract text per page; page numbers are 1-based."""
    path = path.resolve()
    label = source_label or str(path)
    pages: list[PdfPage] = []
    try:
        doc = fitz.open(path)
    except Exception as exc:
        logger.exception("Failed to open PDF: %s", path)
        raise RuntimeError(f"Cannot open PDF: {path}") from exc
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages.append(PdfPage(page_number=i + 1, text=text))
    finally:
        doc.close()
    raw_text = "\n\n".join(p.text for p in pages)
    return RawDocument(
        source=label,
        source_type="pdf",
        raw_text=raw_text,
        path=str(path),
        url=None,
        pages=pages,
        sections=None,
    )
