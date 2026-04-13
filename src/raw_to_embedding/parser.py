"""
Document parsing: PDF pages and website HTML → structured ``RawDocument``.

The canonical implementations live under ``extractors/``; this module
re-exports stable entry points for a clear public API.
"""
from __future__ import annotations

from raw_to_embedding.extractors.pdf_extractor import extract_pdf
from raw_to_embedding.extractors.website_extractor import fetch_website

__all__ = ["extract_pdf", "fetch_website"]
