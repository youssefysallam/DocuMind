"""Deterministic document classification with optional path hints."""

from __future__ import annotations

import re
from pathlib import Path

from raw_to_embedding.models import DocumentType, RawDocument

_SCHOLARLY_PATTERNS = (
    r"\babstract\b",
    r"\bintroduction\b",
    r"\breferences\b",
    r"\bmethods?\b",
    r"\bresults?\b",
    r"\bdiscussion\b",
    r"\bconclusion\b",
    r"\bdoi:\s*10\.",
    r"\barxiv:",
    r"\bjournal of\b",
    r"\bproceedings of\b",
)

_INSTITUTE_PATTERNS = (
    r"\bannual report\b",
    r"\bimpact report\b",
    r"\bour mission\b",
    r"\bour team\b",
    r"\bwho we are\b",
    r"\bstrategic plan\b",
    r"\bdonor\b",
    r"\bboard of directors\b",
)


def _score_patterns(text: str, patterns: tuple[str, ...]) -> int:
    low = text.lower()
    return sum(1 for p in patterns if re.search(p, low, re.IGNORECASE))


def classify_document(raw: RawDocument, input_path: Path | None = None) -> DocumentType:
    """
    Classify into website_page, institute_report_pdf, or scholarly_paper_pdf.

    Rules-first: URL/HTML ingestion is always website_page.
    PDFs use keyword scoring plus optional directory/filename hints.
    """
    if raw.source_type == "website":
        return "website_page"

    hint = ""
    if input_path is not None:
        hint = f"{input_path.as_posix()} {input_path.name} "
    text = (hint + raw.raw_text)[:200000]
    low = text.lower()

    scholarly = _score_patterns(low, _SCHOLARLY_PATTERNS)
    institute = _score_patterns(low, _INSTITUTE_PATTERNS)

    if input_path is not None:
        parts = [p.lower() for p in input_path.parts]
        if any("scholar" in p or "paper" in p or "arxiv" in p for p in parts):
            scholarly += 3
        if any("report" in p or "institute" in p or "annual" in p for p in parts):
            institute += 2

    if scholarly >= 4 and scholarly >= institute:
        return "scholarly_paper_pdf"
    if institute >= 2 and institute > scholarly:
        return "institute_report_pdf"
    if scholarly >= 3:
        return "scholarly_paper_pdf"
    return "institute_report_pdf"
