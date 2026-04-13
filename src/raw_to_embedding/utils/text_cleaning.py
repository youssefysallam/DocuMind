"""Traditional text cleaning before any LLM call."""

from __future__ import annotations

import re
from typing import Iterable


_WS = re.compile(r"[ \t]+")
_NEWLINE_RUN = re.compile(r"\n{3,}")
# Lines that look like isolated page numbers
_PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)


def normalize_whitespace(text: str) -> str:
    """Collapse spaces; normalize newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = _WS.sub(" ", line.strip())
        lines.append(line)
    text = "\n".join(lines)
    text = _NEWLINE_RUN.sub("\n\n", text)
    return text.strip()


def merge_broken_lines(text: str) -> str:
    """Join hyphenated line breaks and soft wraps; keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # hyphen + newline + lowercase continuation
    text = re.sub(r"-\n([a-z])", r"\1", text)
    # single newline inside a paragraph -> space if not heading-like
    lines = text.split("\n")
    merged: list[str] = []
    buf = ""
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append("")
            continue
        if not buf:
            buf = s
            continue
        if s[:1].islower() and buf[-1] not in ".!?:":
            buf = f"{buf} {s}"
        elif len(buf) < 500 and buf[-1] not in ".!?:" and not s.startswith("#"):
            buf = f"{buf} {s}"
        else:
            merged.append(buf)
            buf = s
    if buf:
        merged.append(buf)
    return "\n".join(merged).strip()


def strip_page_number_noise(text: str) -> str:
    """Remove lines that are only small integers (common page markers)."""
    lines = []
    for line in text.split("\n"):
        if _PAGE_NUM_LINE.match(line.strip()) and len(line.strip()) <= 4:
            continue
        lines.append(line)
    return "\n".join(lines)


def drop_repeated_lines(text: str, min_repeats: int = 3) -> str:
    """Remove lines that repeat many times (headers/footers)."""
    from collections import Counter

    raw_lines = text.split("\n")
    counts = Counter(l.strip() for l in raw_lines if l.strip())
    frequent = {k for k, v in counts.items() if v >= min_repeats and len(k) < 200}
    kept = [ln for ln in raw_lines if not (ln.strip() in frequent and len(ln.strip()) > 0)]
    return "\n".join(kept)


def ocr_noise_ratio(text: str) -> float:
    """Fraction of replacement / control characters typical of bad OCR."""
    if not text:
        return 0.0
    replacement = text.count("\ufffd")
    ctrl = sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r")
    return (replacement + ctrl) / max(1, len(text))


def strip_cross_page_repeated_lines(pages: list) -> list:
    """
    Remove short lines that repeat across many pages (running headers, footers, slide titles).
    Only runs when there are enough pages to estimate repetition safely.
    """
    from collections import Counter

    from raw_to_embedding.models import PdfPage

    if len(pages) < 5:
        return pages
    line_counts: Counter[str] = Counter()
    for p in pages:
        for line in (p.text or "").split("\n"):
            s = line.strip()
            if 6 <= len(s) <= 220:
                line_counts[s] += 1
    threshold = max(3, int(len(pages) * 0.35))
    frequent = {k for k, v in line_counts.items() if v >= threshold}
    out: list = []
    for p in pages:
        lines = []
        for line in (p.text or "").split("\n"):
            if line.strip() in frequent:
                continue
            lines.append(line)
        out.append(PdfPage(page_number=p.page_number, text="\n".join(lines)))
    return out


def clean_text(text: str) -> str:
    """Full cleaning pipeline for extracted raw text."""
    text = normalize_whitespace(text)
    text = merge_broken_lines(text)
    text = strip_page_number_noise(text)
    text = drop_repeated_lines(text)
    text = normalize_whitespace(text)
    return text


def clean_iterable(parts: Iterable[str]) -> list[str]:
    """Clean multiple segments."""
    return [clean_text(p) for p in parts if p and clean_text(p)]


def clean_raw_document(raw: "RawDocument") -> "RawDocument":
    """Apply cleaning to extracted document (pages/sections and raw_text)."""
    from raw_to_embedding.models import PdfPage, RawDocument, WebsiteSection

    if raw.pages:
        per_page = [PdfPage(page_number=p.page_number, text=clean_text(p.text)) for p in raw.pages]
        per_page = strip_cross_page_repeated_lines(per_page)
        new_pages = per_page
        new_text = "\n\n".join(p.text for p in new_pages)
        return RawDocument(
            source=raw.source,
            source_type=raw.source_type,
            raw_text=new_text,
            url=raw.url,
            path=raw.path,
            pages=new_pages,
            sections=raw.sections,
        )
    if raw.sections:
        new_sections = [
            WebsiteSection(
                heading_level=s.heading_level,
                heading_text=s.heading_text,
                content=clean_text(s.content),
                url=s.url,
            )
            for s in raw.sections
        ]
        parts: list[str] = []
        for s in new_sections:
            if s.heading_text:
                parts.append(f"{s.heading_text}\n{s.content}")
            else:
                parts.append(s.content)
        new_text = "\n\n".join(parts)
        return RawDocument(
            source=raw.source,
            source_type=raw.source_type,
            raw_text=new_text,
            url=raw.url,
            path=raw.path,
            pages=raw.pages,
            sections=new_sections,
        )
    return RawDocument(
        source=raw.source,
        source_type=raw.source_type,
        raw_text=clean_text(raw.raw_text),
        url=raw.url,
        path=raw.path,
        pages=raw.pages,
        sections=raw.sections,
    )
