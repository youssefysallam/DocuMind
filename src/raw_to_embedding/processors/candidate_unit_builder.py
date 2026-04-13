"""Build candidate units before LLM segmentation (rule-based structure)."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Callable

from raw_to_embedding.models import CandidateUnit, DocumentType, PdfPage, RawDocument, WebsiteSection

# Institute / report-style headings (line- or block-anchored)
_INSTITUTE_HEADING = re.compile(
    r"(?im)^\s*(?:"
    r"\d+(?:\.\d+)*\s+.{3,80}"
    r"|(?:[A-Z][A-Za-z0-9 ,&'-]{2,60})\s*$"
    r"|(?:Our Mission|Our Team|Who We Are|Programs|Partners|Contact|Financials|Overview)\b"
    r")\s*$"
)

# Scholarly section anchors (case-insensitive line starts)
_SCHOLAR_SECTIONS = [
    (re.compile(r"(?im)^\s*abstract\s*$"), "Abstract"),
    (re.compile(r"(?im)^\s*1\.?\s*introduction\s*$"), "Introduction"),
    (re.compile(r"(?im)^\s*(?:2\.?\s*)?methods?\b"), "Methods"),
    (re.compile(r"(?im)^\s*(?:3\.?\s*)?results?\b"), "Results"),
    (re.compile(r"(?im)^\s*(?:4\.?\s*)?discussion\b"), "Discussion"),
    (re.compile(r"(?im)^\s*conclusion[s]?\b"), "Conclusion"),
    (re.compile(r"(?im)^\s*references?\b"), "References"),
    (re.compile(r"(?im)^\s*acknowledgements?\b"), "Acknowledgements"),
]


def _short_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _pdf_concat(pages: list[PdfPage]) -> tuple[str, list[tuple[int, int, int]]]:
    """Full text and (page_number, start, end) char ranges in full text."""
    full_chunks: list[str] = []
    ranges: list[tuple[int, int, int]] = []
    pos = 0
    for p in pages:
        sep = "\n\n" if full_chunks else ""
        if sep:
            pos += len(sep)
            full_chunks.append(sep)
        start = pos
        body = p.text or ""
        full_chunks.append(body)
        end = start + len(body)
        ranges.append((p.page_number, start, end))
        pos = end
    return "".join(full_chunks), ranges


def _span_pages(
    ranges: list[tuple[int, int, int]], start: int, end: int
) -> tuple[int | None, int | None]:
    overlapping = [pg for pg, s, e in ranges if not (e < start or s > end)]
    if not overlapping:
        return None, None
    return min(overlapping), max(overlapping)


def _split_by_regex_blocks(text: str, pattern: re.Pattern[str]) -> list[tuple[str | None, str]]:
    """Split text into (heading_or_none, body) blocks."""
    matches = list(pattern.finditer(text))
    if not matches:
        return [(None, text.strip())]
    blocks: list[tuple[str | None, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        line0 = block.split("\n", 1)[0].strip()
        rest = block[len(line0) :].lstrip() if "\n" in block else ""
        title = line0[:200]
        body = rest if rest else block
        blocks.append((title, body))
    # preamble before first match
    pre = text[: matches[0].start()].strip()
    if pre:
        blocks.insert(0, (None, pre))
    return [b for b in blocks if b[1]]


def _split_institute(text: str) -> list[tuple[str | None, str]]:
    lines = text.split("\n")
    sections: list[tuple[str | None, list[str]]] = []
    current_title: str | None = None
    buf: list[str] = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 120 and (
            _INSTITUTE_HEADING.match(stripped)
            or re.match(r"(?i)^(our mission|programs|contact|overview)\b", stripped)
        ):
            if buf:
                sections.append((current_title, buf))
            current_title = stripped
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((current_title, buf))
    out: list[tuple[str | None, str]] = []
    for title, blines in sections:
        body = "\n".join(blines).strip()
        if body:
            out.append((title, body))
    return out if out else [(None, text.strip())]


def _split_scholarly(text: str) -> list[tuple[str | None, str]]:
    """Split on academic section anchors; merge small preface into first block."""
    positions: list[tuple[int, str, re.Match[str]]] = []
    for rx, label in _SCHOLAR_SECTIONS:
        for m in rx.finditer(text):
            positions.append((m.start(), label, m))
    positions.sort(key=lambda x: x[0])
    if not positions:
        return [(None, text.strip())]
    blocks: list[tuple[str | None, str]] = []
    # preamble
    first = positions[0][0]
    if first > 0:
        pre = text[:first].strip()
        if pre:
            blocks.append((None, pre))
    for i, (_, label, m) in enumerate(positions):
        start = m.start()
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            line = chunk.split("\n", 1)[0].strip()
            body = chunk[len(line) :].lstrip() if "\n" in chunk else chunk
            blocks.append((label, body if body else chunk))
    return blocks if blocks else [(None, text.strip())]


def _make_unit(
    *,
    doc_id: str,
    idx: int,
    document_type: DocumentType,
    title: str,
    section: str | None,
    content: str,
    source: str,
    source_type: str,
    url: str | None,
    page_start: int | None,
    page_end: int | None,
    extra: dict | None = None,
) -> CandidateUnit:
    uid = f"{doc_id}_u{idx}_{_short_id(content[:500])}"
    meta = {"doc_id": doc_id, **(extra or {})}
    return CandidateUnit(
        unit_id=uid,
        document_type=document_type,
        title=title or "Untitled",
        section=section,
        content=content,
        source_type=source_type,  # type: ignore[arg-type]
        source=source,
        url=url,
        page_start=page_start,
        page_end=page_end,
        metadata=meta,
    )


def build_candidate_units(
    raw: RawDocument,
    document_type: DocumentType,
    doc_id: str | None = None,
) -> list[CandidateUnit]:
    """
    Construct candidate units from cleaned RawDocument.

    Website: one unit per heading section (or merged if structure missing).
    PDFs: heuristic sections with page ranges where possible.
    """
    doc_id = doc_id or uuid.uuid4().hex[:12]
    units: list[CandidateUnit] = []

    if document_type == "website_page" and raw.sections:
        for i, sec in enumerate(raw.sections):
            title = sec.heading_text or raw.source
            body = (sec.content or "").strip()
            if not body:
                continue
            units.append(
                _make_unit(
                    doc_id=doc_id,
                    idx=i,
                    document_type=document_type,
                    title=title,
                    section=sec.heading_text,
                    content=body,
                    source=raw.source,
                    source_type="website",
                    url=raw.url,
                    page_start=None,
                    page_end=None,
                    extra={"heading_level": sec.heading_level},
                )
            )
        return units or _fallback_single(raw, document_type, doc_id)

    if raw.source_type == "pdf" and raw.pages:
        full_text, ranges = _pdf_concat(raw.pages)
        splitter: Callable[[str], list[tuple[str | None, str]]]
        if document_type == "scholarly_paper_pdf":
            splitter = _split_scholarly
        else:
            splitter = _split_institute

        blocks = splitter(full_text)
        for i, (title, body) in enumerate(blocks):
            if not body.strip():
                continue
            start = full_text.find(body[: min(80, len(body))])
            if start < 0:
                start = 0
            end = start + len(body)
            ps, pe = _span_pages(ranges, start, end)
            sec_title = title or f"Section {i+1}"
            units.append(
                _make_unit(
                    doc_id=doc_id,
                    idx=i,
                    document_type=document_type,
                    title=sec_title,
                    section=title,
                    content=body.strip(),
                    source=raw.source,
                    source_type="pdf",
                    url=None,
                    page_start=ps,
                    page_end=pe,
                    extra={},
                )
            )
        return units or _fallback_single(raw, document_type, doc_id)

    # Fallback: single blob
    return _fallback_single(raw, document_type, doc_id)


def _fallback_single(
    raw: RawDocument, document_type: DocumentType, doc_id: str
) -> list[CandidateUnit]:
    text = raw.raw_text.strip()
    ps = pe = None
    if raw.pages:
        ps, pe = raw.pages[0].page_number, raw.pages[-1].page_number
    return [
        _make_unit(
            doc_id=doc_id,
            idx=0,
            document_type=document_type,
            title=raw.source,
            section=None,
            content=text,
            source=raw.source,
            source_type=raw.source_type,
            url=raw.url,
            page_start=ps,
            page_end=pe,
            extra={"fallback": True},
        )
    ]
