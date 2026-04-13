"""
Pilot: re-chunk a single PDF with improved semantic chunking strategy.

Produces pilot artifacts under results/raw_to_embedding/pilot_runs/ (configurable).

Usage:
    python -m raw_to_embedding.corpus_build.pilot_single_pdf \
        --pdf "c:/rawdata2embedding/SSL+PDF/Community-Led....pdf"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import raw_to_embedding.config as _config  # noqa: F401
from raw_to_embedding.extractors.pdf_extractor import extract_pdf
from raw_to_embedding.utils.text_cleaning import normalize_whitespace, strip_page_number_noise

PROJECT_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)

# ── Chunking parameters ──────────────────────────────────────────────

MAX_CHUNK_CHARS = 700
MAX_SENTENCES = 5
OVERLAP_SENTENCES = 2
MIN_CHUNK_CHARS = 120


# ═══════════════════════════════════════════════════════════════════════
#  Page-level cleaning
# ═══════════════════════════════════════════════════════════════════════

_JOURNAL_HEADER = re.compile(
    r"^Climate\s+\d{4},\s*\d+,\s*\d+(?:\s+\d+\s+of\s+\d+|\.\s*https?://doi\.org\S*)\s*$",
    re.MULTILINE,
)

# Report-style repeating page headers: "2025 Impact Report", "Preliminary Analysis for Boston Harbor | 5"
_REPORT_PAGE_HEADER = re.compile(
    r"^(?:\d{4}\s+Impact\s+Report|"
    r"Preliminary\s+Analysis\s+for\s+Boston\s+Harbor\s*\|?\s*\d*|"
    r"\d+\s*\|\s*Feasibility\s+of\s+Harbor-wide\s+Barrier\s+Systems|"
    r"Feasibility\s+of\s+Harbor-wide\s+Barrier\s+Systems)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_SCHOLARWORKS = re.compile(
    r"ScholarWorks at UMass Boston|"
    r"University of Massachusetts Boston|"
    r"This [Aa]rticle is (?:brought to you|available).*?(?:scholarworks@umb\.edu|ScholarWorks).*?\.|"
    r"Recommended Citation.*?\.|"
    r"For more information.*?scholarworks@umb\.edu.*?\.|"
    r"an authorized administrator of ScholarWorks.*?\.",
    re.DOTALL,
)

_DOI_LINE = re.compile(r"^https?://(?:doi\.org|www\.mdpi\.com)/.*$", re.MULTILINE)

_MDPI_DISCLAIMER = re.compile(
    r"Disclaimer/Publisher.*?referred to in the content\.",
    re.DOTALL,
)


def _clean_page(text: str) -> str:
    text = normalize_whitespace(text)
    text = _JOURNAL_HEADER.sub("", text)
    text = _REPORT_PAGE_HEADER.sub("", text)
    text = _SCHOLARWORKS.sub("", text)
    text = _DOI_LINE.sub("", text)
    text = _MDPI_DISCLAIMER.sub("", text)
    text = strip_page_number_noise(text)
    text = re.sub(r"-\n([a-z])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_cross_page_repeats(pages: list[dict]) -> list[dict]:
    """Remove short lines that repeat on ≥35% of pages."""
    if len(pages) < 5:
        return pages
    counts: Counter[str] = Counter()
    for p in pages:
        for ln in p["text"].split("\n"):
            s = ln.strip()
            if 6 <= len(s) <= 200:
                counts[s] += 1
    threshold = max(3, int(len(pages) * 0.35))
    frequent = {k for k, v in counts.items() if v >= threshold}
    if not frequent:
        return pages
    out = []
    for p in pages:
        kept = [ln for ln in p["text"].split("\n") if ln.strip() not in frequent]
        out.append({**p, "text": "\n".join(kept)})
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Structural analysis: detect sections, tables, figures
# ═══════════════════════════════════════════════════════════════════════

_TABLE_LABEL_RE = re.compile(
    r"^\s*Table\s+(\d+)\.\s*(.*?)$", re.MULTILINE | re.IGNORECASE,
)
_FIGURE_LABEL_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?)\s+(\d+)\.\s*(.*?)$", re.MULTILINE | re.IGNORECASE,
)
_REF_SIGNALS = re.compile(
    r"\[CrossRef\]|\[PubMed\]|doi\.org|"
    r"Available\s+online:?\s+http|ISBN\s+\d|"
    r"Accessed\s+on\s+\d|^\s*\d+\.\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _is_ref_only_page(text: str) -> bool:
    signals = len(_REF_SIGNALS.findall(text))
    if signals < 6:
        return False
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    ref_lines = sum(1 for l in lines if _REF_SIGNALS.search(l) or re.match(r"^\d+\.\s*$", l))
    return ref_lines / len(lines) > 0.5


def _is_heading_line(line: str) -> tuple[str, str] | None:
    """
    Detect a section heading on a single raw line.
    Returns (number, label) or None.

    Matches:
        Numbered: '7. Conclusions', '4.1.2. Black Americans'
        Unnumbered: 'About', 'Partnerships Highlights', 'New Initiatives'
    Does NOT match:
        Affiliations, reference numbers, table/figure labels, body sentences
    """
    s = line.strip()
    if not s or len(s) > 100 or len(s) < 3:
        return None

    # 1. Numbered headings: "7. Conclusions", "2.1. Background"
    m = re.match(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z,\s:&/'–\-]{3,80})$", s)
    if m:
        num, label = m.group(1), m.group(2).strip()
        # Reject years used as "numbers" (e.g. "2020. This report...")
        if re.match(r"^(19|20)\d{2}$", num):
            return None
        if re.match(r"Department|School|University|Institute|Faculty", label):
            return None
        if len(label) < 5:
            return None
        if label.lower().startswith(("table", "figure", "fig.")):
            return None
        return (num, label)

    # 2. Unnumbered descriptive headings — only high-confidence patterns
    # Must be ALL-CAPS (≥3 chars), not a date/time label
    if s.isupper() and len(s) >= 3 and len(s) <= 60 and " " in s:
        words = s.split()
        if 2 <= len(words) <= 6:
            # Reject month+year patterns like "DEC 2021", "APRIL 2022"
            if re.match(r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s+\d{4}$", s):
                return None
            return ("", s.title())

    # Known structural heading keywords (case-insensitive)
    low = s.lower().strip()
    _KNOWN_HEADINGS = {
        "table of contents", "about", "executive summary",
        "introduction", "background", "methodology", "methods",
        "results", "discussion", "conclusion", "conclusions",
        "recommendations", "acknowledgments", "acknowledgements",
        "appendix", "references", "bibliography", "glossary",
        "thank you", "who we are", "meet our team", "mission",
        "core", "partnerships highlights", "new initiatives",
        "publications", "research highlight", "upcoming research releases",
        "students", "recognitions", "events", "summary",
    }
    if low in _KNOWN_HEADINGS:
        return ("", s)

    return None


class Block:
    __slots__ = ("heading", "number", "text", "page_start", "page_end", "block_type")

    def __init__(self, heading: str, number: str, text: str,
                 page_start: int, page_end: int, block_type: str = "body"):
        self.heading = heading
        self.number = number
        self.text = text
        self.page_start = page_start
        self.page_end = page_end
        self.block_type = block_type


def structural_split(pages: list[dict]) -> list[Block]:
    """
    Two-pass structural splitting:

    Pass 1 (pre-merge): Scan raw lines for standalone section headings, table labels,
    figure labels. These become section boundary markers with clean labels.

    Pass 2: Merge body lines within each section, then assemble blocks.
    """
    # Exclude pages that are purely references
    content_pages = [p for p in pages if not _is_ref_only_page(p["text"])]
    if not content_pages:
        content_pages = pages[:1]

    # Detect reference boundary: find where bibliography entries begin.
    # A reference entry looks like a standalone number line "1." followed
    # by an author line "AuthorLastname, X.X.; ..."
    _REF_ENTRY_START = re.compile(r"^\s*(\d+)\.\s*$")
    _REF_AUTHOR_LINE = re.compile(r"^\s*[A-Z][a-z]+,\s+[A-Z]")

    ref_boundary_page = None
    ref_boundary_line_idx = None
    for p in content_pages:
        lines = p["text"].split("\n")
        for li in range(len(lines) - 1):
            s = lines[li].strip()
            # Pattern: "1." on its own line, followed by author name
            m = _REF_ENTRY_START.match(s)
            if m and _REF_AUTHOR_LINE.match(lines[li + 1].strip()):
                # Confirm it's not a false positive: check for ref signals nearby
                context = "\n".join(lines[li:li + 5])
                if _REF_SIGNALS.search(context) or int(m.group(1)) <= 3:
                    ref_boundary_page = p["page_number"]
                    ref_boundary_line_idx = li
                    break
            # Also detect inline ref entries: "1. AuthorLastname, X.; ..."
            if re.match(r"^\d+\.\s+[A-Z][a-z]+,\s+[A-Z]", s) and _REF_SIGNALS.search(s):
                ref_boundary_page = p["page_number"]
                ref_boundary_line_idx = li
                break
        if ref_boundary_page is not None:
            break

    # Build list of (page_number, line_text) tuples for all content, stopping at references
    all_lines: list[tuple[int, str]] = []
    for p in content_pages:
        pnum = p["page_number"]
        lines = p["text"].split("\n")
        for li, ln in enumerate(lines):
            if pnum == ref_boundary_page and li >= ref_boundary_line_idx:
                break
            all_lines.append((pnum, ln))
        if pnum == ref_boundary_page:
            break

    # Pass 1: identify structural markers
    # Each marker: (line_index, type, number, label)
    markers: list[tuple[int, str, str, str]] = []

    for li, (pnum, ln) in enumerate(all_lines):
        s = ln.strip()
        # Table labels
        tm = _TABLE_LABEL_RE.match(s)
        if tm:
            caption = tm.group(2).strip()[:80] or "Untitled"
            markers.append((li, "table", tm.group(1), f"Table {tm.group(1)}. {caption}"))
            continue
        # Figure labels
        fm = _FIGURE_LABEL_RE.match(s)
        if fm:
            caption = fm.group(2).strip()[:80] or "Untitled"
            markers.append((li, "figure", fm.group(1), f"Figure {fm.group(1)}. {caption}"))
            continue
        # Section headings (numbered or descriptive)
        hd = _is_heading_line(s)
        if hd:
            num, label = hd
            heading_label = f"{num}. {label}" if num else label
            markers.append((li, "section", num, heading_label))

    # Pass 2: build blocks from marker boundaries
    blocks: list[Block] = []

    # Preamble: lines before first marker
    first_marker_li = markers[0][0] if markers else len(all_lines)
    if first_marker_li > 0:
        preamble_lines = [ln for _, ln in all_lines[:first_marker_li]]
        preamble_text = "\n".join(preamble_lines).strip()
        if preamble_text and len(preamble_text) > 30:
            p_start = all_lines[0][0]
            p_end = all_lines[min(first_marker_li - 1, len(all_lines) - 1)][0]
            blocks.append(Block(
                heading="Preamble", number="0", text=preamble_text,
                page_start=p_start, page_end=p_end, block_type="preamble",
            ))

    for mi, (li, mtype, num, label) in enumerate(markers):
        end_li = markers[mi + 1][0] if mi + 1 < len(markers) else len(all_lines)
        # Heading line is at index `li`, body starts at `li + 1`
        body_lines = [ln for _, ln in all_lines[li + 1:end_li]]
        body_text = "\n".join(body_lines).strip()
        if not body_text or len(body_text) < 15:
            continue
        pg_start = all_lines[li][0]
        pg_end = all_lines[min(end_li - 1, len(all_lines) - 1)][0]
        blocks.append(Block(
            heading=label, number=num, text=body_text,
            page_start=pg_start, page_end=pg_end, block_type=mtype,
        ))

    # Fallback: if no structural markers found, split by page boundaries
    if not blocks:
        current_page = None
        page_lines: list[str] = []
        for pnum, ln in all_lines:
            if current_page is not None and pnum != current_page and page_lines:
                text = "\n".join(page_lines).strip()
                if text and len(text) > 30:
                    blocks.append(Block(
                        heading=f"Page {current_page}",
                        number="",
                        text=text,
                        page_start=current_page,
                        page_end=current_page,
                        block_type="body",
                    ))
                page_lines = []
            current_page = pnum
            page_lines.append(ln)
        if page_lines and current_page is not None:
            text = "\n".join(page_lines).strip()
            if text and len(text) > 30:
                blocks.append(Block(
                    heading=f"Page {current_page}",
                    number="",
                    text=text,
                    page_start=current_page,
                    page_end=current_page,
                    block_type="body",
                ))

    if not blocks:
        all_text = "\n".join(ln for _, ln in all_lines)
        blocks.append(Block(
            heading="Document", number="0", text=all_text,
            page_start=all_lines[0][0] if all_lines else 1,
            page_end=all_lines[-1][0] if all_lines else 1,
            block_type="body",
        ))

    return blocks


# ═══════════════════════════════════════════════════════════════════════
#  Sentence splitting & packing
# ═══════════════════════════════════════════════════════════════════════

_SENT_END = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'\(\["\u201C])'
    r"|(?<=[.!?])\s*$",
    re.MULTILINE,
)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        seg = text[start:m.start()].strip()
        if seg:
            parts.append(seg)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts or [text]


def pack_with_overlap(
    sentences: list[str],
    max_group: int = MAX_SENTENCES,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_SENTENCES,
) -> list[list[str]]:
    if not sentences:
        return []
    groups: list[list[str]] = []
    i = 0
    n = len(sentences)
    safe_ov = min(overlap, max_group - 1)
    while i < n:
        grp: list[str] = []
        clen = 0
        j = i
        while j < n and len(grp) < max_group:
            s = sentences[j]
            added = len(s) + (1 if grp else 0)
            if grp and clen + added > max_chars:
                break
            grp.append(s)
            clen += added
            j += 1
        if not grp:
            grp = [sentences[i]]
            j = i + 1
        groups.append(grp)
        if j >= n:
            break
        nxt = max(i + 1, j - safe_ov)
        i = nxt
    return groups


# ═══════════════════════════════════════════════════════════════════════
#  Chunk metadata helpers
# ═══════════════════════════════════════════════════════════════════════

def _chunk_id(pdf_stem: str, idx: int) -> str:
    raw = f"pilot|{pdf_stem}|{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def quality_flag(text: str) -> str:
    t = text or ""
    if len(t.strip()) < 40:
        return "partial"
    bad = sum(1 for c in t if ord(c) > 0xFFFF or (ord(c) < 32 and c not in "\n\t\r"))
    if bad > len(t) * 0.02 or t.count("\ufffd") > 0:
        return "noisy"
    return "clean"


def chunk_type_heuristic(text: str, block_type: str) -> str:
    if block_type == "table":
        return "table"
    if block_type == "figure":
        return "figure_caption"
    low = (text or "").lower()
    if re.search(r"^\s*\|.*\|", text, re.MULTILINE):
        return "table"
    if re.search(r"^\s*[-•*]\s", text, re.MULTILINE) and text.count("\n") >= 2:
        return "list"
    if "figure" in low[:100] or re.search(r"\bfig\.?\s+\d", low):
        return "figure_caption"
    return "paragraph"


# ═══════════════════════════════════════════════════════════════════════
#  Preamble condensation
# ═══════════════════════════════════════════════════════════════════════

def _condense_preamble(text: str) -> str:
    """
    Preamble pages contain ScholarWorks boilerplate, author lists, affiliations,
    copyright notices. Extract only the meaningful parts: title, abstract (if any),
    and authors (compacted).
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Remove common boilerplate patterns
    kept: list[str] = []
    skip_patterns = [
        re.compile(r"^(Authors?|Citation|Received|Revised|Accepted|Published):?\s*$", re.IGNORECASE),
        re.compile(r"^Copyright:?\s", re.IGNORECASE),
        re.compile(r"^Licensee\s+MDPI", re.IGNORECASE),
        re.compile(r"^Academic\s+Editor", re.IGNORECASE),
        re.compile(r"^\*\s*Correspondence", re.IGNORECASE),
        re.compile(r"^https?://", re.IGNORECASE),
        re.compile(r"^open access article", re.IGNORECASE),
        re.compile(r"^This article is an? open access", re.IGNORECASE),
        re.compile(r"^Sustainable Solutions Lab$", re.IGNORECASE),
        re.compile(r"^Research Centers and Institutes$", re.IGNORECASE),
    ]
    for ln in lines:
        if any(p.search(ln) for p in skip_patterns):
            continue
        if len(ln) < 5:
            continue
        kept.append(ln)

    # Deduplicate consecutive identical lines
    deduped: list[str] = []
    for ln in kept:
        if not deduped or ln != deduped[-1]:
            deduped.append(ln)

    return " ".join(deduped)


# ═══════════════════════════════════════════════════════════════════════
#  Table block handling
# ═══════════════════════════════════════════════════════════════════════

def _format_table_block(text: str) -> str:
    """Keep table data as a single chunk (tables shouldn't be sentence-split)."""
    lines = text.split("\n")
    cleaned = [l.strip() for l in lines if l.strip()]
    return "\n".join(cleaned)


# ═══════════════════════════════════════════════════════════════════════
#  Main: run pilot
# ═══════════════════════════════════════════════════════════════════════

def run_pilot(pdf_path: Path) -> dict[str, Any]:
    pdf_path = pdf_path.resolve()
    stem = pdf_path.stem[:60]
    raw = extract_pdf(pdf_path)

    # ── Step 1: page-level cleaning ──
    pages = []
    for p in (raw.pages or []):
        pages.append({"page_number": p.page_number, "text": _clean_page(p.text or "")})
    pages = _strip_cross_page_repeats(pages)

    logger.info("PDF pages: %d (after cleaning)", len(pages))

    # ── Step 2: structural split ──
    blocks = structural_split(pages)
    logger.info("Structural blocks: %d", len(blocks))
    for b in blocks:
        logger.info("  [%s] %s (pages %d-%d, %d chars)",
                     b.block_type, b.heading[:50], b.page_start, b.page_end, len(b.text))

    # ── Step 3: chunk each block ──
    all_chunks: list[dict[str, Any]] = []
    idx = 0

    for blk in blocks:
        if blk.block_type == "preamble":
            content = _condense_preamble(blk.text)
            if len(content) < 30:
                continue
            # Preamble as one or two chunks max
            sents = split_sentences(content)
            groups = pack_with_overlap(sents, max_group=MAX_SENTENCES,
                                       max_chars=MAX_CHUNK_CHARS, overlap=0)
            for grp in groups:
                chunk_text = " ".join(grp).strip()
                if not chunk_text:
                    continue
                all_chunks.append({
                    "chunk_id": _chunk_id(stem, idx),
                    "source_pdf": pdf_path.name,
                    "section_title": "Preamble",
                    "section_path": "Preamble",
                    "page_start": blk.page_start,
                    "page_end": blk.page_end,
                    "chunk_text": chunk_text,
                    "embedding_text": chunk_text,
                    "chunk_type": "metadata",
                    "quality_flag": quality_flag(chunk_text),
                    "chunk_index": idx,
                })
                idx += 1
            continue

        if blk.block_type == "table":
            table_text = _format_table_block(blk.text)
            # Large tables: split into sub-chunks by rows, but keep header
            if len(table_text) > MAX_CHUNK_CHARS:
                rows = table_text.split("\n")
                sub_chunks: list[str] = []
                buf: list[str] = []
                buf_len = 0
                for row in rows:
                    if buf and buf_len + len(row) + 1 > MAX_CHUNK_CHARS:
                        sub_chunks.append("\n".join(buf))
                        buf = []
                        buf_len = 0
                    buf.append(row)
                    buf_len += len(row) + 1
                if buf:
                    sub_chunks.append("\n".join(buf))
                for sc in sub_chunks:
                    if len(sc.strip()) < 20:
                        continue
                    all_chunks.append({
                        "chunk_id": _chunk_id(stem, idx),
                        "source_pdf": pdf_path.name,
                        "section_title": blk.heading,
                        "section_path": blk.heading,
                        "page_start": blk.page_start,
                        "page_end": blk.page_end,
                        "chunk_text": sc,
                        "embedding_text": f"{blk.heading}\n{sc}",
                        "chunk_type": "table",
                        "quality_flag": quality_flag(sc),
                        "chunk_index": idx,
                    })
                    idx += 1
            else:
                all_chunks.append({
                    "chunk_id": _chunk_id(stem, idx),
                    "source_pdf": pdf_path.name,
                    "section_title": blk.heading,
                    "section_path": blk.heading,
                    "page_start": blk.page_start,
                    "page_end": blk.page_end,
                    "chunk_text": table_text,
                    "embedding_text": f"{blk.heading}\n{table_text}",
                    "chunk_type": "table",
                    "quality_flag": quality_flag(table_text),
                    "chunk_index": idx,
                })
                idx += 1
            continue

        if blk.block_type == "figure":
            fig_text = blk.text.strip()
            if len(fig_text) > MAX_CHUNK_CHARS:
                sentences = split_sentences(fig_text)
                groups = pack_with_overlap(sentences, overlap=1)
                for gi, grp in enumerate(groups):
                    content = " ".join(grp).strip()
                    if not content:
                        continue
                    # Prefix first chunk with figure label for context
                    emb_text = content
                    if gi == 0:
                        emb_text = f"{blk.heading}\n{content}"
                    all_chunks.append({
                        "chunk_id": _chunk_id(stem, idx),
                        "source_pdf": pdf_path.name,
                        "section_title": blk.heading,
                        "section_path": blk.heading,
                        "page_start": blk.page_start,
                        "page_end": blk.page_end,
                        "chunk_text": content,
                        "embedding_text": emb_text,
                        "chunk_type": chunk_type_heuristic(content, "body"),
                        "quality_flag": quality_flag(content),
                        "chunk_index": idx,
                    })
                    idx += 1
            else:
                all_chunks.append({
                    "chunk_id": _chunk_id(stem, idx),
                    "source_pdf": pdf_path.name,
                    "section_title": blk.heading,
                    "section_path": blk.heading,
                    "page_start": blk.page_start,
                    "page_end": blk.page_end,
                    "chunk_text": fig_text,
                    "embedding_text": f"{blk.heading}\n{fig_text}",
                    "chunk_type": "figure_caption",
                    "quality_flag": quality_flag(fig_text),
                    "chunk_index": idx,
                })
                idx += 1
            continue

        # Normal body block: sentence-level chunking with overlap
        sentences = split_sentences(blk.text)
        groups = pack_with_overlap(sentences)
        for grp in groups:
            content = " ".join(grp).strip()
            if not content:
                continue
            ctype = chunk_type_heuristic(content, blk.block_type)
            all_chunks.append({
                "chunk_id": _chunk_id(stem, idx),
                "source_pdf": pdf_path.name,
                "section_title": blk.heading,
                "section_path": blk.heading,
                "page_start": blk.page_start,
                "page_end": blk.page_end,
                "chunk_text": content,
                "embedding_text": content,
                "chunk_type": ctype,
                "quality_flag": quality_flag(content),
                "chunk_index": idx,
            })
            idx += 1

    # ── Step 3b: remove chunks that are clearly reference/bibliography entries ──
    def _looks_like_ref_chunk(text: str) -> bool:
        if len(text) > 600:
            return False
        ref_signals = len(re.findall(r"\[CrossRef\]|\[PubMed\]|doi\.org", text))
        has_numbered_ref = bool(re.search(r"^\s*\d+\.\s+[A-Z][a-z]+,", text, re.MULTILINE))
        has_journal_abbr = bool(re.search(r"\b[A-Z][a-z]+\.\s+\d{4},\s+\d+", text))
        return ref_signals >= 1 and (has_numbered_ref or has_journal_abbr)

    all_chunks = [
        c for c in all_chunks
        if not (c["chunk_type"] == "paragraph" and _looks_like_ref_chunk(c["chunk_text"]))
    ]

    # ── Step 4: merge tiny chunks ──
    # Pass A: merge within same section
    merged: list[dict[str, Any]] = []
    for c in all_chunks:
        if (
            merged
            and merged[-1]["section_title"] == c["section_title"]
            and len(c["chunk_text"]) < MIN_CHUNK_CHARS
        ):
            merged[-1]["chunk_text"] += " " + c["chunk_text"]
            merged[-1]["embedding_text"] = merged[-1]["chunk_text"]
            merged[-1]["page_end"] = max(merged[-1]["page_end"], c["page_end"])
        else:
            merged.append(c)

    # Pass B: merge any remaining chunks < 200 chars with their neighbor
    HARD_MIN = 200
    changed = True
    while changed:
        changed = False
        new_merged: list[dict[str, Any]] = []
        for c in merged:
            if len(c["chunk_text"]) < HARD_MIN and new_merged:
                prev = new_merged[-1]
                combined_len = len(prev["chunk_text"]) + len(c["chunk_text"]) + 1
                if combined_len <= MAX_CHUNK_CHARS + 200:
                    prev["chunk_text"] += " " + c["chunk_text"]
                    prev["embedding_text"] = prev["chunk_text"]
                    prev["page_end"] = max(prev["page_end"], c["page_end"])
                    changed = True
                    continue
            new_merged.append(c)
        merged = new_merged

    # Pass C: forward-merge remaining tiny first chunks
    if merged and len(merged[0]["chunk_text"]) < HARD_MIN and len(merged) > 1:
        merged[1]["chunk_text"] = merged[0]["chunk_text"] + " " + merged[1]["chunk_text"]
        merged[1]["embedding_text"] = merged[1]["chunk_text"]
        merged[1]["page_start"] = min(merged[0]["page_start"], merged[1]["page_start"])
        merged = merged[1:]

    all_chunks = merged
    for i, c in enumerate(all_chunks):
        c["chunk_index"] = i
        c["chunk_id"] = _chunk_id(stem, i)
        c["quality_flag"] = quality_flag(c["chunk_text"])

    logger.info("Chunks after merge: %d", len(all_chunks))

    # ── Step 5: embed ──
    texts = [c["embedding_text"] for c in all_chunks]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    mat = np.array(
        model.encode(texts, show_progress_bar=False, normalize_embeddings=True),
        dtype=np.float32,
    )
    for i, c in enumerate(all_chunks):
        c["embedding"] = mat[i].tolist()
        c["embedding_model"] = "all-MiniLM-L6-v2"

    # ── Step 6: FAISS index ──
    import faiss
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)

    # ── Step 7: save artifacts ──
    out_dir = PROJECT_ROOT / "results" / "raw_to_embedding" / "pilot_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]", "_", stem)[:50]

    chunks_path = out_dir / f"pilot_chunks_{safe}.json"
    index_path = out_dir / f"pilot_index_{safe}.faiss"
    meta_path = out_dir / f"pilot_index_metadata_{safe}.json"
    emb_path = out_dir / f"pilot_embeddings_{safe}.npy"
    report_path = out_dir / f"pilot_report_{safe}.json"

    meta_rows = [{k: v for k, v in c.items() if k != "embedding"} for c in all_chunks]
    chunks_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    faiss.write_index(index, str(index_path))
    np.save(str(emb_path), mat)

    # ── Step 8: build report ──
    lens = [len(c["chunk_text"]) for c in all_chunks]
    types = Counter(c["chunk_type"] for c in all_chunks)
    flags = Counter(c["quality_flag"] for c in all_chunks)
    sections = Counter(c["section_title"] for c in all_chunks)
    pages_covered = set()
    for c in all_chunks:
        for pg in range(c["page_start"], c["page_end"] + 1):
            pages_covered.add(pg)

    # Diverse examples: stratified by section
    import random as _rnd
    _rnd.seed(42)
    seen_titles: set[str] = set()
    examples: list[dict] = []
    for c in all_chunks:
        t = c["section_title"]
        if t not in seen_titles and len(examples) < 10:
            seen_titles.add(t)
            examples.append({
                "chunk_id": c["chunk_id"],
                "section_title": c["section_title"],
                "pages": f"{c['page_start']}-{c['page_end']}",
                "chunk_type": c["chunk_type"],
                "quality_flag": c["quality_flag"],
                "char_len": len(c["chunk_text"]),
                "text_preview": c["chunk_text"][:350],
                "embedding_text_preview": c["embedding_text"][:200],
            })

    report = {
        "pdf": pdf_path.name,
        "pilot_config": {
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "max_sentences": MAX_SENTENCES,
            "overlap_sentences": OVERLAP_SENTENCES,
            "min_chunk_chars": MIN_CHUNK_CHARS,
            "embedding_text_strategy": "raw content for body; heading+content for tables/figures",
            "embedding_model": "all-MiniLM-L6-v2",
            "normalize_embeddings": True,
        },
        "stats": {
            "structural_blocks": len(blocks),
            "total_chunks": len(all_chunks),
            "avg_chunk_chars": round(sum(lens) / max(1, len(lens)), 1),
            "min_chunk_chars": min(lens) if lens else 0,
            "max_chunk_chars": max(lens) if lens else 0,
            "median_chunk_chars": sorted(lens)[len(lens) // 2] if lens else 0,
            "chunk_type_counts": dict(types),
            "quality_flag_counts": dict(flags),
            "section_counts": dict(sections.most_common()),
            "pages_covered": sorted(pages_covered),
            "total_pdf_pages": len(raw.pages or []),
        },
        "example_chunks": examples,
        "artifacts": {
            "chunks_json": str(chunks_path),
            "index_faiss": str(index_path),
            "index_metadata": str(meta_path),
            "embeddings_npy": str(emb_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Pilot report written: %s", report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pilot: re-chunk one PDF with improved strategy.")
    ap.add_argument("--pdf", type=Path, required=True, help="Path to a single PDF.")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    report = run_pilot(Path(args.pdf))
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
