"""Sentence splitting and grouping without arbitrary character cuts."""

from __future__ import annotations

import re
from typing import Iterator


# Simple sentence boundary heuristic (no NLTK dependency).
_SENTENCE_END = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|(?<=[.!?])\s*$", re.MULTILINE
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; conservative fallback keeps whole block if empty."""
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        end = m.start()
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    if not parts:
        return [text]
    return parts


def group_sentences(
    sentences: list[str], max_group: int, max_chars: int
) -> Iterator[list[str]]:
    """Group consecutive sentences respecting soft char and count limits."""
    if not sentences:
        return
    current: list[str] = []
    current_len = 0
    for s in sentences:
        sep = 1 if current else 0
        add_len = len(s) + sep
        over_count = len(current) >= max_group
        over_chars = current_len + add_len > max_chars and current
        if over_count or over_chars:
            if current:
                yield current
            current = [s]
            current_len = len(s)
        else:
            current.append(s)
            current_len += add_len
    if current:
        yield current


def pack_chunks_with_overlap(
    sentences: list[str],
    max_group: int,
    max_chars: int,
    overlap: int,
):
    """
    Pack sentences into chunks respecting max size, with 1–overlap sentences repeated
    at the start of the next chunk for continuity.
    """
    if not sentences:
        return
    safe_overlap = max(0, min(overlap, max(0, max_group - 1)))
    i = 0
    n = len(sentences)
    while i < n:
        group: list[str] = []
        char_len = 0
        j = i
        while j < n and len(group) < max_group:
            s = sentences[j]
            extra = len(s) + (1 if group else 0)
            if group and char_len + extra > max_chars:
                break
            group.append(s)
            char_len += extra
            j += 1
        if not group:
            group = [sentences[i]]
            j = i + 1
        yield group
        if j >= n:
            break
        next_i = j - safe_overlap
        if next_i <= i:
            next_i = j
        i = next_i
