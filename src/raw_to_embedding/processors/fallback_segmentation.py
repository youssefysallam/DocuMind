"""Deterministic segmentation when LLM output is unusable."""

from __future__ import annotations

from typing import Any

from raw_to_embedding.models import CandidateUnit, ContentType, SemanticSegment
from raw_to_embedding.utils.sentence_utils import group_sentences, split_sentences


def _trace_meta(unit: CandidateUnit) -> dict[str, Any]:
    return {
        "source_type": unit.source_type,
        "document_type": unit.document_type,
        "source": unit.source,
        "url": unit.url,
        "page_start": unit.page_start,
        "page_end": unit.page_end,
        "title": unit.title,
        "section": unit.section,
        **unit.metadata,
    }


def fallback_segments(
    unit: CandidateUnit,
    *,
    max_chars: int,
    max_sentences: int,
) -> list[SemanticSegment]:
    """
    No-hallucination fallback: preserve original text, optional sentence groups.

    Produces one or more segments with content_type 'other' unless inferred trivially.
    """
    text = unit.content.strip()
    if not text:
        return [
            SemanticSegment(
                unit_id=unit.unit_id,
                segment_index=0,
                title=unit.title,
                section=unit.section,
                content_type="other",
                content="",
                preserve_as_single_chunk=True,
                metadata={**_trace_meta(unit), "fallback": True},
            )
        ]

    sentences = split_sentences(text)
    groups = list(group_sentences(sentences, max_sentences, max_chars))
    if len(groups) == 1:
        return [
            SemanticSegment(
                unit_id=unit.unit_id,
                segment_index=0,
                title=unit.title,
                section=unit.section,
                content_type="other",
                content=text,
                preserve_as_single_chunk=True,
                metadata={**_trace_meta(unit), "fallback": True},
            )
        ]

    segments: list[SemanticSegment] = []
    for i, grp in enumerate(groups):
        chunk_text = " ".join(grp).strip()
        segments.append(
            SemanticSegment(
                unit_id=unit.unit_id,
                segment_index=i,
                title=f"{unit.title} (part {i+1})",
                section=unit.section,
                content_type="other",
                content=chunk_text,
                preserve_as_single_chunk=False,
                metadata={**_trace_meta(unit), "fallback": True, "part": i + 1},
            )
        )
    return segments


def candidate_as_single_segment(unit: CandidateUnit, content_type: ContentType = "other") -> SemanticSegment:
    """Rule-only path: whole unit becomes one semantic segment."""
    return SemanticSegment(
        unit_id=unit.unit_id,
        segment_index=0,
        title=unit.title,
        section=unit.section,
        content_type=content_type,
        content=unit.content.strip(),
        preserve_as_single_chunk=True,
        metadata=_trace_meta(unit),
    )
