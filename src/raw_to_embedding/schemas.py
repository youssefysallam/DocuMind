"""Pydantic schemas for LLM JSON responses (semantic segmentation)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from raw_to_embedding.models import ContentType, DocumentType, SourceType


class LLMSegmentMetadata(BaseModel):
    """Metadata echoed per segment for traceability (must align with source)."""

    unit_id: str
    source_type: SourceType
    document_type: DocumentType
    source: str
    url: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    original_title: str | None = None
    section: str | None = None


class LLMSegmentItem(BaseModel):
    """One semantic segment from the LLM."""

    segment_index: int = Field(..., ge=0)
    title: str
    section: str | None = None
    content_type: ContentType
    content: str
    preserve_as_single_chunk: bool = True
    metadata: LLMSegmentMetadata


class LLMSegmentationResponse(BaseModel):
    """Full JSON object returned by the semantic segmentation LLM."""

    unit_id: str
    is_multi_topic: bool
    reasoning_brief: str
    segments: list[LLMSegmentItem] = Field(..., min_length=1)

    @field_validator("segments")
    @classmethod
    def segment_indices_unique_and_sorted(
        cls, v: list[LLMSegmentItem]
    ) -> list[LLMSegmentItem]:
        indices = [s.segment_index for s in v]
        if len(indices) != len(set(indices)):
            raise ValueError("segment_index values must be unique")
        return v


