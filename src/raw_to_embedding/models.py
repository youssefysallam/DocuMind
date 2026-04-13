"""Core domain models for ingestion and embedding-ready chunks."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DocumentType = Literal["website_page", "institute_report_pdf", "scholarly_paper_pdf"]
SourceType = Literal["pdf", "website"]

ContentType = Literal[
    "mission",
    "people",
    "initiative",
    "event",
    "partnership",
    "publication",
    "organization",
    "contact",
    "program",
    "resource",
    "academic_overview",
    "academic_method",
    "academic_dataset",
    "academic_result",
    "academic_conclusion",
    "other",
]


class PdfPage(BaseModel):
    """One PDF page of extracted text (1-based page index)."""

    page_number: int = Field(..., ge=1, description="1-based page number")
    text: str


class WebsiteSection(BaseModel):
    """Structured website content with optional heading hierarchy."""

    heading_level: int | None = Field(None, ge=1, le=6)
    heading_text: str | None = None
    content: str
    url: str


class RawDocument(BaseModel):
    """Normalized raw document after extraction (before cleaning)."""

    source: str
    source_type: SourceType
    raw_text: str
    url: str | None = None
    path: str | None = None
    pages: list[PdfPage] | None = None
    sections: list[WebsiteSection] | None = None


class CandidateUnit(BaseModel):
    """Pre-LLM candidate unit with traceability metadata."""

    unit_id: str
    document_type: DocumentType
    title: str
    section: str | None = None
    content: str
    source_type: SourceType
    source: str
    url: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticSegment(BaseModel):
    """Post-segmentation semantic unit (may map to one or more embedding chunks)."""

    unit_id: str
    segment_index: int
    title: str
    section: str | None = None
    content_type: ContentType
    content: str
    preserve_as_single_chunk: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingChunk(BaseModel):
    """Final embedding-ready record."""

    chunk_id: str
    unit_id: str
    title: str
    section: str | None = None
    content_type: ContentType
    content: str
    embedding_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
