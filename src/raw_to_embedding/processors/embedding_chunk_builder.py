"""Build embedding-ready chunks from semantic segments with sentence packing."""

from __future__ import annotations

import hashlib
import re

from raw_to_embedding.config import get_settings
from raw_to_embedding.models import ContentType, EmbeddingChunk, SemanticSegment
from raw_to_embedding.utils.sentence_utils import pack_chunks_with_overlap, split_sentences

# Break before a list / enum line (paragraph break or newline + bullet/number).
_LIST_OR_ENUM_START = re.compile(
    r"(?:^|\n\s*\n|\n)(?=\s*(?:[-•*]|\d+\.)\s+\S)",
    re.MULTILINE,
)


def _chunk_id(unit_id: str, segment_index: int, part: int) -> str:
    raw = f"{unit_id}|{segment_index}|{part}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _embedding_text(
    title: str,
    section: str | None,
    content_type: ContentType,
    content: str,
) -> str:
    lines = [f"Title: {title}"]
    if section:
        lines.append(f"Section: {section}")
    lines.append(f"Type: {content_type}")
    lines.append(f"Content: {content}")
    return "\n".join(lines)


def _structural_pieces(text: str) -> list[str]:
    """
    Split segment body on list/enum boundaries so bullets and numbered items
    are less often merged with unrelated prose.
    """
    text = text.strip()
    if not text:
        return []
    raw_parts = _LIST_OR_ENUM_START.split(text)
    pieces = [p.strip() for p in raw_parts if p and p.strip()]
    if len(pieces) <= 1:
        return [text]
    return pieces


def _append_chunk(
    chunks: list[EmbeddingChunk],
    *,
    seg: SemanticSegment,
    part_idx: int,
    title: str,
    content: str,
) -> None:
    content = (content or "").strip()
    cid = _chunk_id(seg.unit_id, seg.segment_index, part_idx)
    meta = {
        **seg.metadata,
        "source_type": seg.metadata.get("source_type"),
        "document_type": seg.metadata.get("document_type"),
        "source": seg.metadata.get("source"),
        "url": seg.metadata.get("url"),
        "page_start": seg.metadata.get("page_start"),
        "page_end": seg.metadata.get("page_end"),
        "content_type": seg.content_type,
        "unit_id": seg.unit_id,
        "chunk_id": cid,
        "segment_part": part_idx + 1,
    }
    chunks.append(
        EmbeddingChunk(
            chunk_id=cid,
            unit_id=seg.unit_id,
            title=title,
            section=seg.section,
            content_type=seg.content_type,
            content=content,
            embedding_text=_embedding_text(title, seg.section, seg.content_type, content),
            metadata=meta,
        )
    )


def segments_to_chunks(segments: list[SemanticSegment]) -> list[EmbeddingChunk]:
    """Convert semantic segments to smaller, single-topic chunks with optional sentence overlap."""
    settings = get_settings()
    chunks: list[EmbeddingChunk] = []
    overlap = min(
        settings.chunk_overlap_sentences,
        max(0, settings.max_sentences_per_group - 1),
    )

    for seg in segments:
        text = seg.content.strip()
        if not text:
            _append_chunk(
                chunks,
                seg=seg,
                part_idx=0,
                title=seg.title,
                content="",
            )
            continue

        pieces = _structural_pieces(text)
        multi_piece = len(pieces) > 1
        part_idx = 0

        for piece in pieces:
            pt = piece.strip()
            if not pt:
                continue
            sentences = split_sentences(pt)
            allow_single = seg.preserve_as_single_chunk and not multi_piece
            if (
                allow_single
                and len(pt) <= settings.max_chunk_chars_soft
                and len(sentences) <= settings.max_sentences_per_group
            ):
                _append_chunk(chunks, seg=seg, part_idx=part_idx, title=seg.title, content=pt)
                part_idx += 1
                continue

            for grp in pack_chunks_with_overlap(
                sentences,
                settings.max_sentences_per_group,
                settings.max_chunk_chars_soft,
                overlap,
            ):
                joined = " ".join(grp).strip()
                if not joined:
                    continue
                part_title = f"{seg.title} (part {part_idx + 1})"
                cid = _chunk_id(seg.unit_id, seg.segment_index, part_idx)
                meta = {
                    **seg.metadata,
                    "source_type": seg.metadata.get("source_type"),
                    "document_type": seg.metadata.get("document_type"),
                    "source": seg.metadata.get("source"),
                    "url": seg.metadata.get("url"),
                    "page_start": seg.metadata.get("page_start"),
                    "page_end": seg.metadata.get("page_end"),
                    "content_type": seg.content_type,
                    "unit_id": seg.unit_id,
                    "chunk_id": cid,
                    "segment_part": part_idx + 1,
                }
                chunks.append(
                    EmbeddingChunk(
                        chunk_id=cid,
                        unit_id=seg.unit_id,
                        title=part_title,
                        section=seg.section,
                        content_type=seg.content_type,
                        content=joined,
                        embedding_text=_embedding_text(
                            part_title, seg.section, seg.content_type, joined
                        ),
                        metadata=meta,
                    )
                )
                part_idx += 1

    return chunks
