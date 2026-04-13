"""
Chunking and segmentation: candidate units → semantic segments → embedding chunks.

Core logic is split across ``processors/candidate_unit_builder``,
``processors/semantic_segmentation_agent``, and ``processors/embedding_chunk_builder``.
"""
from __future__ import annotations

from raw_to_embedding.processors.candidate_unit_builder import build_candidate_units
from raw_to_embedding.processors.embedding_chunk_builder import segments_to_chunks
from raw_to_embedding.processors.semantic_segmentation_agent import segment_units

__all__ = ["build_candidate_units", "segment_units", "segments_to_chunks"]
