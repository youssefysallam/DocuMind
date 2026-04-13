"""LLM-assisted semantic segmentation with validation, repair, and fallback."""

from __future__ import annotations

import logging
import re
from typing import Any

from openai import OpenAI

from raw_to_embedding.config import Settings, get_settings
from raw_to_embedding.models import CandidateUnit, ContentType, DocumentType, SemanticSegment
from raw_to_embedding.prompts import (
    SYSTEM_PROMPT,
    build_general_unit_prompt,
    build_paper_unit_prompt,
    build_repair_prompt,
)
from raw_to_embedding.processors.fallback_segmentation import candidate_as_single_segment, fallback_segments
from raw_to_embedding.validators.llm_output_validator import validate_with_source

logger = logging.getLogger(__name__)


def _base_metadata(unit: CandidateUnit) -> dict[str, Any]:
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


def _is_generic_title(title: str, settings: Settings) -> bool:
    t = title.strip().lower()
    if not t:
        return True
    for p in settings.generic_title_patterns:
        if p and p in t:
            return True
    return bool(re.match(r"^(section|part)\s*\d+$", t))


def _is_noisy(text: str) -> bool:
    if text.count("\n\n") > 25:
        return True
    if text.count("•") + text.count("- ") > 40:
        return True
    return False


def should_use_llm(unit: CandidateUnit, settings: Settings) -> bool:
    """Rule-first: invoke LLM only when heuristics say the unit is hard."""
    n = len(unit.content or "")
    if n > settings.max_unit_chars_for_skip_llm:
        return True
    if n >= settings.min_chars_multi_topic_heuristic:
        return True
    if _is_generic_title(unit.title, settings):
        return True
    if _is_noisy(unit.content):
        return True
    if unit.metadata.get("needs_llm"):
        return True
    return False


def _response_to_segments(
    unit: CandidateUnit, resp: LLMSegmentationResponse
) -> list[SemanticSegment]:
    out: list[SemanticSegment] = []
    base = _base_metadata(unit)
    for seg in sorted(resp.segments, key=lambda s: s.segment_index):
        meta = {
            **base,
            **seg.metadata.model_dump(),
            "original_title": seg.metadata.original_title,
        }
        out.append(
            SemanticSegment(
                unit_id=unit.unit_id,
                segment_index=seg.segment_index,
                title=seg.title,
                section=seg.section,
                content_type=seg.content_type,
                content=seg.content.strip(),
                preserve_as_single_chunk=seg.preserve_as_single_chunk,
                metadata=meta,
            )
        )
    return out


def _heuristic_content_type(document_type: DocumentType) -> ContentType:
    if document_type == "scholarly_paper_pdf":
        return "academic_overview"
    return "other"


def _call_llm(
    client: OpenAI,
    settings: Settings,
    messages: list[dict[str, str]],
) -> str:
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "messages": messages,
    }
    if settings.openai_base_url:
        # Client already has base_url if set via env in OpenAI()
        pass
    kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs, timeout=settings.llm_timeout_seconds)
    choice = resp.choices[0].message.content or ""
    return choice


def segment_unit(
    unit: CandidateUnit,
    *,
    settings: Settings | None = None,
    client: OpenAI | None = None,
) -> list[SemanticSegment]:
    """
    Produce semantic segments for one candidate unit.

    Uses LLM only when `should_use_llm` is true and API key is configured;
    otherwise deterministic single segment or fallback on failure.
    """
    settings = settings or get_settings()
    wants_llm = should_use_llm(unit, settings)

    if not wants_llm:
        return [
            candidate_as_single_segment(
                unit, content_type=_heuristic_content_type(unit.document_type)
            )
        ]

    if not settings.openai_api_key:
        logger.warning(
            "LLM recommended for unit %s but OPENAI_API_KEY missing; using fallback segmentation",
            unit.unit_id,
        )
        return fallback_segments(
            unit,
            max_chars=settings.max_chunk_chars_soft,
            max_sentences=settings.max_sentences_per_group,
        )

    oc = client or OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )

    if unit.document_type == "scholarly_paper_pdf":
        user_prompt = build_paper_unit_prompt(unit)
    else:
        user_prompt = build_general_unit_prompt(unit)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw = ""
    try:
        raw = _call_llm(oc, settings, messages)
        resp = validate_with_source(raw, unit)
        return _response_to_segments(unit, resp)
    except Exception as exc:
        logger.warning("LLM segmentation failed (%s), attempting repair", exc)
        try:
            repair = build_repair_prompt(raw, str(exc))
            messages2 = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair},
            ]
            raw2 = _call_llm(oc, settings, messages2)
            resp2 = validate_with_source(raw2, unit)
            return _response_to_segments(unit, resp2)
        except Exception as exc2:
            logger.warning("Repair failed (%s), using fallback segmentation", exc2)
            return fallback_segments(
                unit,
                max_chars=settings.max_chunk_chars_soft,
                max_sentences=settings.max_sentences_per_group,
            )


def segment_units(
    units: list[CandidateUnit],
    *,
    settings: Settings | None = None,
) -> list[SemanticSegment]:
    """Segment many units sequentially (adequate for course-scale workloads)."""
    settings = settings or get_settings()
    client = (
        OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None)
        if settings.openai_api_key
        else None
    )
    all_seg: list[SemanticSegment] = []
    for u in units:
        all_seg.extend(segment_unit(u, settings=settings, client=client))
    return all_seg
