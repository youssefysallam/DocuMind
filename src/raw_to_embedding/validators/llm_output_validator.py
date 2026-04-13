"""Validate and normalize LLM JSON outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from raw_to_embedding.models import CandidateUnit
from raw_to_embedding.schemas import LLMSegmentationResponse


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse first JSON object from model output (tolerate stray whitespace)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def parse_and_validate(raw_text: str) -> LLMSegmentationResponse:
    """Parse JSON string and validate against pydantic schema."""
    data = extract_json_object(raw_text)
    return LLMSegmentationResponse.model_validate(data)


_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())


def content_derived_from_source(source: str, segment_content: str, *, min_ratio: float = 0.85) -> bool:
    """
    Heuristic: segment text should be largely copyable from source (anti-hallucination).

    Uses word coverage on whitespace-normalized strings; allows minor formatting drift.
    """
    s_src = _normalize(source)
    s_seg = _normalize(segment_content)
    if not s_seg:
        return True
    if s_seg in s_src:
        return True
    words_src = set(s_src.lower().split())
    words_seg = s_seg.lower().split()
    if not words_seg:
        return True
    hit = sum(1 for w in words_seg if w in words_src)
    return (hit / len(words_seg)) >= min_ratio


def validate_against_source(resp: LLMSegmentationResponse, unit: CandidateUnit) -> None:
    """Raise ValueError if segments appear to add content not in the unit."""
    src = unit.content
    for seg in resp.segments:
        if not content_derived_from_source(src, seg.content):
            raise ValueError(
                f"segment {seg.segment_index} content not grounded in source text"
            )


def validate_with_source(raw_text: str, unit: CandidateUnit) -> LLMSegmentationResponse:
    """Full validation pipeline including schema and source grounding."""
    try:
        resp = parse_and_validate(raw_text)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"LLM output invalid: {exc}") from exc
    if resp.unit_id != unit.unit_id:
        raise ValueError("LLM output unit_id does not match candidate unit")
    validate_against_source(resp, unit)
    return resp
