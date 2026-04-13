"""LLM prompts for semantic segmentation (JSON-only, anti-hallucination)."""

from __future__ import annotations

from raw_to_embedding.models import CandidateUnit

SYSTEM_PROMPT = """You are a semantic document segmentation component inside a RAG ingestion pipeline.

Your role is ONLY to split, group, and label existing text so it can be embedded for retrieval.

You are NOT a summarizer. You are NOT a chatbot. You do NOT answer user questions.

Hard rules:
- Preserve factual detail from the source. Do not paraphrase away numbers, names, dates, or technical terms unless you must split a passage (then copy verbatim from the source).
- Do NOT add external knowledge. Do NOT invent names, dates, methods, findings, conclusions, or citations.
- Do NOT over-summarize. Each segment's "content" must be composed ONLY of substrings taken from the provided unit text (you may reorder and group sentences, but must not introduce new facts).
- Split only when semantically necessary: multiple topics, long noisy blocks, or mixed sections that harm retrieval.
- Keep each segment independently meaningful for retrieval.
- Reuse titles when accurate. If you refine a title, do so conservatively and only to reflect what is actually in that segment's text.
- Preserve traceability: every segment must include the required metadata object with correct unit_id, source_type, document_type, source, url, page_start, page_end, original_title, section.

Output format:
- Return a single JSON object ONLY. No markdown. No code fences. No commentary before or after JSON.
- The JSON must match the required schema with all required fields populated.
"""


def build_general_unit_prompt(candidate_unit: CandidateUnit) -> str:
    """User prompt for website_page and institute_report_pdf units."""
    meta = candidate_unit.metadata or {}
    return f"""Segment the following candidate unit for embedding. Document types: website or institute report style.

unit_id: {candidate_unit.unit_id}
document_type: {candidate_unit.document_type}
source_type: {candidate_unit.source_type}
source: {candidate_unit.source}
url: {candidate_unit.url}
page_start: {candidate_unit.page_start}
page_end: {candidate_unit.page_end}
title: {candidate_unit.title}
section: {candidate_unit.section}
extra_metadata_json: {meta}

FULL UNIT TEXT (only source of truth for content):
---
{candidate_unit.content}
---

Return JSON with this structure (values must be real, content-only from the text above):
{{
  "unit_id": "{candidate_unit.unit_id}",
  "is_multi_topic": <boolean>,
  "reasoning_brief": "<short internal justification, no chain-of-thought>",
  "segments": [
    {{
      "segment_index": 0,
      "title": "<string>",
      "section": "<string or null>",
      "content_type": "<one of allowed enum values>",
      "content": "<verbatim-composed from source only>",
      "preserve_as_single_chunk": <boolean>,
      "metadata": {{
        "unit_id": "{candidate_unit.unit_id}",
        "source_type": "{candidate_unit.source_type}",
        "document_type": "{candidate_unit.document_type}",
        "source": "{candidate_unit.source}",
        "url": {repr(candidate_unit.url)},
        "page_start": {repr(candidate_unit.page_start)},
        "page_end": {repr(candidate_unit.page_end)},
        "original_title": {repr(candidate_unit.title)},
        "section": {repr(candidate_unit.section)}
      }}
    }}
  ]
}}

Allowed content_type values:
mission, people, initiative, event, partnership, publication, organization, contact, program, resource,
academic_overview, academic_method, academic_dataset, academic_result, academic_conclusion, other

If the unit is single-topic and coherent, you may return one segment covering the full text (still verbatim from source).
"""


def build_paper_unit_prompt(candidate_unit: CandidateUnit) -> str:
    """User prompt for scholarly_paper_pdf units (section-aware, technical preservation)."""
    meta = candidate_unit.metadata or {}
    return f"""Segment the following scholarly paper candidate unit for embedding.

Priorities:
- Preserve technical terminology, method names, dataset names, and named results exactly as in the source.
- Preserve findings, limitations, and quantitative statements; do not flatten into generic summaries.
- Prefer section-aware academic segmentation (e.g., methods vs results vs discussion) when the text clearly supports it.
- Do NOT invent citations or paper structure not present in the text.

unit_id: {candidate_unit.unit_id}
document_type: {candidate_unit.document_type}
source_type: {candidate_unit.source_type}
source: {candidate_unit.source}
url: {candidate_unit.url}
page_start: {candidate_unit.page_start}
page_end: {candidate_unit.page_end}
title: {candidate_unit.title}
section: {candidate_unit.section}
extra_metadata_json: {meta}

FULL UNIT TEXT (only source of truth for content):
---
{candidate_unit.content}
---

Return JSON with this structure (content must be composed only from the text above):
{{
  "unit_id": "{candidate_unit.unit_id}",
  "is_multi_topic": <boolean>,
  "reasoning_brief": "<short internal justification>",
  "segments": [
    {{
      "segment_index": 0,
      "title": "<string>",
      "section": "<string or null>",
      "content_type": "<one of allowed enum values>",
      "content": "<verbatim-composed from source only>",
      "preserve_as_single_chunk": <boolean>,
      "metadata": {{
        "unit_id": "{candidate_unit.unit_id}",
        "source_type": "{candidate_unit.source_type}",
        "document_type": "{candidate_unit.document_type}",
        "source": "{candidate_unit.source}",
        "url": {repr(candidate_unit.url)},
        "page_start": {repr(candidate_unit.page_start)},
        "page_end": {repr(candidate_unit.page_end)},
        "original_title": {repr(candidate_unit.title)},
        "section": {repr(candidate_unit.section)}
      }}
    }}
  ]
}}

Prefer content_type values aligned with academic structure when appropriate:
academic_overview, academic_method, academic_dataset, academic_result, academic_conclusion, publication, other
"""


def build_repair_prompt(invalid_snippet: str, validation_errors: str) -> str:
    """Ask the model to emit corrected JSON only."""
    return f"""The previous output failed validation.

Validation errors:
{validation_errors}

Invalid or malformed output (may be truncated):
---
{invalid_snippet}
---

Return ONLY a single valid JSON object that satisfies the semantic segmentation schema (same fields as before).
No markdown. No commentary."""
