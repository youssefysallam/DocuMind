# Raw → Embedding Chunks Agent

Production-style **ingestion-only** pipeline for a RAG system: extract raw website/PDF text, clean deterministically, classify the document, build candidate units, optionally refine segmentation with an LLM (JSON-only, anti-hallucination), then emit **embedding-ready chunks** with traceability metadata.

This project is **not** a chatbot, summarizer, retrieval engine, or vector store. It does **not** implement FAISS or answer generation.

## Pipeline

1. **Extract** — PyMuPDF (PDFs, page-level text) or `requests` + BeautifulSoup (websites, heading-aware sections).
2. **Clean** — whitespace normalization, safe line merging, page-number noise, repeated headers/footers (`utils/text_cleaning.py`). Runs **before** any LLM call.
3. **Classify** — `website_page` | `institute_report_pdf` | `scholarly_paper_pdf` (`classifiers/document_classifier.py`).
4. **Candidate units** — rule-based structure by document type (`processors/candidate_unit_builder.py`).
5. **Semantic segmentation** — LLM only when heuristics say the unit is long, noisy, generic-titled, or ambiguous (`processors/semantic_segmentation_agent.py`).
6. **Validate** — Pydantic schemas + source-grounding check (`validators/llm_output_validator.py`).
7. **Repair** — one JSON-only repair pass on validation failure.
8. **Fallback** — deterministic sentence grouping, no paraphrase (`processors/fallback_segmentation.py`).
9. **Embedding chunks** — sentence-aware packing; `embedding_text` built as documented below (`processors/embedding_chunk_builder.py`).

## Anti-hallucination design

- **Rule-first, LLM-when-needed:** short, clean, single-topic units skip the LLM.
- **LLM role:** split, label with `content_type`, reorder **only** text present in the candidate unit; **no** new facts.
- **Validation:** Pydantic + heuristic check that segment text is grounded in the source unit.
- **Fallback:** preserves original wording; splits by sentences only.

## Configuration

Copy `.env.example` to `.env` and set at least `OPENAI_API_KEY` when you want LLM segmentation for difficult units.

Key knobs:

| Variable | Meaning |
|----------|---------|
| `MAX_UNIT_CHARS_SKIP_LLM` | Units **longer** than this (chars) are more likely to use the LLM (with other triggers). |
| `MIN_CHARS_MULTI_TOPIC_HEURISTIC` | Additional length threshold for “likely complex” units. |
| `GENERIC_TITLE_PATTERNS` | Comma-separated substrings; generic titles favor LLM. |
| `MAX_CHUNK_CHARS_SOFT` | Soft cap for packing **sentences** into embedding chunks (not arbitrary character slicing). |

## CLI

```bash
pip install -r requirements.txt
python main.py \
  --pdf ./data/institute_reports/report1.pdf ./data/scholarly_papers/paper1.pdf \
  --url https://example.org/about https://example.org/projects \
  --output ./output/embedding_chunks.json \
  --save-intermediate
```

- **`--save-intermediate`** writes `candidate_units_<doc_id>.json` and `semantic_segments_<doc_id>.json` next to `--output` (or `--intermediate-dir`).
- **`--verbose`** enables DEBUG logs.

## Output JSON

Top-level object:

- `chunks`: list of `EmbeddingChunk` records (`chunk_id`, `unit_id`, `title`, `section`, `content_type`, `content`, `embedding_text`, `metadata`).
- `errors`: per-input error messages (pipeline continues for other inputs).
- `stats`: counts.

`embedding_text` format:

```text
Title: {title}
Section: {section}   # omitted when section is null
Type: {content_type}
Content: {content}
```

## Document types and content types

**`document_type`:** `website_page`, `institute_report_pdf`, `scholarly_paper_pdf`.

**`content_type`:** `mission`, `people`, `initiative`, `event`, `partnership`, `publication`, `organization`, `contact`, `program`, `resource`, `academic_overview`, `academic_method`, `academic_dataset`, `academic_result`, `academic_conclusion`, `other`.

## LLM JSON schema

The segmentation model must return a single JSON object matching `schemas.LLMSegmentationResponse` (see `schemas.py` and `prompts.py`).

## Project layout

See repository tree: `extractors/`, `processors/`, `validators/`, `utils/`, `examples/sample_inputs.md`.

## Requirements

Python 3.10+ recommended. Dependencies: `pymupdf`, `requests`, `beautifulsoup4`, `pydantic`, `python-dotenv`, `openai`.
