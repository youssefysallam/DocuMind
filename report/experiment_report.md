# RAG V1 Experiment Report: Agent-Driven Embedding and Contextualized Hybrid Retrieval for Institutional Knowledge QA

## 1. Introduction

This report documents the design, evaluation, and error analysis of **RAG V1**, a retrieval-augmented generation system built over the complete corpus of the Sustainable Solutions Lab (SSL) at UMass Boston. The system uses an **AI Agent–driven embedding pipeline** (`raw_to_embedding`) that employs LLM-assisted semantic segmentation—with automatic validation, self-repair, and anti-hallucination safeguards—to process 18 PDF research reports and 20+ scraped website pages into a unified 5,227-chunk embedding index. A downstream RAG pipeline then answers stakeholder-style natural-language queries about SSL's mission, people, projects, publications, and research themes.

RAG V1 consolidates seven prior experimental iterations into a single production configuration. It features two key technical contributions: (1) an **agentic document processing pipeline** that decides per-unit whether LLM intelligence is needed for segmentation, validates its own outputs against the source text, and self-repairs on failure; and (2) **contextualized knowledge units**—deterministic natural-language prefixes prepended to each chunk at retrieval time—that provide the downstream LLM with document-level provenance without altering the underlying chunk boundaries or raw text.

## 2. Problem Statement

Institutional knowledge bases present three challenges that generic RAG pipelines handle poorly:

1. **Provenance opacity.** Chunks extracted from PDFs lose their document title, section heading, and organizational attribution. The LLM sees a passage about "financing mechanisms" but cannot determine whether it comes from an SSL report, a government policy document, or a third-party reference.

2. **Tangential evidence hallucination.** When a keyword appears in passing (e.g., "carbon tax" mentioned once in a financing report), a standard pipeline retrieves the chunk and the LLM may fabricate a narrative claiming the institution has a dedicated carbon emissions program.

3. **QA memory dominance.** Curated question–answer pairs intended to improve recall for frequent questions can suppress raw corpus chunks from the top-k, masking retrieval quality and creating over-reliance on pre-authored content.

RAG V1 addresses (1) through contextualized prefixes, (2) through a strict generation prompt with eight grounding rules, and (3) through diversity-constrained merge logic that guarantees at least 3 raw corpus chunks in every final top-5.

## 3. System Design

### 3.1 Pipeline overview

```
Query
  │
  ├─ Intent classifier (keyword heuristic → 6 categories)
  │
  ├─ Dense retrieval (FAISS IndexFlatIP, contextualized embeddings, top-20)
  ├─ Sparse retrieval (BM25Okapi, raw chunk text, top-20)
  │
  ├─ Merge + deduplicate
  │    ├─ Executive-summary / full-report dedup
  │    └─ English / Spanish bilingual dedup
  │
  ├─ Hybrid scoring
  │    hybrid = 0.6 × (dense / max_dense) + 0.4 × (sparse / max_sparse) + source_boost
  │
  ├─ Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  │    final = 0.35 × hybrid + 0.65 × (rerank_score / 10)
  │
  ├─ QA memory retrieval (FAISS over 31 entries, relevance floor 0.35)
  │    ├─ Positive entries: reranked, max 2 in final top-5
  │    └─ Negative entries: injected only when intent = no_evidence AND top raw score < 0.3
  │
  ├─ Diversity-constrained merge (min 3 raw, max 2 QA → final top-5)
  │
  └─ Grounded generation (gpt-4o-mini, temperature=0, max 600 tokens)
       ├─ Structured context: VERIFIED QA KNOWLEDGE / WEBSITE EVIDENCE / PDF RESEARCH EVIDENCE
       └─ 8-rule system prompt with mandatory refusal policy
```

### 3.2 Models

| Component | Model | Dimension / Detail |
|-----------|-------|--------------------|
| Embedding | `all-MiniLM-L6-v2` | 384-d, L2-normalized |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Pointwise reranker |
| Generator | `gpt-4o-mini` | Temperature 0 |
| LLM judge | `gpt-4o-mini` | Evaluation scoring |

### 3.3 Key hyperparameters

| Parameter | Value | Rationale |
|-----------|------:|-----------|
| `TOP_K_DENSE` | 20 | Captures broad dense recall |
| `TOP_K_SPARSE` | 20 | Parallel BM25 candidate pool |
| `TOP_K_FINAL` | 5 | Balances context length and generation quality |
| `MAX_QA_IN_FINAL` | 2 | Prevents QA dominance (reduced to 1 for `publication_finding`, `topic_specific`) |
| `MIN_RAW_IN_FINAL` | 3 | Guarantees raw evidence presence |
| `QA_RELEVANCE_FLOOR` | 0.35 | Filters low-similarity QA matches |
| Hybrid weight (dense:sparse) | 0.6 : 0.4 | Favors dense semantic matching |
| Final blend (hybrid:rerank) | 0.35 : 0.65 | Reranker dominates final ordering |

### 3.4 Intent-dependent source boosts

| Intent | Website boost | PDF boost |
|--------|:---:|:---:|
| `general_overview` | +0.08 | −0.03 |
| `topic_specific` | 0.00 | +0.04 |
| `project_initiative` | +0.04 | +0.04 |
| `publication_finding` | −0.04 | +0.10 |
| `synthesis` | +0.04 | +0.04 |
| `no_evidence` | 0.00 | 0.00 |

## 4. Data Processing Pipeline — AI Agent–Driven Embedding

The data processing stage is not a simple rule-based script; it is implemented as an **autonomous AI Agent** (`src/raw_to_embedding/`) that uses LLM-assisted semantic segmentation with built-in validation, self-repair, and anti-hallucination safeguards.

### 4.1 Source corpus

**PDFs (18 documents):** SSL research reports spanning harbor barrier feasibility, climate governance, financing resilience, municipal vulnerability, climate migration, transient populations, communities of color, East Boston resilience, housing and climate, community-led preparedness, annual reports, and an impact report. Total: ~5,121 chunks.

**Website (20+ pages):** SSL homepage, people directory, project pages, research pages, student pages, board of directors, overview presentation, news articles, and individual researcher profiles. Total: ~106 chunks.

### 4.2 Agent architecture

The embedding agent pipeline consists of five stages, with the third stage being LLM-driven:

**Stage 1 — Extraction.** `extractors/pdf_extractor.py` uses `pdfplumber` for text and table extraction from PDFs; `extractors/website_extractor.py` parses JSON-structured website data from scraped pages.

**Stage 2 — Classification and candidate construction.** `classifiers/document_classifier.py` uses rule-based keyword scoring to classify each document as `website_page`, `institute_report_pdf`, or `scholarly_paper_pdf`. `processors/candidate_unit_builder.py` then segments documents into candidate units based on headings, page breaks, and structural markers.

**Stage 3 — LLM-driven semantic segmentation agent** (`processors/semantic_segmentation_agent.py`). This is the core agentic component:

1. **Rule-based triage (`should_use_llm`):** For each candidate unit, heuristics determine whether LLM segmentation is necessary:
   - Units > 3,500 characters → invoke LLM
   - Units > 4,500 characters (multi-topic heuristic) → invoke LLM
   - Generic or ambiguous section titles → invoke LLM
   - Noisy formatting (excessive bullets, whitespace) → invoke LLM
   - Small, well-titled units → bypass LLM entirely (zero-cost deterministic path)

2. **LLM invocation:** When triggered, the agent calls `gpt-4o-mini` (temperature=0, JSON response mode) with a strict system prompt:
   > *"You are a semantic document segmentation component inside a RAG ingestion pipeline. You are NOT a summarizer. You are NOT a chatbot. Preserve factual detail from the source. Do NOT add external knowledge. Each segment's content must be composed ONLY of substrings taken from the provided unit text."*

   Document-type-specific user prompts (`build_general_unit_prompt` for website/institute reports, `build_paper_unit_prompt` for scholarly papers) provide the unit text and request structured JSON output with segment titles, content types, and metadata.

3. **Validation pipeline:**
   - **Pydantic schema validation** (`schemas.py → LLMSegmentationResponse`): enforces required fields, segment structure, and content type enums.
   - **Anti-hallucination check** (`validators/llm_output_validator.py → content_derived_from_source`): verifies that ≥ 85% of each segment's words can be traced back to the source text. If the LLM injected fabricated content, validation fails.

4. **Self-repair loop:** If validation fails, the agent sends the error message back to the LLM for a correction attempt (`build_repair_prompt`). This gives the LLM a second chance to produce valid output.

5. **Deterministic fallback:** If the repair attempt also fails, the agent falls back to rule-based sentence-level splitting (`processors/fallback_segmentation.py`), ensuring the pipeline never halts.

**Stage 4 — Sentence-aware chunk packing.** `processors/embedding_chunk_builder.py` packs semantic segments into embedding-ready chunks: max 800 characters, max 5 sentences per group, 2-sentence overlap. Quality filters remove chunks below 120 characters.

**Stage 5 — Embedding and index construction.** `all-MiniLM-L6-v2` encodes chunks into 384-dimensional L2-normalized vectors. Per-PDF FAISS `IndexFlatIP` indices are built, then all per-document indices and website chunks are merged into a single unified index.

### 4.3 Why an agent?

Traditional chunking approaches (fixed-size windows, regex-based splitting) frequently break semantic boundaries in complex research reports—splitting a table from its caption, merging unrelated subsections, or cutting mid-paragraph. The agent approach provides:

| Property | Traditional chunker | Agent-based segmentation |
|----------|:---:|:---:|
| Semantic boundary awareness | Low | High (LLM-driven) |
| Cost for simple sections | Low | **Also low** (heuristic bypass) |
| Hallucination risk in output | None | **Controlled** (85% word-coverage validation) |
| Failure resilience | N/A | Self-repair + fallback |
| Reproducibility | Deterministic | Deterministic for bypass; near-deterministic (temp=0) for LLM |

### 4.4 Metadata schema

Each chunk carries: `chunk_id` (hash), `chunk_text`, `source_pdf`/`source_file`, `source_type` (pdf/website), `section_title`, `chunk_type` (paragraph/table/metadata), page range, quality flag, and character count.

### 4.5 Contextualization (RAG V1 addition)

At RAG pipeline load time (not during the embedding agent's execution), each chunk receives a deterministic `contextualized_text` field:

- **PDF chunks:** `"This passage is from the SSL report '[Pretty Title]', published by the Sustainable Solutions Lab at UMass Boston.\nSection: [section_title].\n\n[chunk_text]"`
- **Website chunks:** `"This passage is from the SSL website page '[Page Title]'.\nIt describes [category hint].\nSection: [section_title].\n\n[chunk_text]"`

A mapping of 17 PDF filename stems to human-readable titles and 7 website file identifiers to page names ensures consistent, readable prefixes.

Dense FAISS embeddings are then computed from `contextualized_text`. BM25 operates on raw `chunk_text`. This dual-text strategy preserves keyword retrieval precision while giving dense search access to document-level context.

## 5. Dataset Analysis

### 5.1 Construction

The evaluation set (`data/eval_70/stakeholder_eval_70.json`) contains 70 questions designed to simulate realistic SSL stakeholder queries. Questions span six families:

| Question Type | N | % | Description |
|---------------|--:|---:|-------------|
| `general_overview` | 15 | 21.4 | SSL identity, mission, vision, leadership, location |
| `topic_specific` | 15 | 21.4 | Housing + climate, homelessness, migration, governance, health |
| `no_evidence` | 15 | 21.4 | Out-of-scope: carbon emissions, e-waste, biodiversity, nuclear, etc. |
| `project_initiative` | 12 | 17.1 | C3I, CLIIR, Cape Cod Rail, East Boston, Barr Foundation |
| `synthesis` | 8 | 11.4 | Cross-cutting themes, evolution of portfolio, advisory board analysis |
| `publication_finding` | 5 | 7.1 | Specific report findings, multilingual publications |

### 5.2 Difficulty and source distribution

| Difficulty | Count | | Gold Source Type | Count |
|------------|------:|-|-------------------|------:|
| Easy | 18 | | Website | 20 |
| Medium | 36 | | Mixed (PDF + web) | 20 |
| Hard | 16 | | PDF only | 15 |
| | | | No evidence | 15 |

### 5.3 Design rationale

- **21.4% no-evidence questions** stress-test abstention behavior. Each probes a plausible but unsupported topic (agriculture, mental health, ocean acidification, blockchain).
- **28.6% synthesis or multi-source questions** require combining information from multiple chunks or source types.
- **Gold answers are human-authored** reference responses with source IDs. Expected behavior labels distinguish `answer` (53), `partial_answer` (2), and `refuse` (15).

## 6. Experimental Setup

### 6.1 Evaluation protocol

For each of the 70 questions:
1. **Intent classification:** Keyword-based heuristic → one of 6 intents.
2. **Retrieval:** Hybrid dense + sparse with QA memory merge → top-5 chunks.
3. **Generation:** `gpt-4o-mini` with structured context and grounding prompt.
4. **Retrieval evaluation:** `raw_hit@3` and `raw_hit@5` against gold source IDs (QA entries excluded).
5. **Answer evaluation:** LLM judge (gpt-4o-mini) scores five dimensions on a 0–2 scale: `factual_accuracy`, `completeness`, `appropriate_behavior`, `hallucination`, `is_refusal`. Keyword-based refusal detection overrides the judge when pattern-matched refusal phrases are found.

### 6.2 Metrics definitions

| Metric | Definition |
|--------|------------|
| `answer_accuracy_answerable` | Mean `factual_accuracy / 2` over 55 answerable questions |
| `answer_accuracy_all` | Mean `factual_accuracy / 2` over all 70 questions |
| `answer_completeness` | Mean `completeness / 2` over all 70 questions |
| `hallucination_rate` | `1 − mean(hallucination / 2)` over 70 questions |
| `coverage` | Fraction of questions with `appropriate_behavior ≥ 1` |
| `correct_refusal_rate` | Of 15 refuse-expected questions, fraction where system refused |
| `false_refusal_rate` | Of 55 answerable questions, fraction where system refused |
| `missed_refusal_count` | Number of refuse-expected questions where system did not refuse |
| `raw_corpus_hit@k` | Among answerable questions with gold source IDs, fraction where at least one gold source appears in top-k raw (non-QA) chunks |
| `evidence_supported_accuracy` | Fraction of questions with both `raw_hit@5 = true` AND `factual_accuracy ≥ 1` |

### 6.3 Baseline

The baseline is the same pipeline without contextualized chunk prefixes (dense embeddings over raw `chunk_text`; all other components identical). Archived as `results/final/baseline_v4_metrics.json`.

## 7. Results

### 7.1 Overall performance

| Metric | Baseline | RAG V1 | Δ |
|--------|:--------:|:------:|:-:|
| Answer accuracy (answerable) | 72.7% | **79.1%** | +6.4% |
| Answer accuracy (all) | 67.9% | **71.4%** | +3.6% |
| Completeness | 63.6% | **66.4%** | +2.9% |
| Hallucination rate | 24.3% | 24.3% | — |
| Coverage | 95.7% | 92.9% | −2.9% |
| Correct refusal rate | **100.0%** | 80.0% | −20.0% |
| False refusal rate | 5.5% | **1.8%** | −3.6% |
| Missed refusals | **0** | 3 | +3 |
| Raw corpus hit@5 | 16.4% | **18.2%** | +1.8% |
| Evidence-supported accuracy | 16.4% | **18.2%** | +1.8% |

### 7.2 Performance by question type

| Type | N | Behavior Rate | Avg Accuracy | Perfect (2/2) |
|------|--:|:---:|:---:|:---:|
| `general_overview` | 15 | 100% | **90.0%** | 12 / 15 |
| `topic_specific` | 15 | 100% | **90.0%** | 12 / 15 |
| `project_initiative` | 12 | 100% | 83.3% | 8 / 12 |
| `publication_finding` | 5 | 80% | 60.0% | 2 / 5 |
| `synthesis` | 8 | 100% | 43.8% | 0 / 8 |
| `no_evidence` | 15 | 73.3% | 43.3% | 1 / 15 |

**Baseline comparison (selected types):**

| Type | Baseline Accuracy | RAG V1 Accuracy | Δ |
|------|:---:|:---:|:---:|
| `general_overview` | 80.0% | **90.0%** | +10.0% |
| `topic_specific` | 70.0% | **90.0%** | +20.0% |
| `project_initiative` | 83.3% | 83.3% | — |
| `synthesis` | 50.0% | 43.8% | −6.3% |
| `no_evidence` | 50.0% | 43.3% | −6.7% |

### 7.3 Key takeaways

**Gains:** Contextualization provides the largest improvement on `topic_specific` questions (+20% accuracy), where source attribution helps the LLM ground its answers in the correct report. `general_overview` questions also improve (+10%) as website page names make SSL identity context explicit.

**Regressions:** `no_evidence` behavior rate dropped from 100% to 73.3%, with 3 missed refusals. The baseline's perfect refusal rate was achieved through aggressive keyword-based intent detection and QA-NEG gating. Contextualized prefixes—which mention "SSL" and "Sustainable Solutions Lab" in every chunk—may inadvertently increase the reranker's confidence on tangentially relevant passages, causing the system to generate answers instead of refusing.

## 8. Error Analysis

### 8.1 Error inventory

Five evaluation items received `appropriate_behavior < 2` from the LLM judge:

| ID | Type | Question | Root Cause |
|----|------|----------|------------|
| **PUB-03** | False refusal | "What multilingual publications has SSL produced?" | Retrieval pulled Rajini Srikanth's personal publications instead of the Spanish report *Oportunidad en la Complejidad*. System refused despite evidence existing in corpus. |
| **NEG-01** | Missed refusal | "What is SSL's work on reducing carbon emissions?" | Intent classified as `general_overview` (matched "what is ssl" pattern). QA-NEG never injected. LLM extrapolated from a single carbon-tax mention in Financing report. |
| **NEG-04** | Missed refusal | "How does SSL address food systems and agriculture?" | Intent classified as `general_overview`. LLM fabricated food-systems narrative from flood-control regulatory text—likely an OCR-induced "flood"→"food" confusion in the Governance report. |
| **NEG-09** | Weak refusal | "What international climate work has SSL done outside the US?" | System correctly refused but provided insufficient detail. LLM judge scored `appropriate_behavior = 0` because the refusal omitted relevant context about B.R. Balachandran's international experience. |
| **NEG-10** | Missed refusal | "What is SSL's position on sea-level rise modeling?" | Intent classified as `general_overview`. LLM constructed false narrative from sea-level benchmarks mentioned in Governance report context. |

### 8.2 Error classification

| Error Category | Count | % of errors |
|----------------|------:|:-----------:|
| **Intent misclassification → missed refusal** | 3 | 60% |
| **Retrieval failure → false refusal** | 1 | 20% |
| **Insufficient refusal detail** | 1 | 20% |

### 8.3 Root cause analysis

**Dominant failure: Intent misclassification.** The keyword-based classifier relies on explicit pattern matching (e.g., "recycling", "e-waste" → `no_evidence`). Queries about carbon emissions, food systems, and sea-level modeling are absent from the `no_evidence` keyword list, so they default to `general_overview` or `topic_specific`. Without the `no_evidence` intent signal, QA-NEG gating is bypassed, and the LLM receives no negative-evidence cue.

**Secondary failure: Tangential evidence hallucination.** Even with contextualized prefixes and strict grounding rules, the LLM occasionally treats incidental mentions (a carbon tax recommendation, flood-control regulations) as evidence of dedicated institutional programs. The system prompt's rule 5 ("If a topic is only tangentially mentioned... do NOT treat it as dedicated SSL work") is insufficiently enforced by `gpt-4o-mini` at temperature 0.

**Tertiary failure: Retrieval precision for rare queries.** The multilingual publications query (PUB-03) failed because the embedding for "multilingual publications" is closer to Rajini Srikanth's personal publication list than to the chunk about *Oportunidad en la Complejidad*. This is a classic vocabulary mismatch: the report does not use the word "multilingual."

## 9. System Performance Summary

### 9.1 Strengths

- **High accuracy on factual questions:** 90.0% on both `general_overview` and `topic_specific`, validating contextualization.
- **Low false refusal rate:** 1.8% (1 of 55 answerable questions falsely refused), indicating the system answers when it should.
- **Effective QA memory integration:** Diversity constraints successfully prevent QA dominance while preserving curated answer quality for frequent questions.
- **Reproducible and deterministic:** Contextualized prefixes are template-based (no LLM involved), ensuring reproducibility.

### 9.2 Weaknesses

- **Fragile abstention:** 3 missed refusals where the system should have refused. The keyword-based intent classifier is the bottleneck.
- **Low synthesis quality:** 0 perfect scores on 8 synthesis questions (avg accuracy 43.8%). Multi-document reasoning remains weak.
- **Low raw retrieval hit rate:** 18.2% hit@5 reflects that gold source IDs often do not match the retrieved chunks' file names precisely, and that many questions are answered correctly from QA memory or nearby chunks rather than the exact gold source.
- **Hallucination rate unchanged:** 24.3% hallucination rate persisted from baseline to RAG V1, suggesting the hallucination issue is primarily in the generation model, not the retrieval stage.

## 10. Future Work

### 10.1 High priority

1. **Learned intent classifier.** Replace the keyword-based heuristic with a lightweight classifier (e.g., fine-tuned `all-MiniLM-L6-v2` on the 70-question dataset) to correctly route out-of-scope queries to the `no_evidence` intent, preventing the dominant error mode.

2. **Confidence-based abstention.** Introduce a rerank-score threshold below which the system automatically refuses, independent of intent classification. If the best reranked chunk scores below a calibrated threshold, the system should abstain.

3. **Stronger tangential-evidence filtering.** Implement a binary classifier or rule-based post-retrieval filter that detects when retrieved chunks only mention a topic incidentally (e.g., a single keyword match in a broader context) and suppresses them before generation.

### 10.2 Medium priority

4. **Retrieval improvement for long-tail queries.** Expand BM25 with synonym expansion or query rewriting to handle vocabulary mismatches (e.g., "multilingual" → "Spanish-language version").

5. **Multi-hop synthesis support.** For synthesis questions, retrieve in two passes: first retrieve to identify relevant documents, then re-retrieve within those documents to gather cross-cutting evidence.

6. **Verifier module.** Add a post-generation verification step that checks whether the generated answer is actually supported by the retrieved chunks, using entailment or NLI.

### 10.3 Low priority

7. **Stronger reranker.** Upgrade to a larger cross-encoder (e.g., `ms-marco-MiniLM-L-12-v2` or `bge-reranker-v2-m3`) for better reranking accuracy.

8. **Chunk boundary optimization.** Experiment with variable-length chunks based on document structure (e.g., preserve entire tables, keep multi-paragraph sections intact).

9. **Multilingual embedding model.** If the corpus expands to include more non-English content, consider `paraphrase-multilingual-MiniLM-L12-v2` for cross-lingual retrieval.

---

## Appendix A: Corpus Smoke Test

Before full RAG evaluation, a retrieval smoke test was run on the unified corpus using 14 website-oriented queries.

| Metric | Value |
|--------|------:|
| Total queries | 14 |
| Website hit@5 | 12 / 14 |
| Hit rate | 85.7% |

The two misses were for queries about graduate students (no dedicated student listing in top-5) and the CLIIR initiative (website mentions scattered across multiple low-ranked pages). Full per-query results are in `results/raw_to_embedding/smoke_test/metrics.json`.

## Appendix B: QA Memory Inventory

The QA memory layer contains 31 entries:
- **22 factual entries** (QA-001 through QA-022): SSL identity, mission, leadership, location, partner institutes, research areas, major reports, advisory board, projects (C3I, CLIIR, Cape Cod Rail, Climate Adaptation Forum), research topics (housing, homelessness, migration, communities of color), key people, community engagement methods.
- **9 no-evidence entries** (QA-NEG-001 through QA-NEG-009): Carbon emissions, agriculture/food, e-waste, biodiversity, air quality, renewable energy, mental health, sea-level modeling, international work.

## Appendix C: Artifact Locations

```
data/eval_70/stakeholder_eval_70.json          # 70-question evaluation dataset
data/final_corpus_bundle/merged/               # Unified FAISS index + metadata (5,227 chunks)
data/rag_v1/qa_memory/qa_memory.json           # 31-entry QA memory
data/rag_v1/contextualized_embeddings.npy      # Contextualized FAISS embeddings (regenerated on first run)
data/rag_v1/contextualized_index.faiss         # Contextualized FAISS index (regenerated on first run)
results/final/rag_v1_metrics.json              # Canonical metrics
results/final/rag_v1_eval_results.json         # Per-question evaluation details
results/final/rag_v1_error_analysis.json       # 5 error cases with full retrieval + eval traces
results/final/baseline_v4_metrics.json         # Baseline comparison metrics
results/raw_to_embedding/smoke_test/           # 14-query corpus smoke test
src/rag_v1/pipeline.py                         # Complete pipeline (883 lines)
src/raw_to_embedding/                          # Document ingestion package (34 modules)
```
