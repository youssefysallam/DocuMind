# SSL RAG V1 — Retrieval-Augmented Generation for the Sustainable Solutions Lab

A research-grade RAG system built over the complete corpus of the **Sustainable Solutions Lab (SSL)** at UMass Boston. An **AI Agent–driven embedding pipeline** (`raw_to_embedding`) ingests 18 PDF research reports and 20+ website pages through LLM-assisted semantic segmentation—with automatic validation, self-repair, and anti-hallucination checks—into a unified 5,227-chunk embedding index. The RAG pipeline then answers stakeholder-style questions using hybrid retrieval, curated QA memory, and grounded LLM generation with explicit abstention for unsupported topics.

---

## 1. Project Overview

### What it does

RAG V1 answers natural-language questions about SSL's research, people, projects, and publications. It handles six question families—from simple factual lookups ("Who leads SSL?") to multi-document synthesis ("How has SSL's portfolio evolved over time?") to deliberate out-of-scope probes ("Does SSL study ocean acidification?")—and is designed to **refuse** rather than hallucinate when evidence is insufficient.

### Problem it solves

Standard RAG pipelines over noisy, multi-source corpora suffer from three failure modes:

1. **Source confusion** — the LLM cannot tell which document a chunk came from, leading to misattribution.
2. **Tangential evidence hallucination** — a keyword match on a passing mention triggers a fabricated positive answer.
3. **QA memory dominance** — curated answers suppress raw evidence, masking retrieval quality.

RAG V1 addresses these with **contextualized chunk prefixes** (each chunk is prepended with its source title and SSL attribution), **hybrid dense + sparse retrieval** with cross-encoder reranking, and **gated QA memory** that injects no-evidence entries only under controlled conditions.

---

## 2. System Architecture

The pipeline in `src/rag_v1/pipeline.py` implements four stages:

```
Query → Intent Classification → Hybrid Retrieval → Reranking + QA Merge → Grounded Generation
```

### Stage 1: Intent Classification

A keyword-based heuristic classifies each query into one of six intents:
`general_overview`, `project_initiative`, `topic_specific`, `publication_finding`, `synthesis`, `no_evidence`. The intent determines source-type boosts and QA-NEG gating behavior.

### Stage 2: Hybrid Retrieval

| Channel | Index | Top-K | Text used |
|---------|-------|------:|-----------|
| Dense (FAISS `IndexFlatIP`) | Contextualized embeddings | 20 | `contextualized_text` (prefix + raw) |
| Sparse (BM25) | Token index | 20 | Raw `chunk_text` only |

Candidates are merged, deduplicated (executive-summary vs. full-report; English vs. Spanish bilingual), and scored:

```
hybrid = 0.6 × (dense / max_dense) + 0.4 × (sparse / max_sparse) + source_boost
```

### Stage 3: Cross-Encoder Reranking + QA Memory Merge

- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scores `(query, contextualized_text)` pairs. Final score blends hybrid (35%) and rerank (65%).
- **QA memory (31 entries):** 22 factual + 9 no-evidence. Positive entries are reranked and merged under diversity constraints (min 3 raw, max 2 QA in final top-5). Negative entries are gated: injected only when intent = `no_evidence` **and** top raw rerank score < 0.3.

### Stage 4: Grounded Generation

The LLM (`gpt-4o-mini`) receives structured context in three labeled sections:

```
=== VERIFIED QA KNOWLEDGE ===
=== WEBSITE EVIDENCE ===
=== PDF RESEARCH EVIDENCE ===
```

The system prompt enforces eight strict rules including mandatory refusal when evidence is absent, distrust of tangential mentions, and preference for raw evidence over QA summaries.

---

## 3. Data Processing Pipeline — AI Agent–Driven Embedding

The entire data processing pipeline (`src/raw_to_embedding/`) is implemented as an **autonomous AI Agent** that uses LLM-assisted semantic segmentation. Rather than relying purely on rule-based splitting, the agent invokes `gpt-4o-mini` to intelligently segment complex, multi-topic document sections—with built-in validation, self-repair, and anti-hallucination safeguards.

### Source data

| Source | Count | Location |
|--------|------:|----------|
| PDF reports | 18 | `SSL+PDF/` |
| Website JSON pages | 20+ | `websitedata/website_pages/` |

### Agent pipeline stages (`src/raw_to_embedding/`)

```
Raw PDF/HTML
  │
  ├─ Extraction (pdfplumber for PDFs; JSON parser for website pages)
  ├─ Cleaning & normalization (whitespace, encoding, noise removal)
  ├─ Document classification (rule-based: website_page / institute_report_pdf / scholarly_paper_pdf)
  │
  ├─ Candidate unit construction (structural splitting by headings, page breaks)
  │
  ├─ ★ LLM-Driven Semantic Segmentation Agent ★
  │    ├─ Rule-based triage: should_use_llm(unit) decides per-unit
  │    │    • Units > 3,500 chars → LLM
  │    │    • Units > 4,500 chars (multi-topic heuristic) → LLM
  │    │    • Generic/ambiguous titles → LLM
  │    │    • Noisy formatting (excessive bullets/newlines) → LLM
  │    │    • Small, well-titled units → skip LLM (deterministic single segment)
  │    ├─ LLM call (gpt-4o-mini, temperature=0, JSON mode)
  │    │    • Strict system prompt: "You are NOT a summarizer. Preserve verbatim text."
  │    │    • Anti-hallucination rule: content must be composed only from source substrings
  │    ├─ Pydantic schema validation (LLMSegmentationResponse)
  │    ├─ Anti-hallucination check: 85% word-coverage verification against source text
  │    ├─ Self-repair: if validation fails, error is sent back to LLM for correction
  │    └─ Fallback: if repair also fails, deterministic sentence-based splitting
  │
  ├─ Sentence-aware chunk packing (max 800 chars, max 5 sentences, 2-sentence overlap)
  ├─ Embedding (all-MiniLM-L6-v2, 384-dim, L2-normalized)
  └─ FAISS index construction (per-PDF, then merged into unified index)
```

### Why an Agent?

Traditional chunking uses fixed-size windows or regex-based splitting, which frequently breaks semantic boundaries in complex research reports (e.g., splitting a table from its caption, or merging unrelated subsections). The agent approach:

1. **Decides when intelligence is needed** — simple, well-structured sections bypass the LLM entirely (saving cost and latency). Complex multi-topic blocks get LLM segmentation.
2. **Validates its own output** — a Pydantic schema enforces structural correctness; a word-coverage check (`content_derived_from_source`, 85% threshold) prevents the LLM from injecting fabricated content into chunks.
3. **Self-repairs on failure** — if the LLM's JSON output fails validation, the agent automatically sends the error message back to the LLM for a correction attempt, before falling back to heuristic splitting.

### Chunk metadata fields

Each chunk in `unified_index_metadata.json` contains:

| Field | Description |
|-------|-------------|
| `chunk_id` | Deterministic hash |
| `chunk_text` | Original cleaned text |
| `source_pdf` / `source_file` | Origin document |
| `source_type` | `pdf` or `website` |
| `section_title` | Detected section heading |
| `chunk_type` | `paragraph`, `table`, `metadata` |
| `page_start` / `page_end` | PDF page span |
| `quality_flag` | `clean` or flagged |

### Contextualized chunks (RAG V1 addition)

At RAG pipeline load time, each chunk receives a deterministic natural-language prefix:

**PDF example:**
```
This passage is from the SSL report 'Financing Climate Resilience',
published by the Sustainable Solutions Lab at UMass Boston.
Section: Finance Needs.

[original chunk_text]
```

**Website example:**
```
This passage is from the SSL website page 'SSL Projects & Initiatives'.
It describes SSL projects and initiatives.
Section: C3I.

[original chunk_text]
```

The `contextualized_text` is used for **dense FAISS embeddings** and **cross-encoder reranking input**, while **BM25** operates on the raw `chunk_text` to preserve keyword precision. This separation is a deliberate design choice.

### Final corpus statistics

| Metric | Value |
|--------|------:|
| Total chunks (unified) | 5,227 |
| PDF chunks | ~5,121 |
| Website chunks | ~106 |
| Embedding dimension | 384 |
| QA memory entries | 31 (22 factual + 9 no-evidence) |

---

## 4. Evaluation Dataset (70 Questions)

**Location:** `data/eval_70/stakeholder_eval_70.json`

A stakeholder-aligned benchmark simulating realistic SSL user queries derived from corpus content and meeting priorities.

### Distribution

| Question Type | Count | % |
|---------------|------:|---:|
| `general_overview` | 15 | 21.4% |
| `topic_specific` | 15 | 21.4% |
| `no_evidence` | 15 | 21.4% |
| `project_initiative` | 12 | 17.1% |
| `synthesis` | 8 | 11.4% |
| `publication_finding` | 5 | 7.1% |

| Difficulty | Count |
|------------|------:|
| Easy | 18 |
| Medium | 36 |
| Hard | 16 |

| Expected Behavior | Count |
|--------------------|------:|
| Answer | 53 |
| Partial answer | 2 |
| Refuse | 15 |

| Gold Source Type | Count |
|-------------------|------:|
| Website | 20 |
| Mixed (PDF + website) | 20 |
| PDF only | 15 |
| No evidence | 15 |

---

## 5. Experiment Results

All metrics from `results/final/rag_v1_metrics.json` — no fabricated numbers.

### Overall metrics

| Metric | RAG V1 | Baseline |
|--------|-------:|--------:|
| Answer accuracy (answerable, n=55) | **79.1%** | 72.7% |
| Answer accuracy (all, n=70) | **71.4%** | 67.9% |
| Completeness | **66.4%** | 63.6% |
| Hallucination rate | 24.3% | 24.3% |
| Coverage (appropriate behavior) | 92.9% | 95.7% |
| Correct refusal rate (of 15) | 80.0% | 100.0% |
| False refusal rate (of 55) | **1.8%** | 5.5% |
| Missed refusals | 3 | 0 |
| Raw corpus hit@5 | 18.2% | 16.4% |

*Baseline = archived pre-contextualization configuration on the same 70-question set.*

### By question type

| Type | N | Behavior rate | Avg accuracy | Perfect (score=2) |
|------|--:|:---:|:---:|---:|
| `general_overview` | 15 | 100% | **90.0%** | 12/15 |
| `topic_specific` | 15 | 100% | **90.0%** | 12/15 |
| `project_initiative` | 12 | 100% | 83.3% | 8/12 |
| `publication_finding` | 5 | 80% | 60.0% | 2/5 |
| `synthesis` | 8 | 100% | 43.8% | 0/8 |
| `no_evidence` | 15 | 73.3% | 43.3% | 1/15 |

---

## 6. Error Analysis

From `results/final/rag_v1_error_analysis.json` — 5 items with `appropriate_behavior < 2`:

### Error breakdown

| Error ID | Type | Root cause | Description |
|----------|------|------------|-------------|
| PUB-03 | **False refusal** | Retrieval failure | "Multilingual publications?" — system retrieved Rajini Srikanth's personal publication list instead of the Spanish report *Oportunidad en la Complejidad*. Refused despite evidence existing in corpus. |
| NEG-01 | **Missed refusal** (hallucination) | Intent misclassification | "SSL's work on reducing carbon emissions?" — classified as `general_overview` instead of `no_evidence`, so QA-NEG was not injected. LLM extrapolated from tangential carbon-tax mention in Financing report. |
| NEG-04 | **Missed refusal** (hallucination) | Intent misclassification + tangential evidence | "SSL and food systems?" — classified as `general_overview`. LLM fabricated "food systems" narrative from flood-control regulatory text (OCR confusion between "flood" and "food"). |
| NEG-09 | **Weak refusal** | Partial intent match | "SSL international work?" — correctly refused but explanation lacked detail about B.R. Balachandran's international background. Scored `appropriate_behavior=0` by LLM judge. |
| NEG-10 | **Missed refusal** (hallucination) | Intent misclassification | "SSL sea-level rise modeling?" — classified as `general_overview`. LLM constructed a false narrative from sea-level benchmarks mentioned in Governance report context. |

### Error type summary

| Category | Count | Proportion |
|----------|------:|----------:|
| Missed refusal (should refuse, answered instead) | 3 | 60% |
| False refusal (should answer, refused instead) | 1 | 20% |
| Weak behavior (correct direction, insufficient quality) | 1 | 20% |

**Dominant failure mode:** Intent misclassification for **no-evidence** questions. When a query about an unsupported topic (carbon emissions, food systems, sea-level modeling) is not detected by the keyword-based intent classifier, QA-NEG entries are never injected, and the LLM generates answers from tangentially related corpus passages.

---

## 7. How to Run

### Prerequisites

- Python ≥ 3.10
- An OpenAI-compatible API key (OpenAI direct or [OpenRouter](https://openrouter.ai))

### 7.1 Install

```bash
git clone https://github.com/youssefysallam/institute-rag.git
cd institute-rag
git checkout feat/embeddings
pip install -e .
pip install gradio
```

### 7.2 Configure API key

```bash
copy .env.example .env
```

Open `.env` and fill in the API credentials:

**Option A — OpenRouter (recommended, supports gpt-5.4):**

```
OPENAI_API_KEY=sk-or-v1-your-key-here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
LLM_MODEL_V2=openai/gpt-5.4
```

**Option B — Official OpenAI directly:**

```
OPENAI_API_KEY=sk-your-openai-key
LLM_MODEL=gpt-4o-mini
LLM_MODEL_V2=gpt-5.4
```

> **Important:** If you are NOT using OpenRouter, do NOT set `OPENAI_BASE_URL` — leave it commented out or delete the line entirely. An empty value will cause errors.

### 7.3 Launch RAG V2 Web UI

```bash
set PYTHONPATH=src
python -m rag_v2.app
```

Wait for `[UI] System ready.` to appear in the terminal, then open **http://localhost:7860** in your browser.

The UI has three tabs:
- **Chat** — multi-turn conversation with source attribution and consistency check
- **Evaluation** — V1 vs V2 metrics dashboard
- **System** — configuration and feature summary

> **First launch will be slow (~30s):** it downloads embedding models (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`) and builds the FAISS index. Subsequent launches use the cached index and start in ~5s.

### 7.4 Launch RAG V2 CLI (no browser needed)

```bash
set PYTHONPATH=src
python -m rag_v2.demo
```

Type questions in the terminal. Supports multi-turn conversation. Type `quit` to exit.

### 7.5 Run V1 evaluation (70 questions)

```bash
set PYTHONPATH=src
python -m rag_v1.pipeline
```

This loads the corpus, evaluates all 70 questions, and saves results to `results/final/`.

### 7.6 Run raw-to-embedding agent

```bash
set PYTHONPATH=src
python -m raw_to_embedding.pipeline --pdf path/to/doc.pdf -o output/chunks.json
python -m raw_to_embedding.pipeline --help
```

---

## 8. Repository Structure

```
├── src/
│   ├── rag_v1/                     # RAG V1 pipeline (retrieval + generation + eval)
│   │   ├── pipeline.py             # Main entry point
│   │   └── __init__.py
│   ├── rag_v2/                     # RAG V2 pipeline (multi-turn, intent-driven)
│   │   ├── pipeline.py             # Core ask() function + evaluation
│   │   ├── app.py                  # Gradio Web UI
│   │   ├── demo.py                 # CLI multi-turn demo
│   │   ├── multiturn.py            # Coreference resolution + retrieval strategy
│   │   ├── intent_classifier.py    # Hybrid keyword + LLM intent classification
│   │   ├── query_transform.py      # HyDE, sub-query decomposition
│   │   ├── consistency.py          # Answer-evidence consistency check
│   │   ├── session.py              # Session memory for multi-turn
│   │   └── __init__.py
│   └── raw_to_embedding/           # AI Agent–driven embedding pipeline
│       ├── pipeline.py             # CLI entry point
│       ├── parser.py               # PDF / website extraction
│       ├── chunker.py              # Segmentation → chunks
│       ├── embedder.py             # Sentence-transformer helper
│       ├── corpus_build/           # SSL-specific builders
│       ├── extractors/             # PDF and HTML extractors
│       ├── processors/             # LLM segmentation agent, chunking
│       ├── validators/             # Anti-hallucination validation
│       └── utils/                  # Text cleaning, I/O
├── data/
│   ├── eval_70/                    # 70-question evaluation dataset (Phase 1)
│   ├── eval_phase2/                # 104-question dataset (Phase 2, incl. multi-turn)
│   └── rag_v1/                     # QA memory + contextualized chunks
├── data/final_corpus_bundle/       # Unified PDF + website corpus (embeddings + FAISS)
├── results/
│   ├── final/                      # V1 evaluation outputs
│   └── v2/                         # V2 evaluation outputs + V1-vs-V2 comparison
├── report/                         # Experiment reports (EN + ZH)
├── scripts/                        # Report generation + V1 Phase 2 eval
├── configs/                        # Path conventions
├── pyproject.toml
└── .env.example
```

---

## 9. Notes

- **Single version policy:** Previous experimental iterations (internally numbered v2–v7) were removed during repository normalization. Only the final consolidated implementation is retained as **RAG V1**. Historical evolution can be recovered from Git history.
- **Reproducibility:** Re-running `python -m rag_v1.pipeline` may produce slightly different scores due to LLM judge nondeterminism. The frozen metrics in `results/final/` represent the canonical evaluation.
- **Corpus content:** All corpus material belongs to SSL and upstream publishers. This repository contains code and derived indices for research and demonstration purposes.

---

## License

Research use only. Corpus content belongs to the Sustainable Solutions Lab and its publishers.
