# DocuMind

A multi-turn RAG chatbot that answers questions about a research institute's document corpus with transparent source citations. Two pipeline generations (V1 and V2), a Next.js frontend, and a 70-question evaluation harness.

---

## What it does

Give DocuMind any question about the Sustainable Solutions Lab and it will:
1. Classifies intent and decomposes multi-part queries into sub-questions
2. Retrieves relevant passages with HyDE + BM25 + dense embeddings + cross-encoder reranking
3. Checks answer consistency against retrieved evidence
4. Responds with grounded prose and inline citations — no hallucination on out-of-scope questions

---

## Architecture

### RAG V1 — Hybrid Retrieval Baseline

```
Query
  │
  ├─ BM25 sparse retrieval  ─┐
  │                           ├─ Score fusion → Top-K → Cross-encoder rerank → Answer
  └─ Dense (BGE-small)  ──────┘
```

### RAG V2 — Agentic Multi-turn Pipeline

```
Query + Session Memory
  │
  ├─ [Intent] Classify + decompose (HyDE sub-queries)
  │
  ├─ [Retrieval] BM25 + Dense + Diversity selection + Multi-hop
  │
  ├─ [Rerank] Cross-encoder (ms-marco-MiniLM-L-6-v2)
  │
  ├─ [Consistency] Self-check answer against retrieved evidence
  │
  └─ [Answer] Grounded response + citations
```

V2 adds: HyDE, sub-query decomposition, multi-hop retrieval, session memory with coreference resolution, and consistency checking.

---

## Key Features

- **Multi-turn dialogue** — session memory with a 10-turn window; coreference resolution keeps pronouns coherent across turns
- **HyDE** — generates a hypothetical answer to improve dense retrieval recall
- **Consistency checking** — answer is validated against retrieved evidence before delivery
- **Dual interfaces** — Gradio UI for quick demos; Next.js frontend for the full V2 experience
- **FastAPI backend** — session TTL (1h) + LRU cap (200 sessions) prevents memory leaks under concurrent load
- **70-question evaluation** — stakeholder-curated dataset with precision, recall, and ROUGE metrics

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding | `sentence-transformers` (BGE-small-en-v1.5) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Vector store | FAISS |
| LLM | OpenAI-compatible (configurable) |
| Backend | FastAPI |
| Frontend | Next.js 14 (App Router) + Tailwind CSS |
| Demo UI | Gradio |
| Language | Python 3.11+ |

---

## Project Structure

```
src/
  rag_v1/          Hybrid retrieval baseline (BM25 + dense + rerank)
  rag_v2/
    pipeline.py    Full V2 agentic pipeline
    session.py     Multi-turn session memory
    api.py         FastAPI backend (served to Next.js)
    app.py         Gradio demo UI
  raw_to_embedding/
    pipeline.py    PDF + website ingestion → chunking → embedding → FAISS
frontend/          Next.js 14 chat UI with source panels
data/
  final_corpus_bundle/  Pre-built FAISS indexes + embeddings
  eval_70/              70-question evaluation dataset
```

---

## Quick Start

```bash
pip install -e .
cp .env.example .env   # add OPENAI_API_KEY (or compatible endpoint)

# Run the Gradio demo (V2)
PYTHONPATH=src python src/rag_v2/app.py
# → http://localhost:7860

# Or run the FastAPI + Next.js stack
PYTHONPATH=src uvicorn rag_v2.api:app --port 8000
cd frontend && npm install && npm run dev
# → http://localhost:3000
```

---

## Evaluation Results

A 70-question stakeholder-curated dataset was used. Metrics include exact-match precision, recall, and ROUGE-L. Full results in `data/eval_70/`.

---

## Corpus

The knowledge base covers the Sustainable Solutions Lab at UMass Boston:
institutional reports, climate resilience PDFs, SSL website content, and
academic papers. All source documents are acknowledged in `data/final_corpus_bundle/REPORT/`.
