# RAG V1 Full Experiment Report (formerly development “v7”)

This document is the **complete** experiment write-up for the configuration now shipped as **RAG V1** in this repository. It is the same system that was last labeled **v7** during development: **v4 hybrid retrieval + v4 QA memory merge/NEG gating**, plus **deterministic contextual prefixes** on every corpus chunk, **FAISS built on `contextualized_text`**, **BM25 on raw `chunk_text`**, and the same generation/refusal prompts as v4.

---

## 1. Purpose

Evaluate end-to-end QA quality on a fixed **70-question** stakeholder-style benchmark over the **unified SSL corpus** (PDF + website), with explicit metrics for retrieval, grounding, abstention, and hallucination.

---

## 2. Corpus and indices

| Item | Location |
|------|----------|
| Unified chunk metadata (source of truth) | `data/final_corpus_bundle/merged/unified_index_metadata.json` |
| Unified baseline dense index (pre–RAG-V1) | `data/final_corpus_bundle/merged/unified_index.faiss` (+ `unified_embeddings.npy`) |
| RAG V1 contextualized FAISS | `data/rag_v1/contextualized_index.faiss` (+ `contextualized_embeddings.npy`, generated if missing) |
| Contextualized chunk JSON (optional mirror) | `data/rag_v1/contextualized_chunks.json` |
| Curated QA memory | `data/rag_v1/qa_memory/qa_memory.json` |

**Contextualization rule (deterministic):** each chunk’s `contextualized_text` = short English prefix (report or page title, SSL / UMass Boston attribution, section, optional type hint) + **unchanged** original `chunk_text`.

---

## 3. Evaluation dataset

| Field | Value |
|-------|--------|
| File | `data/eval_70/stakeholder_eval_70.json` |
| Size | 70 questions |
| Answerable (incl. partial) | 55 |
| Refusal expected | 15 |
| Families | `general_overview`, `project_initiative`, `topic_specific`, `publication_finding`, `synthesis`, `no_evidence` |

---

## 4. Models and hyperparameters (RAG V1)

| Component | Setting |
|-----------|---------|
| Dense embedding | `sentence-transformers/all-MiniLM-L6-v2`, `normalize_embeddings=True` |
| Dense index | FAISS `IndexFlatIP` on **contextualized** vectors |
| Sparse | BM25 on **raw** chunk tokens |
| Top-K (dense / sparse) | 20 / 20 |
| Final context size | 5 |
| QA in final | up to 2 (1 for `publication_finding` / `topic_specific`) |
| Min raw in final | 3 |
| QA relevance floor | 0.35 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM (generate + judge) | `gpt-4o-mini` (env: `LLM_MODEL`) |

**QA–NEG policy:** inject no-evidence QA entries only when intent is `no_evidence` **and** top raw rerank score &lt; 0.3 (same as v4).

---

## 5. Metrics (definitions)

- **raw_corpus_hit@k:** among answerable items, gold `gold_source_ids` matched in **non–QA** retrieved sources in top-*k* (substring match on filenames).
- **answer_accuracy_***: LLM judge factual score 0–2, normalized to 0–1 in aggregates.
- **coverage:** fraction of items with `appropriate_behavior` ≥ 1.
- **correct_refusal_rate:** correct refusals / 15 no-evidence items.
- **false_refusal_rate:** refusals on answerable items / 55.
- **evidence_supported_accuracy:** answerable items with raw hit@5 **and** factual score ≥ 1, divided by answerable count.
- **hallucination_rate:** reported as `1 − mean(judge hallucination score)/2` in the metric JSON convention used by the pipeline.

---

## 6. Overall results — Baseline (v4) vs RAG V1 (frozen run)

Values below are taken from `results/final/baseline_v4_metrics.json` and `results/final/rag_v1_metrics.json`.

| Metric | Baseline (v4) | RAG V1 (ex-v7) | Δ |
|--------|---------------|----------------|---|
| raw_hit@3 | 14.6% | 16.4% | +1.8 pp |
| raw_hit@5 | 16.4% | 18.2% | +1.8 pp |
| Accuracy (answerable) | 72.7% | **79.1%** | **+6.4 pp** |
| Accuracy (all) | 67.9% | **71.4%** | **+3.6 pp** |
| Completeness | 63.6% | **66.4%** | +2.9 pp |
| Hallucination rate | 24.3% | 24.3% | 0 |
| Coverage | **95.7%** | 92.9% | −2.8 pp |
| Refusal rate | 25.7% | 18.6% | −7.1 pp |
| Correct refusal | **100%** | 80.0% | −20.0 pp |
| False refusal | 5.5% | **1.8%** | **−3.6 pp** |
| Missed refusals | **0** | **3** | +3 |
| Evidence-supported | 16.4% | 18.2% | +1.8 pp |

**Interpretation:** RAG V1 substantially improves **answer accuracy** and **completeness** on answerable questions and reduces **false refusals**, consistent with contextual prefixes helping the model use evidence. **No-evidence** behavior regresses versus the v4 snapshot (3 missed refusals, lower correct refusal rate), mainly where intent classification does not trigger `no_evidence` and NEG QA is not injected—see prior error analysis in `results/final/rag_v1_error_analysis.json`.

---

## 7. Results by question type

### Baseline (v4)

| Type | N | Behavior rate | Avg accuracy |
|------|---:|---------------:|-------------:|
| general_overview | 15 | 93.3% | 80.0% |
| project_initiative | 12 | 100% | 83.3% |
| topic_specific | 15 | 86.7% | 70.0% |
| publication_finding | 5 | 100% | 70.0% |
| synthesis | 8 | 100% | 50.0% |
| no_evidence | 15 | **100%** | 50.0% |

### RAG V1 (ex-v7)

| Type | N | Behavior rate | Avg accuracy | Perfect (score 2) |
|------|---:|---------------:|-------------:|------------------:|
| general_overview | 15 | **100%** | **90.0%** | 12 |
| project_initiative | 12 | 100% | 83.3% | 8 |
| topic_specific | 15 | **100%** | **90.0%** | 12 |
| publication_finding | 5 | 80.0% | 60.0% | 2 |
| synthesis | 8 | 100% | 43.8% | 0 |
| no_evidence | 15 | 73.3% | 43.3% | 1 |

---

## 8. Artifacts (reproducibility)

| Artifact | Path |
|----------|------|
| Per-question outputs | `results/final/rag_v1_eval_results.json` |
| Aggregated metrics | `results/final/rag_v1_metrics.json` |
| Baseline metrics | `results/final/baseline_v4_metrics.json` |
| Items with behavior &lt; 2 | `results/final/rag_v1_error_analysis.json` |
| Short headline summary | `results/final/RAG_V1_EVAL_SUMMARY.md` |
| Pipeline entrypoint | `src/rag_v1/pipeline.py` |

Re-run full eval:

```bash
pip install -e .
python -m rag_v1.pipeline
```

---

## 9. Relation to repository naming

- **Development name:** v7 (“v4 + contextualized chunks”, QA logic unchanged from v4).
- **Repository name after cleanup:** **RAG V1** (single canonical version).
- **Academic overview (shorter):** `report/experiment_report.md`.

---

## 10. Conclusion

RAG V1 delivers the **best retained trade-off on answer quality** for overview and topic-specific questions on this benchmark, with modest **raw hit@k** gains. The main **residual risk** is **no-evidence** questions when lexical intent routing fails. Recommended next experiments: strengthen `no_evidence` intent detection, optional second-stage “tangential evidence” check before answering, and multi-query retrieval for **synthesis**.
