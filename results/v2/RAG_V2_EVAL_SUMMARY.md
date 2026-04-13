# RAG V2 — Evaluation summary (automated run)

Generated: 2026-04-11 03:09 UTC

## 1. System enhancements over V1

**RAG V2** adds the following on top of V1's contextualized hybrid retrieval:
- **LLM-enhanced intent classification** (keyword fast-path + LLM fallback for no_evidence)
- **HyDE** (Hypothetical Document Embeddings) for improved dense retrieval
- **Sub-query decomposition** for synthesis questions
- **Diversity-aware evidence selection** (max 2 chunks per source)
- **Answer-evidence consistency checking** (post-generation verification)
- **Session memory** and **multi-turn dialogue** support (coreference resolution)
- **Gradio Web UI** with source attribution and retrieval transparency

## 2. Same-dataset comparison (104 questions: V1 vs V2)

| Metric | V1 (Phase2 set) | V2 | Δ |
|--------|:--:|:--:|:---:|
| raw_hit@3 | 15.8% | 15.8% | 0.0% |
| raw_hit@5 | 17.1% | 17.1% | 0.0% |
| Accuracy (answerable) | 72.0% | 75.0% | +3.0% |
| Accuracy (all) | 66.3% | 69.2% | +2.9% |
| Completeness | 62.0% | 62.5% | +0.5% |
| Hallucination | 24.0% | 12.0% | -12.0% |
| Coverage | 90.4% | 96.2% | +5.8% |
| Correct refusal | 86.4% | 90.9% | +4.5% |
| False refusal | 8.5% | 2.4% | -6.1% |
| Evidence-supported | 14.6% | 15.8% | +1.2% |
| Missed refusals | 3 | 2 | |
| Consistency | — | 39.0% | (new) |

## 3. Performance by Question Type

| Type | Total | Behavior | Accuracy | Perfect |
|------|:---:|:---:|:---:|:---:|
| general_overview | 20 | 100% | 82% | 13/20 |
| project_initiative | 17 | 94% | 79% | 11/17 |
| topic_specific | 15 | 100% | 77% | 8/15 |
| publication_finding | 12 | 100% | 79% | 7/12 |
| synthesis | 18 | 94% | 58% | 4/18 |
| no_evidence | 22 | 91% | 48% | 3/22 |

## 4. Artifacts

```
results/v2/
├── rag_v2_eval_results.json
├── rag_v2_metrics.json
├── v1_vs_v2_metrics.json
├── rag_v2_error_analysis.json
└── RAG_V2_EVAL_SUMMARY.md
```