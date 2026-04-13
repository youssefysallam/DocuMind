# RAG V1 — evaluation summary (frozen snapshot)

This file summarizes the last retained evaluation run before repository cleanup.

- **Full narrative + tables (v7-equivalent):** `../../report/RAG_V1_FULL_EXPERIMENT_REPORT.md`
- **Metrics JSON:** `rag_v1_metrics.json`
- **Per-question traces:** `rag_v1_eval_results.json`
- **Baseline pointer:** `baseline_vs_rag_v1_metrics.json` (see also `baseline_v4_metrics.json`)

## Headline metrics (70 questions)

| Metric | Value |
|--------|------:|
| Answer accuracy (answerable) | 79.1% |
| Answer accuracy (all) | 71.4% |
| Completeness | 66.4% |
| Coverage | 92.9% |
| Raw corpus hit@5 | 18.2% |
| Hallucination rate | 24.3% |
| Correct refusal rate | 80.0% |
| False refusal rate | 1.8% |
| Missed refusals | 3 |

Re-run with `python -m rag_v1.pipeline` to regenerate this folder (requires API keys).
