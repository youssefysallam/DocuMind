"""
Run RAG V1 evaluation on the Phase 2 (95-question) dataset for apples-to-apples comparison with V2.

Usage (repo root):
    set PYTHONPATH=src
    python scripts/eval_v1_phase2.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_v1.pipeline import (
    compute_metrics,
    eval_answer,
    eval_retrieval,
    generate_answer,
    load_all,
    openai_client,
    retrieve,
    classify_intent,
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
)

DATASET = PROJECT_ROOT / "data" / "eval_phase2" / "stakeholder_eval_phase2.json"
OUT_DIR = PROJECT_ROOT / "results" / "v2"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"[V1 Phase2] {len(dataset)} questions from {DATASET.name}")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    client = openai_client()
    meta, ctx_idx, qa_items, qa_idx, bm25, _ = load_all(embed_model)

    results = []
    for i, q in enumerate(dataset):
        qid = q["question_id"]
        print(f"  [{i+1}/{len(dataset)}] {qid}: {q['question'][:50]}...")
        intent = classify_intent(q["question"])
        retrieved, ret_log = retrieve(
            q["question"], intent, embed_model, ctx_idx, meta,
            bm25, qa_items, qa_idx, reranker,
        )
        sys_ans = generate_answer(q["question"], retrieved, client)
        reval = eval_retrieval(retrieved, q.get("gold_source_ids", []))
        aeval = eval_answer(sys_ans, q["gold_answer"], q["expected_behavior"], client)
        results.append({
            "question_id": qid,
            "question": q["question"],
            "question_type": q["question_type"],
            "expected_behavior": q["expected_behavior"],
            "gold_answer": q["gold_answer"],
            "system_answer": sys_ans,
            "retrieval_eval": reval,
            "answer_eval": aeval,
            "retrieval_log": ret_log,
        })
        time.sleep(0.3)

    metrics = compute_metrics(results)
    with open(OUT_DIR / "rag_v1_phase2_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "rag_v1_phase2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[V1 Phase2] Saved metrics to {OUT_DIR / 'rag_v1_phase2_metrics.json'}")


if __name__ == "__main__":
    main()
