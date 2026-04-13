"""
Multi-turn dialogue evaluation with LLM-as-Judge scoring.

Runs 5 conversation chains (15 questions total) with shared session memory,
then evaluates each answer against gold standards and produces detailed metrics.

Usage:
    set PYTHONPATH=src
    python -u test_multiturn_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONUNBUFFERED"] = "1"

from sentence_transformers import SentenceTransformer, CrossEncoder

from rag_v1.pipeline import (
    load_all,
    openai_client,
    eval_answer,
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
)
from rag_v2.pipeline import ask
from rag_v2.session import SessionMemory

EVAL_PATH = Path("data/eval_phase2/stakeholder_eval_phase2.json")
OUT_DIR = Path("results/v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_multi_chains(path: Path) -> dict[str, list[dict]]:
    """Load MULTI-* questions grouped by chain letter."""
    with open(path, "r", encoding="utf-8") as f:
        all_qs = json.load(f)

    multi_qs = [q for q in all_qs if q["question_id"].startswith("MULTI-")]
    chains: dict[str, list[dict]] = {}
    for q in multi_qs:
        chain = q.get("multi_turn_chain", "?")
        chains.setdefault(chain, []).append(q)

    for chain in chains:
        chains[chain].sort(key=lambda q: q["question_id"])

    return chains


def main():
    print("=" * 70)
    print("  MULTI-TURN DIALOGUE EVALUATION (LLM-as-Judge)")
    print("=" * 70)

    print("\n[1/4] Loading models and corpus...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    client = openai_client()
    meta, ctx_idx, qa_items, qa_idx, bm25, _ = load_all(embed_model)

    common = dict(
        client=client,
        embed_model=embed_model,
        corpus_idx=ctx_idx,
        corpus_meta=meta,
        bm25=bm25,
        qa_items=qa_items,
        qa_idx=qa_idx,
        reranker=reranker,
    )

    print("\n[2/4] Loading multi-turn chains...")
    chains = load_multi_chains(EVAL_PATH)
    total_questions = sum(len(v) for v in chains.values())
    print(f"  Chains: {len(chains)}  |  Total questions: {total_questions}")
    for chain_id in sorted(chains):
        ids = [q["question_id"] for q in chains[chain_id]]
        print(f"    Chain {chain_id}: {' → '.join(ids)}")

    print(f"\n[3/4] Running evaluation across {len(chains)} chains...\n")
    all_results = []
    t_start = time.time()

    for chain_id in sorted(chains):
        chain_qs = chains[chain_id]
        session = SessionMemory(max_turns=10)
        print(f"--- Chain {chain_id} ({len(chain_qs)} turns) ---")

        for turn_idx, q in enumerate(chain_qs):
            qid = q["question_id"]
            question = q["question"]
            gold = q["gold_answer"]
            expected = q["expected_behavior"]

            print(f"  [{qid}] \"{question}\"")

            t0 = time.time()
            result = ask(question, session=session, **common)
            gen_time = time.time() - t0

            answer = result["answer"]
            resolved = result.get("resolved_query", question)
            intent = result.get("intent", "?")
            coref_changed = resolved != question

            if coref_changed:
                print(f"    >> Resolved: \"{resolved}\"")
            print(f"    Intent: {intent}  |  Gen time: {gen_time:.1f}s")

            t1 = time.time()
            eval_scores = eval_answer(answer, gold, expected, client)
            eval_time = time.time() - t1

            acc = eval_scores.get("factual_accuracy", -1)
            comp = eval_scores.get("completeness", -1)
            behav = eval_scores.get("appropriate_behavior", -1)
            hall = eval_scores.get("hallucination", -1)
            print(f"    Scores: acc={acc} comp={comp} behav={behav} hall={hall}  (eval {eval_time:.1f}s)")

            record = {
                "question_id": qid,
                "chain": chain_id,
                "turn": turn_idx + 1,
                "question": question,
                "resolved_query": resolved,
                "coreference_resolved": coref_changed,
                "intent": intent,
                "expected_behavior": expected,
                "gold_answer": gold,
                "system_answer": answer,
                "answer_eval": eval_scores,
                "generation_time_s": round(gen_time, 2),
                "eval_time_s": round(eval_time, 2),
            }
            all_results.append(record)

        print()

    total_time = time.time() - t_start
    print(f"[4/4] Computing metrics... (total time: {total_time:.1f}s)\n")

    # --- Compute detailed metrics ---
    metrics = compute_multiturn_metrics(all_results)
    metrics["total_time_s"] = round(total_time, 1)
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

    results_path = OUT_DIR / "multiturn_eval_results.json"
    metrics_path = OUT_DIR / "multiturn_eval_metrics.json"
    report_path = OUT_DIR / "MULTITURN_EVAL_REPORT.md"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    write_report(report_path, all_results, metrics)

    print_summary(all_results, metrics)
    print(f"\nResults saved to: {results_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Report saved to:  {report_path}")


def compute_multiturn_metrics(results: list[dict]) -> dict:
    """Compute comprehensive metrics for multi-turn evaluation."""
    N = len(results)
    valid = [r for r in results if r["answer_eval"].get("factual_accuracy", -1) >= 0]
    V = len(valid)

    def avg(key):
        vals = [r["answer_eval"][key] for r in valid if r["answer_eval"].get(key, -1) >= 0]
        return round(sum(vals) / len(vals), 3) if vals else -1

    def score_dist(key):
        vals = [r["answer_eval"].get(key, -1) for r in valid if r["answer_eval"].get(key, -1) >= 0]
        return {str(i): vals.count(i) for i in range(3)}

    overall = {
        "total_questions": N,
        "valid_evals": V,
        "factual_accuracy_avg": avg("factual_accuracy"),
        "completeness_avg": avg("completeness"),
        "appropriate_behavior_avg": avg("appropriate_behavior"),
        "hallucination_avg": avg("hallucination"),
        "factual_accuracy_dist": score_dist("factual_accuracy"),
        "hallucination_dist": score_dist("hallucination"),
    }

    # --- Per-turn-position metrics (Turn 1 vs Turn 2 vs Turn 3) ---
    by_turn = {}
    for turn_num in sorted(set(r["turn"] for r in results)):
        turn_results = [r for r in valid if r["turn"] == turn_num]
        n = len(turn_results)
        if n == 0:
            continue
        by_turn[f"turn_{turn_num}"] = {
            "count": n,
            "factual_accuracy_avg": round(
                sum(r["answer_eval"].get("factual_accuracy", 0) for r in turn_results) / n, 3
            ),
            "completeness_avg": round(
                sum(r["answer_eval"].get("completeness", 0) for r in turn_results) / n, 3
            ),
            "hallucination_avg": round(
                sum(r["answer_eval"].get("hallucination", 0) for r in turn_results) / n, 3
            ),
        }

    # --- Per-chain metrics ---
    by_chain = {}
    for chain_id in sorted(set(r["chain"] for r in results)):
        chain_results = [r for r in valid if r["chain"] == chain_id]
        n = len(chain_results)
        if n == 0:
            continue
        by_chain[f"chain_{chain_id}"] = {
            "count": n,
            "questions": [r["question_id"] for r in chain_results],
            "factual_accuracy_avg": round(
                sum(r["answer_eval"].get("factual_accuracy", 0) for r in chain_results) / n, 3
            ),
            "completeness_avg": round(
                sum(r["answer_eval"].get("completeness", 0) for r in chain_results) / n, 3
            ),
            "hallucination_avg": round(
                sum(r["answer_eval"].get("hallucination", 0) for r in chain_results) / n, 3
            ),
        }

    # --- Coreference resolution stats ---
    coref_questions = [r for r in results if r["turn"] > 1]
    coref_resolved = [r for r in coref_questions if r["coreference_resolved"]]
    coref_stats = {
        "follow_up_questions": len(coref_questions),
        "coreference_triggered": len(coref_resolved),
        "coreference_rate": round(len(coref_resolved) / len(coref_questions), 3)
        if coref_questions
        else 0,
    }
    if coref_resolved:
        coref_valid = [r for r in coref_resolved if r["answer_eval"].get("factual_accuracy", -1) >= 0]
        if coref_valid:
            coref_stats["resolved_factual_accuracy_avg"] = round(
                sum(r["answer_eval"].get("factual_accuracy", 0) for r in coref_valid) / len(coref_valid), 3
            )

    # --- Latency ---
    gen_times = [r["generation_time_s"] for r in results]
    latency = {
        "avg_generation_s": round(sum(gen_times) / len(gen_times), 2),
        "max_generation_s": round(max(gen_times), 2),
        "min_generation_s": round(min(gen_times), 2),
    }

    return {
        "overall": overall,
        "by_turn_position": by_turn,
        "by_chain": by_chain,
        "coreference": coref_stats,
        "latency": latency,
    }


def write_report(path: Path, results: list[dict], metrics: dict):
    """Write a markdown evaluation report."""
    m = metrics
    o = m["overall"]

    lines = [
        "# Multi-Turn Dialogue Evaluation Report",
        "",
        f"**Date:** {m.get('timestamp', 'N/A')}  ",
        f"**Total questions:** {o['total_questions']}  |  **Chains:** {len(m['by_chain'])}  |  "
        f"**Total time:** {m.get('total_time_s', '?')}s",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Score (0-2) |",
        "|--------|-------------|",
        f"| Factual Accuracy | {o['factual_accuracy_avg']} |",
        f"| Completeness | {o['completeness_avg']} |",
        f"| Appropriate Behavior | {o['appropriate_behavior_avg']} |",
        f"| Hallucination (2=none) | {o['hallucination_avg']} |",
        "",
        "## Score Distribution (Factual Accuracy)",
        "",
        "| Score | Count |",
        "|-------|-------|",
    ]
    for s in ("0", "1", "2"):
        lines.append(f"| {s} | {o['factual_accuracy_dist'].get(s, 0)} |")

    lines += [
        "",
        "## Performance by Turn Position",
        "",
        "| Turn | Count | Accuracy | Completeness | Hallucination |",
        "|------|-------|----------|--------------|---------------|",
    ]
    for turn_key in sorted(m["by_turn_position"]):
        t = m["by_turn_position"][turn_key]
        lines.append(f"| {turn_key} | {t['count']} | {t['factual_accuracy_avg']} | "
                      f"{t['completeness_avg']} | {t['hallucination_avg']} |")

    lines += [
        "",
        "## Performance by Chain",
        "",
        "| Chain | Questions | Accuracy | Completeness | Hallucination |",
        "|-------|-----------|----------|--------------|---------------|",
    ]
    for chain_key in sorted(m["by_chain"]):
        c = m["by_chain"][chain_key]
        qids = ", ".join(c["questions"])
        lines.append(f"| {chain_key} | {qids} | {c['factual_accuracy_avg']} | "
                      f"{c['completeness_avg']} | {c['hallucination_avg']} |")

    cs = m["coreference"]
    lines += [
        "",
        "## Coreference Resolution",
        "",
        f"- Follow-up questions (Turn 2+): **{cs['follow_up_questions']}**",
        f"- Coreference triggered: **{cs['coreference_triggered']}**",
        f"- Coreference rate: **{cs['coreference_rate']:.0%}**",
    ]
    if "resolved_factual_accuracy_avg" in cs:
        lines.append(f"- Accuracy after resolution: **{cs['resolved_factual_accuracy_avg']}**")

    lat = m["latency"]
    lines += [
        "",
        "## Latency",
        "",
        f"- Avg generation: **{lat['avg_generation_s']}s**",
        f"- Min / Max: **{lat['min_generation_s']}s** / **{lat['max_generation_s']}s**",
        "",
        "## Detailed Results",
        "",
    ]

    current_chain = None
    for r in results:
        if r["chain"] != current_chain:
            current_chain = r["chain"]
            lines.append(f"### Chain {current_chain}")
            lines.append("")

        ev = r["answer_eval"]
        lines.append(f"**{r['question_id']}** (Turn {r['turn']}): \"{r['question']}\"")
        if r["coreference_resolved"]:
            lines.append(f"  - Resolved → \"{r['resolved_query']}\"")
        lines.append(f"  - Accuracy={ev.get('factual_accuracy', '?')} "
                      f"Completeness={ev.get('completeness', '?')} "
                      f"Behavior={ev.get('appropriate_behavior', '?')} "
                      f"Hallucination={ev.get('hallucination', '?')}")
        lines.append(f"  - Answer: {r['system_answer'][:200]}...")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_summary(results: list[dict], metrics: dict):
    o = metrics["overall"]
    cs = metrics["coreference"]

    print("=" * 70)
    print("  MULTI-TURN EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total questions:    {o['total_questions']}")
    print(f"  Valid evals:        {o['valid_evals']}")
    print()
    print(f"  Factual Accuracy:   {o['factual_accuracy_avg']:.3f} / 2.0")
    print(f"  Completeness:       {o['completeness_avg']:.3f} / 2.0")
    print(f"  Appropriate Behav:  {o['appropriate_behavior_avg']:.3f} / 2.0")
    print(f"  Hallucination:      {o['hallucination_avg']:.3f} / 2.0 (2=none)")
    print()
    print("  --- By Turn Position ---")
    for tk in sorted(metrics["by_turn_position"]):
        t = metrics["by_turn_position"][tk]
        print(f"    {tk}: acc={t['factual_accuracy_avg']:.3f}  "
              f"comp={t['completeness_avg']:.3f}  "
              f"hall={t['hallucination_avg']:.3f}")
    print()
    print(f"  --- Coreference ---")
    print(f"    Follow-up questions:  {cs['follow_up_questions']}")
    print(f"    Coref triggered:      {cs['coreference_triggered']}")
    print(f"    Coref rate:           {cs['coreference_rate']:.0%}")
    if "resolved_factual_accuracy_avg" in cs:
        print(f"    Resolved accuracy:    {cs['resolved_factual_accuracy_avg']:.3f}")
    print()

    print("  --- Per Question Scorecard ---")
    for r in results:
        ev = r["answer_eval"]
        acc = ev.get("factual_accuracy", -1)
        marker = "✓" if acc == 2 else ("△" if acc == 1 else "✗")
        coref = " [coref]" if r["coreference_resolved"] else ""
        print(f"    {marker} {r['question_id']} (T{r['turn']}){coref}: "
              f"acc={acc} comp={ev.get('completeness', '?')} "
              f"hall={ev.get('hallucination', '?')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
