"""
Interactive RAG V2 demo — CLI multi-turn dialogue.

Usage (from repository root):
    set PYTHONPATH=src
    python -m rag_v2.demo
    python -m rag_v2.demo --question "What is SSL's mission?"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_v1.pipeline import (
    EMBED_MODEL_NAME,
    RERANK_MODEL_NAME,
    load_all,
    openai_client,
)
from rag_v2.pipeline import ask
from rag_v2.session import SessionMemory

sys.stdout.reconfigure(encoding="utf-8")


def _print_block(title: str, body: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(body.rstrip())


def _print_result(result: dict) -> None:
    _print_block("QUESTION", result["question"])

    if result["resolved_query"] != result["question"]:
        _print_block("RESOLVED QUERY", result["resolved_query"])

    _print_block("INTENT", result["intent"])

    log = result.get("retrieval_log", {})
    _print_block("RETRIEVAL LOG", json.dumps(log, indent=2, ensure_ascii=False))

    lines = []
    for c in result.get("retrieved", []):
        src = c.get("source", "")
        sec = c.get("section_title", "")
        st = c.get("source_type", "")
        lyr = c.get("layer", "")
        sc = c.get("score", 0)
        txt = (c.get("chunk_text") or "")[:320].replace("\n", " ")
        lines.append(
            f"  [{st} | {lyr}] score={sc:.4f}\n"
            f"    source={src}\n"
            f"    section={sec}\n"
            f"    text: {txt}..."
        )
    _print_block("TOP EVIDENCE (truncated)", "\n".join(lines))

    _print_block("MODEL ANSWER", result["answer"])

    cons = result.get("consistency")
    if cons:
        status = "CONSISTENT" if cons["is_consistent"] else "ISSUES DETECTED"
        _print_block(
            "CONSISTENCY CHECK",
            f"Status: {status} (confidence: {cons['confidence']:.0%})\n"
            + (f"Unsupported: {cons['unsupported_claims']}\n" if cons.get("unsupported_claims") else "")
            + (f"Note: {cons['explanation']}" if cons.get("explanation") else ""),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG V2 interactive demo.")
    parser.add_argument("-q", "--question", type=str, default=None, help="Single question.")
    parser.add_argument("--interactive", action="store_true", help="Multi-turn interactive mode.")
    parser.add_argument("--examples", action="store_true", help="Run preset examples.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    _bu = os.environ.get("OPENAI_BASE_URL")
    if _bu is not None and not str(_bu).strip():
        os.environ.pop("OPENAI_BASE_URL", None)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("[ERROR] OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1

    print("Loading models and corpus...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    client = openai_client()
    meta, ctx_idx, qa_items, qa_idx, bm25, _ = load_all(embed_model)

    session = SessionMemory(max_turns=10)
    common = dict(
        session=session, client=client, embed_model=embed_model,
        corpus_idx=ctx_idx, corpus_meta=meta, bm25=bm25,
        qa_items=qa_items, qa_idx=qa_idx, reranker=reranker,
    )

    if args.examples:
        examples = [
            "What is the Sustainable Solutions Lab?",
            "What projects does it lead?",
            "Has SSL published research on nuclear energy?",
        ]
        for q in examples:
            result = ask(q, **common)
            _print_result(result)
        return 0

    if args.question:
        result = ask(args.question.strip(), **common)
        _print_result(result)
        return 0

    if args.interactive or (not args.question and not args.examples):
        print("\n" + "=" * 70)
        print("  SSL RAG V2 — Interactive multi-turn demo")
        print("  Type your question, or 'quit' to exit, 'clear' to reset session.")
        print("=" * 70)

        while True:
            try:
                q = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if q.lower() == "clear":
                session.clear()
                print("[Session cleared]")
                continue

            result = ask(q, **common)
            _print_result(result)
            print(f"\n[Session: {len(session)} turns]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
