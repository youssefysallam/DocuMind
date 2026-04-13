"""
Interactive RAG V1 demo for classroom / interview (single question, no benchmark eval).

Usage (from repository root):
    set PYTHONPATH=src
    python -m rag_v1.demo --question "What is SSL's mission?"
    python -m rag_v1.demo --examples

Requires: .env with OPENAI_API_KEY, corpus under data/final_corpus_bundle (or symlink),
          data/rag_v1/qa_memory/qa_memory.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

# Reuse production pipeline (paths, retrieval, generation)
from rag_v1.pipeline import (
    classify_intent,
    generate_answer,
    load_all,
    openai_client,
    retrieve,
)

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_EXAMPLES = [
    "What is the Sustainable Solutions Lab?",
    "What is the Climate Careers Curricula Initiative (C3I)?",
    "Has SSL published research on nuclear energy as a climate solution?",
]


def _print_block(title: str, body: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(body.rstrip())


def run_one(
    question: str,
    *,
    embed_model,
    reranker,
    client: OpenAI,
    meta,
    ctx_idx,
    qa_items,
    qa_idx,
    bm25,
) -> None:
    intent = classify_intent(question)
    retrieved, ret_log = retrieve(
        question,
        intent,
        embed_model,
        ctx_idx,
        meta,
        bm25,
        qa_items,
        qa_idx,
        reranker,
    )
    answer = generate_answer(question, retrieved, client)

    _print_block("QUESTION", question)
    _print_block("CLASSIFIED INTENT", intent)
    _print_block("RETRIEVAL LOG", json.dumps(ret_log, indent=2, ensure_ascii=False))

    lines = []
    for c in retrieved:
        src = c.get("source", "")
        sec = c.get("section_title", "")
        st = c.get("source_type", "")
        lyr = c.get("layer", "")
        sc = c.get("score", 0)
        txt = (c.get("chunk_text") or "")[:320].replace("\n", " ")
        lines.append(f"  [{st} | {lyr}] score={sc}\n    source={src}\n    section={sec}\n    text: {txt}...")
    _print_block("TOP-5 EVIDENCE (truncated)", "\n".join(lines))

    _print_block("MODEL ANSWER", answer)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG V1 single-query demo (no eval judge).")
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default=None,
        help="User question in natural language.",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run three preset questions (overview, project, no-evidence).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")
    # Same as pipeline: blank base URL must not override the SDK default.
    _bu = os.environ.get("OPENAI_BASE_URL")
    if _bu is not None and not str(_bu).strip():
        os.environ.pop("OPENAI_BASE_URL", None)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()

    if not api_key:
        print(
            "[ERROR] OPENAI_API_KEY is not set.\n"
            "        Copy .env.example to .env in the repo root and add your key, e.g.:\n"
            "          OPENAI_API_KEY=sk-...\n"
            "        Or set the variable in your shell before running.",
            file=sys.stderr,
        )
        return 1

    if api_key.startswith("sk-or-v1") and not base_url:
        print(
            "[ERROR] Your key looks like OpenRouter (sk-or-v1-...), but OPENAI_BASE_URL is not set.\n"
            "        Add to .env (same folder as this project):\n"
            "          OPENAI_BASE_URL=https://openrouter.ai/api/v1\n"
            "          OPENAI_API_KEY=sk-or-v1-...\n"
            "          LLM_MODEL=openai/gpt-4o-mini\n",
            file=sys.stderr,
        )
        return 1

    bundle = root / "data" / "final_corpus_bundle" / "merged" / "unified_index_metadata.json"
    if not bundle.exists():
        print(
            "[ERROR] Corpus not found at data/final_corpus_bundle/merged/.\n"
            "        If your bundle lives under processed/final_corpus_bundle/, copy or symlink it to data/.",
            file=sys.stderr,
        )
        return 1

    questions: list[str] = []
    if args.examples:
        questions = list(DEFAULT_EXAMPLES)
    elif args.question:
        questions = [args.question.strip()]
    else:
        parser.print_help()
        print("\nProvide --question \"...\" or --examples.", file=sys.stderr)
        return 2

    print("Loading embedding model, reranker, and corpus (first run may build contextualized FAISS)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    client = openai_client()
    meta, ctx_idx, qa_items, qa_idx, bm25, _dataset = load_all(embed_model)

    for q in questions:
        run_one(
            q,
            embed_model=embed_model,
            reranker=reranker,
            client=client,
            meta=meta,
            ctx_idx=ctx_idx,
            qa_items=qa_items,
            qa_idx=qa_idx,
            bm25=bm25,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
