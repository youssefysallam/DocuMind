"""
RAG V2 FastAPI server — backend for the Next.js frontend.

Endpoints match the frontend's api.ts contract:
    POST /api/chat          → ChatResponse
    POST /api/session/clear → 204
    GET  /api/eval          → EvalData
    GET  /api/system        → SystemInfo

Usage (from repo root):
    set PYTHONPATH=src
    python -m rag_v2.server
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

sys.stdout.reconfigure(encoding="utf-8")

_bu = os.environ.get("OPENAI_BASE_URL")
if _bu is not None and not str(_bu).strip():
    os.environ.pop("OPENAI_BASE_URL", None)

TOP_K_DENSE = 20
TOP_K_SPARSE = 20
TOP_K_FINAL = 5
LLM_DISPLAY = os.getenv("LLM_MODEL_V2") or os.getenv("LLM_MODEL", "openai/gpt-5.4")

# ---------------------------------------------------------------------------
# System singleton (loaded once at startup)
# ---------------------------------------------------------------------------
_SYSTEM: dict | None = None
_SESSIONS: dict = {}


def _load_system() -> dict:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from rag_v1.pipeline import (
        EMBED_MODEL_NAME,
        RERANK_MODEL_NAME,
        load_all,
        openai_client,
    )

    print("[Server] Loading models and corpus ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    client = openai_client()
    meta, ctx_idx, qa_items, qa_idx, bm25, dataset = load_all(embed_model)
    print("[Server] System ready.")
    return dict(
        embed_model=embed_model,
        reranker=reranker,
        client=client,
        meta=meta,
        ctx_idx=ctx_idx,
        qa_items=qa_items,
        qa_idx=qa_idx,
        bm25=bm25,
        dataset=dataset,
    )


def _get_system() -> dict:
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = _load_system()
    return _SYSTEM


def _get_session(session_id: str):
    from rag_v2.session import SessionMemory

    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionMemory(max_turns=10)
    return _SESSIONS[session_id]


# ---------------------------------------------------------------------------
# Pydantic models (match frontend lib/types.ts)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str


class ClearRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="InfoWeave RAG V2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat")
def chat(req: ChatRequest):
    from rag_v2.pipeline import ask

    sys_data = _get_system()
    session = _get_session(req.session_id)

    result = ask(
        req.message,
        session=session,
        client=sys_data["client"],
        embed_model=sys_data["embed_model"],
        corpus_idx=sys_data["ctx_idx"],
        corpus_meta=sys_data["meta"],
        bm25=sys_data["bm25"],
        qa_items=sys_data["qa_items"],
        qa_idx=sys_data["qa_idx"],
        reranker=sys_data["reranker"],
    )

    retrieved = []
    for chunk in result.get("retrieved", []):
        retrieved.append(
            {
                "source": chunk.get("source", ""),
                "section_title": chunk.get("section_title", ""),
                "source_type": chunk.get("source_type", ""),
                "score": round(chunk.get("score", 0), 4),
                "layer": chunk.get("layer", ""),
                "chunk_text": (chunk.get("chunk_text") or "")[:300],
            }
        )

    log = result.get("retrieval_log", {})
    retrieval_log = {
        "intent": log.get("intent"),
        "strategy": log.get("strategy"),
        "resolved_query": log.get("resolved_query"),
        "original_query": log.get("original_query"),
        "sub_queries": log.get("sub_queries"),
        "hyde_used": log.get("hyde_used", False),
        "final_raw": log.get("final_raw", 0),
        "final_qa": log.get("final_qa", 0),
        "qa_neg_used": log.get("qa_neg_used", False),
        "multihop": log.get("multihop"),
    }

    consistency = None
    if result.get("consistency"):
        c = result["consistency"]
        consistency = {
            "is_consistent": c.get("is_consistent", True),
            "confidence": c.get("confidence", 0),
            "unsupported_claims": c.get("unsupported_claims"),
            "explanation": c.get("explanation"),
        }

    return {
        "answer": result["answer"],
        "retrieved": retrieved,
        "retrieval_log": retrieval_log,
        "consistency": consistency,
    }


@app.post("/api/session/clear", status_code=204)
def clear_session(req: ClearRequest):
    if req.session_id in _SESSIONS:
        _SESSIONS[req.session_id].clear()


@app.get("/api/eval")
def get_eval():
    comparison_path = PROJECT_ROOT / "results" / "v2" / "v1_vs_v2_metrics.json"
    mt_path = PROJECT_ROOT / "results" / "v2" / "multiturn_eval_metrics.json"

    result: dict = {}

    if comparison_path.exists():
        with open(comparison_path, encoding="utf-8") as f:
            data = json.load(f)
        v1 = data.get("rag_v1_same_dataset_95") or data.get("rag_v1", {})
        v2 = data.get("rag_v2", {})
        result["comparison"] = {
            "rag_v1_same_dataset_95": v1,
            "rag_v2": v2,
        }

    if mt_path.exists():
        with open(mt_path, encoding="utf-8") as f:
            mt = json.load(f)
        result["multiturn"] = {
            "overall": mt.get("overall"),
            "coreference": mt.get("coreference"),
            "by_turn_position": mt.get("by_turn_position"),
        }

    return result


@app.get("/api/system")
def get_system_info():
    return {
        "version": "RAG V2 - Phase 2",
        "llm_model": LLM_DISPLAY,
        "embed_model": "all-MiniLM-L6-v2",
        "sparse_model": "BM25",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": {"dense": TOP_K_DENSE, "sparse": TOP_K_SPARSE, "final": TOP_K_FINAL},
        "features": [
            "HyDE",
            "Intent Classification",
            "Sub-query Decomposition",
            "Multi-hop Retrieval",
            "Diversity Selection",
            "Consistency Check",
            "Session Memory",
            "Coreference Resolution",
            "Multi-turn Dialogue",
        ],
    }


def main():
    print("\n" + "=" * 60)
    print("  InfoWeave RAG V2 — FastAPI Server")
    print("  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
