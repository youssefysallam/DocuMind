"""
FastAPI backend for InfoWeave UI V2.

Exposes the RAG V2 pipeline over HTTP for the Next.js frontend.
The existing Gradio UI (app.py) is NOT modified — both can run independently.

Usage (from repo root, with PYTHONPATH=src):
    set PYTHONPATH=src
    uvicorn rag_v2.api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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

# ── App ──────────────────────────────────────────────────────────

app = FastAPI(title="InfoWeave API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy system loading (same pattern as app.py) ──────────────────

_SYSTEM: dict | None = None
_SESSIONS: dict[str, "SessionMemory"] = {}


def _load_system():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from rag_v1.pipeline import load_all, openai_client, EMBED_MODEL_NAME, RERANK_MODEL_NAME

    print("[API] Loading models and corpus ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    client = openai_client()
    meta, ctx_idx, qa_items, qa_idx, bm25, dataset = load_all(embed_model)
    print("[API] System ready.")
    return dict(
        embed_model=embed_model, reranker=reranker, client=client,
        meta=meta, ctx_idx=ctx_idx, qa_items=qa_items, qa_idx=qa_idx,
        bm25=bm25, dataset=dataset,
    )


def get_system():
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = _load_system()
    return _SYSTEM


def _get_session(session_id: str):
    from rag_v2.session import SessionMemory
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = SessionMemory(max_turns=10)
    return _SESSIONS[session_id]


# ── Request/response models ───────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str


class ClearRequest(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────

@app.post("/api/chat")
def chat(req: ChatRequest):
    from rag_v2.pipeline import ask

    if not req.message.strip():
        return {"answer": "", "retrieved": [], "retrieval_log": {}, "consistency": None}

    sys_data = get_system()
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

    return {
        "answer": result["answer"],
        "retrieved": result.get("retrieved", []),
        "retrieval_log": result.get("retrieval_log", {}),
        "consistency": result.get("consistency"),
    }


@app.post("/api/session/clear")
def clear_session(req: ClearRequest):
    if req.session_id in _SESSIONS:
        _SESSIONS[req.session_id].clear()
    return {"status": "ok"}


@app.get("/api/eval")
def get_eval():
    comparison_path = PROJECT_ROOT / "results" / "v2" / "v1_vs_v2_metrics.json"
    mt_path = PROJECT_ROOT / "results" / "v2" / "multiturn_eval_metrics.json"

    data = {}
    if comparison_path.exists():
        with open(comparison_path, "r", encoding="utf-8") as f:
            data["comparison"] = json.load(f)
    if mt_path.exists():
        with open(mt_path, "r", encoding="utf-8") as f:
            data["multiturn"] = json.load(f)
    return data


@app.get("/api/system")
def get_system_info():
    from rag_v1.pipeline import EMBED_MODEL_NAME, RERANK_MODEL_NAME

    llm_model = os.getenv("LLM_MODEL_V2") or os.getenv("LLM_MODEL", "openai/gpt-5.4")
    return {
        "version": "RAG V2 — Phase 2",
        "llm_model": llm_model,
        "embed_model": EMBED_MODEL_NAME,
        "sparse_model": "BM25",
        "reranker": RERANK_MODEL_NAME,
        "top_k": {"dense": 20, "sparse": 20, "final": 5},
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


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("  InfoWeave API — FastAPI Backend")
    print("  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
