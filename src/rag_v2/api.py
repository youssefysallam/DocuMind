"""
FastAPI backend for DocuMind UI V2.

Exposes the RAG V2 pipeline over HTTP for the Next.js frontend.
The existing Gradio UI (app.py) is NOT modified — both can run independently.

Usage (from repo root, with PYTHONPATH=src):
    set PYTHONPATH=src
    uvicorn rag_v2.api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("documind.api")

_bu = os.environ.get("OPENAI_BASE_URL")
if _bu is not None and not str(_bu).strip():
    os.environ.pop("OPENAI_BASE_URL", None)

# ── App ──────────────────────────────────────────────────────────

app = FastAPI(title="DocuMind API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy system loading ──────────────────────────────────────────

_SYSTEM: dict | None = None
_SESSIONS: dict[str, tuple[float, "SessionMemory"]] = {}  # id → (last_access, session)
_SESSION_TTL = 3600  # evict sessions idle for 1 hour
_MAX_SESSIONS = 200


def _load_system():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from rag_v1.pipeline import load_all, openai_client, EMBED_MODEL_NAME, RERANK_MODEL_NAME

    log.info("Loading models and corpus ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    client = openai_client()
    meta, ctx_idx, qa_items, qa_idx, bm25, dataset = load_all(embed_model)
    log.info("System ready.")
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


def _evict_sessions() -> None:
    now = time.monotonic()
    expired = [k for k, (ts, _) in _SESSIONS.items() if now - ts > _SESSION_TTL]
    for k in expired:
        _SESSIONS.pop(k, None)

    if len(_SESSIONS) > _MAX_SESSIONS:
        oldest = sorted(_SESSIONS.items(), key=lambda kv: kv[1][0])
        for k, _ in oldest[: len(_SESSIONS) - _MAX_SESSIONS]:
            _SESSIONS.pop(k, None)


def _get_session(session_id: str):
    from rag_v2.session import SessionMemory
    _evict_sessions()
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = (time.monotonic(), SessionMemory(max_turns=10))
    else:
        _, mem = _SESSIONS[session_id]
        _SESSIONS[session_id] = (time.monotonic(), mem)
    return _SESSIONS[session_id][1]


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

    try:
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
    except Exception as e:
        log.exception("chat failed for session %s", req.session_id)
        raise HTTPException(500, f"chat failed: {e}") from e

    return {
        "answer": result["answer"],
        "retrieved": result.get("retrieved", []),
        "retrieval_log": result.get("retrieval_log", {}),
        "consistency": result.get("consistency"),
    }


@app.post("/api/session/clear")
def clear_session(req: ClearRequest):
    if req.session_id in _SESSIONS:
        _SESSIONS[req.session_id][1].clear()
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
    log.info("DocuMind API — http://localhost:8000 | Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
