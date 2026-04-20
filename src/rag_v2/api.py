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
import re
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
# Serialise the cold-load path — multiple parallel requests should trigger one load, not N.
_SYSTEM_LOCK = threading.Lock()
_FEATURED_LOCK = threading.Lock()


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
    # Double-checked locking — fast path skips the lock once the system is loaded.
    if _SYSTEM is None:
        with _SYSTEM_LOCK:
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


# ── Guided-query tuning constants ─────────────────────────────────

# Discard matches below this cosine similarity — treat as noise, not related questions.
SUGGEST_MIN_SCORE = 0.25

# Looser floor used only when the strict pass returns nothing — keeps fallback suggestions plausible.
SUGGEST_FALLBACK_MIN_SCORE = 0.05

# Cap fallback suggestions — fewer items signals the lower-confidence intent.
SUGGEST_FALLBACK_TOP_K = 3

# Map a UI source filter to the qa_items source_types it should keep.
# Mixed items include both PDFs and websites, so they survive either filter.
SUGGEST_SOURCE_FILTERS: dict[str, set[str]] = {
    "pdf": {"pdf", "mixed"},
    "website": {"website", "mixed"},
}

# SSL-domain alias groups — typing any term in a group counts as a hit on every other term.
# Keep entries lowercased; the lookup is case-insensitive.
SUGGEST_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("ssl", "sustainable solutions lab"),
    ("c3i", "climate careers curricula initiative", "climate careers"),
    ("ncjrc", "northeast climate justice research collaborative", "climate justice collaborative"),
    ("inenas", "institute for new england native american studies"),
    ("mvp", "municipal vulnerability preparedness", "mvp program"),
    ("grc", "green ribbon commission", "boston green ribbon commission"),
    ("cliir", "climate inequality and integrative resilience"),
    ("epa", "environmental protection agency"),
    ("neefc", "new england environmental finance center", "newin"),
    ("nsf", "national science foundation"),
    ("undp", "united nations development programme"),
    ("ohb", "outer harbor barrier"),
    ("ihb", "inner harbor barrier"),
    ("dif", "district improvement financing"),
    ("bid", "business improvement district"),
    ("pace", "property assessed clean energy"),
    ("bala", "balachandran", "b.r. balachandran"),
    ("umb", "umass boston", "university of massachusetts boston"),
)


def _word_boundary_match(needle: str, haystack: str) -> bool:
    """Return True if needle appears in haystack flanked by word boundaries on both sides."""
    # Escape needle so dots and other regex metacharacters in aliases (e.g. "b.r. balachandran") match literally.
    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None


def _expand_alias_terms(keyword_lc: str) -> list[str]:
    """Return the keyword plus any aliased equivalents that share a group with it.

    Word-boundary guarded — `bala` does NOT trigger on `balance`, `dif` does NOT trigger on `different`.
    """
    expansions: set[str] = {keyword_lc}
    for group in SUGGEST_ALIAS_GROUPS:
        for term in group:
            # Trigger when the term sits inside the keyword as a full word/phrase OR vice-versa.
            if _word_boundary_match(term, keyword_lc) or _word_boundary_match(keyword_lc, term):
                expansions.update(group)
                break
    return list(expansions)


class SuggestRequest(BaseModel):
    keyword: str = Field(max_length=200)
    top_k: int = Field(default=5, ge=1, le=20)
    # None / "all" means no filter — only "pdf" and "website" actually narrow the set.
    source_filter: str | None = None


class SuggestionItem(BaseModel):
    qa_id: str
    question: str
    score: float
    source_type: str


class SuggestResponse(BaseModel):
    suggestions: list[SuggestionItem]
    # True when the strict pass found nothing and these are relaxed semantic-only fallbacks.
    is_fallback: bool = False


# ── Featured-suggestions tuning ───────────────────────────────────

# Cap per-intent picks so no single category dominates the dropdown.
FEATURED_PER_GROUP = 3

# Intent display order — drives both group ordering and pill row layout.
FEATURED_INTENT_ORDER = (
    "general_overview",
    "project_initiative",
    "publication_finding",
    "topic_specific",
    "synthesis",
)

# Map machine intent to user-facing pill label.
FEATURED_INTENT_LABELS = {
    "general_overview": "Overview",
    "project_initiative": "Projects",
    "publication_finding": "Publications",
    "topic_specific": "Topics",
    "synthesis": "Synthesis",
}

# Cache the curated payload — recompute once per process, not per request.
_FEATURED_CACHE: "FeaturedResponse | None" = None


class FeaturedItem(BaseModel):
    qa_id: str
    question: str
    intent: str
    source_type: str


class FeaturedGroup(BaseModel):
    intent: str
    label: str
    items: list[FeaturedItem]


class FeaturedResponse(BaseModel):
    groups: list[FeaturedGroup]


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


@app.post("/api/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest) -> SuggestResponse:
    # Trim whitespace and cache a lowercase form for substring matching downstream.
    keyword = req.keyword.strip()
    if len(keyword) < 2:
        return SuggestResponse(suggestions=[])
    keyword_lc = keyword.lower()

    # Resolve the source filter once — None means "no filter, accept everything".
    allowed_sources = SUGGEST_SOURCE_FILTERS.get(req.source_filter or "")

    # Reuse cached system — embedding model, QA FAISS index, and QA items already live in memory.
    sys_data = get_system()
    qa_idx = sys_data["qa_idx"]
    qa_items = sys_data["qa_items"]

    # Guard against empty QA index — return no suggestions rather than call .search(k=0).
    if qa_idx.ntotal == 0:
        return SuggestResponse(suggestions=[])

    # Encode once and rank every QA item by cosine similarity — 53 items, full scan is cheap.
    q_emb = sys_data["embed_model"].encode(
        [keyword], normalize_embeddings=True
    ).astype("float32")
    scores_arr, indices_arr = qa_idx.search(q_emb, qa_idx.ntotal)

    # Materialise the ranked rows once — reuse for both the strict pass and the fallback pass.
    ranked: list[tuple[float, dict]] = []
    for idx, sem_score in zip(indices_arr[0], scores_arr[0]):
        if idx < 0:
            continue
        item = qa_items[int(idx)]
        # Drop no_evidence refusal templates up front.
        if item.get("answer_type") == "no_evidence":
            continue
        # Apply the source filter early — pruned items skip both passes.
        if allowed_sources is not None and item.get("source_type", "mixed") not in allowed_sources:
            continue
        ranked.append((float(sem_score), item))

    # Expand the keyword once — every alias in the same group counts as an equivalent literal hit.
    alias_terms = _expand_alias_terms(keyword_lc)

    # Build a combined score per item — semantic baseline plus a literal-match bonus.
    candidates: list[tuple[float, dict]] = []
    for sem_score_f, item in ranked:
        # Concatenate canonical + alternates, lowercase, and probe for the strongest term hit.
        haystack = " ".join([
            item.get("canonical_question", ""),
            *item.get("alternate_phrasings", []),
        ]).lower()
        substring_bonus = 0.0
        for term in alias_terms:
            pos = haystack.find(term)
            if pos < 0:
                continue
            # Reward earlier matches — taper from 0.5 at position 0 down to ~0.3 at tail.
            bonus = 0.5 - (pos / max(len(haystack), 1)) * 0.2
            if bonus > substring_bonus:
                substring_bonus = bonus

        # Discard matches that fail both signals — keep the dropdown free of noise.
        if substring_bonus == 0.0 and sem_score_f < SUGGEST_MIN_SCORE:
            continue

        candidates.append((sem_score_f + substring_bonus, item))

    # Sort by combined score, descending — best matches rise to the top.
    candidates.sort(key=lambda t: -t[0])

    # Strict pass produced hits — return those and skip the fallback.
    if candidates:
        suggestions = [
            SuggestionItem(
                qa_id=item["qa_id"],
                question=item["canonical_question"],
                score=round(combined, 3),
                source_type=item.get("source_type", "mixed"),
            )
            for combined, item in candidates[: req.top_k]
        ]
        return SuggestResponse(suggestions=suggestions, is_fallback=False)

    # Fallback pass — relax the floor, drop the substring requirement, surface the closest semantic neighbours.
    fallback: list[SuggestionItem] = []
    for sem_score_f, item in ranked[:SUGGEST_FALLBACK_TOP_K]:
        if sem_score_f < SUGGEST_FALLBACK_MIN_SCORE:
            break
        fallback.append(SuggestionItem(
            qa_id=item["qa_id"],
            question=item["canonical_question"],
            score=round(sem_score_f, 3),
            source_type=item.get("source_type", "mixed"),
        ))

    return SuggestResponse(suggestions=fallback, is_fallback=bool(fallback))


@app.get("/api/featured", response_model=FeaturedResponse)
def featured() -> FeaturedResponse:
    global _FEATURED_CACHE
    # Serve cached payload — curation depends only on the static QA bank.
    if _FEATURED_CACHE is not None:
        return _FEATURED_CACHE

    from rag_v2.intent_classifier import classify_intent_keyword

    sys_data = get_system()

    # Re-check under the lock to avoid duplicate compute when parallel cold requests arrive.
    with _FEATURED_LOCK:
        if _FEATURED_CACHE is not None:
            return _FEATURED_CACHE
        qa_items = sys_data["qa_items"]

        # Bucket each canonical question under one intent — fall through to topic_specific.
        buckets: dict[str, list[tuple[float, dict]]] = {key: [] for key in FEATURED_INTENT_ORDER}
        for item in qa_items:
            # Drop refusal templates — surfacing them as suggestions teaches the wrong lesson.
            if item.get("answer_type") == "no_evidence":
                continue

            question = item.get("canonical_question", "")
            intent = classify_intent_keyword(question) or "topic_specific"
            # Coerce stray no_evidence keyword hits back into a content bucket.
            if intent == "no_evidence":
                intent = "topic_specific"
            if intent not in buckets:
                buckets[intent] = []

            # Rank within bucket by alternate-phrasing count (proxy for popularity) plus confidence.
            score = float(item.get("confidence", 0.0)) + 0.25 * len(item.get("alternate_phrasings", []))
            buckets[intent].append((score, item))

        # Assemble groups in the configured display order, dropping any empty buckets.
        groups: list[FeaturedGroup] = []
        for intent in FEATURED_INTENT_ORDER:
            bucket = buckets.get(intent, [])
            if not bucket:
                continue
            bucket.sort(key=lambda t: (-t[0], t[1].get("qa_id", "")))
            items = [
                FeaturedItem(
                    qa_id=entry["qa_id"],
                    question=entry["canonical_question"],
                    intent=intent,
                    source_type=entry.get("source_type", "mixed"),
                )
                for _, entry in bucket[:FEATURED_PER_GROUP]
            ]
            groups.append(FeaturedGroup(
                intent=intent,
                label=FEATURED_INTENT_LABELS.get(intent, intent.replace("_", " ").title()),
                items=items,
            ))

        _FEATURED_CACHE = FeaturedResponse(groups=groups)
        return _FEATURED_CACHE


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
