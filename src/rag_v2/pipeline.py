"""
RAG V2 pipeline — Phase 2 evolution of RAG V1.

Enhancements over V1:
  1. LLM-enhanced intent classification (keyword fast-path + LLM fallback)
  2. Query transformation: HyDE, sub-query decomposition, query expansion
  3. Session-level memory for multi-turn dialogue
  4. Coreference resolution for follow-up questions
  5. Re-retrieval vs reuse strategy for multi-turn
  6. Answer-evidence consistency checking
  7. Diversity-aware evidence composition
  8. Enhanced generation prompt with conversation context

Reuses V1 core: corpus loading, FAISS+BM25, cross-encoder reranking, QA memory.

Usage (from repo root, with PYTHONPATH=src):
    set PYTHONPATH=src
    python -m rag_v2.pipeline
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_v1.pipeline import (
    BUNDLE,
    DATASET_PATH,
    EMBED_MODEL_NAME,
    QA_MEM_PATH,
    QA_RELEVANCE_FLOOR,
    RAG_DATA,
    RERANK_MODEL_NAME,
    RESULTS_FINAL,
    SOURCE_BOOST,
    TOP_K_DENSE,
    TOP_K_FINAL,
    TOP_K_QA,
    TOP_K_SPARSE,
    MIN_RAW_IN_FINAL,
    MAX_QA_IN_FINAL,
    _EXEC_MAP,
    _REFUSAL_PATTERNS,
    _USELESS,
    _dedup_bilingual,
    _dedup_sources,
    _detect_refusal,
    _is_useless,
    _tokenize,
    build_contextualized_corpus,
    contextualize_chunk,
    eval_answer,
    eval_retrieval,
    load_all,
    openai_client,
)
from rag_v2.consistency import check_consistency
from rag_v2.intent_classifier import classify_intent
from rag_v2.multiturn import RetrievalStrategy, decide_retrieval_strategy, resolve_coreference
from rag_v2.query_transform import decompose_query, expand_query, hyde_generate
from rag_v2.session import SessionMemory, Turn

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# V2 reads LLM_MODEL_V2 first; falls back to LLM_MODEL for backward compat
LLM_MODEL = os.getenv("LLM_MODEL_V2") or os.getenv("LLM_MODEL", "openai/gpt-5.4")

RESULTS_V2 = PROJECT_ROOT / "results" / "v2"
EVAL_PHASE2_DIR = PROJECT_ROOT / "data" / "eval_phase2"
DATASET_V2_PATH = EVAL_PHASE2_DIR / "stakeholder_eval_phase2.json"


# =====================================================================
# Enhanced retrieval — builds on V1 retrieve()
# =====================================================================

def _retrieve_single(
    query: str,
    intent: str,
    embed_model: SentenceTransformer,
    corpus_idx: faiss.IndexFlatIP,
    corpus_meta: list[dict],
    bm25: BM25Okapi,
    qa_items: list[dict],
    qa_idx: faiss.IndexFlatIP,
    reranker: CrossEncoder,
    *,
    hyde_text: str | None = None,
) -> tuple[list[dict], dict]:
    """Single-query retrieval with optional HyDE embedding.

    This is essentially V1's retrieve() with an optional HyDE vector mixed in.
    """
    queries_to_embed = [query]
    if hyde_text:
        queries_to_embed.append(hyde_text)

    q_embs = embed_model.encode(queries_to_embed, normalize_embeddings=True).astype("float32")
    if len(q_embs) > 1:
        q_emb = np.mean(q_embs, axis=0, keepdims=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    else:
        q_emb = q_embs[:1]

    boosts = SOURCE_BOOST.get(intent, SOURCE_BOOST["topic_specific"])

    D_d, I_d = corpus_idx.search(q_emb, TOP_K_DENSE)
    dense_ids = set()
    raw: list[dict] = []
    for i, s in zip(I_d[0], D_d[0]):
        if i < 0:
            continue
        ch = corpus_meta[i]
        if _is_useless(ch["chunk_text"]):
            continue
        dense_ids.add(int(i))
        st = ch.get("source_type", "pdf")
        raw.append({
            "corpus_idx": int(i), "chunk_id": ch["chunk_id"],
            "source_type": st,
            "source": ch.get("source_file", ch.get("source_pdf", "?")),
            "section_title": ch.get("section_title", ""),
            "chunk_text": ch["chunk_text"],
            "contextualized_text": ch["contextualized_text"],
            "dense_score": float(s), "sparse_score": 0.0,
            "ret_src": "dense",
            "boost": boosts.get(st, 0),
            "layer": "corpus",
        })

    bm_scores = bm25.get_scores(_tokenize(query))
    for i in np.argsort(bm_scores)[::-1][:TOP_K_SPARSE]:
        i = int(i)
        bs = float(bm_scores[i])
        if bs <= 0:
            continue
        if i in dense_ids:
            for c in raw:
                if c["corpus_idx"] == i:
                    c["sparse_score"] = bs
                    c["ret_src"] = "both"
                    break
        else:
            ch = corpus_meta[i]
            if _is_useless(ch["chunk_text"]):
                continue
            st = ch.get("source_type", "pdf")
            raw.append({
                "corpus_idx": i, "chunk_id": ch["chunk_id"],
                "source_type": st,
                "source": ch.get("source_file", ch.get("source_pdf", "?")),
                "section_title": ch.get("section_title", ""),
                "chunk_text": ch["chunk_text"],
                "contextualized_text": ch["contextualized_text"],
                "dense_score": 0.0, "sparse_score": bs,
                "ret_src": "sparse",
                "boost": boosts.get(st, 0),
                "layer": "corpus",
            })

    raw = _dedup_sources(raw)
    raw = _dedup_bilingual(raw)

    mx_d = max((c["dense_score"] for c in raw), default=1) or 1
    mx_s = max((c["sparse_score"] for c in raw), default=1) or 1
    for c in raw:
        c["hybrid"] = 0.6 * c["dense_score"] / mx_d + 0.4 * c["sparse_score"] / mx_s + c["boost"]

    if raw:
        ce = reranker.predict(
            [[query, c["contextualized_text"][:512]] for c in raw],
            show_progress_bar=False,
        )
        for c, s in zip(raw, ce):
            c["rerank"] = float(s)
            c["final"] = 0.35 * c["hybrid"] + 0.65 * float(s) / 10.0
    raw.sort(key=lambda x: x.get("final", 0), reverse=True)

    q_emb_qa = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    n_qa = min(TOP_K_QA, qa_idx.ntotal)
    D_q, I_q = qa_idx.search(q_emb_qa, n_qa)
    qa_pos, qa_neg = [], []
    for i, s in zip(I_q[0], D_q[0]):
        if i < 0 or float(s) < QA_RELEVANCE_FLOOR:
            continue
        item = qa_items[i]
        entry = {
            "chunk_id": item["qa_id"], "source_type": "curated_qa",
            "source": "qa_memory",
            "section_title": item["canonical_question"],
            "chunk_text": item["answer"],
            "contextualized_text": item["answer"],
            "raw_score": float(s),
            "qa_conf": item.get("confidence", 1.0),
            "answer_type": item.get("answer_type", "factual"),
            "layer": "qa_memory",
            "is_neg": item.get("answer_type") == "no_evidence",
        }
        (qa_neg if entry["is_neg"] else qa_pos).append(entry)

    if qa_pos:
        ce_q = reranker.predict(
            [[query, c["chunk_text"][:512]] for c in qa_pos],
            show_progress_bar=False,
        )
        for c, s in zip(qa_pos, ce_q):
            c["final"] = float(s) / 10.0
    qa_pos.sort(key=lambda x: x.get("final", 0), reverse=True)

    use_neg = False
    if intent == "no_evidence" and qa_neg:
        top_raw = raw[0]["final"] if raw else 0
        if top_raw < 0.3:
            use_neg = True

    max_qa = 2 if intent not in ("publication_finding", "topic_specific") else 1
    final: list[dict] = []
    seen: set[str] = set()
    for c in raw[:MIN_RAW_IN_FINAL]:
        if c["chunk_id"] not in seen:
            final.append(c)
            seen.add(c["chunk_id"])
    qa_n = 0
    for c in qa_pos:
        if qa_n >= max_qa or len(final) >= TOP_K_FINAL:
            break
        if c["chunk_id"] not in seen:
            final.append(c)
            seen.add(c["chunk_id"])
            qa_n += 1
    if use_neg:
        for c in qa_neg[:1]:
            if len(final) < TOP_K_FINAL and c["chunk_id"] not in seen:
                final.append(c)
                seen.add(c["chunk_id"])
    for c in raw:
        if len(final) >= TOP_K_FINAL:
            break
        if c["chunk_id"] not in seen:
            final.append(c)
            seen.add(c["chunk_id"])

    qa_f = [c for c in final if c["layer"] == "qa_memory"]
    raw_f = sorted(
        [c for c in final if c["layer"] != "qa_memory"],
        key=lambda x: x.get("final", 0), reverse=True,
    )
    final = qa_f + raw_f

    results = [{
        "rank": i + 1, "chunk_id": c["chunk_id"],
        "source_type": c["source_type"], "source": c["source"],
        "section_title": c["section_title"],
        "chunk_text": c["contextualized_text"],
        "score": round(c.get("final", c.get("raw_score", 0)), 4),
        "layer": c["layer"],
    } for i, c in enumerate(final)]

    log = {
        "intent": intent, "qa_neg_used": use_neg,
        "hyde_used": hyde_text is not None,
        "final_qa": sum(1 for r in results if r["source_type"] == "curated_qa"),
        "final_raw": sum(1 for r in results if r["source_type"] != "curated_qa"),
    }
    return results, log


# =====================================================================
# Multi-hop: coverage check + supplemental retrieval
# =====================================================================

def _extract_session_entities(session: SessionMemory) -> list[str]:
    """Extract key entity names mentioned in recent session turns.

    Uses resolved_query (which already has coreference applied) and
    answer text to find proper nouns and known project/report names.
    """
    KNOWN_ENTITIES = {
        "c3i": "C3I",
        "climate careers curricula initiative": "C3I",
        "cape cod rail": "Cape Cod Rail Resilience",
        "east boston": "East Boston",
        "opportunity in the complexity": "Opportunity in the Complexity",
        "harbor barrier": "harbor barrier",
        "feasibility of harbor-wide barrier": "harbor barrier",
        "cliir": "CLIIR",
        "views that matter": "Views that Matter",
        "voices that matter": "Voices that Matter",
        "inenas": "INENAS",
        "institute for new england native american studies": "INENAS",
        "balachandran": "B.R. Balachandran",
        "negrón": "Rosalyn Negrón",
        "negron": "Rosalyn Negrón",
        "gabriela boscio": "Gabriela Boscio Santos",
        "elisa guerrero": "Elisa Guerrero",
        "cedric woods": "Cedric Woods",
        "paul kirshen": "Paul Kirshen",
        "paul watanabe": "Paul Watanabe",
        "barr foundation": "Barr Foundation",
        "liberty mutual": "Liberty Mutual Foundation",
        "northeast climate justice": "Northeast Climate Justice Research Collaborative",
        "mvp program": "MVP Program",
        "who counts": "Who Counts in Climate Resilience",
        "financing climate resilience": "Financing Climate Resilience",
        "governance for a changing climate": "Governance for a Changing Climate",
        "lumbee": "Lumbee Tribe",
        "gastón institute": "Gastón Institute",
        "trotter institute": "Trotter Institute",
        "asian american studies": "Institute for Asian American Studies",
    }

    found: dict[str, str] = {}
    for turn in session.turns[-3:]:
        text_pool = (turn.resolved_query + " " + turn.answer[:500]).lower()
        for pattern, canonical in KNOWN_ENTITIES.items():
            if pattern in text_pool and canonical not in found.values():
                found[pattern] = canonical

    return list(found.values())


def _check_coverage(
    results: list[dict],
    entities: list[str],
    min_coverage_ratio: float = 0.5,
) -> list[str]:
    """Check which session entities are missing from retrieval results.

    Returns list of entity names that have zero representation in the
    retrieved chunks' text content.
    """
    if not entities:
        return []

    result_text = " ".join(
        r.get("chunk_text", "") + " " + r.get("section_title", "")
        for r in results
    ).lower()

    missing = []
    for entity in entities:
        search_terms = entity.lower().split()
        if not any(term in result_text for term in search_terms if len(term) > 2):
            missing.append(entity)

    return missing


def _supplemental_retrieve(
    missing_entities: list[str],
    intent: str,
    embed_model: SentenceTransformer,
    corpus_idx: faiss.IndexFlatIP,
    corpus_meta: list[dict],
    bm25: BM25Okapi,
    reranker: CrossEncoder,
    existing_ids: set[str],
    top_k: int = 3,
) -> list[dict]:
    """Lightweight retrieval for missing entities — no HyDE, no QA memory."""
    supplements: list[dict] = []

    for entity in missing_entities:
        q_emb = embed_model.encode([entity], normalize_embeddings=True).astype("float32")
        D, I = corpus_idx.search(q_emb, TOP_K_DENSE // 2)

        raw: list[dict] = []
        for i, s in zip(I[0], D[0]):
            if i < 0:
                continue
            ch = corpus_meta[i]
            cid = ch["chunk_id"]
            if cid in existing_ids or _is_useless(ch["chunk_text"]):
                continue
            raw.append({
                "chunk_id": cid,
                "source_type": ch.get("source_type", "pdf"),
                "source": ch.get("source_file", ch.get("source_pdf", "?")),
                "section_title": ch.get("section_title", ""),
                "chunk_text": ch["contextualized_text"],
                "dense_score": float(s),
                "layer": "corpus",
            })

        bm_scores = bm25.get_scores(_tokenize(entity))
        for i in np.argsort(bm_scores)[::-1][:TOP_K_SPARSE // 2]:
            i = int(i)
            bs = float(bm_scores[i])
            if bs <= 0:
                continue
            ch = corpus_meta[i]
            cid = ch["chunk_id"]
            if cid in existing_ids or any(r["chunk_id"] == cid for r in raw):
                continue
            if _is_useless(ch["chunk_text"]):
                continue
            raw.append({
                "chunk_id": cid,
                "source_type": ch.get("source_type", "pdf"),
                "source": ch.get("source_file", ch.get("source_pdf", "?")),
                "section_title": ch.get("section_title", ""),
                "chunk_text": ch["contextualized_text"],
                "dense_score": 0.0,
                "layer": "corpus",
            })

        if raw:
            ce = reranker.predict(
                [[entity, c["chunk_text"][:512]] for c in raw],
                show_progress_bar=False,
            )
            for c, sc in zip(raw, ce):
                c["score"] = float(sc) / 10.0
            raw.sort(key=lambda x: x["score"], reverse=True)
            for c in raw[:top_k]:
                existing_ids.add(c["chunk_id"])
                supplements.append(c)

    return supplements


def _apply_diversity(results: list[dict], max_per_source: int = 2) -> list[dict]:
    """Limit chunks per source document to improve diversity."""
    source_count: Counter = Counter()
    diverse: list[dict] = []
    overflow: list[dict] = []
    for r in results:
        src = r["source"]
        if source_count[src] < max_per_source:
            diverse.append(r)
            source_count[src] += 1
        else:
            overflow.append(r)

    while len(diverse) < TOP_K_FINAL and overflow:
        diverse.append(overflow.pop(0))

    for i, r in enumerate(diverse):
        r["rank"] = i + 1
    return diverse


def retrieve_v2(
    question: str,
    *,
    session: SessionMemory,
    client: OpenAI,
    embed_model: SentenceTransformer,
    corpus_idx: faiss.IndexFlatIP,
    corpus_meta: list[dict],
    bm25: BM25Okapi,
    qa_items: list[dict],
    qa_idx: faiss.IndexFlatIP,
    reranker: CrossEncoder,
    enable_hyde: bool = True,
    enable_decompose: bool = True,
    enable_diversity: bool = True,
) -> tuple[list[dict], dict, str, str]:
    """Full V2 retrieval pipeline with multi-turn and query transformation.

    Returns (results, retrieval_log, resolved_query, intent).
    """
    resolved_query = resolve_coreference(question, session, client, LLM_MODEL)

    strategy = decide_retrieval_strategy(question, resolved_query, session, client, LLM_MODEL)

    if strategy == RetrievalStrategy.REUSE_EVIDENCE and session.last_turn:
        intent = session.last_turn.intent
        log = {
            "intent": intent, "strategy": "reuse_evidence",
            "resolved_query": resolved_query,
            "original_query": question,
        }
        return session.get_last_sources(), log, resolved_query, intent

    intent = classify_intent(resolved_query, client=client, model=LLM_MODEL)

    _HYDE_INTENTS = {"synthesis", "topic_specific", "publication_finding"}
    hyde_text = None
    if enable_hyde and intent in _HYDE_INTENTS:
        try:
            hyde_text = hyde_generate(resolved_query, client, LLM_MODEL)
        except Exception:
            hyde_text = None

    if enable_decompose and intent == "synthesis":
        try:
            sub_queries = decompose_query(resolved_query, client, LLM_MODEL)
        except Exception:
            sub_queries = [resolved_query]
    else:
        sub_queries = [resolved_query]

    if len(sub_queries) > 1:
        all_results: list[dict] = []
        seen_ids: set[str] = set()
        sub_logs: list[dict] = []
        for sq in sub_queries:
            sq_results, sq_log = _retrieve_single(
                sq, intent, embed_model, corpus_idx, corpus_meta,
                bm25, qa_items, qa_idx, reranker,
                hyde_text=hyde_text,
            )
            sub_logs.append(sq_log)
            for r in sq_results:
                if r["chunk_id"] not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r["chunk_id"])

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = all_results[:TOP_K_FINAL]
        for i, r in enumerate(results):
            r["rank"] = i + 1
        merge_log = {
            "intent": intent, "strategy": strategy.value,
            "sub_queries": sub_queries, "sub_logs": sub_logs,
            "hyde_used": hyde_text is not None,
            "resolved_query": resolved_query,
            "original_query": question,
        }
    else:
        results, merge_log = _retrieve_single(
            resolved_query, intent, embed_model, corpus_idx, corpus_meta,
            bm25, qa_items, qa_idx, reranker,
            hyde_text=hyde_text,
        )
        merge_log["strategy"] = strategy.value
        merge_log["resolved_query"] = resolved_query
        merge_log["original_query"] = question

    if strategy == RetrievalStrategy.MERGE_EVIDENCE and session.last_turn:
        old_sources = session.get_last_sources()
        old_ids = {r["chunk_id"] for r in results}
        for old in old_sources:
            if old["chunk_id"] not in old_ids and len(results) < TOP_K_FINAL + 2:
                results.append(old)

    # --- Multi-hop: coverage check + supplemental retrieval ---
    multihop_log: dict[str, Any] = {"triggered": False}
    if len(session) > 0:
        entities = _extract_session_entities(session)
        if entities:
            missing = _check_coverage(results, entities)
            if missing:
                existing_ids = {r["chunk_id"] for r in results}
                supplements = _supplemental_retrieve(
                    missing, intent, embed_model, corpus_idx, corpus_meta,
                    bm25, reranker, existing_ids, top_k=2,
                )
                for s in supplements:
                    s["rank"] = len(results) + 1
                    s["layer"] = s.get("layer", "corpus")
                    results.append(s)
                multihop_log = {
                    "triggered": True,
                    "session_entities": entities,
                    "missing_entities": missing,
                    "supplements_added": len(supplements),
                }
    merge_log["multihop"] = multihop_log

    if enable_diversity:
        results = _apply_diversity(results)

    return results, merge_log, resolved_query, intent


# =====================================================================
# Enhanced generation — V2 prompt with conversation context
# =====================================================================

def generate_answer_v2(
    question: str,
    retrieved: list[dict],
    client: OpenAI,
    session: SessionMemory | None = None,
) -> str:
    """Generate answer with optional conversation context injection."""
    qa_parts, web_parts, pdf_parts = [], [], []
    for c in retrieved:
        label = f"[Source: {c['source']} | Section: {c['section_title']}]"
        block = f"{label}\n{c['chunk_text']}"
        if c["source_type"] == "curated_qa":
            qa_parts.append(block)
        elif c["source_type"] == "website":
            web_parts.append(block)
        else:
            pdf_parts.append(block)

    sections = []
    if qa_parts:
        sections.append("=== VERIFIED QA KNOWLEDGE ===\n" + "\n\n".join(qa_parts))
    if web_parts:
        sections.append("=== WEBSITE EVIDENCE ===\n" + "\n\n".join(web_parts))
    if pdf_parts:
        sections.append("=== PDF RESEARCH EVIDENCE ===\n" + "\n\n".join(pdf_parts))
    context = "\n\n---\n\n".join(sections) if sections else "(No evidence)"

    system_prompt = (
        "You are a research assistant for the Sustainable Solutions Lab (SSL) at UMass Boston.\n"
        "Answer ONLY based on the provided evidence. Follow these rules strictly:\n\n"
        "1. If the evidence clearly and directly answers the question, provide a detailed, well-organized answer.\n"
        "2. If the evidence partially addresses the question, answer what you can and note what is missing.\n"
        "3. If the evidence does NOT contain relevant information, you MUST refuse:\n"
        "   Say: 'I could not find clearly supported SSL work on this topic in the current corpus.'\n"
        "4. NEVER guess or infer beyond what the evidence states.\n"
        "5. If a topic is only tangentially mentioned (e.g., mentioned in passing, listed as a keyword, "
        "or part of a broader context), do NOT treat it as dedicated SSL work on that topic.\n"
        "6. When a VERIFIED QA KNOWLEDGE entry says 'No relevant SSL work found,' trust that assessment.\n"
        "7. Prefer specific PDF/website evidence over general QA summaries when both exist.\n"
        "8. Do NOT include source file names, chunk references, or raw evidence text in your answer. "
        "Write a clean, natural response as if you already know the information. "
        "Source attribution is handled separately by the system.\n"
        "9. For follow-up questions, you may reference information from the conversation history "
        "but always prioritize the current evidence."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if session and len(session) > 0:
        conv_context = session.get_context_summary(last_n=2)
        messages.append({
            "role": "user",
            "content": f"[Conversation context for reference — do NOT answer from this alone]\n{conv_context}",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I will use the conversation context only for reference and answer based on the provided evidence.",
        })

    messages.append({
        "role": "user",
        "content": f"Evidence:\n\n{context}\n\n---\n\nQuestion: {question}",
    })

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=600,
            timeout=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[GENERATION ERROR: {e}]"


# =====================================================================
# Metrics — extends V1 compute_metrics with V2 fields
# =====================================================================

def compute_metrics_v2(results: list[dict]) -> dict:
    """Compute V2 metrics (superset of V1 metrics plus consistency stats)."""
    N = len(results)
    valid = [r for r in results if r["answer_eval"].get("factual_accuracy", -1) >= 0]
    answerable = [r for r in results if r["expected_behavior"] in ("answer", "partial_answer")]
    refuse_expected = [r for r in results if r["expected_behavior"] == "refuse"]
    avg = lambda k: float(np.mean([r["answer_eval"][k] for r in valid])) / 2 if valid else 0

    system_refused = [r for r in results if r["answer_eval"].get("is_refusal", 0) == 1]
    correct_refusals = [r for r in system_refused if r["expected_behavior"] == "refuse"]
    false_refusals = [r for r in system_refused if r["expected_behavior"] in ("answer", "partial_answer")]
    missed_refusals = [r for r in refuse_expected if r["answer_eval"].get("is_refusal", 0) != 1]

    answerable_valid = [r for r in answerable if r["answer_eval"].get("factual_accuracy", -1) >= 0]
    ans_accuracy = float(np.mean([r["answer_eval"]["factual_accuracy"] for r in answerable_valid])) / 2 if answerable_valid else 0

    appropriate = sum(1 for r in valid if r["answer_eval"].get("appropriate_behavior", 0) >= 1)
    coverage = appropriate / N if N else 0

    with_gold = [r for r in answerable if r["retrieval_eval"].get("raw_hit5") is not None]
    raw_hit3 = sum(1 for r in with_gold if r["retrieval_eval"]["raw_hit3"]) / max(len(with_gold), 1)
    raw_hit5 = sum(1 for r in with_gold if r["retrieval_eval"]["raw_hit5"]) / max(len(with_gold), 1)

    ev_supported = sum(
        1 for r in with_gold
        if r["retrieval_eval"].get("raw_hit5") and r["answer_eval"].get("factual_accuracy", 0) >= 1
    ) / max(len(with_gold), 1)

    consistency_results = [r for r in results if "consistency" in r]
    consistency_rate = (
        sum(1 for r in consistency_results if r["consistency"].get("is_consistent", True))
        / max(len(consistency_results), 1)
    ) if consistency_results else None

    by_type: dict[str, dict] = {}
    for r in results:
        qt = r["question_type"]
        if qt not in by_type:
            by_type[qt] = {"total": 0, "correct_behavior": 0, "acc_sum": 0, "acc_n": 0, "perfect": 0}
        by_type[qt]["total"] += 1
        if r["answer_eval"].get("appropriate_behavior", 0) >= 2:
            by_type[qt]["correct_behavior"] += 1
        a = r["answer_eval"].get("factual_accuracy", -1)
        if a >= 0:
            by_type[qt]["acc_sum"] += a
            by_type[qt]["acc_n"] += 1
            if a >= 2:
                by_type[qt]["perfect"] += 1

    metrics = {
        "total_questions": N,
        "answerable": len(answerable),
        "refuse_expected": len(refuse_expected),
        "raw_corpus_hit_at_3": round(raw_hit3, 4),
        "raw_corpus_hit_at_5": round(raw_hit5, 4),
        "answer_accuracy_answerable": round(ans_accuracy, 4),
        "answer_accuracy_all": round(avg("factual_accuracy"), 4),
        "answer_completeness": round(avg("completeness"), 4),
        "hallucination_rate": round(1 - avg("hallucination"), 4),
        "coverage": round(coverage, 4),
        "refusal_rate": round(len(system_refused) / N, 4) if N else 0,
        "correct_refusal_rate": round(len(correct_refusals) / max(len(refuse_expected), 1), 4),
        "false_refusal_rate": round(len(false_refusals) / max(len(answerable), 1), 4),
        "missed_refusal_count": len(missed_refusals),
        "evidence_supported_accuracy": round(ev_supported, 4),
        "by_question_type": {
            qt: {
                "total": v["total"],
                "correct_behavior": v["correct_behavior"],
                "behavior_rate": round(v["correct_behavior"] / v["total"], 3),
                "avg_accuracy": round(v["acc_sum"] / max(v["acc_n"], 1) / 2, 3),
                "perfect": v["perfect"],
            }
            for qt, v in by_type.items()
        },
    }
    if consistency_rate is not None:
        metrics["consistency_rate"] = round(consistency_rate, 4)
    return metrics


# =====================================================================
# Single-question interactive entry point (for UI / demo)
# =====================================================================

def ask(
    question: str,
    *,
    session: SessionMemory,
    client: OpenAI,
    embed_model: SentenceTransformer,
    corpus_idx: faiss.IndexFlatIP,
    corpus_meta: list[dict],
    bm25: BM25Okapi,
    qa_items: list[dict],
    qa_idx: faiss.IndexFlatIP,
    reranker: CrossEncoder,
    enable_hyde: bool = True,
    enable_decompose: bool = True,
    enable_diversity: bool = True,
    enable_consistency: bool = False,
) -> dict[str, Any]:
    """Process a single question through the full V2 pipeline. Returns a rich result dict."""
    retrieved, ret_log, resolved_query, intent = retrieve_v2(
        question,
        session=session,
        client=client,
        embed_model=embed_model,
        corpus_idx=corpus_idx,
        corpus_meta=corpus_meta,
        bm25=bm25,
        qa_items=qa_items,
        qa_idx=qa_idx,
        reranker=reranker,
        enable_hyde=enable_hyde,
        enable_decompose=enable_decompose,
        enable_diversity=enable_diversity,
    )

    answer = generate_answer_v2(question, retrieved, client, session=session)

    consistency = None
    if enable_consistency and not _detect_refusal(answer):
        try:
            consistency = check_consistency(answer, retrieved, client, LLM_MODEL)
        except Exception:
            pass

    turn = Turn(
        question=question,
        resolved_query=resolved_query,
        intent=intent,
        answer=answer,
        sources=retrieved,
        retrieval_log=ret_log,
    )
    session.add_turn(turn)

    result: dict[str, Any] = {
        "question": question,
        "resolved_query": resolved_query,
        "intent": intent,
        "answer": answer,
        "retrieved": retrieved,
        "retrieval_log": ret_log,
    }
    if consistency:
        result["consistency"] = {
            "is_consistent": consistency.is_consistent,
            "confidence": consistency.confidence,
            "unsupported_claims": consistency.unsupported_claims,
            "explanation": consistency.explanation,
        }
    return result


# =====================================================================
# Batch evaluation entry point
# =====================================================================

def main():
    print("=" * 70)
    print("  SSL RAG V2 — Phase 2 enhanced pipeline")
    print("=" * 70)

    for d in [RAG_DATA, RAG_DATA / "qa_memory", RESULTS_V2]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[Models] Loading...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    client = openai_client()

    meta, ctx_idx, qa_items, qa_idx, bm25, dataset_v1 = load_all(embed_model)

    if DATASET_V2_PATH.exists():
        with open(DATASET_V2_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"[Dataset] V2 evaluation set loaded: {len(dataset)} questions")
    else:
        dataset = dataset_v1
        print(f"[Dataset] Using V1 evaluation set: {len(dataset)} questions (V2 set not found)")

    print(f"\n{'='*70}\n  EVALUATION ({len(dataset)} questions)\n{'='*70}\n")

    session = SessionMemory(max_turns=10)
    results = []
    for i, q in enumerate(dataset):
        qid = q["question_id"]
        print(f"  [{i+1}/{len(dataset)}] {qid}: {q['question'][:55]}...")

        session.clear()
        out = ask(
            q["question"],
            session=session,
            client=client,
            embed_model=embed_model,
            corpus_idx=ctx_idx,
            corpus_meta=meta,
            bm25=bm25,
            qa_items=qa_items,
            qa_idx=qa_idx,
            reranker=reranker,
            enable_consistency=True,
        )
        retrieved = out["retrieved"]
        ret_log = out["retrieval_log"]
        sys_ans = out["answer"]

        reval = eval_retrieval(retrieved, q.get("gold_source_ids", []))
        aeval = eval_answer(sys_ans, q["gold_answer"], q["expected_behavior"], client)

        result_entry: dict[str, Any] = {
            "question_id": qid, "question": q["question"],
            "question_type": q["question_type"],
            "difficulty": q["difficulty"],
            "expected_behavior": q["expected_behavior"],
            "gold_answer": q["gold_answer"],
            "system_answer": sys_ans,
            "resolved_query": out.get("resolved_query", q["question"]),
            "intent": out.get("intent", ret_log.get("intent")),
            "retrieved_chunks": [{k: v for k, v in c.items() if k != "chunk_text"}
                                 for c in retrieved],
            "retrieval_eval": reval,
            "answer_eval": aeval,
            "retrieval_log": ret_log,
        }
        if "consistency" in out:
            result_entry["consistency"] = out["consistency"]

        results.append(result_entry)
        time.sleep(0.3)

    metrics = compute_metrics_v2(results)

    v1_70_path = RESULTS_FINAL / "rag_v1_metrics.json"
    v1_70 = json.load(open(v1_70_path, encoding="utf-8")) if v1_70_path.exists() else {}
    v1_p2_path = RESULTS_V2 / "rag_v1_phase2_metrics.json"
    v1_p2 = json.load(open(v1_p2_path, encoding="utf-8")) if v1_p2_path.exists() else {}

    print(f"\n{'='*70}\n  RESULTS\n{'='*70}")
    print(f"  raw_hit@3: {metrics['raw_corpus_hit_at_3']:.1%}  raw_hit@5: {metrics['raw_corpus_hit_at_5']:.1%}")
    print(f"  accuracy (answerable): {metrics['answer_accuracy_answerable']:.1%}")
    print(f"  accuracy (all): {metrics['answer_accuracy_all']:.1%}")
    print(f"  completeness: {metrics['answer_completeness']:.1%}")
    print(f"  hallucination: {metrics['hallucination_rate']:.1%}")
    print(f"  coverage: {metrics['coverage']:.1%}")
    print(f"  refusal: {metrics['refusal_rate']:.1%}  correct: {metrics['correct_refusal_rate']:.1%}  false: {metrics['false_refusal_rate']:.1%}")
    print(f"  missed refusals: {metrics['missed_refusal_count']}")
    print(f"  ev_supported: {metrics['evidence_supported_accuracy']:.1%}")
    if "consistency_rate" in metrics:
        print(f"  consistency: {metrics['consistency_rate']:.1%}")
    for qt, v in metrics["by_question_type"].items():
        print(f"    {qt:25s}: beh={v['behavior_rate']:.0%} acc={v['avg_accuracy']:.0%} perfect={v['perfect']}/{v['total']}")

    print(f"\n{'='*70}\n  SAVING\n{'='*70}")

    with open(RESULTS_V2 / "rag_v2_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(RESULTS_V2 / "rag_v2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(RESULTS_V2 / "v1_vs_v2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "rag_v1_baseline_70": v1_70,
                "rag_v1_same_dataset_95": v1_p2,
                "rag_v2": metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    errors = [r for r in results if r["answer_eval"].get("appropriate_behavior", 0) < 2]
    with open(RESULTS_V2 / "rag_v2_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    _write_v2_report(v1_70, v1_p2, metrics, results)
    print(f"\n  Eval artifacts: {RESULTS_V2}")


def _write_v2_report(v1_70: dict, v1_p2: dict, v2m: dict, results: list[dict]) -> None:
    def _get(d, k):
        return d.get(k, 0)

    lines = [
        "# RAG V2 — Evaluation summary (automated run)",
        f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "\n## 1. System enhancements over V1",
        "\n**RAG V2** adds the following on top of V1's contextualized hybrid retrieval:",
        "- **LLM-enhanced intent classification** (keyword fast-path + LLM fallback for no_evidence)",
        "- **HyDE** (Hypothetical Document Embeddings) for improved dense retrieval",
        "- **Sub-query decomposition** for synthesis questions",
        "- **Diversity-aware evidence selection** (max 2 chunks per source)",
        "- **Answer-evidence consistency checking** (post-generation verification)",
        "- **Session memory** and **multi-turn dialogue** support (coreference resolution)",
        "- **Gradio Web UI** with source attribution and retrieval transparency",
        f"\n## 2. Same-dataset comparison ({len(results)} questions: V1 vs V2)",
        "\n| Metric | V1 (Phase2 set) | V2 | Δ |",
        "|--------|:--:|:--:|:---:|",
    ]

    pairs = [
        ("raw_corpus_hit_at_3", "raw_hit@3"),
        ("raw_corpus_hit_at_5", "raw_hit@5"),
        ("answer_accuracy_answerable", "Accuracy (answerable)"),
        ("answer_accuracy_all", "Accuracy (all)"),
        ("answer_completeness", "Completeness"),
        ("hallucination_rate", "Hallucination"),
        ("coverage", "Coverage"),
        ("correct_refusal_rate", "Correct refusal"),
        ("false_refusal_rate", "False refusal"),
        ("evidence_supported_accuracy", "Evidence-supported"),
    ]
    for k, label in pairs:
        v1v = _get(v1_p2, k)
        v2v = _get(v2m, k)
        d = v2v - v1v
        s = "+" if d > 0 else ""
        lines.append(f"| {label} | {v1v:.1%} | {v2v:.1%} | {s}{d:.1%} |")

    lines.append(
        f"| Missed refusals | {v1_p2.get('missed_refusal_count', 0)} | "
        f"{v2m.get('missed_refusal_count', 0)} | |"
    )
    if "consistency_rate" in v2m:
        lines.append(f"| Consistency | — | {v2m['consistency_rate']:.1%} | (new) |")

    lines += [
        "\n## 3. Performance by Question Type",
        "\n| Type | Total | Behavior | Accuracy | Perfect |",
        "|------|:---:|:---:|:---:|:---:|",
    ]
    for qt, v in v2m.get("by_question_type", {}).items():
        lines.append(
            f"| {qt} | {v['total']} | {v['behavior_rate']:.0%} | {v['avg_accuracy']:.0%} | {v['perfect']}/{v['total']} |"
        )

    lines += [
        "\n## 4. Artifacts",
        "\n```",
        "results/v2/",
        "├── rag_v2_eval_results.json",
        "├── rag_v2_metrics.json",
        "├── v1_vs_v2_metrics.json",
        "├── rag_v2_error_analysis.json",
        "└── RAG_V2_EVAL_SUMMARY.md",
        "```",
    ]

    with open(RESULTS_V2 / "RAG_V2_EVAL_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
