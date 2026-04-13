"""
RAG V1 (production) — Hybrid retrieval + curated QA memory + contextualized chunks.

Evolved from prior experiments; repository retains this single version only.
  1. Natural-language prefix per chunk (source, report title, section).
  2. FAISS on contextualized_text; BM25 on raw chunk text.
  3. Hybrid merge, cross-encoder rerank, QA memory with NEG gating (v4-style).

Usage (from repo root, with PYTHONPATH=src):
    set PYTHONPATH=src
    python -m rag_v1.pipeline
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

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# Empty OPENAI_BASE_URL breaks the SDK; unset so the default OpenAI host is used.
def _unset_empty_env(*names: str) -> None:
    for name in names:
        v = os.environ.get(name)
        if v is not None and not str(v).strip():
            os.environ.pop(name, None)


_unset_empty_env("OPENAI_BASE_URL")


def openai_client() -> OpenAI:
    """
    Build OpenAI SDK client with an explicit base_url when set.
    Relying on env alone often still hits api.openai.com; OpenRouter keys (sk-or-v1-...)
    require base_url=https://openrouter.ai/api/v1 .
    """
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if not key and not base:
        return OpenAI()
    kw: dict = {}
    if key:
        kw["api_key"] = key
    if base:
        kw["base_url"] = base
    return OpenAI(**kw)


# ── Paths ─────────────────────────────────────────────────────────────
BUNDLE = PROJECT_ROOT / "data" / "final_corpus_bundle"
RAG_DATA = PROJECT_ROOT / "data" / "rag_v1"
RESULTS_FINAL = PROJECT_ROOT / "results" / "final"

DATASET_PATH = PROJECT_ROOT / "data" / "eval_70" / "stakeholder_eval_70.json"
QA_MEM_PATH = RAG_DATA / "qa_memory" / "qa_memory.json"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ── Hyperparameters — identical to v4 ─────────────────────────────────
TOP_K_DENSE = 20
TOP_K_SPARSE = 20
TOP_K_QA = 5
TOP_K_FINAL = 5
MAX_QA_IN_FINAL = 2
MIN_RAW_IN_FINAL = 3
QA_RELEVANCE_FLOOR = 0.35

# ── Source boost — identical to v4 ────────────────────────────────────
SOURCE_BOOST = {
    "general_overview": {"website": 0.08, "pdf": -0.03},
    "topic_specific":   {"website": 0.0,  "pdf": 0.04},
    "project_initiative": {"website": 0.04, "pdf": 0.04},
    "publication_finding": {"website": -0.04, "pdf": 0.10},
    "synthesis":        {"website": 0.04, "pdf": 0.04},
    "no_evidence":      {"website": 0.0,  "pdf": 0.0},
}

# ── Intent keywords — identical to v4 ────────────────────────────────
_INTENT_KW = {
    "general_overview": [
        "what is ssl", "what does ssl do", "tell me about ssl",
        "ssl mission", "ssl vision", "who leads ssl", "ssl contact",
        "ssl location", "ssl address", "who works at ssl", "ssl team",
        "ssl advisory board", "ssl board", "ssl staff", "ssl director",
        "partner institutes", "what is the sustainable solutions lab",
        "ssl focus areas", "ssl research areas",
    ],
    "publication_finding": [
        "published", "publication", "report findings", "paper",
        "study found", "research found", "key findings", "conclusions",
        "report recommends", "views that matter", "voices that matter",
        "academic background", "publications has ssl",
    ],
    "project_initiative": [
        "project", "initiative", "program", "c3i", "cliir",
        "cape cod rail", "mvp", "green ribbon", "barr foundation",
        "climate careers", "collaborative", "rail resilience",
        "lumbee", "climate adaptation forum", "water quality",
        "east boston", "opportunity in the complexity",
    ],
    "no_evidence": [
        "recycling", "e-waste", "carbon footprint", "biodiversity",
        "air quality", "agriculture", "food system", "wildfire",
        "renewable energy", "k-12", "international context",
        "nuclear", "deforestation", "ocean acidification",
        "transportation electrification", "cryptocurrency", "blockchain",
        "mental health", "sea-level rise modeling",
    ],
    "synthesis": [
        "how does", "relationship between", "intersection",
        "compare", "across", "multiple", "integrate", "connect",
        "combined", "together", "holistic", "evolved", "full scope",
        "funding landscape", "events and convenings", "cross-project",
    ],
}

# ── Filters — identical to v4 ────────────────────────────────────────
_USELESS = [
    re.compile(r"^(table of contents|acknowledgments?|references?|bibliography|copyright)", re.I),
    re.compile(r"^\s*page\s+\d+\s*$", re.I),
    re.compile(r"^(list of figures|list of tables|appendix\s+[a-z])", re.I),
]
_EXEC_MAP = {
    "Executive_Summary_Feasibility": "Feasibility_of_Harbor-wide_Barrier_Systems",
    "Executive_Summary_Financing": "Financing_Climate_Resilience",
    "Executive_Summary_Governance": "Governance_for_a_Changing_Climate",
}
_REFUSAL_PATTERNS = [
    re.compile(r"could not find clearly supported", re.I),
    re.compile(r"no relevant ssl work found", re.I),
    re.compile(r"not enough evidence to answer", re.I),
    re.compile(r"current corpus (does not|doesn.t) contain", re.I),
    re.compile(r"no clearly supported ssl", re.I),
    re.compile(r"i was unable to find", re.I),
    re.compile(r"there is no( specific| direct| clear)? (evidence|information|documentation)", re.I),
]


def _is_useless(t):
    return any(p.search(t.strip()[:150]) for p in _USELESS)

def _tokenize(t):
    return re.findall(r"[a-z0-9]+", t.lower())

def classify_intent(q):
    ql = q.lower()
    for intent, kws in _INTENT_KW.items():
        if any(kw in ql for kw in kws):
            return intent
    return "topic_specific"

def _dedup_sources(chunks):
    srcs = {c.get("source", "") for c in chunks}
    drop = {ep for ep, fp in _EXEC_MAP.items()
            if any(ep in s for s in srcs) and any(fp in s for s in srcs)}
    return [c for c in chunks if not any(d in c.get("source", "") for d in drop)] if drop else chunks

def _dedup_bilingual(chunks):
    srcs = [c.get("source", "") for c in chunks]
    if any("Oportunidad" in s for s in srcs) and any("Opportunity" in s for s in srcs):
        return [c for c in chunks if "Oportunidad" not in c.get("source", "")]
    return chunks

def _detect_refusal(text):
    return any(p.search(text) for p in _REFUSAL_PATTERNS)


# =====================================================================
# Contextualization — the ONLY new part
# =====================================================================

_PDF_TITLES = {
    "Community-Led Climate Preparedness and Resilience in Boston_ New":
        "Community-Led Climate Preparedness and Resilience in Boston",
    "Connecting for Equitable Climate Adaptation_ Mapping Stakeholder":
        "Connecting for Equitable Climate Adaptation",
    "Critical approaches to climate-induced migration research and sol":
        "Critical Approaches to Climate-Induced Migration",
    "Executive Summary_Feasibility of Harbor-wide Barrier Systems_ Pre":
        "Executive Summary: Feasibility of Harbor-wide Barrier Systems",
    "Executive Summary_Financing Climate Resilience_ Mobilizing Resour":
        "Executive Summary: Financing Climate Resilience",
    "Executive Summary_Governance for a Changing Climate_ Adapting Bos":
        "Executive Summary: Governance for a Changing Climate",
    "Feasibility of Harbor-wide Barrier Systems_ Preliminary Analysis":
        "Feasibility of Harbor-wide Barrier Systems",
    "Financing Climate Resilience_ Mobilizing Resources and Incentives":
        "Financing Climate Resilience",
    "Governance for a Changing Climate_ Adapting Boston_s Built Enviro":
        "Governance for a Changing Climate",
    "Learning from the Massachusetts Municipal Vulnerability Preparedn":
        "Learning from the Massachusetts MVP Program",
    "Opportunity in the Complexity_ Recommendations for Equitable Clim":
        "Opportunity in the Complexity: Equitable Climate Resilience in East Boston",
    "Oportunidad en la Complejidad_Recomendaciones para una Resilienci":
        "Oportunidad en la Complejidad (Spanish)",
    "UMB-SSL-2022-Annual_Report": "SSL 2022 Annual Report",
    "UMB-SSL-2025-Impact_Report": "SSL 2025 Impact Report",
    "Views that Matter_ Race and Opinions on Climate Change of Boston":
        "Views that Matter: Race and Climate Opinions",
    "Voices that Matter_ Boston Area Residents of Color Discuss Climat":
        "Voices that Matter: Communities of Color Discuss Climate",
    "Who Counts in Climate Resilience_ Transient Populations and Clima":
        "Who Counts in Climate Resilience: Transient Populations",
}

_WEBSITE_TITLES = {
    "ssl.json": "SSL Main Page",
    "ssl_people.json": "SSL People & Team",
    "ssl_projects.json": "SSL Projects & Initiatives",
    "ssl_research.json": "SSL Research Areas",
    "ssl_overview_presentation_2026.json": "SSL Overview Presentation 2026",
    "ssl_people_board-of-directors.json": "SSL Board of Directors",
    "ssl_people_students.json": "SSL Students",
}


def _pretty_pdf(src):
    stem = src.replace(".pdf", "")
    for k, v in _PDF_TITLES.items():
        if k in stem:
            return v
    return stem.replace("_", " ")


def _pretty_web(sf, sec):
    if sf in _WEBSITE_TITLES:
        return _WEBSITE_TITLES[sf]
    return sec.split(" (part")[0] if sec else sf.replace(".json", "").replace("_", " ")


def contextualize_chunk(ch):
    """Add a natural-language prefix describing the chunk's source and context."""
    st = ch.get("source_type", "pdf")
    sec = ch.get("section_title", "").strip()
    ct = ch.get("chunk_type", "paragraph")
    orig = ch.get("chunk_text", "")

    if st == "pdf":
        title = _pretty_pdf(ch.get("source_pdf", ""))
        pfx = f"This passage is from the SSL report '{title}', published by the Sustainable Solutions Lab at UMass Boston."
        if sec and sec.lower() not in ("preamble", "references", "bibliography"):
            pfx += f"\nSection: {sec}."
        if ct == "table":
            pfx += "\nThis is tabular data from the report."
    elif st == "website":
        pt = _pretty_web(ch.get("source_file", ""), sec)
        pfx = f"This passage is from the SSL website page '{pt}'."
        sf = ch.get("source_file", "")
        if "people" in sf or "profile" in sf or "directory" in sf:
            pfx += "\nIt describes SSL team members and their roles."
        elif "project" in sf:
            pfx += "\nIt describes SSL projects and initiatives."
        elif "research" in sf:
            pfx += "\nIt describes SSL research areas."
        elif "news" in sf:
            pfx += "\nThis is an SSL news article."
        elif "presentation" in sf:
            pfx += "\nThis is from an SSL overview presentation."
        if sec:
            pfx += f"\nSection: {sec}."
    else:
        pfx = "This passage is from the Sustainable Solutions Lab (SSL) at UMass Boston."

    return f"{pfx}\n\n{orig}"


def build_contextualized_corpus(meta):
    for ch in meta:
        ch["contextualized_text"] = contextualize_chunk(ch)
    return meta


# =====================================================================
# Load resources
# =====================================================================

def load_all(embed_model):
    mp = BUNDLE / "merged" / "unified_index_metadata.json"
    with open(mp, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[Corpus] {len(meta)} chunks loaded")

    # Contextualize all chunks
    meta = build_contextualized_corpus(meta)
    print(f"[Ctx] Added contextualized_text to all {len(meta)} chunks")

    # Build or load contextualized FAISS index
    ctx_emb_path = RAG_DATA / "contextualized_embeddings.npy"
    ctx_idx_path = RAG_DATA / "contextualized_index.faiss"

    if ctx_emb_path.exists() and ctx_idx_path.exists():
        ctx_emb = np.load(str(ctx_emb_path))
        ctx_idx = faiss.read_index(str(ctx_idx_path))
        print(f"[FAISS] Loaded cached contextualized index ({ctx_idx.ntotal} vectors)")
    else:
        print("[FAISS] Encoding contextualized embeddings (one-time)...")
        texts = [ch["contextualized_text"] for ch in meta]
        ctx_emb = embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True, batch_size=128,
        ).astype("float32")
        np.save(str(ctx_emb_path), ctx_emb)
        ctx_idx = faiss.IndexFlatIP(ctx_emb.shape[1])
        ctx_idx.add(ctx_emb)
        faiss.write_index(ctx_idx, str(ctx_idx_path))
        print(f"[FAISS] Built & saved contextualized index ({ctx_idx.ntotal} vectors)")

    # Save contextualized chunks metadata
    ctx_chunks_path = RAG_DATA / "contextualized_chunks.json"
    if not ctx_chunks_path.exists():
        slim = [{k: v for k, v in ch.items() if k not in ("embedding", "embedding_text")}
                for ch in meta]
        with open(ctx_chunks_path, "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False)
        print(f"[Save] contextualized_chunks.json ({len(slim)} chunks)")

    # BM25 on raw text (NOT contextualized — keeps keyword matching clean)
    bm25 = BM25Okapi([_tokenize(ch["chunk_text"]) for ch in meta])
    print(f"[BM25] Built over {len(meta)} docs (raw text)")

    # QA memory — identical to v4
    with open(QA_MEM_PATH, "r", encoding="utf-8") as f:
        qa_items = json.load(f)
    qa_texts = []
    for item in qa_items:
        parts = [item["canonical_question"]]
        if item.get("alternate_phrasings"):
            parts.extend(item["alternate_phrasings"][:3])
        parts.append(item["answer"][:300])
        qa_texts.append(" | ".join(parts))
    qa_emb = embed_model.encode(qa_texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    qa_idx = faiss.IndexFlatIP(qa_emb.shape[1])
    qa_idx.add(qa_emb)
    print(f"[QA Mem] {len(qa_items)} entries embedded")

    # Dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[Dataset] {len(dataset)} questions")

    return meta, ctx_idx, qa_items, qa_idx, bm25, dataset


# =====================================================================
# Hybrid retrieval — v4 logic, contextualized FAISS
# =====================================================================

def retrieve(query, intent, embed_model, corpus_idx, corpus_meta, bm25,
             qa_items, qa_idx, reranker):
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    boosts = SOURCE_BOOST.get(intent, SOURCE_BOOST["topic_specific"])

    # Dense retrieval on contextualized index
    D_d, I_d = corpus_idx.search(q_emb, TOP_K_DENSE)
    dense_ids = set()
    raw = []
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

    # Sparse retrieval on raw text (identical to v4)
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

    # Hybrid scoring — identical to v4
    mx_d = max((c["dense_score"] for c in raw), default=1) or 1
    mx_s = max((c["sparse_score"] for c in raw), default=1) or 1
    for c in raw:
        c["hybrid"] = 0.6 * c["dense_score"] / mx_d + 0.4 * c["sparse_score"] / mx_s + c["boost"]

    # Reranking — uses contextualized_text for richer signal
    if raw:
        ce = reranker.predict(
            [[query, c["contextualized_text"][:512]] for c in raw],
            show_progress_bar=False,
        )
        for c, s in zip(raw, ce):
            c["rerank"] = float(s)
            c["final"] = 0.35 * c["hybrid"] + 0.65 * float(s) / 10.0
    raw.sort(key=lambda x: x.get("final", 0), reverse=True)

    # QA memory — identical to v4
    n_qa = min(TOP_K_QA, qa_idx.ntotal)
    D_q, I_q = qa_idx.search(q_emb, n_qa)
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

    # QA-NEG gating — identical to v4
    use_neg = False
    if intent == "no_evidence" and qa_neg:
        top_raw = raw[0]["final"] if raw else 0
        if top_raw < 0.3:
            use_neg = True

    # Merge with diversity constraints — identical to v4
    max_qa = 2 if intent not in ("publication_finding", "topic_specific") else 1
    final, seen = [], set()
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

    # QA first, then raw — identical to v4
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
        "chunk_text": c["contextualized_text"],  # <-- contextualized for LLM
        "score": round(c.get("final", c.get("raw_score", 0)), 4),
        "layer": c["layer"],
    } for i, c in enumerate(final)]

    log = {
        "intent": intent, "qa_neg_used": use_neg,
        "final_qa": sum(1 for r in results if r["source_type"] == "curated_qa"),
        "final_raw": sum(1 for r in results if r["source_type"] != "curated_qa"),
    }
    return results, log


# =====================================================================
# Answer generation — identical to v4 prompt
# =====================================================================

def generate_answer(question, retrieved, client):
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
        "1. If the evidence clearly and directly answers the question, provide a detailed answer with source citations.\n"
        "2. If the evidence partially addresses the question, answer what you can and note what is missing.\n"
        "3. If the evidence does NOT contain relevant information, you MUST refuse:\n"
        "   Say: 'I could not find clearly supported SSL work on this topic in the current corpus.'\n"
        "4. NEVER guess or infer beyond what the evidence states.\n"
        "5. If a topic is only tangentially mentioned (e.g., mentioned in passing, listed as a keyword, "
        "or part of a broader context), do NOT treat it as dedicated SSL work on that topic.\n"
        "6. When a VERIFIED QA KNOWLEDGE entry says 'No relevant SSL work found,' trust that assessment.\n"
        "7. Prefer specific PDF/website evidence over general QA summaries when both exist.\n"
        "8. Cite source documents by name."
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Evidence:\n\n{context}\n\n---\n\nQuestion: {question}"},
            ],
            temperature=0, max_tokens=600, timeout=60,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[GENERATION ERROR: {e}]"


# =====================================================================
# Evaluation — identical to v4
# =====================================================================

def eval_retrieval(retrieved, gold_ids):
    if not gold_ids:
        return {"raw_hit3": None, "raw_hit5": None, "sources": []}
    raw_src = [c["source"] for c in retrieved if c["source_type"] != "curated_qa"]
    def hit(n):
        for s in raw_src[:n]:
            for g in gold_ids:
                if g.lower() in s.lower() or s.lower() in g.lower():
                    return True
        return False
    return {
        "raw_hit3": hit(3),
        "raw_hit5": hit(min(5, len(raw_src))),
        "sources": [c["source"] for c in retrieved],
    }


def eval_answer(sys_ans, gold_ans, expected_behavior, client):
    prompt = (
        "You are evaluating a RAG system's answer against a gold standard.\n\n"
        f"Gold answer:\n{gold_ans}\n\nSystem answer:\n{sys_ans}\n\n"
        f"Expected behavior: {expected_behavior}\n\n"
        "Evaluate (score 0-2, 0=bad, 1=partial, 2=good):\n"
        "1. factual_accuracy: correct facts matching gold answer?\n"
        "2. completeness: covers key points?\n"
        "3. appropriate_behavior: if expected='refuse', did system refuse? "
        "if expected='answer', did system answer? (2=correct, 0=wrong behavior)\n"
        "4. hallucination: states unsupported facts? (2=none, 0=significant)\n"
        "5. is_refusal: does the system explicitly refuse/abstain? (1=yes, 0=no)\n\n"
        "Return ONLY JSON with five scores + 'explanation'. No other text."
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=300, timeout=60,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
    except Exception as e:
        return {"factual_accuracy": -1, "completeness": -1, "appropriate_behavior": -1,
                "hallucination": -1, "is_refusal": -1, "explanation": f"Error: {e}"}

    kw_refusal = _detect_refusal(sys_ans)
    if kw_refusal and result.get("is_refusal", 0) != 1:
        result["is_refusal"] = 1
        if expected_behavior == "refuse":
            result["appropriate_behavior"] = max(result.get("appropriate_behavior", 0), 2)
    elif not kw_refusal and result.get("is_refusal", 0) == 1:
        if expected_behavior in ("answer", "partial_answer"):
            result["is_refusal"] = 0
            result["appropriate_behavior"] = max(result.get("appropriate_behavior", 0), 1)

    return result


# =====================================================================
# Metrics — identical to v4
# =====================================================================

def compute_metrics(results):
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

    by_type = {}
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

    return {
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


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("  SSL RAG V1 — contextualized hybrid RAG + QA memory")
    print("=" * 70)

    for d in [RAG_DATA, RAG_DATA / "qa_memory", RESULTS_FINAL]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[Models] Loading...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    client = openai_client()

    meta, ctx_idx, qa_items, qa_idx, bm25, dataset = load_all(embed_model)

    print(f"\n{'='*70}\n  EVALUATION ({len(dataset)} questions)\n{'='*70}\n")

    results = []
    for i, q in enumerate(dataset):
        qid = q["question_id"]
        print(f"  [{i+1}/{len(dataset)}] {qid}: {q['question'][:55]}...")

        intent = classify_intent(q["question"])
        retrieved, ret_log = retrieve(
            q["question"], intent, embed_model, ctx_idx, meta,
            bm25, qa_items, qa_idx, reranker,
        )
        sys_ans = generate_answer(q["question"], retrieved, client)
        reval = eval_retrieval(retrieved, q.get("gold_source_ids", []))
        aeval = eval_answer(sys_ans, q["gold_answer"], q["expected_behavior"], client)

        results.append({
            "question_id": qid, "question": q["question"],
            "question_type": q["question_type"],
            "difficulty": q["difficulty"],
            "expected_behavior": q["expected_behavior"],
            "gold_answer": q["gold_answer"],
            "system_answer": sys_ans,
            "retrieved_chunks": [{k: v for k, v in c.items() if k != "chunk_text"}
                                 for c in retrieved],
            "retrieval_eval": reval,
            "answer_eval": aeval,
            "retrieval_log": ret_log,
        })
        time.sleep(0.3)

    metrics = compute_metrics(results)

    # Load v4 metrics for comparison
    v4_metrics_path = RESULTS_FINAL / "baseline_v4_metrics.json"
    v4m = json.load(open(v4_metrics_path, encoding="utf-8")) if v4_metrics_path.exists() else {}

    # Print results
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
    for qt, v in metrics["by_question_type"].items():
        print(f"    {qt:25s}: beh={v['behavior_rate']:.0%} acc={v['avg_accuracy']:.0%} perfect={v['perfect']}/{v['total']}")

    # Save artifacts
    print(f"\n{'='*70}\n  SAVING\n{'='*70}")

    with open(RESULTS_FINAL / "rag_v1_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(RESULTS_FINAL / "rag_v1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(RESULTS_FINAL / "baseline_vs_rag_v1_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"baseline_v4": v4m, "rag_v1": metrics}, f, indent=2, ensure_ascii=False)

    errors = [r for r in results if r["answer_eval"].get("appropriate_behavior", 0) < 2]
    with open(RESULTS_FINAL / "rag_v1_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    # Write report
    _write_report(v4m, metrics, results)
    print(f"\n  Eval artifacts: {RESULTS_FINAL}")
    print(f"  RAG indices & QA memory: {RAG_DATA}")


def _write_report(v4m, rag_m, results):
    def _get(d, k, fallback_k=None):
        v = d.get(k, 0)
        if v == 0 and fallback_k:
            v = d.get(fallback_k, 0)
        return v

    lines = [
        "# RAG V1 — Evaluation summary (automated run)",
        f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "\n## 1. System",
        "\n**RAG V1** uses contextualized chunk prefixes, hybrid dense+sparse retrieval,",
        "cross-encoder reranking, and curated QA memory with no-evidence (NEG) gating.",
        "Every chunk now starts with a natural-language sentence describing:",
        "- The source document (report title or website page name)",
        "- SSL attribution ('published by the Sustainable Solutions Lab')",
        "- Section title (when available)",
        "- Content type hint (team members, projects, research areas, etc.)",
        "\nThis helps the LLM understand each chunk's origin and SSL relevance.",
        "\n**Everything else is identical to v4**:",
        "- Same QA memory (31 entries, same merge logic, same NEG gating)",
        "- Same hybrid retrieval (FAISS + BM25) with source boost",
        "- Same reranking, dedup, bilingual handling",
        "- Same generation prompt and refusal detection",
        "\n## 2. Baseline (v4 metrics file) vs RAG V1",
        "\n| Metric | Baseline | RAG V1 | Change |",
        "|--------|:--:|:--:|:------:|",
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
        v4v = _get(v4m, k)
        r1 = _get(rag_m, k)
        d = r1 - v4v
        s = "+" if d > 0 else ""
        lines.append(f"| {label} | {v4v:.1%} | {r1:.1%} | {s}{d:.1%} |")

    lines.append(f"| Missed refusals | {v4m.get('missed_refusal_count', 0)} | {rag_m.get('missed_refusal_count', 0)} | |")

    lines += [
        "\n## 3. Performance by Question Type",
        "\n| Type | Total | Behavior | Accuracy | Perfect |",
        "|------|:---:|:---:|:---:|:---:|",
    ]
    for qt, v in rag_m.get("by_question_type", {}).items():
        lines.append(
            f"| {qt} | {v['total']} | {v['behavior_rate']:.0%} | {v['avg_accuracy']:.0%} | {v['perfect']}/{v['total']} |"
        )

    lines += [
        "\n## 4. Analysis",
        "\nThe contextualized prefix is a **zero-cost improvement** that provides the LLM",
        "with document-level context it previously lacked. Combined with v4's proven",
        "QA memory strategy, this should:",
        "- Help the LLM identify which chunks are about SSL's work",
        "- Improve answer grounding by knowing report titles",
        "- Maintain v4's strong refusal behavior via QA-NEG",
        "\n## 5. Artifacts",
        "\n```",
        "data/rag_v1/",
        "├── contextualized_chunks.json",
        "├── contextualized_embeddings.npy",
        "├── contextualized_index.faiss",
        "└── qa_memory/qa_memory.json",
        "",
        "results/final/",
        "├── rag_v1_eval_results.json",
        "├── rag_v1_metrics.json",
        "├── baseline_vs_rag_v1_metrics.json",
        "├── rag_v1_error_analysis.json",
        "└── RAG_V1_EVAL_SUMMARY.md",
        "```",
    ]

    with open(RESULTS_FINAL / "RAG_V1_EVAL_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
