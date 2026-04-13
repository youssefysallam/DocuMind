#!/usr/bin/env python3
"""
Batch pipeline: PDFs under SSL+PDF → agent segmentation → structured chunks + optional embeddings.

Reuses main._process_one and existing PDF extraction; summaries/keywords are heuristic, not fabricated.
Embeddings: prefer OPENAI_API_KEY + EMBEDDING_MODEL; otherwise install sentence-transformers for local models.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import raw_to_embedding.config as _config  # noqa: F401 — load .env and strip empty OPENAI_BASE_URL

from raw_to_embedding.extractors.pdf_extractor import extract_pdf
from raw_to_embedding.main import _process_one
from raw_to_embedding.models import EmbeddingChunk
from raw_to_embedding.utils.file_utils import write_json
from raw_to_embedding.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

_STOP = frozenset(
    "the a an and or for of in on at to from with by is are was were be been being as it this that these those".split()
)


def _first_sentence(text: str, max_len: int = 320) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    m = re.match(r"^(.+?[.!?])(\s|$)", t, re.DOTALL)
    s = m.group(1).strip() if m else t[:max_len]
    return s[:max_len] + ("…" if len(s) >= max_len else "")


def _keywords(text: str, k: int = 10) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", (text or "").lower())
    words = [w for w in words if w not in _STOP and len(w) > 3]
    if not words:
        return []
    c = Counter(words)
    return [w for w, _ in c.most_common(k)]


def _map_chunk_type(content: str, content_type: str) -> str:
    low = (content or "").lower()
    if re.search(r"^\s*\|.*\|", content, re.MULTILINE) or "\t" in content[:500]:
        return "table"
    if "figure" in low or "fig." in low or "图" in content[:80]:
        return "figure_caption"
    if re.search(r"^\s*[-•*]\s", content, re.MULTILINE) and content.count("\n") > 2:
        return "list"
    if "appendix" in low:
        return "appendix"
    if content_type in ("academic_method", "academic_result", "academic_dataset"):
        return "mixed"
    return "paragraph"


def _quality_flag(content: str) -> str:
    t = content or ""
    if len(t.strip()) < 40:
        return "partial"
    bad = sum(1 for c in t if ord(c) > 0xFFFF or (ord(c) < 32 and c not in "\n\t\r"))
    if bad > len(t) * 0.05:
        return "noisy"
    if t.count("\ufffd") > 0 or (bad / max(1, len(t))) > 0.02:
        return "noisy"
    if len(t) > 8000:
        return "uncertain"
    return "clean"


def _needs_context(content: str) -> bool:
    t = (content or "").strip()
    return len(t) < 120 or t.count(".") == 0 and len(t) < 200


def _enrich_chunk(
    ec: EmbeddingChunk,
    *,
    pdf_name: str,
    section_path: str,
) -> dict[str, Any]:
    meta = dict(ec.metadata or {})
    ct = ec.content_type
    chunk_type = _map_chunk_type(ec.content, ct)
    qf = _quality_flag(ec.content)
    base = {
        "chunk_id": ec.chunk_id,
        "source_pdf": pdf_name,
        "section_title": ec.section or ec.title,
        "section_path": section_path,
        "page_start": meta.get("page_start"),
        "page_end": meta.get("page_end"),
        "chunk_text": ec.content,
        "chunk_summary": _first_sentence(ec.content),
        "keywords": _keywords(ec.content),
        "chunk_type": chunk_type,
        "needs_context": _needs_context(ec.content),
        "context_note": (
            "Short chunk or few sentence breaks; consider adjacent chunks or parent heading for retrieval."
            if _needs_context(ec.content)
            else ""
        ),
        "quality_flag": qf,
        "content_type_agent": ct,
        "embedding_text": ec.embedding_text,
        "unit_id": ec.unit_id,
        "embedding": None,
        "embedding_model": None,
    }
    return base


def _document_summary(raw_text: str, max_chars: int = 800) -> str:
    t = re.sub(r"\s+", " ", (raw_text or "").strip())
    return (t[:max_chars] + "…") if len(t) > max_chars else t


def _page_risks(pages: list) -> list[dict[str, Any]]:
    out = []
    for p in pages:
        txt = (p.text or "").strip()
        if len(txt) < 40:
            out.append(
                {
                    "page_number": p.page_number,
                    "reason": "very_little_extracted_text",
                    "char_count": len(txt),
                }
            )
    return out


def _l2_normalize_vectors(vecs: list[list[float]]) -> list[list[float]]:
    import numpy as np

    m = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    m = m / norms
    return m.tolist()


def _embed_openai(texts: list[str], model: str) -> tuple[list[list[float]], str]:
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    base = os.getenv("OPENAI_BASE_URL") or None
    client = OpenAI(api_key=key, base_url=base)
    # batch in chunks of 64
    all_vec: list[list[float]] = []
    step = 64
    for i in range(0, len(texts), step):
        batch = texts[i : i + step]
        r = client.embeddings.create(model=model, input=batch)
        # preserve order
        data = sorted(r.data, key=lambda d: d.index)
        all_vec.extend([d.embedding for d in data])
    return all_vec, model


def _embed_local(texts: list[str]) -> tuple[list[list[float]] | None, str | None]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, None
    mid = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    model = SentenceTransformer(mid)
    emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return emb.tolist(), mid


def run(
    pdf_dir: Path,
    out_dir: Path,
    *,
    max_files: int | None,
    save_intermediate: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if max_files is not None:
        pdfs = pdfs[: max_files]

    all_records: list[dict[str, Any]] = []
    doc_manifest: list[dict[str, Any]] = []
    errors: list[str] = []
    inter = out_dir / "intermediate"
    inter.mkdir(exist_ok=True)

    for pdf_path in pdfs:
        name = pdf_path.name
        try:
            raw = extract_pdf(pdf_path)
            summary = _document_summary(raw.raw_text)
            risks = _page_risks(raw.pages or [])
            _, _, chunks = _process_one(
                raw,
                input_path=pdf_path,
                input_url=None,
                save_intermediate=save_intermediate,
                intermediate_dir=inter,
            )
            for ec in chunks:
                sp = " / ".join(
                    x for x in (ec.title, ec.section) if x
                )
                rec = _enrich_chunk(ec, pdf_name=name, section_path=sp)
                rec["document_summary"] = summary
                all_records.append(rec)

            doc_manifest.append(
                {
                    "pdf": str(pdf_path),
                    "filename": name,
                    "chunks": len(chunks),
                    "document_summary": summary,
                    "page_count": len(raw.pages or []),
                    "risk_pages": risks,
                }
            )
        except Exception as exc:
            msg = f"{name}: {exc}"
            logger.exception(msg)
            errors.append(msg)
            doc_manifest.append(
                {
                    "pdf": str(pdf_path),
                    "filename": name,
                    "chunks": 0,
                    "error": str(exc),
                }
            )

    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    texts = [r["embedding_text"] for r in all_records]
    emb_meta = "none"
    if texts:
        try:
            vecs, emb_meta = _embed_openai(texts, embed_model)
            vecs = _l2_normalize_vectors(vecs)
            for r, v in zip(all_records, vecs, strict=True):
                r["embedding"] = v
                r["embedding_model"] = emb_meta
        except Exception as exc:
            logger.warning("OpenAI embedding failed: %s; trying local", exc)
            vecs, mid = _embed_local(texts)
            if vecs is not None:
                # sentence-transformers already uses normalize_embeddings=True; re-normalize for safety
                vecs = _l2_normalize_vectors(vecs)
                for r, v in zip(all_records, vecs, strict=True):
                    r["embedding"] = v
                    r["embedding_model"] = mid
                emb_meta = mid or "local"
            else:
                logger.warning("No embeddings produced (set OPENAI_API_KEY or pip install sentence-transformers).")

    bundle = {
        "segmentation_strategy": (
            "raw_to_embedding pipeline: PyMuPDF per-page text → "
            "heuristic candidate units (report-style / paper-style) → segment_units "
            "(LLM semantic split when enabled, else rule fallback) → segments_to_chunks "
            "by sentence groups; this script adds heuristic chunk_type/keywords/quality and optional embeddings."
        ),
        "embedding_note": (
            f"Embedding model: {emb_meta}."
            if all_records and all_records[0].get("embedding") is not None
            else "No embeddings: set OPENAI_API_KEY or pip install sentence-transformers."
        ),
        "documents": doc_manifest,
        "chunks": all_records,
        "errors": errors,
        "stats": {
            "pdf_count": len(pdfs),
            "chunk_count": len(all_records),
            "error_count": len(errors),
        },
    }
    write_json(out_dir / "ssl_pdf_embedding_corpus.json", bundle)

    strategy_md = "\n".join(
        [
            "# PDF segmentation and embedding strategy",
            "",
            "## Built-in agent flow",
            "",
            "1. **Extract**: PyMuPDF `get_text('text')` with per-page page ranges.",
            "2. **Candidate units**: heuristic split by headings/sections (see `candidate_unit_builder.py`).",
            "3. **Semantic segmentation**: `segment_units` — LLM for complex units when `OPENAI_API_KEY` is set, else sentence-group fallback.",
            "4. **Chunk packing**: `embedding_chunk_builder` packs sentence groups with a soft char cap; produces `embedding_text`.",
            "",
            "## Extra fields from this script",
            "",
            "- `chunk_summary` / `keywords` / `chunk_type` / `quality_flag` are **heuristic**, not LLM-invented.",
            "- Sparse table/figure pages are listed per document under `risk_pages`.",
            "",
            "## Embeddings",
            "",
            "- Preferred: `OPENAI_API_KEY` + `EMBEDDING_MODEL` (default text-embedding-3-small).",
            "- Fallback: `pip install sentence-transformers`, env `LOCAL_EMBEDDING_MODEL` (default all-MiniLM-L6-v2).",
            "",
        ]
    )
    (out_dir / "segmentation_strategy.md").write_text(strategy_md, encoding="utf-8")

    risks_out = {
        "documents": [
            {
                "filename": d.get("filename"),
                "risk_pages": d.get("risk_pages", []),
                "error": d.get("error"),
            }
            for d in doc_manifest
        ]
    }
    write_json(out_dir / "risks_and_anomalies.json", risks_out)

    logger.info(
        "Wrote %s chunks for %s PDFs to %s",
        len(all_records),
        len(pdfs),
        out_dir,
    )
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SSL+PDF batch → chunks + optional embeddings")
    p.add_argument(
        "--pdf-dir",
        type=Path,
        default=ROOT.parent / "SSL+PDF",
        help="Directory containing PDFs (default: ../SSL+PDF)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "ssl_pdf_bundle",
        help="Output directory",
    )
    p.add_argument("--max-files", type=int, default=None, help="Process only first N PDFs (debug)")
    p.add_argument("--save-intermediate", action="store_true", help="Save candidate_units / segments JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    pdf_dir = args.pdf_dir.resolve()
    if not pdf_dir.is_dir():
        logger.error("Not a directory: %s", pdf_dir)
        return 2
    return run(
        pdf_dir,
        args.out_dir.resolve(),
        max_files=args.max_files,
        save_intermediate=args.save_intermediate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
