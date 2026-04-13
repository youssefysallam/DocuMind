"""
Website data → cleaned chunks → embeddings → FAISS index.

Loads all JSON files from the websitedata/website_pages directory,
cleans HTML noise, performs semantic chunking (matching the PDF pipeline
parameters), generates embeddings, and builds a FAISS index.

Then merges with the existing PDF corpus to produce a unified
PDF+Website index.

Usage:
    python -m raw_to_embedding.corpus_build.website_to_chunks
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEBSITE_DIR = PROJECT_ROOT / "websitedata" / "website_pages"
PILOT_DIR = PROJECT_ROOT / "results" / "raw_to_embedding" / "pilot_runs"
OUTPUT_DIR = PROJECT_ROOT / "results" / "raw_to_embedding" / "workspace"

# ── Chunking parameters (match PDF pipeline) ──────────────────────────
MAX_CHUNK_CHARS = 700
MAX_SENTENCES = 5
OVERLAP_SENTENCES = 2
MIN_CHUNK_CHARS = 120
HARD_MIN = 200
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Nav / boilerplate noise patterns ─────────────────────────────────
_NAV_BLOCK = re.compile(
    r"^(?:UMass Boston|Request Info|Visit|Apply|Give|Home|"
    r"Menu|menu|Toggle (?:Main Menu|Search)|Skip to (?:Main Content|"
    r"Main Navigation|Search|Footer Links)|honeypot link|"
    r"Current Students|Parents & Families|Faculty & Staff|Alumni|"
    r"Send Email|Phone:\s*N/A|phone:\s*N/A|Directory|"
    r"Board Of Directors|Students|University Affiliates|"
    r"Projects & Initiatives|Research & Publications|"
    r"People|Recent News|Federal, State & Policy News|Publications|"
    r"2025|2024|2023|2022|2021|2020|Research Magazine Vol\.\s*\d+)\s*$",
    re.MULTILINE,
)
_MENU_LINE = re.compile(r"^\s*Menu\s*$", re.MULTILINE)
_NAV_BREADCRUMB = re.compile(
    r"^Home\n(?:Sustainable Solutions Lab\n)?(?:People\n)?(?:News\n)?"
    r"(?:Recent News\n)?.*?\s*\n\s*\n",
    re.MULTILINE,
)
_CONTACT_FOOTER = re.compile(
    r"Contact Us\s*\n\s*Sustainable Solutions Lab\n.*?(?:ssl@umb\.edu|$)",
    re.MULTILINE | re.DOTALL,
)
_OPEN_IMAGE = re.compile(r"\s*Open Image Modal\s*", re.MULTILINE)
_PHOTO_COURTESY = re.compile(r"Photo courtesy of .+$", re.MULTILINE)
_WHITESPACE = re.compile(r"\n{3,}")
_LINK_LINES = re.compile(r"^(?:LinkedIn|Website|Join .*|Explore .*|Browse .*|Download .*|Learn About .*)\s*$", re.MULTILINE)


def _clean_website_text(text: str) -> str:
    """Remove navigation, boilerplate, and UI noise from scraped website text."""
    text = _NAV_BLOCK.sub("", text)
    text = _MENU_LINE.sub("", text)
    text = _OPEN_IMAGE.sub("", text)
    text = _PHOTO_COURTESY.sub("", text)
    text = _CONTACT_FOOTER.sub("", text)
    text = _LINK_LINES.sub("", text)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if s in ("", "\r"):
            continue
        if re.match(r"^\s*\r?\s*$", s):
            continue
        cleaned.append(s)
    text = "\n".join(cleaned)
    text = re.sub(r"\r\n?", "\n", text)
    text = _WHITESPACE.sub("\n\n", text)
    return text.strip()


# ── Sentence splitting (same as PDF pipeline) ────────────────────────
_SENT_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z\"\u201c(])")


def split_sentences(text: str) -> list[str]:
    raw = _SENT_END.split(text)
    result: list[str] = []
    for seg in raw:
        seg = seg.strip()
        if not seg:
            continue
        if result and len(result[-1]) < 60 and not result[-1].endswith((".", "!", "?", ";")):
            result[-1] += " " + seg
        else:
            result.append(seg)
    return result


def pack_with_overlap(sentences: list[str], overlap: int = OVERLAP_SENTENCES) -> list[list[str]]:
    groups: list[list[str]] = []
    i = 0
    while i < len(sentences):
        grp: list[str] = []
        total = 0
        j = i
        while j < len(sentences) and len(grp) < MAX_SENTENCES:
            s = sentences[j]
            if total + len(s) > MAX_CHUNK_CHARS and grp:
                break
            grp.append(s)
            total += len(s) + 1
            j += 1
        if grp:
            groups.append(grp)
        if j >= len(sentences):
            break
        i = max(i + 1, j - overlap)
    return groups


# ── Chunk ID ─────────────────────────────────────────────────────────
def _chunk_id(source: str, idx: int) -> str:
    return hashlib.md5(f"{source}::{idx}".encode()).hexdigest()[:16]


# ── Quality flag ─────────────────────────────────────────────────────
def quality_flag(text: str) -> str:
    if len(text) < 80:
        return "too_short"
    if len(text) > 900:
        return "too_long"
    return "clean"


# ── Chunk type heuristic ─────────────────────────────────────────────
def chunk_type_heuristic(text: str, source_type: str) -> str:
    low = text[:200].lower()
    if any(k in low for k in ("biography", "bio:", "area of expertise", "degrees", "ph.d")):
        return "biography"
    if any(k in low for k in ("publication", "journal", "book chapter", "refereed")):
        return "publication_list"
    if source_type == "person_profile":
        return "person_profile"
    if source_type == "institute_presentation":
        return "presentation_slide"
    if any(k in low for k in ("project", "initiative", "program", "funded by")):
        return "project_description"
    if any(k in low for k in ("event", "forum", "workshop", "lecture")):
        return "event_info"
    return "paragraph"


# ── Section detection for long pages ─────────────────────────────────
_SECTION_HEADING = re.compile(
    r"^(?:Our Vision|Our Mission|What We Do|Our Staff|"
    r"External (?:Affiliates|Advisory Board)|Visiting Scholars|"
    r"Graduate Students and Interns|"
    r"Biography|Area of Expertise|Degrees|"
    r"Professional Publications & Contributions|"
    r"Doctoral dissertation|Refereed Journal Articles|"
    r"Book chapters|Manuscripts under preparation|"
    r"Additional Information|"
    r"Northeast Climate Justice Research Collaborative|"
    r"Climate Adaptation Forum|"
    r"Climate Careers Curricula Initiative.*|"
    r"Cape Cod Rail Resilience.*|"
    r"Climate Inequality and Integrative Resilience.*|"
    r"Explore our Publications|Annual Reports|Publications Archive|"
    r"Current Projects and Programs|Recently Completed)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _structural_split_website(text: str, source_type: str) -> list[dict[str, str]]:
    """Split cleaned website text into structural sections."""
    lines = text.split("\n")
    markers: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        s = line.strip()
        if _SECTION_HEADING.match(s):
            markers.append((i, s))

    if not markers:
        return [{"heading": "", "text": text}]

    blocks = []
    for mi, (line_idx, heading) in enumerate(markers):
        start = line_idx
        end = markers[mi + 1][0] if mi + 1 < len(markers) else len(lines)
        block_text = "\n".join(lines[start + 1:end]).strip()
        if block_text:
            blocks.append({"heading": heading, "text": block_text})

    preamble = "\n".join(lines[:markers[0][0]]).strip()
    if preamble and len(preamble) > 50:
        blocks.insert(0, {"heading": "Overview", "text": preamble})

    return blocks if blocks else [{"heading": "", "text": text}]


# ── Person-based splitting for people pages ──────────────────────────
_PERSON_NAME_PATTERN = re.compile(
    r"^([A-Z][a-zA-ZáéíóúñÁÉÍÓÚÑ\.\-' ]+(?:\([^)]+\))?)\s*$"
)


def _split_people_page(text: str) -> list[dict[str, str]]:
    """Split pages listing multiple people into per-person blocks."""
    lines = text.split("\n")
    blocks: list[dict[str, str]] = []
    current_name = ""
    current_lines: list[str] = []
    preamble_lines: list[str] = []

    for line in lines:
        s = line.strip()
        m = _PERSON_NAME_PATTERN.match(s)
        if m and len(s) > 5 and len(s) < 60:
            candidate = m.group(1).strip()
            if candidate.istitle() or candidate[0].isupper():
                if current_name:
                    body = "\n".join(current_lines).strip()
                    if body:
                        blocks.append({"heading": current_name, "text": body})
                elif current_lines:
                    preamble_lines = current_lines[:]
                current_name = candidate
                current_lines = []
                continue
        current_lines.append(s)

    if current_name and current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            blocks.append({"heading": current_name, "text": body})

    if preamble_lines:
        preamble = "\n".join(preamble_lines).strip()
        if preamble and len(preamble) > 50:
            blocks.insert(0, {"heading": "Overview", "text": preamble})

    return blocks if blocks else [{"heading": "", "text": text}]


# ── Presentation slide chunking ──────────────────────────────────────
def _chunk_presentation(data: dict) -> list[dict[str, Any]]:
    """Chunk a presentation by grouping related slides."""
    slides = data.get("slides", [])
    if not slides:
        return []

    chunks = []
    source = data.get("url", "presentation")
    title = data.get("title", "SSL Presentation")

    slide_groups = []
    current_group: list[dict] = []
    current_len = 0

    for slide in slides:
        slide_text = slide.get("text", "").strip()
        if not slide_text:
            continue
        if current_len + len(slide_text) > MAX_CHUNK_CHARS and current_group:
            slide_groups.append(current_group)
            current_group = []
            current_len = 0
        current_group.append(slide)
        current_len += len(slide_text) + 10

    if current_group:
        slide_groups.append(current_group)

    for gi, group in enumerate(slide_groups):
        slide_nums = [s["slide_number"] for s in group]
        combined = "\n\n".join(s["text"] for s in group)
        first_line = group[0]["text"].split("\n")[0].strip()
        section = first_line if len(first_line) < 60 else f"Slides {slide_nums[0]}-{slide_nums[-1]}"

        chunk = {
            "chunk_id": _chunk_id(f"presentation_{title}", gi),
            "source_url": source,
            "source_type": "website",
            "source_file": "ssl_overview_presentation_2026.json",
            "section_title": section,
            "section_path": f"{title} > {section}",
            "page_start": slide_nums[0],
            "page_end": slide_nums[-1],
            "chunk_text": combined,
            "embedding_text": combined,
            "chunk_type": "presentation_slide",
            "quality_flag": quality_flag(combined),
            "chunk_index": gi,
        }
        chunks.append(chunk)

    return chunks


# ── Generic page chunking ────────────────────────────────────────────
def _chunk_page(data: dict, filename: str) -> list[dict[str, Any]]:
    """Chunk a generic website page JSON."""
    text = data.get("text", "")
    if not text or len(text.strip()) < 50:
        return []

    source_url = data.get("url", "")
    source_type = data.get("source_type", "website_page")
    title = data.get("title", filename)
    page_title = re.sub(r"\s*[-–|]\s*UMass Boston\s*$", "", title).strip()

    text = _clean_website_text(text)
    if len(text) < 50:
        return []

    is_people_page = any(k in filename.lower() for k in (
        "board-of-directors", "students", "ssl_people.",
    ))
    is_profile = source_type == "person_profile" or filename.startswith("profile_")

    if is_profile:
        blocks = _structural_split_website(text, source_type)
    elif is_people_page:
        blocks = _split_people_page(text)
    else:
        blocks = _structural_split_website(text, source_type)

    chunks: list[dict[str, Any]] = []
    idx = 0

    for block in blocks:
        heading = block["heading"]
        block_text = block["text"].strip()
        if not block_text or len(block_text) < 30:
            continue

        if len(block_text) <= MAX_CHUNK_CHARS:
            emb = f"{heading}\n{block_text}" if heading else block_text
            chunks.append({
                "chunk_id": _chunk_id(f"web_{filename}", idx),
                "source_url": source_url,
                "source_type": "website",
                "source_file": filename,
                "section_title": heading or page_title,
                "section_path": f"{page_title} > {heading}" if heading else page_title,
                "page_start": 0,
                "page_end": 0,
                "chunk_text": block_text,
                "embedding_text": emb,
                "chunk_type": chunk_type_heuristic(block_text, source_type),
                "quality_flag": quality_flag(block_text),
                "chunk_index": idx,
            })
            idx += 1
        else:
            sentences = split_sentences(block_text)
            groups = pack_with_overlap(sentences)
            for gi, grp in enumerate(groups):
                content = " ".join(grp).strip()
                if not content:
                    continue
                section_label = heading or page_title
                if len(groups) > 1:
                    section_label = f"{section_label} (part {gi + 1})"
                emb = f"{heading}\n{content}" if heading else content
                chunks.append({
                    "chunk_id": _chunk_id(f"web_{filename}", idx),
                    "source_url": source_url,
                    "source_type": "website",
                    "source_file": filename,
                    "section_title": section_label,
                    "section_path": f"{page_title} > {section_label}" if heading else page_title,
                    "page_start": 0,
                    "page_end": 0,
                    "chunk_text": content,
                    "embedding_text": emb,
                    "chunk_type": chunk_type_heuristic(content, source_type),
                    "quality_flag": quality_flag(content),
                    "chunk_index": idx,
                })
                idx += 1

    # Merge small chunks
    merged: list[dict[str, Any]] = []
    for c in chunks:
        if (
            merged
            and len(c["chunk_text"]) < HARD_MIN
            and merged[-1]["section_title"].split(" (part")[0] == c["section_title"].split(" (part")[0]
        ):
            merged[-1]["chunk_text"] += " " + c["chunk_text"]
            merged[-1]["embedding_text"] = merged[-1]["chunk_text"]
        else:
            merged.append(c)

    # Second pass: merge any remaining tiny chunks
    final: list[dict[str, Any]] = []
    for c in merged:
        if final and len(c["chunk_text"]) < HARD_MIN:
            combined = len(final[-1]["chunk_text"]) + len(c["chunk_text"]) + 1
            if combined <= MAX_CHUNK_CHARS + 200:
                final[-1]["chunk_text"] += " " + c["chunk_text"]
                final[-1]["embedding_text"] = final[-1]["chunk_text"]
                continue
        final.append(c)

    for i, c in enumerate(final):
        c["chunk_index"] = i
        c["chunk_id"] = _chunk_id(f"web_{filename}", i)
        c["quality_flag"] = quality_flag(c["chunk_text"])

    return final


# ── Main pipeline ────────────────────────────────────────────────────
def run_website_pipeline() -> dict[str, Any]:
    """Process all website JSON files → chunks → embeddings → FAISS index."""
    sys.stdout.reconfigure(encoding="utf-8")

    json_files = sorted(WEBSITE_DIR.glob("*.json"))
    skip_files = {
        "ssl_people_university-affiliates.json",
        "directory__department_sustainable_solutions_lab.json",
    }

    print(f"Found {len(json_files)} JSON files in {WEBSITE_DIR}")

    all_chunks: list[dict[str, Any]] = []
    per_file_stats: list[dict[str, Any]] = []

    for jf in json_files:
        if jf.name in skip_files:
            print(f"  [SKIP] {jf.name} (empty placeholder)")
            continue

        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_type = data.get("source_type", "")

        if source_type == "institute_presentation":
            chunks = _chunk_presentation(data)
        else:
            chunks = _chunk_page(data, jf.name)

        if not chunks:
            print(f"  [SKIP] {jf.name} (no usable content)")
            continue

        lengths = [len(c["chunk_text"]) for c in chunks]
        stats = {
            "file": jf.name,
            "chunks": len(chunks),
            "avg_len": int(np.mean(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }
        per_file_stats.append(stats)
        print(f"  [OK] {jf.name}: {len(chunks)} chunks, "
              f"avg={stats['avg_len']}, range=[{stats['min_len']}, {stats['max_len']}]")

        all_chunks.extend(chunks)

    print(f"\nTotal website chunks: {len(all_chunks)}")

    # ── Generate embeddings ──────────────────────────────────────────
    print(f"\nLoading {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["embedding_text"] for c in all_chunks]
    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Attach embedding to each chunk
    for i, c in enumerate(all_chunks):
        c["embedding_model"] = EMBED_MODEL

    # ── Save website artifacts ───────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = OUTPUT_DIR / "website_index_metadata.json"
    meta_records = []
    for c in all_chunks:
        rec = {k: v for k, v in c.items() if k != "embedding"}
        meta_records.append(rec)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_records, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metadata: {meta_path} ({len(meta_records)} records)")

    chunks_with_emb_path = OUTPUT_DIR / "website_chunks_with_embeddings.json"
    emb_records = []
    for i, c in enumerate(all_chunks):
        rec = dict(c)
        rec["embedding"] = embeddings[i].tolist()
        emb_records.append(rec)
    with open(chunks_with_emb_path, "w", encoding="utf-8") as f:
        json.dump(emb_records, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks+embeddings: {chunks_with_emb_path}")

    emb_npy_path = OUTPUT_DIR / "website_embeddings.npy"
    np.save(emb_npy_path, embeddings)
    print(f"Saved embeddings array: {emb_npy_path} ({embeddings.shape})")

    # ── Build website-only FAISS index ───────────────────────────────
    dim = embeddings.shape[1]
    web_index = faiss.IndexFlatIP(dim)
    web_index.add(embeddings)
    web_faiss_path = OUTPUT_DIR / "website_index.faiss"
    faiss.write_index(web_index, str(web_faiss_path))
    print(f"Saved FAISS index: {web_faiss_path} ({web_index.ntotal} vectors)")

    # ── Merge with PDF corpus ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MERGING PDF + WEBSITE CORPUS")
    print("=" * 60)

    pdf_chunks_files = sorted(PILOT_DIR.glob("pilot_chunks_*.json"))
    pdf_all_chunks: list[dict[str, Any]] = []
    pdf_embeddings_list: list[np.ndarray] = []

    for pcf in pdf_chunks_files:
        with open(pcf, "r", encoding="utf-8") as f:
            pdf_chunks = json.load(f)
        for chunk in pdf_chunks:
            emb_vec = chunk.pop("embedding", None)
            chunk["source_type"] = "pdf"
            if emb_vec:
                pdf_embeddings_list.append(np.array(emb_vec, dtype=np.float32))
                pdf_all_chunks.append(chunk)

    print(f"PDF chunks loaded: {len(pdf_all_chunks)}")
    print(f"Website chunks: {len(all_chunks)}")

    if pdf_all_chunks:
        pdf_embeddings = np.array(pdf_embeddings_list, dtype=np.float32)
        unified_embeddings = np.vstack([pdf_embeddings, embeddings])
    else:
        unified_embeddings = embeddings

    unified_meta: list[dict[str, Any]] = []
    for c in pdf_all_chunks:
        rec = {k: v for k, v in c.items()}
        unified_meta.append(rec)
    for c in all_chunks:
        rec = {k: v for k, v in c.items()}
        unified_meta.append(rec)

    total = len(unified_meta)
    print(f"Unified corpus: {total} chunks ({len(pdf_all_chunks)} PDF + {len(all_chunks)} website)")

    # Save unified metadata
    unified_meta_path = OUTPUT_DIR / "unified_index_metadata.json"
    with open(unified_meta_path, "w", encoding="utf-8") as f:
        json.dump(unified_meta, f, ensure_ascii=False, indent=2)
    print(f"Saved: {unified_meta_path}")

    # Build unified FAISS index
    unified_index = faiss.IndexFlatIP(unified_embeddings.shape[1])
    unified_index.add(unified_embeddings)
    unified_faiss_path = OUTPUT_DIR / "unified_index.faiss"
    faiss.write_index(unified_index, str(unified_faiss_path))
    print(f"Saved: {unified_faiss_path} ({unified_index.ntotal} vectors)")

    np.save(OUTPUT_DIR / "unified_embeddings.npy", unified_embeddings)

    # ── Smoke test ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RETRIEVAL SMOKE TEST (Website-specific queries)")
    print("=" * 60)

    queries = [
        "Who is B.R. Balachandran and what is his role at SSL?",
        "What is the Climate Careers Curricula Initiative C3I?",
        "Who are the members of SSL's External Advisory Board?",
        "What is the Northeast Climate Justice Research Collaborative?",
        "What are SSL's current research projects?",
        "What is the Cape Cod Rail Resilience Project?",
        "Who are the graduate students working at SSL?",
        "What is SSL's vision and mission?",
        "What recent events has SSL organized in 2025-2026?",
        "What is the CLIIR initiative and why was it suspended?",
        "What is Rajini Srikanth's research expertise?",
        "What is Paul Kirshen's work on climate resilience?",
        "How does SSL collaborate with Just A Start?",
        "What is the Lumbee Tribe climate adaptation project?",
    ]

    q_embs = model.encode(queries, normalize_embeddings=True)
    D, I = unified_index.search(np.array(q_embs, dtype=np.float32), 5)

    hits = 0
    total_q = len(queries)
    results: list[dict] = []

    for qi, query in enumerate(queries):
        top_indices = I[qi]
        top_scores = D[qi]
        top_chunks = []
        found_website = False

        for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
            meta = unified_meta[idx]
            src = meta.get("source_type", "unknown")
            source = meta.get("source_file", meta.get("source_pdf", "?"))
            top_chunks.append({
                "rank": rank + 1,
                "source_type": src,
                "source": source,
                "section": meta.get("section_title", ""),
                "score": round(float(score), 4),
                "snippet": meta.get("chunk_text", "")[:100],
            })
            if src == "website":
                found_website = True

        if found_website:
            hits += 1

        results.append({
            "query": query,
            "has_website_hit": found_website,
            "top_results": top_chunks,
        })

    hit_rate = hits / total_q
    print(f"\nWebsite hit@5: {hits}/{total_q} = {hit_rate:.1%}")
    print()

    for r in results:
        status = "✓" if r["has_website_hit"] else "✗"
        print(f"  {status} Q: {r['query'][:70]}")
        for t in r["top_results"][:3]:
            print(f"      [{t['rank']}] {t['source_type']:7s} | {t['source'][:40]:40s} | {t['score']:.3f} | {t['snippet'][:60]}")
        print()

    # ── Report ───────────────────────────────────────────────────────
    report = {
        "website_chunks_total": len(all_chunks),
        "pdf_chunks_total": len(pdf_all_chunks),
        "unified_total": total,
        "per_file_stats": per_file_stats,
        "smoke_test": {
            "total_queries": total_q,
            "website_hit_at_5": hits,
            "hit_rate": round(hit_rate, 4),
            "results": results,
        },
        "artifacts": {
            "website_metadata": str(meta_path),
            "website_chunks": str(chunks_with_emb_path),
            "website_embeddings": str(emb_npy_path),
            "website_faiss": str(web_faiss_path),
            "unified_metadata": str(unified_meta_path),
            "unified_faiss": str(unified_faiss_path),
        },
    }

    report_path = OUTPUT_DIR / "website_pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull report: {report_path}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_website_pipeline()
