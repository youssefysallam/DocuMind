# SSL RAG Corpus Bundle — Final Report

**Generated:** 2026-03-28  
**Bundle location:** `rag_system/data/final_corpus_bundle/`

---

## 1. Overview

This bundle contains the complete, production-ready corpus for the Sustainable Solutions Lab (SSL) RAG question-answering system. It combines two data sources — **17 PDF research documents** and **21 website pages** — into a unified retrieval index.

**Purpose:** Provide a single, self-contained directory that any downstream retrieval, reranking, or evaluation pipeline can load directly, without needing to understand the original processing history.

**Key numbers:**
- **5,227** total chunks (4,959 PDF + 268 website)
- **5,227** embedding vectors (384-dimensional, `all-MiniLM-L6-v2`)
- **0** duplicate chunk IDs
- All counts verified: metadata records = embedding vectors = FAISS index entries

---

## 2. Data Sources

### 2.1 PDF Corpus

| # | PDF Filename | Chunks |
|---|-------------|--------|
| 1 | Community-Led Climate Preparedness and Resilience in Boston_ New.pdf | ~173 |
| 2 | Connecting for Equitable Climate Adaptation_ Mapping Stakeholder.pdf | ~87 |
| 3 | Critical approaches to climate-induced migration research and sol.pdf | ~123 |
| 4 | Executive Summary_Feasibility of Harbor-wide Barrier Systems_ Pre.pdf | ~126 |
| 5 | Executive Summary_Financing Climate Resilience_ Mobilizing Resour.pdf | ~97 |
| 6 | Executive Summary_Governance for a Changing Climate_ Adapting Bos.pdf | ~85 |
| 7 | Feasibility of Harbor-wide Barrier Systems_ Preliminary Analysis.pdf | ~763 |
| 8 | Financing Climate Resilience_ Mobilizing Resources and Incentives.pdf | ~563 |
| 9 | Governance for a Changing Climate_ Adapting Boston_s Built Enviro.pdf | ~944 |
| 10 | Learning from the Massachusetts Municipal Vulnerability Preparedn.pdf | ~492 |
| 11 | Oportunidad en la Complejidad_Recomendaciones para una Resilienci.pdf | ~479 |
| 12 | Opportunity in the Complexity_ Recommendations for Equitable Clim.pdf | ~419 |
| 13 | UMB-SSL-2022-Annual_Report.pdf | ~62 |
| 14 | UMB-SSL-2025-Impact_Report.pdf | ~57 |
| 15 | Views that Matter_ Race and Opinions on Climate Change of Boston.pdf | ~162 |
| 16 | Voices that Matter_ Boston Area Residents of Color Discuss Climat.pdf | ~193 |
| 17 | Who Counts in Climate Resilience_ Transient Populations and Clima.pdf | ~140 |

**Source directory:** `c:\rawdata2embedding\SSL+PDF\`  
**Total PDF chunks:** 4,959  
**Processing pipeline:** `rag_system/pilot_single_pdf.py` (semantic chunking with structural splitting, sentence overlap, three-pass merging, boilerplate removal)

### 2.2 Website Corpus

| # | Source File | Type | Chunks |
|---|-----------|------|--------|
| 1 | ssl.json | Homepage | 7 |
| 2 | ssl_people.json | Staff & affiliates | 10 |
| 3 | ssl_projects.json | Projects & initiatives | 18 |
| 4 | ssl_research.json | Publications listing | 14 |
| 5 | ssl_overview_presentation_2026.json | 21-slide presentation (Mar 2026) | 13 |
| 6 | ssl_people_board-of-directors.json | Advisory board bios | 23 |
| 7 | ssl_people_students.json | Graduate students & interns | 29 |
| 8 | news_recent-news_beyond-survival-Sustainable-Solutions-Labs-Drive.json | News article | 16 |
| 9 | profile_directory_brbalachandran.json | Personal profile | 16 |
| 10 | profile_directory_rajinisrikanth.json | Personal profile | 40 |
| 11 | profile_directory_paulkirshen.json | Personal profile | 21 |
| 12 | profile_directory_rosalynnegron.json | Personal profile | 18 |
| 13 | profile_directory_antonioraciti.json | Personal profile | 14 |
| 14 | profile_directory_evanstewart.json | Personal profile | 10 |
| 15 | profile_directory_paulwatanabe.json | Personal profile | 5 |
| 16 | profile_directory_elisaguerrero.json | Personal profile | 5 |
| 17 | profile_directory_gabrielaboscio.json | Personal profile | 5 |
| 18 | profile_directory_ceciliaidikakalu.json | Personal profile | 1 |
| 19 | profile_directory_cedricwoods.json | Personal profile | 1 |
| 20 | profile_directory_johnnaflahive001.json | Personal profile | 1 |
| 21 | profile_directory_patriciobelloy001.json | Personal profile | 1 |

**Source domain:** `umb.edu` (SSL pages + UMB directory profiles)  
**Source directory:** `c:\rawdata2embedding\websitedata\website_pages\`  
**Total website chunks:** 268  
**Processing pipeline:** `rag_system/website_to_chunks.py`

**Skipped files (not included):**
- `ssl_people_university-affiliates.json` — empty placeholder page ("Coming soon!")
- `directory__department_sustainable_solutions_lab.json` — pure UMB directory navigation noise

**Failed to scrape (404):**
- `robertchen` — URL not found
- `ellenmdouglas` — URL not found
- `lorenaestradademartinez` — URL not found

---

## 3. File Inventory

### 3.1 `pdf/` — 68 files

For each of the 17 PDFs, there are 4 file types:

| Pattern | Type | Format | Description |
|---------|------|--------|-------------|
| `pilot_chunks_{name}.json` | chunks | JSON array | Full chunks with embedded vectors (384-dim float list per record) |
| `pilot_embeddings_{name}.npy` | embeddings | NumPy `.npy` | Embedding matrix (N × 384, float32) |
| `pilot_index_metadata_{name}.json` | metadata | JSON array | Chunk metadata without embedding vectors |
| `pilot_index_{name}.faiss` | index | FAISS `IndexFlatIP` | Per-PDF FAISS index for inner-product search |

**Total files:** 17 × 4 = 68

### 3.2 `website/` — 4 files

| File | Type | Format | Description |
|------|------|--------|-------------|
| `website_chunks_with_embeddings.json` | chunks | JSON array | 268 chunks, each with 384-dim embedding vector |
| `website_index_metadata.json` | metadata | JSON array | 268 records, metadata only (no vectors) |
| `website_embeddings.npy` | embeddings | NumPy `.npy` | (268, 384) float32 matrix |
| `website_index.faiss` | index | FAISS `IndexFlatIP` | Website-only FAISS index |

### 3.3 `merged/` — 3 files

| File | Type | Format | Description |
|------|------|--------|-------------|
| `unified_index_metadata.json` | metadata | JSON array | 5,227 records (4,959 PDF + 268 website), no vectors |
| `unified_embeddings.npy` | embeddings | NumPy `.npy` | (5,227, 384) float32 matrix |
| `unified_index.faiss` | index | FAISS `IndexFlatIP` | Complete unified FAISS index |

**Important:** The row order of `unified_index_metadata.json` corresponds exactly to `unified_embeddings.npy` and the vector IDs in `unified_index.faiss`. Record `i` in the metadata maps to row `i` in the embeddings and vector ID `i` in FAISS.

### 3.4 `REPORT/` — provenance files

| File | Description |
|------|-------------|
| `FINAL_CORPUS_REPORT.md` | This document |
| `full_pipeline_report.json` | PDF-only full pipeline run report (18 PDFs processed, smoke test results) |
| `per_pdf_stats.json` | Per-PDF chunking statistics from full pipeline |
| `website_pipeline_report.json` | Website pipeline report (per-file stats, smoke test results) |
| `audit_results.json` | Chunking quality audit results (C1–C7 criteria) |

---

## 4. Chunk Schema

### 4.1 Common fields (present in both PDF and website chunks)

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | 16-char MD5-based unique identifier |
| `chunk_index` | int | Zero-based position within its source document |
| `chunk_text` | string | Raw text content of the chunk |
| `embedding_text` | string | Text used for embedding generation (may include section heading as prefix) |
| `section_title` | string | Heading of the section this chunk belongs to |
| `section_path` | string | Hierarchical path (e.g., "Report > Section > Subsection") |
| `page_start` | int | Starting page (1-based for PDFs, 0 for website, slide number for presentations) |
| `page_end` | int | Ending page |
| `chunk_type` | string | Semantic category (see values below) |
| `quality_flag` | string | `"clean"`, `"partial"`, `"too_short"`, or `"too_long"` |
| `embedding_model` | string | Always `"all-MiniLM-L6-v2"` |
| `source_type` | string | `"pdf"` or `"website"` (in unified metadata only) |

### 4.2 PDF-only fields

| Field | Type | Description |
|-------|------|-------------|
| `source_pdf` | string | Original PDF filename (e.g., `"UMB-SSL-2025-Impact_Report.pdf"`) |

### 4.3 Website-only fields

| Field | Type | Description |
|-------|------|-------------|
| `source_url` | string | Original page URL (e.g., `"https://www.umb.edu/ssl/"`) |
| `source_file` | string | JSON filename in websitedata (e.g., `"ssl.json"`) |

### 4.4 `chunk_type` values

**PDF chunk types:**
| Value | Count | Description |
|-------|-------|-------------|
| `paragraph` | 4,427 | Standard body text |
| `list` | 333 | Bulleted or numbered list content |
| `table` | 94 | Table or structured data |
| `figure_caption` | 63 | Figure caption or description |
| `metadata` | 42 | Document metadata (title pages, headers) |

**Website chunk types:**
| Value | Count | Description |
|-------|-------|-------------|
| `person_profile` | 127 | Individual person bio/profile |
| `paragraph` | 66 | General content |
| `project_description` | 35 | Project/initiative description |
| `publication_list` | 13 | Publication entries |
| `presentation_slide` | 13 | Presentation slide content |
| `biography` | 8 | Detailed biography section |
| `event_info` | 6 | Event listing or description |

---

## 5. Embedding Information

| Property | Value |
|----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimension | 384 |
| Normalization | `normalize_embeddings=True` (unit vectors) |
| Similarity metric | Inner product (cosine similarity, since vectors are L2-normalized) |
| Total vectors | 5,227 |
| FAISS index type | `IndexFlatIP` (exact inner-product search) |
| dtype | float32 |

### Embedding storage formats

1. **Per-PDF `.npy` files** — each is an (N, 384) float32 array for that PDF
2. **`website_embeddings.npy`** — (268, 384) float32 for all website chunks
3. **`unified_embeddings.npy`** — (5,227, 384) float32 for the complete corpus
4. **`pilot_chunks_*.json`** and **`website_chunks_with_embeddings.json`** — each JSON record contains an `"embedding"` field (list of 384 floats)

---

## 6. Corpus Statistics

| Metric | PDF | Website | Merged |
|--------|-----|---------|--------|
| Total chunks | 4,959 | 268 | 5,227 |
| Source documents | 17 PDFs | 21 web pages | 38 sources |
| Avg chunk length (chars) | 560 | 489 | 556 |
| Min chunk length | 32 | 88 | 32 |
| Max chunk length | 4,339 | 1,027 | 4,339 |
| Embedding vectors | 4,959 | 268 | 5,227 |
| Embedding dimension | 384 | 384 | 384 |
| Duplicate chunk_ids | 0 | 0 | 0 |
| Quality: `clean` | 4,958 | 267 | 5,225 |
| Quality: `partial` | 1 | 0 | 1 |
| Quality: `too_long` | 0 | 1 | 1 |

---

## 7. Data Provenance

### 7.1 PDF pipeline

```
Source PDFs (SSL+PDF/)
    → pdf_extractor.py (PyMuPDF text extraction)
    → pilot_single_pdf.py:
        ├── Page-level cleaning (headers, footers, page numbers, journal boilerplate)
        ├── Two-pass structural splitting (heading detection + reference boundaries)
        ├── Sentence-level chunking (max 700 chars, max 5 sentences, 2-sentence overlap)
        ├── Three-pass merge (same-section → cross-section → forward-merge)
        ├── Reference chunk filtering
        └── Embedding (all-MiniLM-L6-v2, normalize=True)
    → Per-PDF artifacts (pilot_chunks_*, pilot_embeddings_*, pilot_index_*)
```

### 7.2 Website pipeline

```
Source: umb.edu SSL pages + UMB directory profiles
    → Team member scraper (scrape_website_pages.py, scrape_profiles.py)
    → 23 JSON files in websitedata/website_pages/
    → website_to_chunks.py:
        ├── HTML noise removal (nav, menus, footers, breadcrumbs)
        ├── Structure-aware splitting (section headings, per-person blocks, slide groups)
        ├── Sentence-level chunking (same parameters as PDF)
        ├── Two-pass merge for small chunks
        └── Embedding (all-MiniLM-L6-v2, normalize=True)
    → Website artifacts (website_chunks_*, website_embeddings.npy, website_index.faiss)
```

### 7.3 Merging

```
PDF chunks (4,959) + Website chunks (268)
    → Concatenated metadata (unified_index_metadata.json)
    → Stacked embeddings (unified_embeddings.npy)
    → Combined FAISS index (unified_index.faiss)
    
Row ordering: PDF chunks [0..4958] followed by website chunks [4959..5226]
```

### 7.4 Superseded data (NOT included in this bundle)

| Location | Content | Why superseded |
|----------|---------|---------------|
| `data/index.faiss` + `data/index_metadata.json` | Legacy v1 pipeline (1,649 chunks) | Pre-pilot; avg chunk 1,276 chars, max 66,697 chars; no structural splitting |
| `data/full_corpus/` | Intermediate PDF-only corpus (5,121 chunks) | Created by `_run_full_pipeline.py` before final per-PDF merging optimization; chunk count differs from final (5,121 vs 4,959) |

---

## 8. Known Issues

### 8.1 Field schema divergence between PDF and website

The unified metadata uses **different source identifier fields** depending on `source_type`:
- PDF records have `source_pdf` (no `source_url`, no `source_file`)
- Website records have `source_url` and `source_file` (no `source_pdf`)

**Impact:** Downstream code must check `source_type` and use the appropriate field. A unified `source` field was not created to avoid modifying data.

### 8.2 One chunk flagged `too_long` (website)

One website chunk from `profile_directory_rosalynnegron.json` is 1,027 characters (exceeds the 900-char threshold for `too_long`). This is a single long biography paragraph that resists further splitting without breaking mid-sentence.

### 8.3 One chunk flagged `partial` (PDF)

One PDF chunk has `quality_flag: "partial"`, indicating it may have incomplete content (e.g., at a page boundary).

### 8.4 PDF chunks with very long text

The maximum PDF chunk length is 4,339 characters (well above the 700-char target). These are table-heavy or figure-heavy sections where structural content could not be cleanly split without losing meaning.

### 8.5 Missing individual profile pages (3 people)

Three UMB directory pages returned 404 and are not in the corpus:
- Robert Chen (`/directory/robertchen/`)
- Ellen M. Douglas (`/directory/ellenmdouglas/`)
- Lorena Estrada de Martinez (`/directory/lorenaestradademartinez/`)

These people are still mentioned in other documents (PDFs, team pages) but lack dedicated profile chunks.

### 8.6 `page_start` / `page_end` = 0 for website chunks

Website chunks use `0` for both `page_start` and `page_end` since web pages don't have page numbers. Presentation slides use slide numbers instead.

---

## 9. Readiness for Next Step

### Ready for:

| Capability | Status | Notes |
|-----------|--------|-------|
| Retrieval experiments | **Ready** | Load `merged/unified_index.faiss` + `merged/unified_index_metadata.json` |
| Semantic search | **Ready** | Encode query with `all-MiniLM-L6-v2` (normalize=True), search FAISS |
| Reranking | **Ready** | Chunk text available in metadata for cross-encoder reranking |
| Evaluation / test set generation | **Ready** | Chunks contain section_title, source info for answer verification |
| Source-specific retrieval | **Ready** | Use `pdf/` or `website/` subdirectories for source-isolated experiments |
| RAG QA system | **Ready** | `chunk_text` for generation context, `embedding_text` for retrieval |

### Quick-start code

```python
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

BUNDLE = "rag_system/data/final_corpus_bundle/merged"

# Load
with open(f"{BUNDLE}/unified_index_metadata.json") as f:
    metadata = json.load(f)
index = faiss.read_index(f"{BUNDLE}/unified_index.faiss")

# Query
model = SentenceTransformer("all-MiniLM-L6-v2")
q_emb = model.encode(["What is SSL's mission?"], normalize_embeddings=True)
D, I = index.search(q_emb.astype("float32"), k=5)

for rank, (idx, score) in enumerate(zip(I[0], D[0])):
    chunk = metadata[idx]
    print(f"[{rank+1}] score={score:.3f} | {chunk['source_type']} | {chunk['section_title']}")
    print(f"    {chunk['chunk_text'][:120]}...")
```

---

## Appendix: Bundle directory tree

```
final_corpus_bundle/
├── pdf/                                    # 68 files
│   ├── pilot_chunks_{pdf_name}.json        # × 17 — chunks with embedding vectors
│   ├── pilot_embeddings_{pdf_name}.npy     # × 17 — embedding matrices
│   ├── pilot_index_metadata_{pdf_name}.json# × 17 — metadata without vectors
│   └── pilot_index_{pdf_name}.faiss        # × 17 — per-PDF FAISS indices
├── website/                                # 4 files
│   ├── website_chunks_with_embeddings.json # 268 chunks with embedding vectors
│   ├── website_index_metadata.json         # 268 records metadata only
│   ├── website_embeddings.npy              # (268, 384) float32
│   └── website_index.faiss                 # website-only FAISS index
├── merged/                                 # 3 files
│   ├── unified_index_metadata.json         # 5,227 records (PDF + website)
│   ├── unified_embeddings.npy              # (5,227, 384) float32
│   └── unified_index.faiss                 # complete FAISS index
└── REPORT/                                 # documentation
    ├── FINAL_CORPUS_REPORT.md              # this file
    ├── full_pipeline_report.json           # PDF pipeline run report
    ├── per_pdf_stats.json                  # per-PDF chunking stats
    ├── website_pipeline_report.json        # website pipeline report
    └── audit_results.json                  # chunking quality audit
```
