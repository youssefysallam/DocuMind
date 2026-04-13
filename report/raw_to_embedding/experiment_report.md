# Experiment Report: Raw-to-Embedding Pipeline (SSL Corpus)

## 1. Introduction

This report documents the **final** embedding corpus construction workflow for SSL research products: **PDF reports** from `SSL+PDF/` and **website** text from `websitedata/website_pages/`. Intermediate experimental trees and duplicate bundles were removed in favor of a single **canonical artifact**: `data/final_corpus_bundle/`.

## 2. Pipeline Objective

Produce a **unified** retrieval index:

1. **Semantic chunks** for each PDF with structural awareness (sections, lists, boilerplate removal, controlled overlap).
2. **Semantic chunks** for each website JSON page with HTML/boilerplate cleaning.
3. **Embeddings** with a consistent sentence-transformer model.
4. A **merged FAISS** index and aligned metadata for downstream RAG.

## 3. Data Processing Workflow

**PDF path (`corpus_build/pilot_single_pdf.py`).** PDFs are parsed to text, cleaned, split using sentence-aware limits (character cap, max sentences, overlap), merged for micro-fragments, and enriched with metadata (source PDF, section, page span, chunk type).

**Website path (`corpus_build/website_to_chunks.py`).** JSON pages are loaded, denoised (navigation, modal strings, repeated menu lines), chunked with parameters matched to the PDF pipeline, embedded, indexed, and **merged** with the PDF chunk pool to produce `unified_index_metadata.json`, `unified_embeddings.npy`, and `unified_index.faiss` in the bundle’s `merged/` directory.

**Generic agent (`raw_to_embedding.pipeline`).** For arbitrary PDFs/URLs, the package provides extraction → classification → (optional LLM) segmentation → embedding-oriented JSON chunks; this path is orthogonal to the SSL-specific builders but shares extractors and processors.

## 4. Final System Design

The **authoritative** corpus layout is documented in `data/final_corpus_bundle/REPORT/FINAL_CORPUS_REPORT.md` and mirrored under `results/raw_to_embedding/final/` for archival convenience. It contains:

- `pdf/` — per-PDF chunk JSON and per-PDF indices (historical artifacts from the batch run).
- `website/` — website chunk JSON, embeddings, and website-local FAISS.
- `merged/` — **unified** metadata and index consumed by RAG before contextualized re-embedding.

## 5. Smoke Test Setup

After merging, the website builder executed a **retrieval smoke test**: a fixed list of natural questions embedded with the same model, queried against the unified FAISS index, and checked for the presence of at least one **website** chunk in the top-5 results.

## 6. Final Experimental Outputs

| Artifact | Role |
|----------|------|
| `results/raw_to_embedding/final/full_pipeline_report.json` | PDF batch statistics (18 PDFs, chunk counts, timing). |
| `results/raw_to_embedding/final/website_pipeline_report.json` | Website processing stats + embedded smoke test. |
| `results/raw_to_embedding/final/FINAL_CORPUS_REPORT.md` | Human-readable corpus inventory and conventions. |
| `results/raw_to_embedding/final/metrics.json` | Short index pointing to the above files. |
| `results/raw_to_embedding/smoke_test/*` | **Extracted** smoke-test slice (`metrics.json`, `summary.md`, `outputs.json`) for quick review. |

Artifact paths inside JSON were normalized to **repository-relative** strings pointing at `data/final_corpus_bundle/...` so clones are not tied to obsolete absolute directories.

## 7. Observations

- Unified merging materially increases coverage for **people**, **projects**, and **news** queries compared to PDF-only indices.
- Smoke-test hit rates indicated strong website recall for profile- and role-centric queries, while some queries naturally retrieve PDF hits first (still valid for RAG).

## 8. Limitations

- Website text depends on crawl quality and template noise; chunking cannot recover information missing from HTML extracts.
- Per-PDF pilot files remain large JSON assets; Git LFS or external hosting may be preferable for some deployments.
- Regenerating the unified bundle requires consistent versions of `sentence-transformers` and FAISS for bitwise-identical vectors.

## 9. Conclusion

The **final** SSL embedding corpus is frozen as `data/final_corpus_bundle/`, with reproducible reports and an explicit **smoke test** record under `results/raw_to_embedding/smoke_test/`. Downstream RAG V1 consumes this bundle and optionally rebuilds a **contextualized** dense index without altering stored raw `chunk_text` in the source metadata file.
