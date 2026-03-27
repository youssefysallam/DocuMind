# SSL RAG Chatbot — Raw Data Collection

**CS 438/638 — UMass Boston | Phase 1 (Due: March 30)**

---

## Folder Structure

```
rawdata/
├── website_pages/        ← SSL website pages (JSON, one file per page)
├── institute_report/     ← SSL institute reports (PDF + JSON metadata)
├── scholary_papers/      ← Scholarly papers by SSL researchers (PDF + JSON metadata)
├── metadata_index.json   ← Master index of all documents (auto-generated)
│
├── scrape_website_pages.py    ← Step 1: scrape the SSL website
├── scrape_scholarworks.py     ← Step 2: download reports & papers
└── build_metadata_index.py    ← Step 3: build the master index
```

---

## Data Sources

| Source | URL | Saved To |
|--------|-----|----------|
| SSL Website | https://www.umb.edu/ssl/ | `website_pages/` |
| ScholarWorks Repository | https://scholarworks.umb.edu/ssl/ | `institute_report/` or `scholary_papers/` |

---

## How to Run (in order)

### 1. Install dependencies
```bash
pip install requests beautifulsoup4
```

### 2. Scrape SSL website pages
```bash
python scrape_website_pages.py
```
Saves each page as a JSON file in `website_pages/`.

### 3. Download institute reports & papers from ScholarWorks
```bash
python scrape_scholarworks.py
```
Downloads PDFs and extracts metadata. Saves to `institute_report/` or `scholary_papers/`
depending on whether the document looks like a policy report or a scholarly article.

### 4. Build the master metadata index
```bash
python build_metadata_index.py
```
Produces `metadata_index.json` — a single file listing every document with its
title, URL, source type, and file path. Your RAG pipeline will load this to know
what's in the knowledge base.

---

## File Format

Every saved document has a **JSON sidecar** with these fields:

```json
{
  "title":        "Financing Climate Resilience...",
  "url":          "https://scholarworks.umb.edu/ssl/12/",
  "source_type":  "institute_report",
  "text":         "...(full extracted text)...",
  "abstract":     "...",
  "authors":      ["Author Name"],
  "pub_date":     "2021",
  "pdf_file":     "ssl_12.pdf",
  "downloaded_at": "2026-03-26T..."
}
```

---

## Next Steps (after data collection)

1. **Chunking** — split long documents into ~500-token passages
2. **Embeddings** — embed each chunk using sentence-transformers or OpenAI
3. **Vector store** — load into ChromaDB or FAISS for retrieval
4. **RAG pipeline** — retrieve top-k chunks → pass to LLM → return answer + citations

---

## Notes

- Re-run all three scripts any time new content is added to the SSL website.
- The metadata_index.json is the single source of truth for the ingestion pipeline.
- PDFs that can't be downloaded still get a JSON metadata file (useful for retrieval index).
