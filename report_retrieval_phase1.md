# **Phase 1 Retrieval Summary**

## **Overview**

For Phase 1, I implemented the baseline retrieval component in `src/retriever.py`.

This component takes a user query, embeds it, retrieves the top-k most relevant chunks from the corpus, and returns structured results for generation and citation.

It uses the embedding-ready dataset and FAISS index from the `feat/embeddings` branch.

---

## **System Flow**

The retrieval pipeline is:

1. **Load FAISS index**

   * `processed/final_corpus_bundle/merged/unified_index.faiss`

2. **Load metadata**

   * `unified_index_metadata.json`
   * maps FAISS indices → actual chunk data

3. **Encode query**

   * `SentenceTransformer("all-MiniLM-L6-v2")`
   * uses `normalize_embeddings=True`

4. **Search**

   * FAISS top-k nearest neighbors

5. **Map results**

   * convert indices → full chunk records

---

## **Output Format**

The retriever returns a list of dicts like:

```python
[
  {
    "score": float,
    "chunk_id": str,
    "chunk_text": str,
    "source_type": str,
    "section_title": str,
    "section_path": str,
    "page_start": int,
    "page_end": int,
    "chunk_type": str,
    "quality_flag": str,
    "source_pdf": str (optional),
    "source_url": str (optional),
    "source_file": str (optional)
  }
]
```

This keeps everything needed for:

* prompt building
* citations
* debugging

---

## **Prompt Formatting**

I added:

```python
format_results_for_prompt(results)
```

This turns results into a clean context string like:

```
[1] (pdf) Mission | UMB-SSL-2025-Impact_Report.pdf
<chunk text>

[2] (pdf) About | UMB-SSL-2022-Annual_Report.pdf
<chunk text>
```

This is what generation can plug directly into prompts.
This enables the chatbot to produce grounded answers with citations.

---

## **What Works**

* Retrieval returns relevant chunks for natural queries
* Source info is preserved (PDF / URL → supports citations)
* Output format is clean and easy to use downstream

Example:
For *“What is SSL’s mission?”*, the retriever returned “About” and “Mission” sections containing relevant context.

---

## **Limitations**

* Some results are related but not direct answers

* “Mission” sections sometimes include:

  * planning history
  * admin/biographical content

* Dense retrieval alone doesn’t guarantee best answer chunk

---

## **Failure Example**

For the mission query:

* one result focused on strategic planning history
* another included background info instead of a clear mission

This shows that retrieval can return semantically related but non-answer-focused chunks.

---

## **Phase 2 Improvements**

* filter by `quality_flag`
* tune `k`
* reranking
* hybrid retrieval (BM25 + dense)
* better chunking

---

## **How to Run**

```bash
python src/retriever.py
```

Runs a test query and prints:

* top-k results
* formatted context for prompt use

---

## **Conclusion**

The retriever is working end-to-end and integrated with the corpus.

It meets Phase 1 requirements:

* FAISS retrieval
* structured output
* source grounding for citations

It’s not perfect, but it’s a solid baseline to build on.
These limitations will be addressed in Phase 2 through improved retrieval strategies.

---