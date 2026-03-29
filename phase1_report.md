# **Phase 1 Report — RAG System for SSL Knowledge Base**

---

## **1. Problem & Use Case**

Our project focuses on building a retrieval-augmented generation (RAG) system for the **Sustainable Solutions Lab (SSL) at UMass Boston**.

The goal is to create a system that can answer stakeholder-style questions about:

* SSL’s mission, team, and organization
* research areas and publications
* projects and initiatives
* broader climate-related topics connected to SSL

The system must also correctly **refuse to answer** when no relevant information exists in the corpus.

This is challenging because:

* institutional documents lose context when chunked
* retrieval may return only partially relevant information
* LLMs may hallucinate when evidence is weak

---

## **2. System Overview**

### **High-Level Pipeline**

```
Raw Data → Embedding Pipeline → Retrieval → Reranking → Generation → Answer
```

---

### **Data Sources**

* 18 SSL PDF reports
* 20+ SSL website pages
* Total: ~5,200 text chunks

---

### **Ingestion (Embedding Pipeline)**

An AI-assisted pipeline processes raw documents into chunks:

* extracts text from PDFs and website data
* splits into semantically meaningful chunks
* validates chunks to prevent hallucinated content
* encodes chunks into embeddings

---

### **Retrieval System**

The system uses a **hybrid retrieval approach**:

* **Dense retrieval (FAISS)**

  * semantic similarity using MiniLM embeddings

* **Sparse retrieval (BM25)**

  * keyword matching

* **Hybrid scoring**

  ```
  0.6 × dense + 0.4 × sparse
  ```

* **Reranking**

  * cross-encoder model improves ranking of top results

* **Final selection**

  * top 5 chunks
  * ensures at least 3 real corpus chunks

---

### **Key Improvement: Contextualized Chunks**

Each chunk is enhanced with a prefix:

> “This passage is from the SSL report ‘[title]’…”

This helps the model understand:

* where the information comes from
* how it relates to SSL

This significantly improves answer accuracy 

---

### **Generation**

* Model: `gpt-4o-mini`
* Uses retrieved chunks as context
* Follows strict rules:

  * answer only from evidence
  * refuse if no relevant information
  * include source grounding

---

## **3. Demo**

Example question:

**“What is SSL’s mission?”**

System behavior:

* retrieves relevant “Mission” and “About” sections
* generates a grounded response using those chunks
* includes references to source documents

The system is able to provide accurate, citation-supported answers for common stakeholder questions.

---

## **4. What Works / What Doesn’t**

### ✅ What Works

* **Improved accuracy with contextualized chunks**

  * Answer accuracy increased from ~72.7% → ~79.1% 

* **Hybrid retrieval improves relevance**

  * combining semantic + keyword search improves results

* **Low false refusal rate**

  * system usually answers when it should

---

### ❌ What Doesn’t

#### 1. Missed Refusals

The system sometimes answers questions it should refuse.

Example:

* “What is SSL’s work on carbon emissions?”
* system generates an answer from weak evidence instead of refusing

Cause:

* weak intent classification
* missing detection of “no evidence” queries

---

#### 2. Tangential Retrieval

Some retrieved chunks are only loosely related.

Example:

* chunk mentions a keyword once
* system treats it as strong evidence

---

#### 3. Weak Multi-Document Reasoning

“Synthesis” questions (combining multiple sources) perform poorly.

* lowest accuracy category (~43.8%) 

---

## **5. Next Steps (Phase 2)**

Planned improvements:

### High Priority

* improve intent classification (detect no-evidence queries better)
* add confidence-based refusal logic
* filter weak or tangential chunks before generation

### Medium Priority

* improve retrieval for rare queries (synonyms, query expansion)
* support multi-step retrieval for synthesis questions

### Low Priority

* experiment with stronger reranking models
* refine chunking strategy
* explore multilingual embeddings

---

## **Conclusion**

The Phase 1 system successfully implements a working RAG pipeline:

* ingestion
* retrieval
* generation
* evaluation

The system produces grounded answers with citations and demonstrates clear improvements over a baseline approach.

While limitations remain—especially in refusal handling and complex reasoning—the system provides a strong foundation for further improvement in Phase 2.

---