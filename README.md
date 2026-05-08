# DocuMind — Knowledge Chatbot
> CS 438/638 Applied Machine Learning — Term Project
> University of Massachusetts Boston

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a real research institute using its own documents. The system ingests institutional materials, retrieves relevant passages, and generates grounded answers with transparent source citations.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team](#team)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Chatbot](#running-the-chatbot)
- [Running Evaluation](#running-evaluation)
- [Running on HPC](#running-on-hpc)
- [Environment Variables](#environment-variables)
- [Phase 1 Deliverables](#phase-1-deliverables)
- [Contributing](#contributing)

---

## Project Overview

Modern research institutes hold large amounts of valuable information spread across websites, PDFs, reports, and staff pages. This project builds an AI assistant that:

- Ingests and indexes all institute documents automatically
- Retrieves the most relevant passages for any user question
- Generates accurate, grounded answers using a large language model
- Cites every source used in the answer
- Refuses to answer when reliable information is not available

The system is built around a RAG pipeline — no model training required for the base system.

---

## Team

| Person | Role | Branch |
|--------|------|--------|
| P1 | Data / Ingestion | `feat/ingestion` |
| P2 | Embeddings + Vector Database | `feat/embeddings` |
| P3 | Retrieval System | `feat/retrieval` |
| P4 | LLM / Answer Generation + UI | `feat/generation` |
| P5 | Integration + Evaluation | `feat/evaluation` |

---

## Architecture

```
User question
     │
     ▼
[Retriever] ──── vector search + BM25 hybrid ────► ChromaDB
     │
     ▼
Retrieved passages (top-k chunks with metadata)
     │
     ▼
[RAG Chain] ──── prompt builder + LLM API call
     │
     ▼
Grounded answer with inline citations [Source 1], [Source 2]
     │
     ▼
[Streamlit UI] ──── answer panel + expandable source panel
```

**Ingestion pipeline (runs once, or when documents change):**
```
Institute website / PDFs
     │
     ▼
[ingest.py] ──── scrape → parse → chunk → extract metadata
     │
     ▼
Clean JSON: { text, chunk_id, source, url, date }
     │
     ▼
[embedder.py] ──── embed each chunk ──► ChromaDB
```

---

## Folder Structure

```
DocuMind/
├── data/
│   ├── raw/                  # Scraped HTML and downloaded PDFs (not committed)
│   ├── processed/            # Chunked JSON output from ingest.py
│   └── eval/                 # 30-question evaluation dataset and results
├── src/
│   ├── __init__.py
│   ├── ingest.py             # P1 — scrape, parse, chunk, extract metadata
│   ├── embedder.py           # P2 — embed chunks using sentence-transformers
│   ├── vector_store.py       # P2 — ChromaDB setup, load, and query interface
│   ├── retriever.py          # P3 — hybrid vector + BM25 search
│   ├── rag_chain.py          # P4 — prompt builder, LLM call, citations, refusal
│   ├── evaluator.py          # P5 — classification and regression metrics
│   └── finetune/             # P2 — bonus embedding fine-tuning scripts
├── app/
│   └── streamlit_app.py      # P4 — web interface
├── hpc/
│   └── eval_job.slurm        # P5 — HPC batch evaluation job script
├── .env.example              # API key template — copy to .env and fill in
├── .gitignore
├── pyproject.toml
├── CONTRIBUTING.md
└── README.md
```

---

## Requirements

- Python 3.10 or higher
- pip
- A virtual environment (strongly recommended)
- An OpenAI API key **or** a HuggingFace account (for free embedding model)
- Git

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/DocuMind.git
cd DocuMind
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -e .
```

### 4. Configure environment variables
```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` and fill in your keys:
```
OPENAI_API_KEY=sk-...
CHROMA_DB_PATH=./chroma_db
```

> Never commit your `.env` file. It is listed in `.gitignore` and must stay local.

---

## Running the Pipeline

The ingestion pipeline must be run once before the chatbot can answer questions. It scrapes the institute website, parses PDFs, chunks text, and loads everything into ChromaDB.

```bash
python src/ingest.py
```

This will:
1. Scrape all pages from the configured institute URL
2. Parse any PDFs found
3. Chunk text into 300–500 token overlapping segments
4. Save clean JSON to `data/processed/`

Then load the chunks into the vector database:
```bash
python src/embedder.py
```

This will:
1. Read all chunks from `data/processed/`
2. Generate embeddings using the configured model
3. Store vectors and metadata in ChromaDB at `chroma_db/`

> Re-run both scripts any time the institute adds new documents.

---

## Running the Chatbot

```bash
streamlit run app/streamlit_app.py
```

Open your browser to `http://localhost:8501`

The interface shows:
- A chat input for your question
- The generated answer with inline citations
- An expandable panel showing the retrieved source passages and URLs

### Running the DocuMind V2 UI

The improved UI in `frontend/` runs separately from the legacy chatbot UI and
expects the FastAPI backend in `src/rag_v2/api.py`.

Run these in two terminals:

```bash
# Terminal 1
set PYTHONPATH=src
python -m rag_v2.api
```

```bash
# Terminal 2
cd frontend
npm run dev
```

Then open:

```text
http://localhost:3000/chat
```

If port `3000` is already in use, run:

```bash
cd frontend
npm run dev -- --port 3001
```

and then open `http://localhost:3001/chat`.

The V2 UI proxies `/api/*` requests to the FastAPI backend on port `8000`, so
the backend must be running for chat to work.

---

## Running Evaluation

The evaluator runs the full 30-question dataset through the system and scores each answer.

```bash
python src/evaluator.py --dataset data/eval/questions.json --output data/eval/results.json
```

This produces:
- **Classification metrics** (Pass/Fail): answer correctness, citation presence, retrieval relevance, correct refusal
- **Regression metrics** (1–5 scale): answer quality, grounding quality, helpfulness
- A summary report printed to the console and saved to `data/eval/results.json`

---

## Running on HPC

All final evaluation must be run on the UMass HPC cluster.

### Submit the batch job
```bash
sbatch hpc/eval_job.slurm
```

### Check job status
```bash
squeue -u YOUR_USERNAME
```

### View output logs
```bash
cat hpc/logs/eval_output.txt
```

Make sure your HPC environment has Python 3.10+ and all dependencies installed. See `hpc/eval_job.slurm` for the full job configuration including memory, CPU, and time limit settings.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (if using OpenAI) | API key for GPT-4o-mini LLM calls and/or embeddings |
| `ANTHROPIC_API_KEY` | Yes (if using Claude) | API key for Claude Haiku LLM calls |
| `CHROMA_DB_PATH` | Yes | Local path for ChromaDB persistence (default: `./chroma_db`) |
| `EMBEDDING_MODEL` | No | HuggingFace model name (default: `all-MiniLM-L6-v2`) |
| `LLM_MODEL` | No | LLM model name (default: `gpt-4o-mini`) |
| `TOP_K_RESULTS` | No | Number of passages to retrieve per query (default: `5`) |

---

## Phase 1 Deliverables
**Due: March 30, 12:00 PM**

- [x] GitHub repo set up with branch protection and team access
- [ ] Working ingestion pipeline (`ingest.py` runs end-to-end)
- [ ] ChromaDB loaded with embedded institute documents
- [ ] Retrieval returning relevant passages for test queries
- [ ] Chatbot answering 10 demo questions with citations
- [ ] Institute partner meeting completed
- [ ] Institute meeting summary written
- [ ] 10-question demo dataset in `data/eval/`
- [ ] Failure analysis — 5 documented cases
- [ ] README complete (this file)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow including branching rules, commit message format, PR process, and file ownership.

---

## License

This project is for academic use only as part of CS 438/638 at UMass Boston.
