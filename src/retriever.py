import json
import faiss
from sentence_transformers import SentenceTransformer

BUNDLE_PATH = "processed/final_corpus_bundle/merged"
INDEX_PATH = f"{BUNDLE_PATH}/unified_index.faiss"
METADATA_PATH = f"{BUNDLE_PATH}/unified_index_metadata.json"

# Load FAISS vector index from disk
def load_index():
    return faiss.read_index(INDEX_PATH)

# Load metadata that maps index positions to text chunks
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Load embedding model for query encoding
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Retrieve top-k relevant chunks for a query
def retrieve(query, k=5):
    index = load_index()
    metadata = load_metadata()
    model = load_model()

    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec.astype("float32"), k)

    results = []

    # Pair each retrieved index with its score, then use index to get actual chunk from metadata
    for idx, score in zip(I[0], D[0]):
        chunk = metadata[idx]

        # Build a clean result dictionary with key fields for retrieval and citation
        result = {
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "chunk_text": chunk["chunk_text"],
            "source_type": chunk["source_type"],
            "section_title": chunk["section_title"],
            "section_path": chunk["section_path"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk_type": chunk["chunk_type"],
            "quality_flag": chunk["quality_flag"]
        }

        # Add optional source fields depending on chunk type (pdf or website)
        if "source_pdf" in chunk:
            result["source_pdf"] = chunk["source_pdf"]

        if "source_url" in chunk:
            result["source_url"] = chunk["source_url"]

        if "source_file" in chunk:
            result["source_file"] = chunk["source_file"]

        results.append(result)

    return results

# Format retrieved results into a readable context string for prompt building
def format_results_for_prompt(results):
    context_parts = []

    for i, r in enumerate(results):
        source = r.get("source_pdf", r.get("source_url", "unknown source"))

        context_parts.append(
            f"[{i+1}] ({r['source_type']}) {r['section_title']} | {source}\n"
            f"{r['chunk_text']}\n"
        )

    return "\n".join(context_parts)

# Simple Testing
if __name__ == "__main__":
    results = retrieve("What is SSL's mission?", k=3)

    for i, r in enumerate(results):
        source = r.get("source_pdf", r.get("source_url", "unknown source"))
        print(f"[{i+1}] score={r['score']:.3f} | {r['source_type']} | {r['section_title']} | {source}")
        print(f"    {r['chunk_text'][:300]}...\n")

    print("----- FORMATTED CONTEXT -----")
    print(format_results_for_prompt(results))