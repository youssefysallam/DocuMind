"""Test multi-turn dialogue with session memory and coreference resolution."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONUNBUFFERED"] = "1"

from sentence_transformers import SentenceTransformer, CrossEncoder
from rag_v1.pipeline import load_all, openai_client, EMBED_MODEL_NAME, RERANK_MODEL_NAME
from rag_v2.pipeline import ask
from rag_v2.session import SessionMemory

print("Loading models...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
reranker = CrossEncoder(RERANK_MODEL_NAME)
client = openai_client()
meta, ctx_idx, qa_items, qa_idx, bm25, _ = load_all(embed_model)

session = SessionMemory(max_turns=10)
common = dict(
    session=session, client=client, embed_model=embed_model,
    corpus_idx=ctx_idx, corpus_meta=meta, bm25=bm25,
    qa_items=qa_items, qa_idx=qa_idx, reranker=reranker,
)

test_conversations = [
    [
        ("Tell me about C3I.", None),
        ("Who funds it and how much did they give?", "Liberty Mutual"),
        ("How many participants does it aim to serve?", "90"),
    ],
    [
        ("What is the Cape Cod Rail Resilience project?", None),
        ("How does that project compare to SSL's work in East Boston?", "East Boston"),
        ("What about the harbor barrier study?", "barrier"),
    ],
    [
        ("Who leads SSL?", None),
        ("Can you tell me more about the executive director's background?", "Balachandran"),
        ("What about the research director?", "Negrón"),
    ],
]

print("\n" + "=" * 70)
print("  MULTI-TURN DIALOGUE TEST")
print("=" * 70)

total_tests = 0
passed_tests = 0

for conv_idx, conversation in enumerate(test_conversations):
    session.clear()
    print(f"\n--- Conversation {conv_idx + 1} ---")
    
    for turn_idx, (question, expected_keyword) in enumerate(conversation):
        result = ask(question, **common)
        
        resolved = result.get("resolved_query", question)
        intent = result.get("intent", "?")
        answer = result["answer"]
        
        resolved_changed = resolved != question
        
        print(f"\n  Turn {turn_idx + 1}: \"{question}\"")
        if resolved_changed:
            print(f"  >> Resolved to: \"{resolved}\"")
        print(f"  Intent: {intent}")
        print(f"  Answer (first 200 chars): {answer[:200]}...")
        
        if expected_keyword:
            total_tests += 1
            kw_lower = expected_keyword.lower()
            if kw_lower in answer.lower() or kw_lower in resolved.lower():
                passed_tests += 1
                print(f"  CHECK: PASS ('{expected_keyword}' found)")
            else:
                print(f"  CHECK: FAIL ('{expected_keyword}' NOT found in answer or resolved query)")

print(f"\n{'=' * 70}")
print(f"  RESULTS: {passed_tests}/{total_tests} keyword checks passed")
print(f"  Sessions tested: {len(test_conversations)}")
print(f"{'=' * 70}")
