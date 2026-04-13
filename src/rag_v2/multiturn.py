"""
Multi-turn dialogue support for RAG V2.

Handles:
  - Coreference resolution: rewrite follow-ups into self-contained queries
  - Re-retrieval strategy: decide whether to re-search or reuse prior evidence
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from rag_v2.session import SessionMemory

if TYPE_CHECKING:
    from openai import OpenAI


class RetrievalStrategy(Enum):
    FULL_RETRIEVAL = "full_retrieval"
    REUSE_EVIDENCE = "reuse_evidence"
    MERGE_EVIDENCE = "merge_evidence"


def resolve_coreference(
    current_query: str,
    session: SessionMemory,
    client: "OpenAI",
    model: str,
) -> str:
    """Rewrite a follow-up question into a self-contained query using conversation history."""
    if len(session) == 0:
        return current_query

    history_lines = []
    for t in session.turns[-3:]:
        history_lines.append(f"User: {t.question}")
        history_lines.append(f"Assistant: {t.answer[:300]}")
    history_text = "\n".join(history_lines)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite follow-up questions into self-contained queries.\n"
                    "Given a conversation history and a new question, rewrite the new question "
                    "so it can be understood WITHOUT the conversation history.\n"
                    "Rules:\n"
                    "- Replace pronouns (it, they, this, that) with their referents from history.\n"
                    "- Include relevant context from prior turns if needed.\n"
                    "- If the question is already self-contained, return it unchanged.\n"
                    "- Return ONLY the rewritten question. No explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"New question: {current_query}\n\n"
                    "Rewritten question:"
                ),
            },
        ],
        temperature=0,
        max_tokens=150,
        timeout=20,
    )
    rewritten = resp.choices[0].message.content.strip()
    return rewritten if rewritten else current_query


def decide_retrieval_strategy(
    current_query: str,
    resolved_query: str,
    session: SessionMemory,
    client: "OpenAI",
    model: str,
) -> RetrievalStrategy:
    """Decide whether to re-retrieve, reuse, or merge evidence."""
    if len(session) == 0:
        return RetrievalStrategy.FULL_RETRIEVAL

    last = session.last_turn
    if last is None:
        return RetrievalStrategy.FULL_RETRIEVAL

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You decide how a RAG system should handle a follow-up question.\n"
                    "Given the previous question+answer and the new question, choose ONE strategy:\n\n"
                    "- FULL_RETRIEVAL: The new question is about a completely different topic. "
                    "Search the corpus from scratch.\n"
                    "- REUSE_EVIDENCE: The new question asks for more detail, clarification, or "
                    "rephrasing of the same information. Reuse the same evidence.\n"
                    "- MERGE_EVIDENCE: The new question relates the previous topic to something new. "
                    "Search for new evidence AND keep some old evidence.\n\n"
                    "Return ONLY one of: FULL_RETRIEVAL, REUSE_EVIDENCE, MERGE_EVIDENCE"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous question: {last.question}\n"
                    f"Previous answer (excerpt): {last.answer[:300]}\n\n"
                    f"New question: {resolved_query}"
                ),
            },
        ],
        temperature=0,
        max_tokens=20,
        timeout=15,
    )
    text = resp.choices[0].message.content.strip().upper()
    for s in RetrievalStrategy:
        if s.value.upper() in text:
            return s
    return RetrievalStrategy.FULL_RETRIEVAL
