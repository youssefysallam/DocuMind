"""
Query transformation module for RAG V2.

Provides:
  - HyDE (Hypothetical Document Embeddings): generate a hypothetical answer, embed it for retrieval
  - Sub-query decomposition: split complex/synthesis questions into atomic sub-queries
  - Query expansion: enrich short queries with synonyms and related terms
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


def hyde_generate(question: str, client: "OpenAI", model: str) -> str:
    """Generate a hypothetical answer passage for HyDE retrieval."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant for the Sustainable Solutions Lab (SSL) at UMass Boston. "
                    "Write a short paragraph (3-5 sentences) that would be a plausible answer to the "
                    "following question, as if it appeared in an SSL report or webpage. "
                    "Use factual-sounding language even if you are uncertain."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=200,
        timeout=30,
    )
    return resp.choices[0].message.content.strip()


def decompose_query(question: str, client: "OpenAI", model: str) -> list[str]:
    """Break a complex question into 2-4 simpler sub-queries.

    Returns the original question as-is if decomposition is unnecessary.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You decompose complex research questions into simpler sub-questions.\n"
                    "Rules:\n"
                    "- If the question is already simple and atomic, return it unchanged as a single-element list.\n"
                    "- If the question asks about relationships, comparisons, or multiple topics, "
                    "break it into 2-4 focused sub-questions.\n"
                    "- Each sub-question should be self-contained (no pronouns referring to other sub-questions).\n"
                    "- Return ONLY a JSON array of strings. No other text."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=300,
        timeout=30,
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        subs = json.loads(text)
        if isinstance(subs, list) and all(isinstance(s, str) for s in subs):
            return subs if len(subs) > 1 else [question]
    except (json.JSONDecodeError, TypeError):
        pass
    return [question]


def expand_query(question: str, client: "OpenAI", model: str) -> str:
    """Expand a short query with related terms for better sparse retrieval."""
    if len(question.split()) > 12:
        return question

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Expand the following short question by adding 3-5 related keywords or phrases "
                    "that would help retrieve relevant documents about the Sustainable Solutions Lab (SSL) "
                    "at UMass Boston. Return ONLY the expanded query as a single line. "
                    "Keep the original question intact and append the expansions."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=100,
        timeout=20,
    )
    return resp.choices[0].message.content.strip()
