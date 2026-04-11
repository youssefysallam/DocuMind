"""
LLM-enhanced intent classification for RAG V2.

Phase 1 used keyword-only classification which missed many no_evidence cases.
Phase 2 adds an LLM fallback: keyword match is tried first (fast path),
and if it falls through to the default, the LLM is consulted.
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

VALID_INTENTS = frozenset({
    "general_overview",
    "project_initiative",
    "topic_specific",
    "publication_finding",
    "synthesis",
    "no_evidence",
})

_INTENT_KW: dict[str, list[str]] = {
    "general_overview": [
        "what is ssl", "what does ssl do", "tell me about ssl",
        "ssl mission", "ssl vision", "who leads ssl", "ssl contact",
        "ssl location", "ssl address", "who works at ssl", "ssl team",
        "ssl advisory board", "ssl board", "ssl staff", "ssl director",
        "partner institutes", "what is the sustainable solutions lab",
        "ssl focus areas", "ssl research areas",
    ],
    "publication_finding": [
        "published", "publication", "report findings", "paper",
        "study found", "research found", "key findings", "conclusions",
        "report recommends", "views that matter", "voices that matter",
        "academic background", "publications has ssl",
    ],
    "project_initiative": [
        "project", "initiative", "program", "c3i", "cliir",
        "cape cod rail", "mvp", "green ribbon", "barr foundation",
        "climate careers", "collaborative", "rail resilience",
        "lumbee", "climate adaptation forum", "water quality",
        "east boston", "opportunity in the complexity",
    ],
    "no_evidence": [
        "recycling", "e-waste", "carbon footprint", "biodiversity",
        "air quality", "agriculture", "food system", "wildfire",
        "renewable energy", "k-12", "international context",
        "nuclear", "deforestation", "ocean acidification",
        "transportation electrification", "cryptocurrency", "blockchain",
        "mental health", "sea-level rise modeling",
    ],
    "synthesis": [
        "how does", "relationship between", "intersection",
        "compare", "across", "multiple", "integrate", "connect",
        "combined", "together", "holistic", "evolved", "full scope",
        "funding landscape", "events and convenings", "cross-project",
    ],
}


def classify_intent_keyword(question: str) -> str | None:
    """Fast keyword-based classification. Returns None if no keyword matches."""
    ql = question.lower()
    for intent, kws in _INTENT_KW.items():
        if any(kw in ql for kw in kws):
            return intent
    return None


def classify_intent_llm(question: str, client: "OpenAI", model: str) -> str:
    """LLM-based intent classification as fallback."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You classify questions about the Sustainable Solutions Lab (SSL) at UMass Boston "
                    "into exactly one category. The categories are:\n\n"
                    "- general_overview: questions about what SSL is, its mission, leadership, contact info\n"
                    "- project_initiative: questions about specific SSL projects or programs\n"
                    "- topic_specific: questions about a specific topic SSL has researched\n"
                    "- publication_finding: questions about findings from a specific SSL report\n"
                    "- synthesis: questions requiring information from multiple sources or comparing topics\n"
                    "- no_evidence: questions about topics SSL has NOT studied or that are outside SSL's scope\n\n"
                    "SSL focuses on: climate justice, equitable adaptation, urban resilience, governance, "
                    "housing & climate, transient populations, climate migration, workforce development (C3I), "
                    "harbor barriers, rail resilience, community engagement in Greater Boston.\n\n"
                    "SSL does NOT research: carbon emissions reduction, renewable energy technology, "
                    "agriculture/food, biodiversity, air pollution, K-12 education, international contexts, "
                    "nuclear energy, cryptocurrency, mental health, sea-level modeling (physics).\n\n"
                    "Return ONLY the category name, nothing else."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=20,
        timeout=15,
    )
    result = resp.choices[0].message.content.strip().lower().replace(" ", "_")
    if result in VALID_INTENTS:
        return result
    return "topic_specific"


def classify_intent(
    question: str,
    client: "OpenAI | None" = None,
    model: str = "openai/gpt-5.4",
    use_llm_fallback: bool = True,
) -> str:
    """Hybrid intent classification: keyword fast-path + LLM fallback.

    When *use_llm_fallback* is False, behaves identically to V1.
    """
    kw_result = classify_intent_keyword(question)
    if kw_result is not None:
        return kw_result

    if use_llm_fallback and client is not None:
        return classify_intent_llm(question, client, model)

    return "topic_specific"
