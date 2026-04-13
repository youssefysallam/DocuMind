"""
Answer-evidence consistency checker for RAG V2.

Lightweight post-generation check that verifies the answer's key claims
are supported by the retrieved evidence.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


@dataclass
class ConsistencyResult:
    is_consistent: bool
    confidence: float
    unsupported_claims: list[str] = field(default_factory=list)
    explanation: str = ""


def check_consistency(
    answer: str,
    evidence: list[dict],
    client: "OpenAI",
    model: str,
) -> ConsistencyResult:
    """Check whether the answer's claims are supported by the evidence."""
    evidence_text = "\n\n".join(
        f"[{e.get('source', '?')}] {e.get('chunk_text', '')[:500]}"
        for e in evidence[:5]
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You verify whether a system's answer is fully supported by the provided evidence.\n"
                    "Identify any claims in the answer that are NOT supported by the evidence.\n\n"
                    "Return ONLY a JSON object with:\n"
                    '  "is_consistent": true/false,\n'
                    '  "confidence": 0.0-1.0,\n'
                    '  "unsupported_claims": ["claim1", ...] (empty if consistent),\n'
                    '  "explanation": "brief explanation"\n'
                    "No other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{evidence_text}\n\n"
                    f"Answer:\n{answer}"
                ),
            },
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
        data = json.loads(text)
        return ConsistencyResult(
            is_consistent=bool(data.get("is_consistent", True)),
            confidence=float(data.get("confidence", 0.5)),
            unsupported_claims=data.get("unsupported_claims", []),
            explanation=data.get("explanation", ""),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return ConsistencyResult(
            is_consistent=True, confidence=0.0,
            explanation="Consistency check failed to parse LLM response.",
        )
