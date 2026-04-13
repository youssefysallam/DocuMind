"""
Lightweight session-level memory for RAG V2.

Stores recent (question, answer, sources) tuples so the system can:
  - resolve coreferences in follow-up questions
  - inject conversation context into generation prompts
  - decide whether to re-retrieve or reuse prior evidence

This is NOT a knowledge source — facts still come from the corpus.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    question: str
    resolved_query: str
    intent: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieval_log: dict[str, Any] = field(default_factory=dict)


class SessionMemory:
    """Fixed-window conversation memory."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    @property
    def last_turn(self) -> Turn | None:
        return self._turns[-1] if self._turns else None

    def add_turn(self, turn: Turn) -> None:
        self._turns.append(turn)
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def get_context_summary(self, last_n: int = 3) -> str:
        """Build a concise summary of recent turns for prompt injection."""
        recent = self._turns[-last_n:]
        if not recent:
            return ""
        lines = ["Previous conversation:"]
        for i, t in enumerate(recent, 1):
            src_names = ", ".join(
                s.get("source", "?") for s in t.sources[:3]
            )
            lines.append(
                f"  Q{i}: {t.question}\n"
                f"  A{i}: {t.answer[:200]}{'...' if len(t.answer) > 200 else ''}\n"
                f"  Sources: {src_names}"
            )
        return "\n".join(lines)

    def get_last_sources(self) -> list[dict[str, Any]]:
        """Return retrieved sources from the most recent turn."""
        if not self._turns:
            return []
        return self._turns[-1].sources

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
