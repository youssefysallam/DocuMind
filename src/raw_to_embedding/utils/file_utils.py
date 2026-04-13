"""Filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    """Create parent directories if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Write UTF-8 JSON with stable formatting."""
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")


def read_text_safe(path: Path) -> str:
    """Read text with UTF-8."""
    return path.read_text(encoding="utf-8", errors="replace")
