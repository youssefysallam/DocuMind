"""
High-level CLI for the generic raw-to-embedding agent (PDFs / URLs).

Delegates to ``main.main``. Run from repository root::

    pip install -e .
    python -m raw_to_embedding.pipeline --help
"""
from __future__ import annotations

from raw_to_embedding.main import main


if __name__ == "__main__":
    raise SystemExit(main())
