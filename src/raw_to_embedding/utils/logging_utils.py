"""Logging setup for CLI and pipeline."""

from __future__ import annotations

import logging
import sys
from typing import TextIO


def setup_logging(level: int = logging.INFO, stream: TextIO | None = None) -> None:
    """Configure root logger once with a concise format."""
    stream = stream or sys.stderr
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=stream,
        force=True,
    )
