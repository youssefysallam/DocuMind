#!/usr/bin/env python3
"""
Entry point: read URL lists from data/urls and run the full main pipeline.

Examples:
  python run_urls_from_data.py
  python run_urls_from_data.py --save-intermediate -v
  python run_urls_from_data.py -o ./output/from_urls.json

Extra arguments are forwarded to main.py (see python main.py --help).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def main() -> int:
    from raw_to_embedding.main import main as pipeline_main

    url_dir = _ROOT / "data" / "urls"
    out = _ROOT / "output" / "embedding_chunks.json"
    argv = [
        "--url-dir",
        str(url_dir),
        "-o",
        str(out),
        *sys.argv[1:],
    ]
    return pipeline_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
