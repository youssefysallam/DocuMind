#!/usr/bin/env python3
"""
Fetch SSL-related HTML pages and save visible main text (not PDFs) for RAG.

Reads URL lists (one URL per line, # comments allowed), strips nav/footer/scripts,
writes one .txt per page + manifest.json under data/ssl_crawl/page_text_corpus/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raw_to_embedding.utils.text_from_html import html_to_page_text

DEFAULT_URL_FILE = ROOT / "rawdata" / "urls" / "all_crawled_pages.txt"
DEFAULT_OUT = ROOT / "data" / "ssl_crawl" / "page_text_corpus"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 SSL-RAG-Corpus/1.0"
)


def _read_urls(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    # dedupe preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def _safe_filename(url: str) -> str:
    p = urlparse(url)
    host = p.netloc.replace(".", "_")
    path = (p.path or "/").strip("/").replace("/", "_")
    if not path:
        path = "index"
    base = f"{host}_{path}"[:120]
    base = re.sub(r"[^\w\-_.]", "_", base)
    h = hashlib.sha256(url.encode()).hexdigest()[:10]
    return f"{base}__{h}.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect visible page text for SSL RAG corpus.")
    ap.add_argument(
        "--url-file",
        type=Path,
        default=DEFAULT_URL_FILE,
        help=f"Text file with URLs (default: {DEFAULT_URL_FILE})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout seconds",
    )
    args = ap.parse_args()

    url_file = args.url_file.resolve()
    if not url_file.is_file():
        print(f"Missing URL file: {url_file}", file=sys.stderr)
        return 1

    out_dir: Path = args.out.resolve()
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    records: list[dict[str, object]] = []
    for url in _read_urls(url_file):
        try:
            r = session.get(url, timeout=args.timeout, allow_redirects=True)
            r.raise_for_status()
        except requests.RequestException as exc:
            records.append({"url": url, "ok": False, "error": str(exc)})
            continue

        ctype = (r.headers.get("Content-Type") or "").lower()
        raw = r.text or ""
        looks_html = "html" in ctype or "<html" in raw[:2000].lower() or "<!doctype" in raw[:200].lower()
        if not looks_html:
            records.append(
                {
                    "url": url,
                    "ok": False,
                    "error": f"non-html content-type: {ctype}",
                }
            )
            continue

        title, text = html_to_page_text(raw)
        fname = _safe_filename(r.url)
        rel = f"pages/{fname}"
        dest = pages_dir / fname
        body = ""
        if title:
            body += f"{title}\n\n"
        body += text
        dest.write_text(body, encoding="utf-8")

        records.append(
            {
                "url": url,
                "final_url": r.url,
                "ok": True,
                "title": title,
                "text_file": rel,
                "chars": len(text),
                "bytes_written": dest.stat().st_size,
            }
        )

    manifest = {
        "source_url_file": str(url_file),
        "pages_ok": sum(1 for x in records if x.get("ok")),
        "pages_failed": sum(1 for x in records if not x.get("ok")),
        "records": records,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "out": str(out_dir),
                "ok": manifest["pages_ok"],
                "failed": manifest["pages_failed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if manifest["pages_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
