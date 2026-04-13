"""Fetch ScholarWorks SSL index HTML + extract PDF link list (no PDF bytes — AWS WAF)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse, urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://scholarworks.umb.edu/ssl/"
UA = "Mozilla/5.0 (compatible; SSL-Export/1.0; +https://www.umb.edu/ssl/)"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "ssl_crawl"


def main() -> None:
    r = requests.get(BASE, timeout=60, headers={"User-Agent": UA})
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    pdfs: list[dict[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if "viewcontent.cgi" not in href:
            continue
        abs_url = urljoin(BASE, href)
        title = a.get("aria-label") or a.get("title") or ""
        m = re.search(r"PDF of (.+?)(?:\s*\(\d)", title)
        label = m.group(1).strip() if m else title
        art = (parse_qs(urlparse(abs_url).query).get("article") or [""])[0]
        pdfs.append({"url": abs_url, "article_id": art, "title": label})

    seen: set[str] = set()
    uniq: list[dict[str, str]] = []
    for p in pdfs:
        if p["url"] not in seen:
            seen.add(p["url"])
            uniq.append(p)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scholarworks_ssl_index.txt").write_text(text, encoding="utf-8")
    (OUT_DIR / "scholarworks_ssl_index.html").write_text(html, encoding="utf-8")

    url_lines = [p["url"] for p in uniq]
    (OUT_DIR / "scholarworks_pdf_urls.txt").write_text(
        "\n".join(url_lines) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "source_urls": [
            BASE,
            "https://scholarworks.umb.edu/ssl/index.html#article",
            "https://scholarworks.umb.edu/ssl/index.html#executive_summary",
            "https://scholarworks.umb.edu/ssl/index.html#researchreport",
        ],
        "note": "Download PDFs with Playwright; see scripts/fetch_scholarworks_ssl_playwright.py",
        "pdf_links": uniq,
    }
    (OUT_DIR / "scholarworks_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PDF links found:", len(uniq), "text chars:", len(text))


if __name__ == "__main__":
    main()
