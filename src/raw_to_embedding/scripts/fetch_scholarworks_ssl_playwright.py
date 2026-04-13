"""Download ScholarWorks SSL PDFs via Playwright (bypasses AWS WAF JS challenge)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "ssl_crawl"
PDF_DIR = OUT_DIR / "scholarworks_pdfs"
BASE = "https://scholarworks.umb.edu/ssl/"


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()
        page.goto(BASE, wait_until="domcontentloaded", timeout=120000)
        time.sleep(5)

        html = page.content()
        (OUT_DIR / "scholarworks_ssl_index.html").write_text(html, encoding="utf-8")
        text = page.inner_text("body")
        (OUT_DIR / "scholarworks_ssl_index.txt").write_text(text, encoding="utf-8")

        hrefs = page.eval_on_selector_all(
            'a[href*="viewcontent.cgi"]',
            "els => [...new Set(els.map(e => e.href))]",
        )
        print("Found viewcontent links:", len(hrefs))

        # One fresh page per PDF: avoids "Download is starting" / navigation race on reused page
        for href in hrefs:
            art = (parse_qs(urlparse(href).query).get("article") or [""])[0]
            p2 = context.new_page()
            try:
                with p2.expect_download(timeout=300000) as dl:
                    try:
                        p2.goto(href, wait_until="commit", timeout=120000)
                    except PlaywrightError as exc:
                        if "Download is starting" not in str(exc):
                            raise
                download = dl.value
                dest = PDF_DIR / f"article{art}.pdf"
                download.save_as(dest)
                results.append(
                    {"url": href, "article_id": art, "path": str(dest), "bytes": dest.stat().st_size}
                )
                print("OK", art, dest.name, dest.stat().st_size)
            except Exception as exc:
                results.append({"url": href, "article_id": art, "error": str(exc)})
                print("FAIL", art, exc)
            finally:
                p2.close()

        browser.close()

    manifest = {
        "source_urls": [
            BASE,
            "https://scholarworks.umb.edu/ssl/index.html#article",
            "https://scholarworks.umb.edu/ssl/index.html#executive_summary",
            "https://scholarworks.umb.edu/ssl/index.html#researchreport",
        ],
        "note": "PDFs downloaded via Playwright (bypasses AWS WAF). Anchors reference sections on the same index page.",
        "downloads": results,
        "page_html": str(OUT_DIR / "scholarworks_ssl_index.html"),
        "page_text": str(OUT_DIR / "scholarworks_ssl_index.txt"),
    }
    (OUT_DIR / "scholarworks_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
