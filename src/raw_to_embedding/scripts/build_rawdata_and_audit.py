#!/usr/bin/env python3
"""
Normalize SSL crawl outputs into rawdata/, HTTP-check URLs, and gap-check ScholarWorks index links.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "rawdata"
SRC_CRAWL = ROOT / "data" / "ssl_crawl"

WWW_URLS = [
    "https://www.umb.edu/ssl",
    "https://www.umb.edu/ssl/people",
    "https://www.umb.edu/ssl/projects",
    "https://www.umb.edu/ssl/research",
    "https://www.umb.edu/ssl/people/board-of-directors",
    "https://www.umb.edu/ssl/people/students",
    "https://www.umb.edu/ssl/people/university-affiliates",
]
SW_PAGES = [
    "https://scholarworks.umb.edu/ssl",
    "https://scholarworks.umb.edu/ssl/10",
    "https://scholarworks.umb.edu/ssl/1",
    "https://scholarworks.umb.edu/ssl/15",
    "https://scholarworks.umb.edu/ssl/11",
    "https://scholarworks.umb.edu/ssl/13",
    "https://scholarworks.umb.edu/ssl/7",
    "https://scholarworks.umb.edu/ssl/6",
    "https://scholarworks.umb.edu/ssl/8",
    "https://scholarworks.umb.edu/ssl/3",
    "https://scholarworks.umb.edu/ssl/9",
    "https://scholarworks.umb.edu/ssl/4",
    "https://scholarworks.umb.edu/ssl/5",
    "https://scholarworks.umb.edu/ssl/14",
    "https://scholarworks.umb.edu/ssl/2",
    "https://scholarworks.umb.edu/ssl/12",
    "https://scholarworks.umb.edu/ssl/announcements.html",
]
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def write_lines(path: Path, lines: list[str], header: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    if header:
        parts.append(header.rstrip() + "\n")
    parts.extend(l + "\n" for l in lines if l.strip())
    path.write_text("".join(parts), encoding="utf-8")


def copy_pdfs() -> dict[str, list[str]]:
    """Copy PDFs into rawdata/pdfs/*; return relative paths for manifest."""
    out: dict[str, list[str]] = {"www_umb_media": [], "scholarworks": []}
    src_media = SRC_CRAWL / "pdfs"
    dst_media = RAW / "pdfs" / "www_umb_media"
    if src_media.is_dir():
        dst_media.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_media.glob("*.pdf")):
            shutil.copy2(f, dst_media / f.name)
            out["www_umb_media"].append(str((dst_media / f.name).relative_to(ROOT)))

    src_sw = SRC_CRAWL / "scholarworks_pdfs"
    dst_sw = RAW / "pdfs" / "scholarworks"
    if src_sw.is_dir():
        dst_sw.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_sw.glob("article*.pdf")):
            shutil.copy2(f, dst_sw / f.name)
            out["scholarworks"].append(str((dst_sw / f.name).relative_to(ROOT)))

    return out


def http_check(urls: list[str]) -> list[dict]:
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    rows = []
    for url in urls:
        try:
            r = sess.get(url, timeout=45, allow_redirects=True)
            ct = (r.headers.get("Content-Type") or "")[:80]
            row: dict = {
                "url": url,
                "final_url": str(r.url),
                "status": r.status_code,
                "content_type": ct,
                "bytes": len(r.content),
                "ok": r.status_code < 400,
            }
            if "viewcontent.cgi" in url and r.status_code in (202, 403) and len(r.content) < 5000:
                row["note"] = (
                    "ScholarWorks viewcontent: plain GET often hits WAF or returns an empty body; "
                    "use PDFs under rawdata/pdfs/scholarworks/ (downloaded via Playwright) as source of truth."
                )
            rows.append(row)
        except requests.RequestException as exc:
            rows.append({"url": url, "error": str(exc), "ok": False})
    return rows


def gap_check_scholarworks_index(listed_viewcontent: set[str]) -> dict:
    """Compare index HTML on disk / live fetch for viewcontent links not in our list."""
    index_path = SRC_CRAWL / "scholarworks_ssl_index.html"
    base = "https://scholarworks.umb.edu/ssl/"
    from_file: set[str] = set()
    if index_path.is_file():
        soup = BeautifulSoup(index_path.read_text(encoding="utf-8"), "html.parser")
        for a in soup.find_all("a", href=True):
            h = (a.get("href") or "").strip()
            if "viewcontent.cgi" in h:
                from_file.add(urljoin(base, h))

    live_found: set[str] = set()
    live_error: str | None = None
    try:
        r = requests.get(base, timeout=60, headers={"User-Agent": UA})
        if r.ok:
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                h = (a.get("href") or "").strip()
                if "viewcontent.cgi" in h:
                    live_found.add(urljoin(base, h))
        else:
            live_error = f"HTTP {r.status_code}"
    except requests.RequestException as exc:
        live_error = str(exc)

    union = from_file | live_found
    new_on_live = sorted(live_found - from_file) if live_found else []
    missing_from_txt = sorted(union - listed_viewcontent)

    return {
        "viewcontent_from_saved_index_html": len(from_file),
        "viewcontent_from_live_index": len(live_found),
        "live_fetch_error": live_error,
        "urls_in_page_but_not_in_scholarworks_pdf_urls_txt": missing_from_txt,
        "new_links_on_live_vs_saved_html": new_on_live,
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "urls").mkdir(parents=True, exist_ok=True)
    (RAW / "audit").mkdir(parents=True, exist_ok=True)

    all_pages = sorted(set(WWW_URLS + SW_PAGES))
    write_lines(
        RAW / "urls" / "www_ssl_pages.txt",
        WWW_URLS,
        "# UMass Boston marketing site — SSL section",
    )
    write_lines(
        RAW / "urls" / "scholarworks_ssl_pages.txt",
        SW_PAGES,
        "# ScholarWorks — SSL digital series (HTML landing pages)",
    )
    write_lines(
        RAW / "urls" / "all_crawled_pages.txt",
        all_pages,
        "# All HTML pages from prior crawl (deduped)",
    )

    # PDF URLs
    media_urls: list[str] = []
    pj = SRC_CRAWL / "pdf_links.json"
    if pj.is_file():
        for item in json.loads(pj.read_text(encoding="utf-8")):
            media_urls.append(item["normalized_url"])
    write_lines(
        RAW / "urls" / "www_umb_media_pdf_urls.txt",
        sorted(set(media_urls)),
        "# Direct PDFs on www.umb.edu (from SSL crawl — research/home)",
    )

    sw_pdf_urls: list[str] = []
    sp = SRC_CRAWL / "scholarworks_pdf_urls.txt"
    if sp.is_file():
        for line in sp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                sw_pdf_urls.append(line)
    write_lines(
        RAW / "urls" / "scholarworks_viewcontent_pdf_urls.txt",
        sorted(set(sw_pdf_urls)),
        "# ScholarWorks viewcontent.cgi (same set as index #article / #executive_summary / #researchreport)",
    )

    pdf_files = copy_pdfs()

    listed_vc = set(sw_pdf_urls)
    gap = gap_check_scholarworks_index(listed_vc)

    page_checks = http_check(all_pages)
    pdf_url_checks = http_check(sorted(set(media_urls + sw_pdf_urls)))

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "page_url_checks": page_checks,
        "pdf_url_checks": pdf_url_checks,
        "scholarworks_gap_analysis": gap,
        "pdf_files_copied": pdf_files,
        "counts": {
            "html_pages": len(all_pages),
            "www_umb_media_pdf_urls": len(media_urls),
            "scholarworks_pdf_urls": len(sw_pdf_urls),
            "files_www_umb_media": len(pdf_files.get("www_umb_media", [])),
            "files_scholarworks": len(pdf_files.get("scholarworks", [])),
        },
    }
    (RAW / "audit" / "audit_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Root manifest
    ingest = RAW / "urls" / "ingest"
    ingest.mkdir(parents=True, exist_ok=True)
    write_lines(
        ingest / "urls.txt",
        all_pages,
        "# Single merged URL list for --url-dir ./rawdata/urls/ingest (avoid duplicate txts in the same folder)",
    )

    manifest = {
        "description": "SSL crawl rawdata: page URLs + PDF paths for raw_to_embedding",
        "urls_dir": "rawdata/urls/",
        "ingest_url_dir": "rawdata/urls/ingest/",
        "pdfs_dir": "rawdata/pdfs/",
        "audit": "rawdata/audit/audit_report.json",
        "embedding_agent_examples": {
            "url_dir": "python main.py --url-dir ./rawdata/urls/ingest",
            "pdf_globs": "python main.py --pdf ./rawdata/pdfs/www_umb_media/*.pdf ./rawdata/pdfs/scholarworks/*.pdf",
        },
    }
    (RAW / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("rawdata ready:", RAW)
    print(json.dumps(audit["counts"], indent=2))


if __name__ == "__main__":
    main()
