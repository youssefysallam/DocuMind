"""
scrape_scholarworks.py
----------------------
Downloads all SSL publications from scholarworks.umb.edu/ssl/.

* Institute reports (type = "report")  -> saved to rawdata/institute_report/
* Scholarly papers (type = "article")  -> saved to rawdata/scholary_papers/

Each download saves:
  1. The PDF file itself (e.g. "ssl_12.pdf")
  2. A metadata JSON sidecar (e.g. "ssl_12.json") with:
       - title, authors, abstract, url, source_type, pub_date, downloaded_at

Usage:
    python scrape_scholarworks.py
"""

import os
import json
import time
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# --- CONFIG ------------------------------------------------------------------
BASE_URL    = "https://scholarworks.umb.edu/ssl/"
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "institute_report")
PAPERS_DIR  = os.path.join(os.path.dirname(__file__), "scholary_papers")
DELAY       = 1.5
MAX_PAGES   = 50   # how many numbered entries to try (ssl/1 ... ssl/MAX_PAGES)
HEADERS     = {"User-Agent": "Mozilla/5.0 (UMass RAG Project Crawler)"}
# -----------------------------------------------------------------------------

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PAPERS_DIR,  exist_ok=True)


def classify_document(soup, title, abstract):
    """
    Decide if this is an institute report or a scholarly paper.
    Heuristic: look for 'report', 'brief', 'summary' keywords -> report
    Otherwise -> scholarly paper / article
    """
    text = (title + " " + abstract).lower()
    report_keywords = ["report", "brief", "summary", "executive", "white paper",
                       "policy", "recommendations", "assessment"]
    for kw in report_keywords:
        if kw in text:
            return "institute_report"
    return "scholarly_paper"


def clean_author_name(raw):
    """Remove 'Follow', affiliation noise, and extra whitespace from an author name."""
    name = re.sub(r"Follow.*", "", raw, flags=re.IGNORECASE).strip()
    name = re.sub(r",\s*(University|UMass|Massachusetts).*", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"\s{2,}", " ", name).strip(" ,;")
    return name


def extract_metadata(url, soup):
    """Pull title, authors, abstract, pub_date from a bepress Digital Commons page."""
    title    = ""
    authors  = []
    abstract = ""
    pub_date = ""

    # --- TITLE ---------------------------------------------------------------
    # bepress uses <meta name="citation_title"> or an <h1> / #title
    meta_title = soup.find("meta", attrs={"name": "citation_title"})
    if meta_title and meta_title.get("content"):
        title = meta_title["content"].strip()
    else:
        for sel in ["#title", "h1", "h2"]:
            tag = soup.select_one(sel)
            if tag:
                title = tag.get_text(strip=True)
                break

    # --- AUTHORS -------------------------------------------------------------
    # Strategy 1: <meta name="citation_author"> tags (most reliable)
    meta_authors = soup.find_all("meta", attrs={"name": "citation_author"})
    if meta_authors:
        for ma in meta_authors:
            name = (ma.get("content") or "").strip()
            if name:
                authors.append(name)
    else:
        # Strategy 2: #bp_authors or #authors container
        author_container = soup.find(id="bp_authors") or soup.find(id="authors")
        if author_container:
            for a in author_container.find_all("a"):
                name = clean_author_name(a.get_text(strip=True))
                if name and len(name) > 2:
                    authors.append(name)

        # Strategy 3: any element with class containing "author"
        if not authors:
            for tag in soup.find_all(["p", "div", "span", "a"]):
                cls = " ".join(tag.get("class", []))
                if "author" in cls.lower():
                    raw_text = tag.get_text(strip=True)
                    # Split on "Follow" boundaries (bepress-specific)
                    parts = re.split(r"Follow", raw_text)
                    for part in parts:
                        name = clean_author_name(part)
                        if name and len(name) > 2 and name not in authors:
                            authors.append(name)

    # Deduplicate while preserving order
    seen = set()
    unique_authors = []
    for a in authors:
        if a.lower() not in seen:
            seen.add(a.lower())
            unique_authors.append(a)
    authors = unique_authors

    # --- ABSTRACT ------------------------------------------------------------
    # Strategy 1: <meta name="description"> (common in bepress)
    meta_desc = soup.find("meta", attrs={"name": "description"})

    # Strategy 2: #abstract div (bepress standard)
    abstract_div = (
        soup.find(id="abstract")
        or soup.find(id="bp_abstract")
        or soup.find("div", class_=re.compile(r"abstract", re.I))
        or soup.find("blockquote")
    )
    if abstract_div:
        abstract = abstract_div.get_text(strip=True)
        abstract = re.sub(r"^Abstract\s*", "", abstract, flags=re.IGNORECASE).strip()

    # Fall back to meta description if we got nothing
    if not abstract and meta_desc and meta_desc.get("content"):
        abstract = meta_desc["content"].strip()

    # Strategy 3: first long paragraph in the main content area
    if not abstract:
        main_area = (soup.find("div", id="main-content")
                     or soup.find("article")
                     or soup.find("main")
                     or soup)
        for p in main_area.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 100 and "cookie" not in text.lower() and "javascript" not in text.lower():
                abstract = text
                break

    # --- PUBLICATION DATE ----------------------------------------------------
    # Strategy 1: citation meta tags
    for meta_name in ["citation_date", "citation_publication_date", "citation_online_date"]:
        md = soup.find("meta", attrs={"name": meta_name})
        if md and md.get("content"):
            pub_date = md["content"].strip()
            break

    # Strategy 2: regex fallback for a 4-digit year
    if not pub_date:
        date_pat = re.compile(r"\b(19|20)\d{2}\b")
        for tag in soup.find_all(["p", "span", "td", "li", "div"]):
            txt = tag.get_text()
            m = date_pat.search(txt)
            if m:
                pub_date = m.group()
                break

    return {
        "title":         title or url,
        "authors":       authors,
        "abstract":      abstract,
        "url":           url,
        "pub_date":      pub_date,
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
    }


def find_pdf_link(soup, page_url, entry_num):
    """
    Find the PDF download URL on a bepress Digital Commons page.
    Tries multiple strategies since bepress does not use simple .pdf links.
    """
    base = "https://scholarworks.umb.edu"

    # Strategy 1: <meta name="citation_pdf_url"> (most reliable in bepress)
    meta_pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta_pdf and meta_pdf.get("content"):
        url = meta_pdf["content"].strip()
        return url if url.startswith("http") else base + url

    # Strategy 2: link containing "viewcontent.cgi" or "download"
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "viewcontent.cgi" in href or "download" in href.lower():
            return href if href.startswith("http") else base + href

    # Strategy 3: direct .pdf links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            return href if href.startswith("http") else base + href

    # Strategy 4: link text says "Download" or "Full Text"
    for a in soup.find_all("a", href=True):
        link_text = a.get_text(strip=True).lower()
        if "download" in link_text or "full text" in link_text:
            href = a["href"]
            return href if href.startswith("http") else base + href

    # Strategy 5: guess the bepress viewcontent URL and verify with a HEAD request
    guessed_url = "{}/cgi/viewcontent.cgi?article={:04d}&context=ssl".format(base, entry_num)
    try:
        head_resp = requests.head(guessed_url, headers=HEADERS, timeout=10, allow_redirects=True)
        if head_resp.status_code == 200:
            content_type = head_resp.headers.get("Content-Type", "")
            if "pdf" in content_type.lower() or "octet" in content_type.lower():
                return guessed_url
    except Exception:
        pass

    return None


def download_entry(entry_num):
    url = "https://scholarworks.umb.edu/ssl/{}/".format(entry_num)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            return False   # entry does not exist
        resp.raise_for_status()
    except Exception as e:
        print("  [FAIL] {}: {}".format(url, e))
        return True  # might be a temporary error, keep going

    soup = BeautifulSoup(resp.text, "html.parser")
    meta = extract_metadata(url, soup)

    doc_type = classify_document(soup, meta["title"], meta["abstract"])
    out_dir  = REPORTS_DIR if doc_type == "institute_report" else PAPERS_DIR
    meta["source_type"] = doc_type

    slug      = "ssl_{}".format(entry_num)
    json_path = os.path.join(out_dir, slug + ".json")
    pdf_path  = os.path.join(out_dir, slug + ".pdf")

    # Save metadata JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Try to download PDF
    pdf_url = find_pdf_link(soup, url, entry_num)
    if pdf_url:
        try:
            pdf_resp = requests.get(pdf_url, headers=HEADERS, timeout=30, allow_redirects=True)
            pdf_resp.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(pdf_resp.content)
            meta["pdf_file"]    = os.path.basename(pdf_path)
            meta["pdf_size_kb"] = round(len(pdf_resp.content) / 1024, 1)
            print("  [OK] [{}] #{}: {}  (PDF {} KB)".format(
                doc_type, entry_num, meta["title"][:55], meta["pdf_size_kb"]))
        except Exception as e:
            print("  [!]  [{}] #{}: metadata saved but PDF failed: {}".format(
                doc_type, entry_num, e))
    else:
        print("  [OK] [{}] #{}: {}  (no PDF found - metadata only)".format(
            doc_type, entry_num, meta["title"][:55]))

    # Re-save JSON with updated pdf info
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return True


if __name__ == "__main__":
    print("Starting ScholarWorks SSL scrape (entries 1-{})\n".format(MAX_PAGES))
    print("Reports ->", REPORTS_DIR)
    print("Papers  ->", PAPERS_DIR)
    print()

    consecutive_misses = 0

    for i in range(1, MAX_PAGES + 1):
        print("-> Entry {}:".format(i))
        found = download_entry(i)
        if not found:
            consecutive_misses += 1
            print("   Entry {} not found (404)".format(i))
            if consecutive_misses >= 5:
                print("5 consecutive 404s - assuming end of collection.")
                break
        else:
            consecutive_misses = 0
        time.sleep(DELAY)

    print("\nDone.")
