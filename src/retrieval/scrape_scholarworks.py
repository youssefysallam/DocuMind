"""
scrape_scholarworks.py
----------------------
Downloads all SSL publications from scholarworks.umb.edu/ssl/.

Saves institute reports and scholarly papers into the PARENT directory's
institute_report/ and scholary_papers/ folders.

Usage:
    python scrape_scholarworks.py
"""

import os
import json
import time
import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

# --- CONFIG ------------------------------------------------------------------
RAWDATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(RAWDATA_DIR, "institute_report")
PAPERS_DIR  = os.path.join(RAWDATA_DIR, "scholary_papers")
DELAY       = 2.0
MAX_PAGES   = 50

# Use a real browser User-Agent - ScholarWorks blocks bot-like agents
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
# -----------------------------------------------------------------------------

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PAPERS_DIR,  exist_ok=True)

# Use a persistent session (keeps cookies between requests - critical for bepress)
session = requests.Session()
session.headers.update(BROWSER_HEADERS)


def classify_document(soup, title, abstract):
    text = (title + " " + abstract).lower()
    report_keywords = ["report", "brief", "summary", "executive", "white paper",
                       "policy", "recommendations", "assessment", "feasibility"]
    for kw in report_keywords:
        if kw in text:
            return "institute_report"
    return "scholarly_paper"


def clean_author_name(raw):
    name = re.sub(r"Follow.*", "", raw, flags=re.IGNORECASE).strip()
    name = re.sub(r",\s*(University|UMass|Massachusetts).*", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"\s{2,}", " ", name).strip(" ,;")
    return name


def extract_metadata(url, soup):
    title    = ""
    authors  = []
    abstract = ""
    pub_date = ""

    # TITLE
    meta_title = soup.find("meta", attrs={"name": "citation_title"})
    if meta_title and meta_title.get("content"):
        title = meta_title["content"].strip()
    else:
        for sel in ["#title", "h1", "h2"]:
            tag = soup.select_one(sel)
            if tag:
                title = tag.get_text(strip=True)
                break

    # AUTHORS
    meta_authors = soup.find_all("meta", attrs={"name": "citation_author"})
    if meta_authors:
        for ma in meta_authors:
            name = (ma.get("content") or "").strip()
            if name:
                authors.append(name)
    else:
        author_container = soup.find(id="bp_authors") or soup.find(id="authors")
        if author_container:
            for a in author_container.find_all("a"):
                name = clean_author_name(a.get_text(strip=True))
                if name and len(name) > 2:
                    authors.append(name)
        if not authors:
            for tag in soup.find_all(["p", "div", "span", "a"]):
                cls = " ".join(tag.get("class", []))
                if "author" in cls.lower():
                    parts = re.split(r"Follow", tag.get_text(strip=True))
                    for part in parts:
                        name = clean_author_name(part)
                        if name and len(name) > 2 and name not in authors:
                            authors.append(name)

    seen = set()
    unique_authors = []
    for a in authors:
        if a.lower() not in seen:
            seen.add(a.lower())
            unique_authors.append(a)
    authors = unique_authors

    # ABSTRACT
    meta_desc = soup.find("meta", attrs={"name": "description"})
    abstract_div = (
        soup.find(id="abstract")
        or soup.find(id="bp_abstract")
        or soup.find("div", class_=re.compile(r"abstract", re.I))
        or soup.find("blockquote")
    )
    if abstract_div:
        abstract = abstract_div.get_text(strip=True)
        abstract = re.sub(r"^Abstract\s*", "", abstract, flags=re.IGNORECASE).strip()
    if not abstract and meta_desc and meta_desc.get("content"):
        abstract = meta_desc["content"].strip()
    if not abstract:
        main_area = soup.find("div", id="main-content") or soup.find("article") or soup
        for p in main_area.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 100 and "cookie" not in text.lower():
                abstract = text
                break

    # DATE
    for meta_name in ["citation_date", "citation_publication_date", "citation_online_date"]:
        md = soup.find("meta", attrs={"name": meta_name})
        if md and md.get("content"):
            pub_date = md["content"].strip()
            break
    if not pub_date:
        date_pat = re.compile(r"\b(19|20)\d{2}\b")
        for tag in soup.find_all(["p", "span", "td", "li", "div"]):
            m = date_pat.search(tag.get_text())
            if m:
                pub_date = m.group()
                break

    return {
        "title":         title or url,
        "authors":       authors,
        "abstract":      abstract,
        "url":           url,
        "pub_date":      pub_date,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def find_pdf_link(soup, page_url, entry_num):
    base = "https://scholarworks.umb.edu"

    # Strategy 1: citation_pdf_url meta tag
    meta_pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta_pdf and meta_pdf.get("content"):
        url = meta_pdf["content"].strip()
        return url if url.startswith("http") else base + url

    # Strategy 2: viewcontent.cgi links on the page
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "viewcontent.cgi" in href:
            return href if href.startswith("http") else base + href

    # Strategy 3: any download-looking link
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf") or "download" in href.lower():
            return href if href.startswith("http") else base + href

    # Strategy 4: button text
    for a in soup.find_all("a", href=True):
        link_text = a.get_text(strip=True).lower()
        if "download" in link_text or "full text" in link_text:
            href = a["href"]
            return href if href.startswith("http") else base + href

    return None


def download_pdf(pdf_url, pdf_path, page_url):
    """Download a PDF using a session with proper headers to avoid 403."""
    download_headers = {
        "Referer": page_url,
        "Accept": "application/pdf,*/*",
    }
    try:
        resp = session.get(pdf_url, headers=download_headers, timeout=30,
                           allow_redirects=True, stream=True)

        # If 403, try without Referer (some servers are picky)
        if resp.status_code == 403:
            resp = session.get(pdf_url, timeout=30, allow_redirects=True, stream=True)

        # If still 403, try adding an explicit cookie acceptance
        if resp.status_code == 403:
            session.get("https://scholarworks.umb.edu/ssl/", timeout=10)
            time.sleep(0.5)
            resp = session.get(pdf_url, headers=download_headers, timeout=30,
                               allow_redirects=True, stream=True)

        resp.raise_for_status()

        # Verify we actually got a PDF (not an HTML error page)
        content_type = resp.headers.get("Content-Type", "").lower()
        if "html" in content_type and "pdf" not in content_type:
            return False, "Got HTML instead of PDF (likely blocked)"

        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        size_kb = round(os.path.getsize(pdf_path) / 1024, 1)
        return True, size_kb

    except Exception as e:
        return False, str(e)


def download_entry(entry_num):
    page_url = "https://scholarworks.umb.edu/ssl/{}/".format(entry_num)
    try:
        resp = session.get(page_url, timeout=15)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
    except Exception as e:
        print("  [FAIL] {}: {}".format(page_url, e))
        return True

    soup = BeautifulSoup(resp.text, "html.parser")
    meta = extract_metadata(page_url, soup)

    doc_type = classify_document(soup, meta["title"], meta["abstract"])
    out_dir  = REPORTS_DIR if doc_type == "institute_report" else PAPERS_DIR
    meta["source_type"] = doc_type

    slug      = "ssl_{}".format(entry_num)
    json_path = os.path.join(out_dir, slug + ".json")
    pdf_path  = os.path.join(out_dir, slug + ".pdf")

    # Try to download PDF
    pdf_url = find_pdf_link(soup, page_url, entry_num)
    if pdf_url:
        meta["pdf_url"] = pdf_url
        ok, result = download_pdf(pdf_url, pdf_path, page_url)
        if ok:
            meta["pdf_file"]    = os.path.basename(pdf_path)
            meta["pdf_size_kb"] = result
            print("  [OK] [{}] #{}: {}  (PDF {} KB)".format(
                doc_type, entry_num, meta["title"][:50], result))
        else:
            print("  [!]  [{}] #{}: {}  (PDF failed: {})".format(
                doc_type, entry_num, meta["title"][:50], result))
    else:
        print("  [OK] [{}] #{}: {}  (no PDF link found)".format(
            doc_type, entry_num, meta["title"][:50]))

    # Save metadata JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return True


if __name__ == "__main__":
    # First, visit the main SSL page to establish cookies
    print("Initializing session (getting cookies)...")
    session.get("https://scholarworks.umb.edu/ssl/", timeout=15)
    time.sleep(1)

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
