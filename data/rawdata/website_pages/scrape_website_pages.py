"""
scrape_website_pages.py
-----------------------
Scrapes the SSL website (www.umb.edu/ssl/) and saves each page as a
JSON file in rawdata/website_pages/.

Each saved file contains:
  - title       : page title
  - url         : source URL
  - source_type : "website_page"
  - text        : clean extracted text
  - scraped_at  : timestamp

Usage:
    python scrape_website_pages.py
"""

import os
import json
import time
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL   = "https://www.umb.edu/ssl"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "website_pages")
DELAY      = 1.0   # seconds between requests (be polite to the server)
HEADERS    = {"User-Agent": "Mozilla/5.0 (UMass RAG Project Crawler)"}

# Seed pages to start from — add more as you discover them
SEED_URLS = [
    "https://www.umb.edu/ssl/",
    "https://www.umb.edu/ssl/about/",
    "https://www.umb.edu/ssl/research/",
    "https://www.umb.edu/ssl/people/",
    "https://www.umb.edu/ssl/people/board-of-directors/",
    "https://www.umb.edu/ssl/people/students/",
    "https://www.umb.edu/ssl/people/university-affiliates/",
    "https://www.umb.edu/ssl/projects/",
    "https://www.umb.edu/ssl/news/",
    "https://www.umb.edu/ssl/events/",
    "https://www.umb.edu/ssl/community-partners/",
    "https://www.umb.edu/ssl/contact/",
    "https://www.umb.edu/directory/?department=sustainable+solutions+lab",
]
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(soup):
    """Remove nav/footer/script noise and return clean body text."""
    for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def slugify(url):
    """Turn a URL into a safe filename."""
    slug = re.sub(r"https?://[^/]+/", "", url).rstrip("/")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug) or "index"
    return slug[:80]

def scrape_and_save(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ✗ Failed to fetch {url}: {e}")
        return None

    soup  = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string.strip() if soup.title else url
    text  = clean_text(soup)

    record = {
        "title":       title,
        "url":         url,
        "source_type": "website_page",
        "text":        text,
        "scraped_at":  datetime.utcnow().isoformat() + "Z",
    }

    fname = slugify(url) + ".json"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved: {fname}  ({len(text)} chars)")

    # Discover additional SSL sub-pages linked from this page
    discovered = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/ssl/") or href.startswith("https://www.umb.edu/ssl/"):
            full = "https://www.umb.edu" + href if href.startswith("/") else href
            full = full.split("?")[0].split("#")[0]   # strip query/fragment
            if full not in visited and full.startswith("https://www.umb.edu/ssl/"):
                discovered.append(full)
    return discovered

visited = set()
queue   = list(SEED_URLS)

print(f"Starting SSL website scrape → {OUTPUT_DIR}\n")

while queue:
    url = queue.pop(0)
    if url in visited:
        continue
    visited.add(url)
    print(f"→ {url}")
    new_links = scrape_and_save(url)
    if new_links:
        for link in new_links:
            if link not in visited:
                queue.append(link)
    time.sleep(DELAY)

print(f"\nDone. Scraped {len(visited)} pages → {OUTPUT_DIR}")
