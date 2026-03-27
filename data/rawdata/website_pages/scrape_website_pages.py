"""
scrape_website_pages.py
-----------------------
Scrapes the SSL website (www.umb.edu/ssl/) and saves each page as a
JSON file in the SAME directory as this script.

Usage:
    python scrape_website_pages.py
"""

import os
import json
import time
import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

# --- CONFIG ------------------------------------------------------------------
BASE_URL   = "https://www.umb.edu/ssl"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # save next to this script
DELAY      = 1.0
HEADERS    = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Seed pages -- only URLs confirmed to exist (no trailing slash on events)
SEED_URLS = [
    "https://www.umb.edu/ssl/",
    "https://www.umb.edu/ssl/research/",
    "https://www.umb.edu/ssl/people/",
    "https://www.umb.edu/ssl/people/board-of-directors/",
    "https://www.umb.edu/ssl/people/students/",
    "https://www.umb.edu/ssl/people/university-affiliates/",
    "https://www.umb.edu/ssl/projects/",
    "https://www.umb.edu/ssl/events",                                        # no trailing slash!
    "https://www.umb.edu/directory/?department=sustainable+solutions+lab",
    "https://caps.umb.edu/ssl/faculty_grants/community_of_practice",
    "https://www.umb.edu/news/recent-news/beyond-survival-Sustainable-Solutions-Labs-Drive",
]
# -----------------------------------------------------------------------------

def clean_text(soup):
    for tag in soup(["nav", "footer", "script", "style", "header", "aside",
                     "noscript", "iframe", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Remove common UMB nav noise
    noise = [
        "honeypot link", "Current Students", "Parents & Families",
        "Faculty & Staff", "Alumni", "Skip to Main Content",
    ]
    for n in noise:
        text = text.replace(n, "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def slugify(url):
    slug = re.sub(r"https?://[^/]+/", "", url).rstrip("/")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", slug) or "index"
    return slug[:80]

def scrape_and_save(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print("  [FAIL] {}: {}".format(url, e))
        return None

    soup  = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    text  = clean_text(soup)

    record = {
        "title":       title,
        "url":         resp.url,  # use final URL after redirects
        "source_type": "website_page",
        "text":        text,
        "scraped_at":  datetime.now(timezone.utc).isoformat(),
    }

    fname = slugify(url) + ".json"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print("  [OK] Saved: {}  ({} chars)".format(fname, len(text)))

    discovered = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/ssl/") or href.startswith("https://www.umb.edu/ssl/"):
            full = "https://www.umb.edu" + href if href.startswith("/") else href
            full = full.split("#")[0]
            if full not in visited and "umb.edu/ssl" in full:
                discovered.append(full)
    return discovered

visited = set()
queue   = list(SEED_URLS)

print("Starting SSL website scrape -> {}\n".format(OUTPUT_DIR))

while queue:
    url = queue.pop(0)
    if url in visited:
        continue
    visited.add(url)
    print("-> {}".format(url))
    new_links = scrape_and_save(url)
    if new_links:
        for link in new_links:
            if link not in visited:
                queue.append(link)
    time.sleep(DELAY)

print("\nDone. Scraped {} pages -> {}".format(len(visited), OUTPUT_DIR))
