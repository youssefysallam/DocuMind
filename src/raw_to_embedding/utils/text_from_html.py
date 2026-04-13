"""Extract main visible text from HTML for RAG (noise-stripped)."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

_BOILER_SELECTORS = (
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "script",
    "style",
    "noscript",
    "[role=navigation]",
    "[aria-label*=cookie i]",
    "[class*=cookie i]",
    "[id*=cookie i]",
)

_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


def strip_boilerplate(soup: BeautifulSoup) -> None:
    for sel in _BOILER_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()


def pick_main_container(soup: BeautifulSoup) -> Tag:
    for sel in ("main", "[role=main]", "article", "#content", "#main", ".main-content"):
        node = soup.select_one(sel)
        if node:
            return node
    return soup.body or soup


def clean_whitespace(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        line = _WS.sub(" ", line.strip())
        if line:
            lines.append(line)
    text = "\n".join(lines)
    text = _NL.sub("\n\n", text)
    return text.strip()


def html_to_page_text(html: str) -> tuple[str | None, str]:
    """
    Return (title, main_visible_text) with boilerplate removed.
    """
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None
    strip_boilerplate(soup)
    main = pick_main_container(soup)
    text = main.get_text("\n", strip=True)
    text = clean_whitespace(text)
    return title, text
