"""Website content extraction with requests + BeautifulSoup."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from raw_to_embedding.config import get_settings
from raw_to_embedding.models import RawDocument, WebsiteSection

logger = logging.getLogger(__name__)

_NOISE_SELECTORS = (
    "nav",
    "header",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "form",
    "[role=navigation]",
    "[aria-label*=cookie i]",
    "[class*=cookie i]",
    "[id*=cookie i]",
    "[class*=banner i]",
    "[class*=newsletter i]",
)


def _strip_noise(soup: BeautifulSoup) -> None:
    for sel in _NOISE_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()


def _ordered_blocks(main: BeautifulSoup) -> list:
    """Collect h1–h3, p, li in document order; avoid duplicate nested p inside li."""
    blocks = []
    for el in main.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
        if el.name == "p" and el.find_parent("li") is not None:
            continue
        blocks.append(el)
    return blocks


def _extract_sections(soup: BeautifulSoup, url: str) -> list[WebsiteSection]:
    """Split main/article/body into sections on heading boundaries."""
    main = soup.find("main") or soup.find("article") or soup.body
    if not main:
        return [
            WebsiteSection(
                heading_level=None,
                heading_text=None,
                content=soup.get_text("\n", strip=True),
                url=url,
            )
        ]

    blocks = _ordered_blocks(main)
    sections: list[WebsiteSection] = []
    cur_level: int | None = None
    cur_heading: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, cur_level, cur_heading
        text = "\n".join(t for t in buf if t).strip()
        if text:
            sections.append(
                WebsiteSection(
                    heading_level=cur_level,
                    heading_text=cur_heading,
                    content=text,
                    url=url,
                )
            )
        buf = []

    for el in blocks:
        if el.name in {"h1", "h2", "h3"}:
            flush()
            cur_level = int(el.name[1])
            cur_heading = el.get_text(strip=True)
            continue
        piece = el.get_text(" ", strip=True)
        if piece:
            buf.append(piece)
    flush()

    if not sections:
        text = main.get_text("\n", strip=True)
        sections.append(WebsiteSection(heading_level=None, heading_text=None, content=text, url=url))
    return sections


def fetch_website(url: str) -> RawDocument:
    """Fetch URL and return structured website sections."""
    settings = get_settings()
    headers = {"User-Agent": settings.http_user_agent}
    try:
        resp = requests.get(url, headers=headers, timeout=settings.http_timeout_seconds)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("HTTP error for %s: %s", url, exc)
        raise RuntimeError(f"Failed to fetch URL: {url}") from exc

    ctype = resp.headers.get("Content-Type", "")
    if "html" not in ctype.lower():
        logger.warning("Unexpected content-type for %s: %s", url, ctype)

    soup = BeautifulSoup(resp.text, "html.parser")
    _strip_noise(soup)
    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else ""
    sections = _extract_sections(soup, url)
    raw_parts: list[str] = []
    for sec in sections:
        if sec.heading_text:
            raw_parts.append(f"{sec.heading_text}\n{sec.content}")
        else:
            raw_parts.append(sec.content)
    raw_text = "\n\n".join(raw_parts)
    if page_title:
        raw_text = f"{page_title}\n\n{raw_text}"
    parsed = urlparse(url)
    source = f"{parsed.netloc}{parsed.path}" or url
    return RawDocument(
        source=source,
        source_type="website",
        raw_text=raw_text,
        url=url,
        path=None,
        pages=None,
        sections=sections,
    )
