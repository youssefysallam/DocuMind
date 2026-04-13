"""Read URL lists from text files in a directory (one URL per line)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Match first http(s) URL on a line (allows trailing punctuation trim)
_URL_RE = re.compile(
    r"https?://[^\s#]+", re.IGNORECASE
)

_GLOB_PATTERNS = ("*.txt", "*.url", "*.list")


def _parse_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith(("http://", "https://")):
        return line.rstrip()
    m = _URL_RE.search(line)
    if not m:
        return None
    u = m.group(0).rstrip(".,);]\"'")
    if u.startswith(("http://", "https://")):
        return u
    return None


def read_urls_from_file(path: Path) -> list[str]:
    """Parse one UTF-8 text file; one URL per line; # starts a comment line."""
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []
    for line in text.splitlines():
        u = _parse_line(line)
        if u:
            out.append(u)
    return out


def read_urls_from_directory(
    directory: Path,
    *,
    recursive: bool = False,
) -> list[str]:
    """
    Read all *.txt, *.url, *.list files under `directory`.

    Order: files sorted by name; within each file, top-to-bottom.
    De-duplicates URLs while preserving first-seen order.
    """
    directory = directory.resolve()
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))

    files_set: dict[str, Path] = {}
    if recursive:
        for pat in _GLOB_PATTERNS:
            for f in directory.rglob(pat):
                files_set[f.resolve().as_posix()] = f
    else:
        for pat in _GLOB_PATTERNS:
            for f in directory.glob(pat):
                files_set[f.resolve().as_posix()] = f

    files = sorted(files_set.values(), key=lambda p: p.as_posix().lower())
    # Skip hidden / backup names
    files = [f for f in files if not f.name.startswith(".")]

    seen: dict[str, None] = {}
    ordered: list[str] = []
    for fp in files:
        try:
            for u in read_urls_from_file(fp):
                if u not in seen:
                    seen[u] = None
                    ordered.append(u)
        except OSError as exc:
            logger.warning("Skip URL file %s: %s", fp, exc)
    logger.info(
        "Loaded %d unique URL(s) from %d file(s) in %s",
        len(ordered),
        len(files),
        directory,
    )
    return ordered
