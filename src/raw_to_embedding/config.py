"""Application configuration and tunable segmentation thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# If .env sets OPENAI_BASE_URL to an empty value, dotenv stores "" and the OpenAI SDK treats it as a base URL.
# Drop keys that are only whitespace so "unset" uses the default API host.
def _unset_empty_env(*names: str) -> None:
    for name in names:
        v = os.environ.get(name)
        if v is not None and not str(v).strip():
            os.environ.pop(name, None)


_unset_empty_env("OPENAI_BASE_URL")


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment with sensible defaults."""

    openai_api_key: str | None
    openai_base_url: str | None
    llm_model: str
    llm_temperature: float
    llm_timeout_seconds: float

    # When to invoke LLM semantic segmentation
    max_unit_chars_for_skip_llm: int
    min_chars_multi_topic_heuristic: int
    generic_title_patterns: tuple[str, ...]

    # Embedding chunk packing (sentence groups)
    max_chunk_chars_soft: int
    max_sentences_per_group: int
    chunk_overlap_sentences: int

    # HTTP
    http_timeout_seconds: float
    http_user_agent: str

    # Debug / IO
    default_output_dir: Path


def _env_str(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val


def get_settings() -> Settings:
    """Load settings from environment variables."""
    generic_raw = _env_str(
        "GENERIC_TITLE_PATTERNS",
        "overview,introduction,programs,new initiatives,updates,news,about us,our work",
    )
    generic_patterns = tuple(
        p.strip().lower() for p in (generic_raw or "").split(",") if p.strip()
    )
    return Settings(
        openai_api_key=_env_str("OPENAI_API_KEY"),
        openai_base_url=_env_str("OPENAI_BASE_URL"),
        llm_model=_env_str("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        llm_temperature=float(_env_str("LLM_TEMPERATURE", "0") or "0"),
        llm_timeout_seconds=float(_env_str("LLM_TIMEOUT_SECONDS", "120") or "120"),
        max_unit_chars_for_skip_llm=int(_env_str("MAX_UNIT_CHARS_SKIP_LLM", "3500") or "3500"),
        min_chars_multi_topic_heuristic=int(
            _env_str("MIN_CHARS_MULTI_TOPIC_HEURISTIC", "4500") or "4500"
        ),
        generic_title_patterns=generic_patterns,
        max_chunk_chars_soft=int(_env_str("MAX_CHUNK_CHARS_SOFT", "800") or "800"),
        max_sentences_per_group=int(_env_str("MAX_SENTENCES_PER_GROUP", "5") or "5"),
        chunk_overlap_sentences=int(_env_str("CHUNK_OVERLAP_SENTENCES", "2") or "2"),
        http_timeout_seconds=float(_env_str("HTTP_TIMEOUT_SECONDS", "30") or "30"),
        http_user_agent=_env_str(
            "HTTP_USER_AGENT",
            "raw-to-embedding-agent/1.0 (+https://example.org)",
        )
        or "raw-to-embedding-agent/1.0",
        default_output_dir=Path(_env_str("DEFAULT_OUTPUT_DIR", "./output") or "./output"),
    )
