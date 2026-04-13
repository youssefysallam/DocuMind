#!/usr/bin/env python3
"""
CLI: raw inputs (PDF paths, URLs) → extraction → cleaning → classification →
candidate units → semantic segmentation → embedding-ready JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from raw_to_embedding.classifiers.document_classifier import classify_document
from raw_to_embedding.config import get_settings
from raw_to_embedding.extractors.pdf_extractor import extract_pdf
from raw_to_embedding.extractors.website_extractor import fetch_website
from raw_to_embedding.models import CandidateUnit, DocumentType, EmbeddingChunk, RawDocument, SemanticSegment
from raw_to_embedding.processors.candidate_unit_builder import build_candidate_units
from raw_to_embedding.processors.embedding_chunk_builder import segments_to_chunks
from raw_to_embedding.processors.semantic_segmentation_agent import segment_units
from raw_to_embedding.utils.file_utils import write_json
from raw_to_embedding.utils.logging_utils import setup_logging
from raw_to_embedding.utils.text_cleaning import clean_raw_document
from raw_to_embedding.utils.url_feed import read_urls_from_directory

logger = logging.getLogger(__name__)


def _doc_id_for(path: Path | None, url: str | None) -> str:
    if path is not None:
        return hashlib.sha256(path.resolve().as_posix().encode()).hexdigest()[:12]
    if url:
        return hashlib.sha256(url.encode()).hexdigest()[:12]
    return "unknown"


def _load_pdf(path: Path) -> RawDocument:
    logger.info("Extracting PDF: %s", path)
    return extract_pdf(path)


def _load_url(url: str) -> RawDocument:
    logger.info("Fetching URL: %s", url)
    return fetch_website(url)


def _process_one(
    raw: RawDocument,
    *,
    input_path: Path | None,
    input_url: str | None,
    save_intermediate: bool,
    intermediate_dir: Path,
) -> tuple[list[CandidateUnit], list[SemanticSegment], list[EmbeddingChunk]]:
    doc_id = _doc_id_for(input_path, input_url or raw.url)
    raw = clean_raw_document(raw)
    doc_type: DocumentType = classify_document(raw, input_path)
    logger.info("Classified document %s as %s", doc_id, doc_type)

    units = build_candidate_units(raw, doc_type, doc_id=doc_id)
    logger.info("Built %d candidate units", len(units))

    if save_intermediate:
        write_json(
            intermediate_dir / f"candidate_units_{doc_id}.json",
            [u.model_dump() for u in units],
        )

    segments = segment_units(units)
    logger.info("Produced %d semantic segments", len(segments))

    if save_intermediate:
        write_json(
            intermediate_dir / f"semantic_segments_{doc_id}.json",
            [s.model_dump() for s in segments],
        )

    chunks = segments_to_chunks(segments)
    logger.info("Built %d embedding chunks", len(chunks))
    return units, segments, chunks


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest PDFs and web pages into embedding-ready chunks (RAG ingestion only)."
    )
    p.add_argument(
        "--pdf",
        nargs="*",
        default=[],
        metavar="PATH",
        help="One or more local PDF file paths",
    )
    p.add_argument(
        "--url",
        nargs="*",
        default=[],
        metavar="URL",
        help="One or more http(s) URLs",
    )
    p.add_argument(
        "--url-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of *.txt / *.url / *.list files (one URL per line, # for comments)",
    )
    p.add_argument(
        "--url-dir-recursive",
        action="store_true",
        help="With --url-dir, also scan subfolders for URL list files",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output JSON path for embedding chunks (default: ./output/embedding_chunks.json)",
    )
    p.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save candidate units and semantic segments per document for debugging",
    )
    p.add_argument(
        "--intermediate-dir",
        type=Path,
        default=None,
        help="Directory for intermediate JSON (default: next to --output)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="DEBUG logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    settings = get_settings()
    out_path = args.output or (settings.default_output_dir / "embedding_chunks.json")
    inter_dir = args.intermediate_dir or out_path.parent

    pdfs = [Path(p) for p in args.pdf]
    urls = list(args.url)
    if args.url_dir is not None:
        try:
            extra = read_urls_from_directory(
                args.url_dir, recursive=args.url_dir_recursive
            )
            urls.extend(extra)
        except (OSError, NotADirectoryError) as exc:
            logger.error("Invalid --url-dir %s: %s", args.url_dir, exc)
            return 2

    if not pdfs and not urls:
        logger.error("Provide at least one --pdf, --url, or --url-dir")
        return 2

    all_chunks: list[EmbeddingChunk] = []
    errors: list[str] = []

    for pdf_path in pdfs:
        try:
            if not pdf_path.exists():
                raise FileNotFoundError(str(pdf_path))
            raw = _load_pdf(pdf_path)
            _, _, chunks = _process_one(
                raw,
                input_path=pdf_path,
                input_url=None,
                save_intermediate=args.save_intermediate,
                intermediate_dir=inter_dir,
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            msg = f"PDF {pdf_path}: {exc}"
            logger.exception(msg)
            errors.append(msg)

    for url in urls:
        try:
            raw = _load_url(url)
            _, _, chunks = _process_one(
                raw,
                input_path=None,
                input_url=url,
                save_intermediate=args.save_intermediate,
                intermediate_dir=inter_dir,
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            msg = f"URL {url}: {exc}"
            logger.exception(msg)
            errors.append(msg)

    payload = {
        "chunks": [c.model_dump() for c in all_chunks],
        "errors": errors,
        "stats": {
            "chunk_count": len(all_chunks),
            "error_count": len(errors),
        },
    }
    write_json(out_path, payload)
    logger.info("Wrote %d chunks to %s", len(all_chunks), out_path)
    if errors:
        logger.warning("Completed with %d input errors", len(errors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
