"""
Embedding helpers for chunk lists (SentenceTransformer + numpy/FAISS live in corpus scripts).

For the SSL unified corpus, see ``corpus_build.website_to_chunks`` and FAISS artifacts
under ``data/final_corpus_bundle/merged/``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


def encode_texts(
    model_name: str,
    texts: list[str],
    *,
    batch_size: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    """Encode strings with a sentence-transformers model."""
    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        normalize_embeddings=normalize,
        show_progress_bar=True,
        batch_size=batch_size,
    ).astype("float32")


__all__ = ["encode_texts"]
