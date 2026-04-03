"""
coreinsight/embeddings.py — Shared embedding utility

Single source of truth for embedding model loading used by both
memory.py (OptimizationMemory) and indexer.py (RepoIndexer).

Tries to load all-MiniLM-L6-v2 from local cache first.
Falls back to a deterministic hash-based embedder when offline
or when the model has not yet been downloaded.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# All models cached here — never hits the network if already present
MODEL_CACHE_DIR = Path.home() / ".coreinsight" / "models"
MODEL_NAME      = "all-MiniLM-L6-v2"


class _HashEmbeddingFunction:
    """
    Deterministic offline fallback embedder.

    Produces a 384-dim float vector from token overlap — no downloads,
    no GPU, no network. Semantic quality is lower than MiniLM but RAG
    and memory lookup still work via keyword/structural matching.

    Run `coreinsight index` once while online to cache the real model.
    """
    DIM = 384

    def __call__(self, input: List[str]) -> List[List[float]]:
        results = []
        for text in input:
            tokens = text.lower().split()
            vec    = [0.0] * self.DIM
            for tok in tokens:
                h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
                vec[h % self.DIM] += 1.0
            # L2 normalise so cosine distance works correctly
            mag = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([x / mag for x in vec])
        return results


def load_embedding_fn() -> Tuple[object, str]:
    """
    Load the sentence-transformer embedding function.

    Returns:
        (embedding_fn, label) where label is a human-readable string
        indicating which embedder is active — shown in CLI output.

    Strategy:
        1. Pin HuggingFace cache to ~/.coreinsight/models so the model
           is never re-downloaded on subsequent runs.
        2. Probe the model with a dummy call to force-load weights now
           rather than silently failing later during indexing or lookup.
        3. On any failure (network error, disk full, offline) fall back
           to _HashEmbeddingFunction with a visible warning.
    """
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Pin cache dirs — must be set before chromadb.utils imports torch
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_CACHE_DIR))
    os.environ.setdefault("HF_HUB_CACHE",              str(MODEL_CACHE_DIR))
    # Allow download when online; callers that want strict offline can
    # set HF_HUB_OFFLINE=1 in their environment before importing.
    os.environ.setdefault("HF_HUB_OFFLINE", "0")

    try:
        from chromadb.utils import embedding_functions as _ef

        fn = _ef.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)

        # Force-load now so we catch errors here, not mid-analysis.
        fn(["probe"])

        label = f"{MODEL_NAME} (cached)"
        logger.debug(f"Embedding model loaded: {label}")
        return fn, label

    except Exception as exc:
        logger.warning(
            f"SentenceTransformer unavailable ({exc}). "
            f"Using offline hash embedder — semantic quality reduced. "
            f"Run `coreinsight index` once while online to cache the model."
        )
        from rich.console import Console as _Console
        _Console().print(
            "[yellow]⚠  Embedding model unavailable (offline or not yet downloaded). "
            "Using keyword-based fallback — RAG and memory recall will work but with "
            "reduced semantic accuracy. "
            "Run [cyan]coreinsight index[/cyan] once while online to cache the model.[/yellow]"
        )
        return _HashEmbeddingFunction(), "hash-based (offline fallback)"