"""
coreinsight/memory.py — Optimization memory layer (v0.2.0)

Stores every verified optimization in a local ChromaDB vector database.
On new analysis, retrieves structurally similar past optimizations before
calling the LLM. If a match is found above the similarity threshold, the
LLM call, harness generation, sandbox, and verification are all skipped.

Storage layout:
  ~/.coreinsight/memory_db/        ← ChromaDB persistent store
  ~/.coreinsight/memory_db/code/   ← optimized source files, keyed by hash
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MEMORY_DIR  = Path.home() / ".coreinsight" / "memory_db"
CODE_DIR    = MEMORY_DIR / "code"
COLLECTION  = "optimization_memory"
EMBED_MODEL = "all-MiniLM-L6-v2"   # same model as RepoIndexer — no extra download

# ChromaDB uses cosine *distance* (lower = more similar).
# 0.15 distance ≈ 0.85 cosine similarity for this embedding model.
DEFAULT_DISTANCE_THRESHOLD = 0.15


@dataclass
class MemoryHit:
    func_name:         str
    optimized_code:    str
    avg_speedup:       float
    issue:             str
    reasoning:         str
    similarity:        float    # 0–1, higher is better
    is_exact:          bool     # True = SHA-256 match, not just semantic
    timestamp:         str
    language:          str
    severity:          str = "High"
    correctness_cases: int = 0
    profiler_summary:  str = ""


class OptimizationMemory:
    """
    Local vector database of verified optimizations.

    Reads are thread-safe (ChromaDB handles concurrent queries).
    Writes are called from the main thread after each future completes,
    so no write contention across worker threads.
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR) -> None:
        self._memory_dir = memory_dir
        self._code_dir   = memory_dir / "code"
        self._client     = None
        self._collection = None
        self._embed_fn   = None
        self._init_error = ""

    # ------------------------------------------------------------------ #
    # Lazy init — avoids slow import at startup
    # ------------------------------------------------------------------ #

    def _ensure_db(self) -> bool:
        if self._collection is not None:
            return True
        if self._init_error:
            return False
        try:
            try:
                import chromadb
                from chromadb.utils import embedding_functions
            except Exception as sqlite_exc:
                self._init_error = (
                    f"ChromaDB unavailable (likely outdated SQLite): {sqlite_exc}. "
                    "Optimization memory disabled. "
                    "Fix: pip install pysqlite3-binary and add the following to the top of memory.py:\n"
                    "  import pysqlite3, sys; sys.modules['sqlite3'] = pysqlite3"
                )
                return False

            self._memory_dir.mkdir(parents=True, exist_ok=True)
            self._code_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=str(self._memory_dir))
            self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBED_MODEL
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION,
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            return True
        except Exception as exc:
            self._init_error = str(exc)
            logger.debug(f"OptimizationMemory init failed: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @staticmethod
    def ast_hash(code: str) -> str:
        """SHA-256 of normalised function code — whitespace and comment agnostic."""
        normalised = re.sub(r"#.*",  "", code)          # strip Python comments
        normalised = re.sub(r"//.*", "", normalised)    # strip C++ line comments
        normalised = re.sub(r"\s+",  " ", normalised).strip()
        return hashlib.sha256(normalised.encode()).hexdigest()

    def lookup(
        self,
        code:      str,
        language:  str,
        threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    ) -> Optional[MemoryHit]:
        """
        Look up a verified optimization for `code`.

        Strategy:
          1. Exact match by AST hash — instant, no embedding needed.
          2. Semantic similarity search — best hit returned if distance < threshold.

        Returns None on no match or any internal error.
        """
        if not self._ensure_db():
            return None
        try:
            if self._collection.count() == 0:
                return None

            h = self.ast_hash(code)

            # ── 1. Exact hash match ────────────────────────────────────
            try:
                exact = self._collection.get(ids=[h], include=["metadatas"])
                if exact and exact["ids"]:
                    meta    = exact["metadatas"][0]
                    opt     = self._load_code(h)
                    if opt and meta.get("language") == language:
                        logger.info(f"Memory: exact hit for hash {h[:8]}…")
                        return self._build_hit(meta, opt, similarity=1.0, is_exact=True)
            except Exception:
                pass    # hash not found — fall through to semantic search

            # ── 2. Semantic similarity search ─────────────────────────
            total = self._collection.count()
            results = self._collection.query(
                query_texts=[code],
                n_results=min(3, total),    # guard: n_results must not exceed count
                include=["metadatas", "distances", "ids"],
            )
            if not results["ids"] or not results["ids"][0]:
                return None

            # Find the best hit that matches the requested language
            for dist, meta, rid in zip(
                results["distances"][0],
                results["metadatas"][0],
                results["ids"][0],
            ):
                if meta.get("language") != language:
                    continue
                if dist > threshold:
                    break   # results are sorted by distance ascending
                opt = self._load_code(rid)
                if not opt:
                    continue
                similarity = max(0.0, 1.0 - dist)
                logger.info(f"Memory: semantic hit (similarity={similarity:.2%})")
                return self._build_hit(meta, opt, similarity=similarity, is_exact=False)

            return None

        except Exception as exc:
            logger.debug(f"Memory lookup failed: {exc}")
            return None

    def store(
        self,
        original_code:  str,
        func_name:      str,
        language:       str,
        result:         Dict[str, Any],
        verification,                       # VerificationResult
        profiler_result=None,               # ProfilerResult | None
    ) -> bool:
        """
        Persist a verified optimization.
        Only called when verification.fully_verified is True.
        Uses upsert so re-verifying the same function updates the record.
        """
        if not self._ensure_db():
            return False
        try:
            h           = self.ast_hash(original_code)
            opt_code    = result.get("optimized_code", "") or ""
            avg_speedup = 0.0
            if verification.speedup.computed_speedups:
                avg_speedup = (
                    sum(verification.speedup.computed_speedups)
                    / len(verification.speedup.computed_speedups)
                )

            profiler_summary = ""
            if profiler_result and profiler_result.available and profiler_result.metrics:
                parts = [
                    f"{m.name}: {m.delta}"
                    for m in profiler_result.metrics[:2]
                ]
                profiler_summary = " | ".join(parts)

            self._save_code(h, language, opt_code)

            meta = {
                "func_name":         func_name,
                "language":          language,
                "avg_speedup":       round(avg_speedup, 4),
                "issue":             (result.get("issue")     or "")[:500],
                "reasoning":         (result.get("reasoning") or "")[:1000],
                "severity":          result.get("severity", "High"),
                "correctness_cases": verification.correctness.passed_cases,
                "profiler_summary":  profiler_summary[:200],
                "timestamp":         datetime.now(timezone.utc).isoformat(),
            }

            self._collection.upsert(
                ids=[h],
                documents=[original_code],
                metadatas=[meta],
            )
            logger.info(
                f"Memory: stored '{func_name}' "
                f"(hash={h[:8]}…, speedup={avg_speedup:.2f}x)"
            )
            return True

        except Exception as exc:
            logger.debug(f"Memory store failed: {exc}")
            return False

    def stats(self) -> Dict[str, Any]:
        if not self._ensure_db():
            return {"count": 0, "error": self._init_error}
        try:
            return {"count": self._collection.count()}
        except Exception as exc:
            return {"count": 0, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _save_code(self, h: str, language: str, code: str) -> None:
        ext  = {"python": "py", "cpp": "cpp", "c++": "cpp", "cuda": "cu"}.get(language, "txt")
        path = self._code_dir / f"{h}.{ext}"
        path.write_text(code, encoding="utf-8")

    def _load_code(self, h: str) -> Optional[str]:
        for ext in ("py", "cpp", "cu", "txt"):
            path = self._code_dir / f"{h}.{ext}"
            if path.exists():
                return path.read_text(encoding="utf-8")
        return None

    @staticmethod
    def _build_hit(
        meta:       Dict[str, Any],
        opt_code:   str,
        similarity: float,
        is_exact:   bool,
    ) -> MemoryHit:
        return MemoryHit(
            func_name=         meta.get("func_name",         ""),
            optimized_code=    opt_code,
            avg_speedup=       float(meta.get("avg_speedup", 0.0)),
            issue=             meta.get("issue",             ""),
            reasoning=         meta.get("reasoning",         ""),
            similarity=        similarity,
            is_exact=          is_exact,
            timestamp=         meta.get("timestamp",         ""),
            language=          meta.get("language",          ""),
            severity=          meta.get("severity",          "High"),
            correctness_cases= int(meta.get("correctness_cases", 0)),
            profiler_summary=  meta.get("profiler_summary",  ""),
        )