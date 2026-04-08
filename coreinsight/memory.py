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

from coreinsight.embeddings import load_embedding_fn

logger = logging.getLogger(__name__)

MEMORY_DIR  = Path.home() / ".coreinsight" / "memory_db"
CODE_DIR    = MEMORY_DIR / "code"
COLLECTION  = "optimization_memory"

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
    total_cases:       int = 0
    test_cases:        list = field(default_factory=list)
    profiler_summary:  str = ""


class OptimizationMemory:
    """
    Local vector database of verified optimizations.

    Reads are thread-safe (ChromaDB handles concurrent queries).
    Writes are serialized via _write_lock since store() can be called
    from concurrent threads in process_function's as_completed loop.
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR) -> None:
        import threading
        self._memory_dir  = memory_dir
        self._code_dir    = memory_dir / "code"
        self._client      = None
        self._collection  = None
        self._embed_fn    = None
        self._init_error  = ""
        self._write_lock  = threading.Lock()

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
            except Exception as sqlite_exc:
                self._init_error = (
                    f"ChromaDB unavailable (likely outdated SQLite): {sqlite_exc}. "
                    "Optimization memory disabled. "
                    "Fix: pip install coreinsight-cli[compat]"
                )
                return False

            self._memory_dir.mkdir(parents=True, exist_ok=True)
            self._code_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=str(self._memory_dir))
            self._embed_fn, _embed_label = load_embedding_fn()
            logger.debug(f"Memory embedder: {_embed_label}")
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

    def export(self, output_path: str, fmt: str = "csv") -> int:
        """
        Export all stored optimizations to CSV or Markdown.
        Returns the number of records written, or 0 on failure.
        """
        import csv as _csv
        from pathlib import Path as _Path

        if not self._ensure_db():
            return 0

        try:
            all_records = self._collection.get(include=["metadatas"])
            metadatas   = all_records.get("metadatas", []) or []
        except Exception as e:
            logger.error(f"Export failed reading store: {e}")
            return 0

        if not metadatas:
            return 0

        # Sort most recent first
        metadatas = sorted(
            metadatas,
            key=lambda m: m.get("timestamp", ""),
            reverse=True,
        )

        out = _Path(output_path)

        if fmt == "csv":
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = _csv.writer(f)
                writer.writerow([
                    "func_name", "language", "severity", "issue",
                    "avg_speedup", "correctness_cases",
                    "hardware_evidence", "verified_at", "optimized_code",
                ])
                for m in metadatas:
                    writer.writerow([
                        m.get("func_name",        ""),
                        m.get("language",         ""),
                        m.get("severity",         ""),
                        m.get("issue",            ""),
                        m.get("avg_speedup",      ""),
                        m.get("correctness_cases",""),
                        m.get("profiler_summary", ""),
                        m.get("timestamp",        "")[:19].replace("T", " "),
                        m.get("optimized_code",   ""),
                    ])

        elif fmt == "md":
            with open(out, "w", encoding="utf-8") as f:
                f.write("# CoreInsight Optimization Memory Export\n\n")
                for i, m in enumerate(metadatas, 1):
                    lang = m.get("language", "")
                    f.write(f"## {i}. `{m.get('func_name', 'unknown')}` ({lang})\n\n")
                    f.write(f"- **Severity:** {m.get('severity', '')}\n")
                    f.write(f"- **Issue:** {m.get('issue', '')}\n")
                    f.write(f"- **Avg speedup:** {float(m.get('avg_speedup', 0)):.2f}x\n")
                    f.write(f"- **Correctness cases:** {m.get('correctness_cases', '')}\n")
                    f.write(f"- **Verified at:** {m.get('timestamp', '')[:19].replace('T', ' ')}\n")
                    if m.get("profiler_summary"):
                        f.write(f"- **Hardware evidence:** {m.get('profiler_summary')}\n")
                    code = m.get("optimized_code", "").strip()
                    if code:
                        f.write(f"\n```{lang}\n{code}\n```\n")
                    f.write("\n---\n\n")

        return len(metadatas)

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
        with self._write_lock:
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
                    "total_cases":       verification.correctness.total_cases,
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

    def lookup_test_cases(self, original_code: str) -> Optional[list]:
        """
        Return stored test cases for `original_code`, or None if not found.
        Used to re-run correctness without regenerating via LLM.
        """
        if not self._ensure_db():
            return None
        h = self.ast_hash(original_code)
        return self._load_test_cases(h)

    def store_test_cases(self, original_code: str, test_cases: list) -> None:
        """
        Persist test cases for a function, keyed by AST hash.
        Called from process_function immediately after test cases are generated,
        so `coreinsight test` can re-run verification without the LLM.
        """
        if not self._ensure_db():
            return
        h = self.ast_hash(original_code)
        with self._write_lock:
            try:
                self._code_dir.mkdir(parents=True, exist_ok=True)
                self._save_test_cases(h, test_cases)
            except Exception as exc:
                logger.debug(f"store_test_cases failed: {exc}")

    def lookup_by_name(self, func_name: str) -> Optional[dict]:
        """
        Find the most recent memory record whose func_name matches exactly.
        Returns a dict with keys: func_name, language, original_code,
        optimized_code, test_cases, meta. Returns None on no match.
        """
        if not self._ensure_db():
            return None
        try:
            all_records = self._collection.get(
                include=["metadatas", "documents"]
            )
            matches = [
                (meta, doc, rid)
                for meta, doc, rid in zip(
                    all_records.get("metadatas", []),
                    all_records.get("documents", []),
                    all_records.get("ids", []),
                )
                if meta.get("func_name") == func_name
            ]
            if not matches:
                return None
            # Most recent first
            matches.sort(key=lambda x: x[0].get("timestamp", ""), reverse=True)
            meta, original_code, h = matches[0]
            return {
                "func_name":      func_name,
                "language":       meta.get("language", ""),
                "original_code":  original_code or "",
                "optimized_code": self._load_code(h) or "",
                "test_cases":     self._load_test_cases(h) or [],
                "meta":           meta,
            }
        except Exception as exc:
            logger.debug(f"lookup_by_name failed: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _save_test_cases(self, h: str, cases: list) -> None:
        path = self._code_dir / f"{h}.test_cases.json"
        path.write_text(json.dumps(cases), encoding="utf-8")

    def _load_test_cases(self, h: str) -> Optional[list]:
        path = self._code_dir / f"{h}.test_cases.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

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
            total_cases=       int(meta.get("total_cases",       0)),
            profiler_summary=  meta.get("profiler_summary",  ""),
        )