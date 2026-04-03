import os
import logging
import hashlib
import math
from pathlib import Path
from rich.console import Console
from rich.progress import track
import chromadb
from chromadb.utils import embedding_functions

from coreinsight.parser import CodeParser
from coreinsight.embeddings import load_embedding_fn

console = Console()
logger  = logging.getLogger(__name__)


class RepoIndexer:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.db_path   = self.repo_path / ".coreinsight_db"
        self.parser    = CodeParser()
        self._chroma_client = None
        self._embedding_fn  = None
        self._embedding_label = ""
        self._collection    = None

    def _ensure_db(self) -> bool:
        """Lazy-initialize ChromaDB and the embedding model on first real use."""
        if self._collection is not None:
            return True
        try:
            self._chroma_client   = chromadb.PersistentClient(path=str(self.db_path))
            self._embedding_fn, self._embedding_label = load_embedding_fn()
            self._collection      = self._chroma_client.get_or_create_collection(
                name="codebase_context",
                embedding_function=self._embedding_fn,
            )
            return True
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            return False

    @property
    def collection(self):
        self._ensure_db()
        return self._collection

    def index_repository(self):
        """Scans the repo and embeds all code blocks into the vector DB."""
        if not self._ensure_db():
            console.print("[red]Could not initialise vector DB — skipping index.[/red]")
            return

        valid_extensions = {".py", ".cpp", ".cc", ".h", ".hpp", ".cu", ".cuh"}
        files_to_index   = []

        for root, _, files in os.walk(self.repo_path):
            if any(skip in root for skip in [".coreinsight_db", ".git", "venv"]):
                continue
            for file in files:
                path = Path(root) / file
                if path.suffix in valid_extensions:
                    files_to_index.append(path)

        if not files_to_index:
            console.print("[yellow]No source files found to index.[/yellow]")
            return

        console.print(
            f"[cyan]Found {len(files_to_index)} files. "
            f"Building vector index (embedder: {self._embedding_label})...[/cyan]"
        )

        documents, metadatas, ids = [], [], []

        for filepath in track(files_to_index, description="Parsing and Embedding..."):
            try:
                content   = filepath.read_bytes()
                functions = self.parser.parse_file(str(filepath), content)
                for idx, func in enumerate(functions):
                    documents.append(func['code'])
                    metadatas.append({
                        "file":     str(filepath.relative_to(self.repo_path)),
                        "name":     func['name'],
                        "language": func['language'],
                    })
                    ids.append(f"{filepath.name}_{func['name']}_{idx}")
            except Exception as e:
                console.print(f"[dim red]Skipped {filepath.name}: {e}[/dim red]")

        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            console.print(
                f"[bold green]✅ Indexed {len(documents)} code chunks "
                f"(embedder: {self._embedding_label})[/bold green]"
            )

    def get_context_for_code(self, code_snippet: str, n_results: int = 3) -> str:
        """Retrieves related code chunks to feed to the LLM as context."""
        if not self.db_path.exists():
            return ""
        try:
            if not self._ensure_db():
                return ""
            if self._collection.count() == 0:
                return ""
            results = self._collection.query(
                query_texts=[code_snippet],
                n_results=n_results,
            )
            if not results['documents'] or not results['documents'][0]:
                return ""
            context_blocks = []
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                context_blocks.append(f"// From {meta['file']} ({meta['name']}):\n{doc}")
            return "\n\n".join(context_blocks)
        except Exception as e:
            logger.warning(f"ChromaDB context retrieval failed, continuing without RAG: {e}")
            return ""