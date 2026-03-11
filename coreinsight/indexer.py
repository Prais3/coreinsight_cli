import os
from pathlib import Path
from rich.console import Console
from rich.progress import track
import chromadb
from chromadb.utils import embedding_functions

from coreinsight.parser import CodeParser

console = Console()

class RepoIndexer:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.db_path = self.repo_path / ".coreinsight_db"
        self.parser = CodeParser()
        self._chroma_client = None
        self._embedding_fn = None
        self._collection = None

    def _ensure_db(self):
        """Lazy-initialize ChromaDB and the embedding model on first real use."""
        if self._collection is not None:
            return
        self._chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name="codebase_context",
            embedding_function=self._embedding_fn,
        )

    @property
    def collection(self):
        self._ensure_db()
        return self._collection

    def index_repository(self):
        """Scans the repo and embeds all code blocks into the vector DB."""
        valid_extensions = {".py", ".cpp", ".cc", ".h", ".hpp", ".cu", ".cuh"}
        files_to_index = []
        
        for root, _, files in os.walk(self.repo_path):
            if ".coreinsight_db" in root or ".git" in root or "venv" in root:
                continue
            for file in files:
                path = Path(root) / file
                if path.suffix in valid_extensions:
                    files_to_index.append(path)
                    
        if not files_to_index:
            console.print("[yellow]No source files found to index.[/yellow]")
            return

        console.print(f"[cyan]Found {len(files_to_index)} files. Building vector index...[/cyan]")
        
        documents = []
        metadatas = []
        ids = []
        
        for filepath in track(files_to_index, description="Parsing and Embedding..."):
            try:
                content = filepath.read_bytes()
                # Use your existing AST parser to chunk the file!
                functions = self.parser.parse_file(str(filepath), content)
                
                for idx, func in enumerate(functions):
                    documents.append(func['code'])
                    metadatas.append({
                        "file": str(filepath.relative_to(self.repo_path)),
                        "name": func['name'],
                        "language": func['language']
                    })
                    ids.append(f"{filepath.name}_{func['name']}_{idx}")
            except Exception as e:
                console.print(f"[dim red]Skipped {filepath.name}: {e}[/dim red]")

        if documents:
            # Batch add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            console.print(f"[bold green]✅ Successfully indexed {len(documents)} code chunks into local Vector DB![/bold green]")

    def get_context_for_code(self, code_snippet: str, n_results: int = 3) -> str:
        """Retrieves related code (like structs/classes) to feed to the LLM."""
        if not self.db_path.exists():
            return ""
        if self.collection.count() == 0:
            return ""
            
        results = self.collection.query(
            query_texts=[code_snippet],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return ""
            
        context_blocks = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            context_blocks.append(f"// From {meta['file']} ({meta['name']}):\n{doc}")
            
        return "\n\n".join(context_blocks)