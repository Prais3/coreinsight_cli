import os
from pathlib import Path
from rich.console import Console
from rich.table import Table

from coreinsight.parser import CodeParser

console = Console()

class ProjectScanner:
    def __init__(self, directory: str):
        self.directory = Path(directory).resolve()
        self.parser = CodeParser()

    def _calculate_complexity(self, code: str) -> float:
        """
        A fast, static heuristic to rank hotspots without running them or using AI.
        Penalizes nested loops heavily, as they are the primary cause of CPU/GPU bottlenecks.
        """
        score = 0.0
        lines = code.split('\n')
        
        # Base penalty for massive functions
        score += len(lines) * 0.1  
        
        indent_levels = []
        for line in lines:
            stripped = line.strip()
            
            # Catch common loop declarations
            if stripped.startswith("for ") or stripped.startswith("for(") or stripped.startswith("while ") or stripped.startswith("while("):
                score += 5.0
                
                # Heuristic for nesting based on leading whitespace
                indent = len(line) - len(line.lstrip())
                
                # If this loop's indent is deeper than the last recorded loop, it is nested
                if indent_levels and indent > indent_levels[-1]:
                    score += 15.0  # Massive penalty for O(N^2)+ nesting
                
                indent_levels.append(indent)
                
            # Catch known slow Python list operations
            if ".insert(0" in stripped or "deepcopy(" in stripped:
                score += 10.0
                
        return round(score, 1)

    def scan_project(self, max_results: int = 10):
        valid_extensions = {".py", ".cpp", ".cc", ".h", ".hpp", ".cu", ".cuh"}
        all_functions = []
        
        with console.status("[yellow]Scanning project files and calculating AST complexity...[/yellow]"):
            for root, _, files in os.walk(self.directory):
                # Skip hidden directories and virtual envs
                if any(ignored in root for ignored in [".git", "venv", ".coreinsight_db", "__pycache__", "node_modules"]):
                    continue
                    
                for file in files:
                    path = Path(root) / file
                    if path.suffix in valid_extensions:
                        try:
                            content = path.read_bytes()
                            functions = self.parser.parse_file(str(path), content)
                            
                            for func in functions:
                                func['file'] = str(path.relative_to(self.directory))
                                func['complexity'] = self._calculate_complexity(func['code'])
                                all_functions.append(func)
                        except Exception:
                            # Safely skip unparseable files
                            pass

        if not all_functions:
            console.print("[red]No valid source files found to scan.[/red]")
            return []

        # Sort by complexity descending
        all_functions.sort(key=lambda x: x['complexity'], reverse=True)
        top_hotspots = all_functions[:max_results]

        self._print_results(top_hotspots)
        return top_hotspots

    def _print_results(self, hotspots: list):
        table = Table(title="🔥 Top Project Hotspots (Static Complexity Analysis)")
        table.add_column("Rank", justify="center", style="cyan", no_wrap=True)
        table.add_column("Function", style="magenta")
        table.add_column("File", style="green")
        table.add_column("Language", style="blue")
        table.add_column("Complexity Score", justify="right", style="red bold")

        for i, func in enumerate(hotspots):
            table.add_row(
                str(i + 1),
                func['name'],
                func['file'],
                func['language'].upper(),
                str(func['complexity'])
            )
        
        console.print(table)
        console.print("\n[dim]💡 Next Step: Run `coreinsight analyze <file>` on the worst offenders to generate and verify hardware optimizations.[/dim]\n")