import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.spinner import Spinner
from rich.live import Live

from coreinsight.config import load_config, run_configure
from coreinsight.parser import CodeParser
from coreinsight.analyzer import AnalyzerAgent
from coreinsight.sandbox import CodeSandbox

console = Console()

def get_language_from_ext(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext in ['.cu', '.cuh']: return "cuda"
    elif ext in ['.cpp', '.cc', '.h', '.hpp']: return "cpp"
    elif ext == '.py': return "python"
    return "unknown"

def run_analysis(file_path: str):
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        console.print(f"[red]Error: File '{file_path}' not found.[/red]")
        sys.exit(1)

    language = get_language_from_ext(path)
    if language == "unknown":
        console.print(f"[red]Error: Unsupported file type '{path.suffix}'. Use .cpp, .cu, or .py[/red]")
        sys.exit(1)

    config = load_config()
    provider = config.get("provider", "ollama")
    model_name = config.get("model_name", "llama3.2")
    api_keys = config.get("api_keys", {})

    console.print(Panel.fit(f"🚀 CoreInsight: Profiling [bold cyan]{path.name}[/bold cyan] via [bold]{provider}[/bold]"))

    # 1. PARSE
    with console.status("[yellow]Parsing AST and extracting hot loops...[/yellow]"):
        parser = CodeParser()
        content_bytes = path.read_bytes()
        functions = parser.parse_file(str(path), content_bytes)
    
    if not functions:
        console.print("[red]No parseable functions found in the file.[/red]")
        sys.exit(1)

    # Initialize heavy lifters
    try:
        agent = AnalyzerAgent(provider=provider, model_name=model_name, api_keys=api_keys)
        sandbox = CodeSandbox()
    except Exception as e:
        console.print(f"[red]Initialization Error:[/red] {e}")
        sys.exit(1)

    # 2. ANALYZE & 3. BENCHMARK
    for func in functions:
        func_name = func['name']
        original_code = func['code']
        
        console.print(f"\n[bold]Targeting Kernel:[/bold] `{func_name}`")
        
        with Live(Spinner("dots", text=f"[cyan]AI analyzing hardware bottlenecks in {func_name}...[/cyan]"), refresh_per_second=10, transient=True) as live:
            result = agent.analyze(original_code, language)
            
            if result.get("severity") in ["Error", "Low"] or not result.get("optimized_code"):
                live.update("")
                console.print(f"[dim]No critical bottlenecks detected in {func_name}. Skipping benchmark.[/dim]")
                continue

            live.update(Spinner("dots", text=f"[yellow]Generating C++/CUDA benchmark harness for {func_name}...[/yellow]"))
            optimized_code = result["optimized_code"]
            harness_code = agent.generate_harness(original_code, optimized_code, language)
            
            live.update(Spinner("dots", text=f"[magenta]Compiling and executing in secure Docker sandbox...[/magenta]"))
            success, logs = sandbox.execute_benchmark(harness_code, language)
            live.update("")

        # 4. REPORT
        console.print("="*60)
        severity = result.get('severity', 'Medium')
        color = "red" if severity == "Critical" else "yellow" if severity == "High" else "cyan"
        
        console.print(f"[{color} bold]⚠️ {severity} Bottleneck Detected[/{color} bold]")
        console.print(f"[bold]Issue:[/bold] {result.get('issue')}")
        console.print("\n[bold]Hardware Reasoning:[/bold]")
        console.print(Markdown(result.get('reasoning', '')))
        
        console.print("\n[bold green]💡 AI Optimized Implementation:[/bold green]")
        syntax = Syntax(optimized_code, language, theme="monokai", line_numbers=True)
        console.print(Panel(syntax))

        console.print("\n[bold blue]🔬 Sandbox Verification Results:[/bold blue]")
        if success:
            console.print(f"[green]{logs.strip()}[/green]")
        else:
            console.print(f"[red]Benchmark Failed:\n{logs.strip()}[/red]")

def main_cli():
    parser = argparse.ArgumentParser(description="CoreInsight CLI - Local Hardware Optimization")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("configure", help="Set up AI providers and API keys")
    
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a local code file")
    analyze_parser.add_argument("file", help="The .cpp, .cu, or .py file to analyze")
    
    args = parser.parse_args()
    
    if args.command == "configure":
        run_configure()
    elif args.command == "analyze":
        run_analysis(args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()