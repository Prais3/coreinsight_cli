import sys
import argparse
import concurrent.futures
import threading
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from coreinsight.config import load_config, run_configure
from coreinsight.parser import CodeParser
from coreinsight.analyzer import AnalyzerAgent
from coreinsight.sandbox import CodeSandbox
from coreinsight.indexer import RepoIndexer

console = Console()

# Thread locks to prevent garbled output when multiple threads finish simultaneously
print_lock = threading.Lock()
file_lock = threading.Lock()

def get_language_from_ext(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext in ['.cu', '.cuh']: return "cuda"
    elif ext in ['.cpp', '.cc', '.h', '.hpp']: return "cpp"
    elif ext == '.py': return "python"
    return "unknown"

def process_function(func: dict, language: str, agent: AnalyzerAgent, sandbox: CodeSandbox, indexer: RepoIndexer):
    """Worker function to analyze and benchmark a single code kernel."""
    func_name = func['name']
    original_code = func['code']
    
    try:
        # 0. Fetch Global Context via RAG
        context = indexer.get_context_for_code(original_code) if indexer else ""
        
        # 1. Analyze
        result = agent.analyze(original_code, language, context=context)
        
        if result.get("severity") in ["Error", "Low"] or not result.get("optimized_code"):
            return func_name, None, None, "✅ No critical bottlenecks detected. Code is optimal."

        # 2. Benchmark in Sandbox
        optimized_code = result["optimized_code"]
        harness_code = agent.generate_harness(func_name, original_code, optimized_code, language)
        success, logs = sandbox.execute_benchmark(harness_code, language)

        # TODO: Commenting this out for now due to slowness -- will fix later        
        # # 3. AUTONOMOUS RETRY LOOP (Self-Healing)
        # max_retries = 2
        # retry_count = 0
        
        # while not success and retry_count < max_retries:
        #     # The harness failed or timed out. Ask the AI to fix it based on the error logs.
        #     harness_code = agent.fix_harness(func_name, original_code, harness_code, logs, language)
        #     # Re-run the sandbox with the new code
        #     success, logs = sandbox.execute_benchmark(harness_code, language)
        #     retry_count += 1

        # # Append retry info to logs if it took multiple attempts
        # if success and retry_count > 0:
        #     logs = f"(Succeeded after {retry_count} AI retries)\n" + logs
        # elif not success and retry_count > 0:
        #     logs = f"(Failed after {retry_count} AI retries)\n" + logs

        return func_name, result, (success, logs), None
        
    except Exception as e:
        return func_name, None, None, f"❌ Analysis failed: {str(e)}"

def format_report_markdown(func_name: str, result: dict, sandbox_res: tuple, msg: str, language: str) -> str:
    """Formats the output into a clean Markdown string."""
    if not result:
        return f"## Kernel: `{func_name}`\n{msg}\n\n---\n"
        
    success, logs = sandbox_res
    severity = result.get('severity', 'Medium')
    
    # Emojis for quick visual scanning
    sev_icon = "🔴" if severity == "Critical" else "🟠" if severity == "High" else "🟡"
    sandbox_icon = "🟢" if success else "❌"
    
    md = f"## Kernel: `{func_name}`\n\n"
    md += f"**Severity:** {sev_icon} {severity}\n\n"
    md += f"**Issue Detected:** {result.get('issue')}\n\n"
    md += f"**Hardware Reasoning:**\n{result.get('reasoning')}\n\n"
    md += f"### 💡 AI Optimized Implementation\n"
    md += f"```{language}\n{result.get('optimized_code')}\n```\n\n"
    md += f"### 🔬 Sandbox Verification\n"
    md += f"**Execution Status:** {sandbox_icon} {'Success' if success else 'Failed'}\n"
    md += f"```text\n{logs.strip()}\n```\n\n---\n"
    
    return md

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

    # 1. PARSE (Synchronous, since it's fast)
    with console.status("[yellow]Parsing AST and extracting hot loops...[/yellow]"):
        parser = CodeParser()
        content_bytes = path.read_bytes()
        functions = parser.parse_file(str(path), content_bytes)
    
    if not functions:
        console.print("[red]No parseable functions found in the file.[/red]")
        sys.exit(1)

    console.print(f"[green]✅ Extracted {len(functions)} functional kernels.[/green]\n")

    # Initialize heavy lifters
    try:
        agent = AnalyzerAgent(provider=provider, model_name=model_name, api_keys=api_keys)
        sandbox = CodeSandbox()
        indexer = RepoIndexer(str(path.parent))
    except Exception as e:
        console.print(f"[red]Initialization Error:[/red] {e}")
        sys.exit(1)

    # Prepare Live Markdown File
    report_path = path.with_name(f"{path.stem}_coreinsight_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# CoreInsight Performance Report: `{path.name}`\n\n")
        f.write("> **Note:** This file updates live as the AI completes hardware verification.\n\n---\n\n")
    
    console.print(f"[bold blue]📄 Live report created at:[/bold blue] [underline]{report_path.absolute()}[/underline]\n")
    console.print("[dim]Analyzing functions in parallel. Results will appear as they complete...[/dim]\n")

    # 2. PARALLEL EXECUTION
    # Limit max_workers to 4 so we don't overwhelm local Docker engine or local Ollama GPU VRAM
    max_workers = min(4, len(functions)) 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all functions to the thread pool
        future_to_func = {
            executor.submit(process_function, func, language, agent, sandbox, indexer): func 
            for func in functions
        }

        # As each thread finishes, process its output instantly
        for future in concurrent.futures.as_completed(future_to_func):
            func = future_to_func[future]
            try:
                func_name, result, sandbox_res, msg = future.result()
                
                # Format to Markdown
                output_md = format_report_markdown(func_name, result, sandbox_res, msg, language)
                
                # Write to File (Safely)
                with file_lock:
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write(output_md)
                
                # Print to Console (Safely)
                with print_lock:
                    console.print(Markdown(output_md))
                    
            except Exception as exc:
                with print_lock:
                    console.print(f"[bold red]❌ Critical failure in thread processing {func['name']}:[/bold red] {exc}")

    console.print(Panel.fit(f"✅ [bold green]Analysis Complete![/bold green] Final report saved to:\n{report_path.absolute()}"))

def main_cli():
    parser = argparse.ArgumentParser(description="CoreInsight CLI - Local Hardware Optimization")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("configure", help="Set up AI providers and API keys")
    
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a local code file")
    analyze_parser.add_argument("file", help="The .cpp, .cu, or .py file to analyze")
    
    index_parser = subparsers.add_parser("index", help="Index the current repository for global AI context")
    index_parser.add_argument("--dir", default=".", help="Directory to index")
    
    args = parser.parse_args()
    
    if args.command == "configure":
        run_configure()
    elif args.command == "analyze":
        run_analysis(args.file)
    elif args.command == "index":
        indexer = RepoIndexer(args.dir)
        indexer.index_repository()
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()