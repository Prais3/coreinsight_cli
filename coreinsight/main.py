import sys
import argparse
import concurrent.futures
import threading
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group

from coreinsight.config import load_config, run_configure
from coreinsight.parser import CodeParser
from coreinsight.analyzer import AnalyzerAgent
from coreinsight.sandbox import CodeSandbox
from coreinsight.indexer import RepoIndexer
from coreinsight.hardware import HardwareDetector
from coreinsight.scanner import ProjectScanner

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

def process_function(func: dict, language: str, agent: AnalyzerAgent, sandbox: CodeSandbox, indexer: RepoIndexer, hardware_target: str):
    """Worker function to analyze and benchmark a single code kernel."""
    func_name = func['name']
    original_code = func['code']
    
    try:
        # 0. Fetch Global Context via RAG
        context = indexer.get_context_for_code(original_code) if indexer else ""
        
        # 1. Analyze
        result = agent.analyze(original_code, language, context=context, hardware_target=hardware_target)
        
        if result.get("severity") in ["Error", "Low"] or not result.get("optimized_code"):
            return func_name, None, None, "✅ No critical bottlenecks detected. Code is optimal."

        # 2. Benchmark in Sandbox
        optimized_code = result["optimized_code"]
        harness_code = agent.generate_harness(func_name, original_code, optimized_code, language, context, hardware_target=hardware_target)
        success, logs, plot_data = sandbox.execute_benchmark(harness_code, language)

        # TODO: Commenting this out for now due to slowness -- will fix later        
        # 3. AUTONOMOUS RETRY LOOP (Self-Healing & Speedup Verification)
        max_retries = 2
        retry_count = 0
        
        def check_speedup_success(success_status, output_logs):
            if not success_status: return False
            
            # Parse the CSV logs to find the final speedup
            try:
                lines = output_logs.strip().split('\n')
                for line in reversed(lines):
                    parts = line.split(',')
                    # Make sure it actually found our CSV format
                    if len(parts) == 4 and parts[0].strip().isdigit():
                        speedup = float(parts[3])
                        return speedup >= 1.05 # Require at least a 5% speedup
            except Exception:
                pass
                
            # If we reach here, it means the script ran successfully but DID NOT print the CSV.
            # We must return False so the retry loop forces it to print the table.
            return False

        is_valid_optimization = check_speedup_success(success, logs)
        
        while not is_valid_optimization and retry_count < max_retries:
            if success: 
                # It ran without crashing, but failed our validation
                if "N,Original_Time" not in logs:
                    logs += "\nERROR: You ran successfully but DID NOT print the CSV table to stdout! You MUST print the strict CSV format."
                else:
                    logs += "\nERROR: Your optimized code was SLOWER than the original. Rewrite it to be faster."
            
            harness_code = agent.fix_harness(func_name, original_code, harness_code, logs, language, context=context)
            success, logs, plot_data = sandbox.execute_benchmark(harness_code, language)
            is_valid_optimization = check_speedup_success(success, logs)
            retry_count += 1

        if is_valid_optimization and retry_count > 0:
            logs = f"(Succeeded after {retry_count} AI retries)\n" + logs
        elif not is_valid_optimization:
            logs = f"(Failed to achieve valid CSV/Speedup after {retry_count} AI retries)\n" + logs
            success = False # Force the UI to show a red X

        return func_name, result, (success, logs, plot_data), None
        
    except Exception as e:
        return func_name, None, None, f"❌ Analysis failed: {str(e)}"

def parse_csv_logs(logs: str):
    """Safely extracts CSV data from the sandbox logs."""
    data = []
    for line in logs.strip().split('\n'):
        if line.startswith("N,") or line.startswith("(Succeeded") or line.startswith("(Failed"): 
            continue
        parts = line.split(',')
        if len(parts) == 4 and parts[0].isdigit():
            data.append(parts)
    return data

def format_report_markdown(func_name: str, result: dict, sandbox_res: tuple, msg: str, language: str, target_dir: Path) -> str:
    """Generates the text for the .md file."""
    if not result:
        return f"## Kernel: `{func_name}`\n{msg}\n\n---\n"
        
    success, logs, plot_data = sandbox_res
    severity = result.get('severity', 'Medium')
    sandbox_icon = "🟢" if success else "❌"
    
    md = f"## Kernel: `{func_name}`\n\n"
    md += f"**Severity:** {severity}\n\n"
    md += f"**Issue:** {result.get('issue')}\n\n"
    md += f"**Reasoning:**\n{result.get('reasoning')}\n\n"
    md += f"### Optimized Code\n```{language}\n{result.get('optimized_code')}\n```\n\n"
    md += f"### Verification: {sandbox_icon} {'Success' if success else 'Failed'}\n"
    
    csv_data = parse_csv_logs(logs)
    if success and csv_data:
        md += "| N (Size) | Orig Time (s) | Opt Time (s) | Speedup |\n| :--- | :--- | :--- | :--- |\n"
        for p in csv_data:
            md += f"| {p[0]} | {p[1]} | {p[2]} | **{p[3]}x** |\n"
    else:
        md += f"```text\n{logs.strip()}\n```\n"

    if plot_data:
        plot_path = target_dir / f"{func_name}_benchmark_plot.png"
        with open(plot_path, "wb") as f:
            f.write(plot_data)
        md += f"\n![Scaling Plot](./{plot_path.name})\n"

    md += "\n---\n"
    return md

def print_console_report(func_name: str, result: dict, sandbox_res: tuple, msg: str, language: str):
    """Renders a beautiful, native Rich dashboard for the terminal."""
    if not result:
        console.print(Panel(f"[green]{msg}[/green]", title=f"Kernel: {func_name}"))
        return

    success, logs, plot_data = sandbox_res
    severity = result.get('severity', 'Medium')
    
    color = "red" if severity == "Critical" else "yellow" if severity == "High" else "cyan"
    
    # 1. Header Information
    content = [
        Text(f"Severity: {severity}", style=f"bold {color}"),
        Text(f"Issue: {result.get('issue')}", style="bold white"),
        Text(f"Reasoning: {result.get('reasoning')}\n", style="dim white")
    ]

    # 2. Results Table
    csv_data = parse_csv_logs(logs)
    if csv_data:
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("N (Size)", justify="right")
        table.add_column("Original (s)", justify="right", style="dim")
        table.add_column("Optimized (s)", justify="right", style="green")
        table.add_column("Speedup", justify="right", style="bold cyan")
        
        for p in csv_data:
            table.add_row(p[0], p[1], p[2], f"{p[3]}x")
        
        content.append(table)
    else:
        # Fallback if no CSV was parsed (e.g. compilation error)
        status_color = "green" if success else "red"
        content.append(Panel(logs.strip(), title="Sandbox Logs", border_style=status_color))

    if plot_data:
        content.append(Text(f"\n📈 Plot saved: {func_name}_benchmark_plot.png", style="italic blue"))

    console.print(Panel(Group(*content), title=f"🚀 Analysis: [bold]{func_name}[/bold]", border_style=color))

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
        hardware_specs_dict = HardwareDetector.get_system_specs()
        hardware_target_str = HardwareDetector.format_for_llm(hardware_specs_dict)
        
        console.print(f"[dim]🎯 Target Hardware Detected: {hardware_target_str}[/dim]")
        
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
            executor.submit(process_function, func, language, agent, sandbox, indexer, hardware_target_str): func 
            for func in functions
        }

        # As each thread finishes, process its output instantly
        for future in concurrent.futures.as_completed(future_to_func):
            func = future_to_func[future]
            try:
                func_name, result, sandbox_res, msg = future.result()
                
                # 1. Format and save to Markdown file
                output_md = format_report_markdown(func_name, result, sandbox_res, msg, language, path.parent)
                
                # Write to File (Safely)
                with file_lock:
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write(output_md)
                        
                # 2. Print beautiful native UI to the terminal
                with print_lock:
                    print_console_report(func_name, result, sandbox_res, msg, language)
                
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
    
    scan_parser = subparsers.add_parser("scan", help="Scan directory for complex hotspots")
    scan_parser.add_argument("--dir", default=".", help="Directory to scan")
    scan_parser.add_argument("--top", type=int, default=10, help="Number of hotspots to show")
    
    args = parser.parse_args()
    
    if args.command == "configure":
        run_configure()
    elif args.command == "analyze":
        run_analysis(args.file)
    elif args.command == "scan":
        scanner = ProjectScanner(args.dir)
        scanner.scan_project(max_results=args.top)
    elif args.command == "index":
        indexer = RepoIndexer(args.dir)
        indexer.index_repository()
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()