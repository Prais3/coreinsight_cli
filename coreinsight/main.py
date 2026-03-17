import sys
import os
import warnings

# ── Suppress noisy third-party warnings before any imports trigger them ──────
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")
warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Suppress chromadb's BertModel load report via transformers logger
import logging as _logging
_logging.getLogger("transformers").setLevel(_logging.ERROR)
_logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
_logging.getLogger("chromadb").setLevel(_logging.ERROR)

import json
import argparse
import concurrent.futures
import threading
import urllib.request
import urllib.error
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Group

from coreinsight.config import load_config, run_configure, is_pro, get_tier_limits, PRO_WAITLIST_URL, get_model_tier
from coreinsight.parser import CodeParser
from coreinsight.analyzer import AnalyzerAgent
from coreinsight.sandbox import CodeSandbox
from coreinsight.profiler import HardwareProfiler
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

def _log(func_name: str, msg: str, style: str = "dim"):
    """Thread-safe single-line progress log."""
    with print_lock:
        console.log(f"[{style}][ {func_name} ][/{style}] {msg}")

def process_function(func: dict, language: str, agent: AnalyzerAgent, sandbox: CodeSandbox, indexer: RepoIndexer, hardware_target: str, tier_limits: dict, profiler=None, file_content: str = "", source_dir: str = ""):
    """Worker function to analyze and benchmark a single code kernel."""
    func_name = func['name']
    original_code = func['code']
    
    try:
        # 0. Fetch Global Context via RAG
        _log(func_name, "Fetching RAG context...")
        context = indexer.get_context_for_code(original_code) if indexer else ""
        
        # 1. Analyze
        _log(func_name, "Calling LLM for bottleneck analysis...")
        result = agent.analyze(original_code, language, context=context, hardware_target=hardware_target)
        
        if result.get("severity") == "Error":
            error_msg = result.get("issue", "Unknown error during analysis.")
            return func_name, None, None, f"❌ Analysis error: {error_msg}", None, None

        if result.get("severity") == "Low" or not result.get("optimized_code"):
            return func_name, None, None, "✅ No critical bottlenecks detected. Code is optimal.", None, None

        # 2. Benchmark in Sandbox
        optimized_code = result["optimized_code"]
        _log(func_name, "Generating benchmark harness...")
        harness_code = agent.generate_harness(func_name, original_code, optimized_code, language, context, hardware_target=hardware_target)

        # # Debug statement to test what AI output is generated        
        # # Before calling sandbox, see exactly what the model generated:
        # print("\n" + "="*40 + " AI GENERATED HARNESS " + "="*40)
        # print(harness_code)
        # print("="*102 + "\n")
        
        _log(func_name, "Running benchmark in Docker sandbox...")
        success, logs, plot_data = sandbox.execute_benchmark(harness_code, language)

        # AUTONOMOUS RETRY LOOP (Self-Healing & Speedup Verification)
        max_retries = tier_limits["max_retries"]
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
            
            _log(func_name, f"Retry {retry_count + 1}/{max_retries} — asking LLM to fix harness...", style="yellow")
            harness_code = agent.fix_harness(func_name, original_code, harness_code, logs, language, context=context)
            _log(func_name, f"Retry {retry_count + 1}/{max_retries} — re-running sandbox...", style="yellow")
            success, logs, plot_data = sandbox.execute_benchmark(harness_code, language)
            is_valid_optimization = check_speedup_success(success, logs)
            retry_count += 1

        if is_valid_optimization and retry_count > 0:
            logs = f"(Succeeded after {retry_count} AI retries)\n" + logs
        elif not is_valid_optimization:
            logs = f"(Failed to achieve valid CSV/Speedup after {retry_count} AI retries)\n" + logs
            success = False

        # 3. Verification + AI-free hardware profiling
        verification    = None
        profiler_result = None
        if is_valid_optimization:
            _log(func_name, "Generating correctness test cases...")
            test_cases = agent.generate_test_cases(func_name, original_code, language, context, num_cases=tier_limits["num_test_cases"])
            _log(func_name, "Running correctness verification in Docker sandbox...")
            verification = sandbox.verify(
                csv_output=logs,
                original_code=original_code,
                optimized_code=optimized_code,
                original_func_name=func_name,
                optimized_func_name=func_name,
                test_cases=test_cases,
                language=language,
            )

            # AI-free hardware profiling (pro-gated)
            if profiler is not None and tier_limits.get("hardware_profiling"):
                _log(func_name, "Running AI-free hardware profiling...", style="blue")
                profiler_result = profiler.profile(
                    original_code=original_code,
                    optimized_code=optimized_code,
                    func_name=func_name,
                    language=language,
                    test_cases=test_cases,
                    original_file_content=file_content,
                    source_dir=source_dir,
                )

        _log(func_name, "Done ✓", style="bold green")
        return func_name, result, (success, logs, plot_data), None, verification, profiler_result

    except Exception as e:
        _log(func_name, f"Failed: {e}", style="bold red")
        return func_name, None, None, f"❌ Analysis failed: {str(e)}", None, None

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

def format_report_markdown(func_name: str, result: dict, sandbox_res: tuple, msg: str, language: str, target_dir: Path, verification=None, profiler_result=None) -> str:
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

    if verification is not None:
        v = verification
        speedup_icon    = "🟢" if v.speedup.verified     else "❌"
        correct_icon    = "🟢" if v.correctness.verified  else "❌"
        md += f"\n### Verification Report\n"
        md += f"- {speedup_icon} **Speedup integrity:** {v.speedup.details}\n"
        md += f"- {correct_icon} **Output correctness:** {v.correctness.details}\n"
        if v.speedup.suspicious_flags:
            for flag in v.speedup.suspicious_flags:
                md += f"  - ⚠️ {flag}\n"
        if v.correctness.failures:
            for failure in v.correctness.failures:
                md += f"  - ✗ {failure}\n"

    if profiler_result is not None:
        pr = profiler_result
        md += f"\n### 🔬 Hardware Evidence (AI-Free Profiling)\n\n"
        if pr.available:
            md += f"> Measured by **`{pr.tool}`** inside Docker · no LLM involved · deterministic\n\n"
            if pr.metrics:
                md += "| Metric | Original | Optimized | Δ Delta | Note |\n"
                md += "|:---|---:|---:|---:|:---|\n"
                for m in pr.metrics:
                    md += f"| {m.name} | {m.original} | {m.optimized} | **{m.delta}** | {m.note} |\n"
            if pr.host_tool_metrics:
                md += f"\n#### `{pr.host_tool_name}` — CPU hardware counters\n\n"
                md += "| Metric | Original | Optimized | Δ Delta | Note |\n"
                md += "|:---|---:|---:|---:|:---|\n"
                for m in pr.host_tool_metrics:
                    md += f"| {m.name} | {m.original} | {m.optimized} | **{m.delta}** | {m.note} |\n"
            if pr.raw_original or pr.raw_optimized:
                md += "\n<details><summary>cProfile detail (original)</summary>\n\n"
                md += f"```\n{pr.raw_original.strip()}\n```\n\n</details>\n"
                md += "\n<details><summary>cProfile detail (optimized)</summary>\n\n"
                md += f"```\n{pr.raw_optimized.strip()}\n```\n\n</details>\n"
        else:
            md += f"> ⚠️ Hardware profiling unavailable: {pr.error}\n"

    md += "\n---\n"
    return md

def print_console_report(func_name: str, result: dict, sandbox_res: tuple, msg: str, language: str, verification=None, profiler_result=None):
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

    if verification is not None:
        v = verification
        speedup_color  = "green" if v.speedup.verified    else "red"
        correct_color  = "green" if v.correctness.verified else "red"

        vtable = Table(show_header=False, box=None, padding=(0, 1))
        vtable.add_column(style="bold")
        vtable.add_column()
        vtable.add_row(
            f"[{speedup_color}]Speedup integrity[/{speedup_color}]",
            f"[{speedup_color}]{v.speedup.details}[/{speedup_color}]",
        )
        vtable.add_row(
            f"[{correct_color}]Output correctness[/{correct_color}]",
            f"[{correct_color}]{v.correctness.details}[/{correct_color}]",
        )
        for flag in v.speedup.suspicious_flags:
            vtable.add_row("[yellow]⚠ Warning[/yellow]", f"[yellow]{flag}[/yellow]")
        for failure in v.correctness.failures:
            vtable.add_row("[red]✗ Failure[/red]", f"[red]{failure}[/red]")

        content.append(Panel(vtable, title="🔬 Verification", border_style="dim"))

    if profiler_result is not None and profiler_result.available:
        pr = profiler_result
        all_metrics = pr.metrics + pr.host_tool_metrics
        if all_metrics:
            ptable = Table(show_header=True, header_style="bold blue",
                           box=None, padding=(0, 1), expand=True)
            ptable.add_column("Metric",    style="dim")
            ptable.add_column("Original",  justify="right")
            ptable.add_column("Optimized", justify="right", style="cyan")
            ptable.add_column("Δ Delta",   justify="right", style="bold")
            ptable.add_column("Note",      style="dim italic")
            for m in all_metrics:
                delta_color = "green" if m.delta.startswith("-") else (
                              "red"   if m.delta.startswith("+") else "white")
                ptable.add_row(
                    m.name, m.original, m.optimized,
                    f"[{delta_color}]{m.delta}[/{delta_color}]",
                    m.note,
                )
            tool_label = pr.tool
            from rich.markup import escape as _escape
            content.append(Panel(
                ptable,
                title=f"🔬 Hardware Evidence  \[{_escape(pr.tool)}]",
                border_style="blue",
            ))

    console.print(Panel(Group(*content), title=f"🚀 Analysis: [bold]{func_name}[/bold]", border_style=color))

def _preflight_checks(provider: str, model_name: str) -> bool:
    """
    Fast, cheap checks run before any expensive work (parsing, image building, threads).
    Prints an actionable Rich panel and returns False on failure.
    """
    import docker as _docker

    # ── 1. Docker ────────────────────────────────────────────────────────────
    try:
        _docker.from_env().ping()
    except _docker.errors.DockerException as e:
        console.print(Panel(
            "[bold]CoreInsight requires Docker to run its verification sandboxes.[/bold]\n\n"
            "[yellow]To fix:[/yellow]\n"
            "  • [cyan]Mac / Windows[/cyan] — Open Docker Desktop and wait for the whale icon to stop animating\n"
            "  • [cyan]Linux[/cyan]          — Run: [cyan]sudo systemctl start docker[/cyan]\n\n"
            f"[dim]Docker error: {e}[/dim]",
            title="❌  Docker is not running",
            border_style="red",
        ))
        return False
    except Exception as e:
        console.print(f"[red]Unexpected error reaching Docker: {e}[/red]")
        return False

    # ── 2. Ollama (only when provider == "ollama") ────────────────────────────
    if provider == "ollama":
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
                data = json.loads(resp.read())

            available = [m["name"] for m in data.get("models", [])]
            base = model_name.split(":")[0]   # "llama3.2:latest" → "llama3.2"

            if not any(base in m for m in available):
                model_list = "\n".join(f"  • {m}" for m in available) or "  (none pulled yet)"
                console.print(Panel(
                    f"Model [cyan]{model_name}[/cyan] is not available in Ollama.\n\n"
                    f"[yellow]To fix:[/yellow]\n"
                    f"  Run: [cyan]ollama pull {model_name}[/cyan]\n\n"
                    f"[bold]Models currently pulled:[/bold]\n{model_list}",
                    title="❌  Ollama model not found",
                    border_style="red",
                ))
                return False

        except urllib.error.URLError:
            console.print(Panel(
                "[bold]CoreInsight defaults to Ollama for free local inference.[/bold]\n\n"
                "[yellow]To fix:[/yellow]\n"
                "  1. Install:    [cyan]https://ollama.com/download[/cyan]\n"
                "  2. Start it:   [cyan]ollama serve[/cyan]  (auto-starts on most systems after install)\n"
                "  3. Pull model: [cyan]ollama pull llama3.2[/cyan]\n\n"
                "[bold]Prefer a cloud model instead?[/bold]\n"
                "  Run: [cyan]coreinsight configure[/cyan]",
                title="❌  Ollama is not running",
                border_style="red",
            ))
            return False

    return True

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
    tier_limits = get_tier_limits(config)
    pro_user = is_pro(config)

    # ── Preflight: fail fast before AST parsing, image building, or thread spawning ──
    if not _preflight_checks(provider, model_name):
        sys.exit(1)

    tier_label = "[bold green]Pro[/bold green]" if pro_user else "[bold yellow]Free[/bold yellow]"
    console.print(Panel.fit(f"🚀 CoreInsight: Profiling [bold cyan]{path.name}[/bold cyan] via [bold]{provider}[/bold] · {tier_label}"))

    # 1. PARSE (Synchronous, since it's fast)
    with console.status("[yellow]Parsing AST and extracting hot loops...[/yellow]"):
        parser = CodeParser()
        content_bytes = path.read_bytes()
        file_content  = content_bytes.decode("utf-8", errors="replace")
        functions = parser.parse_file(str(path), content_bytes)
    
    if not functions:
        console.print("[red]No parseable functions found in the file.[/red]")
        sys.exit(1)

    max_fn = tier_limits["max_functions"]
    if max_fn is not None and len(functions) > max_fn:
        console.print(
            f"[yellow]⚠ Free tier: analysing the first {max_fn} of {len(functions)} functions. "
            f"Upgrade to Pro for unlimited → [cyan underline]{PRO_WAITLIST_URL}[/cyan underline][/yellow]"
        )
        functions = functions[:max_fn]

    console.print(f"[green]✅ Extracted {len(functions)} functional kernels.[/green]\n")

    # Initialize heavy lifters
    try:
        hardware_specs_dict = HardwareDetector.get_system_specs()
        hardware_target_str = HardwareDetector.format_for_llm(hardware_specs_dict)
        
        console.print(f"[dim]🎯 Target Hardware Detected: {hardware_target_str}[/dim]")

        model_tier = get_model_tier(provider, model_name)
        agent   = AnalyzerAgent(provider=provider, model_name=model_name, api_keys=api_keys, model_tier=model_tier)
        sandbox = CodeSandbox()
        db_path = path.parent / ".coreinsight_db"
        indexer  = RepoIndexer(str(path.parent)) if db_path.exists() else None
        profiler = HardwareProfiler() if pro_user else None
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
            executor.submit(process_function, func, language, agent, sandbox, indexer, hardware_target_str, tier_limits, profiler, file_content, str(path.parent)): func
            for func in functions
        }

        # As each thread finishes, process its output instantly
        for future in concurrent.futures.as_completed(future_to_func):
            func = future_to_func[future]
            try:
                func_name, result, sandbox_res, msg, verification, profiler_result = future.result()

                # 1. Format and save to Markdown file
                output_md = format_report_markdown(func_name, result, sandbox_res, msg, language, path.parent, verification, profiler_result)

                # Write to File (Safely)
                with file_lock:
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write(output_md)

                # 2. Print beautiful native UI to the terminal
                with print_lock:
                    print_console_report(func_name, result, sandbox_res, msg, language, verification, profiler_result)
                
            except Exception as exc:
                with print_lock:
                    console.print(f"[bold red]❌ Critical failure in thread processing {func['name']}:[/bold red] {exc}")

    console.print(Panel.fit(f"✅ [bold green]Analysis Complete![/bold green] Final report saved to:\n{report_path.absolute()}"))

    if not pro_user:
        console.print(Panel(
            "[bold]Enjoyed CoreInsight?[/bold] Pro unlocks:\n"
            "  • [cyan]Cloud models[/cyan] (GPT-4o, Claude, Gemini)\n"
            "  • [cyan]Unlimited functions[/cyan] per file\n"
            "  • [cyan]5 retry attempts[/cyan] + deeper test coverage\n\n"
            f"[yellow]Join the waitlist:[/yellow] [cyan underline]{PRO_WAITLIST_URL}[/cyan underline]",
            title="⚡  Upgrade to Pro",
            border_style="yellow",
        ))

def run_demo(lang: str = "python"):
    import shutil
    import importlib.resources

    # ── Config awareness: guide unconfigured users before doing any work ─────
    config = load_config()
    provider = config.get("provider", "ollama")
    model_name = config.get("model_name", "llama3.2")

    is_default = not (Path.home() / ".coreinsight" / "config.json").exists()
    if is_default:
        console.print(Panel(
            "[bold]No configuration found.[/bold] CoreInsight defaults to [cyan]Ollama + llama3.2[/cyan].\n\n"
            "[yellow]Quick setup options:[/yellow]\n"
            "  [bold]Option A[/bold] — Local (free, no API key):\n"
            "    1. Install Ollama: [cyan]https://ollama.com/download[/cyan]\n"
            "    2. Pull a model:   [cyan]ollama pull deepseek-coder[/cyan]\n"
            "    3. Run this again: [cyan]coreinsight demo[/cyan]\n\n"
            "  [bold]Option B[/bold] — Cloud (requires API key):\n"
            "    Run [cyan]coreinsight configure[/cyan] and choose your provider.\n\n"
            "[dim]Continuing with default config — preflight will catch any issues.[/dim]",
            title="⚙️  Setup Required",
            border_style="yellow",
        ))
    else:
        console.print(
            f"[dim]Using configured provider: [bold]{provider}[/bold] / {model_name}[/dim]"
        )

    demo_dir = Path.cwd() / "coreinsight_demo"
    demo_dir.mkdir(exist_ok=True)

    # Clear stale ChromaDB between language switches — a DB indexed for
    # Python files is incompatible with a C++ demo run and causes tenant errors
    stale_db = demo_dir / ".coreinsight_db"
    if stale_db.exists():
        import shutil as _shutil
        _shutil.rmtree(stale_db, ignore_errors=True)

    if lang == "python":
        demo_files = ["bad_loop.py", "data_processor.py"]
        entry_file = "data_processor.py"
    else:
        demo_files = ["slow.cpp"]
        entry_file = "slow.cpp"

    # Copy bundled demo files into the local demo directory
    for fname in demo_files:
        src = importlib.resources.files("coreinsight.demo").joinpath(fname)
        with importlib.resources.as_file(src) as p:
            shutil.copy(str(p), demo_dir / fname)

    console.print(Panel(
        f"CoreInsight will analyse a built-in [bold cyan]{lang.upper()}[/bold cyan] example.\n\n"
        f"[dim]Demo files copied to:[/dim] [underline]{demo_dir}[/underline]\n"
        f"[dim]The full report will be saved there when analysis completes.[/dim]",
        title="🎬  CoreInsight Demo",
        border_style="cyan",
    ))

    # For Python: auto-index so RAG cross-file context is showcased
    if lang == "python":
        console.print("[dim]Auto-indexing demo files to showcase RAG cross-file context...[/dim]")
        from coreinsight.indexer import RepoIndexer as _RepoIndexer
        _RepoIndexer(str(demo_dir)).index_repository()
        console.print()

    run_analysis(str(demo_dir / entry_file))

def main_cli():
    parser = argparse.ArgumentParser(description="CoreInsight CLI - Local Hardware Optimization")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    configure_parser = subparsers.add_parser("configure", help="Set up AI providers and API keys")
    configure_parser.add_argument("--pro-key", dest="pro_key", help="Activate Pro tier with your key", default=None)

    demo_parser = subparsers.add_parser("demo", help="Run CoreInsight on a built-in example file")
    demo_parser.add_argument(
        "--lang",
        choices=["python", "cpp"],
        default="python",
        help="Language of the demo file (default: python)",
    )
    
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a local code file")
    analyze_parser.add_argument("file", help="The .cpp, .cu, or .py file to analyze")
    
    index_parser = subparsers.add_parser("index", help="Index the current repository for global AI context")
    index_parser.add_argument("--dir", default=".", help="Directory to index")
    
    scan_parser = subparsers.add_parser("scan", help="Scan directory for complex hotspots")
    scan_parser.add_argument("--dir", default=".", help="Directory to scan")
    scan_parser.add_argument("--top", type=int, default=10, help="Number of hotspots to show")
    
    args = parser.parse_args()
    
    if args.command == "configure":
        run_configure(pro_key=getattr(args, "pro_key", None))
    elif args.command == "demo":
        run_demo(getattr(args, "lang", "python"))
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