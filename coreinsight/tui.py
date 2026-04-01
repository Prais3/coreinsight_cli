"""
coreinsight/tui.py — Interactive TUI for CoreInsight
Launch with: coreinsight view
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DirectoryTree,
    Footer,
    Header,
    Label,
    RichLog,
    Static,
    Switch,
)
from textual.widgets._directory_tree import DirEntry

from coreinsight.config import (
    load_config,
    is_pro,
    get_tier_limits,
    PRO_WAITLIST_URL,
)
from coreinsight.memory import OptimizationMemory

# ---------------------------------------------------------------------------
# Sentinel used to detect --no-docker skips (imported for display logic)
# ---------------------------------------------------------------------------
try:
    from coreinsight.sandbox import SANDBOX_SKIPPED_MSG
except ImportError:
    SANDBOX_SKIPPED_MSG = "Verification skipped (--no-docker)."


# ---------------------------------------------------------------------------
# TuiConsole — drop-in replacement for rich.Console that writes to RichLog
# ---------------------------------------------------------------------------

class _NoopStatus:
    """Stub so console.status(...) calls don't crash inside the TUI."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


class TuiConsole:
    """
    Mimics the rich.Console API used in main.py so run_analysis() output
    is captured and displayed in the TUI's RichLog widget in real time.
    """

    def __init__(self, log_widget: RichLog):
        self._log = log_widget
        self._lock = threading.Lock()

    # Main output method — handles str, Rich renderables, Panel, etc.
    def print(self, *args, **kwargs):
        with self._lock:
            for arg in args:
                try:
                    self._log.write(arg)
                except Exception:
                    self._log.write(str(arg))

    # console.log() is used by _log() helper in main.py
    def log(self, *args, **kwargs):
        self.print(*args)

    # console.status() is used for spinners — no-op in TUI
    def status(self, *args, **kwargs):
        return _NoopStatus()


# ---------------------------------------------------------------------------
# Memory modal
# ---------------------------------------------------------------------------

class MemoryModal(ModalScreen):
    """Full-screen modal showing the optimization memory store."""

    BINDINGS = [Binding("escape,q", "dismiss", "Close")]

    DEFAULT_CSS = """
    MemoryModal {
        align: center middle;
    }
    MemoryModal > Container {
        width: 90%;
        height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    MemoryModal #memory-log {
        height: 1fr;
    }
    MemoryModal #close-memory {
        margin-top: 1;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Container():
            yield Label("Optimization Memory", id="memory-title")
            yield RichLog(id="memory-log", highlight=True, markup=True)
            yield Button("Close  [Esc]", id="close-memory", variant="default")

    def on_mount(self) -> None:
        log = self.query_one("#memory-log", RichLog)
        self._populate(log)

    def _populate(self, log: RichLog) -> None:
        from rich.table import Table

        mem = OptimizationMemory()
        stats = mem.stats()
        count = stats.get("count", 0)

        if count == 0:
            log.write("[dim]No optimizations stored yet.[/dim]")
            log.write(
                f"\nRun [cyan]coreinsight analyze[/cyan] or use the Analyze action "
                f"to start building your memory store."
            )
            return

        try:
            if not mem._ensure_db():
                log.write("[red]Could not open memory store.[/red]")
                return
            all_records = mem._collection.get(include=["metadatas"])
            metadatas   = all_records.get("metadatas", []) or []
            ids         = all_records.get("ids",       []) or []
        except Exception as exc:
            log.write(f"[red]Failed to read memory store: {exc}[/red]")
            return

        table = Table(
            show_header=True, header_style="bold cyan",
            expand=True, show_lines=True,
        )
        table.add_column("#",          justify="right",  style="dim",        width=4)
        table.add_column("Function",   justify="left",   style="bold white")
        table.add_column("Language",   justify="center", style="cyan",       width=10)
        table.add_column("Speedup",    justify="right",  style="bold green", width=9)
        table.add_column("Severity",   justify="center",                     width=10)
        table.add_column("Issue",      justify="left",   style="dim white")
        table.add_column("Verified",   justify="left",   style="dim",        width=20)

        severity_colors = {
            "Critical": "red", "High": "yellow",
            "Medium": "cyan",  "Low": "green",
        }

        paired = sorted(
            zip(metadatas, ids),
            key=lambda x: x[0].get("timestamp", ""),
            reverse=True,
        )
        for i, (meta, rid) in enumerate(paired, start=1):
            sev   = meta.get("severity", "High")
            sev_c = severity_colors.get(sev, "white")
            ts    = meta.get("timestamp", "")[:19].replace("T", " ")
            issue = (meta.get("issue", "") or "—")[:55]
            if len(meta.get("issue", "")) > 55:
                issue += "…"
            table.add_row(
                str(i),
                meta.get("func_name", rid[:12]),
                meta.get("language", "?"),
                f"{float(meta.get('avg_speedup', 0)):.2f}x",
                f"[{sev_c}]{sev}[/{sev_c}]",
                issue,
                ts,
            )

        log.write(table)
        log.write(f"\n[dim]Store location: ~/.coreinsight/memory_db[/dim]")

    @on(Button.Pressed, "#close-memory")
    def close(self) -> None:
        self.dismiss()


# ---------------------------------------------------------------------------
# Confirm modal — used for destructive actions like memory clear
# ---------------------------------------------------------------------------

class ConfirmModal(ModalScreen[bool]):
    """Simple yes/no confirmation dialog."""

    BINDINGS = [Binding("escape", "dismiss_false", "Cancel")]

    DEFAULT_CSS = """
    ConfirmModal {
        align: center middle;
    }
    ConfirmModal > Container {
        width: 50;
        height: 10;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
        align: center middle;
    }
    ConfirmModal Horizontal {
        align: center middle;
        margin-top: 1;
    }
    ConfirmModal Button {
        margin: 0 1;
    }
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(**kwargs)
        self._message = message

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(self._message, id="confirm-msg")
            with Horizontal():
                yield Button("Yes", id="confirm-yes", variant="error")
                yield Button("No",  id="confirm-no",  variant="default")

    @on(Button.Pressed, "#confirm-yes")
    def yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def no(self) -> None:
        self.dismiss(False)

    def action_dismiss_false(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# Settings modal
# ---------------------------------------------------------------------------

class SettingsModal(ModalScreen):
    """Read-only settings viewer. Directs user to coreinsight configure for changes."""

    BINDINGS = [Binding("escape,q", "dismiss", "Close")]

    DEFAULT_CSS = """
    SettingsModal {
        align: center middle;
    }
    SettingsModal > Container {
        width: 60;
        height: 22;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    SettingsModal .setting-row {
        height: 1;
        margin-bottom: 1;
    }
    SettingsModal #close-settings {
        margin-top: 1;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        config    = load_config()
        pro_user  = is_pro(config)
        provider  = config.get("provider",   "ollama")
        model     = config.get("model_name", "llama3.2")
        tier      = "Pro" if pro_user else "Free"
        tier_col  = "bold green" if pro_user else "bold yellow"
        agent_mode = config.get("agent_mode", "auto")

        with Container():
            yield Label("Current Configuration", id="settings-title")
            yield Static(f"  Tier      : [{tier_col}]{tier}[/{tier_col}]", classes="setting-row")
            yield Static(f"  Provider  : [cyan]{provider}[/cyan]",          classes="setting-row")
            yield Static(f"  Model     : [cyan]{model}[/cyan]",              classes="setting-row")
            yield Static(f"  Agent mode: [cyan]{agent_mode}[/cyan]",         classes="setting-row")
            yield Static("", classes="setting-row")
            yield Static(
                "  To change settings, run:\n"
                "  [cyan]coreinsight configure[/cyan]",
                classes="setting-row",
            )
            if not pro_user:
                yield Static(
                    f"\n  [yellow]Unlock Pro (free during beta):[/yellow]\n"
                    f"  [cyan underline]{PRO_WAITLIST_URL}[/cyan underline]",
                    classes="setting-row",
                )
            yield Button("Close  [Esc]", id="close-settings", variant="default")

    @on(Button.Pressed, "#close-settings")
    def close(self) -> None:
        self.dismiss()


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------

class CoreInsightApp(App):

    TITLE    = "CoreInsight"
    CSS_PATH = None   # all CSS inline below

    BINDINGS = [
        Binding("q",     "quit",           "Quit"),
        Binding("a",     "analyze",        "Analyze"),
        Binding("i",     "index",          "Index"),
        Binding("m",     "view_memory",    "Memory"),
        Binding("s",     "view_settings",  "Settings"),
        Binding("ctrl+c","quit",           "Quit", show=False),
    ]

    CSS = """
    /* ── Layout ─────────────────────────────────────────────────────────── */
    Screen {
        layout: vertical;
    }

    #main-area {
        layout: horizontal;
        height: 1fr;
    }

    /* ── Left panel ─────────────────────────────────────────────────────── */
    #left-panel {
        width: 32;
        min-width: 24;
        layout: vertical;
        border-right: solid $primary-darken-2;
    }

    #file-panel {
        height: 1fr;
        border-bottom: solid $primary-darken-2;
        padding: 0 0;
    }

    #file-label {
        background: $primary-darken-2;
        color: $text;
        padding: 0 1;
        height: 1;
    }

    #file-tree {
        height: 1fr;
        padding: 0;
    }

    #selected-count {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface-darken-1;
    }

    /* ── Action panel ───────────────────────────────────────────────────── */
    #action-panel {
        height: auto;
        padding: 1;
    }

    #action-label {
        color: $text-muted;
        margin-bottom: 1;
    }

    #action-panel Button {
        width: 100%;
        margin-bottom: 1;
    }

    #no-docker-row {
        height: 3;
        align: left middle;
        margin-top: 1;
    }

    #no-docker-label {
        padding: 0 1;
        color: $text-muted;
    }

    /* ── Right panel (output) ───────────────────────────────────────────── */
    #right-panel {
        width: 1fr;
        layout: vertical;
    }

    #output-label {
        background: $primary-darken-2;
        color: $text;
        padding: 0 1;
        height: 1;
    }

    #output-log {
        height: 1fr;
        padding: 0 1;
    }

    /* ── Status bar ─────────────────────────────────────────────────────── */
    #status-bar {
        height: 1;
        background: $primary-darken-3;
        color: $text-muted;
        padding: 0 1;
        dock: bottom;
    }
    """

    def __init__(self, start_dir: str = "."):
        super().__init__()
        self._start_dir  = str(Path(start_dir).resolve())
        self._selected: set[str] = set()
        self._busy       = False

        config           = load_config()
        self._pro        = is_pro(config)
        self._tier_limits = get_tier_limits(config)

    # ── Compose ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="main-area"):

            # Left panel
            with Vertical(id="left-panel"):
                with Vertical(id="file-panel"):
                    yield Label(" Files", id="file-label")
                    yield DirectoryTree(self._start_dir, id="file-tree")
                    yield Label("No files selected", id="selected-count")

                with Vertical(id="action-panel"):
                    yield Label("Actions", id="action-label")
                    yield Button("Analyze [a]",      id="btn-analyze",  variant="success")
                    yield Button("Index   [i]",      id="btn-index",    variant="primary")
                    yield Button("Memory  [m]",      id="btn-memory",   variant="default")
                    yield Button("Settings [s]",     id="btn-settings", variant="default")
                    with Horizontal(id="no-docker-row"):
                        yield Switch(value=False, id="no-docker-switch")
                        yield Label("Skip Docker", id="no-docker-label")

            # Right panel — live output
            with Vertical(id="right-panel"):
                yield Label(" Output", id="output-label")
                yield RichLog(
                    id="output-log",
                    highlight=True,
                    markup=True,
                    wrap=True,
                )

        yield Label(" Ready — select files to begin.", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#output-log", RichLog)
        config   = load_config()
        pro_user = is_pro(config)
        tier_str = "[bold green]Pro[/bold green]" if pro_user else "[bold yellow]Free[/bold yellow]"

        log.write(f"[bold cyan]CoreInsight[/bold cyan]  {tier_str}")
        log.write("[dim]Select one or more files in the browser, then press Analyze.[/dim]")
        if not pro_user:
            log.write(
                f"[dim]Free tier: up to 2 files per run. "
                f"Pro is free during beta → [cyan]{PRO_WAITLIST_URL}[/cyan][/dim]"
            )
        log.write("")

    # ── File selection ────────────────────────────────────────────────────

    @on(DirectoryTree.FileSelected)
    def file_selected(self, event: DirectoryTree.FileSelected) -> None:
        path = str(event.path)

        # Only allow supported file types
        supported = {".py", ".cpp", ".cc", ".h", ".hpp", ".cu", ".cuh"}
        if Path(path).suffix.lower() not in supported:
            self._set_status(f"Unsupported file type: {Path(path).suffix}")
            return

        # Toggle selection
        if path in self._selected:
            self._selected.discard(path)
        else:
            # Enforce free-tier file limit
            max_files = self._tier_limits.get("max_files")
            if max_files and len(self._selected) >= max_files:
                log = self.query_one("#output-log", RichLog)
                log.write(
                    f"\n[yellow]Free tier: max {max_files} file(s) per run. "
                    f"Upgrade to Pro for unlimited → [cyan]{PRO_WAITLIST_URL}[/cyan][/yellow]"
                )
                self._set_status(f"Free tier: max {max_files} files. Upgrade to Pro for unlimited.")
                return
            self._selected.add(path)

        self._refresh_selected_label()

    def _refresh_selected_label(self) -> None:
        label = self.query_one("#selected-count", Label)
        n = len(self._selected)
        if n == 0:
            label.update("No files selected")
        elif n == 1:
            name = Path(next(iter(self._selected))).name
            label.update(f"1 selected: {name}")
        else:
            label.update(f"{n} files selected")

    # ── Actions ───────────────────────────────────────────────────────────

    @on(Button.Pressed, "#btn-analyze")
    def action_analyze(self) -> None:
        if self._busy:
            self._set_status("Already running — please wait.")
            return
        if not self._selected:
            self._set_status("Select at least one file first.")
            return
        no_docker = self.query_one("#no-docker-switch", Switch).value
        self._start_analysis(list(self._selected), no_docker)

    @on(Button.Pressed, "#btn-index")
    def action_index(self) -> None:
        if self._busy:
            self._set_status("Already running — please wait.")
            return
        if not self._selected:
            self._set_status("Select at least one file to determine its directory.")
            return
        # Index the parent directory of the first selected file
        target_dir = str(Path(next(iter(self._selected))).parent)
        self._start_index(target_dir)

    @on(Button.Pressed, "#btn-memory")
    def action_view_memory(self) -> None:
        self.push_screen(MemoryModal())

    @on(Button.Pressed, "#btn-settings")
    def action_view_settings(self) -> None:
        self.push_screen(SettingsModal())

    # ── Workers ───────────────────────────────────────────────────────────

    @work(thread=True)
    def _start_analysis(self, files: list[str], no_docker: bool) -> None:
        from coreinsight.main import run_analysis

        log = self.query_one("#output-log", RichLog)
        tui_console = TuiConsole(log)
        self._busy = True

        for i, file_path in enumerate(files, 1):
            name = Path(file_path).name
            self.call_from_thread(
                self._set_status,
                f"Analyzing {name} ({i}/{len(files)})...",
            )
            self.call_from_thread(
                log.write,
                f"\n[bold cyan]{'─' * 60}[/bold cyan]"
                f"\n[bold]Analyzing:[/bold] [cyan]{name}[/cyan]\n",
            )
            try:
                run_analysis(file_path, no_docker=no_docker, tui_console=tui_console)
            except SystemExit:
                # run_analysis calls sys.exit(1) on bad input — catch it
                pass
            except Exception as exc:
                self.call_from_thread(
                    log.write,
                    f"[red]Unexpected error analyzing {name}: {exc}[/red]",
                )

        self._busy = False
        self.call_from_thread(
            self._set_status,
            f"Done. {len(files)} file(s) analyzed.",
        )

    @work(thread=True)
    def _start_index(self, target_dir: str) -> None:
        from coreinsight.indexer import RepoIndexer

        log = self.query_one("#output-log", RichLog)
        self._busy = True
        self.call_from_thread(
            self._set_status,
            f"Indexing {target_dir}...",
        )
        self.call_from_thread(
            log.write,
            f"\n[bold cyan]Indexing directory:[/bold cyan] [cyan]{target_dir}[/cyan]\n",
        )
        try:
            indexer = RepoIndexer(target_dir)
            indexer.index_repository()
            self.call_from_thread(
                log.write,
                "[bold green]Indexing complete.[/bold green]\n",
            )
        except Exception as exc:
            self.call_from_thread(
                log.write,
                f"[red]Indexing failed: {exc}[/red]\n",
            )
        finally:
            self._busy = False
            self.call_from_thread(self._set_status, "Indexing done.")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-bar", Label).update(f" {msg}")


# ---------------------------------------------------------------------------
# Entry point called from main.py
# ---------------------------------------------------------------------------

def run_tui(start_dir: str = ".") -> None:
    """Launch the CoreInsight TUI. Called by `coreinsight view`."""
    app = CoreInsightApp(start_dir=start_dir)
    app.run()