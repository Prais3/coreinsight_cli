import json
import hashlib
import urllib
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()
CONFIG_FILE = Path.home() / ".coreinsight" / "config.json"

PRO_WAITLIST_URL = "https://tally.so/r/xXZ9YE"

# Cloudflare Worker endpoint for Pro key validation (v0.3.0+)
PRO_KEY_VALIDATION_URL = "https://coreinsight.coreinsight-dev.workers.dev/"

CLOUD_PROVIDERS = ["openai", "anthropic", "google"]

FREE_TIER_LIMITS = {
    "max_functions":      None,  # unlimited
    "max_retries":        3,
    "num_test_cases":     5,
    "hardware_profiling": False,
    "max_files":          None,
}

PRO_TIER_LIMITS = {
    "max_functions":     None,   # unlimited
    "max_retries":       5,
    "num_test_cases":    15,
    "hardware_profiling": True,
    "max_files": None,
}

SMALL_MODELS  = ["llama3.2:3b", "llama3.2:1b", "codellama:7b", "llama3:7b", "mistral:7b"]
MEDIUM_MODELS = ["codellama:13b", "llama3:13b", "mistral:13b", "llama3.1:8b"]
CLOUD_MODELS  = ["openai", "anthropic", "google"]  # always large tier

def get_model_tier(provider: str, model_name: str) -> str:
    from coreinsight.prompts import ModelTier
    if provider in CLOUD_MODELS:
        return ModelTier.LARGE
    name = model_name.lower()
    if any(m in name for m in SMALL_MODELS):
        return ModelTier.SMALL
    if any(m in name for m in MEDIUM_MODELS):
        return ModelTier.MEDIUM
    # Unknown local model: default to MEDIUM (more guidance, less noise than SMALL)
    return ModelTier.MEDIUM


def is_pro(config: dict) -> bool:
    return bool(config.get("pro", False))


def get_tier_limits(config: dict) -> dict:
    return PRO_TIER_LIMITS if is_pro(config) else FREE_TIER_LIMITS


def get_agent_mode(config: dict) -> str:
    """
    Returns "multi" or "single".

    Priority:
      1. Explicit user override stored in config ("agent_mode" key)
      2. Auto-selection based on model tier:
           - small / medium local models  → "multi"
             (focused prompts compensate for smaller context windows)
           - large / cloud models         → "single"
             (large models handle full context fine; saves API cost)
    """
    explicit = config.get("agent_mode")
    if explicit in ("single", "multi"):
        return explicit

    provider   = config.get("provider",   "ollama")
    model_name = config.get("model_name", "llama3.2")
    tier       = get_model_tier(provider, model_name)

    from coreinsight.prompts import ModelTier
    if tier in (ModelTier.SMALL, ModelTier.MEDIUM):
        return "multi"
    return "single"

def load_config():
    if not CONFIG_FILE.exists():
        return {"provider": "ollama", "model_name": "llama3.2", "api_keys": {}}
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(config_data):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)

def run_configure(pro_key: str = None, agent_mode: str = None):
    """Interactive CLI to set up models and API keys."""
    console.print("[bold cyan]⚙️ CoreInsight Configuration[/bold cyan]")

    config = load_config()

    if pro_key is not None:
        pro_key = pro_key.strip()
        if not pro_key:
            console.print("[red]❌ Invalid pro key format.[/red]")
            return

        console.print("[cyan]Verifying Pro key...[/cyan]")
        key_hash = hashlib.sha256(pro_key.encode()).hexdigest()

        try:
            payload = json.dumps({"hash": key_hash}).encode()
            req = urllib.request.Request(
                PRO_KEY_VALIDATION_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "coreinsight-cli/0.3.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                result = json.loads(response.read().decode())

            if result.get("valid"):
                config["pro"] = True
                save_config(config)
                console.print("[bold green]✅ Pro tier activated![/bold green]")
            else:
                config["pro"] = False
                save_config(config)
                console.print("[red]❌ Invalid or revoked Pro key.[/red]")
        except Exception:
            console.print(
                "[red]⚠️ Could not verify key — check your internet connection "
                "or try again later.[/red]"
            )
        return

    if agent_mode is not None:
        if agent_mode in ("single", "multi"):
            config["agent_mode"] = agent_mode
            save_config(config)
            console.print(
                f"[bold green]✅ Agent mode set to [cyan]{agent_mode}[/cyan].[/bold green]\n"
                f"[dim]Use [cyan]coreinsight configure --agent-mode auto[/cyan] "
                f"to restore automatic selection.[/dim]"
            )
        elif agent_mode == "auto":
            config.pop("agent_mode", None)
            save_config(config)
            console.print(
                "[bold green]✅ Agent mode reset to automatic selection.[/bold green]"
            )
        else:
            console.print(
                "[red]Invalid agent mode. Choose from: single, multi, auto[/red]"
            )
        return
    
    provider = Prompt.ask(
        "Which AI provider do you want to use?",
        choices=["ollama", "local_server", "openai", "anthropic", "google"],
        default=config.get("provider", "ollama"),
    )

    if provider in CLOUD_PROVIDERS and not is_pro(config):
        from rich.panel import Panel
        console.print(Panel(
            f"[bold]Cloud providers ({', '.join(CLOUD_PROVIDERS)}) are a Pro feature.[/bold]\n\n"
            f"Free tier supports [cyan]ollama[/cyan] and [cyan]local_server[/cyan] only.\n\n"
            f"[yellow]Join the Pro waitlist:[/yellow] [cyan underline]{PRO_WAITLIST_URL}[/cyan underline]\n\n"
            "[dim]Already have a Pro key? Run [cyan]coreinsight configure --pro-key <key>[/cyan] to unlock.[/dim]",
            title="🔒  Pro Feature",
            border_style="yellow",
        ))
        provider = Prompt.ask(
            "Switch to a free provider instead?",
            choices=["ollama", "local_server"],
            default="ollama",
        )

    config["provider"] = provider
    
    if provider == "ollama":
        config["model_name"] = Prompt.ask("Ollama model name", default=config.get("model_name", "llama3.2"))
    elif provider == "local_server":
        from rich.panel import Panel
        console.print(Panel(
            "[bold]Local inference server setup[/bold]\n\n"
            "CoreInsight talks to any OpenAI-compatible local server.\n"
            "Choose the option that matches how you loaded your weights:\n\n"
            "[bold cyan]Option A — GGUF weights (llama.cpp):[/bold cyan]\n"
            "  pip install llama-cpp-python\\[server]\n"
            "  python -m llama_cpp.server --model your_model.gguf --port 1234\n\n"
            "[bold cyan]Option B — PyTorch / HuggingFace weights (vLLM):[/bold cyan]\n"
            "  pip install vllm\n"
            "  python -m vllm.entrypoints.openai.api_server \\\\\n"
            "      --model /path/to/weights --port 1234\n\n"
            "[bold cyan]Option C — LM Studio (GUI, easiest):[/bold cyan]\n"
            "  1. Load your model in LM Studio\n"
            "  2. Click [bold]Start Server[/bold] (defaults to localhost:1234)\n"
            "  3. Enter the URL below\n\n"
            "[dim]All three expose an OpenAI-compatible API on the URL you provide.[/dim]",
            title="⚙️  Local Inference Server",
            border_style="cyan",
        ))
        config["model_name"] = Prompt.ask(
            "Model name (shown in server logs, or 'local-model')",
            default=config.get("model_name", "local-model"),
        )
        config["api_keys"]["local_url"] = Prompt.ask(
            "Server base URL",
            default=config.get("api_keys", {}).get("local_url", "http://localhost:1234/v1"),
        )
    elif provider == "openai":
        config["model_name"] = Prompt.ask("OpenAI model name", default="gpt-4o")
        config["api_keys"]["openai"] = Prompt.ask("OpenAI API Key (hidden)", password=True)
    elif provider == "anthropic":
        config["model_name"] = Prompt.ask("Claude model name", default="claude-3-5-sonnet-latest")
        config["api_keys"]["anthropic"] = Prompt.ask("Anthropic API Key (hidden)", password=True)
    elif provider == "google":
        config["model_name"] = Prompt.ask("Gemini model name", default="gemini-1.5-pro")
        config["api_keys"]["google"] = Prompt.ask("Google Gemini API Key (hidden)", password=True)

    save_config(config)
    console.print("\n[bold green]✅ Configuration saved successfully![/bold green]")
    console.print(f"CoreInsight is now using [bold]{provider}[/bold] ({config['model_name']}).")