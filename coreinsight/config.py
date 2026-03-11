import json
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()
CONFIG_FILE = Path.home() / ".coreinsight" / "config.json"

# TODO: swap for your real URL
PRO_WAITLIST_URL = "https://tally.so/r/coreinsight-pro"

CLOUD_PROVIDERS = ["openai", "anthropic", "google"]

FREE_TIER_LIMITS = {
    "max_functions":  3,
    "max_retries":    2,
    "num_test_cases": 8,
}

PRO_TIER_LIMITS = {
    "max_functions":  None,   # unlimited
    "max_retries":    5,
    "num_test_cases": 15,
}


def is_pro(config: dict) -> bool:
    return bool(config.get("pro", False))


def get_tier_limits(config: dict) -> dict:
    return PRO_TIER_LIMITS if is_pro(config) else FREE_TIER_LIMITS

def load_config():
    if not CONFIG_FILE.exists():
        return {"provider": "ollama", "model_name": "llama3.2", "api_keys": {}}
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(config_data):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)

def run_configure():
    """Interactive CLI to set up models and API keys."""
    console.print("[bold cyan]⚙️ CoreInsight Configuration[/bold cyan]")
    
    config = load_config()
    
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
        config["model_name"] = Prompt.ask("Local model name (optional)", default=config.get("model_name", "local-model"))
        config["api_keys"]["local_url"] = Prompt.ask("Local Server Base URL", default="http://localhost:1234/v1")
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