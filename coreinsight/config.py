import json
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm

console = Console()
CONFIG_FILE = Path.home() / ".coreinsight" / "config.json"

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
        default=config.get("provider", "ollama")
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