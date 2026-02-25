import platform
import psutil
import subprocess

class HardwareDetector:
    @staticmethod
    def get_system_specs() -> dict:
        """Dynamically detects the host system's CPU, RAM, and GPU."""
        specs = {
            "os": platform.system(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu": "None detected",
            "vram_gb": 0.0
        }
        
        # Attempt to detect NVIDIA GPU and VRAM
        try:
            smi_output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                encoding="utf-8",
                stderr=subprocess.DEVNULL
            ).strip()
            
            if smi_output:
                # E.g., "NVIDIA GeForce RTX 4090, 24564 MiB"
                parts = smi_output.split(',')
                specs["gpu"] = parts[0].strip()
                specs["vram_gb"] = round(int(parts[1].replace('MiB', '').strip()) / 1024, 2)
        except Exception:
            pass # No NVIDIA GPU found or drivers not installed

        return specs

    @staticmethod
    def format_for_llm(specs: dict) -> str:
        """Formats the dictionary into a readable block for the LLM prompt."""
        return (
            f"OS: {specs['os']} | "
            f"CPU: {specs['cpu_cores']} Cores ({specs['cpu_threads']} Threads) | "
            f"System RAM: {specs['ram_gb']} GB | "
            f"GPU: {specs['gpu']} | "
            f"GPU VRAM: {specs['vram_gb']} GB"
        )