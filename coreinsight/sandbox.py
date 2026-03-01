import os
import tempfile
import docker
import logging
import importlib
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

SANDBOX_IMAGES = {
    "python": "coreinsight-python-sandbox:latest",
    "cpp":    "coreinsight-cpp-sandbox:latest",
}

DOCKERFILES = {
    "python": "Dockerfile.python-sandbox",
    "cpp":    "Dockerfile.cpp-sandbox",
}


class CodeSandbox:
    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to connect to Docker daemon: {e}")
            self.client = None
            return

        # Prebake all non-CUDA images at startup
        for lang in ["python", "cpp"]:
            self._ensure_sandbox_image(lang)

    def _ensure_sandbox_image(self, lang: str):
        image_tag = SANDBOX_IMAGES[lang]
        try:
            self.client.images.get(image_tag)
            logger.info(f"{image_tag} already exists, skipping build.")
        except docker.errors.ImageNotFound:
            self._build_sandbox_image(lang)

    def _build_sandbox_image(self, lang: str):
        from rich.console import Console
        console = Console()
        label = "Python" if lang == "python" else "C++"
        console.print(f"[yellow]First run: building {label} sandbox image (one-time, ~30s)...[/yellow]")

        dockerfile_path = importlib.resources.files("coreinsight").joinpath(DOCKERFILES[lang])
        with importlib.resources.as_file(dockerfile_path) as dockerfile:
            _, logs = self.client.images.build(
                path=str(dockerfile.parent),
                dockerfile=dockerfile.name,
                tag=SANDBOX_IMAGES[lang],
                rm=True,
            )
            for chunk in logs:
                if "stream" in chunk:
                    logger.debug(chunk["stream"].strip())

        console.print(f"[green]✓ {label} sandbox image built successfully.[/green]")

    def execute_benchmark(self, code: str, language: str = "cpp", timeout_seconds: int = 120) -> Tuple[bool, str, Optional[bytes]]:
        if not self.client:
            return False, "Docker is not running on the host machine.", None

        if language in ["cuda", "cu", "cuh"]:
                timeout_seconds = max(timeout_seconds, 300)

        if language in ["cuda", "cu", "cuh"]:
            image = "nvidia/cuda:12.2.0-devel-ubuntu22.04"
            compile_cmd = "nvcc -O3 source.cu -o benchmark && ./benchmark"
            filename = "source.cu"
            device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
        elif language == "python":
            image = SANDBOX_IMAGES["python"]
            compile_cmd = "python -u -W ignore source.py"
            filename = "source.py"
            device_requests = []
        else:
            image = SANDBOX_IMAGES["cpp"]
            compile_cmd = "g++ -O3 -std=c++17 source.cpp -o benchmark && ./benchmark"
            filename = "source.cpp"
            device_requests = []

        with tempfile.TemporaryDirectory() as temp_dir:
            os.chmod(temp_dir, 0o777)
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                f.write(code)
            os.chmod(file_path, 0o777)

            container = None
            try:
                logger.info(f"Spinning up {image} sandbox for {filename}...")
                container = self.client.containers.run(
                    image=image,
                    command=["/bin/sh", "-c", f"timeout {timeout_seconds}s {compile_cmd}"],
                    volumes={temp_dir: {'bind': '/workspace', 'mode': 'rw'}},
                    working_dir="/workspace",
                    detach=True,
                    mem_limit="2g",
                    network_disabled=True,  # Only needs network if pip runs
                    device_requests=device_requests,
                    environment={"OMP_NUM_THREADS": "1"},
                    pids_limit=200,
                    cap_drop=["ALL"] if language not in ["cuda", "cu", "cuh"] else [],
                )

                exit_code = container.wait(timeout=timeout_seconds + 10)["StatusCode"]
                raw_logs = container.logs(stdout=True, stderr=True).decode("utf-8")

                plot_data = None
                plot_path = os.path.join(temp_dir, "benchmark_plot.png")
                if os.path.exists(plot_path):
                    with open(plot_path, "rb") as pf:
                        plot_data = pf.read()

                if exit_code == 124:
                    return False, f"⚠️ TIMEOUT: Killed after {timeout_seconds}s. Likely an infinite loop.\nPartial output:\n{raw_logs}", None

                csv_header = "N,Original_Time,Optimized_Time,Speedup"
                if csv_header in raw_logs:
                    clean_csv = raw_logs[raw_logs.find(csv_header):].strip()
                    return (exit_code == 0), clean_csv, plot_data
                else:
                    return False, f"Missing CSV output (exit {exit_code}).\nFull output:\n{raw_logs}", None

            except Exception as e:
                return False, f"Sandbox error: {str(e)}", None

            finally:
                if container:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass