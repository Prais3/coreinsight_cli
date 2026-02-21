import os
import tempfile
import docker
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class CodeSandbox:
    def __init__(self):
        try:
            # Connects to the local Docker daemon
            self.client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to connect to Docker daemon: {e}")
            self.client = None

    def execute_benchmark(self, code: str, language: str = "cpp", timeout_seconds: int = 10) -> Tuple[bool, str]:
        """
        Compiles and runs the code in an isolated, ephemeral Docker container.
        Returns (success_boolean, output_string_or_error)
        """
        if not self.client:
            return False, "Docker is not running on the host machine."

        # 1. Setup image and execution command based on language
        if language in ["cuda", "cu", "cuh"]:
            image = "nvidia/cuda:12.2.0-devel-ubuntu22.04"
            compile_cmd = "nvcc -O3 source.cu -o benchmark && ./benchmark"
            filename = "source.cu"
            device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])]
        elif language == "python":
            image = "python:3.11-slim"
            compile_cmd = "python source.py"
            filename = "source.py"
            device_requests = []
        else:
            # Default to C++
            image = "gcc:latest"
            compile_cmd = "g++ -O3 -std=c++17 source.cpp -o benchmark && ./benchmark"
            filename = "source.cpp"
            device_requests = []

        # 2. Enforce strict timeout using the OS-level 'timeout' command
        # This guarantees the Celery worker won't hang if the code has an infinite loop
        safe_command = f"/bin/sh -c 'timeout {timeout_seconds}s sh -c \"{compile_cmd}\"'"

        # Create a temporary directory to share the file with the container
        with tempfile.TemporaryDirectory() as temp_dir:
            # FIX: Force permissions wide open so the Docker container can read/write
            os.chmod(temp_dir, 0o777) 
            
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                f.write(code)
            os.chmod(file_path, 0o777) # Force file permissions too

            try:
                # Run the ephemeral container
                logger.info(f"Spinning up {image} sandbox...")
                
                # detach=False blocks and returns the byte logs directly
                logs = self.client.containers.run(
                    image=image,
                    command=safe_command,
                    volumes={temp_dir: {'bind': '/workspace', 'mode': 'rw'}},
                    working_dir="/workspace",
                    remove=True, # Auto-destroy container when done
                    detach=False,
                    mem_limit="512m", 
                    network_disabled=True, 
                    device_requests=device_requests,
                    environment={"OMP_NUM_THREADS": "1"},
                    pids_limit=50,   # Security: Prevent fork bombs
                    cap_drop=["ALL"] # Security: Drop unnecessary Linux capabilities
                )
                
                return True, logs.decode('utf-8')

            except docker.errors.ContainerError as e:
                # Exit code 124 is the standard code for the Linux 'timeout' command
                if e.exit_status == 124:
                    return False, f"⚠️ **TIMEOUT EXCEEDED:**\nThe execution was killed after {timeout_seconds} seconds. The AI likely generated an infinite loop."
                
                # The compilation failed, or the code crashed at runtime
                error_logs = e.stderr.decode('utf-8') if e.stderr else str(e)
                return False, f"Execution/Compilation Failed:\n{error_logs}"
            
            except docker.errors.APIError as e:
                # Catch the specific missing GPU driver error
                if "could not select device driver" in str(e) or "capabilities: [[gpu]]" in str(e):
                    logger.warning("Host lacks NVIDIA Container Toolkit or GPU. Skipping CUDA benchmark.")
                    return False, "⚠️ **HARDWARE LIMITATION:**\nThe host worker node does not have an NVIDIA GPU or the NVIDIA Container Toolkit installed. The AI generated the optimization, but we cannot physically run the CUDA benchmark on this specific server."
                return False, f"Docker API Error: {str(e)}"
                
            except Exception as e:
                return False, f"Sandbox System Error: {str(e)}"