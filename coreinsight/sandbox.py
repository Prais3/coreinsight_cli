import os
import tempfile
import docker
import logging
import importlib
from typing import Tuple, Optional, List, Dict, Any

import io
import csv
import math
import json
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SANDBOX_IMAGES = {
    "python": "coreinsight-python-sandbox:latest",
    "cpp":    "coreinsight-cpp-sandbox:latest",
}

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DOCKERFILES = {
    "python": os.path.join(ROOT_DIR, "docker", "Dockerfile.python-sandbox"),
    "cpp":    os.path.join(ROOT_DIR, "docker", "Dockerfile.cpp-sandbox"),
}

# ---------------------------------------------------------------------------
# Verification constants
# ---------------------------------------------------------------------------
SANDBOX_SKIPPED_MSG = "Verification skipped (--no-docker)."
SPEEDUP_DISCREPANCY_TOLERANCE = 0.05  # max relative delta: computed vs reported speedup
MIN_TIMING_ROWS = 2                   # minimum CSV rows to trust timing statistics
FLOAT_RTOL = 1e-5                     # relative tolerance for output comparison
FLOAT_ATOL = 1e-8                     # absolute tolerance for output comparison

_PYTHON_CORRECTNESS_HARNESS = '''
import json, sys, math, importlib.util, traceback

spec = importlib.util.spec_from_file_location("user_module", "/workspace/source.py")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

original_fn  = getattr(mod, "{original_func_name}",  None)
optimized_fn = getattr(mod, "{optimized_func_name}", None)

if original_fn is None:
    print(json.dumps({{"error": "source.py has no function named \'{original_func_name}\'"}}))
    sys.exit(1)
if optimized_fn is None:
    print(json.dumps({{"error": "source.py has no function named \'{optimized_func_name}\'"}}))
    sys.exit(1)

with open("/workspace/test_cases.json") as fh:
    test_cases = json.load(fh)

FLOAT_RTOL = {rtol}
FLOAT_ATOL = {atol}

def _approx_equal(a, b, path=""):
    if isinstance(a, float) and isinstance(b, float):
        if not (math.isfinite(a) and math.isfinite(b)):
            ok = (math.isnan(a) == math.isnan(b)) and (math.isinf(a) == math.isinf(b))
            return ok, f"{{path}}: {{a}} vs {{b}}"
        ok = abs(a - b) <= FLOAT_ATOL + FLOAT_RTOL * max(abs(a), abs(b))
        return ok, f"{{path}}: {{a}} vs {{b}} (delta={{abs(a-b):.3e}})"
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False, f"{{path}}: length {{len(a)}} vs {{len(b)}}"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, msg = _approx_equal(x, y, f"{{path}}[{{i}}]")
            if not ok:
                return False, msg
        return True, ""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False, f"{{path}}: keys differ"
        for k in a:
            ok, msg = _approx_equal(a[k], b[k], f"{{path}}.{{k}}")
            if not ok:
                return False, msg
        return True, ""
    return (a == b), f"{{path}}: {{repr(a)}} vs {{repr(b)}}"

results = []
for i, case in enumerate(test_cases):
    args   = case.get("args", [])
    kwargs = case.get("kwargs", {{}})
    try:
        out_orig = original_fn(*args, **kwargs)
        out_opt  = optimized_fn(*args, **kwargs)
        ok, msg  = _approx_equal(out_orig, out_opt, "output")
        results.append({{"case": i, "passed": ok, "detail": msg or "outputs match"}})
    except Exception:
        results.append({{"case": i, "passed": False, "detail": traceback.format_exc()}})

print(json.dumps(results))
'''.strip()


@dataclass
class SpeedupVerification:
    verified: bool
    computed_speedups: List[float] = field(default_factory=list)
    reported_speedups: List[float] = field(default_factory=list)
    max_discrepancy: Optional[float] = None
    suspicious_flags: List[str] = field(default_factory=list)
    details: str = ""


@dataclass
class CorrectnessVerification:
    verified: bool
    total_cases: int = 0
    passed_cases: int = 0
    failures: List[str] = field(default_factory=list)
    details: str = ""


@dataclass
class VerificationResult:
    speedup: SpeedupVerification
    correctness: CorrectnessVerification

    @property
    def fully_verified(self) -> bool:
        return self.speedup.verified and self.correctness.verified

    def summary(self) -> str:
        lines = [
            "=== Verification Report ===",
            f"Speedup integrity : {'✓ PASS' if self.speedup.verified else '✗ FAIL'}",
            f"Output correctness: {'✓ PASS' if self.correctness.verified else '✗ FAIL'}",
        ]
        if self.speedup.computed_speedups:
            avg = sum(self.speedup.computed_speedups) / len(self.speedup.computed_speedups)
            lines.append(f"  Computed avg speedup : {avg:.3f}x")
        if self.speedup.max_discrepancy is not None:
            lines.append(f"  Max speedup discrepancy vs reported: {self.speedup.max_discrepancy:.2%}")
        if self.speedup.suspicious_flags:
            lines.append("  Suspicious flags: " + "; ".join(self.speedup.suspicious_flags))
        if self.correctness.total_cases:
            lines.append(f"  Correctness: {self.correctness.passed_cases}/{self.correctness.total_cases} test cases passed")
        for f in self.correctness.failures:
            lines.append(f"    ✗ {f}")
        if self.speedup.details:
            lines.append(f"  Speedup detail: {self.speedup.details}")
        if self.correctness.details:
            lines.append(f"  Correctness detail: {self.correctness.details}")
        return "\n".join(lines)


class CodeSandbox:
    def __init__(self, disabled: bool = False):
        self.disabled = disabled
        if disabled:
            self.client = None
            self._init_error = None
            return
        self._init_error: Optional[str] = None
        try:
            self.client = docker.from_env()
            self.client.ping()          # actually proves the daemon is alive
        except docker.errors.DockerException as e:
            self.client = None
            self._init_error = str(e)
            logger.error(f"Docker unavailable: {e}")
            return
        except Exception as e:
            self.client = None
            self._init_error = str(e)
            logger.error(f"Unexpected Docker error: {e}")
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

        dockerfile_full = DOCKERFILES[lang]
        if not os.path.exists(dockerfile_full):
            raise FileNotFoundError(
                f"Dockerfile not found at {dockerfile_full}. "
                f"Expected docker/ directory at project root."
            )
        _, logs = self.client.images.build(
            path=os.path.dirname(dockerfile_full),
            dockerfile=os.path.basename(dockerfile_full),
            tag=SANDBOX_IMAGES[lang],
            rm=True,
        )
        for chunk in logs:
            if "stream" in chunk:
                logger.debug(chunk["stream"].strip())

        console.print(f"[green]✓ {label} sandbox image built successfully.[/green]")

    def execute_benchmark(self, code: str, language: str = "cpp", timeout_seconds: int = 120) -> Tuple[bool, str, Optional[bytes]]:
        if self.disabled:
            return False, SANDBOX_SKIPPED_MSG, None
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
                    environment={"OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
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
                    
    def verify(
        self,
        csv_output: str,
        original_code: str,
        optimized_code: str,
        original_func_name: str,
        optimized_func_name: str,
        test_cases: List[Dict[str, Any]],
        language: str = "python",
        timeout_seconds: int = 60,
        context: str = "",
    ) -> VerificationResult:
        if self.disabled:
            return VerificationResult(
                speedup=SpeedupVerification(verified=False, details=SANDBOX_SKIPPED_MSG),
                correctness=CorrectnessVerification(verified=False, details=SANDBOX_SKIPPED_MSG),
            )
        speedup_result     = self._verify_speedup(csv_output)
        correctness_result = self._verify_correctness(
            original_code, optimized_code,
            original_func_name, optimized_func_name,
            test_cases, language, timeout_seconds,
            context=context,
        )
        return VerificationResult(speedup=speedup_result, correctness=correctness_result)

    def verify_correctness_only(
        self,
        original_code:       str,
        optimized_code:      str,
        original_func_name:  str,
        optimized_func_name: str,
        test_cases:          List[Dict[str, Any]],
        language:            str = "python",
        timeout_seconds:     int = 60,
        context:             str = "",
    ) -> CorrectnessVerification:
        """
        Re-run correctness sandbox only — no speedup check, no LLM.
        Used by `coreinsight test <function_name>`.

        C++ and CUDA are not supported: their correctness harness is a
        main() block embedded by HarnessAgent at analysis time and cannot
        be reconstructed from stored test cases alone.
        """
        if self.disabled:
            return CorrectnessVerification(verified=False, details=SANDBOX_SKIPPED_MSG)
        if not self.client:
            return CorrectnessVerification(verified=False, details="Docker unavailable.")
        if language in ("cpp", "c++", "cuda"):
            return CorrectnessVerification(
                verified=False,
                details=(
                    f"Re-verification not supported for {language}: "
                    "correctness harness is embedded at analysis time. "
                    "See stored pass rate in `coreinsight memory`."
                ),
            )
        return self._verify_correctness(
            original_code, optimized_code,
            original_func_name, optimized_func_name,
            test_cases, language, timeout_seconds,
            context=context,
        )

    def _verify_speedup(self, csv_output: str) -> SpeedupVerification:
        result = SpeedupVerification(verified=False)
        try:
            csv_header = "N,Original_Time,Optimized_Time,Speedup"
            if csv_header in csv_output:
                csv_output = csv_output[csv_output.find(csv_header):]
            rows = list(csv.DictReader(io.StringIO(csv_output)))
        except Exception as e:
            result.details = f"CSV parse error: {e}"
            return result

        if not rows or not {"Original_Time", "Optimized_Time"}.issubset(rows[0].keys()):
            result.details = "CSV is missing required timing columns."
            return result
        if len(rows) < MIN_TIMING_ROWS:
            result.details = f"Too few timing rows ({len(rows)}); need ≥ {MIN_TIMING_ROWS}."
            return result

        computed_speedups, reported_speedups, orig_times, flags = [], [], [], []
        max_discrepancy = 0.0

        for i, row in enumerate(rows):
            try:
                t_orig = float(row["Original_Time"])
                t_opt  = float(row["Optimized_Time"])
            except ValueError:
                flags.append(f"Row {i}: non-numeric timing values.")
                continue
            if t_orig <= 0 or t_opt <= 0:
                # Small N often rounds to 0.000 at CSV precision — skip row but don't fail
                flags.append(
                    f"Row {i}: timing rounds to zero at this N (orig={t_orig}, opt={t_opt}) — insufficient timer resolution, row skipped."
                )
                continue
            if t_orig == t_opt:
                flags.append(f"Row {i}: original and optimized times are identical — likely a harness copy-paste error.")

            computed = t_orig / t_opt
            computed_speedups.append(computed)
            orig_times.append(t_orig)

            if "Speedup" in row:
                try:
                    reported = float(row["Speedup"])
                    # Only cross-check rows where our computed speedup is
                    # meaningful — skip rows where timer resolution was
                    # insufficient (computed would be based on near-zero times)
                    if t_opt < 1e-6:
                        flags.append(
                            f"Row {i}: optimized time too small to verify reported "
                            f"speedup {reported:.4f}x — harness needs higher-resolution timing."
                        )
                        continue
                    reported_speedups.append(reported)
                    discrepancy = abs(computed - reported) / max(abs(computed), 1e-12)
                    max_discrepancy = max(max_discrepancy, discrepancy)
                    if discrepancy > SPEEDUP_DISCREPANCY_TOLERANCE:
                        flags.append(
                            f"Row {i}: reported speedup {reported:.4f}x ≠ computed {computed:.4f}x "
                            f"(Δ={discrepancy:.2%} > {SPEEDUP_DISCREPANCY_TOLERANCE:.0%} threshold)."
                        )
                except ValueError:
                    flags.append(f"Row {i}: non-numeric Speedup column value.")

        if not computed_speedups:
            result.details = "No valid timing rows after filtering."
            return result

        # Monotonicity: times should grow with N
        if len(orig_times) >= 3:
            inversions = sum(1 for a, b in zip(orig_times, orig_times[1:]) if b < a * 0.5)
            if inversions > len(orig_times) // 3:
                flags.append("Original timing is non-monotone with N — possible fabricated values.")

        # Suspiciously round speedups
        if len(computed_speedups) > 1 and all(abs(s - round(s)) < 1e-6 for s in computed_speedups):
            flags.append("All computed speedups are exact integers — verify timer resolution.")

        # High coefficient of variation
        if len(computed_speedups) > 2:
            mean = sum(computed_speedups) / len(computed_speedups)
            std  = math.sqrt(sum((s - mean) ** 2 for s in computed_speedups) / len(computed_speedups))
            cv = std / mean if mean else 0
            # High CV is expected when algorithmic complexity class changes (e.g. O(N²)→O(N)),
            # since speedup grows with N by design. Only flag if speedups are also non-monotone
            # AND the variance is extreme enough to suggest fabrication rather than complexity gain.
            speedups_monotone = all(a <= b * 1.2 for a, b in zip(computed_speedups, computed_speedups[1:]))
            if cv > 1.0 and not speedups_monotone:
                flags.append(f"High speedup variance (CV={cv:.2f}) with non-monotone speedup — results may be fabricated.")

        result.computed_speedups = computed_speedups
        result.reported_speedups = reported_speedups
        result.max_discrepancy   = max_discrepancy if reported_speedups else None
        result.suspicious_flags  = flags

        critical = [f for f in flags if "≠ computed" in f or "non-numeric" in f]
        result.verified = (len(critical) == 0)
        avg = sum(computed_speedups) / len(computed_speedups)
        result.details = (
            f"Recomputed {len(computed_speedups)} speedup(s). Avg={avg:.3f}x. "
            + (f"Max discrepancy vs reported: {max_discrepancy:.2%}." if reported_speedups else "No Speedup column to cross-check.")
        )
        return result

    def _verify_correctness(
        self,
        original_code: str,
        optimized_code: str,
        original_func_name: str,
        optimized_func_name: str,
        test_cases: List[Dict[str, Any]],
        language: str,
        timeout_seconds: int,
        context: str = "",
    ) -> CorrectnessVerification:
        if not self.client:
            return CorrectnessVerification(verified=False, details="Docker unavailable.")
        if not test_cases:
            return CorrectnessVerification(verified=False, details="No test cases provided — skipping correctness check.")
        if language == "python":
            return self._correctness_python(
                original_code, optimized_code, original_func_name, optimized_func_name,
                test_cases, timeout_seconds, context=context,
            )
        if language in ["cpp", "c++"]:
            return self._correctness_cpp(
                original_code, optimized_code, original_func_name, optimized_func_name,
                test_cases, timeout_seconds, context=context,
            )
        return CorrectnessVerification(verified=False, details=f"Correctness not implemented for '{language}'.")

    def _correctness_python(
        self,
        original_code: str,
        optimized_code: str,
        original_func_name: str,
        optimized_func_name: str,
        test_cases: List[Dict[str, Any]],
        timeout_seconds: int,
        context: str = "",
    ) -> CorrectnessVerification:
        # Prepend RAG context (helper functions / imports the target function
        # depends on) so the verification container has everything it needs.
        # Without this, any cross-function dependency causes a NameError and
        # every test case fails despite the optimization being correct.
        context_block = (
            "# === context (helper functions) ===\n" + context.strip() + "\n\n"
            if context and context.strip()
            else ""
        )
        merged = (
            context_block
            + "# === original ===\n" + original_code.strip()
            + "\n\n# === optimized ===\n" + optimized_code.strip() + "\n"
        )
        harness = _PYTHON_CORRECTNESS_HARNESS.format(
            original_func_name=original_func_name,
            optimized_func_name=optimized_func_name,
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
        )
        temp_dir = tempfile.mkdtemp()
        try:
            os.chmod(temp_dir, 0o777)
            for fname, content in [
                ("source.py",       merged),
                ("harness.py",      harness),
                ("test_cases.json", json.dumps(test_cases)),
            ]:
                p = os.path.join(temp_dir, fname)
                with open(p, "w") as f:
                    f.write(content)
                os.chmod(p, 0o777)

            container = None
            try:
                container = self.client.containers.run(
                    image=SANDBOX_IMAGES["python"],
                    command=["python", "-u", "-B", "-W", "ignore", "/workspace/harness.py"],
                    volumes={temp_dir: {"bind": "/workspace", "mode": "rw"}},
                    working_dir="/workspace",
                    detach=True,
                    mem_limit="1g",
                    network_disabled=True,
                    environment={"OMP_NUM_THREADS": "1"},
                    pids_limit=100,
                    cap_drop=["ALL"],
                )
                exit_code  = container.wait(timeout=timeout_seconds + 5)["StatusCode"]
                raw_output = container.logs(stdout=True, stderr=True).decode("utf-8").strip()
            except Exception as e:
                return CorrectnessVerification(verified=False, details=f"Container error: {e}")
            finally:
                if container:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass

        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        return self._parse_python_correctness_output(raw_output, exit_code, len(test_cases))

    def _parse_python_correctness_output(
        self, raw_output: str, exit_code: int, expected_cases: int
    ) -> CorrectnessVerification:
        result = CorrectnessVerification(verified=False, total_cases=expected_cases)
        json_line = next(
            (l.strip() for l in reversed(raw_output.splitlines()) if l.strip().startswith("[")),
            None,
        )
        if not json_line:
            result.details = f"No JSON output from harness (exit={exit_code}).\n{raw_output[:500]}"
            return result
        try:
            cases = json.loads(json_line)
        except json.JSONDecodeError as e:
            result.details = f"JSON parse error: {e}"
            return result

        passed   = sum(1 for c in cases if c.get("passed"))
        failures = [f"Case {c['case']}: {c.get('detail','')}" for c in cases if not c.get("passed")]
        result.total_cases  = len(cases)
        result.passed_cases = passed
        result.failures     = failures
        result.verified     = (passed == len(cases) and len(cases) > 0)
        result.details      = f"{passed}/{len(cases)} test cases passed."
        return result

    def _correctness_cpp(
        self,
        original_code: str,
        optimized_code: str,
        original_func_name: str,
        optimized_func_name: str,
        test_cases: List[Dict[str, Any]],
        timeout_seconds: int,
        context: str = "",
    ) -> CorrectnessVerification:
        # Same rationale as Python path — C++ functions frequently depend on
        # structs or helpers defined elsewhere in the file.
        # RAG context may contain C++-style metadata headers ("// From file.py")
        # which are syntax errors in Python. Rewrite them as # comments.
        import re as _re
        safe_context = _re.sub(r"^//", "#", context.strip(), flags=_re.MULTILINE)
        context_block = (
            "# === context (helper functions) ===\n" + safe_context + "\n\n"
            if safe_context
            else ""
        )
        merged = context_block + optimized_code.strip()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chmod(temp_dir, 0o777)
            for fname, content in [
                ("source.cpp",      merged),
                ("test_cases.json", json.dumps(test_cases)),
            ]:
                p = os.path.join(temp_dir, fname)
                with open(p, "w") as f:
                    f.write(content)
                os.chmod(p, 0o777)

            container = None
            try:
                container = self.client.containers.run(
                    image=SANDBOX_IMAGES["cpp"],
                    command=["/bin/sh", "-c",
                        f"timeout {timeout_seconds}s "
                        f"g++ -O0 -DCOREINSIGHT_CORRECTNESS -std=c++17 source.cpp -o check && ./check"
                    ],
                    volumes={temp_dir: {"bind": "/workspace", "mode": "rw"}},
                    working_dir="/workspace",
                    detach=True,
                    mem_limit="1g",
                    network_disabled=True,
                    environment={"OMP_NUM_THREADS": "1"},
                    pids_limit=100,
                    cap_drop=["ALL"],
                )
                exit_code  = container.wait(timeout=timeout_seconds + 5)["StatusCode"]
                raw_output = container.logs(stdout=True, stderr=True).decode("utf-8")
            except Exception as e:
                return CorrectnessVerification(verified=False, details=f"Container error: {e}")
            finally:
                if container:
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass

        return self._parse_cpp_correctness_output(raw_output, exit_code, len(test_cases))

    def _parse_cpp_correctness_output(
        self, raw_output: str, exit_code: int, expected_cases: int
    ) -> CorrectnessVerification:
        result   = CorrectnessVerification(verified=False, total_cases=expected_cases)
        passed, failures, parsed = 0, [], 0
        for line in raw_output.splitlines():
            parts = line.strip().split(None, 3)
            if not parts or parts[0] != "CASE" or len(parts) < 3:
                continue
            parsed += 1
            if parts[2].upper() == "PASS":
                passed += 1
            else:
                failures.append(f"Case {parts[1]}: {parts[3] if len(parts) > 3 else ''}")
        if parsed == 0:
            result.details = f"No CASE lines in output (exit={exit_code}).\n{raw_output[:500]}"
            return result
        result.total_cases  = parsed
        result.passed_cases = passed
        result.failures     = failures
        result.verified     = (passed == parsed)
        result.details      = f"{passed}/{parsed} test cases passed."
        return result