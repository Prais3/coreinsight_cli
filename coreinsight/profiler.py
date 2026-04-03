"""
AI-free hardware profiling layer (addition in v0.1.6)

Instruments both the original and optimised function with real profiling
tools and returns structured, LLM-independent timing and hardware-counter
data.  NO LLM is involved at any stage of this module.

Tool priority:
  Python : cProfile + timeit (Docker, always) → perf stat (host, Linux only)
  C++    : perf stat (host, Linux only)        [v0.2.0: valgrind callgrind]
  CUDA   : nsys / nvprof detection             [v0.2.0: full implementation]
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProfilerMetric:
    name:      str
    original:  str
    optimized: str
    delta:     str          # e.g. "-73.4%"  (always a string, pre-formatted)
    unit:      str = ""
    note:      str = ""     # e.g. "lower is better"


@dataclass
class ProfilerResult:
    available:         bool           # False → profiler skipped or fully errored
    tool:              str            # "cProfile", "perf stat + cProfile", …
    language:          str
    metrics:           List[ProfilerMetric] = field(default_factory=list)
    raw_original:      str = ""       # cProfile text dump for original fn
    raw_optimized:     str = ""       # cProfile text dump for optimized fn
    host_tool_name:    str = ""       # "perf stat", "nsys", …
    host_tool_metrics: List[ProfilerMetric] = field(default_factory=list)
    error:             str = ""       # non-fatal — surfaced in report as a note


# ---------------------------------------------------------------------------
# Docker harness templates
# ---------------------------------------------------------------------------

# Runs inside the existing coreinsight-python-sandbox Docker image.
# cProfile is part of the Python stdlib — no image changes required.
_PYTHON_CPROFILE_HARNESS = '''\
import cProfile, pstats, io, json, time, sys, importlib.util, traceback

def _load(path, fn_name):
    spec = importlib.util.spec_from_file_location("_m_" + fn_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name, None)

orig_fn = _load("/workspace/original.py", "{func_name}")
opt_fn  = _load("/workspace/optimized.py",  "{func_name}")

if orig_fn is None or opt_fn is None:
    print(json.dumps({{"error": "function '{func_name}' not found in one or both source files"}}))
    sys.exit(1)

with open("/workspace/profile_input.json") as _fh:
    _inp = json.load(_fh)
_args   = _inp.get("args",   [])
_kwargs = _inp.get("kwargs", {{}})
N_ITER  = {n_iter}

output = {{}}
for _label, _fn in [("original", orig_fn), ("optimized", opt_fn)]:
    # Warmup pass (not measured)
    try:
        _fn(*_args, **_kwargs)
    except Exception:
        pass

    # cProfile pass
    _pr = cProfile.Profile()
    _pr.enable()
    try:
        for _ in range(N_ITER):
            _fn(*_args, **_kwargs)
    except Exception:
        output[_label] = {{"error": traceback.format_exc(limit=3)}}
        _pr.disable()
        continue
    _pr.disable()

    _s = io.StringIO()
    pstats.Stats(_pr, stream=_s).sort_stats("cumulative").print_stats(10)

    # Separate wall-clock pass (cProfile itself adds overhead)
    _t0 = time.perf_counter()
    for _ in range(N_ITER):
        _fn(*_args, **_kwargs)
    _elapsed = (time.perf_counter() - _t0) / N_ITER

    _stats_obj = pstats.Stats(_pr)
    output[_label] = {{
        "total_calls": _stats_obj.total_calls,
        "avg_wall_s":  _elapsed,
        "profile_top": _s.getvalue(),
    }}

print(json.dumps(output))
'''.strip()

# Minimal driver written to a temp file for host-level `perf stat` runs.
_PYTHON_PERF_DRIVER = '''\
import importlib.util, json
spec = importlib.util.spec_from_file_location("_tgt", "{source_path}")
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = getattr(mod, "{func_name}")
with open("{input_path}") as f:
    inp = json.load(f)
args, kwargs = inp.get("args", []), inp.get("kwargs", {{}})
for _ in range({n_iter}):
    fn(*args, **kwargs)
'''.strip()


# Minimal C++ wrapper — compiles the full source file and runs it under perf stat.
# The source file must be compilable as a standalone program (has a main or we inject one).
_CPP_PERF_COMPILE_CMD = "g++ -O3 -std=c++17 {src} -o {out}"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct_delta(orig: float, opt: float) -> str:
    """Return a pre-formatted percentage delta string, e.g. '-73.4%'."""
    if orig == 0:
        return "N/A"
    delta = (opt - orig) / orig * 100.0
    sign  = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}%"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _parse_nsys_stats(output: str) -> Dict[str, Any]:
    """
    Parse `nsys profile --stats=true` stdout into structured metrics.
    Extracts kernel timing and memory throughput from the summary tables.
    """
    result: Dict[str, Any] = {}

    # ── Kernel statistics ────────────────────────────────────────────────
    # Header: Time(%)  Total Time (ns)  Instances  Avg (ns)  ...  Name
    kernel_section = False
    kernels = []
    for line in output.splitlines():
        if "CUDA Kernel Statistics" in line or "GPU Kernel Summary" in line:
            kernel_section = True
            continue
        if kernel_section:
            if line.strip() == "" or line.startswith("="):
                if kernels:
                    kernel_section = False
                continue
            # Skip header/separator lines
            if "Time(%)" in line or "----" in line:
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    kernels.append({
                        "pct":       float(parts[0]),
                        "total_ns":  float(parts[1].replace(",", "")),
                        "instances": int(parts[2].replace(",", "")),
                        "avg_ns":    float(parts[3].replace(",", "")),
                        "name":      " ".join(parts[7:]) if len(parts) > 7 else parts[-1],
                    })
                except (ValueError, IndexError):
                    continue

    if kernels:
        # Top kernel by total time
        top = max(kernels, key=lambda k: k["total_ns"])
        result["top_kernel_name"]     = top["name"]
        result["top_kernel_avg_ns"]   = top["avg_ns"]
        result["top_kernel_total_ns"] = top["total_ns"]
        result["top_kernel_instances"]= top["instances"]
        result["total_kernel_ns"]     = sum(k["total_ns"] for k in kernels)

    # ── Memory throughput ────────────────────────────────────────────────
    # Look for "Memory Throughput" or HtoD/DtoH transfer lines
    mem_section = False
    total_mem_ns = 0.0
    for line in output.splitlines():
        if "Memory Operation" in line or "Memory Throughput" in line:
            mem_section = True
            continue
        if mem_section:
            if line.strip() == "" or line.startswith("="):
                mem_section = False
                continue
            if "Time(%)" in line or "----" in line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    total_mem_ns += float(parts[1].replace(",", ""))
                except (ValueError, IndexError):
                    continue

    if total_mem_ns:
        result["total_mem_transfer_ns"] = total_mem_ns

    return result


def _parse_perf_stat(stderr: str) -> Dict[str, float]:
    """Extract hardware counter values from `perf stat` stderr output."""
    targets = {
        "cache-misses":     "cache_misses",
        "cache-references": "cache_refs",
        "instructions":     "instructions",
        "cycles":           "cycles",
        "branch-misses":    "branch_misses",
    }
    result: Dict[str, float] = {}
    for line in stderr.splitlines():
        for raw_key, attr in targets.items():
            if raw_key in line:
                m = re.match(r"\s*([\d,]+)\s+" + re.escape(raw_key), line)
                if m:
                    result[attr] = float(m.group(1).replace(",", ""))
    return result


def _detect_host_tools(language: str) -> Dict[str, bool]:
    """
    Check which profiling binaries are present on the host PATH.
    `perf` is Linux-only; all others are cross-platform in principle.
    """
    on_linux = platform.system() == "Linux"
    return {
        "perf":    on_linux and shutil.which("perf") is not None,
        "py_spy":  language == "python" and shutil.which("py-spy") is not None,
        "valgrind":language in ("cpp", "c++") and shutil.which("valgrind") is not None,
        "nsys":    language in ("cuda", "cu", "cuh") and shutil.which("nsys") is not None,
        "nvprof":  language in ("cuda", "cu", "cuh") and shutil.which("nvprof") is not None,
    }


def _build_cprofile_metrics(
    orig: Dict[str, Any],
    opt:  Dict[str, Any],
) -> List[ProfilerMetric]:
    metrics: List[ProfilerMetric] = []

    ow  = orig.get("avg_wall_s", 0.0)
    opw = opt.get("avg_wall_s",  0.0)
    if ow and opw:
        metrics.append(ProfilerMetric(
            name="Avg wall time",
            original=f"{ow  * 1000:.3f} ms",
            optimized=f"{opw * 1000:.3f} ms",
            delta=_pct_delta(ow, opw),
            note="lower is better",
        ))

    oc  = orig.get("total_calls", 0)
    opc = opt.get("total_calls",  0)
    if oc and opc:
        metrics.append(ProfilerMetric(
            name="Total fn calls",
            original=_fmt_int(oc),
            optimized=_fmt_int(opc),
            delta=_pct_delta(float(oc), float(opc)),
            note="lower is better",
        ))

    return metrics


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HardwareProfiler:
    """
    AI-free profiling layer.

    Usage (called once per run_analysis, shared across threads):
        profiler = HardwareProfiler()
        result   = profiler.profile(original_code, optimized_code, ...)
    """

    def __init__(self) -> None:
        self._docker_client = None
        try:
            import docker
            client = docker.from_env()
            client.ping()
            self._docker_client = client
        except Exception:
            logger.debug("HardwareProfiler: Docker unavailable — cProfile path disabled.")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def profile(
        self,
        original_code:         str,
        optimized_code:        str,
        func_name:             str,
        language:              str,
        test_cases:            Optional[List[Dict[str, Any]]] = None,
        n_iter:                int = 50,
        original_file_content: str = "",
        source_dir:            str = "",
    ) -> ProfilerResult:
        """
        Run all available profiling tools.  Never raises — errors are
        captured in ProfilerResult.error and surfaced in the report.
        """
        test_input = (test_cases[0] if test_cases else None) or {}
        detected   = _detect_host_tools(language)

        try:
            if language == "python":
                return self._profile_python(
                    original_code, optimized_code, func_name,
                    test_input, detected, n_iter,
                    original_file_content=original_file_content,
                    source_dir=source_dir,
                )
            if language in ("cpp", "c++"):
                return self._profile_cpp(
                    detected,
                    original_code=original_code,
                    optimized_code=optimized_code,
                    func_name=func_name,
                    original_file_content=original_file_content,
                    source_dir=source_dir,
                )
            if language in ("cuda", "cu", "cuh"):
                return self._profile_cuda(
                    detected,
                    original_code=original_code,
                    optimized_code=optimized_code,
                    func_name=func_name,
                    original_file_content=original_file_content,
                    source_dir=source_dir,
                )
        except Exception as exc:
            logger.debug(f"HardwareProfiler.profile exception: {exc}", exc_info=True)
            return ProfilerResult(
                available=False, tool="unknown", language=language,
                error=f"Unexpected profiler error: {exc}",
            )

        return ProfilerResult(
            available=False, tool="none", language=language,
            error=f"No profiler implemented for language '{language}'.",
        )

    # ------------------------------------------------------------------ #
    # Python path
    # ------------------------------------------------------------------ #

    def _profile_python(
        self,
        original_code:  str,
        optimized_code: str,
        func_name:      str,
        test_input:     Dict[str, Any],
        detected:       Dict[str, bool],
        n_iter:         int,
        original_file_content: str = "",
        source_dir:            str = "",
    ) -> ProfilerResult:
        result = ProfilerResult(available=False, tool="cProfile", language="python")

        # ── Primary: cProfile inside Docker ─────────────────────────────
        if self._docker_client:
            raw = self._cprofile_in_docker(
                original_code, optimized_code, func_name, test_input, n_iter,
                original_file_content=original_file_content,
                source_dir=source_dir,
            )
            err = raw.get("error", "")
            orig_data = raw.get("original", {})
            opt_data  = raw.get("optimized", {})

            if err:
                result.error = err
            elif isinstance(orig_data, dict) and isinstance(opt_data, dict) \
                    and "error" not in orig_data and "error" not in opt_data:
                result.available     = True
                result.raw_original  = orig_data.get("profile_top", "")
                result.raw_optimized = opt_data.get("profile_top", "")
                result.metrics       = _build_cprofile_metrics(orig_data, opt_data)
            else:
                sub_err = orig_data.get("error", "") or opt_data.get("error", "")
                result.error = f"cProfile harness error: {sub_err}"
        else:
            result.error = "Docker unavailable — cProfile skipped."

        # ── Supplementary: perf stat on host (Linux only) ───────────────
        if detected["perf"]:
            host_metrics = self._run_perf_stat_python(
                original_code, optimized_code, func_name, test_input, n_iter
            )
            if host_metrics:
                result.host_tool_name    = "perf stat"
                result.host_tool_metrics = host_metrics
                # Upgrade tool label to reflect both sources
                if result.available:
                    result.tool = "cProfile + perf stat"

        return result

    def _cprofile_in_docker(
        self,
        original_code:  str,
        optimized_code: str,
        func_name:      str,
        test_input:     Dict[str, Any],
        n_iter:         int,
        original_file_content: str = "",
        source_dir:            str = "",
    ) -> Dict[str, Any]:
        from coreinsight.sandbox import SANDBOX_IMAGES

        harness = _PYTHON_CPROFILE_HARNESS.format(
            func_name=func_name,
            n_iter=n_iter,
        )

        # Use the full file so helper functions / imports are available.
        # For optimized.py: append the optimized function — Python's last
        # definition wins, so it shadows the original cleanly.
        orig_file = original_file_content.strip() if original_file_content else original_code
        opt_file  = (
            orig_file + "\n\n# --- optimized replacement ---\n" + optimized_code.strip()
            if original_file_content
            else optimized_code
        )

        temp_dir = tempfile.mkdtemp()
        container = None
        try:
            os.chmod(temp_dir, 0o777)

            # Copy sibling .py files so cross-file imports resolve inside Docker.
            # original.py and optimized.py are written explicitly below and will
            # overwrite any copy of the analysed file that lands here.
            if source_dir and os.path.isdir(source_dir):
                import glob
                for sibling in glob.glob(os.path.join(source_dir, "*.py")):
                    dest = os.path.join(temp_dir, os.path.basename(sibling))
                    try:
                        shutil.copy2(sibling, dest)
                        os.chmod(dest, 0o777)
                    except Exception:
                        pass

            for fname, content in [
                ("original.py",          orig_file),
                ("optimized.py",         opt_file),
                ("profiler_harness.py",  harness),
                ("profile_input.json",   json.dumps(test_input)),
            ]:
                p = os.path.join(temp_dir, fname)
                with open(p, "w") as fh:
                    fh.write(content)
                os.chmod(p, 0o777)

            container = self._docker_client.containers.run(
                image=SANDBOX_IMAGES["python"],
                command=["python", "-u", "-B", "-W", "ignore",
                         "/workspace/profiler_harness.py"],
                volumes={temp_dir: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                detach=True,
                mem_limit="1g",
                network_disabled=True,
                environment={"OMP_NUM_THREADS": "1"},
                pids_limit=100,
                cap_drop=["ALL"],
            )
            container.wait(timeout=120)
            raw_output = container.logs(stdout=True, stderr=True).decode("utf-8").strip()
        except Exception as exc:
            return {"error": f"Container error: {exc}"}
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            shutil.rmtree(temp_dir, ignore_errors=True)

        # The harness always prints a single JSON object as its last line
        json_line = next(
            (ln.strip() for ln in reversed(raw_output.splitlines())
             if ln.strip().startswith("{")),
            None,
        )
        if not json_line:
            return {"error": f"No JSON from profiler harness.\n{raw_output[:300]}"}
        try:
            return json.loads(json_line)
        except json.JSONDecodeError as exc:
            return {"error": f"JSON parse error: {exc}\n{raw_output[:300]}"}

    def _run_perf_stat_python(
        self,
        original_code:  str,
        optimized_code: str,
        func_name:      str,
        test_input:     Dict[str, Any],
        n_iter:         int,
    ) -> Optional[List[ProfilerMetric]]:
        """
        Run `perf stat` on the host for both original and optimised code.
        Returns None on any failure (permission denied, tool absent, timeout).
        This runs the code directly on the host — only use for developer-owned code.
        """
        events = "cache-misses,cache-references,instructions,cycles,branch-misses"
        counters_per_label: Dict[str, Dict[str, float]] = {}

        for label, code in [("original", original_code), ("optimized", optimized_code)]:
            tmp = tempfile.mkdtemp()
            try:
                src_path    = os.path.join(tmp, "source.py")
                input_path  = os.path.join(tmp, "input.json")
                driver_path = os.path.join(tmp, "driver.py")

                with open(src_path,    "w") as fh: fh.write(code)
                with open(input_path,  "w") as fh: json.dump(test_input, fh)
                with open(driver_path, "w") as fh:
                    fh.write(_PYTHON_PERF_DRIVER.format(
                        source_path=src_path,
                        func_name=func_name,
                        input_path=input_path,
                        n_iter=n_iter,
                    ))

                proc = subprocess.run(
                    ["perf", "stat", "-e", events, sys.executable, driver_path],
                    capture_output=True, text=True, timeout=60,
                )
                parsed = _parse_perf_stat(proc.stderr)
                if not parsed:
                    logger.debug(f"perf stat: no counters parsed for {label}. "
                                 f"stderr: {proc.stderr[:200]}")
                    return None
                counters_per_label[label] = parsed

            except subprocess.TimeoutExpired:
                logger.debug("perf stat timed out.")
                return None
            except PermissionError:
                logger.debug("perf stat: permission denied "
                             "(hint: sudo sysctl kernel.perf_event_paranoid=1).")
                return None
            except Exception as exc:
                logger.debug(f"perf stat failed: {exc}")
                return None
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        if len(counters_per_label) < 2:
            return None

        orig_c = counters_per_label["original"]
        opt_c  = counters_per_label["optimized"]

        display_map = [
            ("cache_misses",  "Cache misses",  "lower is better"),
            ("cache_refs",    "Cache refs",    ""),
            ("instructions",  "Instructions",  "lower is better"),
            ("cycles",        "CPU cycles",    "lower is better"),
            ("branch_misses", "Branch misses", "lower is better"),
        ]
        metrics: List[ProfilerMetric] = []
        for key, display, note in display_map:
            if key in orig_c and key in opt_c:
                ov, opv = orig_c[key], opt_c[key]
                metrics.append(ProfilerMetric(
                    name=display,
                    original=f"{ov:,.0f}",
                    optimized=f"{opv:,.0f}",
                    delta=_pct_delta(ov, opv),
                    note=note,
                ))
        return metrics or None

    # ------------------------------------------------------------------ #
    # C++ path — perf stat on host (v0.2.1)
    # ------------------------------------------------------------------ #

    def _profile_cpp(
        self,
        detected:              Dict[str, bool],
        original_code:         str = "",
        optimized_code:        str = "",
        func_name:             str = "",
        original_file_content: str = "",
        source_dir:            str = "",
        n_iter:                int = 1,
    ) -> ProfilerResult:
        result = ProfilerResult(available=False, tool="perf stat", language="cpp")

        has_perf = detected.get("perf", False)
        has_gpp  = shutil.which("g++") is not None

        if not has_perf:
            result.error = (
                "perf not found on host (Linux only). "
                "Install via: sudo apt install linux-perf"
            )
            return result
        if not has_gpp:
            result.error = "g++ not found on host — required to compile for perf stat."
            return result
        if not original_file_content:
            result.error = "No source file content available for C++ profiling."
            return result

        host_metrics = self._run_perf_stat_cpp(
            original_file_content, optimized_code, func_name, n_iter
        )
        if host_metrics is None:
            result.error = (
                "perf stat produced no counters — check kernel.perf_event_paranoid "
                "(sudo sysctl kernel.perf_event_paranoid=1) or compilation errors."
            )
            return result

        result.available        = True
        result.host_tool_name   = "perf stat"
        result.host_tool_metrics = host_metrics
        return result

    def _substitute_cpp_function(
        self, source: str, func_name: str, new_body: str
    ) -> Optional[str]:
        """
        Best-effort substitution of a C++ function by name.
        Finds the first occurrence of `func_name` followed by a parameter list
        and replaces the entire brace-delimited body.  Returns None on failure.
        """
        # Find the function signature line
        sig_pattern = re.compile(
            r"[^\n]*\b" + re.escape(func_name) + r"\s*\([^)]*\)[^\{]*\{",
            re.DOTALL,
        )
        m = sig_pattern.search(source)
        if not m:
            return None

        # Walk forward to find the matching closing brace
        start = m.start()
        brace_start = source.index("{", m.start())
        depth, i = 0, brace_start
        while i < len(source):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            return None

        # Replace everything from the signature start to the closing brace
        replacement = source[start : brace_start + 1] + "\n" + new_body.strip() + "\n}"
        return source[:start] + replacement + source[i + 1:]

    def _run_perf_stat_cpp(
        self,
        original_file_content: str,
        optimized_code:        str,
        func_name:             str,
        n_iter:                int,
    ) -> Optional[List[ProfilerMetric]]:
        events = "cache-misses,cache-references,instructions,cycles,branch-misses"
        counters_per_label: Dict[str, Dict[str, float]] = {}

        # Build optimized source — substitute function if we can, else skip opt run
        opt_source = None
        if optimized_code and func_name:
            opt_source = self._substitute_cpp_function(
                original_file_content, func_name, optimized_code
            )

        sources = [("original", original_file_content)]
        if opt_source:
            sources.append(("optimized", opt_source))

        tmp = tempfile.mkdtemp()
        try:
            for label, src in sources:
                src_path = os.path.join(tmp, f"{label}.cpp")
                bin_path = os.path.join(tmp, label)
                with open(src_path, "w") as fh:
                    fh.write(src)

                # Compile
                compile_proc = subprocess.run(
                    ["g++", "-O3", "-std=c++17", src_path, "-o", bin_path],
                    capture_output=True, text=True, timeout=60,
                )
                if compile_proc.returncode != 0:
                    logger.debug(
                        f"C++ perf: compile failed for {label}:\n"
                        f"{compile_proc.stderr[:400]}"
                    )
                    return None

                # Run under perf stat
                try:
                    perf_proc = subprocess.run(
                        ["perf", "stat", "-e", events, bin_path],
                        capture_output=True, text=True, timeout=120,
                    )
                    parsed = _parse_perf_stat(perf_proc.stderr)
                    if not parsed:
                        logger.debug(
                            f"C++ perf: no counters for {label}. "
                            f"stderr: {perf_proc.stderr[:200]}"
                        )
                        return None
                    counters_per_label[label] = parsed
                except subprocess.TimeoutExpired:
                    logger.debug("C++ perf stat timed out.")
                    return None
                except PermissionError:
                    logger.debug("C++ perf stat: permission denied.")
                    return None

        except Exception as exc:
            logger.debug(f"C++ perf stat error: {exc}")
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        if "original" not in counters_per_label:
            return None

        orig_c = counters_per_label["original"]
        # If substitution failed we only have original — report as baseline only
        opt_c  = counters_per_label.get("optimized", orig_c)

        display_map = [
            ("cache_misses",  "Cache misses",  "lower is better"),
            ("cache_refs",    "Cache refs",    ""),
            ("instructions",  "Instructions",  "lower is better"),
            ("cycles",        "CPU cycles",    "lower is better"),
            ("branch_misses", "Branch misses", "lower is better"),
        ]
        metrics: List[ProfilerMetric] = []
        for key, display, note in display_map:
            if key in orig_c:
                ov  = orig_c[key]
                opv = opt_c.get(key, ov)
                metrics.append(ProfilerMetric(
                    name=display,
                    original=f"{ov:,.0f}",
                    optimized=f"{opv:,.0f}",
                    delta=_pct_delta(ov, opv) if opv != ov else "—",
                    note=note,
                ))
        return metrics or None

    # ------------------------------------------------------------------ #
    # CUDA path — nsys CLI profiling
    # ------------------------------------------------------------------ #

    def _profile_cuda(
        self,
        detected:              Dict[str, bool],
        original_code:         str = "",
        optimized_code:        str = "",
        func_name:             str = "",
        original_file_content: str = "",
        source_dir:            str = "",
    ) -> ProfilerResult:
        result = ProfilerResult(available=False, tool="nsys", language="cuda")

        if not detected.get("nsys"):
            if detected.get("nvprof"):
                result.error = (
                    "nvprof detected but not yet supported — install nsys "
                    "from CUDA Toolkit 11.0+ for hardware profiling."
                )
            else:
                result.error = (
                    "No CUDA profiling tools found on PATH. "
                    "Install nsys: https://developer.nvidia.com/nsight-systems"
                )
            return result

        if not shutil.which("nvcc"):
            result.error = "nvcc not found — required to compile CUDA sources for profiling."
            return result

        if not original_file_content:
            result.error = "No CUDA source content available for profiling."
            return result

        stats_per_label: Dict[str, Dict[str, Any]] = {}

        # Build optimized source by appending the optimized kernel —
        # last __global__ definition with the same name wins at link time
        # only if we can safely substitute; otherwise skip optimized run.
        sources = [("original", original_file_content)]
        if optimized_code and func_name:
            opt_src = (
                original_file_content.strip()
                + "\n\n// --- CoreInsight optimized replacement ---\n"
                + optimized_code.strip()
            )
            sources.append(("optimized", opt_src))

        tmp = tempfile.mkdtemp()
        try:
            for label, src in sources:
                src_path = os.path.join(tmp, f"{label}.cu")
                bin_path = os.path.join(tmp, label)

                with open(src_path, "w") as fh:
                    fh.write(src)

                # Compile
                compile_proc = subprocess.run(
                    ["nvcc", "-O3", "-arch=native", src_path, "-o", bin_path],
                    capture_output=True, text=True, timeout=120,
                )
                if compile_proc.returncode != 0:
                    # Try without -arch=native (older nvcc versions)
                    compile_proc = subprocess.run(
                        ["nvcc", "-O3", src_path, "-o", bin_path],
                        capture_output=True, text=True, timeout=120,
                    )
                if compile_proc.returncode != 0:
                    logger.debug(
                        f"CUDA compile failed for {label}:\n"
                        f"{compile_proc.stderr[:400]}"
                    )
                    result.error = (
                        f"nvcc compilation failed for {label} version.\n"
                        f"{compile_proc.stderr[:300]}"
                    )
                    return result

                # Profile with nsys
                nsys_out_base = os.path.join(tmp, f"nsys_{label}")
                try:
                    nsys_proc = subprocess.run(
                        [
                            "nsys", "profile",
                            "--stats=true",
                            "--force-overwrite=true",
                            "-o", nsys_out_base,
                            bin_path,
                        ],
                        capture_output=True, text=True, timeout=300,
                    )
                    # nsys writes stats to stdout; combined output in stderr too
                    combined = nsys_proc.stdout + nsys_proc.stderr
                    parsed   = _parse_nsys_stats(combined)

                    if not parsed:
                        logger.debug(
                            f"nsys: no stats parsed for {label}.\n"
                            f"nsys stdout: {nsys_proc.stdout[:300]}\n"
                            f"nsys stderr: {nsys_proc.stderr[:300]}"
                        )
                        result.error = (
                            f"nsys ran but produced no parseable stats for {label}. "
                            f"Ensure the binary launches at least one CUDA kernel."
                        )
                        return result

                    stats_per_label[label] = parsed

                except subprocess.TimeoutExpired:
                    result.error = "nsys profiling timed out (300s)."
                    return result
                except Exception as exc:
                    result.error = f"nsys execution error: {exc}"
                    return result

        except Exception as exc:
            logger.debug(f"CUDA profiling error: {exc}")
            result.error = f"CUDA profiling failed: {exc}"
            return result
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        if "original" not in stats_per_label:
            result.error = "No profiling data collected."
            return result

        orig_s = stats_per_label["original"]
        opt_s  = stats_per_label.get("optimized", orig_s)

        metrics: List[ProfilerMetric] = []

        # ── Kernel timing ─────────────────────────────────────────────
        orig_ns = orig_s.get("top_kernel_avg_ns", 0.0)
        opt_ns  = opt_s.get("top_kernel_avg_ns",  orig_ns)
        if orig_ns:
            metrics.append(ProfilerMetric(
                name=f"Kernel avg time [{orig_s.get('top_kernel_name', 'top kernel')}]",
                original=f"{orig_ns / 1000:.2f} µs",
                optimized=f"{opt_ns / 1000:.2f} µs",
                delta=_pct_delta(orig_ns, opt_ns),
                note="lower is better",
            ))

        orig_total = orig_s.get("total_kernel_ns", 0.0)
        opt_total  = opt_s.get("total_kernel_ns",  orig_total)
        if orig_total:
            metrics.append(ProfilerMetric(
                name="Total kernel time",
                original=f"{orig_total / 1e6:.3f} ms",
                optimized=f"{opt_total / 1e6:.3f} ms",
                delta=_pct_delta(orig_total, opt_total),
                note="lower is better",
            ))

        orig_inst = orig_s.get("top_kernel_instances", 0)
        if orig_inst:
            metrics.append(ProfilerMetric(
                name="Kernel launches",
                original=str(orig_inst),
                optimized=str(opt_s.get("top_kernel_instances", orig_inst)),
                delta="—",
                note="",
            ))

        # ── Memory transfers ──────────────────────────────────────────
        orig_mem = orig_s.get("total_mem_transfer_ns", 0.0)
        opt_mem  = opt_s.get("total_mem_transfer_ns",  orig_mem)
        if orig_mem:
            metrics.append(ProfilerMetric(
                name="Total memory transfer time",
                original=f"{orig_mem / 1e6:.3f} ms",
                optimized=f"{opt_mem / 1e6:.3f} ms",
                delta=_pct_delta(orig_mem, opt_mem),
                note="lower is better",
            ))

        if not metrics:
            result.error = "nsys ran but no timing metrics could be extracted."
            return result

        result.available        = True
        result.host_tool_name   = "nsys"
        result.host_tool_metrics = metrics
        return result