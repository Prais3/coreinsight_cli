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
                return self._profile_cpp(detected)
            if language in ("cuda", "cu", "cuh"):
                return self._profile_cuda(detected)
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
    # C++ path (v0.2.0: valgrind callgrind)
    # ------------------------------------------------------------------ #

    def _profile_cpp(self, detected: Dict[str, bool]) -> ProfilerResult:
        found = [k for k, v in detected.items() if v]
        note  = f"Detected: {', '.join(found)}." if found else "No C++ profiling tools found."
        return ProfilerResult(
            available=False, tool="none", language="cpp",
            error=f"{note} Full C++ profiling (perf stat + valgrind) coming in v0.2.0.",
        )

    # ------------------------------------------------------------------ #
    # CUDA path (v0.2.0: nsys / nvprof)
    # ------------------------------------------------------------------ #

    def _profile_cuda(self, detected: Dict[str, bool]) -> ProfilerResult:
        if detected.get("nsys"):
            note = "nsys detected."
        elif detected.get("nvprof"):
            note = "nvprof detected."
        else:
            note = "No CUDA profiling tools found (install nsys from CUDA Toolkit)."
        return ProfilerResult(
            available=False, tool="none", language="cuda",
            error=f"{note} CUDA profiling coming in v0.2.0.",
        )