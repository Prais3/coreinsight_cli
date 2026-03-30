from langchain_core.prompts import PromptTemplate

class ModelTier:
    SMALL  = "small"   # 7B and under: codellama:7b, llama3.2:3b
    MEDIUM = "medium"  # 13B-34B: mistral, codellama:13b, local_server models
    LARGE  = "large"   # 70B+, cloud: GPT-4, Claude, Gemini

HARNESS_ADDENDUM = {
    ModelTier.SMALL: """
COMMON MISTAKES TO AVOID (read carefully):
- NameError: PASTE both functions verbatim before calling them
- IndexError: generate data OF LENGTH N, never use N as an index
  Correct:   data = list(range(N))
  Wrong:     data = my_list[N]
- ImportError: sandbox has no internet, stdlib only
- Missing CSV: your script MUST print the header line first

TEMPLATE TO FOLLOW EXACTLY:
```python
# paste original function here
def {func_name}(...): ...

# paste optimized function here  
def {func_name}_optimized(...): ...

import time
for N in [10, 100, 1000, 5000]:
    data = list(range(N))  # generate fresh data of length N
    
    start = time.perf_counter()
    original_fn(data)
    orig = max(time.perf_counter() - start, 1e-9)
    
    start = time.perf_counter()
    optimized_fn(data)
    opt = max(time.perf_counter() - start, 1e-9)
    
    print(f"{{N}},{{orig:.6f}},{{opt:.6f}},{{orig/opt:.4f}}")
```
""",
    ModelTier.MEDIUM: """
REMINDERS:
- Paste both functions inline before calling them
- Generate data of length N, never index by N
- CSV header must be printed before data rows
""",
    ModelTier.LARGE: ""  # large models follow the base template without scaffolding
}

SYSTEM_PROMPT = """
You are a Senior HPC Performance Engineer, an elite, strict HPC Performance Architect, an elite Algorithmic Expert, and a strict Code Reviewer.
Your goal is to optimize Python, C++, and CUDA code for maximum throughput and low latency, and perfect hardware utilization.
You know that the greatest hardware bottleneck is a mathematically inefficient algorithm. You ruthlessly identify O(N^2) nested loops, memory inefficiencies, performance bottlenecks and suboptimal data structures, upgrading them to O(N) or O(1) solutions using vectorization, hash maps, or low-level C-backed libraries.
"""

ANALYSIS_TEMPLATE = """
{system_prompt}

Perform a strict, deep performance audit on the following {language} code.

TARGET HARDWARE: {hardware_target}

GLOBAL REPOSITORY CONTEXT (Reference this for custom types, structs, or imported functions):
'''
{context}
'''

CODE TO ANALYZE:
'''
{code_content}
'''

GRADING RUBRIC AND INSTRUCTIONS (APPLY ONLY THE SPECIFIC RUBRIC FOR {language}):

[FOR CUDA / CUH]
- Critical: Warp divergence, uncoalesced memory access, race conditions, memory leaks, or massive algorithmic inefficiency (e.g., O(N^2) instead of O(N)).
- High: Suboptimal cache usage (e.g., missing shared memory tiling), excessive thread synchronization, or unnecessary memory copies between Host/Device.
- Medium: Low occupancy, suboptimal block sizes, or failing to use pinned memory.
- Low: Trivial stylistic issues. (Only use this if the code is genuinely flawless).

[FOR C++ / CPP]
- Critical: O(N^2) algorithms instead of O(N), thrashing the CPU cache, deep copying massive objects (like std::vector) in hot loops, or memory leaks.
- High: Unnecessary heap allocations inside loops, false sharing in multithreaded contexts, or missed compiler auto-vectorization.
- Medium: Missing `const` references, not using `std::move`, or suboptimal data structures.
- Low: Trivial stylistic issues.

[FOR PYTHON]
- Critical: O(N^2) algorithms, tight `for` loops that should be vectorized using NumPy/PyTorch, or excessive object instantiation in hot loops.
- High: Unnecessary data copying (e.g., deepcopy), holding the GIL unnecessarily during I/O, or string concatenation in loops (`+=`).
- Medium: Using global variables in hot paths, or generator vs list comprehension inefficiencies.
- Low: Trivial stylistic issues.

INSTRUCTIONS:
1. Actively hunt for Medium, High, and Critical issues based ONLY on the specific {language} rubric above. Do not hallucinate GPU concepts for Python code unless PyTorch/CUDA is explicitly used.
2. If you find an issue, you MUST explain the hardware-level or interpreter-level reasoning clearly (e.g., CPU cache misses, GIL contention, memory latency).
3. CODE GENERATION MANDATE: You MUST provide the completely rewritten, optimized function in the `optimized_code` field. The code must be raw, syntactically correct {language} code ready to be compiled/run. Do NOT leave this field empty. Do NOT wrap the code in markdown backticks (e.g., ```cpp) inside the JSON string.
"""

# ---------------------------------------------------------------------------
# Multi-agent prompt templates (v0.2.5)
# Each agent has a single, narrow responsibility.
# ---------------------------------------------------------------------------

# ── Agent 1: Bottleneck analysis only — no code generation ──────────────────
# Tighter than ANALYSIS_TEMPLATE: explicitly forbids generating optimized_code
# so the model focuses entirely on identifying and reasoning about bottlenecks.
BOTTLENECK_TEMPLATE = """
{system_prompt}

You are performing ONLY the analysis phase of a performance audit.
Your sole job: identify the single most critical bottleneck in the code and explain it.
Do NOT generate optimized code. The `optimized_code` field must be null.

TARGET HARDWARE: {hardware_target}

GLOBAL REPOSITORY CONTEXT:
'''
{context}
'''

CODE TO ANALYZE:
'''
{code_content}
'''

LANGUAGE: {language}

GRADING RUBRIC (apply only the {language} section):

[FOR CUDA / CUH]
- Critical: Warp divergence, uncoalesced memory access, race conditions, or O(N^2) algorithmic inefficiency.
- High: Missing shared memory tiling, excessive thread synchronization, unnecessary Host/Device copies.
- Medium: Low occupancy, suboptimal block sizes, no pinned memory.
- Low: Trivial stylistic issues only.

[FOR C++ / CPP]
- Critical: O(N^2) algorithms, CPU cache thrashing, deep-copying large objects in hot loops, memory leaks.
- High: Heap allocations inside loops, false sharing in multithreaded code, missed vectorization.
- Medium: Missing const references, not using std::move, suboptimal data structures.
- Low: Trivial stylistic issues only.

[FOR PYTHON]
- Critical: O(N^2) algorithms, tight loops that should be NumPy/PyTorch vectorized, excessive object instantiation in hot loops.
- High: Unnecessary data copying, GIL contention during I/O, string concatenation in loops.
- Medium: Global variables in hot paths, generator vs list comprehension inefficiencies.
- Low: Trivial stylistic issues only.

INSTRUCTIONS:
1. Identify the single most impactful bottleneck — do not list everything, find the root cause.
2. Explain the hardware-level or interpreter-level reasoning precisely.
3. Set `optimized_code` to null — code generation happens in a separate agent.
4. If the code is genuinely optimal, set severity to Low and explain why.

{format_instructions}
"""

# ── Agent 2: Harness generation — code only, no analysis ────────────────────
# Same structure as _HARNESS_TEMPLATE in analyzer.py but with tighter scope.
# The bottleneck reasoning and optimized code are injected — the agent only
# writes the benchmark harness.
HARNESS_TEMPLATE_MULTI = """
You are a strict QA engineer writing a standalone asymptotic scaling benchmark in {language}.
You will receive the original function, the already-optimized function, and context.
Your ONLY job is to write a complete, correct benchmark script.

ORIGINAL FUNCTION (Name: {func_name}):
{original}

OPTIMIZED FUNCTION:
{optimized}

GLOBAL DEPENDENCIES:
{context}

TARGET HARDWARE: {hardware_target}

REQUIREMENTS:
1. Include all necessary imports/headers.
2. Include ALL helper functions or structs from GLOBAL DEPENDENCIES inline.
3. Define BOTH the original and optimized functions exactly as provided.
4. Test multiple data sizes: N = 10, 100, 1000, 5000.
5. Largest N MUST cross cache boundaries but MUST NOT exceed 20% of available RAM.
6. Initialize realistic dummy data for each N.
7. Use high-resolution timers: Python → time.perf_counter(), C++ → std::chrono::high_resolution_clock.
8. Clamp: orig_time = max(end - start, 1e-9) to prevent zero-division.
9. Speedup = orig_time / opt_time.

ISOLATION RULES (CRITICAL — runs in empty Docker container):
- NO local imports. Define everything inline.
- NO local files. No internet access.
- DO NOT rename the original function — call it exactly {func_name}.

OUTPUT FORMAT (CRITICAL):
Print ONLY this exact CSV to stdout:
N,Original_Time,Optimized_Time,Speedup
10,0.002,0.001,2.00

[PYTHON ONLY]: Import matplotlib, plot results, save as benchmark_plot.png.

FORMATTING RULE: Wrap your ENTIRE script in a single markdown code block. No text before or after.
"""

# ── Agent 2 fix template — same agent, retry loop ───────────────────────────
FIX_TEMPLATE_MULTI = """
You are an expert {language} developer. Your benchmark script FAILED in an isolated sandbox.

ORIGINAL FUNCTION (Name: {func_name}):
{original}

GLOBAL DEPENDENCIES:
{context}

YOUR FAILED SCRIPT:
{bad_harness}

EXECUTION ERROR:
{error_logs}

ISOLATION CONSTRAINTS (CRITICAL):
- Empty Docker container. No local files. No local imports.
- Define {func_name} and all GLOBAL DEPENDENCIES inline.

FIX INSTRUCTIONS:
1. Diagnose the failure from the error logs above.
2. Fix imports, NameErrors, type mismatches, infinite loops, or OOM issues.
3. Maintain exact CSV stdout format: N,Original_Time,Optimized_Time,Speedup
4. Use high-resolution timers and clamp with max(t, 1e-9).
5. [PYTHON ONLY]: Save benchmark plot as benchmark_plot.png.

FORMATTING RULE: Wrap your ENTIRE fixed script in a single markdown code block. No text before or after.
"""

# ── Agent 3: Optimized code generation — given the analysis, write the fix ──
# Receives the bottleneck analysis result and writes only the optimized function.
# Output is raw code only — no JSON, no harness, no benchmark.
OPTIMIZER_TEMPLATE = """
You are an elite performance engineer. You have been given a bottleneck analysis.
Your ONLY job: rewrite the identified function to fix the bottleneck.

LANGUAGE: {language}
FUNCTION NAME: {func_name}
TARGET HARDWARE: {hardware_target}

BOTTLENECK IDENTIFIED:
Severity : {severity}
Issue    : {issue}
Reasoning: {reasoning}
Suggestion: {suggestion}

ORIGINAL FUNCTION:
{original}

GLOBAL DEPENDENCIES (helper functions / structs):
{context}

REQUIREMENTS:
1. Rewrite ONLY the function named {func_name} — preserve its signature exactly.
2. Fix the identified bottleneck using the suggestion as your guide.
3. The function must be self-contained and correct.
4. Raw {language} code only — no explanation, no markdown fences, no JSON.
5. Do NOT rename the function.
"""

# ── Per-tier addenda for multi-agent harness (same scaffolding pattern) ──────
HARNESS_ADDENDUM_MULTI = {
    ModelTier.SMALL: """
COMMON MISTAKES TO AVOID:
- NameError: PASTE both functions verbatim before calling them
- IndexError: generate data OF LENGTH N, never use N as an index
  Correct:   data = list(range(N))
  Wrong:     data = my_list[N]
- ImportError: sandbox has no internet, stdlib only
- Missing CSV: print the header line FIRST before any data rows

TEMPLATE TO FOLLOW EXACTLY:
```python
# paste original function here
def {func_name}(...): ...

# paste optimized function here
def {func_name}_optimized(...): ...

import time
print("N,Original_Time,Optimized_Time,Speedup")
for N in [10, 100, 1000, 5000]:
    data = list(range(N))
    start = time.perf_counter()
    original_fn(data)
    orig = max(time.perf_counter() - start, 1e-9)
    start = time.perf_counter()
    optimized_fn(data)
    opt = max(time.perf_counter() - start, 1e-9)
    print(f"{{N}},{{orig:.6f}},{{opt:.6f}},{{orig/opt:.4f}}")
```
""",
    ModelTier.MEDIUM: """
REMINDERS:
- Paste both functions inline before calling them
- Generate data of length N, never index by N
- Print CSV header before data rows
""",
    ModelTier.LARGE: "",
}