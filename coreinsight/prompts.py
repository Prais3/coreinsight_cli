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