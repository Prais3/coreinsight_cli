"""
coreinsight/prompts/bottleneck.py
Bottleneck analysis prompt — Agent 1 in multi-agent pipeline.
Three variants: SMALL / MEDIUM / LARGE (selected by analyzer.py via BOTTLENECK_TEMPLATES).
"""
from coreinsight.prompts._base import ModelTier

# ── LARGE — full rubric, used for GPT-4, Claude, Gemini ─────────────────────
BOTTLENECK_TEMPLATE_LARGE = """
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
3. CRITICAL: Set `optimized_code` to null. Code generation is handled by a separate agent.
4. SEVERITY BIAS: When uncertain, always choose the higher severity. Only assign Low if you can explicitly prove algorithmic optimality.

{format_instructions}
"""

# ── MEDIUM — trimmed rubric, fewer instructions ──────────────────────────────
BOTTLENECK_TEMPLATE_MEDIUM = """
{system_prompt}

Find the single biggest performance bottleneck in this code. Do NOT generate optimized code.

LANGUAGE: {language}
HARDWARE: {hardware_target}

CONTEXT:
{context}

CODE:
{code_content}

SEVERITY LEVELS for {language}:
- Critical: O(N^2) algorithms, unvectorized tight loops, cache thrashing, warp divergence.
- High: Unnecessary copies, GIL held during I/O, heap allocs in hot loops.
- Medium: Suboptimal data structures, missing const refs, global vars in hot paths.
- Low: Trivial style only — use ONLY if the code is provably optimal.

RULES:
1. Find ONE root cause — not a list of symptoms.
2. Name the exact mechanism: cache line size, GIL, algorithmic complexity.
3. Set `optimized_code` to null — another agent writes the fix.
4. When unsure between two severities, pick the higher one.

{format_instructions}
"""

# ── SMALL — minimal instructions, few-shot examples to prevent parroting ─────
BOTTLENECK_TEMPLATE_SMALL = """
Find the biggest performance problem in this {language} code.

HARDWARE: {hardware_target}

CODE:
{code_content}

CONTEXT:
{context}

RULES:
1. Find ONE problem. Name the exact mechanism (example: "O(N²) nested loop", "dict rebuilt every call", "GIL held during sleep").
2. Do NOT copy words from this prompt. Use your own words based on the actual code.
3. Set optimized_code to null.
4. If unsure between two severities, pick the higher one.

GOOD OUTPUT EXAMPLE:
  severity: "Critical"
  issue: "Nested loop scans full list for every element"
  reasoning: "For N=10000 this runs 100M iterations. Each inner scan is O(N),
              making the total O(N²). A dict keyed by match_id reduces each
              lookup to O(1) amortized, making the whole function O(N)."

BAD OUTPUT EXAMPLE (do not do this — these are words from the prompt, not analysis):
  severity: "Critical"
  issue: "O(N^2) algorithms, tight loops that should be NumPy/PyTorch vectorized"
  reasoning: "This code has critical performance bottlenecks that severely limit
              scalability and throughput for large inputs on the target hardware."

{format_instructions}
"""

# ── Tier selector ─────────────────────────────────────────────────────────────
BOTTLENECK_TEMPLATES = {
    ModelTier.SMALL:  BOTTLENECK_TEMPLATE_SMALL,
    ModelTier.MEDIUM: BOTTLENECK_TEMPLATE_MEDIUM,
    ModelTier.LARGE:  BOTTLENECK_TEMPLATE_LARGE,
}

# Backward-compatible alias — existing code imports BOTTLENECK_TEMPLATE
BOTTLENECK_TEMPLATE = BOTTLENECK_TEMPLATE_LARGE