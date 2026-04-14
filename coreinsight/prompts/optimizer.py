"""
coreinsight/prompts/optimizer.py
Optimizer prompt — Agent 3 in multi-agent pipeline.
Receives bottleneck analysis, outputs only the rewritten function.
"""
from coreinsight.prompts._base import ModelTier

# ── LARGE ────────────────────────────────────────────────────────────────────
OPTIMIZER_TEMPLATE_LARGE = """
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
4. VERIFICATION: Before outputting, mentally confirm: does the rewrite directly eliminate the identified bottleneck? If the issue was O(N²), confirm the new complexity is O(N log N) or better. If the issue was a Python loop, confirm it is vectorized with NumPy/PyTorch. If the issue was a deep copy, confirm it is eliminated.
5. Raw {language} code only — no explanation, no markdown fences, no JSON.
6. Do NOT rename the function.
"""

# ── MEDIUM ───────────────────────────────────────────────────────────────────
OPTIMIZER_TEMPLATE_MEDIUM = """
Rewrite the function below to fix the identified performance problem.

LANGUAGE: {language}
FUNCTION: {func_name}
HARDWARE: {hardware_target}

PROBLEM:
  Severity: {severity}
  Issue: {issue}
  Fix: {suggestion}

ORIGINAL:
{original}

DEPENDENCIES:
{context}

RULES:
1. Output ONLY the rewritten function — no explanation, no markdown fences.
2. Keep the exact same function name and signature.
3. The fix must directly address the issue above.
4. Do not introduce new dependencies beyond stdlib + numpy/pandas.
"""

# ── SMALL — stripped to essentials, no multi-step reasoning ──────────────────
OPTIMIZER_TEMPLATE_SMALL = """
Rewrite this {language} function to fix the problem described below.

FUNCTION NAME: {func_name}

PROBLEM TO FIX:
  Issue: {issue}
  How: {suggestion}

ORIGINAL FUNCTION:
{original}

DEPENDENCIES:
{context}

OUTPUT RULES:
- Output ONLY the rewritten function. No explanation. No markdown. No JSON.
- Keep the exact same function name: {func_name}
- Keep the exact same parameters.

GOOD OUTPUT: starts with "def {func_name}(" and nothing else before or after.
BAD OUTPUT: any explanation, any markdown fence, any import outside the function.
"""

# ── Tier selector ─────────────────────────────────────────────────────────────
OPTIMIZER_TEMPLATES = {
    ModelTier.SMALL:  OPTIMIZER_TEMPLATE_SMALL,
    ModelTier.MEDIUM: OPTIMIZER_TEMPLATE_MEDIUM,
    ModelTier.LARGE:  OPTIMIZER_TEMPLATE_LARGE,
}

# Backward-compatible alias
OPTIMIZER_TEMPLATE = OPTIMIZER_TEMPLATE_LARGE