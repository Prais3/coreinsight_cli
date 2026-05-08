"""
coreinsight/prompts/test_cases.py
Test case generation prompt — used by generate_test_cases() in analyzer.py.
"""

TEST_CASES_TEMPLATE = """
You are a QA engineer writing correctness test cases for a function.

FUNCTION NAME: {func_name}
LANGUAGE: {language}

FUNCTION SIGNATURE AND BODY:
{original}

GLOBAL DEPENDENCIES (helper functions / structs this function relies on):
{context}

Your task: generate {num_cases} diverse test cases that call `{func_name}` with different
arguments. The cases must cover:
  - Small inputs (N ~ 10)
  - Medium inputs (N ~ 100-500)
  - Edge cases: empty collections, single-element, all-zeros, negative values (where applicable)
  - Boundary conditions specific to this function's logic

OUTPUT FORMAT — respond with ONLY a valid JSON array, nothing else. No markdown fences,
no explanation. Each element must be a JSON object with exactly two keys:
  "args"  : a JSON array of positional arguments (use only JSON-serialisable types:
            numbers, strings, booleans, arrays, objects — NO numpy, NO bytes)
  "kwargs": a JSON object of keyword arguments (may be empty {{}})

Example (do NOT copy this — generate cases specific to {func_name}):
[
  {{"args": [[1, 2, 3]], "kwargs": {{}}}},
  {{"args": [[]], "kwargs": {{}}}},
  {{"args": [[9, -1, 4, 0, 7]], "kwargs": {{"reverse": true}}}}
]

CONSTRAINTS:
- All values must be plain JSON types — no numpy arrays, no custom objects.
- Do NOT instantiate any classes (no ClassName(...), no dataclasses, no enums). Use plain dicts instead.
- If the function operates on a matrix, represent it as a list-of-lists.
- If the function takes a size integer N, generate concrete data of that size inline.
- Do NOT include function calls or expressions — only literal values.
- Produce exactly {num_cases} test cases.
- SANDBOX RULE: these test cases run in an isolated environment where project-specific
  imports (custom classes, dataclasses, enums) have been stripped. If the function
  accepts custom objects, pass plain dicts with the same field names instead of class
  instances — e.g. {{"id": 1, "amount": 100.0}} not Transaction(id=1, amount=100.0).
  The function code may use attribute access (obj.field) — dict args are fine for that
  because the sandbox converts them via SimpleNamespace automatically.
- IN-PLACE MUTATION: if the function mutates its input in-place (e.g. sorts a list,
  fills an array) and returns None, add an optional "expected_mutation" key to the
  test case object — a copy of the input argument in its expected post-call state.
  The harness will compare the mutated arg against expected_mutation if present.
"""