import re
import logging
from typing import Callable, Optional, List
from pydantic import BaseModel, Field

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from coreinsight.prompts import SYSTEM_PROMPT, ANALYSIS_TEMPLATE, HARNESS_ADDENDUM

# Phrases that appear at the start of a truncated LLM response
_TRUNCATION_HINTS = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "token limit",
    "finish_reason: length",
    "finish_reason\":\"length",
)

def _is_truncated(raw: str) -> bool:
    """
    Returns True if the raw LLM output looks like it was cut off mid-generation.
    Catches both explicit error messages and structural truncation signs.
    """
    if not raw or len(raw.strip()) < 20:
        return True
    low = raw.lower()
    if any(hint in low for hint in _TRUNCATION_HINTS):
        return True
    stripped = raw.strip()
    # JSON truncation: opened but never closed
    if stripped.startswith("{") and not stripped.endswith("}"):
        return True
    # Code truncation: opens a block but ends mid-statement
    if stripped.endswith(("...", "/*", "//", "\"", "'")):
        return True
    return False

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------
# Prompt compression - SMALL-tier models (≤7B) within their 4 096-token context budget.
# -------------------------------------------------------------------------------------
_SMALL_CONTEXT_CHAR_LIMIT = 1_200   # ~300 tokens — enough for signatures
_SMALL_CODE_CHAR_LIMIT    = 2_000   # ~500 tokens — function body cap

def _compress_for_small_model(
    context: str,
    code: str,
    model_tier: str,
) -> tuple:
    """
    Aggressively trims RAG context and target code for SMALL-tier models so
    the entire prompt + format instructions + response fit within 4 096 tokens.
    Returns (compressed_context, compressed_code). No-op for MEDIUM / LARGE.
    """
    from coreinsight.prompts import ModelTier
    if model_tier != ModelTier.SMALL:
        return context, code

    if context and len(context) > _SMALL_CONTEXT_CHAR_LIMIT:
        context = (
            context[:_SMALL_CONTEXT_CHAR_LIMIT]
            + "\n\n# [context truncated — top dependencies shown only]"
        )

    if code and len(code) > _SMALL_CODE_CHAR_LIMIT:
        lines = code.splitlines()
        kept  = lines[:60]
        if len(lines) > 60:
            kept.append(
                f"# ... [{len(lines) - 60} lines truncated for small model]"
            )
        code = "\n".join(kept)

    return context, code


class Bottleneck(BaseModel):
    line: int = Field(description="The approximate line number of the issue in the original code")
    severity: str = Field(description="Critical, High, Medium, Low")
    message: str = Field(description="Specific hardware or algorithmic bottleneck at this line")
    suggestion: str = Field(description="How to fix this specific line")


class AuditResult(BaseModel):
    severity: str = Field(description="Overall severity: Critical, High, Medium, Low")
    issue: str = Field(description="Brief overall description of the bottleneck")
    reasoning: str = Field(description="Step-by-step hardware-level reasoning for the proposed changes")
    suggestion: str = Field(description="Overall specific fix strategy")
    bottlenecks: List[Bottleneck] = Field(description="List of specific line-level bottlenecks", default_factory=list)
    optimized_code: Optional[str] = Field(description="The entirely rewritten optimized code, ready to drop in", default=None)


_HARNESS_TEMPLATE = """
You are a strict QA engineer writing a standalone asymptotic scaling benchmark script in {language}.

ORIGINAL FUNCTION (Name: {func_name}):
{original}

OPTIMIZED FUNCTION:
{optimized}

GLOBAL DEPENDENCIES (Helper functions/structs required to run the code):
{context}

Write the complete executable script (e.g., `int main()` or `if __name__ == "__main__":`) that:
1. Includes necessary imports/headers.
2. Includes ALL required helper functions or structs from GLOBAL DEPENDENCIES so the script is fully standalone.
3. Defines BOTH the original and optimized functions exactly as provided above.
4. Tests multiple data sizes (e.g., N=10, 100, 1000, 5000).
5. Target Hardware: {hardware_target}. The largest N MUST cross cache boundaries but MUST NOT exceed 20% of available RAM to prevent OOM crashes.
6. Initializes realistic dummy data for each size N.
7. Times execution of original vs optimized using high-resolution timers.

CRITICAL TIMING:
- Python: use `time.perf_counter()`. C++: use `std::chrono::high_resolution_clock`.
- Clamp: `orig_time = max(end - start, 1e-9)` to prevent zero-division.
- Speedup: `speedup = orig_time / opt_time`.

ISOLATION RULES (CRITICAL):
- This runs in an empty Docker container. NO local files exist.
- DO NOT use local imports. Define everything inline.
- DO NOT rename the original function — call it exactly `{func_name}`.

OUTPUT FORMAT (CRITICAL):
Print ONLY this exact CSV to stdout, no other text:
N,Original_Time,Optimized_Time,Speedup
10,0.002,0.001,2.00

[PYTHON ONLY]: Also import matplotlib, plot results, and save as `benchmark_plot.png`.

FORMATTING RULE: Wrap your ENTIRE script in a single markdown code block. No text before or after.
"""

_FIX_TEMPLATE = """
You are an expert {language} developer. Your previous benchmark script FAILED in an isolated sandbox.

ORIGINAL FUNCTION (Name: {func_name}):
{original}

GLOBAL DEPENDENCIES:
{context}

YOUR FAILED SCRIPT:
{bad_harness}

EXECUTION ERROR LOGS:
{error_logs}

ISOLATION CONSTRAINTS (CRITICAL):
- Empty Docker container. No local files. NO local imports.
- Define `{func_name}` and all GLOBAL DEPENDENCIES inline.

FIX INSTRUCTIONS:
1. Diagnose the failure from the error logs above.
2. Fix imports, NameErrors, type mismatches, infinite loops, or OOM issues.
3. Maintain the CSV stdout format exactly: N,Original_Time,Optimized_Time,Speedup
4. Use high-resolution timers and clamp with `max(t, 1e-9)`.
5. [PYTHON ONLY]: Save benchmark plot as `benchmark_plot.png`.

FORMATTING RULE: Wrap your ENTIRE fixed script in a single markdown code block. No text before or after.
"""

_TEST_CASES_TEMPLATE = """
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
- If the function operates on a matrix, represent it as a list-of-lists.
- If the function takes a size integer N, generate concrete data of that size inline.
- Do NOT include function calls or expressions — only literal values.
- Produce exactly {num_cases} test cases.
"""


class AnalyzerAgent:
    def __init__(self, provider="ollama", model_name="llama3.2", api_keys=None, model_tier="large"):
        self.model_tier = model_tier
        self.parser = JsonOutputParser(pydantic_object=AuditResult)
        self.provider = provider
        api_keys = api_keys or {}

        if provider == "openai":
            if not api_keys.get("openai"):
                raise ValueError("OpenAI API Key required.")
            self.base_llm = ChatOpenAI(
                model=model_name,
                api_key=api_keys["openai"],
                temperature=0.1,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            self.json_llm = self.base_llm

        elif provider == "local_server":
            from coreinsight.prompts import ModelTier
            base_url   = api_keys.get("local_url", "http://localhost:1234/v1")
            _max_tokens = 2048 if model_tier == ModelTier.SMALL else 4096
            self.base_llm = ChatOpenAI(
                model=model_name,
                api_key="not-needed",
                base_url=base_url,
                temperature=0.1,
                max_tokens=_max_tokens,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            self.json_llm = self.base_llm

        elif provider == "anthropic":
            if not api_keys.get("anthropic"):
                raise ValueError("Anthropic API Key required.")
            self.base_llm = ChatAnthropic(
                model=model_name,
                api_key=api_keys["anthropic"],
                temperature=0.1,
            )
            # Anthropic doesn't support response_format; JSON is enforced via prompt only
            self.json_llm = self.base_llm

        elif provider == "google":
            if not api_keys.get("google"):
                raise ValueError("Google Gemini API Key required.")
            self.base_llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_keys["google"],
                temperature=0.1,
                convert_system_message_to_human=True,
            )
            self.json_llm = self.base_llm

        else:  # Ollama default
            from coreinsight.prompts import ModelTier
            # Small models (7B) typically have 4096 native context.
            # Asking for more causes silent degradation or OOM on the host.
            # Medium/large local models can handle 8192 comfortably.
            _ctx = 4096 if model_tier == ModelTier.SMALL else 8192
            # num_predict: small models need room for JSON + code in one shot.
            # Capping at 2048 for small prevents runaway generation that hits
            # the limit mid-JSON and returns truncated garbage.
            _predict = 2048 if model_tier == ModelTier.SMALL else 4096
            self.base_llm = ChatOllama(
                model=model_name,
                temperature=0.1,
                num_predict=_predict,
                num_ctx=_ctx,
            )
            self.json_llm = self.base_llm.bind(format="json")

        self.prompt = PromptTemplate(
            template=ANALYSIS_TEMPLATE + "\n\n{format_instructions}",
            input_variables=["language", "code_content", "context", "hardware_target"],
            partial_variables={
                "system_prompt": SYSTEM_PROMPT,
                "format_instructions": self.parser.get_format_instructions(),
            },
        )
        self.chain = self.prompt | self.json_llm | self.parser

    def analyze(
        self,
        code: str,
        language: str,
        context: str = "",
        hardware_target: str = "Generic CPU",
        stream_callback: Optional[Callable[[str], None]] = None,
    ):
        context, code = _compress_for_small_model(context, code, self.model_tier)
        try:
            if stream_callback is not None:
                # Stream raw tokens → accumulate → parse at end.
                # Keeps the cursor alive on slow local models instead of hanging.
                raw_chain   = self.prompt | self.json_llm
                accumulated = ""
                for chunk in raw_chain.stream({
                    "language":        language,
                    "code_content":    code,
                    "context":         context,
                    "hardware_target": hardware_target,
                }):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if isinstance(token, list):
                        token = "".join(
                            t.get("text", "") if isinstance(t, dict) else str(t)
                            for t in token
                        )
                    if token:
                        accumulated += token
                        stream_callback(token)
                return self.parser.parse(accumulated)
            return self.chain.invoke({
                "language":        language,
                "code_content":    code,
                "context":         context,
                "hardware_target": hardware_target,
            })
        except OutputParserException:
            return {
                "severity":       "Error",
                "issue":          "AI Output Parsing Failed",
                "reasoning":      "The model failed to return valid JSON.",
                "suggestion":     "Try running the analysis again or use a larger parameter model.",
                "bottlenecks":    [],
                "optimized_code": None,
            }
        except Exception as e:
            return {
                "severity":       "Error",
                "issue":          str(e),
                "reasoning":      "System error during analysis pipeline.",
                "suggestion":     "Check LLM API keys and connectivity.",
                "bottlenecks":    [],
                "optimized_code": None,
            }

    def _extract_executable_code(self, response_text: str) -> str:
        """Extract the first/longest fenced code block from model output."""
        blocks = re.findall(r"```[a-zA-Z+#]*\s*\n(.*?)```", response_text, re.DOTALL)
        if blocks:
            return max(blocks, key=len).strip()

        # Fallback: strip fence markers line by line to avoid corrupting code
        lines = response_text.strip().split("\n")
        lines = [l for l in lines if not re.match(r"^```", l)]
        while lines and lines[0].lower().startswith(("here is", "sure", "certainly", "output:")) \
                and not lines[0].strip().startswith(("#", "//")):
            lines.pop(0)
        return "\n".join(lines).strip()

    def _invoke_code_chain(
        self,
        template: str,
        variables: dict,
        language: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Shared invocation + extraction logic for harness and fix chains."""
        chain = PromptTemplate.from_template(template) | self.base_llm
        try:
            if stream_callback is not None:
                accumulated = ""
                for chunk in chain.stream(variables):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if isinstance(token, list):
                        token = "".join(
                            t.get("text", "") if isinstance(t, dict) else str(t)
                            for t in token
                        )
                    if token:
                        accumulated += token
                        stream_callback(token)
                raw = accumulated
            else:
                result = chain.invoke(variables)
                raw = result.content if hasattr(result, "content") else str(result)
                if isinstance(raw, list):
                    raw = "\n".join(
                        item["text"] if isinstance(item, dict) and "text" in item else str(item)
                        for item in raw
                    )
        except Exception as e:
            err = str(e).lower()
            if any(h in err for h in _TRUNCATION_HINTS):
                raise RuntimeError(
                    f"Model hit its context limit. Try a smaller file, fewer functions, "
                    f"or a model with a larger context window. Detail: {e}"
                ) from e
            raise
        if _is_truncated(raw):
            logger.warning(
                f"LLM output appears truncated (len={len(raw)}). "
                f"Model likely hit its context/predict limit."
            )
            raise RuntimeError(
                "Model output was truncated — hit context or token limit. "
                "Try a model with a larger context window, or reduce the function size."
            )
        return self._extract_executable_code(raw)

    def generate_harness(
        self,
        func_name: str,
        original_code: str,
        optimized_code: str,
        language: str,
        context: str = "",
        hardware_target: str = "Generic CPU",
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        try:
            context, original_code = _compress_for_small_model(
                context, original_code, self.model_tier
            )
            tiered_template = _HARNESS_TEMPLATE + HARNESS_ADDENDUM.get(self.model_tier, "")

            return self._invoke_code_chain(
                tiered_template,
                {
                    "language":        language,
                    "func_name":       func_name,
                    "original":        original_code,
                    "optimized":       optimized_code,
                    "context":         context,
                    "hardware_target": hardware_target,
                },
                language,
                stream_callback=stream_callback,
            )
        except Exception as e:
            is_python = language.lower() == "python"
            opener = "#" if is_python else "//"
            entry = 'if __name__ == "__main__":' if is_python else "int main() {"
            stub = "    pass" if is_python else "    return 1;\n}"
            return f"{opener} Failed to generate harness: {e}\n{entry}\n{stub}"

    def fix_harness(
        self,
        func_name: str,
        original_code: str,
        bad_harness: str,
        error_logs: str,
        language: str,
        context: str = "",
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        try:
            context, original_code = _compress_for_small_model(
                context, original_code, self.model_tier
            )
            tiered_template = _FIX_TEMPLATE + HARNESS_ADDENDUM.get(self.model_tier, "")

            return self._invoke_code_chain(
                tiered_template,
                {
                    "language":   language,
                    "func_name":  func_name,
                    "original":   original_code,
                    "bad_harness":bad_harness,
                    "error_logs": error_logs,
                    "context":    context,
                },
                language,
                stream_callback=stream_callback,
            )
        except Exception as e:
            is_python = language.lower() == "python"
            opener = "#" if is_python else "//"
            return f"{opener} Failed to fix harness: {e}"
        
    def generate_test_cases(
        self,
        func_name: str,
        original_code: str,
        language: str,
        context: str = "",
        num_cases: int = 8,
    ) -> list:
        """
        Ask the LLM to generate diverse, JSON-serialisable test cases for
        `func_name` so the sandbox can run correctness verification.

        Returns a list of {"args": [...], "kwargs": {...}} dicts, or an
        empty list if generation or parsing fails (sandbox skips gracefully).
        """
        import json

        chain = PromptTemplate.from_template(_TEST_CASES_TEMPLATE) | self.base_llm
        try:
            result = chain.invoke({
                "func_name": func_name,
                "language": language,
                "original": original_code,
                "context": context or "None",
                "num_cases": num_cases,
            })
            raw = result.content if hasattr(result, "content") else str(result)
            if isinstance(raw, list):
                raw = "\n".join(
                    item["text"] if isinstance(item, dict) and "text" in item else str(item)
                    for item in raw
                )

            # Strip markdown fences if the model wrapped anyway
            raw = re.sub(r"```[a-zA-Z]*\s*", "", raw).strip()
            raw = re.sub(r"```", "", raw).strip()

            # Sanitize Python literals that are invalid JSON
            raw = re.sub(r"\bNone\b", "null", raw)
            raw = re.sub(r"\bTrue\b", "true", raw)
            raw = re.sub(r"\bFalse\b", "false", raw)
            # Remove trailing commas before ] or }
            raw = re.sub(r",\s*([\]}])", r"\1", raw)

            # Extract the JSON array if extra text surrounds it
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                raw = match.group(0)

            try:
                cases = json.loads(raw)
            except json.JSONDecodeError:
                # Fallback: ast.literal_eval handles Python literals
                # (None, True, False, tuples) that LLMs commonly produce
                import ast
                cases = ast.literal_eval(raw)

            # Validate structure — drop malformed entries silently
            return [
                case for case in cases
                if isinstance(case, dict)
                and isinstance(case.get("args"), list)
                and isinstance(case.get("kwargs"), dict)
            ]

        except Exception as e:
            logger.warning(f"generate_test_cases failed for '{func_name}': {e}")
            return []
        
# ---------------------------------------------------------------------------
# Multi-agent support (v0.2.5)
# ---------------------------------------------------------------------------

def _build_llm(provider: str, model_name: str, api_keys: dict):
    """
    Shared LLM factory for all multi-agent classes.
    Returns (base_llm, json_llm) — same pattern as AnalyzerAgent.__init__.
    Raises ValueError on missing credentials.
    """
    api_keys = api_keys or {}

    if provider == "openai":
        if not api_keys.get("openai"):
            raise ValueError("OpenAI API key required.")
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_keys["openai"],
            temperature=0.1,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        return llm, llm

    if provider == "local_server":
        base_url    = api_keys.get("local_url", "http://localhost:1234/v1")
        _max_tokens = api_keys.pop("_predict", 4096)  # reuse same key as Ollama path
        llm = ChatOpenAI(
            model=model_name,
            api_key="not-needed",
            base_url=base_url,
            temperature=0.1,
            max_tokens=_max_tokens,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        return llm, llm

    if provider == "anthropic":
        if not api_keys.get("anthropic"):
            raise ValueError("Anthropic API key required.")
        llm = ChatAnthropic(
            model=model_name,
            api_key=api_keys["anthropic"],
            temperature=0.1,
        )
        return llm, llm

    if provider == "google":
        if not api_keys.get("google"):
            raise ValueError("Google Gemini API key required.")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_keys["google"],
            temperature=0.1,
            convert_system_message_to_human=True,
        )
        return llm, llm

    # Ollama default — context and predict budget are passed in from the
    # calling agent which knows its own model_tier.
    # Default to medium-safe values; callers override via kwargs if needed.
    _ctx     = api_keys.pop("_ctx",     8192)
    _predict = api_keys.pop("_predict", 4096)
    base = ChatOllama(
        model=model_name,
        temperature=0.1,
        num_predict=_predict,
        num_ctx=_ctx,
    )
    return base, base.bind(format="json")


def _build_llm_tiered(provider: str, model_name: str, api_keys: dict, model_tier: str):
    """Wraps _build_llm with tier-aware context settings for local providers."""
    from coreinsight.prompts import ModelTier
    keys = dict(api_keys or {})
    if provider == "ollama":
        keys["_ctx"]     = 4096 if model_tier == ModelTier.SMALL else 8192
        keys["_predict"] = 2048 if model_tier == ModelTier.SMALL else 4096
    elif provider == "local_server":
        # max_tokens controls response length — context window is server-side
        keys["_predict"] = 2048 if model_tier == ModelTier.SMALL else 4096
    return _build_llm(provider, model_name, keys)


class BottleneckAgent:
    """
    Agent 1 — analysis only.
    Identifies the single most critical bottleneck and returns the same
    dict structure as AnalyzerAgent.analyze() so process_function cannot
    tell the difference.  optimized_code is always None from this agent.
    """

    def __init__(
        self,
        provider:   str,
        model_name: str,
        api_keys:   dict,
        model_tier: str,
    ) -> None:
        from coreinsight.prompts import BOTTLENECK_TEMPLATE, SYSTEM_PROMPT
        self.model_tier = model_tier
        self.parser     = JsonOutputParser(pydantic_object=AuditResult)
        self._base_llm, self._json_llm = _build_llm_tiered(provider, model_name, api_keys, model_tier)

        self._prompt = PromptTemplate(
            template=BOTTLENECK_TEMPLATE,
            input_variables=[
                "language", "code_content", "context", "hardware_target",
            ],
            partial_variables={
                "system_prompt":      SYSTEM_PROMPT,
                "format_instructions": self.parser.get_format_instructions(),
            },
        )
        self._chain = self._prompt | self._json_llm | self.parser

    def analyze(
        self,
        code:            str,
        language:        str,
        context:         str = "",
        hardware_target: str = "Generic CPU",
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        context, code = _compress_for_small_model(context, code, self.model_tier)
        try:
            if stream_callback is not None:
                raw_chain   = self._prompt | self._json_llm
                accumulated = ""
                for chunk in raw_chain.stream({
                    "language":        language,
                    "code_content":    code,
                    "context":         context,
                    "hardware_target": hardware_target,
                }):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if isinstance(token, list):
                        token = "".join(
                            t.get("text", "") if isinstance(t, dict) else str(t)
                            for t in token
                        )
                    if token:
                        accumulated += token
                        stream_callback(token)
                return self.parser.parse(accumulated)
            return self._chain.invoke({
                "language":        language,
                "code_content":    code,
                "context":         context,
                "hardware_target": hardware_target,
            })
        except OutputParserException:
            return {
                "severity":       "Error",
                "issue":          "AI Output Parsing Failed",
                "reasoning":      "The model failed to return valid JSON.",
                "suggestion":     "Try running again or use a larger model.",
                "bottlenecks":    [],
                "optimized_code": None,
            }
        except Exception as e:
            return {
                "severity":       "Error",
                "issue":          str(e),
                "reasoning":      "System error during bottleneck analysis.",
                "suggestion":     "Check LLM API keys and connectivity.",
                "bottlenecks":    [],
                "optimized_code": None,
            }


class OptimizerAgent:
    """
    Agent 2 — code generation only.
    Receives the bottleneck analysis result and writes the optimized function.
    Returns raw code as a string (no JSON, no harness).
    """

    def __init__(
        self,
        provider:   str,
        model_name: str,
        api_keys:   dict,
        model_tier: str,
    ) -> None:
        from coreinsight.prompts import OPTIMIZER_TEMPLATE
        self.model_tier = model_tier
        self._base_llm, _ = _build_llm_tiered(provider, model_name, api_keys, model_tier)
        self._template = OPTIMIZER_TEMPLATE

    def _extract_code(self, raw: str) -> str:
        """Reuse the same extraction logic as AnalyzerAgent."""
        blocks = re.findall(r"```[a-zA-Z+#]*\s*\n(.*?)```", raw, re.DOTALL)
        if blocks:
            return max(blocks, key=len).strip()
        lines = raw.strip().split("\n")
        lines = [l for l in lines if not re.match(r"^```", l)]
        while lines and lines[0].lower().startswith(
            ("here is", "sure", "certainly", "output:")
        ) and not lines[0].strip().startswith(("#", "//")):
            lines.pop(0)
        return "\n".join(lines).strip()

    def generate(
        self,
        func_name:       str,
        original_code:   str,
        analysis:        dict,
        language:        str,
        context:         str = "",
        hardware_target: str = "Generic CPU",
    ) -> str:
        """
        Returns the optimized function as a raw code string.
        Returns original_code on any failure so the pipeline can continue.
        """
        try:
            context, original_code = _compress_for_small_model(
                context or "", original_code, self.model_tier
            )
            chain  = PromptTemplate.from_template(self._template) | self._base_llm
            result = chain.invoke({
                "language":        language,
                "func_name":       func_name,
                "hardware_target": hardware_target,
                "severity":        analysis.get("severity",  ""),
                "issue":           analysis.get("issue",     ""),
                "reasoning":       analysis.get("reasoning", ""),
                "suggestion":      analysis.get("suggestion",""),
                "original":        original_code,
                "context":         context or "None",
            })
            raw = result.content if hasattr(result, "content") else str(result)
            if isinstance(raw, list):
                raw = "\n".join(
                    item["text"] if isinstance(item, dict) and "text" in item
                    else str(item)
                    for item in raw
                )
            code = self._extract_code(raw)
            return code if code else original_code
        except Exception as e:
            logger.warning(f"OptimizerAgent.generate failed: {e}")
            return original_code


class HarnessAgent:
    """
    Agent 3 — harness generation and fix loop.
    Owns the entire retry loop so process_function stays clean.
    Returns (harness_code, success, logs, plot_data) after running in sandbox.
    """

    def __init__(
        self,
        provider:   str,
        model_name: str,
        api_keys:   dict,
        model_tier: str,
    ) -> None:
        from coreinsight.prompts import (
            HARNESS_TEMPLATE_MULTI,
            FIX_TEMPLATE_MULTI,
            HARNESS_ADDENDUM_MULTI,
        )
        self.model_tier      = model_tier
        self._base_llm, _    = _build_llm_tiered(provider, model_name, api_keys, model_tier)
        self._harness_tmpl   = HARNESS_TEMPLATE_MULTI + HARNESS_ADDENDUM_MULTI.get(model_tier, "")
        self._fix_tmpl       = FIX_TEMPLATE_MULTI     + HARNESS_ADDENDUM_MULTI.get(model_tier, "")

    def _extract_code(self, raw: str) -> str:
        blocks = re.findall(r"```[a-zA-Z+#]*\s*\n(.*?)```", raw, re.DOTALL)
        if blocks:
            return max(blocks, key=len).strip()
        lines = raw.strip().split("\n")
        lines = [l for l in lines if not re.match(r"^```", l)]
        while lines and lines[0].lower().startswith(
            ("here is", "sure", "certainly", "output:")
        ) and not lines[0].strip().startswith(("#", "//")):
            lines.pop(0)
        return "\n".join(lines).strip()

    def _invoke(self, template: str, variables: dict) -> str:
        chain  = PromptTemplate.from_template(template) | self._base_llm
        try:
            result = chain.invoke(variables)
        except Exception as e:
            err = str(e).lower()
            if any(h in err for h in _TRUNCATION_HINTS):
                raise RuntimeError(
                    f"Model hit its context limit during harness generation. "
                    f"Detail: {e}"
                ) from e
            raise
        raw = result.content if hasattr(result, "content") else str(result)
        if isinstance(raw, list):
            raw = "\n".join(
                item["text"] if isinstance(item, dict) and "text" in item
                else str(item)
                for item in raw
            )
        if _is_truncated(raw):
            raise RuntimeError(
                "Harness output was truncated — model hit its token limit. "
                "Switching to fix loop with truncation note."
            )
        return self._extract_code(raw)

    def _check_speedup(self, success: bool, logs: str) -> bool:
        if not success:
            return False
        try:
            for line in reversed(logs.strip().split("\n")):
                parts = line.split(",")
                if len(parts) == 4 and parts[0].strip().isdigit():
                    return float(parts[3]) >= 1.05
        except Exception:
            pass
        return False

    def run(
        self,
        func_name:       str,
        original_code:   str,
        optimized_code:  str,
        language:        str,
        context:         str,
        hardware_target: str,
        sandbox,                    # CodeSandbox instance
        max_retries:     int = 2,
    ) -> tuple:
        """
        Generates harness, runs in sandbox, retries on failure.
        Returns (success, logs, plot_data, retry_count).
        """
        try:
            harness = self._invoke(self._harness_tmpl, {
                "language":        language,
                "func_name":       func_name,
                "original":        original_code,
                "optimized":       optimized_code,
                "context":         context,
                "hardware_target": hardware_target,
            })
        except Exception as e:
            return False, f"Harness generation failed: {e}", None, 0

        success, logs, plot_data = sandbox.execute_benchmark(harness, language)
        is_valid  = self._check_speedup(success, logs)
        retries   = 0

        while not is_valid and retries < max_retries:
            if success and "N,Original_Time" not in logs:
                logs += "\nERROR: Script ran but did NOT print the CSV table. You MUST print the strict CSV format."
            elif success:
                logs += "\nERROR: Optimized code was SLOWER than original. Rewrite to be faster."

            try:
                harness = self._invoke(self._fix_tmpl, {
                    "language":   language,
                    "func_name":  func_name,
                    "original":   original_code,
                    "bad_harness":harness,
                    "error_logs": logs,
                    "context":    context,
                })
            except Exception as e:
                logs += f"\nFix generation failed: {e}"
                break

            success, logs, plot_data = sandbox.execute_benchmark(harness, language)
            is_valid = self._check_speedup(success, logs)
            retries += 1

        if getattr(sandbox, 'disabled', False):
            pass  # skipped intentionally — don't annotate as failed
        elif is_valid and retries > 0:
            logs = f"(Succeeded after {retries} retries)\n" + logs
        elif not is_valid:
            logs    = f"(Failed after {retries} retries)\n" + logs
            success = False

        return success, logs, plot_data, retries


class TestCaseAgent:
    """
    Agent 4 — test case generation only.
    Identical logic to AnalyzerAgent.generate_test_cases but as a
    standalone class so it can be called from a separate thread.
    """

    def __init__(
        self,
        provider:   str,
        model_name: str,
        api_keys:   dict,
        model_tier: str,
    ) -> None:
        self.model_tier   = model_tier
        self._base_llm, _ = _build_llm_tiered(provider, model_name, api_keys, model_tier)

    def generate(
        self,
        func_name:     str,
        original_code: str,
        language:      str,
        context:       str = "",
        num_cases:     int = 8,
    ) -> list:
        """
        Same return contract as AnalyzerAgent.generate_test_cases:
        list of {"args": [...], "kwargs": {...}} or [] on failure.
        """
        import json as _json

        chain = PromptTemplate.from_template(_TEST_CASES_TEMPLATE) | self._base_llm
        try:
            result = chain.invoke({
                "func_name": func_name,
                "language":  language,
                "original":  original_code,
                "context":   context or "None",
                "num_cases": num_cases,
            })
            raw = result.content if hasattr(result, "content") else str(result)
            if isinstance(raw, list):
                raw = "\n".join(
                    item["text"] if isinstance(item, dict) and "text" in item
                    else str(item)
                    for item in raw
                )

            raw = re.sub(r"```[a-zA-Z]*\s*", "", raw).strip()
            raw = re.sub(r"```",              "", raw).strip()
            raw = re.sub(r"\bNone\b",  "null",  raw)
            raw = re.sub(r"\bTrue\b",  "true",  raw)
            raw = re.sub(r"\bFalse\b", "false", raw)
            raw = re.sub(r",\s*([\]}])", r"\1", raw)

            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                raw = match.group(0)

            try:
                cases = _json.loads(raw)
            except _json.JSONDecodeError:
                import ast
                cases = ast.literal_eval(raw)

            return [
                c for c in cases
                if isinstance(c, dict)
                and isinstance(c.get("args"),   list)
                and isinstance(c.get("kwargs"), dict)
            ]
        except Exception as e:
            logger.warning(f"TestCaseAgent.generate failed for '{func_name}': {e}")
            return []