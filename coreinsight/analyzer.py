import re
import logging
from typing import Optional, List
from pydantic import BaseModel, Field

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from coreinsight.prompts import SYSTEM_PROMPT, ANALYSIS_TEMPLATE, HARNESS_ADDENDUM

logger = logging.getLogger(__name__)


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
            base_url = api_keys.get("local_url", "http://localhost:1234/v1")
            self.base_llm = ChatOpenAI(
                model=model_name,
                api_key="not-needed",
                base_url=base_url,
                temperature=0.1,
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
            self.base_llm = ChatOllama(
                model=model_name,
                temperature=0.1,
                num_predict=4096,
                num_ctx=8192,
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

    def analyze(self, code: str, language: str, context: str = "", hardware_target: str = "Generic CPU"):
        try:
            return self.chain.invoke({
                "language": language,
                "code_content": code,
                "context": context,
                "hardware_target": hardware_target,
            })
        except OutputParserException:
            return {
                "severity": "Error",
                "issue": "AI Output Parsing Failed",
                "reasoning": "The model failed to return valid JSON.",
                "suggestion": "Try running the analysis again or use a larger parameter model.",
                "bottlenecks": [],
                "optimized_code": None,
            }
        except Exception as e:
            return {
                "severity": "Error",
                "issue": str(e),
                "reasoning": "System error during analysis pipeline.",
                "suggestion": "Check LLM API keys and connectivity.",
                "bottlenecks": [],
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

    def _invoke_code_chain(self, template: str, variables: dict, language: str) -> str:
        """Shared invocation + extraction logic for harness and fix chains."""
        chain = PromptTemplate.from_template(template) | self.base_llm
        result = chain.invoke(variables)
        raw = result.content if hasattr(result, "content") else str(result)
        # Handle Anthropic returning a list of content blocks
        if isinstance(raw, list):
            raw = "\n".join(
                item["text"] if isinstance(item, dict) and "text" in item else str(item)
                for item in raw
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
    ) -> str:
        try:
            tiered_template = _HARNESS_TEMPLATE + HARNESS_ADDENDUM.get(self.model_tier, "")
            
            return self._invoke_code_chain(
                tiered_template,
                {
                    "language": language,
                    "func_name": func_name,
                    "original": original_code,
                    "optimized": optimized_code,
                    "context": context,
                    "hardware_target": hardware_target,
                },
                language,
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
    ) -> str:
        try:
            tiered_template = _FIX_TEMPLATE + HARNESS_ADDENDUM.get(self.model_tier, "")
            
            return self._invoke_code_chain(
                tiered_template,
                {
                    "language": language,
                    "func_name": func_name,
                    "original": original_code,
                    "bad_harness": bad_harness,
                    "error_logs": error_logs,
                    "context": context,
                },
                language,
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