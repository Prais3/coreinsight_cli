import re
from typing import Optional, List
from pydantic import BaseModel, Field

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException

# Provider Imports
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from coreinsight.prompts import SYSTEM_PROMPT, ANALYSIS_TEMPLATE

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
    
class AnalyzerAgent:
    def __init__(self, provider="ollama", model_name="llama3.2", api_keys=None):
        self.parser = JsonOutputParser(pydantic_object=AuditResult)
        api_keys = api_keys or {}
        
        # 1. Initialize the Base LLM
        if provider == "openai":
            if not api_keys.get("openai"):
                raise ValueError("OpenAI API Key required.")
            self.base_llm = ChatOpenAI(
                model=model_name,
                api_key=api_keys["openai"],
                temperature=0.1,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            self.json_llm = self.base_llm

        elif provider == "local_server":
            base_url = api_keys.get("local_url", "http://localhost:1234/v1")
            self.base_llm = ChatOpenAI(
                model=model_name,
                api_key="not-needed", # Local servers usually ignore this
                base_url=base_url,
                temperature=0.1,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            self.json_llm = self.base_llm

        elif provider == "anthropic":
            if not api_keys.get("anthropic"):
                raise ValueError("Anthropic API Key required.")
            self.base_llm = ChatAnthropic(
                model=model_name,
                api_key=api_keys["anthropic"],
                temperature=0.1
            )
            self.json_llm = self.base_llm

        elif provider == "google":
            if not api_keys.get("google"):
                raise ValueError("Google Gemini API Key required.")
            self.base_llm = ChatGoogleGenerativeAI(
                model=model_name, 
                google_api_key=api_keys["google"],
                temperature=0.1,
                convert_system_message_to_human=True
            )
            self.json_llm = self.base_llm 

        else: # Default to Ollama
            self.base_llm = ChatOllama(
                model=model_name,
                temperature=0.1,
                num_predict=4096,
                num_ctx=8192
            )
            self.json_llm = self.base_llm.bind(format="json")
        
        # 2. Bind the JSON format specifically for the analysis chain
        self.prompt = PromptTemplate(
            template=ANALYSIS_TEMPLATE + "\n\n{format_instructions}",
            input_variables=["language", "code_content", "context", "hardware_target"],
            partial_variables={
                "system_prompt": SYSTEM_PROMPT,
                "format_instructions": self.parser.get_format_instructions()
            }
        )
        self.chain = self.prompt | self.json_llm | self.parser

    def analyze(self, code: str, language: str, context: str = "", hardware_target: str = "Generic CPU"):
        try:
            return self.chain.invoke({
                "language": language,
                "code_content": code,
                "context": context,
                "hardware_target": hardware_target
            })
        except OutputParserException:
            return {
                "severity": "Error",
                "issue": "AI Output Parsing Failed",
                "reasoning": "The model failed to return valid JSON.",
                "suggestion": "Try running the analysis again or use a larger parameter model.",
                "bottlenecks": [],
                "optimized_code": None
            }
        except Exception as e:
            return {
                "severity": "Error",
                "issue": str(e),
                "reasoning": "System error during analysis pipeline.",
                "suggestion": "Check LLM API keys and connectivity.",
                "bottlenecks": [],
                "optimized_code": None
            }
    
    def generate_harness(self, func_name: str, original_code: str, optimized_code: str, language: str, context: str = "", hardware_target: str = "Generic CPU") -> str:
        harness_prompt = """
        You are a strict QA engineer writing a standalone asymptotic scaling benchmark script in {language}.

        ORIGINAL FUNCTION (Name: {func_name}):
        {original}
        
        OPTIMIZED FUNCTION:
        {optimized}
        
        GLOBAL DEPENDENCIES (Helper functions/structs required to run the code):
        {context}
        
        Write the complete executable script (e.g., `int main()` or `if __name__ == "__main__":`) that:
        1. Includes necessary imports/headers.
        2. INCLUDES all required helper functions or structs from the GLOBAL DEPENDENCIES above so the script can run standalone.
        3. Defines BOTH the original and optimized functions exactly as provided above. 
        4. Creates a loop to test multiple data sizes (e.g., N=10, 100, 1000, 5000).
        5. Target Hardware: {hardware_target}. Ensure your max N crosses cache boundaries but DOES NOT exceed available RAM to prevent OOM crashes.
        6. Initializes realistic dummy data for the arguments at each size N.
        7. Time the execution of the original vs optimized functions.
        
        CRITICAL TIMING MATH:
        - Use HIGH-RESOLUTION timers (e.g., `time.perf_counter()` in Python, `std::chrono::high_resolution_clock` in C++).
        - Calculate duration strictly as `end_time - start_time`.
        - Prevent zero-division by clamping both times: `orig_time = max(end_time - start_time, 1e-9)` and `opt_time = max(end_time - start_time, 1e-9)`.
        - Calculate the speedup dynamically: `speedup = orig_time / opt_time`.

        ENVIRONMENT CONSTRAINTS (CRITICAL):
        This script will run in a totally empty, isolated Docker container. There are NO other files in the directory.
        Therefore, you CANNOT use local imports (e.g., `from data_processor import...`). 
        You MUST define `{func_name}`, the optimized function, and any GLOBAL DEPENDENCIES inline directly inside this script.
        
        CRITICAL CONSTRAINTS:
        - DO NOT rename the original function. You MUST call it exactly `{func_name}`.
        - [FOR PYTHON ONLY]: Import `matplotlib.pyplot as plt`, plot Original vs Optimized times against N, and save the plot as `benchmark_plot.png` in the current working directory.
        
        STDOUT FORMATTING (CRITICAL):
        You MUST print the final results to stdout in EXACTLY this CSV format, with this exact header, and absolutely no other text:
        N,Original_Time,Optimized_Time,Speedup
        10,0.002,0.001,2.00
        100,0.020,0.005,4.00
        
        Output ONLY the raw executable code. No markdown formatting, no backticks, no explanations.
        """
        
        try:
            chain = PromptTemplate.from_template(harness_prompt) | self.base_llm
            result = chain.invoke({
                "language": language,
                "func_name": func_name,
                "original": original_code,
                "optimized": optimized_code,
                "context": context
            })
            
            code = result.content if hasattr(result, 'content') else str(result)
            
            match = re.search(r'```(?:cpp|python|c\+\+|cu|cuh|c)?\n(.*?)```', code, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
            else:
                code = code.replace("```cpp", "").replace("```python", "").replace("```", "").strip()
                
            return code
        except Exception as e:
            return f"// Failed to generate harness: {e}\nint main() {{ return 1; }}"
        
    def fix_harness(self, func_name: str, original_code: str, bad_harness: str, error_logs: str, language: str, context: str = "") -> str:
        fix_prompt = """
        You are an expert {language} developer. You previously wrote an ASYMPTOTIC benchmark script that FAILED in the execution sandbox.
        
        ORIGINAL FUNCTION TO BENCHMARK (Name: {func_name}):
        {original}
        
        GLOBAL DEPENDENCIES (Helper functions/structs required to run the code):
        {context}
        
        YOUR FAILED SCRIPT:
        {bad_harness}
        
        EXECUTION ERROR / TIMEOUT LOGS:
        {error_logs}

        ENVIRONMENT CONSTRAINTS (CRITICAL):
        This script will run in a totally empty, isolated Docker container. There are NO other files in the directory.
        Therefore, you CANNOT use local imports (e.g., `from data_processor import...`). 
        You MUST define `{func_name}`, the optimized function, and any GLOBAL DEPENDENCIES inline directly inside this script.

        INSTRUCTIONS:
        1. Identify why the code failed (e.g., infinite loop, missing import, NameError).
        2. Fix the error (e.g., missing dependencies, NameError, ModuleNotFoundError). Check GLOBAL DEPENDENCIES if needed: {context}
        3. STRICT ISOLATION RULE: DO NOT import local files! If you get a `ModuleNotFoundError` for a local file, it's because you tried to import it instead of defining the function directly in the script.
        4. CRITICAL TIMING: Use `time.perf_counter()` (Python) or `std::chrono` (C++). Clamp times using `max(end - start, 1e-9)` to prevent negative zeros and division by zero.
        5. YOU MUST MAINTAIN the strict CSV printing loop: "N,Original_Time,Optimized_Time,Speedup".
        6. REMEMBER: Test multiple dataset sizes (N). If Python, save `benchmark_plot.png`. Print ONLY the exact CSV format to stdout.
        7. Output ONLY the raw executable {language} code. No markdown wrappers, no explanations.
        """
        
        try:
            chain = PromptTemplate.from_template(fix_prompt) | self.base_llm
            result = chain.invoke({
                "language": language,
                "func_name": func_name,
                "original": original_code,
                "context": context,
                "bad_harness": bad_harness,
                "error_logs": error_logs
            })
            
            code = result.content if hasattr(result, 'content') else str(result)
            
            match = re.search(r'```(?:cpp|python|c\+\+|cu|cuh|c)?\n(.*?)```', code, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
            else:
                code = code.replace("```cpp", "").replace("```python", "").replace("```", "").strip()
                
            return code
        except Exception as e:
            return bad_harness # Fallback to the original bad harness if the LLM fails