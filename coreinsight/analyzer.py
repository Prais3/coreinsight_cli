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
            input_variables=["language", "code_content", "hardware_target"],
            partial_variables={
                "system_prompt": SYSTEM_PROMPT,
                "format_instructions": self.parser.get_format_instructions()
            }
        )
        self.chain = self.prompt | self.json_llm | self.parser

    def analyze(self, code: str, language: str, hardware_target: str = "Generic CPU"):
        try:
            return self.chain.invoke({
                "language": language,
                "code_content": code,
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
    
    def generate_harness(self, original_code: str, optimized_code: str, language: str) -> str:
        harness_prompt = """
        You are a strict QA engineer. I have an original function and an optimized function.
        Write a standalone benchmark script that:
        1. Includes necessary headers/imports (e.g., <iostream>, <chrono>, <vector>).
        2. Initializes realistic dummy data for the functions.
        3. Executes the original function and times it.
        4. Executes the optimized function and times it.
        5. Verifies the outputs match.
        6. Prints the execution times and the speedup multiplier to stdout.

        Language: {language}
        
        Original Code Signature/Logic:
        {original}
        
        Optimized Code Signature/Logic:
        {optimized}
        
        CRITICAL: Output ONLY the raw executable code (like the `int main()` block). Do not use markdown wrappers. Just the code.
        """
        
        try:
            chain = PromptTemplate.from_template(harness_prompt) | self.base_llm
            result = chain.invoke({
                "language": language,
                "original": original_code,
                "optimized": optimized_code
            })
            
            code = result.content if hasattr(result, 'content') else str(result)
            
            # Robust Extraction
            match = re.search(r'```(?:cpp|python|c\+\+|cu|cuh|c)?\n(.*?)```', code, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
            else:
                code = code.replace("```cpp", "").replace("```python", "").replace("```", "").strip()
                
            return code
        except Exception as e:
            return f"// Failed to generate harness: {e}\nint main() {{ return 1; }}"