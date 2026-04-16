import json
from pydantic import BaseModel, Field
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from typing import Any, List, Optional
from typing import cast, Any, List, Optional, Union

# 1. Define the exact Schema DeepEval expects
# This forces the LLM to provide 'truths' and 'claims' specifically.
class FaithfulnessSchema(BaseModel):
    truths: List[str] = Field(description="List of factual statements extracted from the retrieval context.")
    claims: List[str] = Field(description="List of claims made in the actual output.")

class OllamaDeepEval(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama3.1:8b"):
        # We use the 8B model for higher reasoning accuracy
        self.model = ChatOllama(
            model=model_name,
            format="json",
            temperature=0.0
        )
        # Bind the schema to the model for Structured Output
        self.structured_model = self.model.with_structured_output(FaithfulnessSchema)

    def load_model(self) -> Any:
        return self.model

    def generate(self, prompt: str) -> str:
        try:
            # We tell Pyright: 'The result will be a FaithfulnessSchema OR a dict'
            res = self.structured_model.invoke(prompt)
            
            # Handle if it's already a dictionary (failsafe)
            if isinstance(res, dict):
                return json.dumps(res)
            
            # Use 'cast' to access Pydantic methods safely
            return cast(FaithfulnessSchema, res).model_dump_json()
            
        except Exception as e:
            print(f"--- STRUCTURED GEN ERROR: {e} ---")
            return json.dumps({"truths": [], "claims": []})

    async def a_generate(self, prompt: str) -> str:
        try:
            res = await self.structured_model.ainvoke(prompt)
            
            if isinstance(res, dict):
                return json.dumps(res)
                
            return cast(FaithfulnessSchema, res).model_dump_json()
            
        except Exception as e:
            print(f"--- STRUCTURED ASYNC GEN ERROR: {e} ---")
            return json.dumps({"truths": [], "claims": []})
    def get_model_name(self) -> str:
        return "Llama 3.1 8B (Structured)"

def check_faithfulness(question: str, context: str, answer: str):
    judge_model = OllamaDeepEval()
    
    # Threshold 0.7: If faithfulness is low, we know the AI halluncinated.
    metric = FaithfulnessMetric(threshold=0.7, model=judge_model)
    
    test_case = LLMTestCase(
        input=str(question),
        actual_output=str(answer),
        retrieval_context=[str(context)]
    )
    
    try:
        metric.measure(test_case)
        # DeepEval might return None if it fails internally, so we use 'or 0.0'
        score = float(metric.score) if metric.score is not None else 0.5
        reason = str(metric.reason) if metric.reason else "Reasoning failed but check completed."
        
        return score, reason
    except Exception as e:
        # Final safety net so the FastAPI endpoint doesn't return 500
        print(f"⚠️ DeepEval Critical Failure: {e}")
        return 0.5, f"Evaluation error: {str(e)}"