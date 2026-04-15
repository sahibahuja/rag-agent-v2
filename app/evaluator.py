from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from typing import Any, Union, List

# We wrap your Ollama model so DeepEval can talk to it
class OllamaDeepEval(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama3.2:3b"):
        # We don't call super().__init__ because DeepEvalBaseLLM doesn't require it
        
        # 🚨 THE FIX: Force JSON mode and make the model 100% deterministic
        self.model = ChatOllama(
            model=model_name,
            format="json",
            temperature=0.0
        )

    # FIX 1: The return type must technically be 'Any' or 'DeepEvalBaseLLM'
    # but since this method is meant to return the underlying model, we use Any
    def load_model(self) -> Any:
        return self.model

    # FIX 2 & 3: DeepEval expects a string, but LangChain content can be str or list.
    # We force it to a string to satisfy the return type.
    def generate(self, prompt: str) -> str:
        res = self.model.invoke(prompt)
        return str(res.content)

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return str(res.content)

    def get_model_name(self) -> str:
        return "Llama 3.2 (Ollama)"

def check_faithfulness(question: str, context: str, answer: str):
    # Use Llama 3.2 as the judge
    judge_model = OllamaDeepEval()
    
    # Define the metric
    metric = FaithfulnessMetric(threshold=0.7, model=judge_model)
    
    # Create a test case
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=[context]
    )
    
    metric.measure(test_case)
    return float(metric.score or 0.0), str(metric.reason or "No reason provided")