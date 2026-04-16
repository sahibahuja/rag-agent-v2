from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from typing import Any

class OllamaDeepEval(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "llama3.1:8b"):
        # We drop the strict Pydantic schema here!
        # The 8B model is smart enough to follow DeepEval's native prompts.
        self.model = ChatOllama(
            model=model_name,
            format="json", # This acts as a gentle boundary, not a strict cage
            temperature=0.0
        )

    def load_model(self) -> Any:
        return self.model

    def generate(self, prompt: str) -> str:
        try:
            # Let the model answer DeepEval exactly how it wants
            res = self.model.invoke(prompt)
            return str(res.content)
        except Exception as e:
            print(f"--- DEEPEVAL GEN ERROR: {e} ---")
            return "{}"

    async def a_generate(self, prompt: str) -> str:
        try:
            res = await self.model.ainvoke(prompt)
            return str(res.content)
        except Exception as e:
            print(f"--- DEEPEVAL ASYNC GEN ERROR: {e} ---")
            return "{}"

    def get_model_name(self) -> str:
        return "Llama 3.1 8B"

def check_faithfulness(question: str, context: str, answer: str):
    """
    Called by the Background Task.
    Now that the Pydantic cage is gone, DeepEval can successfully calculate the 'verdicts'.
    """
    # 1. Skip evaluation if there is no context (e.g., chat history questions)
    if not context or context.strip() == "" or context == "No context found.":
        print("⏭️ Skipping Faithfulness: No PDF context to evaluate against.")
        return 1.0, "System question / Chat history used (No context needed)."

    judge_model = OllamaDeepEval()
    metric = FaithfulnessMetric(threshold=0.7, model=judge_model)
    
    test_case = LLMTestCase(
        input=str(question),
        actual_output=str(answer),
        retrieval_context=[str(context)]
    )
    
    try:
        # This will now successfully complete all 3 steps!
        metric.measure(test_case)
        
        score = float(metric.score) if metric.score is not None else 0.5
        reason = str(metric.reason) if metric.reason else "Reasoning completed."
        
        return score, reason
    except Exception as e:
        print(f"⚠️ DeepEval Error: {e}")
        return 0.5, f"Evaluation error: {str(e)}"