from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated, TypedDict
import operator
from typing_extensions import NotRequired, Literal

# --- Phase 1: Data Contracts ---

class StorePayload(BaseModel):
    """Payload for indexing new documents via Docling"""
    file_path: str = Field(..., description="Absolute path to the Document/File")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional tags for Qdrant")

class Message(BaseModel):
    """Standard message format for history (OpenAI compatible)"""
    role: str  # 'user' or 'assistant'
    content: str

# --- Phase 2: Agentic Contracts ---

class ChatPayload(BaseModel):
    """Payload for the Agentic Chat endpoint"""
    question: str
    thread_id: Optional[str] = Field(None, description="Unique ID for session persistence in Redis")

class ChatResponse(BaseModel):
    """Structured response for the final API output"""
    answer: str
    sources: List[str] = Field(default_factory=list)
    iteration_count: int
    faithfulness_score: float = 0.0
    faithfulness_reason: str = ""

# --- Pillar 2: Structured Output Schemas (The Permanent Fix) ---

class RouteQuery(BaseModel):
    """Routing logic: Where should the question go?"""
    # CHANGED: 'str' is now 'Literal["vector_store", "chat_history"]'
    datasource: Literal["vector_store", "chat_history"] = Field(
        description="You MUST choose exactly 'vector_store' or 'chat_history'. Do not use any other words."
    )

class GradeSchema(BaseModel):
    """Relevance check: Is the document useful?"""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

class FaithfulnessSchema(BaseModel):
    """Judge logic: Did the AI hallucinate?"""
    truths: List[str] = Field(description="Factual statements from context")
    claims: List[str] = Field(description="Statements made in the answer")

# --- Pillar 3: Graph State Definition ---

class GraphState(TypedDict):
    question: str              # Original user input (Sacred)
    search_query: str          # Optimized search keywords
    iteration_count: int
    history: Annotated[list, operator.add]
    context: NotRequired[List[str]]
    response: NotRequired[str]
    is_relevant: NotRequired[str]
    sources: NotRequired[List[str]]
    faithfulness_score: float 
    faithfulness_reason: str