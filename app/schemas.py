from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from typing import List, TypedDict
from typing_extensions import NotRequired
# --- Phase 1 Migration: Data Contracts ---

class StorePayload(BaseModel):
    """Payload for indexing new documents via Docling"""
    file_path: str = Field(..., description="Absolute path to the PDF/File")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional tags for Qdrant")

class Message(BaseModel):
    """Standard message format for history (OpenAI compatible)"""
    role: str  # 'user' or 'assistant'
    content: str

# --- Phase 2 Additions: Agentic Contracts ---

class ChatPayload(BaseModel):
    """Payload for the Agentic Chat endpoint"""
    question: str
    history: List[Message] = Field(default_factory=list)
    # Added for Redis persistence in Phase 2
    thread_id: Optional[str] = Field(None, description="Unique ID for session persistence in Redis")

class ChatResponse(BaseModel):
    """Structured response from the LangGraph agent"""
    answer: str
    sources: List[str] = Field(default_factory=list)
    # Helpful for debugging in Phoenix/Postman
    iteration_count: int
    faithfulness_score: float = 0.0
    faithfulness_reason: str = ""

# 1. Define the State
class GraphState(TypedDict):
    question: str
    iteration_count: int
    search_query: NotRequired[str]
    context: NotRequired[List[str]]
    response: NotRequired[str]
    is_relevant: NotRequired[str]
    sources: NotRequired[List[str]]
    faithfulness_score: float 
    faithfulness_reason: str
