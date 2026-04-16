import uvicorn
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from app.schemas import StorePayload, ChatPayload, ChatResponse
from app.database import init_db
from app.engine import process_file
from app.observability import setup_tracing
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_core.runnables import RunnableConfig
from typing import cast

# --- 1. Lifespan Orchestration ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Phase 2 Agent: Initializing Qdrant Database...")
    try:
        setup_tracing()
        init_db()
        print("✅ Qdrant and FastEmbed ready.")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
    
    yield  
    print("🛑 Shutting down Agentic RAG services...")

# --- 2. FastAPI Setup ---
app = FastAPI(
    title="Phase 2: Agentic RAG (Docling + LangGraph)",
    description="Senior Software Engineer - AI Initiative",
    lifespan=lifespan
)

# --- 3. Endpoints ---

@app.post("/v2/ingest/file")
async def ingest_file(payload: StorePayload, background_tasks: BackgroundTasks):
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail="File path not found.")
    
    try:
        # Kept exactly as you had it
        count = process_file(payload.file_path, payload.metadata)
        return {"status": "success", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/v2/agent/chat")
async def chat_endpoint(payload: ChatPayload):
    # Respecting your import paths
    from app.graph import agent_builder, GraphState
    
    thread_id = payload.thread_id or "default_session_1"
    
    # FIX 1: Kept your cast to RunnableConfig
    config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    
    # FIX 2: Updated to include PILLAR 3 (search_query) and SOURCE tracking
    # We do NOT pass 'history' here so the Redis checkpointer can load it automatically.
    inputs = cast(GraphState, {
        "question": payload.question,
        "search_query": "",      # Pillar 3: Initialize search_query separately
        "iteration_count": 0,
        "response": "",  
        "context": [],
        "sources": [],           # Respecting your source logic
        "faithfulness_score": 0.0,
        "faithfulness_reason": ""
    })
    
    # FIX 3: Redis URI setup
    redis_uri = "redis://localhost:6379"
    
    async with AsyncRedisSaver.from_conn_string(redis_uri) as checkpointer:
        # Compile with the memory checkpointer
        agent_graph = agent_builder.compile(checkpointer=checkpointer)
        
        # Run it synchronously (Fast Synchronous Option)
        # This ensures you get the score in Postman
        final_state = await agent_graph.ainvoke(inputs, config) 
    
    # Respecting your final response structure exactly
    return {
        "answer": final_state.get("response", "I'm sorry, I couldn't find an answer."),
        "sources": final_state.get("sources", []),
        "iteration_count": final_state.get("iteration_count", 0),
        "faithfulness_score": final_state.get("faithfulness_score", 0.0),
        "faithfulness_reason": final_state.get("faithfulness_reason", "Evaluation skipped.")
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)