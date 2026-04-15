import uvicorn
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from app.schemas import StorePayload, ChatPayload, ChatResponse
from app.database import init_db
from app.engine import process_file
from app.observability import setup_tracing
# We will import the graph 'app' once we've finalized nodes.py
# from app.graph import app as agent_app 

# --- 1. Lifespan Orchestration ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the Knowledge Base
    print("🚀 Phase 2 Agent: Initializing Qdrant Database...")
    try:
        setup_tracing()
        init_db()
        print("✅ Qdrant and FastEmbed ready.")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
    
    # Optional: Start Phoenix Tracing here if you want it automated
    # from app.observability import setup_tracing
    # setup_tracing()

    yield  # Application is serving requests
    
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
    """
    Uses the 10-page window logic from engine.py to index PDFs.
    Using BackgroundTasks for larger files to prevent timeouts.
    """
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail="File path not found.")
    
    try:
        # For very large docs, we run this as a background task
        # so the user gets an immediate 'Processing' response
        count = process_file(payload.file_path, payload.metadata)
        return {"status": "success", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/v2/agent/chat")
async def chat_endpoint(payload: ChatPayload):
    from app.graph import app as agent_graph,GraphState # Import the compiled graph
    
    inputs = {
        "question": payload.question,
        "iteration_count": 0,
        "response": "",  # Initialize empty strings for safety
        "context": []
    }
    from typing import cast
    # This actually runs the LangGraph nodes you've been editing
    final_state = await agent_graph.ainvoke(cast(GraphState, inputs)) 
    
    return {
        # Mapping 'response' from Graph to 'answer' in API
        "answer": final_state.get("response", "I'm sorry, I couldn't find an answer."),
        "sources": final_state.get("sources", []),
        "iteration_count": final_state.get("iteration_count", 0),
        "faithfulness_score": final_state.get("faithfulness_score", 0.0),
        "faithfulness_reason": final_state.get("faithfulness_reason", "Evaluation skipped.")
        }
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)