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
from opentelemetry import trace
from fastapi.responses import StreamingResponse
import json

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
    title="Phase 2: Agentic RAG (Background Eval)",
    description="Senior Software Engineer - AI Initiative",
    lifespan=lifespan
)

# --- 3. Background Task Helper ---
def run_background_eval(question: str, context: str, answer: str, thread_id: str):
    """Runs DeepEval as an independent trace, linked by Correlation ID"""
    tracer = trace.get_tracer(__name__)
    
    # Start a clean, independent span
    with tracer.start_as_current_span("DeepEval_Background_Check") as span:
        from app.evaluator import check_faithfulness
        print(f"\n🕵️‍♂️ [BACKGROUND] Starting DeepEval for thread: {thread_id}...")
        
        # 🚨 THE MAGIC: Tag this span with the Correlation ID
        span.set_attribute("session.thread_id", thread_id)
        span.set_attribute("question", question)
        
        score, reason = check_faithfulness(question, context, answer)
        
        span.set_attribute("evaluation.faithfulness_score", score)
        span.set_attribute("evaluation.reason", reason)
        
        print(f"✅ [BACKGROUND] Eval Complete! Score: {score}")

# --- 4. Endpoints ---
@app.post("/v2/ingest/file")
async def ingest_file(payload: StorePayload, background_tasks: BackgroundTasks):
    if not os.path.exists(payload.file_path):
        raise HTTPException(status_code=404, detail="File path not found.")
    
    try:
        count = process_file(payload.file_path, payload.metadata)
        return {"status": "success", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/v2/agent/chat")
async def chat_endpoint(payload: ChatPayload, background_tasks: BackgroundTasks): 
    from app.graph import agent_builder, GraphState
    
    thread_id = payload.thread_id or "default_session_1"
    config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    
    inputs = cast(GraphState, {
        "question": payload.question,
        "search_query": "",      
        "iteration_count": 0,
        "response": "",  
        "context": [],
        "sources": [],           
        "faithfulness_score": 0.0,
        "faithfulness_reason": ""
    })

    # 🚨 THE STREAMING GENERATOR 🚨
    async def stream_generator():
        redis_uri = "redis://localhost:6379"
        full_answer = ""
        final_state = None
        
        async with AsyncRedisSaver.from_conn_string(redis_uri) as checkpointer:
            agent_graph = agent_builder.compile(checkpointer=checkpointer)
            
            async for event in agent_graph.astream_events(inputs, config, version="v2"):
                
                # 🚨 THE FIX: Identify WHICH node is currently running
                current_node = event.get("metadata", {}).get("langgraph_node")
                
                # ONLY stream if the event is a stream AND the node is your generator
                # Note: Change "generate" to whatever you named your final node in graph.py!
                if event["event"] == "on_chat_model_stream" and current_node == "generate":
                    
                    chunk_obj = event["data"].get("chunk")
                    
                    if chunk_obj and hasattr(chunk_obj, "content"):
                        chunk_text = chunk_obj.content
                        
                        if chunk_text:
                            full_answer += chunk_text
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk_text})}\n\n"
            
            # Grab the final state after the stream finishes
            final_state = await agent_graph.aget_state(config)
            
        # The LLM is done talking. Now extract sources.
        context_str = "\n".join(final_state.values.get("context", []))
        sources = final_state.values.get("sources", [])
        
        # Send the final metadata (sources) to the frontend
        yield f"data: {json.dumps({'type': 'metadata', 'sources': sources})}\n\n"
        
        # Trigger DeepEval in the background NOW that the full answer is complete
        background_tasks.add_task(
            run_background_eval, 
            payload.question, 
            context_str, 
            full_answer,
            thread_id 
        )

    # Return the stream instead of a static JSON dictionary
    return StreamingResponse(stream_generator(), media_type="text/event-stream")
# 🚨 UPGRADED ENDPOINT FOR STREAMLIT 🚨
@app.get("/v2/agent/history/{thread_id}")
async def get_history(thread_id: str):
    """
    Time-travels through Redis to reconstruct the entire chat history 
    for a specific thread ID.
    """
    from app.graph import agent_builder
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    from langchain_core.runnables import RunnableConfig
    from typing import cast
    
    redis_uri = "redis://localhost:6379"
    config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    
    async with AsyncRedisSaver.from_conn_string(redis_uri) as checkpointer:
        graph = agent_builder.compile(checkpointer=checkpointer)
        
        full_chat_history = []
        seen_questions = set()
        
        # Time-travel backwards through all saved checkpoints
        async for snapshot in graph.aget_state_history(config):
            q = snapshot.values.get("question", "")
            r = snapshot.values.get("response", "")
            s = snapshot.values.get("sources", [])
            
            # If we found a valid pair we haven't processed yet
            if q and r and q not in seen_questions:
                seen_questions.add(q)
                # Insert at the beginning so the oldest messages stay at the top!
                full_chat_history.insert(0, {"role": "assistant", "content": r, "sources": s})
                full_chat_history.insert(0, {"role": "user", "content": q})
                
        return {"messages": full_chat_history}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)