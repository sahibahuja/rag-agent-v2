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
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
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
def run_background_eval(question: str, context: str, answer: str, trace_id: int, span_id: int):
    """Runs DeepEval and mathematically forces the Trace ID using a Phantom Parent"""
    tracer = trace.get_tracer(__name__)
    
    # 🚨 THE ULTIMATE FIX: Reconstruct the dead parent mathematically
    phantom_parent_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,           # Tells OTEL "Trust me, the parent is elsewhere"
        trace_flags=TraceFlags(1) # Ensures the trace is recorded
    )
    
    # Wrap it in a context
    ctx = trace.set_span_in_context(NonRecordingSpan(phantom_parent_context))
    
    # Start the span forcing that exact context
    with tracer.start_as_current_span("DeepEval_Background_Check", context=ctx) as span:
        from app.evaluator import check_faithfulness
        print(f"\n🕵️‍♂️ [BACKGROUND] Starting DeepEval with forced Trace ID...")
        
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
async def chat_endpoint(payload: ChatPayload, background_tasks: BackgroundTasks): # <--- Added BackgroundTasks
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
    
    redis_uri = "redis://localhost:6379"
    async with AsyncRedisSaver.from_conn_string(redis_uri) as checkpointer:
        agent_graph = agent_builder.compile(checkpointer=checkpointer)
        
        # 1. Run the graph (this is now fast!)
        final_state = await agent_graph.ainvoke(inputs, config) 
    
    # 2. Extract results
    answer = final_state.get("response", "I'm sorry, I couldn't find an answer.")
    context_str = "\n".join(final_state.get("context", []))
    sources = final_state.get("sources", [])
    
   # 3a. Extract the raw mathematical IDs before FastAPI ends the trace!
    current_span_context = trace.get_current_span().get_span_context()
    raw_trace_id = current_span_context.trace_id
    raw_span_id = current_span_context.span_id
    
    # 3b. Pass the integers, not the object!
    background_tasks.add_task(
        run_background_eval, 
        payload.question, 
        context_str, 
        answer,
        raw_trace_id,
        raw_span_id
    )
    
    # 4. Return to user immediately!
    return {
        "answer": answer,
        "sources": sources,
        "iteration_count": final_state.get("iteration_count", 0),
        "faithfulness_score": -1.0, # Float to pass Pydantic validation
        "faithfulness_reason": "Evaluation is running in the background. Check server logs."
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)