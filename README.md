🤖 Phase 2: Agentic RAG Pipeline (Production-Grade)
A highly optimized Agentic Retrieval-Augmented Generation (RAG) system utilizing LangGraph for state-machine orchestration, Redis for persistent memory, and DeepEval for asynchronous faithfulness validation.

Powered by Docling (ingestion), Qdrant (vector storage), Arize Phoenix (observability), and RedisInsight (memory UI).

🏗️ Architecture & Workflow (The 4-Pillar Design)
This architecture is designed for low-latency user experience while maintaining high-quality validation.

The Agentic Graph Flow
Intent Routing (The Express Lane): A classification node intercepts the user's prompt.

vector_store: Routed to PDF search.

chat_history: Bypasses search entirely for instant conversational responses.

Multi-Query Expansion & Retrieval: The original question is preserved while a specialized search_query is generated to fetch chunks from Qdrant.

Structured Binary Grading: A local Llama 3.1 8B model grades the retrieved context as yes or no using strict Pydantic validation.

Speed-Capped Loops: If context is irrelevant, the agent rewrites the query. To prevent "death loops" and high latency, the system is strictly limited to 1 rewrite iteration.

Context-Aware Generation: Final answers are generated using the intersection of retrieved PDF context and persistent Redis chat history.

The Validation Pillar (Option B: Async)
To prevent the user from waiting 2+ minutes, the DeepEval Faithfulness Check is decoupled from the main request.

The API returns an answer to the user in ~5-10 seconds.

A FastAPI Background Task triggers DeepEval.

Distributed Tracing: The background evaluation is mathematically linked to the original user request via OpenTelemetry context propagation, appearing in the same Arize Phoenix session.

🛠️ Prerequisites
Python: 3.10+

Ollama: Installed locally (Running llama3.1:8b).

Docker & Docker Compose: For running Qdrant, Redis, RedisInsight, and Arize Phoenix.

🚀 Setup & Installation
1. Start Infrastructure
Bash
docker-compose up -d
Qdrant: http://localhost:6333 (Vector DB)

Arize Phoenix: http://localhost:6006 (Tracing)

RedisInsight: http://localhost:8001 (Memory UI)

2. Pull Local LLM
This project requires the 8B parameter model for reliable tool-calling and structured JSON output.

Bash
ollama pull llama3.1:8b
🏃 Running the Server
Bash
python app/main.py
API Base URL: http://localhost:8080

Swagger UI: http://localhost:8080/docs

🧪 API Endpoints & Usage
1. Agentic Chat (Async Eval)
POST /v2/agent/chat

JSON
{
    "question": "What is the document's take on renewable energy?",
    "thread_id": "user_session_99" 
}
Response (Immediate):

JSON
{
    "answer": "The document states...",
    "sources": ["..."],
    "iteration_count": 0,
    "faithfulness_score": -1.0,
    "faithfulness_reason": "Evaluation is running in the background. Check server logs/Phoenix."
}
Note: A score of -1.0 indicates the background worker is currently calculating the metric.

📊 Observability & Memory Management
Arize Phoenix (Tracing)
View the full lifecycle of a request, including the Background DeepEval Check. The system uses W3C Trace Context propagation to ensure that even though the evaluation runs later, it is linked to the original user's trace_id.

RedisInsight (Memory UI)
Navigate to http://localhost:8001 to visualize the chat history.

Keys: Stored as checkpoint_thread_id:<your_id>.

Management: You can delete keys in RedisInsight to manually reset the agent's memory for a specific user.

📂 Project Structure
app/main.py: FastAPI entry point with BackgroundTasks and OTEL context injection.

app/schemas.py: Pydantic models with Literal types for bulletproof routing.

app/nodes.py: Graph logic (Routing, Retrieval, Grading, Rewriting, Generation).

app/evaluator.py: Optimized DeepEval wrapper for Llama 3.1 8B (no Pydantic cage to allow native 'verdicts').

app/graph.py: The StateGraph topology (Ghost nodes removed for maximum speed).

app/llm.py: Centralized Ollama configuration with format="json".