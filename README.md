# 🤖 Phase 2: Agentic RAG Pipeline (Production-Grade)

A highly optimized Agentic Retrieval-Augmented Generation (RAG) system utilizing **LangGraph** for state-machine orchestration, **Redis** for persistent memory, and **DeepEval** for asynchronous faithfulness validation.

Powered by **Docling** (multi-format document ingestion), **Qdrant** (vector storage), **Arize Phoenix** (observability), **RedisInsight** (memory UI), and **Streamlit** (interactive frontend).

---

## ✨ Phase 2: Core Features & Implementations

This phase transitioned the system from a basic "query-and-retrieve" script into a resilient, distributed agentic architecture. 

* **Intent Routing (The Express Lane):** The agent dynamically classifies the user's prompt. If the user asks a conversational question (e.g., "What did I just ask you?"), it bypasses the vector database entirely to save time and compute.
* **Structured Binary Grading:** Replaced fragile prompt-based grading with strict **Pydantic Structured Outputs**. A local Llama 3.1 8B model grades the retrieved context as exactly `yes` or `no`, eliminating JSON parsing crashes.
* **State-Separated Rewriting:** If context is graded `no`, the agent rewrites the query to try again. Crucially, it stores the new query in a `search_query` state variable while leaving the user's original `question` untouched, ensuring the final answer addresses the exact original request.
* **Speed-Capped Loops (Anti-Death Loop):** The graph is hard-coded to allow a maximum of 1 rewrite iteration. If it still fails to find relevant context, it falls back gracefully rather than looping infinitely.
* **Asynchronous LLM-as-a-Judge (DeepEval):** The user receives their answer in ~5 seconds. Meanwhile, a decoupled FastAPI `BackgroundTask` runs DeepEval's `FaithfulnessMetric` to score the output for hallucinations. 
* **W3C Distributed Tracing:** Using OpenTelemetry context propagation, the background DeepEval task mathematically inherits the original request's `trace_id`. In Arize Phoenix, the background validation appears in the exact same waterfall trace as the user's original chat.
* **Time-Travel Memory (Redis):** Integrated `AsyncRedisSaver`. The Streamlit UI can query the backend to walk backward through Redis checkpoints and perfectly reconstruct a user's entire chat history when they switch `thread_id`s.

---

## 🏗️ The Agentic Graph Flow

1. **User Input:** Prompt enters via Streamlit UI.
2. **Router Node:** Decides `vector_store` vs. `chat_history`.
3. **Retrieval Node:** Generates alternative keywords and fetches top `k=3` chunks from Qdrant.
4. **Grader Node:** Pydantic validation confirms if chunks are relevant.
5. **Rewriter Node (Conditional):** Modifies search terms if grading fails (max 1 loop).
6. **Generator Node:** Blends retrieved context + Redis chat history -> Returns Answer.
7. **Background Worker:** Runs DeepEval against the generated answer and logs to Phoenix.

---

## 🛠️ Prerequisites

* **Python:** 3.10+
* **Ollama:** Installed locally (Running `llama3.1:8b`).
* **Docker & Docker Compose:** For running Qdrant, Redis, RedisInsight, and Arize Phoenix.

---

## 🚀 Setup & Installation

### 1. Start Infrastructure
```bash
docker-compose up -d
Qdrant: http://localhost:6333 (Vector DB)

Arize Phoenix: http://localhost:6006 (Tracing)

RedisInsight: http://localhost:8001 (Memory UI)

2. Pull Local LLM
This project specifically requires the 8B parameter model for reliable tool-calling and structured JSON schema adherence.

Bash
ollama pull llama3.1:8b
3. Install Requirements
Ensure all dependencies are installed.

Bash
pip install -r requirements.txt
🏃 Running the Application
This project operates with a decoupled backend and frontend. You will need two terminal windows.

Terminal 1: Start the Backend (FastAPI)

Bash
python app/main.py
API Base URL: http://localhost:8080

Swagger UI: http://localhost:8080/docs

Terminal 2: Start the Frontend (Streamlit)

Bash
streamlit run frontend/streamlit_app.py
Interactive UI: http://localhost:8501

🧪 API Endpoints & Usage
1. Document Ingestion
Uses Docling to parse and chunk multi-format files (PDF, Word, PPT, HTML, Markdown) into Qdrant.
POST /v2/ingest/file

JSON
{
    "file_path": "C:/absolute/path/to/your/document.pdf",
    "metadata": { "category": "manual" }
}
2. Agentic Chat (Async Eval)
POST /v2/agent/chat

JSON
{
    "question": "What are the key takeaways from the document?",
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
3. Time-Travel Chat History
GET /v2/agent/history/{thread_id}
Used by the Streamlit frontend to read Redis memory and reconstruct the exact chat history when switching active threads.

📊 Observability & Memory Management
Arize Phoenix (Tracing)
View the full lifecycle of a request, including the Background DeepEval Check. The system uses W3C Trace Context propagation to ensure that even though the evaluation runs later, it is linked to the original user's trace_id.

RedisInsight (Memory UI)
Navigate to http://localhost:8001 to visualize the chat history.

Keys: Stored as checkpoint_thread_id:<your_id>.

Management: You can delete keys in RedisInsight to manually reset the agent's memory for a specific user without restarting the server.

📂 Project Structure (The Monorepo)
app/main.py: FastAPI entry point with BackgroundTasks, Time-Travel history endpoints, and OTEL context injection.

app/schemas.py: Pydantic models with Literal types for bulletproof routing.

app/nodes.py: Graph logic (Routing, Retrieval, Grading, Rewriting, Generation).

app/evaluator.py: Optimized DeepEval wrapper for Llama 3.1 8B (no Pydantic cage to allow native 'verdicts').

app/graph.py: The StateGraph topology (Ghost nodes removed for maximum speed).

app/engine.py: Docling ingestion logic and fast embedding configurations.

app/llm.py: Centralized Ollama configuration with format="json".

frontend/streamlit_app.py: The interactive chat UI with state-synced thread management.