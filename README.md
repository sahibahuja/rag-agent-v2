Markdown
# 🤖 Phase 2: Agentic RAG Pipeline (Intent-Driven LangGraph)

A production-grade, highly optimized Agentic Retrieval-Augmented Generation (RAG) system. This Phase 2 architecture moves beyond basic RAG by utilizing an advanced state-machine workflow (LangGraph) with **Intent Routing**, **Structured Pydantic Outputs**, and **Redis Memory Checkpointing** to deliver self-correcting and validated answers.

Powered by Docling (ingestion), Qdrant (vector storage), Arize Phoenix (observability), and **DeepEval** (LLM-as-a-Judge validation).

---

## 🏗️ Architecture & Workflow (The 4-Pillar Design)

When modifying this codebase, keep these 4 architectural pillars in mind. They were designed to prevent "Death Loops", "Amnesia", and JSON parsing crashes.

### The Agentic Graph Flow
1. **Intent Routing (The Express Lane):** A classification node intercepts the user's prompt. 
   * If it's a question about the document -> Routed to Vector Store. 
   * If it's a question about the conversation history -> Bypasses search entirely for instant generation.
2. **Multi-Query Expansion & Retrieval:** The user's prompt is rewritten to generate alternative keywords. Chunks are fetched from Qdrant (capped at `k=3` to prevent token bloat).
3. **Structured Binary Grading:** A local LLM strictly grades the retrieved context as `yes` (relevant) or `no` (irrelevant) using Pydantic validation.
4. **Iterative Rewriting (State Separated):** If graded `no`, the agent rewrites the query. **Crucial:** It stores this in `search_query` while leaving the original `question` pure, ensuring the AI remembers exactly what the user asked. Loops are capped at 2 iterations.
5. **Context-Aware Generation:** An answer is generated using a blend of the retrieved context and the persistent Redis chat history.
6. **DeepEval Validation:** The final output is evaluated by DeepEval's `FaithfulnessMetric` (wrapped in a Pydantic schema) to ensure the LLM didn't hallucinate outside the retrieved context.

---

## 🛠️ Prerequisites

* **Python:** 3.10+
* **Ollama:** Installed locally.
* **Docker & Docker Compose:** For running Qdrant, Redis, and Arize Phoenix.

---

## 🚀 Setup & Installation

### 1. Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd rag-agent-v2

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2. Start Infrastructure
Ensure Docker is running, then spin up the vector database, Redis memory checkpointer, and tracing server:

Bash
docker-compose up -d
Qdrant: http://localhost:6333

Arize Phoenix: http://localhost:6006

Redis Checkpointer: redis://localhost:6379

3. Pull Local LLM
Design Note: This project specifically requires the 8-Billion parameter model (llama3.1:8b). Smaller 3B models lack the native tool-calling capabilities required to output stable Pydantic JSON schemas, which will cause pipeline crashes.

Bash
ollama pull llama3.1:8b
🏃 Running the Server
Start the FastAPI application. The server runs on port 8080 to avoid conflicts with system-level processes or Docker socket leaks.

Bash
python app/main.py
API Base URL: http://localhost:8080

Swagger UI: http://localhost:8080/docs

🧪 API Endpoints & Usage
1. Document Ingestion
Uses Docling to chunk and index Documents into Qdrant. Large files are handled safely via the engine.py chunking logic.

POST /v2/ingest/file

JSON
{
    "file_path": "C:/absolute/path/to/your/document.pdf",
    "metadata": {
        "category": "manual",
        "version": "1.0"
    }
}
2. Agentic Chat
Triggers the LangGraph state machine. Note: Chat history is NOT passed in the payload; it is managed automatically via the Redis Checkpointer using the thread_id.

POST /v2/agent/chat

JSON
{
    "question": "What is the recommended tire pressure?",
    "thread_id": "user_session_123" 
}
Example Response:

JSON
{
    "answer": "The recommended tire pressure is 32 PSI.",
    "sources": ["C:/documents/manual.pdf"],
    "iteration_count": 1,
    "faithfulness_score": 1.0,
    "faithfulness_reason": "The generated answer directly matches the specifications listed in the provided context."
}
Note: A faithfulness_score of 1.0 means the agent did not hallucinate. A score of 0.0 indicates a failure to validate.

📊 Observability (Tracing)
Every execution of the LangGraph agent is automatically traced using Arize Phoenix.

Open your browser to http://localhost:6006.

Navigate to the Traces tab.

View the Waterfall Trace to inspect the inputs, outputs, structured JSON schema validations, token counts, and latency of individual nodes.

📂 Project Structure (Developer Guide)
When maintaining this code, here is where everything lives:

app/schemas.py: The Data Contracts. Contains all Pydantic models (RouteQuery, GradeSchema, FaithfulnessSchema) that force the LLM to output clean JSON. Also contains the LangGraph GraphState.

app/graph.py: The Topological Wiring. Defines the nodes, the conditional entry point (Router), and the edges that control the retry loops.

app/nodes.py: The Business Logic. Contains the actual Python functions executed at each node in the graph (Retrieval, Grading, Rewriting, Generation).

app/evaluator.py: DeepEval Integration. Wraps the local ChatOllama model so DeepEval can use it as an LLM-as-a-Judge for calculating faithfulness metrics.

app/main.py: FastAPI setup, Redis Checkpointer initialization, and API endpoint definitions.

app/engine.py: Docling ingestion logic and Qdrant integration.

app/llm.py: Centralized model configuration (ensure this points to llama3.1:8b with format="json").

app/database.py: Qdrant client connection setup.

app/observability.py: OpenTelemetry setup mapping to Arize Phoenix.