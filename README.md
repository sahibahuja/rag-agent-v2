🤖 Phase 2: Agentic RAG Pipeline (LangGraph + DeepEval)
A production-grade Agentic Retrieval-Augmented Generation (RAG) system. This Phase 2 architecture utilizes a state-machine workflow (LangGraph) to self-correct and validate its own answers before returning them to the user.

Powered by Docling (ingestion), Qdrant (vector storage), Arize Phoenix (observability), and DeepEval (LLM-as-a-Judge validation).

🏗️ Architecture & Workflow
Unlike standard RAG pipelines that simply retrieve and answer, this agent utilizes a "Self-Correcting" loop:

Multi-Query Expansion: The user's prompt is rewritten into multiple search queries to cast a wider semantic net.

Vector Retrieval: Chunks are fetched from Qdrant.

Binary Grading: A local LLM (Llama 3.2) acts as a bouncer, grading the retrieved context as YES (relevant) or NO (irrelevant).

Iterative Rewriting: If graded NO, the agent rewrites the search query and tries again, looping until it finds good context (or hits a maximum iteration safety limit).

Generation: An answer is generated strictly using the approved context.

Validation (DeepEval): An independent "Judge" LLM evaluates the final output against the retrieved text, calculating a faithfulness_score to prevent hallucinations.

🛠️ Prerequisites
Python: 3.10+

Ollama: Installed locally.

Docker & Docker Compose: For running Qdrant and Arize Phoenix.

🚀 Setup & Installation
1. Environment Setup
Bash
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
Ensure Docker is running, then spin up the vector database and tracing server:

Bash
docker-compose up -d
Qdrant: http://localhost:6333

Arize Phoenix: http://localhost:6006

3. Pull Local LLM
This project is configured to use the 3-billion parameter version of Llama 3.2 for both generation and evaluation.

Bash
ollama pull llama3.2:3b
🏃 Running the Server
Start the FastAPI application. Note: The server runs on port 8080 to avoid conflicts with system-level processes or Docker socket leaks.

Bash
python app/main.py
API Base URL: http://localhost:8080

Swagger UI: http://localhost:8080/docs

🧪 API Endpoints & Usage
1. Document Ingestion
Uses Docling to chunk and index PDFs into Qdrant. Runs as a background task for large files.

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
Triggers the LangGraph state machine.

POST /v2/agent/chat

JSON
{
    "question": "What is the recommended tire pressure?",
    "history": [],
    "thread_id": "session_123" 
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

View the Waterfall Trace to inspect the inputs, outputs, token counts, and latency of individual nodes (retrieve, grade, rewrite, generate, validate).

📂 Project Structure
app/main.py: FastAPI setup, lifespan events, and API route definitions.

app/schemas.py: Pydantic data contracts (ChatPayload, StorePayload) and the LangGraph GraphState TypedDict.

app/graph.py: Compiles the StateGraph, defining nodes and conditional edges.

app/nodes.py: The core business logic for the agent (Retrieval, Grading, Rewriting, Generation, Validation).

app/evaluator.py: DeepEval integration wrapping ChatOllama for LLM-as-a-Judge metrics.

app/engine.py: Docling ingestion logic and Qdrant embedding integration.

app/database.py: Qdrant client initialization.

app/observability.py: OpenTelemetry setup for Arize Phoenix.