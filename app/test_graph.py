from app.graph import app, GraphState
from app.database import init_db

def run_test():
    """
    Validation script for Phase 2 Agentic RAG.
    Ensures DB initialization, proper typing, and graph flow.
    """
    # 1. Initialize DB to prevent 404 Collection Not Found errors
    print("🚀 Initializing Qdrant and FastEmbed...")
    init_db()

    # 2. Define inputs with explicit GraphState type to satisfy Pylance
    inputs: GraphState = {
        "question": "What is the main objective discussed in the uploaded documents?",
        "iteration_count": 0
    }

    print("\n--- Starting Graph Execution ---")
    try:
        # 3. Stream the execution to watch node transitions in the terminal
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"✅ Node '{key}' finished.")
                
                # Print the final answer if we reached the generator
                if key == "generate" and "answer" in value:
                    print(f"\n🤖 Agent Answer:\n{value['answer']}\n")
                    
    except Exception as e:
        print(f"❌ Graph Execution Error: {e}")

if __name__ == "__main__":
    run_test()