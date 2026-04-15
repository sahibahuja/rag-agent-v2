from langgraph.graph import StateGraph, END
from app.schemas import GraphState
from nodes import rewrite_query, retrieve_docs, grade_documents, generate_answer, validate_answer

# 3. Define the Router (Conditional Logic)
def decide_to_generate(state: GraphState):
    print("--- DECIDING NEXT STEP ---")
    relevance = state.get("is_relevant", "no")
    count = state.get("iteration_count", 0)
    
    print(f"DEBUG: Current iteration: {count}, Relevance: {relevance}")

    # FORCE STOP: If we've tried 3 times, we MUST generate even if it's 'no'
    if relevance == "yes" or count >= 3:
        print("--- DECISION: GENERATE (Limit reached or relevant) ---")
        return "generate"
    else:
        print("--- DECISION: REWRITE ---")
        return "rewrite"

# 4. Build the Graph
def create_graph():
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("validate", validate_answer)
    # Define the Flow
    workflow.set_entry_point("retrieve") # Start by searching

    workflow.add_edge("retrieve", "grade")
    
    # The "Brain" of the Agent: Conditional branching
    workflow.add_conditional_edges(
        "grade",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", "validate") 
    workflow.add_edge("validate", END)

    return workflow.compile()

# Final Compiled App
app = create_graph()