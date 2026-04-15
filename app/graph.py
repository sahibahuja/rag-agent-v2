from langgraph.graph import StateGraph, END
from app.schemas import GraphState
from app.nodes import rewrite_query, retrieve_docs, grade_documents, generate_answer, validate_answer

# 1. Define the Router (Conditional Logic)
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

# 2. Initialize the Graph Builder
builder = StateGraph(GraphState)

# 3. Add Nodes
builder.add_node("rewrite", rewrite_query)
builder.add_node("retrieve", retrieve_docs)
builder.add_node("grade", grade_documents)
builder.add_node("generate", generate_answer)
builder.add_node("validate", validate_answer)

# 4. Define the Flow
builder.set_entry_point("retrieve") # Start by searching
builder.add_edge("retrieve", "grade")

# The "Brain" of the Agent: Conditional branching
builder.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", "validate") 
builder.add_edge("validate", END)

# EXPORT THE BUILDER 
# (We DO NOT compile it here. We compile it in main.py with the Redis Checkpointer attached!)
agent_builder = builder