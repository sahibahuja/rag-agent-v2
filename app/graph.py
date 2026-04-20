from langgraph.graph import StateGraph, END
from app.schemas import GraphState
from app.nodes import (
    rewrite_query, 
    retrieve_docs, 
    grade_documents, 
    generate_answer, 
    route_question,  # 🚨 ADDED: Import the router!
    decide_to_generate
)


builder = StateGraph(GraphState)

# Add Nodes (Removed 'validate')
builder.add_node("rewrite", rewrite_query)
builder.add_node("retrieve", retrieve_docs)
builder.add_node("grade", grade_documents)
builder.add_node("generate", generate_answer)

# Define the Flow
builder.set_conditional_entry_point(
    route_question,
    {
        "vector_store": "retrieve",  
        "chat_history": "generate"   
    }
)

builder.add_edge("retrieve", "grade")

builder.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

builder.add_edge("rewrite", "retrieve")

# 🚨 THE FIX: Generate goes straight to END now!
builder.add_edge("generate", END) 

agent_builder = builder