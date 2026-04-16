from typing import List, cast, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm import llm
from app.graph import GraphState
from app.engine import get_context_from_qdrant
from app.schemas import RouteQuery, GradeSchema
from app.evaluator import check_faithfulness

# --- 1. The Router Node (PILLAR 1) ---
def route_question(state: GraphState):
    print("--- NODE: ROUTING QUESTION ---")
    question = state.get("question", "")
    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    response = cast(RouteQuery, structured_llm_router.invoke([
        ("system", "Route the user query. Use 'vector_store' for Document questions and 'chat_history' for greetings or memory."),
        ("human", question)
    ]))
    
    print(f"DEBUG: Routing to -> {response.datasource}")
    return response.datasource

# --- 2. The Retrieval Node (PILLAR 4 Optimized + SOURCES) ---
def retrieve_docs(state: GraphState):
    print("--- NODE: RETRIEVING FROM QDRANT ---")
    search_query = state.get("search_query") or state.get("question", "")
    
    # Pillar 4: Generate 1 alternative query to save time
    system_msg = "Generate 1 alternative search query. Output ONLY the query text."
    response = llm.invoke([
        ("system", system_msg),
        ("human", f"Query: {search_query}")
    ])
    
    queries = [search_query, str(response.content).strip()]
    
    # 🚨 KEPT: Your source-tracking engine call
    context_str, sources = get_context_from_qdrant(queries, limit=3)
    
    return {"context": [context_str], "sources": sources}

# --- 3. The Grader Node (PILLAR 2 Structured) ---
def grade_documents(state: GraphState):
    print("--- NODE: GRADING RELEVANCE ---")
    structured_grader = llm.with_structured_output(GradeSchema)
    
    context = "\n\n".join(state.get("context", []))
    question = state.get("question", "")

    response = cast(GradeSchema, structured_grader.invoke([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question. Answer 'yes' or 'no'."),
        ("human", f"Context: {context}\n\nQuestion: {question}")
    ]))
    
    grade = response.binary_score.lower().strip()
    print(f"--- GRADER RESULT: {grade} ---")
    
    return {"is_relevant": grade}

# --- 4. The Decision Node (PILLAR 4 Speed Cap) ---
def decide_to_generate(state: GraphState):
    print("--- DECIDING NEXT STEP ---")
    if state.get("is_relevant") == "yes" or state.get("iteration_count", 0) >= 2:
        return "generate"
    return "rewrite"

# --- 5. The Rewriter Node ---
def rewrite_query(state: GraphState):
    print("--- NODE: REWRITING QUERY ---")
    question = state.get("question", "")
    history = state.get("history", [])
    current_count = state.get("iteration_count", 0)
    
    history_str = "\n".join(history) if history else "No previous conversation."
    system_msg = "You are a search optimizer. Output ONLY optimized search keywords based on history."
    
    response = llm.invoke([
        ("system", system_msg),
        ("human", f"History: {history_str}\n\nQuestion: {question}")
    ])

    clean_query = str(response.content).strip().split('\n')[-1].replace('"', '')
    return {
        "search_query": clean_query, 
        "iteration_count": current_count + 1
    }

# --- 6. The Generator Node (SOURCES RESTORED) ---
def generate_answer(state: GraphState):
    print("--- NODE: GENERATING ANSWER ---")
    context_list = state.get("context", [])
    context = "\n\n".join(context_list) if context_list else "No context found."
    question = state.get("question", "")
    
    # 🚨 KEPT: Pulling sources from the state
    sources = state.get("sources", [])
    
    history = state.get("history", [])
    history_str = "\n".join(history) if history else "No previous conversation."

    system_msg = (
        "You are a helpful assistant. "
        "Use CONTEXT for document facts and HISTORY for conversation questions."
    )
    
    response = llm.invoke([
        ("system", system_msg),
        ("human", f"History:\n{history_str}\n\nContext:\n{context}\n\nQuestion: {question}")
    ])

    # 🚨 KEPT: Returning 'sources' in the final payload
    return {
        "response": str(response.content), 
        "sources": sources,
        "history": [f"User: {question}", f"AI: {response.content}"]
    }

# --- 7. The Validator Node ---
def validate_answer(state: GraphState):
    print("--- NODE: VALIDATING ---")
    score, reason = check_faithfulness(
        state.get("question", ""),
        "\n".join(state.get("context", [])),
        state.get("response", "")
    )
    return {"faithfulness_score": score, "faithfulness_reason": reason}