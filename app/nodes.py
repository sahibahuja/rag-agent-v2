from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm import llm
from app.graph import GraphState
from app.engine import get_context_from_qdrant
from typing import cast
from typing import Any

from app.evaluator import check_faithfulness

def validate_answer(state: GraphState):
    print("--- NODE: VALIDATING WITH DEEPEVAL ---")
    
    question = state.get("question", "")
    # SYNCED: Now matches what generate_answer returns
    answer = state.get("response", "") 
    
    # SYNCED: Using 'context' as that's what retrieve_docs provides
    context_list = state.get("context", [])
    context = "\n".join(context_list) if context_list else "No context"
    if not answer or answer.strip() == "":
        return {"faithfulness_score": 0.0, "faithfulness_reason": "No answer generated to validate."}
    # Run DeepEval
    score, reason = check_faithfulness(question, context, answer)
    
    print(f"⭐ DeepEval Faithfulness Score: {score}")
    print(f"📝 Reason: {reason}")
    
    return {
        "faithfulness_score": score,
        "faithfulness_reason": reason
    }

# --- 1. The Rewriter Node ---
def rewrite_query(state: GraphState):
    """
    Optimizes the user's question for Vector Search and increments the loop counter.
    """
    print("--- NODE: REWRITING QUERY ---")
    question = state.get("question", "")
    
    # 1. IMPORTANT: Get and increment the current count
    current_count = state.get("iteration_count", 0)
    new_count = current_count + 1
    
    # 2. Optimize the query
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a search optimizer. Convert the user's input into a "
                   "single, high-quality standalone search query. "
                   "Output ONLY the query string, no preamble."),
        ("human", "{question}")
    ])
    
    # Using the pipe is fine here as long as we treat the output as a string
    chain = prompt | llm | StrOutputParser()
    new_query = chain.invoke({"question": question})
    
    # 3. Return BOTH the new query and the incremented count
    # This ensures the GraphState is updated so decide_to_generate can stop the loop
    return {
        "search_query": str(new_query).strip(),
        "iteration_count": new_count
    }
def retrieve_docs(state: GraphState):
    print("--- NODE: RETRIEVING FROM QDRANT ---")
    search_query = state.get("search_query", state.get("question", ""))
    
    # We skip the 'Chain' (|) and Parser for this specific node to kill the error
    system_msg = "Generate 2 alternative search queries. Output one per line."
    human_msg = f"Query: {search_query}"
    
    # Call LLM directly - this returns a BaseMessage object
    response = llm.invoke([
        ("system", system_msg),
        ("human", human_msg)
    ])
    
    # Force cast the content to string and perform the split
    # Since response.content is explicitly a string/list union, 
    # we force it to string to stop Pylance from guessing.
    raw_text = str(response.content)
    
    # Process the lines
    queries = [search_query]
    for line in raw_text.split("\n"):
        clean_line = line.strip()
        if clean_line:
            queries.append(clean_line)
    
    # Pass to your engine
    context_str, sources = get_context_from_qdrant(queries, limit=3)
    
    return {"context": [context_str], "sources": sources}
# --- 3. The Grader Node ---
def grade_documents(state: GraphState):
    print("--- NODE: GRADING RELEVANCE ---")
    context_list = state.get("context", [])
    context = context_list[0] if context_list else ""
    question = state.get("question", "")

    # Structured prompt to force 1-word output
    system_msg = (
        "You are a binary classifier. "
        "Rules:\n1. If the context helps answer the question, say 'YES'.\n"
        "2. If not, say 'NO'.\n"
        "3. DO NOT EXPLAIN. DO NOT SAY 'I CAN HELP'. ONLY SAY 'YES' OR 'NO'."
    )
    
    response = llm.invoke([
        ("system", system_msg),
        ("human", f"Context: {context}\n\nQuestion: {question}\n\nResult:")
    ])
    
    raw_grade = str(response.content).upper().strip()
    
    # Logic check: if 'YES' is anywhere, it's a 'yes'
    grade = "yes" if "YES" in raw_grade else "no"
    
    print(f"--- GRADER RESULT: {grade} (Raw: {raw_grade[:30]}) ---")
    return {"is_relevant": grade}
# --- 4. The Generator Node ---
def generate_answer(state: GraphState):
    print("--- NODE: GENERATING ANSWER ---")
    context_list = state.get("context", [])
    context = "\n\n".join(context_list) if context_list else "No context found."
    question = state.get("question", "")
    sources = state.get("sources", [])

    system_msg = (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the context doesn't have the answer, say you don't know."
    )
    
    response = llm.invoke([
        ("system", system_msg),
        ("human", f"Context: {context}\n\nQuestion: {question}")
    ])

    # KEY CHANGE: Returning 'response' to match validate_answer
    return {
        "response": str(response.content), 
        "sources": sources
    }
