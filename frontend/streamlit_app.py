import streamlit as st
import requests

# --- Configuration ---
FASTAPI_BASE_URL = "http://localhost:8080"
st.set_page_config(page_title="Phase 2: Agentic UI", layout="wide")

# --- Sidebar: Thread ID Selection ---
with st.sidebar:
    st.title("⚙️ Session Management")
    st.markdown("Change the ID below to switch conversations or start a new one.")
    
    # 🚨 THE THREAD SELECTOR 🚨
    selected_thread_id = st.text_input("Active Thread ID:", value="default_user_1")
    
    st.divider()
    
    st.title("📂 Document Ingestion")
    uploaded_file = st.text_input("Enter absolute Document path to ingest:")
    if st.button("Process Document"):
        with st.spinner("Chunking and Vectorizing..."):
            res = requests.post(f"{FASTAPI_BASE_URL}/v2/ingest/file", json={"file_path": uploaded_file})
            if res.status_code == 200:
                st.success("Document ingested into Qdrant!")
            else:
                st.error(f"Failed: {res.text}")

# --- State Management (Detecting Thread Changes) ---
# If the user types a new Thread ID, we must clear the screen and fetch new history
# --- State Management (Detecting Thread Changes) ---
if "current_thread_id" not in st.session_state or st.session_state.current_thread_id != selected_thread_id:
    st.session_state.current_thread_id = selected_thread_id
    st.session_state.messages = []
    
    # Fetch FULL history from FastAPI/Redis
    try:
        history_res = requests.get(f"{FASTAPI_BASE_URL}/v2/agent/history/{selected_thread_id}").json()
        
        # 🚨 THE FIX: Load the entire array of messages at once!
        if "messages" in history_res and history_res["messages"]:
            st.session_state.messages = history_res["messages"]
            
    except Exception as e:
        st.warning("Could not connect to backend to fetch history.")

# --- Main Chat UI ---
st.title(f"🤖 RAG Agent (Session: `{selected_thread_id}`)")

# Draw the chat history on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Show assistant message (thinking -> responding)
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            # Call the FastAPI endpoint with the active Thread ID
            payload = {
                "question": prompt,
                "thread_id": selected_thread_id
            }
            response = requests.post(f"{FASTAPI_BASE_URL}/v2/agent/chat", json=payload).json()
            
            answer = response.get("answer", "Error generating response.")
            st.markdown(answer)
            
            # Save to local UI state
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Handle the Background Eval UI
            if response.get("faithfulness_score") == -1.0:
                with st.expander("DeepEval Status", expanded=False):
                    st.info("Validation is running in the background. Check Arize Phoenix for the final faithfulness score.")