import os  # 🚨 Moved to the top!
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
if "current_thread_id" not in st.session_state or st.session_state.current_thread_id != selected_thread_id:
    st.session_state.current_thread_id = selected_thread_id
    st.session_state.messages = []
    
    # Fetch FULL history from FastAPI/Redis
    try:
        history_res = requests.get(f"{FASTAPI_BASE_URL}/v2/agent/history/{selected_thread_id}").json()
        
        # Load the entire array of messages at once!
        if "messages" in history_res and history_res["messages"]:
            st.session_state.messages = history_res["messages"]
            
    except Exception as e:
        st.warning("Could not connect to backend to fetch history.")

# --- Main Chat UI ---
st.title(f"🤖 RAG Agent (Session: `{selected_thread_id}`)")

# Draw the chat history on the screen
# 🚨 THE FIX: Use enumerate to track the index (i) for unique button keys
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 🚨 THE FIX: Draw sources if they exist in the saved history
        if msg.get("sources"):
            st.markdown("**🔗 Source Documents:**")
            unique_sources = list(set(msg["sources"]))
            
            for source_path in unique_sources:
                if os.path.exists(source_path):
                    with open(source_path, "rb") as file:
                        st.download_button(
                            label=f"📄 Download {os.path.basename(source_path)}",
                            data=file,
                            file_name=os.path.basename(source_path),
                            key=f"dl_{i}_{os.path.basename(source_path)}" # 🚨 CRITICAL: Unique key prevents Streamlit crash!
                        )
                else:
                    st.caption(f"📁 Source: {os.path.basename(source_path)} (File unavailable)")

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
            sources = response.get("sources", [])
            
            # 🚨 THE FIX: Save BOTH the answer and the sources to session state!
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
            
            # Handle the Background Eval UI (We just save this logic for the active response)
            if response.get("faithfulness_score") == -1.0:
                with st.expander("DeepEval Status", expanded=False):
                    st.info("Validation is running in the background. Check Arize Phoenix for the final faithfulness score.")
            
            # 🚨 THE FIX: Force Streamlit to instantly redraw the screen from the top down.
            # This ensures the download buttons are rendered by the main history loop safely.
            st.rerun()