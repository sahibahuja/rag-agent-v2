import os  # 🚨 Moved to the top!
import streamlit as st
import requests
import json

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
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Show assistant message (thinking -> responding)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 🚨 THE NEW UX: Custom CSS for Bouncing Dots Animation 🚨
        thinking_html = """
        <style>
        .typing-indicator { display: flex; align-items: center; gap: 6px; padding: 10px 5px; }
        .typing-indicator span { width: 8px; height: 8px; background-color: #888888; border-radius: 50%; display: inline-block; animation: bounce 1.4s infinite ease-in-out both; }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
        </style>
        <div class="typing-indicator"><span></span><span></span><span></span></div>
        """
        # Inject the animation into the empty placeholder immediately
        message_placeholder.markdown(thinking_html, unsafe_allow_html=True)
        
        full_response = ""
        sources = []
        
        payload = {"question": prompt, "thread_id": selected_thread_id}
        
        # Open a streaming connection to FastAPI
        with requests.post(f"{FASTAPI_BASE_URL}/v2/agent/chat", json=payload, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data = json.loads(decoded_line[6:])
                        
                        # 1. First token arrives! The animation is instantly overwritten.
                        if data["type"] == "token":
                            full_response += data["content"]
                            message_placeholder.markdown(full_response + "▌") 
                            
                        # 2. Grab the metadata
                        elif data["type"] == "metadata":
                            sources = data["sources"]
        
        # The stream is finished! Remove the cursor
        message_placeholder.markdown(full_response)
        
        # Save to local UI state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "sources": sources
        })
        
        # Force a rerun so the main loop can draw the download buttons!
        st.rerun()