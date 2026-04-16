from langchain_ollama import ChatOllama

# Using Llama 3.2 as it's fast and smart for reasoning
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0, # Keep it deterministic for logic
    base_url="http://localhost:11434"
)