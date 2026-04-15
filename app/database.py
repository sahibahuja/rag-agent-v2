import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load variables from .env
load_dotenv()

# Global variable to hold the client instance for Singleton pattern
_client = None

# CONFIGURATION
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_docs")
# Standardizing names for Phase 2
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3") 

def get_client() -> QdrantClient:
    """
    Returns a single instance of the Qdrant client. 
    Initialization happens only when this function is called.
    """
    global _client
    if _client is None:
        db_host = os.getenv("QDRANT_HOST", "localhost")
        
        # Logic to handle both Local Path and Docker/Remote Host
        if db_host.startswith("./") or "/" in db_host or "\\" in db_host:
            _client = QdrantClient(path=db_host)
        else:
            _client = QdrantClient(
                host=db_host, 
                port=int(os.getenv("QDRANT_PORT", 6333))
            )
        
        # Porting your Phase 1 logic: Set the model for FastEmbed
        _client.set_model(EMBED_MODEL)
        
    return _client

def init_db():
    """Ensures the collection exists and handles dimension safety checks"""
    q_client = get_client()
    
    if q_client.collection_exists(COLLECTION_NAME):
        info = q_client.get_collection(COLLECTION_NAME)
        
        # 1. Safe access to the vector config
        vector_config = info.config.params.vectors
        
        # 2. Check if it's a dict (multiple vectors) or a single VectorParams object
        if vector_config is not None:
            # Handle the case where it's a dictionary of vectors (common with FastEmbed)
            if isinstance(vector_config, dict):
                current_vector = vector_config.get(EMBED_MODEL)
                current_size = current_vector.size if current_vector else None
            else:
                # Handle the case where it's a single VectorParams object
                current_size = getattr(vector_config, 'size', None)

            # 3. Perform the mismatch check if we successfully got a size
            if current_size:
                expected_size = 1024 if "m3" in EMBED_MODEL.lower() or "large" in EMBED_MODEL.lower() else 384
                
                if current_size != expected_size:
                    print(f"⚠️ Mismatch: Current {current_size} != Expected {expected_size}. Recreating...")
                    q_client.delete_collection(COLLECTION_NAME)