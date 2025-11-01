import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    
    # Model Configuration
    LLM_MODEL: str = "gemini-2.5-flash-lite"
    LLM_TEMPERATURE: float = 0.3
    
    # MemGPT Configuration
    MAX_CONTEXT_TOKENS: int = 8000
    MEMORY_WARNING_THRESHOLD: float = 0.7
    FLUSH_THRESHOLD: float = 0.9
    
    # Database Configuration (for production)
    POSTGRES_URI: str = os.getenv(
        "POSTGRES_URI", 
        "postgresql://user:pass@localhost:5432/travel_planner"
    )
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE: str = "chroma"  # chroma, pinecone, pgvector
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    # MCP Server
    MCP_SERVER_URL: str = "http://localhost:8000"
    
    class Config:
        env_file = ".env"

settings = Settings()