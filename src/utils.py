"""Utility functions for the RAG assistant."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_api_key(provider: str = "gemini") -> str:
    """Get API key for the specified provider."""
    if provider.lower() == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        return key
    elif provider.lower() == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        return key
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_vector_db_path() -> Path:
    """Get the vector DB path from env or default."""
    path = os.getenv("VECTOR_DB_PATH", "./data/chroma")
    db_path = Path(path)
    db_path.mkdir(parents=True, exist_ok=True)
    return db_path


def get_embedding_model_name() -> str:
    """Get the embedding model name."""
    return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The full text to split
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap
    
    return chunks