"""Centralized configuration and settings for Data Ingestion Pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory: root of the project
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
ENV_FILE: Path = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv()


class Settings:
    """Application settings loaded from environment or defaults."""

    # Environment
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Application metadata
    APP_TITLE: str = os.getenv("APP_TITLE", "Data Ingestion Pipeline")
    STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

    # File paths
    BASE_DIR: Path = BASE_DIR
    UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
    PROCESSED_DIR: Path = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")
    VECTOR_STORE_DIR: Path = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "data/vector_store")

    # File constraints
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS: list[str] = [
        "pdf", "docx", "txt", "md", "png", "jpg", "jpeg"
    ]

    # Chunking configuration defaults
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))

    # Embedding configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    NORMALIZE_EMBEDDINGS: bool = os.getenv("NORMALIZE_EMBEDDINGS", "True").lower() in ("true", "1", "yes")
    FAKE_EMBEDDING_DIM: int = int(os.getenv("FAKE_EMBEDDING_DIM", "384"))

    # Vector store & Retrieval configuration
    VECTOR_COLLECTION_NAME: str = os.getenv("VECTOR_COLLECTION_NAME", "canonix_knowledge_base")
    DEFAULT_ACCESS_LEVEL: str = os.getenv("DEFAULT_ACCESS_LEVEL", "internal")
    DEFAULT_ALLOWED_ROLES: str = os.getenv("DEFAULT_ALLOWED_ROLES", "all")
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))

    # OCR / External Tools
    TESSERACT_CMD: str | None = os.getenv("TESSERACT_CMD") or None

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure necessary data directories exist."""
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
settings.ensure_directories()
