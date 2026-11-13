"""Embedding package for document chunks and vector representations.

Exports embedding factory, configuration error, manager, and result dataclass.
"""

from embedding.embedding_factory import (
    EmbeddingConfigurationError,
    clear_embedding_cache,
    get_embedding_model,
)
from embedding.embedding_manager import (
    EmbeddingError,
    EmbeddingManager,
    EmbeddingResult,
)

__all__ = [
    "EmbeddingConfigurationError",
    "clear_embedding_cache",
    "get_embedding_model",
    "EmbeddingError",
    "EmbeddingManager",
    "EmbeddingResult",
]
