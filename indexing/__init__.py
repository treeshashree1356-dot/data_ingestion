"""Indexing package for vector database storage and Chroma integration.

Exports vector store manager, indexing result dataclass, and error definitions.
"""

from indexing.vector_store import (
    IndexingError,
    IndexingResult,
    VectorStoreManager,
)

__all__ = [
    "IndexingError",
    "IndexingResult",
    "VectorStoreManager",
]
