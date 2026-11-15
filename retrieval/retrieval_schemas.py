"""Data structures and schemas for the retrieval pipeline and similarity search.

Provides strongly typed representations of search results, retrieval diagnostics,
and similarity scores.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from query.query_schemas import ProcessedQuery


class RetrievalError(RuntimeError):
    """Raised when an error occurs during similarity search or retrieval."""
    pass


@dataclass
class SearchResult:
    """Encapsulates a single retrieved document chunk with relevance score and metadata.

    Supports tuple unpacking:
        doc, score = search_result
    """

    document: Document
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    vector_id: str = ""

    def __post_init__(self) -> None:
        """Populate metadata shortcuts and identifiers from document if not provided."""
        if self.document and self.document.metadata:
            if not self.metadata:
                self.metadata = dict(self.document.metadata)
            if not self.chunk_id:
                self.chunk_id = str(self.document.metadata.get("chunk_id", ""))
            if not self.vector_id:
                self.vector_id = str(self.document.metadata.get("vector_id", ""))

    @property
    def page_content(self) -> str:
        """Convenience property returning the text content of the underlying document."""
        return self.document.page_content if self.document else ""

    @property
    def category(self) -> str:
        """Convenience property returning the category from document metadata."""
        return str(self.metadata.get("category", "general"))

    @property
    def source_name(self) -> str:
        """Convenience property returning the source name from document metadata."""
        return str(self.metadata.get("source_name", self.metadata.get("source", "unknown")))

    @property
    def access_level(self) -> str:
        """Convenience property returning the access level from document metadata."""
        return str(self.metadata.get("access_level", "internal"))

    def __iter__(self) -> Iterator[Any]:
        """Support tuple unpacking: (document, score) = search_result."""
        yield self.document
        yield self.score

    def __repr__(self) -> str:
        return f"SearchResult(chunk_id={self.chunk_id!r}, score={self.score:.4f}, category={self.category!r})"


@dataclass
class RetrievalResult:
    """Comprehensive diagnostic outcome of a similarity search retrieval execution.

    Attributes:
        results: Ranked list of SearchResult objects.
        query: ProcessedQuery that was executed.
        query_vector: High-dimensional vector embedding generated for the query.
        top_k: Top-k parameter applied.
        total_found: Total number of documents matching query and filters.
        duration_seconds: Wall-clock time spent generating embeddings and searching.
    """

    results: list[SearchResult]
    query: ProcessedQuery
    query_vector: list[float]
    top_k: int
    total_found: int
    duration_seconds: float
