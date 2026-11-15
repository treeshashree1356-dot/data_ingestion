"""Abstract base classes for search retrieval engines and rerankers.

Defines the contract for similarity search, future hybrid search (vector + keyword),
and downstream reranking components.
"""

from abc import ABC, abstractmethod
from typing import Any

from query.query_schemas import ProcessedQuery
from retrieval.retrieval_schemas import SearchResult


class BaseRetriever(ABC):
    """Abstract base class for all retrieval components."""

    @abstractmethod
    def search(
        self,
        query: ProcessedQuery | str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Execute search against the knowledge store.

        Args:
            query: ProcessedQuery instance or raw query string.
            k: Top-k result limit.
            filters: Optional metadata filters overriding or augmenting query.filters.
            **kwargs: Additional engine-specific arguments.

        Returns:
            Ranked list of SearchResult objects sorted by relevance score descending.
        """
        raise NotImplementedError


class BaseReranker(ABC):
    """Abstract base class for result reranking modules (e.g. Score-based, Cross-Encoder, LLM Reranker)."""

    @abstractmethod
    def rerank(
        self,
        query: ProcessedQuery | str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank preliminary search candidates to improve precision.

        Args:
            query: ProcessedQuery instance or raw query string.
            results: Preliminary candidates retrieved by vector or hybrid search.
            top_k: Optional limit on the final reranked list.

        Returns:
            Reranked list of SearchResult objects with updated scores.
        """
        raise NotImplementedError
