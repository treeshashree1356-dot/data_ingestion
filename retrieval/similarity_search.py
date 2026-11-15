"""Modular Similarity Search component for the retrieval pipeline.

Generates query embeddings using EmbeddingManager, applies metadata and RBAC
filters to VectorStoreManager, and returns ranked SearchResult documents.
Supports pluggable rerankers for future hybrid search or cross-encoder enhancements.
"""

import logging
import time
from typing import Any

from langchain_core.documents import Document

from config.settings import settings
from embedding.embedding_manager import EmbeddingError, EmbeddingManager
from indexing.vector_store import IndexingError, VectorStoreManager
from query.query_processor import QueryProcessor
from query.query_schemas import ProcessedQuery, QueryValidationError
from retrieval.base import BaseReranker, BaseRetriever
from retrieval.retrieval_schemas import RetrievalError, RetrievalResult, SearchResult

logger = logging.getLogger(__name__)


class SimilaritySearcher(BaseRetriever):
    """Executes semantic vector similarity search against the persistent vector store."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager | None = None,
        vector_store: VectorStoreManager | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        """Initialize SimilaritySearcher with optional injected dependencies.

        Dependencies are lazily initialized upon first access if omitted.

        Args:
            embedding_manager: Optional EmbeddingManager instance.
            vector_store: Optional VectorStoreManager instance.
            reranker: Optional BaseReranker instance for downstream score refinement.
        """
        self._embedding_manager = embedding_manager
        self._vector_store = vector_store
        self.reranker = reranker

    @property
    def embedding_manager(self) -> EmbeddingManager:
        """Return or lazily instantiate EmbeddingManager."""
        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager()
        return self._embedding_manager

    @property
    def vector_store(self) -> VectorStoreManager:
        """Return or lazily instantiate VectorStoreManager."""
        if self._vector_store is None:
            self._vector_store = VectorStoreManager()
        return self._vector_store

    def _resolve_query(self, query: ProcessedQuery | str) -> ProcessedQuery:
        """Ensure input query is a validated ProcessedQuery instance."""
        if isinstance(query, ProcessedQuery):
            if not query.cleaned_query or not query.cleaned_query.strip():
                raise QueryValidationError("ProcessedQuery cleaned_query cannot be empty.")
            return query

        if isinstance(query, str):
            processor = QueryProcessor()
            return processor.process(query)

        raise QueryValidationError(
            f"Expected ProcessedQuery or str, got {type(query).__name__}"
        )

    def search(
        self,
        query: ProcessedQuery | str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Execute semantic similarity search for a processed user query.

        Args:
            query: ProcessedQuery instance or raw text query.
            k: Maximum number of results to retrieve. Defaults to settings.RETRIEVAL_TOP_K.
            filters: Optional metadata filters overriding or augmenting query.filters.
            **kwargs: Additional parameters.

        Returns:
            Ranked list of SearchResult objects sorted by similarity score descending.

        Raises:
            QueryValidationError: If query text is invalid.
            RetrievalError: If embedding generation or vector retrieval fails.
        """
        retrieval_res = self.search_with_diagnostics(query=query, k=k, filters=filters, **kwargs)
        return retrieval_res.results

    def search_with_diagnostics(
        self,
        query: ProcessedQuery | str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute similarity search and return comprehensive diagnostic timing and vector data.

        Args:
            query: ProcessedQuery instance or raw text query.
            k: Maximum number of results to retrieve.
            filters: Optional metadata filters.
            **kwargs: Additional parameters.

        Returns:
            RetrievalResult containing ranked SearchResults, query vector, and metrics.
        """
        start_time = time.perf_counter()

        # 1. Resolve and validate ProcessedQuery
        processed = self._resolve_query(query)
        top_k = k if k is not None and k > 0 else int(getattr(settings, "RETRIEVAL_TOP_K", 4))

        # 2. Check if vector store is empty
        if self.vector_store.count() == 0:
            logger.info("Vector store collection '%s' is empty. Returning 0 results.", self.vector_store.collection_name)
            return RetrievalResult(
                results=[],
                query=processed,
                query_vector=[],
                top_k=top_k,
                total_found=0,
                duration_seconds=time.perf_counter() - start_time,
            )

        # 3. Determine search text (incorporating expanded terms if available)
        search_text = processed.cleaned_query
        if processed.expanded_terms:
            search_text = f"{processed.cleaned_query} {' '.join(processed.expanded_terms)}"

        # 4. Generate query embedding using EmbeddingManager
        try:
            query_vector = self.embedding_manager.embed_query(search_text)
        except EmbeddingError as exc:
            raise RetrievalError(f"Failed to generate query embedding: {exc}") from exc
        except Exception as exc:
            raise RetrievalError(f"Unexpected error generating query embedding: {exc}") from exc

        # 5. Merge query filters (RBAC, category) with any runtime filter overrides
        combined_filters = dict(processed.filters) if processed.filters else {}
        if filters:
            combined_filters.update({k_f: v_f for k_f, v_f in filters.items() if v_f is not None})
        effective_filters = combined_filters if combined_filters else None

        # 6. Search vector store using precomputed vector
        raw_candidates: list[tuple[Document, float]] = []
        try:
            if hasattr(self.vector_store, "similarity_search_by_vector"):
                raw_candidates = self.vector_store.similarity_search_by_vector(
                    embedding=query_vector,
                    k=top_k,
                    filters=effective_filters,
                )
            else:
                # Fallback for custom or legacy vector stores
                raw_candidates = self.vector_store.similarity_search(
                    query=search_text,
                    k=top_k,
                    filters=effective_filters,
                )
        except (IndexingError, Exception) as exc:
            logger.error("Vector search failed on query '%s': %s", search_text, exc)
            raise RetrievalError(f"Vector store search failed: {exc}") from exc

        # 7. Convert to SearchResult models and ensure descending sort by score
        search_results: list[SearchResult] = []
        for doc, score in raw_candidates:
            res = SearchResult(
                document=doc,
                score=float(score),
                metadata=dict(doc.metadata) if doc.metadata else {},
            )
            search_results.append(res)

        search_results.sort(key=lambda item: item.score, reverse=True)

        # 8. Optional Reranking Hook
        if self.reranker is not None and search_results:
            logger.debug("Applying reranker %s to %d candidates", type(self.reranker).__name__, len(search_results))
            search_results = self.reranker.rerank(
                query=processed,
                results=search_results,
                top_k=top_k,
            )

        duration = time.perf_counter() - start_time
        logger.info(
            "Similarity search retrieved %d results in %.3fs (top_k=%d, filters=%s)",
            len(search_results),
            duration,
            top_k,
            effective_filters,
        )

        return RetrievalResult(
            results=search_results,
            query=processed,
            query_vector=query_vector,
            top_k=top_k,
            total_found=len(search_results),
            duration_seconds=duration,
        )
