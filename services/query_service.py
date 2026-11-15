"""High-level service coordinating user query processing, vector store retrieval, and result ranking.

Encapsulates retrieval and ranking orchestration so UI, agent, and API consumers
remain purely presentational, while delegating processing to domain modules.
"""

import logging
import time
from typing import Any

from langchain_core.documents import Document

from config.settings import settings
from indexing.vector_store import VectorStoreManager
from query.query_processor import QueryProcessor
from query.query_schemas import QueryContext, QueryResponse
from retrieval.ranking_pipeline import RankedContext, RankingOptions, ResultRankingPipeline
from retrieval.similarity_search import SimilaritySearcher

logger = logging.getLogger(__name__)


class QueryService:
    """Service orchestrating query preprocessing, similarity retrieval, result ranking, and context assembly."""

    def __init__(
        self,
        processor: QueryProcessor | None = None,
        vector_store: VectorStoreManager | None = None,
        similarity_searcher: SimilaritySearcher | None = None,
        ranking_pipeline: ResultRankingPipeline | None = None,
    ) -> None:
        """Initialize QueryService with optional injected dependencies.

        Dependencies are lazily initialized upon first access if not supplied.

        Args:
            processor: Optional QueryProcessor instance.
            vector_store: Optional VectorStoreManager instance.
            similarity_searcher: Optional SimilaritySearcher instance.
            ranking_pipeline: Optional ResultRankingPipeline instance.
        """
        self._processor = processor
        self._vector_store = vector_store
        self._similarity_searcher = similarity_searcher
        self._ranking_pipeline = ranking_pipeline

    @property
    def processor(self) -> QueryProcessor:
        """Return or lazily instantiate QueryProcessor."""
        if self._processor is None:
            self._processor = QueryProcessor()
        return self._processor

    @property
    def vector_store(self) -> VectorStoreManager:
        """Return or lazily instantiate VectorStoreManager."""
        if self._vector_store is None:
            self._vector_store = VectorStoreManager()
        return self._vector_store

    @property
    def similarity_searcher(self) -> SimilaritySearcher:
        """Return or lazily instantiate SimilaritySearcher bound to current vector store."""
        if self._similarity_searcher is None:
            self._similarity_searcher = SimilaritySearcher(
                vector_store=self.vector_store,
            )
        return self._similarity_searcher

    @property
    def ranking_pipeline(self) -> ResultRankingPipeline:
        """Return or lazily instantiate ResultRankingPipeline."""
        if self._ranking_pipeline is None:
            self._ranking_pipeline = ResultRankingPipeline()
        return self._ranking_pipeline

    def execute_query(
        self,
        query: str,
        context: QueryContext | None = None,
        ranking_options: RankingOptions | None = None,
    ) -> QueryResponse:
        """Process user query, execute similarity search, and apply ranking and deduplication.

        Args:
            query: Raw user query string.
            context: Optional QueryContext configuring roles, overrides, and limits.
            ranking_options: Optional RankingOptions for fine-grained ranking/filtering.

        Returns:
            QueryResponse containing ranked documents, scores, and full diagnostic details.
        """
        start_time = time.perf_counter()
        ctx = context or QueryContext()

        logger.info("Executing query pipeline for: '%s' (role: %s)", query, ctx.user_role)

        # 1. Process query through domain layer
        processed = self.processor.process(query, context=ctx)

        # 2. Retrieve candidates via SimilaritySearcher (retrieve 2x candidates to allow effective dedup)
        k_val = ctx.top_k if ctx.top_k else int(getattr(settings, "RETRIEVAL_TOP_K", 4))
        candidate_k = max(k_val * 2, 8) if (ranking_options and (ranking_options.deduplicate_exact or ranking_options.deduplicate_overlapping)) else k_val

        search_results = self.similarity_searcher.search(
            query=processed,
            k=candidate_k,
        )

        # 3. Apply ResultRankingPipeline for deduplication, filtering, and stable ordering
        opts = ranking_options or RankingOptions(top_k=k_val)
        ranked_context = self.ranking_pipeline.process(
            query=processed,
            results=search_results,
            options=opts,
        )

        duration = time.perf_counter() - start_time
        logger.info(
            "Query returned %d ranked results in %.3fs (intent: %s, category: %s)",
            len(ranked_context.results),
            duration,
            processed.intent.intent_type.value,
            processed.detected_category,
        )

        # Convert SearchResults to list of tuples for backward-compatible QueryResponse results
        tuple_results: list[tuple[Document, float]] = [(r.document, r.score) for r in ranked_context.results]

        return QueryResponse(
            results=tuple_results,
            processed_query=processed,
            total_found=len(ranked_context.results),
            duration_seconds=duration,
        )

    def get_ranked_context(
        self,
        query: str,
        context: QueryContext | None = None,
        options: RankingOptions | None = None,
    ) -> RankedContext:
        """Execute query pipeline and return assembled context specifically formatted for LLM consumption.

        Args:
            query: Raw user query string.
            context: Optional QueryContext configuring user roles and filters.
            options: Optional RankingOptions configuring thresholds and limits.

        Returns:
            RankedContext with formatted context text, citations, and ranked results.
        """
        ctx = context or QueryContext()
        processed = self.processor.process(query, context=ctx)

        k_val = options.top_k if options and options.top_k else ctx.top_k
        candidate_k = max(k_val * 2, 8)

        candidates = self.similarity_searcher.search(query=processed, k=candidate_k)
        return self.ranking_pipeline.process(query=processed, results=candidates, options=options)

    def search(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Convenience search method for backward compatibility with IndexingService callers.

        Args:
            query: Natural language query string.
            k: Top-k result limit.
            filters: Optional metadata filters.

        Returns:
            List of (Document, score) tuples.
        """
        ctx = QueryContext(
            top_k=k or int(getattr(settings, "RETRIEVAL_TOP_K", 4)),
            auto_detect_category=filters is None or "category" not in filters,
            extra_filters=filters or {},
        )
        response = self.execute_query(query=query, context=ctx)
        return response.results


# Module-level singleton instance
query_service = QueryService()
