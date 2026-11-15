"""Concrete reranker implementations and base templates for downstream refinement.

Includes the default ScoreBasedReranker with deterministic tie-breaking,
and extensible base classes for future Cross-Encoder, BGE, or LLM-based rerankers.
"""

import logging
from typing import Any

from query.query_schemas import ProcessedQuery
from retrieval.base import BaseReranker
from retrieval.retrieval_schemas import SearchResult

logger = logging.getLogger(__name__)


class RerankingError(RuntimeError):
    """Raised when an error occurs during candidate reranking."""
    pass


class ScoreBasedReranker(BaseReranker):
    """Default reranker ordering candidates primarily by similarity score with deterministic tie-breaking.

    Tie-breaking hierarchy when similarity scores are identical:
    1. Similarity score descending (-score)
    2. Document character length descending (-len(page_content)) to prefer more informative chunks
    3. Deterministic chunk ID / vector ID ascending for 100% stable, reproducible ordering
    """

    def rerank(
        self,
        query: ProcessedQuery | str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rank candidates by score with deterministic tie-breaking.

        Args:
            query: ProcessedQuery instance or raw text query (unused in pure score ranking).
            results: List of SearchResult candidates.
            top_k: Optional limit on the number of returned results.

        Returns:
            Ranked list of SearchResult objects.
        """
        if not results:
            return []

        # Sort with deterministic tie-breaking
        ranked = sorted(
            results,
            key=lambda r: (
                -float(r.score),
                -len(r.page_content),
                str(r.chunk_id or r.vector_id or ""),
            ),
        )

        limit = top_k if top_k is not None and top_k > 0 else len(ranked)
        return ranked[:limit]


class WeightedScoreReranker(BaseReranker):
    """Extensible reranker template combining base similarity with category or source weights.

    Demonstrates how future semantic or business-logic rerankers can plug into the pipeline.
    """

    def __init__(
        self,
        category_weights: dict[str, float] | None = None,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize WeightedScoreReranker with optional multiplier weights.

        Args:
            category_weights: Multiplier map for categories (e.g. {'policy': 1.1}).
            source_weights: Multiplier map for specific source documents.
        """
        self.category_weights = category_weights or {}
        self.source_weights = source_weights or {}

    def rerank(
        self,
        query: ProcessedQuery | str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank candidates by applying category/source weight multipliers to base scores.

        Args:
            query: ProcessedQuery or query string.
            results: List of SearchResult candidates.
            top_k: Optional top-k limit.

        Returns:
            Reranked list of SearchResults with updated scores.
        """
        if not results:
            return []

        adjusted_results: list[SearchResult] = []
        for r in results:
            cat_multiplier = self.category_weights.get(r.category, 1.0)
            src_multiplier = self.source_weights.get(r.source_name, 1.0)
            new_score = round(float(r.score) * cat_multiplier * src_multiplier, 6)

            adjusted_results.append(
                SearchResult(
                    document=r.document,
                    score=new_score,
                    metadata=dict(r.metadata),
                    chunk_id=r.chunk_id,
                    vector_id=r.vector_id,
                )
            )

        # Apply deterministic score-based sorting
        reranker = ScoreBasedReranker()
        return reranker.rerank(query=query, results=adjusted_results, top_k=top_k)
