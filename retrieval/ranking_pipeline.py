"""Result Ranking and Filtering pipeline coordinating deduplication, filtering, and LLM context assembly.

Combines ResultFilter, BaseReranker, and context formatting into a unified,
extensible post-retrieval pipeline.
"""

from dataclasses import dataclass, field
import logging
from typing import Any

from query.query_schemas import ProcessedQuery
from retrieval.base import BaseReranker
from retrieval.rerankers import ScoreBasedReranker
from retrieval.result_filter import ResultFilter
from retrieval.retrieval_schemas import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RankingOptions:
    """Options governing post-retrieval ranking, deduplication, and metadata filtering.

    Attributes:
        top_k: Maximum number of final ranked chunks to retain.
        min_score: Minimum similarity score threshold for candidate retention.
        deduplicate_exact: Whether to eliminate exact duplicate chunks.
        deduplicate_overlapping: Whether to suppress near-duplicate chunks from the same source.
        overlap_threshold: Jaccard similarity threshold for near-duplicate suppression.
        max_per_source: Maximum chunks allowed from any single source document (diversity).
        category_filter: Optional explicit category requirement.
        allowed_access_levels: Optional list of permitted access levels.
        allowed_sources: Optional whitelist of source names.
        excluded_sources: Optional blacklist of source names.
    """

    top_k: int = 4
    min_score: float = 0.0
    deduplicate_exact: bool = True
    deduplicate_overlapping: bool = True
    overlap_threshold: float = 0.85
    max_per_source: int | None = None
    category_filter: str | None = None
    allowed_access_levels: list[str] | None = None
    allowed_sources: list[str] | None = None
    excluded_sources: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate options."""
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k <= 0:
            self.top_k = 4
        if self.min_score < 0.0:
            self.min_score = 0.0


@dataclass
class RankedContext:
    """Structured context packaged specifically for downstream LLM generation / RAG consumption.

    Attributes:
        results: Final ranked list of SearchResult chunks.
        context_text: Assembled, formatted context text ready for LLM prompt injection.
        total_candidates: Number of candidate chunks before filtering.
        retained_count: Number of chunks retained after filtering and ranking.
        sources_cited: Unique source names included in the context text.
        char_count: Total character count of the assembled context text.
        word_count: Total word count of the assembled context text.
    """

    results: list[SearchResult]
    context_text: str
    total_candidates: int
    retained_count: int
    sources_cited: list[str] = field(default_factory=list)
    char_count: int = 0
    word_count: int = 0

    def __post_init__(self) -> None:
        """Compute statistics if not explicitly provided."""
        if self.context_text:
            if not self.char_count:
                self.char_count = len(self.context_text)
            if not self.word_count:
                self.word_count = len(self.context_text.split())
        if self.results and not self.sources_cited:
            sources: set[str] = {r.source_name for r in self.results if r.source_name}
            self.sources_cited = sorted(sources)


def format_context_for_llm(results: list[SearchResult]) -> str:
    """Format ranked SearchResults into a clean, structured context string for LLM prompts.

    Args:
        results: Ranked list of SearchResult chunks.

    Returns:
        Structured context string with clear document separators and metadata tags.
    """
    if not results:
        return ""

    context_blocks: list[str] = []
    for idx, r in enumerate(results, start=1):
        score_str = f"{r.score:.4f}" if isinstance(r.score, float) else str(r.score)
        header = (
            f"--- Document {idx}: {r.source_name} "
            f"[Category: {r.category} | Access: {r.access_level} | Relevance: {score_str}] ---"
        )
        block = f"{header}\n{r.page_content.strip()}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


class ResultRankingPipeline(BaseReranker):
    """Pipeline orchestrating deduplication, metadata filtering, reranking, and context assembly."""

    def __init__(
        self,
        reranker: BaseReranker | None = None,
        filter_engine: ResultFilter | None = None,
    ) -> None:
        """Initialize ResultRankingPipeline with optional reranker and filter components.

        Args:
            reranker: Pluggable reranker instance (defaults to ScoreBasedReranker).
            filter_engine: Pluggable filter and deduplication engine (defaults to ResultFilter).
        """
        self.reranker = reranker or ScoreBasedReranker()
        self.filter_engine = filter_engine or ResultFilter()

    def process(
        self,
        query: ProcessedQuery | str,
        results: list[SearchResult],
        options: RankingOptions | None = None,
    ) -> RankedContext:
        """Execute end-to-end filtering, deduplication, reranking, and context assembly.

        Args:
            query: ProcessedQuery instance or query string.
            results: Raw candidate SearchResults from similarity search.
            options: Optional RankingOptions configuring thresholds and limits.

        Returns:
            RankedContext ready for downstream LLM generation.
        """
        opts = options or RankingOptions()
        total_candidates = len(results)

        if not results:
            logger.info("No candidates provided to ResultRankingPipeline. Returning empty context.")
            return RankedContext(
                results=[],
                context_text="",
                total_candidates=0,
                retained_count=0,
            )

        candidates = list(results)

        # 1. Exact Deduplication
        if opts.deduplicate_exact:
            candidates = self.filter_engine.deduplicate_exact(candidates)

        # 2. Metadata & Score Threshold Filtering
        candidates = self.filter_engine.filter_by_metadata(
            results=candidates,
            min_score=opts.min_score,
            category=opts.category_filter,
            allowed_access_levels=opts.allowed_access_levels,
            allowed_sources=opts.allowed_sources,
            excluded_sources=opts.excluded_sources,
            max_per_source=opts.max_per_source,
        )

        # 3. Reranking (applies ScoreBasedReranker or injected custom reranker)
        ranked = self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=None,  # Do not truncate before overlap dedup
        )

        # 4. Overlap & Near-Duplicate Suppression (applied in ranked order)
        if opts.deduplicate_overlapping:
            ranked = self.filter_engine.deduplicate_overlapping(
                results=ranked,
                overlap_threshold=opts.overlap_threshold,
            )

        # 5. Top-K Truncation
        final_results = ranked[: opts.top_k]

        # 6. Assemble context for downstream LLM consumption
        context_text = format_context_for_llm(final_results)

        logger.info(
            "Ranking pipeline finished: %d candidates -> %d final chunks (top_k=%d)",
            total_candidates,
            len(final_results),
            opts.top_k,
        )

        return RankedContext(
            results=final_results,
            context_text=context_text,
            total_candidates=total_candidates,
            retained_count=len(final_results),
        )

    def rerank(
        self,
        query: ProcessedQuery | str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """BaseReranker interface implementation enabling direct injection into SimilaritySearcher.

        Args:
            query: ProcessedQuery or query string.
            results: List of SearchResult candidates.
            top_k: Optional top-k limit.

        Returns:
            Reranked and deduplicated list of SearchResults.
        """
        opts = RankingOptions(top_k=top_k if top_k is not None and top_k > 0 else 4)
        ranked_context = self.process(query=query, results=results, options=opts)
        return ranked_context.results
