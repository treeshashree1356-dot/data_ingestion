"""Comprehensive unit tests for Result Ranking & Filtering, deduplication, and LLM context assembly."""

from unittest.mock import MagicMock

from langchain_core.documents import Document
import pytest

from query.query_processor import QueryProcessor
from query.query_schemas import ProcessedQuery
from retrieval.base import BaseReranker
from retrieval.ranking_pipeline import (
    RankedContext,
    RankingOptions,
    ResultRankingPipeline,
    format_context_for_llm,
)
from retrieval.rerankers import ScoreBasedReranker, WeightedScoreReranker
from retrieval.result_filter import ResultFilter
from retrieval.retrieval_schemas import SearchResult
from services.query_service import QueryService


@pytest.fixture
def sample_candidates() -> list[SearchResult]:
    """Fixture providing a diverse set of SearchResult candidates."""
    doc1 = Document(
        page_content="Canonix Security Policy: All employee passwords must be rotated every 90 days.",
        metadata={
            "chunk_id": "policy_doc_01",
            "content_hash": "hash_policy_01",
            "category": "policy",
            "access_level": "internal",
            "source_name": "security_policy.pdf",
        },
    )
    doc2 = Document(
        page_content="Deployment Guide: Step-by-step instructions to deploy microservices using Docker containers.",
        metadata={
            "chunk_id": "guide_doc_02",
            "content_hash": "hash_guide_02",
            "category": "guide",
            "access_level": "internal",
            "source_name": "deploy_guide.docx",
        },
    )
    doc3 = Document(
        page_content="Public FAQ: Frequently asked questions about customer login and password resets.",
        metadata={
            "chunk_id": "faq_doc_03",
            "content_hash": "hash_faq_03",
            "category": "faq",
            "access_level": "public",
            "source_name": "customer_faq.md",
        },
    )
    doc4 = Document(
        page_content="Executive Meeting Notes: Confidential board meeting minutes and quarterly financial strategy.",
        metadata={
            "chunk_id": "internal_doc_04",
            "content_hash": "hash_internal_04",
            "category": "internal",
            "access_level": "restricted",
            "source_name": "board_minutes.docx",
        },
    )

    return [
        SearchResult(document=doc1, score=0.82),
        SearchResult(document=doc2, score=0.91),
        SearchResult(document=doc3, score=0.74),
        SearchResult(document=doc4, score=0.65),
    ]


@pytest.fixture
def ranking_pipeline() -> ResultRankingPipeline:
    """Fixture providing a default ResultRankingPipeline instance."""
    return ResultRankingPipeline()


# ==============================================================================
# 1. Ranking by Similarity Score Tests
# ==============================================================================

def test_ranking_by_similarity_score_orders_descending(
    ranking_pipeline: ResultRankingPipeline,
    sample_candidates: list[SearchResult],
) -> None:
    """1. Test that candidates are ranked strictly in descending order of similarity score."""
    ranked_context = ranking_pipeline.process(
        query="System query",
        results=sample_candidates,
    )

    results = ranked_context.results
    assert len(results) == len(sample_candidates)
    assert results[0].score == 0.91
    assert results[0].chunk_id == "guide_doc_02"
    assert results[1].score == 0.82
    assert results[2].score == 0.74
    assert results[3].score == 0.65


# ==============================================================================
# 2. Duplicate Removal Tests
# ==============================================================================

def test_duplicate_removal_exact_content_hash_and_text(
    ranking_pipeline: ResultRankingPipeline,
) -> None:
    """2. Test that exact duplicate chunks are removed, keeping the highest-scoring occurrence."""
    doc_dup_low = Document(
        page_content="Duplicate policy text content for compliance.",
        metadata={"chunk_id": "chunk_dup_1", "content_hash": "hash_dup_same", "source_name": "doc_a.pdf"},
    )
    doc_dup_high = Document(
        page_content="Duplicate policy text content for compliance.",
        metadata={"chunk_id": "chunk_dup_2", "content_hash": "hash_dup_same", "source_name": "doc_a.pdf"},
    )
    doc_other = Document(
        page_content="Completely different unique text content.",
        metadata={"chunk_id": "chunk_other", "content_hash": "hash_other", "source_name": "doc_b.pdf"},
    )

    candidates = [
        SearchResult(document=doc_dup_low, score=0.70),
        SearchResult(document=doc_dup_high, score=0.95),
        SearchResult(document=doc_other, score=0.80),
    ]

    opts = RankingOptions(deduplicate_exact=True, top_k=10)
    ranked_context = ranking_pipeline.process(query="Query", results=candidates, options=opts)

    assert ranked_context.total_candidates == 3
    assert ranked_context.retained_count == 2
    # The higher scoring duplicate (0.95) should be retained
    assert ranked_context.results[0].score == 0.95
    assert ranked_context.results[0].chunk_id == "chunk_dup_2"


def test_duplicate_removal_overlapping_chunks_from_same_source(
    ranking_pipeline: ResultRankingPipeline,
) -> None:
    """3. Test near-duplicate suppression for overlapping chunks from the same source document."""
    # Chunk A and Chunk B from same source with 90% word overlap
    text_a = "Microservice architecture requires mutual TLS encryption between all internal service endpoints."
    text_b = "Microservice architecture requires mutual TLS encryption between all internal service endpoints securely."

    doc_a = Document(page_content=text_a, metadata={"chunk_id": "c_a", "source_name": "service_specs.pdf"})
    doc_b = Document(page_content=text_b, metadata={"chunk_id": "c_b", "source_name": "service_specs.pdf"})

    candidates = [
        SearchResult(document=doc_a, score=0.92),
        SearchResult(document=doc_b, score=0.88),  # Near-duplicate of doc_a with lower score
    ]

    opts = RankingOptions(deduplicate_overlapping=True, overlap_threshold=0.80, top_k=5)
    ranked_context = ranking_pipeline.process(query="Query", results=candidates, options=opts)

    # Lower scoring near-duplicate should be suppressed
    assert len(ranked_context.results) == 1
    assert ranked_context.results[0].chunk_id == "c_a"
    assert ranked_context.results[0].score == 0.92


# ==============================================================================
# 3. Top-K Ranked Results Tests
# ==============================================================================

def test_top_k_ranked_results_truncation(
    ranking_pipeline: ResultRankingPipeline,
    sample_candidates: list[SearchResult],
) -> None:
    """4. Test that top_k restricts the number of returned chunks."""
    opts = RankingOptions(top_k=2)
    ranked_context = ranking_pipeline.process(
        query="System query",
        results=sample_candidates,
        options=opts,
    )

    assert ranked_context.total_candidates == 4
    assert ranked_context.retained_count == 2
    assert len(ranked_context.results) == 2
    assert ranked_context.results[0].score >= ranked_context.results[1].score


# ==============================================================================
# 4. Metadata Filtering Tests
# ==============================================================================

def test_metadata_filtering_by_category(
    ranking_pipeline: ResultRankingPipeline,
    sample_candidates: list[SearchResult],
) -> None:
    """5. Test post-retrieval filtering by document category."""
    opts = RankingOptions(category_filter="policy", top_k=5)
    ranked_context = ranking_pipeline.process(
        query="Policy query",
        results=sample_candidates,
        options=opts,
    )

    for r in ranked_context.results:
        assert r.category == "policy"


def test_metadata_filtering_by_access_level(
    ranking_pipeline: ResultRankingPipeline,
    sample_candidates: list[SearchResult],
) -> None:
    """6. Test post-retrieval filtering by allowed access levels."""
    opts = RankingOptions(allowed_access_levels=["public"], top_k=5)
    ranked_context = ranking_pipeline.process(
        query="Public query",
        results=sample_candidates,
        options=opts,
    )

    for r in ranked_context.results:
        assert r.access_level == "public"


def test_metadata_filtering_by_minimum_score_threshold(
    ranking_pipeline: ResultRankingPipeline,
    sample_candidates: list[SearchResult],
) -> None:
    """7. Test that candidates below min_score threshold are eliminated."""
    opts = RankingOptions(min_score=0.80, top_k=5)
    ranked_context = ranking_pipeline.process(
        query="Threshold query",
        results=sample_candidates,
        options=opts,
    )

    # Only candidates with score >= 0.80 should remain (0.91 and 0.82)
    assert len(ranked_context.results) == 2
    for r in ranked_context.results:
        assert r.score >= 0.80


def test_metadata_filtering_source_diversity(
    ranking_pipeline: ResultRankingPipeline,
) -> None:
    """8. Test that max_per_source enforces source diversity."""
    docs = [
        SearchResult(Document(page_content="Chunk 1", metadata={"source_name": "doc_a.pdf"}), score=0.95),
        SearchResult(Document(page_content="Chunk 2", metadata={"source_name": "doc_a.pdf"}), score=0.90),
        SearchResult(Document(page_content="Chunk 3", metadata={"source_name": "doc_a.pdf"}), score=0.85),
        SearchResult(Document(page_content="Chunk 4", metadata={"source_name": "doc_b.pdf"}), score=0.80),
    ]

    opts = RankingOptions(max_per_source=1, top_k=5)
    ranked_context = ranking_pipeline.process(query="Query", results=docs, options=opts)

    # doc_a should only have 1 chunk retained, doc_b has 1
    assert len(ranked_context.results) == 2
    sources = [r.source_name for r in ranked_context.results]
    assert sources == ["doc_a.pdf", "doc_b.pdf"]


# ==============================================================================
# 5. Empty Result Handling Tests
# ==============================================================================

def test_empty_candidates_handling(
    ranking_pipeline: ResultRankingPipeline,
) -> None:
    """9. Test that passing an empty candidate list returns clean empty RankedContext."""
    ranked_context = ranking_pipeline.process(
        query="Any query",
        results=[],
    )

    assert ranked_context.total_candidates == 0
    assert ranked_context.retained_count == 0
    assert ranked_context.results == []
    assert ranked_context.context_text == ""
    assert ranked_context.sources_cited == []


# ==============================================================================
# 6. Custom Reranker Extensibility Tests
# ==============================================================================

def test_custom_reranker_extensibility(
    sample_candidates: list[SearchResult],
) -> None:
    """10. Test that a custom BaseReranker (e.g. WeightedScoreReranker) is successfully invoked."""
    # Apply category multiplier of 1.5 to 'policy' chunks, boosting 0.82 -> 1.23
    weighted_reranker = WeightedScoreReranker(category_weights={"policy": 1.5})
    custom_pipeline = ResultRankingPipeline(reranker=weighted_reranker)

    ranked_context = custom_pipeline.process(
        query="Policy query",
        results=sample_candidates,
    )

    # Policy chunk should now be ranked #1
    assert ranked_context.results[0].category == "policy"
    assert ranked_context.results[0].score > 1.20


# ==============================================================================
# 7. Stable Ordering When Scores are Equal Tests
# ==============================================================================

def test_stable_ordering_when_scores_are_equal(
    ranking_pipeline: ResultRankingPipeline,
) -> None:
    """11. Test deterministic tie-breaking when similarity scores are identical."""
    # Identical score (0.85): doc_long has 50 chars, doc_short has 20 chars
    doc_short = Document(page_content="Short text.", metadata={"chunk_id": "chunk_b", "source_name": "b.txt"})
    doc_long = Document(page_content="This is a much longer and more detailed text content.", metadata={"chunk_id": "chunk_a", "source_name": "a.txt"})

    cand1 = SearchResult(document=doc_short, score=0.85)
    cand2 = SearchResult(document=doc_long, score=0.85)

    # Test order 1: [short, long]
    res1 = ranking_pipeline.process(query="Query", results=[cand1, cand2], options=RankingOptions(top_k=5))
    # Test order 2: [long, short]
    res2 = ranking_pipeline.process(query="Query", results=[cand2, cand1], options=RankingOptions(top_k=5))

    # In both cases, longer document is deterministically ranked first
    assert res1.results[0].chunk_id == "chunk_a"
    assert res2.results[0].chunk_id == "chunk_a"
    assert [r.chunk_id for r in res1.results] == [r.chunk_id for r in res2.results]


# ==============================================================================
# 8. LLM Context Formatting Tests
# ==============================================================================

def test_formatted_context_for_llm_consumption(
    sample_candidates: list[SearchResult],
) -> None:
    """12. Test that format_context_for_llm produces clean structured text with metadata headers."""
    formatted_text = format_context_for_llm(sample_candidates[:2])

    assert "--- Document 1:" in formatted_text
    assert "--- Document 2:" in formatted_text
    assert "Category:" in formatted_text
    assert "Relevance:" in formatted_text
    assert "Canonix Security Policy" in formatted_text
    assert "Deployment Guide" in formatted_text


# ==============================================================================
# 9. Backward Compatibility with QueryService Tests
# ==============================================================================

def test_backward_compatibility_with_query_service() -> None:
    """13. Test that QueryService coordinates retrieval and ranking transparently."""
    mock_vs = MagicMock()
    doc = Document(
        page_content="Internal policy guidelines on remote data access.",
        metadata={"chunk_id": "policy_01", "category": "policy", "source_name": "policy.pdf"},
    )
    mock_vs.count.return_value = 1
    mock_vs.similarity_search_by_vector.return_value = [(doc, 0.90)]
    mock_vs.similarity_search.return_value = [(doc, 0.90)]

    service = QueryService(vector_store=mock_vs)

    # 1. execute_query() returns QueryResponse with tuple results
    response = service.execute_query("What are the remote data policies?")
    assert len(response.results) == 1
    d, score = response.results[0]
    assert isinstance(d, Document)
    assert score == 0.90

    # 2. get_ranked_context() returns RankedContext with formatted text
    ranked_ctx = service.get_ranked_context("What are the remote data policies?")
    assert isinstance(ranked_ctx, RankedContext)
    assert len(ranked_ctx.results) == 1
    assert "--- Document 1: policy.pdf" in ranked_ctx.context_text
    assert "policy.pdf" in ranked_ctx.sources_cited
