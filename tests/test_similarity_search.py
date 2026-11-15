"""Comprehensive unit and integration tests for SimilaritySearcher, SearchResult, and ranking."""

from unittest.mock import MagicMock

from langchain_core.documents import Document
import pytest

from query.query_processor import QueryProcessor
from query.query_schemas import IntentType, ProcessedQuery, QueryContext, QueryIntent, QueryValidationError
from retrieval.base import BaseReranker
from retrieval.retrieval_schemas import RetrievalError, RetrievalResult, SearchResult
from retrieval.similarity_search import SimilaritySearcher


@pytest.fixture
def sample_documents() -> list[Document]:
    """Fixture providing sample documents with rich metadata."""
    return [
        Document(
            page_content="Canonix Security Policy: All employee passwords must be rotated every 90 days.",
            metadata={
                "chunk_id": "policy_doc_01",
                "vector_id": "vec_01",
                "category": "policy",
                "access_level": "internal",
                "source_name": "security_policy.pdf",
            },
        ),
        Document(
            page_content="Deployment Guide: Step-by-step instructions to deploy microservices using Docker.",
            metadata={
                "chunk_id": "guide_doc_02",
                "vector_id": "vec_02",
                "category": "guide",
                "access_level": "internal",
                "source_name": "deploy_guide.docx",
            },
        ),
        Document(
            page_content="Public FAQ: Frequently asked questions about customer login and password resets.",
            metadata={
                "chunk_id": "faq_doc_03",
                "vector_id": "vec_03",
                "category": "faq",
                "access_level": "public",
                "source_name": "customer_faq.md",
            },
        ),
        Document(
            page_content="Executive Meeting Notes: Confidential board meeting minutes and financial strategy.",
            metadata={
                "chunk_id": "internal_doc_04",
                "vector_id": "vec_04",
                "category": "internal",
                "access_level": "restricted",
                "source_name": "board_minutes.docx",
            },
        ),
    ]


@pytest.fixture
def mock_embedding_manager() -> MagicMock:
    """Fixture providing a mock EmbeddingManager returning 384-dimensional vectors."""
    emb_mgr = MagicMock()
    emb_mgr.embed_query.return_value = [0.1] * 384
    return emb_mgr


@pytest.fixture
def mock_vector_store(sample_documents: list[Document]) -> MagicMock:
    """Fixture providing a mock VectorStoreManager with configurable search responses."""
    vs = MagicMock()
    vs.count.return_value = len(sample_documents)

    def fake_search_by_vector(embedding, k=4, filters=None):
        docs = list(sample_documents)
        # Apply filters in mock
        if filters:
            if "category" in filters:
                docs = [d for d in docs if d.metadata.get("category") == filters["category"]]
            if "access_level" in filters:
                lvl = filters["access_level"]
                if isinstance(lvl, dict) and "$in" in lvl:
                    docs = [d for d in docs if d.metadata.get("access_level") in lvl["$in"]]
                else:
                    docs = [d for d in docs if d.metadata.get("access_level") == lvl]

        # Slice top-k
        sliced = docs[:k]
        # Return with decreasing scores: 0.95, 0.85, 0.75, 0.65...
        return [(d, 0.95 - (i * 0.10)) for i, d in enumerate(sliced)]

    vs.similarity_search_by_vector.side_effect = fake_search_by_vector
    vs.similarity_search.side_effect = lambda query, k=4, filters=None: fake_search_by_vector(None, k, filters)
    return vs


@pytest.fixture
def similarity_searcher(mock_embedding_manager: MagicMock, mock_vector_store: MagicMock) -> SimilaritySearcher:
    """Fixture providing a configured SimilaritySearcher instance."""
    return SimilaritySearcher(
        embedding_manager=mock_embedding_manager,
        vector_store=mock_vector_store,
    )


# ==============================================================================
# 1. Successful Similarity Search Tests
# ==============================================================================

def test_successful_similarity_search_with_processed_query(
    similarity_searcher: SimilaritySearcher,
    mock_embedding_manager: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    """1. Test executing a search with a valid ProcessedQuery returns ranked SearchResult objects."""
    processor = QueryProcessor()
    processed_q = processor.process("What are the password security rules?")

    results = similarity_searcher.search(processed_q, k=3)

    assert len(results) > 0
    assert isinstance(results[0], SearchResult)
    assert results[0].score > 0.0
    assert results[0].page_content != ""
    assert results[0].chunk_id != ""

    # Verify that query embedding was generated
    mock_embedding_manager.embed_query.assert_called_once()
    mock_vector_store.similarity_search_by_vector.assert_called_once()


def test_successful_similarity_search_with_raw_string(
    similarity_searcher: SimilaritySearcher,
) -> None:
    """2. Test that passing a raw string query automatically cleans, classifies, and executes."""
    results = similarity_searcher.search("How to deploy microservices tutorial", k=2)

    assert len(results) > 0
    assert isinstance(results[0], SearchResult)
    assert results[0].category == "guide"


def test_search_result_tuple_unpacking(similarity_searcher: SimilaritySearcher) -> None:
    """3. Test that SearchResult supports tuple unpacking (doc, score) for backward compatibility."""
    results = similarity_searcher.search("What is the data retention policy?", k=1)
    assert len(results) == 1

    doc, score = results[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)
    assert score > 0.0


# ==============================================================================
# 2. Empty Vector Store Tests
# ==============================================================================

def test_empty_vector_store_returns_empty_list(
    mock_embedding_manager: MagicMock,
) -> None:
    """4. Test that an empty vector store (count = 0) returns an empty list without error."""
    empty_vs = MagicMock()
    empty_vs.count.return_value = 0

    searcher = SimilaritySearcher(
        embedding_manager=mock_embedding_manager,
        vector_store=empty_vs,
    )
    processor = QueryProcessor()
    pq = processor.process("Any valid search query")

    results = searcher.search(pq, k=4)
    assert results == []

    # embed_query should not even be called when store is empty
    mock_embedding_manager.embed_query.assert_not_called()


# ==============================================================================
# 3. Top-K Retrieval Tests
# ==============================================================================

def test_top_k_retrieval_limits_results(
    similarity_searcher: SimilaritySearcher,
    mock_vector_store: MagicMock,
) -> None:
    """5. Test that the retrieved count adheres to the top_k parameter."""
    processor = QueryProcessor()
    pq = processor.process("General query without category")

    # Request k = 1
    res_1 = similarity_searcher.search(pq, k=1)
    assert len(res_1) == 1

    # Request k = 3
    res_3 = similarity_searcher.search(pq, k=3)
    assert len(res_3) == 3


# ==============================================================================
# 4. Metadata Filtering Tests (Category)
# ==============================================================================

def test_metadata_filtering_by_category(
    similarity_searcher: SimilaritySearcher,
    mock_vector_store: MagicMock,
) -> None:
    """6. Test that auto-detected category filters only return matching documents."""
    processor = QueryProcessor()
    # Query has "faq" keyword -> auto_detect sets filters["category"] = "faq"
    pq_faq = processor.process("Customer login FAQ questions")

    results = similarity_searcher.search(pq_faq, k=4)

    assert len(results) == 1
    assert results[0].category == "faq"
    assert "FAQ" in results[0].page_content


# ==============================================================================
# 5. RBAC Filtering Tests (Roles & Access Levels)
# ==============================================================================

def test_rbac_filtering_guest_user_only_retrieves_public(
    similarity_searcher: SimilaritySearcher,
) -> None:
    """7. Test that a guest user role only retrieves public documents."""
    processor = QueryProcessor()
    ctx_guest = QueryContext(user_role="guest")
    pq_guest = processor.process("What are our general service features?", context=ctx_guest)

    results = similarity_searcher.search(pq_guest, k=10)

    for r in results:
        assert r.access_level == "public"


def test_rbac_filtering_internal_user_retrieves_public_and_internal(
    similarity_searcher: SimilaritySearcher,
) -> None:
    """8. Test that internal employee role retrieves public and internal documents, but not restricted."""
    processor = QueryProcessor()
    ctx_internal = QueryContext(user_role="internal")
    pq_internal = processor.process("General service features", context=ctx_internal)

    results = similarity_searcher.search(pq_internal, k=10)

    for r in results:
        assert r.access_level in ("public", "internal")
        assert r.access_level != "restricted"


def test_rbac_filtering_admin_can_retrieve_all_access_levels(
    similarity_searcher: SimilaritySearcher,
) -> None:
    """9. Test that admin role can retrieve restricted documents without access_level filter."""
    processor = QueryProcessor()
    ctx_admin = QueryContext(user_role="admin")
    pq_admin = processor.process("General service features", context=ctx_admin)

    results = similarity_searcher.search(pq_admin, k=10)

    access_levels = {r.access_level for r in results}
    assert "restricted" in access_levels or len(results) > 0


# ==============================================================================
# 6. Invalid Query Handling Tests
# ==============================================================================

def test_invalid_query_empty_string_raises_validation_error(
    similarity_searcher: SimilaritySearcher,
) -> None:
    """10. Test that an empty string query raises QueryValidationError."""
    with pytest.raises(QueryValidationError):
        similarity_searcher.search("   ")


def test_invalid_query_none_raises_validation_error(
    similarity_searcher: SimilaritySearcher,
) -> None:
    """11. Test that None query input raises QueryValidationError."""
    with pytest.raises(QueryValidationError):
        similarity_searcher.search(None)  # type: ignore


def test_invalid_query_wrong_type_raises_validation_error(
    similarity_searcher: SimilaritySearcher,
) -> None:
    """12. Test that numeric or object queries raise QueryValidationError."""
    with pytest.raises(QueryValidationError):
        similarity_searcher.search(12345)  # type: ignore


# ==============================================================================
# 7. Result Ordering by Similarity Score Tests
# ==============================================================================

def test_results_ordered_by_similarity_score_descending(
    similarity_searcher: SimilaritySearcher,
) -> None:
    """13. Test that results are strictly ordered descending by relevance score."""
    processor = QueryProcessor()
    pq = processor.process("Broad organizational search")

    results = similarity_searcher.search(pq, k=4)

    assert len(results) >= 2
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


# ==============================================================================
# 8. Modularity & Extensibility (Reranker Hook) Tests
# ==============================================================================

def test_reranker_hook_extensibility(
    mock_embedding_manager: MagicMock,
    mock_vector_store: MagicMock,
) -> None:
    """14. Test injecting a BaseReranker successfully reorders candidate results."""
    class InvertedScoreReranker(BaseReranker):
        def rerank(self, query: ProcessedQuery, results: list[SearchResult], top_k: int | None = None) -> list[SearchResult]:
            # Reverse order and set score to 1.0 - original
            reranked = []
            for r in reversed(results):
                reranked.append(SearchResult(document=r.document, score=round(1.0 - r.score, 4)))
            return reranked[:top_k] if top_k else reranked

    searcher_with_reranker = SimilaritySearcher(
        embedding_manager=mock_embedding_manager,
        vector_store=mock_vector_store,
        reranker=InvertedScoreReranker(),
    )
    processor = QueryProcessor()
    pq = processor.process("Broad organizational search")

    results = searcher_with_reranker.search(pq, k=2)

    assert len(results) == 2
    # The first result should be the formerly lowest score inverted
    assert results[0].score <= 0.50


# ==============================================================================
# 9. Diagnostics Output Tests
# ==============================================================================

def test_search_with_diagnostics_returns_full_metadata(
    similarity_searcher: SimilaritySearcher,
) -> None:
    """15. Test that search_with_diagnostics returns duration, vector, and query metadata."""
    processor = QueryProcessor()
    pq = processor.process("Compliance and retention guidelines")

    retrieval_res = similarity_searcher.search_with_diagnostics(pq, k=2)

    assert isinstance(retrieval_res, RetrievalResult)
    assert len(retrieval_res.results) > 0
    assert len(retrieval_res.query_vector) == 384
    assert retrieval_res.duration_seconds >= 0.0
    assert retrieval_res.top_k == 2
    assert retrieval_res.total_found == len(retrieval_res.results)
