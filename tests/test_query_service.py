"""Unit and integration tests for QueryService retrieval orchestration and backward compatibility."""

from unittest.mock import MagicMock

from langchain_core.documents import Document
import pytest

from query.query_processor import QueryProcessor
from query.query_schemas import IntentType, QueryContext, QueryResponse, QueryValidationError
from services.indexing_service import IndexingService
from services.query_service import QueryService


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Fixture providing a mock VectorStoreManager."""
    vs = MagicMock()
    doc1 = Document(
        page_content="Canonix Security Policy: Passwords must be rotated every 90 days.",
        metadata={"category": "policy", "access_level": "internal", "chunk_id": "chunk_01"},
    )
    vs.count.return_value = 1
    vs.similarity_search.return_value = [(doc1, 0.95)]
    vs.similarity_search_by_vector.return_value = [(doc1, 0.95)]
    return vs


@pytest.fixture
def query_service(mock_vector_store: MagicMock) -> QueryService:
    """Fixture providing QueryService with injected mock VectorStoreManager."""
    return QueryService(
        processor=QueryProcessor(),
        vector_store=mock_vector_store,
    )


def test_query_service_lazy_initialization() -> None:
    """Test that QueryService lazily initializes default processor and vector store."""
    service = QueryService()
    assert service.processor is not None
    assert isinstance(service.processor, QueryProcessor)
    # Accessing vector_store initializes without error
    assert service.vector_store is not None


def test_query_service_execute_query_orchestration(
    query_service: QueryService,
    mock_vector_store: MagicMock,
) -> None:
    """Test that execute_query processes the query, calls vector store, and returns QueryResponse."""
    query = "What is the compliance policy for password rotation?"
    ctx = QueryContext(user_role="internal", top_k=2)

    response = query_service.execute_query(query, context=ctx)

    assert isinstance(response, QueryResponse)
    assert response.total_found == 1
    assert response.duration_seconds >= 0.0
    assert len(response.results) == 1

    # Verify vector store call arguments
    assert mock_vector_store.similarity_search_by_vector.called or mock_vector_store.similarity_search.called
    call = mock_vector_store.similarity_search_by_vector.call_args or mock_vector_store.similarity_search.call_args
    call_kwargs = call[1]

    assert call_kwargs["k"] == 2
    # Category was auto-detected as 'policy', access_level derived from role 'internal'
    assert call_kwargs["filters"]["category"] == "policy"
    assert call_kwargs["filters"]["access_level"] == {"$in": ["public", "internal"]}


def test_query_service_execute_query_with_expansion(
    query_service: QueryService,
    mock_vector_store: MagicMock,
) -> None:
    """Test that query expansion augments the query with expanded terms."""
    query = "Setup auth with k8s"
    ctx = QueryContext(enable_expansion=True, top_k=3)

    response = query_service.execute_query(query, context=ctx)

    assert mock_vector_store.similarity_search_by_vector.called or mock_vector_store.similarity_search.called
    assert len(response.processed_query.expanded_terms) > 0
    assert "kubernetes" in response.processed_query.expanded_terms


def test_query_service_search_backward_compatibility(
    query_service: QueryService,
    mock_vector_store: MagicMock,
) -> None:
    """Test that QueryService.search() returns list[tuple[Document, float]] matching previous signature."""
    results = query_service.search(
        query="Password policy guidelines",
        k=5,
        filters={"category": "policy"},
    )
    assert isinstance(results, list)
    assert len(results) == 1
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert score == 0.95


def test_query_service_propagates_validation_error(query_service: QueryService) -> None:
    """Test that QueryService propagates QueryValidationError for invalid inputs."""
    with pytest.raises(QueryValidationError):
        query_service.execute_query("   ")


def test_indexing_service_delegates_search_to_query_service(mock_vector_store: MagicMock) -> None:
    """Test that IndexingService.search() seamlessly delegates through QueryService."""
    indexing_svc = IndexingService(vector_store=mock_vector_store)
    results = indexing_svc.search(
        query="What is the data retention policy?",
        k=3,
    )
    assert len(results) == 1
    assert mock_vector_store.similarity_search_by_vector.called or mock_vector_store.similarity_search.called
