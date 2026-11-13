"""Unit tests for IndexingService business logic and orchestration.

Verifies end-to-end pipeline execution, individual stage idempotency, collection statistics,
serialization roundtrips, and transactional failure safety.
"""

from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
import pytest

from chunking.text_splitter import deserialize_chunks_from_json
from embedding.embedding_manager import EmbeddingError, EmbeddingManager
from enrichment.metadata_enricher import MetadataEnricher
from indexing.vector_store import VectorStoreManager
from services.indexing_service import IndexingService


def test_1_run_full_pipeline_with_injected_fakes(vector_store: VectorStoreManager, fake_embeddings: DeterministicFakeEmbedding, sample_chunks: list[Document]) -> None:
    """1. Test that run_full_pipeline progresses from chunks to indexed vectors with injected fakes."""
    service = IndexingService(
        enricher=MetadataEnricher(),
        embedding_manager=EmbeddingManager(embeddings=fake_embeddings),
        vector_store=vector_store,
    )
    result = service.run_full_pipeline(sample_chunks)

    assert result.indexed_count == len(sample_chunks)
    assert result.total_in_collection == len(sample_chunks)
    assert vector_store.count() == len(sample_chunks)


def test_2_each_stage_called_separately_gives_same_count(vector_store: VectorStoreManager, fake_embeddings: DeterministicFakeEmbedding, sample_chunks: list[Document]) -> None:
    """2. Test that calling each stage separately gives the same final indexed count."""
    service = IndexingService(
        enricher=MetadataEnricher(),
        embedding_manager=EmbeddingManager(embeddings=fake_embeddings),
        vector_store=vector_store,
    )

    enriched = service.enrich_chunks(sample_chunks)
    assert len(enriched) == len(sample_chunks)

    emb_result = service.embed_chunks(enriched)
    assert len(emb_result.vectors) == len(sample_chunks)

    idx_result = service.index_embeddings(emb_result)
    assert idx_result.indexed_count == len(sample_chunks)
    assert vector_store.count() == len(sample_chunks)


def test_3_get_collection_stats_returns_all_keys(vector_store: VectorStoreManager, fake_embeddings: DeterministicFakeEmbedding, sample_chunks: list[Document]) -> None:
    """3. Test that get_collection_stats returns all expected diagnostic keys."""
    service = IndexingService(
        enricher=MetadataEnricher(),
        embedding_manager=EmbeddingManager(embeddings=fake_embeddings),
        vector_store=vector_store,
    )
    service.run_full_pipeline(sample_chunks)
    stats = service.get_collection_stats()

    expected_keys = {"collection_name", "count", "sources", "embedding_model", "persist_directory"}
    assert expected_keys.issubset(set(stats.keys()))
    assert stats["count"] == len(sample_chunks)
    assert len(stats["sources"]) > 0


def test_4_export_enriched_json_roundtrip(sample_chunks: list[Document]) -> None:
    """4. Test that export_enriched_json round-trips through deserialize_chunks_from_json with metadata intact."""
    service = IndexingService()
    enriched = service.enrich_chunks(sample_chunks)

    json_str = service.export_enriched_json(enriched)
    restored = deserialize_chunks_from_json(json_str)

    assert len(restored) == len(enriched)
    for orig, res in zip(enriched, restored):
        assert orig.page_content == res.page_content
        assert orig.metadata["vector_id"] == res.metadata["vector_id"]
        assert orig.metadata["category"] == res.metadata["category"]
        assert orig.metadata["access_level"] == res.metadata["access_level"]


def test_5_embedding_failure_leaves_collection_unchanged(vector_store: VectorStoreManager, sample_chunks: list[Document]) -> None:
    """5. Test that an embedding failure leaves the vector store collection unchanged."""
    failing_embedder = MagicMock()
    failing_embedder.embed_chunks.side_effect = EmbeddingError("Failed embedding")

    service = IndexingService(
        enricher=MetadataEnricher(),
        embedding_manager=failing_embedder,
        vector_store=vector_store,
    )

    initial_count = vector_store.count()
    with pytest.raises(EmbeddingError):
        service.run_full_pipeline(sample_chunks)

    assert vector_store.count() == initial_count
