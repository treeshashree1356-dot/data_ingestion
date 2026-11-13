"""Unit tests for embedding generation and factory orchestration.

Verifies model caching, batch execution, progress reporting, dimension verification,
deterministic output, and error wrapping.
"""

import os
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
import pytest

from embedding.embedding_factory import (
    EmbeddingConfigurationError,
    clear_embedding_cache,
    get_embedding_model,
)
from embedding.embedding_manager import (
    EmbeddingError,
    EmbeddingManager,
    EmbeddingResult,
)


def test_1_factory_returns_fake_and_rejects_unknown() -> None:
    """1. Test that factory returns DeterministicFakeEmbedding for 'fake' and raises for unknown."""
    clear_embedding_cache()
    model = get_embedding_model(provider="fake")
    assert isinstance(model, DeterministicFakeEmbedding)

    with pytest.raises(EmbeddingConfigurationError, match="Unsupported embedding provider"):
        get_embedding_model(provider="unsupported_provider_xyz")


def test_2_factory_caches_instances_and_clear_resets() -> None:
    """2. Test that factory caches instances by (provider, model_name) and clear_embedding_cache resets it."""
    clear_embedding_cache()
    m1 = get_embedding_model(provider="fake", model_name="test_model_1")
    m2 = get_embedding_model(provider="fake", model_name="test_model_1")
    assert m1 is m2

    clear_embedding_cache()
    m3 = get_embedding_model(provider="fake", model_name="test_model_1")
    assert m3 is not m1


def test_3_manager_init_does_not_call_factory() -> None:
    """3. Test that EmbeddingManager.__init__ does not call get_embedding_model."""
    with patch("embedding.embedding_manager.get_embedding_model") as mock_factory:
        manager = EmbeddingManager(provider="fake")
        mock_factory.assert_not_called()
        assert manager.batch_size > 0


def test_4_vector_count_equals_chunk_count_and_expected_dimension(fake_embeddings: DeterministicFakeEmbedding) -> None:
    """4. Test that vector count matches chunk count and every vector has expected dimension."""
    manager = EmbeddingManager(embeddings=fake_embeddings)
    chunks = [
        Document(page_content="Chunk 1", metadata={"chunk_id": "c1"}),
        Document(page_content="Chunk 2", metadata={"chunk_id": "c2"}),
        Document(page_content="Chunk 3", metadata={"chunk_id": "c3"}),
    ]
    result = manager.embed_chunks(chunks)

    assert len(result.vectors) == 3
    assert result.dimension == 384
    for vec in result.vectors:
        assert len(vec) == 384


def test_5_batching_and_progress_callback(fake_embeddings: DeterministicFakeEmbedding) -> None:
    """5. Test batching: with batch_size=2 and 5 chunks, embed_documents called 3 times with progress callback."""
    mock_engine = MagicMock(wraps=fake_embeddings)
    manager = EmbeddingManager(embeddings=mock_engine, batch_size=2)
    chunks = [Document(page_content=f"Chunk {i}") for i in range(5)]

    progress_calls = []

    def callback(done: int, total: int) -> None:
        progress_calls.append((done, total))

    result = manager.embed_chunks(chunks, progress_callback=callback)

    assert mock_engine.embed_documents.call_count == 3
    assert progress_calls == [(2, 5), (4, 5), (5, 5)]
    assert len(result.vectors) == 5


def test_6_same_text_gives_same_vector(fake_embeddings: DeterministicFakeEmbedding) -> None:
    """6. Test that identical text produces identical embedding vectors (deterministic)."""
    manager = EmbeddingManager(embeddings=fake_embeddings)
    doc_a = Document(page_content="Consistent semantic text.")
    doc_b = Document(page_content="Consistent semantic text.")

    res = manager.embed_chunks([doc_a, doc_b])
    assert res.vectors[0] == res.vectors[1]


def test_7_empty_input_returns_empty_result_without_loading_model() -> None:
    """7. Test that empty input returns an empty result and never triggers model loading."""
    with patch("embedding.embedding_manager.get_embedding_model") as mock_factory:
        manager = EmbeddingManager(provider="fake")
        res = manager.embed_chunks([])

        mock_factory.assert_not_called()
        assert len(res.chunks) == 0
        assert len(res.vectors) == 0
        assert res.dimension == 0


def test_8_model_exception_wrapped_in_embedding_error() -> None:
    """8. Test that exceptions from the underlying model are caught and wrapped in EmbeddingError."""
    failing_model = MagicMock()
    failing_model.embed_documents.side_effect = RuntimeError("GPU Out of Memory")

    manager = EmbeddingManager(embeddings=failing_model)
    doc = Document(page_content="Failing chunk content.")

    with pytest.raises(EmbeddingError, match="Embedding failed at batch 1.*GPU Out of Memory"):
        manager.embed_chunks([doc])


def test_9_embed_query_dimension_and_empty_validation(fake_embeddings: DeterministicFakeEmbedding) -> None:
    """9. Test embed_query returns correct dimension and empty query raises ValueError."""
    manager = EmbeddingManager(embeddings=fake_embeddings)
    vec = manager.embed_query("Valid test query string")
    assert len(vec) == 384
    assert isinstance(vec[0], float)

    with pytest.raises(ValueError, match="cannot be empty"):
        manager.embed_query("")

    with pytest.raises(ValueError, match="cannot be empty"):
        manager.embed_query("    ")


def test_10_embedding_model_and_dim_added_to_metadata(fake_embeddings: DeterministicFakeEmbedding) -> None:
    """10. Test that embedding_model and embedding_dim are appended to each chunk's metadata."""
    manager = EmbeddingManager(embeddings=fake_embeddings, model_name="custom/test-model")
    doc = Document(page_content="Enriched metadata test chunk.", metadata={"chunk_id": "c1"})
    result = manager.embed_chunks([doc])

    assert len(result.chunks) == 1
    chunk_meta = result.chunks[0].metadata
    assert chunk_meta["embedding_model"] == "custom/test-model"
    assert chunk_meta["embedding_dim"] == 384


@pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1",
    reason="Live HuggingFace model download test skipped unless RUN_MODEL_TESTS=1",
)
def test_11_real_huggingface_model_execution() -> None:
    """11. Optional real model test verifying all-MiniLM-L6-v2 dimension and semantic ranking."""
    clear_embedding_cache()
    model = get_embedding_model(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2")
    manager = EmbeddingManager(embeddings=model)

    docs = [
        Document(page_content="The company policy requires data encryption."),
        Document(page_content="Information security guidelines enforce encrypted storage."),
        Document(page_content="Recipe for baking delicious chocolate chip cookies."),
    ]
    res = manager.embed_chunks(docs)
    assert res.dimension == 384

    # Calculate cosine similarities
    def dot_product(v1: list[float], v2: list[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2))

    sim_related = dot_product(res.vectors[0], res.vectors[1])
    sim_unrelated = dot_product(res.vectors[0], res.vectors[2])
    assert sim_related > sim_unrelated, f"Related score {sim_related} not > unrelated {sim_unrelated}"
