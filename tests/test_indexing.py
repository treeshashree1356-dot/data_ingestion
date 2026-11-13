"""Unit tests for Chroma vector database indexing and querying.

Verifies lazy initialization, idempotency, deduplication, metadata persistence,
filtering, persistence across restarts, and LangChain handoff.
"""

from pathlib import Path
import shutil

from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
import pytest

from embedding.embedding_manager import EmbeddingResult
from indexing.vector_store import IndexingError, VectorStoreManager


def test_1_init_does_not_create_files_on_disk(tmp_path: Path, fake_embeddings: DeterministicFakeEmbedding) -> None:
    """1. Test that creating VectorStoreManager does not create files on disk until first use."""
    vstore_dir = tmp_path / "lazy_vstore"
    assert not vstore_dir.exists()

    manager = VectorStoreManager(persist_directory=vstore_dir, embeddings=fake_embeddings)
    assert not vstore_dir.exists(), "Directory was prematurely created on disk during __init__"


def test_2_indexing_stores_n_vectors_and_count(vector_store: VectorStoreManager) -> None:
    """2. Test that indexing stores exactly N vectors and count() returns N."""
    chunks = [
        Document(page_content="Alpha chunk", metadata={"vector_id": "v1", "source": "s1.txt", "content_hash": "h1"}),
        Document(page_content="Beta chunk", metadata={"vector_id": "v2", "source": "s1.txt", "content_hash": "h2"}),
    ]
    res = EmbeddingResult(
        chunks=chunks,
        vectors=[[0.1] * 384, [0.2] * 384],
        model_name="fake",
        provider="fake",
        dimension=384,
        duration_seconds=0.1,
    )
    idx_res = vector_store.index(res)

    assert idx_res.indexed_count == 2
    assert vector_store.count() == 2


def test_3_idempotent_indexing_keeps_count(vector_store: VectorStoreManager) -> None:
    """3. Test that indexing the same result twice keeps the count at N (idempotent upsert)."""
    chunks = [
        Document(page_content="Alpha chunk", metadata={"vector_id": "v1", "source": "s1.txt", "content_hash": "h1"}),
    ]
    res = EmbeddingResult(
        chunks=chunks,
        vectors=[[0.1] * 384],
        model_name="fake",
        provider="fake",
        dimension=384,
        duration_seconds=0.1,
    )
    vector_store.index(res)
    assert vector_store.count() == 1

    # Index same chunk again
    idx_res2 = vector_store.index(res)
    assert vector_store.count() == 1
    assert idx_res2.total_in_collection == 1


def test_4_duplicate_content_inside_batch_skipped(vector_store: VectorStoreManager) -> None:
    """4. Test that duplicate content_hash values inside the same batch are skipped and counted."""
    chunks = [
        Document(page_content="Same content", metadata={"vector_id": "v1", "source": "s1.txt", "content_hash": "dup_hash"}),
        Document(page_content="Same content", metadata={"vector_id": "v2", "source": "s2.txt", "content_hash": "dup_hash"}),
        Document(page_content="Unique content", metadata={"vector_id": "v3", "source": "s3.txt", "content_hash": "uniq_hash"}),
    ]
    res = EmbeddingResult(
        chunks=chunks,
        vectors=[[0.1] * 384, [0.1] * 384, [0.3] * 384],
        model_name="fake",
        provider="fake",
        dimension=384,
        duration_seconds=0.1,
    )
    idx_res = vector_store.index(res)

    assert idx_res.indexed_count == 2
    assert idx_res.skipped_duplicates == 1
    assert vector_store.count() == 2


def test_5_replace_existing_sources_purges_old_vectors(vector_store: VectorStoreManager) -> None:
    """5. Test that replace_existing_sources=True removes old vectors for that source before inserting."""
    c1 = Document(page_content="Old version", metadata={"vector_id": "old_1", "source": "guide.pdf", "content_hash": "h_old"})
    res1 = EmbeddingResult(chunks=[c1], vectors=[[0.1] * 384], model_name="fake", provider="fake", dimension=384, duration_seconds=0.1)
    vector_store.index(res1)
    assert vector_store.count() == 1

    c2 = Document(page_content="New version", metadata={"vector_id": "new_1", "source": "guide.pdf", "content_hash": "h_new"})
    res2 = EmbeddingResult(chunks=[c2], vectors=[[0.2] * 384], model_name="fake", provider="fake", dimension=384, duration_seconds=0.1)
    idx_res2 = vector_store.index(res2, replace_existing_sources=True)

    assert "guide.pdf" in idx_res2.replaced_sources
    assert vector_store.count() == 1
    # Check old ID is gone
    data = vector_store._get_collection().get(ids=["old_1"])
    assert len(data["ids"]) == 0


def test_6_metadata_survives_round_trip(vector_store: VectorStoreManager) -> None:
    """6. Test that category, access_level, chunk_id, and source_name survive the indexing round trip."""
    doc = Document(
        page_content="Roundtrip text content.",
        metadata={
            "vector_id": "rt_1",
            "chunk_id": "doc_d0_c0",
            "source": "manual.docx",
            "source_name": "manual.docx",
            "category": "guide",
            "access_level": "internal",
            "allowed_roles": "engineering",
            "word_count": 3,
            "content_hash": "rt_hash",
        },
    )
    res = EmbeddingResult(chunks=[doc], vectors=[[0.15] * 384], model_name="fake", provider="fake", dimension=384, duration_seconds=0.1)
    vector_store.index(res)

    results = vector_store.similarity_search("Roundtrip", k=1)
    assert len(results) == 1
    retrieved_doc, _ = results[0]

    assert retrieved_doc.metadata["chunk_id"] == "doc_d0_c0"
    assert retrieved_doc.metadata["category"] == "guide"
    assert retrieved_doc.metadata["access_level"] == "internal"
    assert retrieved_doc.metadata["source_name"] == "manual.docx"


def test_7_similarity_search_ranking_and_limit(vector_store: VectorStoreManager, fake_embeddings: DeterministicFakeEmbedding) -> None:
    """7. Test that similarity search returns at most k results and exact match text returns first."""
    target_text = "Specific target query phrase for ranking test."
    other_text = "Completely different content about planetary astrophysics."

    target_vec = fake_embeddings.embed_documents([target_text])[0]
    other_vec = fake_embeddings.embed_documents([other_text])[0]

    chunks = [
        Document(page_content=other_text, metadata={"vector_id": "v_other", "source": "s1", "content_hash": "h_other"}),
        Document(page_content=target_text, metadata={"vector_id": "v_target", "source": "s2", "content_hash": "h_target"}),
    ]
    res = EmbeddingResult(chunks=chunks, vectors=[other_vec, target_vec], model_name="fake", provider="fake", dimension=384, duration_seconds=0.1)
    vector_store.index(res)

    results = vector_store.similarity_search(target_text, k=1)
    assert len(results) == 1
    top_doc, _ = results[0]
    assert top_doc.page_content == target_text


def test_8_metadata_filters_single_and_combined(vector_store: VectorStoreManager) -> None:
    """8. Test that single and combined metadata filters function correctly."""
    docs = [
        Document(page_content="Policy doc", metadata={"vector_id": "f1", "category": "policy", "access_level": "public", "source": "s1", "content_hash": "h1"}),
        Document(page_content="FAQ doc", metadata={"vector_id": "f2", "category": "faq", "access_level": "public", "source": "s2", "content_hash": "h2"}),
        Document(page_content="Internal FAQ", metadata={"vector_id": "f3", "category": "faq", "access_level": "internal", "source": "s3", "content_hash": "h3"}),
    ]
    res = EmbeddingResult(chunks=docs, vectors=[[0.1] * 384, [0.2] * 384, [0.3] * 384], model_name="fake", provider="fake", dimension=384, duration_seconds=0.1)
    vector_store.index(res)

    # Single filter
    faq_results = vector_store.similarity_search("doc", k=5, filters={"category": "faq"})
    assert len(faq_results) == 2
    for d, _ in faq_results:
        assert d.metadata["category"] == "faq"

    # Combined filter
    combined_results = vector_store.similarity_search("doc", k=5, filters={"category": "faq", "access_level": "public"})
    assert len(combined_results) == 1
    assert combined_results[0][0].metadata["vector_id"] == "f2"


def test_9_data_persists_across_manager_reinstantiation(tmp_path: Path, fake_embeddings: DeterministicFakeEmbedding) -> None:
    """9. Test that data persists when closing the manager and opening a new one on the same directory."""
    vstore_dir = tmp_path / "persist_dir"
    m1 = VectorStoreManager(persist_directory=vstore_dir, collection_name="persist_col", embeddings=fake_embeddings)

    doc = Document(page_content="Persistent text", metadata={"vector_id": "p1", "source": "p.txt", "content_hash": "hp1"})
    res = EmbeddingResult(chunks=[doc], vectors=[[0.1] * 384], model_name="fake", provider="fake", dimension=384, duration_seconds=0.1)
    m1.index(res)
    assert m1.count() == 1
    m1.close()

    # Reopen
    m2 = VectorStoreManager(persist_directory=vstore_dir, collection_name="persist_col", embeddings=fake_embeddings)
    assert m2.count() == 1
    m2.close()


def test_10_dimension_mismatch_raises_indexing_error(vector_store: VectorStoreManager) -> None:
    """10. Test that vector dimension mismatch against existing collection raises IndexingError."""
    doc1 = Document(page_content="Dim 384 doc", metadata={"vector_id": "d1", "source": "s1", "content_hash": "hd1"})
    res1 = EmbeddingResult(chunks=[doc1], vectors=[[0.1] * 384], model_name="fake", provider="fake", dimension=384, duration_seconds=0.1)
    vector_store.index(res1)

    # Incoming result with dimension 512
    doc2 = Document(page_content="Dim 512 doc", metadata={"vector_id": "d2", "source": "s2", "content_hash": "hd2"})
    res2 = EmbeddingResult(chunks=[doc2], vectors=[[0.1] * 512], model_name="fake", provider="fake", dimension=512, duration_seconds=0.1)

    with pytest.raises(IndexingError, match="Vector dimension mismatch"):
        vector_store.index(res2)


def test_11_collection_utility_methods(vector_store: VectorStoreManager) -> None:
    """11. Test delete_by_source, list_sources, and reset_collection behaviors."""
    docs = [
        Document(page_content="Doc A", metadata={"vector_id": "u1", "source": "src_a.pdf", "content_hash": "ha"}),
        Document(page_content="Doc B", metadata={"vector_id": "u2", "source": "src_b.pdf", "content_hash": "hb"}),
    ]
    res = EmbeddingResult(chunks=docs, vectors=[[0.1] * 384, [0.2] * 384], model_name="fake", provider="fake", dimension=384, duration_seconds=0.1)
    vector_store.index(res)

    sources = vector_store.list_sources()
    assert sources == ["src_a.pdf", "src_b.pdf"]

    deleted = vector_store.delete_by_source("src_a.pdf")
    assert deleted == 1
    assert vector_store.count() == 1
    assert vector_store.list_sources() == ["src_b.pdf"]

    vector_store.reset_collection()
    assert vector_store.count() == 0
    assert vector_store.list_sources() == []


def test_12_as_langchain_vectorstore_handoff(vector_store: VectorStoreManager) -> None:
    """12. Test that as_langchain_vectorstore returns a Chroma instance capable of similarity search."""
    doc = Document(page_content="Handoff test document", metadata={"vector_id": "h1", "source": "h.txt", "content_hash": "hh"})
    res = EmbeddingResult(chunks=[doc], vectors=[[0.1] * 384], model_name="fake", provider="fake", dimension=384, duration_seconds=0.1)
    vector_store.index(res)

    lc_vs = vector_store.as_langchain_vectorstore()
    assert isinstance(lc_vs, Chroma)

    results = lc_vs.similarity_search("Handoff", k=1)
    assert len(results) == 1
    assert results[0].page_content == "Handoff test document"


def test_13_collection_configured_with_cosine_distance(vector_store: VectorStoreManager) -> None:
    """13. Test that the created Chroma collection explicitly uses cosine distance metric."""
    col = vector_store._get_collection()
    assert col.metadata.get("hnsw:space") == "cosine"
