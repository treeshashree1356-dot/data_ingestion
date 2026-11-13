"""UI integration tests for metadata enrichment, embedding, and indexing tabs.

Uses Streamlit AppTest to verify rendering of 6 tabs, button states, session state
transitions, downstream resets, and lazy embedding loading.
"""

from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_1_app_renders_six_tabs_without_exception() -> None:
    """1. Test that the Streamlit app renders all 6 tabs without throwing exceptions."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert len(at.tabs) == 6


def test_2_buttons_disabled_or_show_info_when_empty() -> None:
    """2. Test that enrichment, embedding, and indexing buttons are disabled or show info messages when inputs are empty."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    btn_enrich = at.button(key="btn_enrich_metadata")
    assert btn_enrich.disabled is True

    btn_embed = at.button(key="btn_generate_embeddings")
    assert btn_embed.disabled is True

    btn_index = at.button(key="btn_index_vectors")
    assert btn_index.disabled is True

    # Confirm guidance info messages are present
    info_messages = [info.value for info in at.info]
    assert any("Generate chunks in Step 3" in msg for msg in info_messages)
    assert any("Enrich metadata in Step 4" in msg for msg in info_messages)
    assert any("Generate embeddings in Step 5" in msg for msg in info_messages)


def test_3_seeding_chunks_and_enriching_fills_state() -> None:
    """3. Test that seeding generated_chunks and clicking btn_enrich_metadata populates enriched_chunks."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    at.session_state["generated_chunks"] = [
        Document(page_content="Policy guidelines handbook.", metadata={"source": "policy.pdf", "chunk_id": "c1"})
    ]
    at.run()

    btn_enrich = at.button(key="btn_enrich_metadata")
    assert btn_enrich.disabled is False

    btn_enrich.click().run()
    assert not at.exception
    assert len(at.session_state["enriched_chunks"]) == 1
    assert at.session_state["enriched_chunks"][0].metadata["category"] == "policy"


def test_4_generating_embeddings_fills_state() -> None:
    """4. Test that clicking btn_generate_embeddings populates embedding_result."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    at.session_state["enriched_chunks"] = [
        Document(
            page_content="Semantic embedding test chunk.",
            metadata={
                "source": "manual.pdf",
                "chunk_id": "c1",
                "content_hash": "h1",
                "vector_id": "v1",
                "category": "guide",
                "access_level": "internal",
                "allowed_roles": "all",
                "source_name": "manual.pdf",
                "source_type": "pdf",
                "ingested_at": "2026-01-01T00:00:00Z",
                "enriched_at": "2026-01-01T00:00:00Z",
                "doc_id": "d1",
                "word_count": 5,
                "char_count": 30,
                "tags": "",
                "enrichment_version": "1.0",
            },
        )
    ]
    at.run()

    btn_embed = at.button(key="btn_generate_embeddings")
    assert btn_embed.disabled is False

    btn_embed.click().run()
    assert not at.exception
    res = at.session_state["embedding_result"]
    assert res is not None
    assert len(res.vectors) == 1
    assert res.dimension == 384


def test_5_indexing_vectors_fills_result() -> None:
    """5. Test that clicking btn_index_vectors populates indexing_result with correct count."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    doc = Document(
        page_content="Index test chunk content.",
        metadata={
            "source": "test.txt",
            "chunk_id": "c1",
            "vector_id": "idx_test_v1",
            "content_hash": "idx_test_h1",
            "category": "guide",
            "access_level": "internal",
            "allowed_roles": "all",
            "source_name": "test.txt",
            "source_type": "text",
            "ingested_at": "2026-01-01T00:00:00Z",
            "enriched_at": "2026-01-01T00:00:00Z",
            "doc_id": "d1",
            "word_count": 4,
            "char_count": 25,
            "tags": "",
            "enrichment_version": "1.0",
        },
    )
    from embedding.embedding_manager import EmbeddingResult
    at.session_state["embedding_result"] = EmbeddingResult(
        chunks=[doc],
        vectors=[[0.05] * 384],
        model_name="fake",
        provider="fake",
        dimension=384,
        duration_seconds=0.1,
    )
    at.run()

    btn_index = at.button(key="btn_index_vectors")
    assert btn_index.disabled is False

    btn_index.click().run()
    assert not at.exception
    idx_res = at.session_state["indexing_result"]
    assert idx_res is not None
    assert idx_res.indexed_count == 1


def test_6_test_search_returns_results_after_indexing() -> None:
    """6. Test that Test Retrieval search button returns results after vectors are indexed."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    doc = Document(
        page_content="Specific searchable retrieval phrase.",
        metadata={
            "source": "search_doc.txt",
            "chunk_id": "s_c1",
            "vector_id": "search_v1",
            "content_hash": "search_h1",
            "category": "faq",
            "access_level": "public",
            "allowed_roles": "all",
            "source_name": "search_doc.txt",
            "source_type": "text",
            "ingested_at": "2026-01-01T00:00:00Z",
            "enriched_at": "2026-01-01T00:00:00Z",
            "doc_id": "sd1",
            "word_count": 4,
            "char_count": 36,
            "tags": "",
            "enrichment_version": "1.0",
        },
    )
    from embedding.embedding_manager import EmbeddingManager
    mgr = EmbeddingManager()
    emb_res = mgr.embed_chunks([doc])
    at.session_state["embedding_result"] = emb_res
    at.run()

    # Index first
    at.button(key="btn_index_vectors").click().run()
    assert not at.exception

    # Execute search
    at.text_input(key="input_test_query").input("searchable retrieval phrase").run()
    at.button(key="btn_test_search").click().run()

    assert not at.exception
    search_results = at.session_state.get("search_results", [])
    assert len(search_results) >= 1
    assert any("searchable retrieval phrase" in res[0].page_content for res in search_results)


def test_7_regenerating_chunks_resets_downstream() -> None:
    """7. Test that regenerating chunks clears enriched_chunks, embedding_result, and indexing_result."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    at.session_state["extracted_documents"] = [Document(page_content="Extracted doc for chunk regeneration.")]
    at.session_state["enriched_chunks"] = [Document(page_content="Old enriched chunk")]
    at.session_state["embedding_result"] = "mock_embedding"
    at.session_state["indexing_result"] = "mock_indexing"
    at.run()

    at.button(key="btn_generate_chunks").click().run()

    assert len(at.session_state["enriched_chunks"]) == 0
    assert at.session_state["embedding_result"] is None
    assert at.session_state["indexing_result"] is None


def test_8_sidebar_reset_clears_all_new_keys() -> None:
    """8. Test that the sidebar Reset Pipeline State button clears all new session keys."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    at.session_state["enriched_chunks"] = [Document(page_content="Sample")]
    at.session_state["embedding_result"] = "res"
    at.session_state["indexing_result"] = "idx"
    at.session_state["search_results"] = ["sr"]
    at.run()

    # Find and click reset button in sidebar
    reset_btn = at.sidebar.button[0]
    reset_btn.click().run()

    assert at.session_state["enriched_chunks"] == []
    assert at.session_state["embedding_result"] is None
    assert at.session_state["indexing_result"] is None
    assert at.session_state["search_results"] == []


def test_9_app_startup_does_not_call_embedding_factory() -> None:
    """9. Test that launching the application does not invoke the embedding model factory."""
    with patch("embedding.embedding_factory.get_embedding_model") as mock_factory:
        at = AppTest.from_file(APP_PATH)
        at.run()

        mock_factory.assert_not_called()
        assert not at.exception
