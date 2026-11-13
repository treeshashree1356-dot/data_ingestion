"""Full end-to-end integration tests for stages 1 through 6 of the RAG pipeline.

Covers:
Sources -> Extraction -> Chunking -> Metadata Enrichment -> Embedding -> Vector Indexing -> Semantic Search.
Tests execution both through programmatic service coordination and Streamlit AppTest UI clicks.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
import pytest
from streamlit.testing.v1 import AppTest

from services.indexing_service import IndexingService
from services.ingestion_service import IngestionService

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_1_full_rag_pipeline_via_services(vector_store, fake_embeddings: DeterministicFakeEmbedding) -> None:
    """1. Test full end-to-end flow from manual text through services to search verification."""
    ingestion_svc = IngestionService()
    indexing_svc = IndexingService(
        embedding_manager=None,
        vector_store=vector_store,
    )
    # Inject fake embeddings into vector_store and embedding_manager
    indexing_svc.embedding_manager._embeddings = fake_embeddings

    # 1. Extraction from manual text
    text_content = (
        "Canonix Engineering Guide: Microservice communication standards. "
        "All internal RPC calls must enforce TLS 1.3 encryption and mutual authentication."
    )
    extract_res = ingestion_svc.process_sources(
        manual_text=text_content,
        manual_title="engineering_standards",
    )
    assert extract_res.successful_count == 1
    assert len(extract_res.documents) == 1

    # 2. Chunking
    chunks = ingestion_svc.chunk_documents(extract_res.documents, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 1

    # 3. Enrichment
    enriched_chunks = indexing_svc.enrich_chunks(chunks)
    assert len(enriched_chunks) == len(chunks)
    assert enriched_chunks[0].metadata["category"] == "guide"
    assert "vector_id" in enriched_chunks[0].metadata

    # 4. Embedding
    emb_res = indexing_svc.embed_chunks(enriched_chunks)
    assert len(emb_res.vectors) == len(chunks)
    assert emb_res.dimension == 384

    # 5. Indexing
    idx_res = indexing_svc.index_embeddings(emb_res)
    assert idx_res.indexed_count == len(chunks)
    assert vector_store.count() == len(chunks)

    # 6. Semantic Search
    search_res = indexing_svc.search(query="TLS 1.3 mutual authentication", k=1)
    assert len(search_res) == 1
    found_doc, score = search_res[0]
    assert "TLS 1.3" in found_doc.page_content


def test_2_full_rag_pipeline_via_apptest_ui_interactions() -> None:
    """2. Test full end-to-end flow driven entirely through AppTest button clicks and widget inputs."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception

    # Step 1: Input manual text
    prompt_text = "Security compliance rules dictate that all audit logs must be retained for 7 years."
    at.text_area(key="source_manual_text").input(prompt_text).run()

    # Step 2: Click Extract
    at.button(key="btn_extract_data").click().run()
    assert not at.exception
    assert len(at.session_state["extracted_documents"]) == 1

    # Step 3: Click Generate Chunks
    at.button(key="btn_generate_chunks").click().run()
    assert not at.exception
    assert len(at.session_state["generated_chunks"]) >= 1

    # Step 4: Click Enrich Metadata
    at.button(key="btn_enrich_metadata").click().run()
    assert not at.exception
    assert len(at.session_state["enriched_chunks"]) >= 1
    assert at.session_state["enriched_chunks"][0].metadata["category"] in ("policy", "rules", "general")

    # Step 5: Click Generate Embeddings
    at.button(key="btn_generate_embeddings").click().run()
    assert not at.exception
    assert at.session_state["embedding_result"] is not None

    # Step 6: Click Index into Vector Database
    at.button(key="btn_index_vectors").click().run()
    assert not at.exception
    assert at.session_state["indexing_result"] is not None
    assert at.session_state["indexing_result"].indexed_count >= 1

    # Step 7: Test Retrieval Search
    at.text_input(key="input_test_query").input("audit logs retention").run()
    at.button(key="btn_test_search").click().run()
    assert not at.exception

    results = at.session_state.get("search_results", [])
    assert len(results) >= 1
    assert "audit logs" in results[0][0].page_content
