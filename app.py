"""Streamlit Application Entrypoint for Data Ingestion Pipeline.

Connects modular UI components, manages persistent session state, and orchestrates
the ingestion, chunking, enrichment, embedding, and indexing workflows without
duplicating business logic.
"""

import streamlit as st

from config.settings import settings
from ui.chunking_view import render_chunking_view
from ui.embedding_view import render_embedding_view
from ui.extraction_view import render_extraction_view
from ui.indexing_view import render_indexing_view
from ui.metadata_view import render_metadata_view
from ui.source_selection import render_source_selection


def init_session_state() -> None:
    """Initialize persistent Streamlit session state keys with safe defaults."""
    if "extracted_documents" not in st.session_state:
        st.session_state["extracted_documents"] = []

    if "extraction_result" not in st.session_state:
        st.session_state["extraction_result"] = None

    if "generated_chunks" not in st.session_state:
        st.session_state["generated_chunks"] = []

    if "chunk_size" not in st.session_state:
        st.session_state["chunk_size"] = getattr(settings, "DEFAULT_CHUNK_SIZE", 1000)

    if "chunk_overlap" not in st.session_state:
        st.session_state["chunk_overlap"] = getattr(settings, "DEFAULT_CHUNK_OVERLAP", 200)

    if "source_inputs" not in st.session_state:
        st.session_state["source_inputs"] = {}

    # Downstream pipeline session keys
    if "enriched_chunks" not in st.session_state:
        st.session_state["enriched_chunks"] = []

    if "embedding_result" not in st.session_state:
        st.session_state["embedding_result"] = None

    if "indexing_result" not in st.session_state:
        st.session_state["indexing_result"] = None

    if "enrichment_options" not in st.session_state:
        st.session_state["enrichment_options"] = {}

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []


def main() -> None:
    """Main application layout and dashboard orchestration."""
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon="📥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Header section
    st.title(f"📥 {settings.APP_TITLE}")
    st.markdown(
        """
        Unified document ingestion, extraction, chunking, enrichment, embedding, and indexing pipeline.
        Extract text from **PDFs, Word DOCX, Plain Text, Markdown, Images (OCR), and Web URLs**, 
        segment content using **LangChain RecursiveCharacterTextSplitter**, enrich with governance metadata,
        generate local semantic embeddings via **HuggingFace**, and index vectors into **ChromaDB**.
        """
    )
    st.markdown("---")

    # Sidebar: Pipeline Status & Reset
    with st.sidebar:
        st.header("⚙️ Pipeline Status")
        st.caption(f"Environment: **{settings.APP_ENV}**")

        extracted_count = len(st.session_state.get("extracted_documents", []))
        chunks_count = len(st.session_state.get("generated_chunks", []))
        enriched_count = len(st.session_state.get("enriched_chunks", []))

        emb_result = st.session_state.get("embedding_result")
        vectors_count = len(emb_result.vectors) if emb_result and hasattr(emb_result, "vectors") else 0

        idx_result = st.session_state.get("indexing_result")
        indexed_count = idx_result.indexed_count if idx_result and hasattr(idx_result, "indexed_count") else 0

        st.metric("Documents Ingested", extracted_count)
        st.metric("Total Chunks Ready", chunks_count)
        st.metric("Enriched Chunks", enriched_count)
        st.metric("Vectors Embedded", vectors_count)
        st.metric("Vectors Indexed", indexed_count)

        st.markdown("---")
        st.markdown("### 🛠️ Configuration")
        st.write(f"- **Max File Size:** {settings.MAX_FILE_SIZE_MB} MB")
        st.write(f"- **Default Chunk:** {settings.DEFAULT_CHUNK_SIZE} chars")
        st.write(f"- **Default Overlap:** {settings.DEFAULT_CHUNK_OVERLAP} chars")
        st.write(f"- **Embedding Model:** {settings.EMBEDDING_MODEL_NAME.split('/')[-1]}")
        st.write(f"- **Collection Name:** {settings.VECTOR_COLLECTION_NAME}")

        st.markdown("---")
        if st.button("🔄 Reset Pipeline State", use_container_width=True):
            st.session_state["extracted_documents"] = []
            st.session_state["extraction_result"] = None
            st.session_state["generated_chunks"] = []
            st.session_state["source_inputs"] = {}
            st.session_state["enriched_chunks"] = []
            st.session_state["embedding_result"] = None
            st.session_state["indexing_result"] = None
            st.session_state["enrichment_options"] = {}
            st.session_state["search_results"] = []
            st.rerun()

    # Main Dashboard Tabs
    (
        tab_sources,
        tab_extract,
        tab_chunk,
        tab_enrich,
        tab_embed,
        tab_index,
    ) = st.tabs([
        "1️⃣ Knowledge Sources",
        "2️⃣ Data Extraction",
        "3️⃣ Chunking & Export",
        "4️⃣ Metadata Enrichment",
        "5️⃣ Embedding",
        "6️⃣ Indexing",
    ])

    with tab_sources:
        render_source_selection()

    with tab_extract:
        render_extraction_view()

    with tab_chunk:
        render_chunking_view()

    with tab_enrich:
        render_metadata_view()

    with tab_embed:
        render_embedding_view()

    with tab_index:
        render_indexing_view()


if __name__ == "__main__":
    main()
