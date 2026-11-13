"""UI component for vector store indexing, collection management, and retrieval verification."""

import streamlit as st

from config.settings import settings
from enrichment.category_rules import CATEGORY_RULES
from enrichment.metadata_enricher import ALLOWED_ACCESS_LEVELS
from indexing.vector_store import IndexingError
from services.indexing_service import indexing_service


def render_indexing_view() -> None:
    """Render the vector database indexing controls, statistics, and retrieval preview."""
    st.subheader("6. Vector Store Indexing & Retrieval Verification")
    st.caption("Store high-dimensional vectors and metadata in persistent ChromaDB for fast semantic search.")

    emb_result = st.session_state.get("embedding_result")
    has_embeddings = emb_result is not None and bool(emb_result.vectors)

    if not has_embeddings:
        st.info("👈 Generate embeddings in Step 5 before indexing into vector store.")

    col1, col2 = st.columns([2, 1])
    with col1:
        replace_sources = st.checkbox(
            "Replace existing vectors for these sources",
            value=False,
            help="Purge prior records matching the incoming documents' sources before inserting.",
            key="chk_replace_sources",
        )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        index_clicked = st.button(
            "🗄️ Index into Vector Database",
            type="primary",
            use_container_width=True,
            disabled=not has_embeddings,
            key="btn_index_vectors",
        )

    if index_clicked and has_embeddings:
        try:
            with st.spinner("Indexing vector embeddings and sanitized metadata into ChromaDB..."):
                idx_res = indexing_service.index_embeddings(
                    emb_result,
                    replace_existing_sources=replace_sources,
                )
                st.session_state["indexing_result"] = idx_res
                st.success(f"✅ Successfully indexed **{idx_res.indexed_count} vector(s)**!")
        except IndexingError as exc:
            st.error(f"⚠️ Indexing error: {exc}")
        except Exception as exc:
            st.error(f"⚠️ An unexpected error occurred while indexing: {exc}")

    # Display Indexing Metrics
    idx_result = st.session_state.get("indexing_result")
    stats = indexing_service.get_collection_stats()

    if idx_result or stats["count"] > 0:
        st.markdown("---")
        st.markdown("### 📊 Vector Store Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Indexed in Last Run", idx_result.indexed_count if idx_result else 0)
        m2.metric("Duplicates Skipped", idx_result.skipped_duplicates if idx_result else 0)
        m3.metric("Total in Collection", stats["count"])
        m4.metric("Collection Name", stats["collection_name"])

        st.caption(f"📁 Persist Directory: `{stats['persist_directory']}`")

    # Test Retrieval Section
    st.markdown("---")
    st.markdown("### 🔎 Test Retrieval")
    st.caption("Verify retrieval sanity with semantic similarity search and metadata filtering.")

    q_col1, q_col2 = st.columns([3, 1])
    with q_col1:
        test_query = st.text_input(
            "Test Query",
            placeholder="e.g. What are the policy guidelines for code of conduct?",
            key="input_test_query",
        )
    with q_col2:
        top_k = st.number_input(
            "Top K Results",
            min_value=1,
            max_value=20,
            value=int(getattr(settings, "RETRIEVAL_TOP_K", 4)),
            key="input_top_k",
        )

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        filter_cat = st.selectbox(
            "Filter Category (Optional)",
            options=["All"] + list(CATEGORY_RULES.keys()),
            index=0,
            key="select_filter_category",
        )
    with f_col2:
        filter_level = st.selectbox(
            "Filter Access Level (Optional)",
            options=["All"] + list(ALLOWED_ACCESS_LEVELS),
            index=0,
            key="select_filter_access_level",
        )

    search_clicked = st.button("Search", type="secondary", key="btn_test_search")

    if search_clicked:
        if not test_query or not test_query.strip():
            st.warning("Please enter a non-empty query string.")
        elif stats["count"] == 0:
            st.warning("Vector database collection is currently empty. Please index documents first.")
        else:
            filters = {}
            if filter_cat != "All":
                filters["category"] = filter_cat
            if filter_level != "All":
                filters["access_level"] = filter_level

            with st.spinner("Querying vector database..."):
                try:
                    search_results = indexing_service.search(
                        query=test_query.strip(),
                        k=top_k,
                        filters=filters or None,
                    )
                    st.session_state["search_results"] = search_results
                except Exception as exc:
                    st.error(f"Search failed: {exc}")

    search_results = st.session_state.get("search_results", [])
    if search_results:
        st.markdown(f"**Found {len(search_results)} result(s):**")
        for idx, (doc, score) in enumerate(search_results):
            meta = doc.metadata or {}
            chunk_id = meta.get("chunk_id", f"chunk_{idx}")
            src_name = meta.get("source_name", meta.get("source", "Unknown"))
            cat = meta.get("category", "General")
            score_text = f"{score:.4f}" if isinstance(score, float) else str(score)

            with st.expander(f"#{idx+1} [Score: {score_text}] [{cat}] {chunk_id} - {src_name}"):
                st.write(f"**Score:** `{score_text}` | **Category:** `{cat}` | **Access Level:** `{meta.get('access_level', '')}`")
                st.write(f"**Source:** `{src_name}`")
                st.markdown("**Chunk Text:**")
                st.info(doc.page_content)
                with st.expander("Raw Metadata"):
                    st.json(meta)

    # Danger Zone: Reset Collection
    st.markdown("---")
    with st.expander("⚠️ Danger Zone: Reset Collection"):
        st.write("Resetting the collection will permanently delete all stored vectors and documents.")
        confirm_reset = st.checkbox(
            "I understand that this action is irreversible.",
            value=False,
            key="chk_confirm_reset",
        )
        if st.button("🗑️ Reset Collection", disabled=not confirm_reset, key="btn_reset_collection"):
            indexing_service.vector_store.reset_collection()
            st.session_state["indexing_result"] = None
            st.session_state["search_results"] = []
            st.success("Vector collection has been completely reset.")
            st.rerun()
