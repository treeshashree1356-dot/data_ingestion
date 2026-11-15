"""UI component for vector store indexing, collection management, and retrieval verification."""

import streamlit as st

from config.settings import settings
from enrichment.category_rules import CATEGORY_RULES
from enrichment.metadata_enricher import ALLOWED_ACCESS_LEVELS
from indexing.vector_store import IndexingError
from query.query_schemas import QueryContext, QueryProcessingError, QueryValidationError
from services.indexing_service import indexing_service
from services.query_service import query_service


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
    st.markdown("### 🔎 Test Retrieval & Query Processing")
    st.caption("Verify retrieval sanity with user query processing, intent detection, category inference, and RBAC filtering.")

    q_col1, q_col2 = st.columns([3, 1])
    with q_col1:
        test_query = st.text_input(
            "User Query",
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

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        user_role = st.selectbox(
            "User Role (RBAC)",
            options=["all", "admin", "engineering", "internal", "guest"],
            index=0,
            help="Simulates the user's authorization role to test RBAC query filtering.",
            key="select_user_role",
        )
    with f_col2:
        filter_cat = st.selectbox(
            "Category Filter",
            options=["All (Auto-detect)"] + list(CATEGORY_RULES.keys()),
            index=0,
            help="Override or allow automatic keyword detection for category classification.",
            key="select_filter_category",
        )
    with f_col3:
        filter_level = st.selectbox(
            "Access Level Filter",
            options=["All (Derive from role)"] + list(ALLOWED_ACCESS_LEVELS),
            index=0,
            help="Override or allow automatic derivation of access levels from user role.",
            key="select_filter_access_level",
        )

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        auto_detect_cat = st.checkbox(
            "Auto-detect category from query keywords",
            value=True,
            help="Uses existing CATEGORY_RULES keyword patterns to infer the category.",
            key="chk_auto_detect_category",
        )
    with opt_col2:
        enable_expansion = st.checkbox(
            "Enable query expansion (acronyms & synonyms)",
            value=False,
            help="Expands technical abbreviations (e.g. 'auth', 'k8s', 'rpc') to improve recall.",
            key="chk_enable_expansion",
        )

    search_clicked = st.button("🔎 Process Query & Search", type="secondary", key="btn_test_search")

    if search_clicked:
        if not test_query or not test_query.strip():
            st.warning("Please enter a non-empty query string.")
        elif stats["count"] == 0:
            st.warning("Vector database collection is currently empty. Please index documents first.")
        else:
            cat_override = None if filter_cat.startswith("All") else filter_cat
            level_override = None if filter_level.startswith("All") else filter_level

            query_ctx = QueryContext(
                user_role=user_role,
                target_category=cat_override,
                access_level=level_override,
                top_k=top_k,
                auto_detect_category=auto_detect_cat,
                enable_expansion=enable_expansion,
            )

            with st.spinner("Processing query and querying vector database..."):
                try:
                    response = query_service.execute_query(
                        query=test_query,
                        context=query_ctx,
                    )
                    st.session_state["query_response"] = response
                    st.session_state["search_results"] = response.results
                except (QueryValidationError, QueryProcessingError) as exc:
                    st.error(f"Query Error: {exc}")
                except Exception as exc:
                    st.error(f"Search failed: {exc}")

    # Display Query Diagnostics
    query_response = st.session_state.get("query_response")
    if query_response and hasattr(query_response, "processed_query"):
        pq = query_response.processed_query
        with st.expander("🛠️ Query Diagnostics & Pipeline Breakdown", expanded=True):
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Intent", pq.intent.intent_type.value)
            d2.metric("Intent Confidence", f"{pq.intent.confidence:.0%}")
            d3.metric("Detected Category", pq.detected_category or "None")
            d4.metric("Latency", f"{query_response.duration_seconds * 1000:.1f} ms")

            st.write(f"- **Cleaned Query:** `{pq.cleaned_query}`")
            st.write(f"- **Intent Description:** {pq.intent.description}")
            if pq.expanded_terms:
                st.write(f"- **Expanded Terms:** `{', '.join(pq.expanded_terms)}`")

            st.markdown("**Applied ChromaDB Filter Payload:**")
            st.json(pq.filters if pq.filters else {"status": "No filters applied (unrestricted)"})

    # Display Search Results
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
            st.session_state["query_response"] = None
            st.success("Vector collection has been completely reset.")
            st.rerun()
