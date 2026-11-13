"""UI component for chunking configuration, chunk visualization, and JSON export."""

import streamlit as st
from services.ingestion_service import ingestion_service
from ui.state import reset_downstream


def render_chunking_view() -> None:
    """Render the document chunking interface and chunk inspection dashboard."""
    st.subheader("3. Document Chunking & Export")
    st.caption("Segment extracted documents using LangChain RecursiveCharacterTextSplitter.")

    extracted_docs = st.session_state.get("extracted_documents", [])
    has_docs = bool(extracted_docs)

    # Chunking Controls
    col1, col2 = st.columns(2)
    with col1:
        current_size = st.session_state.get("chunk_size", 1000)
        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=1,
            max_value=20000,
            value=current_size,
            step=50,
            help="Maximum character length per chunk.",
            key="input_chunk_size",
        )

    with col2:
        current_overlap = st.session_state.get("chunk_overlap", 200)
        chunk_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=5000,
            value=current_overlap,
            step=25,
            help="Overlapping characters between adjacent chunks. Must be smaller than chunk size.",
            key="input_chunk_overlap",
        )

    # Invalidate old chunks if chunk settings changed
    if (chunk_size != current_size) or (chunk_overlap != current_overlap):
        st.session_state["chunk_size"] = chunk_size
        st.session_state["chunk_overlap"] = chunk_overlap
        st.session_state["generated_chunks"] = []
        reset_downstream("chunking")

    # Validation message
    is_config_valid = chunk_size > 0 and chunk_overlap >= 0 and chunk_overlap < chunk_size
    if not is_config_valid:
        if chunk_overlap >= chunk_size:
            st.error(f"⚠️ Chunk overlap ({chunk_overlap}) must be strictly smaller than chunk size ({chunk_size}).")
        elif chunk_size <= 0:
            st.error("⚠️ Chunk size must be greater than zero.")

    col_btn, col_msg = st.columns([1, 3])
    with col_btn:
        generate_clicked = st.button(
            "✂️ Generate Chunks",
            type="primary",
            use_container_width=True,
            disabled=not (has_docs and is_config_valid),
            key="btn_generate_chunks",
        )

    with col_msg:
        if not has_docs:
            st.info("👈 Extract documents in Step 2 before generating chunks.")

    if generate_clicked and has_docs and is_config_valid:
        with st.spinner("Generating chunks..."):
            chunks = ingestion_service.chunk_documents(
                documents=extracted_docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state["generated_chunks"] = chunks
            reset_downstream("chunking")

    # Chunk Display & Export
    generated_chunks = st.session_state.get("generated_chunks", [])
    if generated_chunks:
        st.markdown("---")
        hcol1, hcol2 = st.columns([2, 1])
        with hcol1:
            st.success(
                f"✅ Successfully created **{len(generated_chunks):,} chunks** "
                f"from {len(extracted_docs)} source document(s)!"
            )
        with hcol2:
            # Reusable JSON export functionality
            json_payload = ingestion_service.export_chunks_json(generated_chunks)
            st.download_button(
                label="📥 Download Chunks (JSON)",
                data=json_payload,
                file_name="chunks_export.json",
                mime="application/json",
                use_container_width=True,
                key="btn_download_json",
            )

        # Chunk inspection list
        st.markdown("### 🔍 Individual Chunk Inspector")
        for idx, chunk in enumerate(generated_chunks):
            meta = chunk.metadata or {}
            chunk_id = meta.get("chunk_id", f"chunk_{idx}")
            source = meta.get("filename") or meta.get("source") or "unknown"
            char_len = len(chunk.page_content)
            title = f"#{idx+1} | ID: {chunk_id} | Source: {source} ({char_len:,} chars)"

            with st.expander(title):
                st.json(meta, expanded=False)
                st.text_area(
                    "Chunk Content",
                    value=chunk.page_content,
                    height=120,
                    disabled=True,
                    key=f"chunk_view_{idx}",
                )
