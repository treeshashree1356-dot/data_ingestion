"""UI component for executing extraction, previewing extracted text, and monitoring status."""

import streamlit as st
from services.ingestion_service import IngestionResult, ingestion_service
from ui.state import reset_downstream


def render_extraction_view() -> None:
    """Render the data extraction interface and preview extracted documents."""
    st.subheader("2. Data Extraction & Document Preview")
    st.caption("Trigger ingestion and preview extracted text and metadata.")

    source_inputs = st.session_state.get("source_inputs", {})
    uploaded_files = source_inputs.get("uploaded_files", [])
    urls = source_inputs.get("urls", [])
    manual_text = source_inputs.get("manual_text", "")
    manual_title = source_inputs.get("manual_title", "notes")

    has_inputs = bool(uploaded_files or urls or manual_text)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        extract_clicked = st.button(
            "🚀 Extract Data",
            type="primary",
            use_container_width=True,
            key="btn_extract_data",
        )

    with col_info:
        if not has_inputs:
            st.info("👈 Please select or enter at least one knowledge source above to extract.")

    if extract_clicked:
        if not has_inputs:
            st.warning("No knowledge sources provided. Upload files, provide URLs, or enter manual text.")
            return

        with st.spinner("Extracting content from sources..."):
            progress_bar = st.progress(0, text="Initializing extraction...")
            progress_bar.progress(30, text="Processing files and URLs...")

            # Business logic delegation to IngestionService
            result = ingestion_service.process_sources(
                uploaded_files=uploaded_files,
                urls=urls,
                manual_text=manual_text,
                manual_title=manual_title,
            )

            progress_bar.progress(100, text="Extraction complete!")
            st.session_state["extraction_result"] = result
            st.session_state["extracted_documents"] = result.documents

            # Invalidate previously generated chunks because source documents changed
            st.session_state["generated_chunks"] = []
            reset_downstream("extraction")

    # Display Results & Diagnostics
    result: IngestionResult | None = st.session_state.get("extraction_result")
    if result:
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Documents Extracted", len(result.documents))
        mcol2.metric("Successful Sources", result.successful_count)
        mcol3.metric("Failed Sources", result.failed_count)
        mcol4.metric(
            "Total Characters",
            sum(len(d.page_content) for d in result.documents),
        )

        # Show extraction errors if any occurred
        if result.errors:
            error_details = "\n".join([f"- **{err['source']}**: `{err['error']}`" for err in result.errors])
            st.error(f"⚠️ {len(result.errors)} extraction issue(s) encountered:\n{error_details}")

        # Display extracted document previews
        if result.documents:
            st.markdown("### 📄 Extracted Document Previews")
            for idx, doc in enumerate(result.documents):
                meta = doc.metadata or {}
                source_label = meta.get("filename") or meta.get("source") or f"Document {idx+1}"
                source_type = meta.get("source_type", "document").upper()
                page_info = f" | Page {meta['page_number']}" if "page_number" in meta else ""
                char_info = f" | {len(doc.page_content):,} chars"

                with st.expander(f"[{source_type}] {source_label}{page_info}{char_info}"):
                    # Metadata pill summary
                    st.json(meta, expanded=False)
                    st.text_area(
                        "Extracted Content Preview",
                        value=doc.page_content,
                        height=160,
                        disabled=True,
                        key=f"extracted_preview_{idx}",
                    )
        elif not result.errors:
            st.info("No documents extracted from the provided sources.")
