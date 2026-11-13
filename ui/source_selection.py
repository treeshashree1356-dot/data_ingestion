"""UI component for knowledge source selection, multi-file uploads, URLs, and manual text."""

from typing import Any
import streamlit as st


def render_source_selection() -> dict[str, Any]:
    """Render the knowledge source selection interface.

    Allows users to:
    - Upload multiple PDF, DOCX, TXT, Markdown, JPG, JPEG, and PNG files.
    - Input multiple website URLs.
    - Paste manual text notes.

    Returns:
        Dictionary containing the user-provided inputs.
    """
    st.subheader("1. Select Knowledge Sources")
    st.caption("Upload documents/images, supply live URLs, or enter manual text.")

    # 1. File Uploader
    uploaded_files = st.file_uploader(
        "Upload Documents & Images",
        type=["pdf", "docx", "txt", "md", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, Markdown, JPG, JPEG, PNG",
        key="source_file_uploader",
    )

    col1, col2 = st.columns(2)

    with col1:
        # 2. Website URLs
        urls_input = st.text_area(
            "Website URLs (One per line)",
            placeholder="https://en.wikipedia.org/wiki/Information_extraction\nhttps://example.com/guide",
            help="Enter one HTTP or HTTPS website URL per line.",
            key="source_urls_input",
            height=130,
        )

    with col2:
        # 3. Manual Text
        manual_title = st.text_input(
            "Manual Text Title / Label",
            value="quick_notes",
            help="Optional label to identify manual text snippets.",
            key="source_manual_title",
        )
        manual_text = st.text_area(
            "Paste Manual Text",
            placeholder="Paste raw notes, research excerpts, or transcription text here...",
            key="source_manual_text",
            height=85,
        )

    # Parse and clean URLs
    parsed_urls = [u.strip() for u in urls_input.splitlines() if u.strip()] if urls_input else []

    source_data = {
        "uploaded_files": uploaded_files or [],
        "urls": parsed_urls,
        "manual_text": manual_text.strip() if manual_text else "",
        "manual_title": manual_title.strip() if manual_title else "manual_text",
    }

    st.session_state["source_inputs"] = source_data
    return source_data
