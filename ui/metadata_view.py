"""UI component for metadata enrichment, governance tagging, and categorization preview."""

import pandas as pd
import streamlit as st

from config.settings import settings
from enrichment.category_rules import CATEGORY_RULES
from enrichment.metadata_enricher import ALLOWED_ACCESS_LEVELS, EnrichmentOptions
from services.indexing_service import indexing_service
from ui.state import reset_downstream


def render_metadata_view() -> None:
    """Render the metadata enrichment interface, controls, and chunk inspection table."""
    st.subheader("4. Metadata Enrichment & Governance")
    st.caption("Classify categories, assign access permissions, and generate identity hashes.")

    generated_chunks = st.session_state.get("generated_chunks", [])
    has_chunks = bool(generated_chunks)

    if not has_chunks:
        st.info("👈 Generate chunks in Step 3 first.")

    # Enrichment Configuration Controls
    col1, col2 = st.columns(2)
    with col1:
        category_options = ["Auto-detect"] + list(CATEGORY_RULES.keys())
        selected_category = st.selectbox(
            "Category Override (Optional)",
            options=category_options,
            index=0,
            help="Force a specific category or allow auto-detection based on rules and keywords.",
            key="select_category_override",
        )

        access_level = st.selectbox(
            "Access Level",
            options=list(ALLOWED_ACCESS_LEVELS),
            index=list(ALLOWED_ACCESS_LEVELS).index(
                getattr(settings, "DEFAULT_ACCESS_LEVEL", "internal")
            )
            if getattr(settings, "DEFAULT_ACCESS_LEVEL", "internal") in ALLOWED_ACCESS_LEVELS
            else 1,
            help="Role-based access classification for retrieval filtering.",
            key="select_access_level",
        )

    with col2:
        allowed_roles_input = st.text_input(
            "Allowed Roles (comma-separated)",
            value=getattr(settings, "DEFAULT_ALLOWED_ROLES", "all"),
            help="Specify roles permitted to access these chunks (e.g., 'all' or 'hr,admin').",
            key="input_allowed_roles",
        )

        tags_input = st.text_input(
            "Additional Tags (comma-separated)",
            value="",
            placeholder="e.g. q1_review, confidential, compliance_2026",
            help="Custom tags attached to chunks for filtering and audit trails.",
            key="input_tags",
        )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        enrich_clicked = st.button(
            "🏷️ Enrich Metadata",
            type="primary",
            use_container_width=True,
            disabled=not has_chunks,
            key="btn_enrich_metadata",
        )

    if enrich_clicked and has_chunks:
        cat_override = None if selected_category == "Auto-detect" else selected_category
        extra_tags = [t.strip() for t in tags_input.split(",") if t.strip()]

        options = EnrichmentOptions(
            category_override=cat_override,
            access_level=access_level,
            allowed_roles=allowed_roles_input,
            extra_tags=extra_tags,
        )

        with st.spinner("Enriching metadata and calculating identity hashes..."):
            enriched = indexing_service.enrich_chunks(generated_chunks, options=options)
            st.session_state["enriched_chunks"] = enriched
            reset_downstream("enrichment")

    enriched_chunks = st.session_state.get("enriched_chunks", [])
    if enriched_chunks:
        st.markdown("---")
        hcol1, hcol2 = st.columns([2, 1])
        with hcol1:
            st.success(f"✅ Successfully enriched **{len(enriched_chunks):,} chunks**!")
        with hcol2:
            json_payload = indexing_service.export_enriched_json(enriched_chunks)
            st.download_button(
                label="📥 Download Enriched Chunks (JSON)",
                data=json_payload,
                file_name="enriched_chunks.json",
                mime="application/json",
                use_container_width=True,
                key="btn_download_enriched_json",
            )

        # Overview Table
        table_rows = []
        cat_counts: dict[str, int] = {}
        for c in enriched_chunks:
            meta = c.metadata or {}
            c_cat = meta.get("category", "general")
            cat_counts[c_cat] = cat_counts.get(c_cat, 0) + 1
            table_rows.append(
                {
                    "Chunk ID": meta.get("chunk_id", ""),
                    "Source Name": meta.get("source_name", meta.get("source", "")),
                    "Category": c_cat,
                    "Access Level": meta.get("access_level", ""),
                    "Word Count": meta.get("word_count", 0),
                }
            )

        st.markdown("### 📊 Metadata Distribution")
        df_overview = pd.DataFrame(table_rows)
        st.dataframe(df_overview, use_container_width=True)

        # Per-category metrics / bar chart
        if cat_counts:
            chart_df = pd.DataFrame(
                list(cat_counts.items()), columns=["Category", "Count"]
            ).set_index("Category")
            st.bar_chart(chart_df)

        # Detailed Inspect Expanders
        st.markdown("### 🔍 Inspect Enriched Chunks")
        for idx, doc in enumerate(enriched_chunks[:50]):
            meta = doc.metadata or {}
            cid = meta.get("chunk_id", f"chunk_{idx}")
            cat = meta.get("category", "")
            with st.expander(f"[{cat.upper()}] {cid} ({meta.get('source_name', '')})"):
                st.json(meta)
                st.text_area("Chunk Content", value=doc.page_content, height=100, disabled=True, key=f"preview_enrich_{idx}")

        if len(enriched_chunks) > 50:
            st.caption(f"Showing first 50 of {len(enriched_chunks)} chunks.")
