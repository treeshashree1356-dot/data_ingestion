"""UI component for batch embedding generation, progress visualization, and vector inspection."""

import math
import streamlit as st

from config.settings import settings
from embedding.embedding_factory import get_embedding_model
from embedding.embedding_manager import EmbeddingError, EmbeddingManager
from services.indexing_service import indexing_service
from ui.state import reset_downstream


@st.cache_resource(show_spinner=False)
def _get_cached_embedding_engine(provider: str, model_name: str):
    """Cache the embedding model instance across Streamlit reruns."""
    return get_embedding_model(provider=provider, model_name=model_name)


def render_embedding_view() -> None:
    """Render the embedding execution panel, configuration metrics, and vector inspector."""
    st.subheader("5. Vector Embedding Generation")
    st.caption("Generate high-dimensional semantic embeddings locally using HuggingFace sentence-transformers.")

    enriched_chunks = st.session_state.get("enriched_chunks", [])
    has_chunks = bool(enriched_chunks)

    if not has_chunks:
        st.info("👈 Enrich metadata in Step 4 before generating embeddings.")

    # Model Configuration Summary (reads strictly from settings, zero model download)
    c1, c2, c3 = st.columns(3)
    c1.metric("Provider", getattr(settings, "EMBEDDING_PROVIDER", "huggingface"))
    c2.metric("Model Name", getattr(settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2").split("/")[-1])
    c3.metric("Batch Size", getattr(settings, "EMBEDDING_BATCH_SIZE", 32))

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        generate_clicked = st.button(
            "🧮 Generate Embeddings",
            type="primary",
            use_container_width=True,
            disabled=not has_chunks,
            key="btn_generate_embeddings",
        )

    if generate_clicked and has_chunks:
        progress_bar = st.progress(0, text="Initializing embedding engine...")

        def _on_progress(done: int, total: int) -> None:
            pct = int((done / max(total, 1)) * 100)
            progress_bar.progress(min(pct, 100), text=f"Embedding chunks: {done} / {total} ({pct}%)")

        try:
            with st.spinner("Embedding chunks... Note: The first run downloads the model (approx 90 MB)."):
                # Use cached model in UI
                cached_engine = _get_cached_embedding_engine(
                    getattr(settings, "EMBEDDING_PROVIDER", "huggingface"),
                    getattr(settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                )
                manager = EmbeddingManager(embeddings=cached_engine)
                emb_res = manager.embed_chunks(enriched_chunks, progress_callback=_on_progress)

                st.session_state["embedding_result"] = emb_res
                reset_downstream("embedding")
                progress_bar.progress(100, text="Embeddings successfully generated!")
        except EmbeddingError as exc:
            st.error(f"⚠️ Embedding generation failed: {exc}")
        except Exception as exc:
            st.error(f"⚠️ An unexpected error occurred while embedding: {exc}")

    emb_result = st.session_state.get("embedding_result")
    if emb_result and emb_result.vectors:
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Vectors Created", len(emb_result.vectors))
        m2.metric("Vector Dimension", emb_result.dimension)
        m3.metric("Time Taken", f"{emb_result.duration_seconds:.2f}s")
        m4.metric("Model", emb_result.model_name.split("/")[-1])

        st.markdown("### 🔬 Inspect Embedding Vector")
        chunk_labels = [
            f"{c.metadata.get('chunk_id', f'chunk_{i}')} - {c.page_content[:40]}..."
            for i, c in enumerate(emb_result.chunks)
        ]

        selected_idx = st.selectbox(
            "Select Chunk to Inspect Vector",
            options=range(len(chunk_labels)),
            format_func=lambda idx: chunk_labels[idx],
            key="select_inspect_vector",
        )

        selected_vector = emb_result.vectors[selected_idx]
        norm = math.sqrt(sum(x * x for x in selected_vector))

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.write(f"**L2 Norm (Magnitude):** `{norm:.4f}`")
            st.write(f"**Full Length:** `{len(selected_vector)} dimensions`")
        with vcol2:
            st.write(f"**First 10 Dimensions:**")
            first_10 = [round(val, 6) for val in selected_vector[:10]]
            st.code(str(first_10), language="python")
