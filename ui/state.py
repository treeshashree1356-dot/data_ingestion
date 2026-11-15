"""Session state management and downstream pipeline state invalidation.

Ensures that whenever an upstream ingestion, extraction, chunking, or embedding
parameter changes, downstream artifacts are automatically purged to prevent stale data
from entering the vector database.
"""

import streamlit as st


def reset_downstream(from_stage: str) -> None:
    """Invalidate and reset downstream pipeline states based on the triggering stage.

    Stage progression:
    - 'extraction' or 'chunking': invalidates enriched_chunks, embedding_result,
      indexing_result, search_results, query_response.
    - 'enrichment': invalidates embedding_result, indexing_result.
    - 'embedding': invalidates indexing_result.

    Args:
        from_stage: Identifier for the trigger stage ('extraction', 'chunking',
            'enrichment', 'embedding').
    """
    stage = from_stage.lower().strip()

    if stage in ("extraction", "chunking"):
        st.session_state["enriched_chunks"] = []
        st.session_state["embedding_result"] = None
        st.session_state["indexing_result"] = None
        st.session_state["search_results"] = []
        st.session_state["query_response"] = None
    elif stage == "enrichment":
        st.session_state["embedding_result"] = None
        st.session_state["indexing_result"] = None
    elif stage == "embedding":
        st.session_state["indexing_result"] = None
