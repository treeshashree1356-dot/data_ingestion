"""Phase 6 Streamlit UI Component and Integration Tests."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
import pytest
from streamlit.testing.v1 import AppTest

import app
from chunking.text_splitter import export_chunks_to_json, validate_chunking_config
from services.ingestion_service import IngestionResult
import ui
from ui.chunking_view import render_chunking_view
from ui.extraction_view import render_extraction_view
from ui.source_selection import render_source_selection

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_module_imports_successfully() -> None:
    """1. Test that app.py imports cleanly and main entrypoint is callable."""
    assert hasattr(app, "main"), "app.py does not define main()"
    assert hasattr(app, "init_session_state"), "app.py does not define init_session_state()"


def test_ui_modules_import_successfully() -> None:
    """2. Test that all UI modules and view functions can be imported."""
    assert hasattr(ui, "source_selection"), "ui package missing source_selection"
    assert hasattr(ui, "extraction_view"), "ui package missing extraction_view"
    assert hasattr(ui, "chunking_view"), "ui package missing chunking_view"
    assert callable(render_source_selection)
    assert callable(render_extraction_view)
    assert callable(render_chunking_view)


def test_source_selection_handles_supported_formats() -> None:
    """3. Test that source selection file uploader handles all expected formats."""
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception

    # Locate file uploader widget
    file_uploaders = at.file_uploader
    assert len(file_uploaders) >= 1
    uploader = file_uploaders[0]
    expected_types = ["pdf", "docx", "txt", "md", "jpg", "jpeg", "png"]
    supported_extensions = [t.lstrip(".") for t in uploader.proto.type]
    for ext in expected_types:
        assert ext in supported_extensions, f"Uploader missing support for extension {ext}"


def test_extraction_button_calls_ingestion_service() -> None:
    """4. Test that the extraction button invokes the ingestion service."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Seed mock manual text input
    at.text_area(key="source_manual_text").input("Manual testing notes").run()

    mock_result = IngestionResult(
        documents=[Document(page_content="Manual testing notes", metadata={"source": "manual"})],
        successful_count=1,
        failed_count=0,
    )

    with patch("services.ingestion_service.IngestionService.process_sources", return_value=mock_result) as mock_svc:
        btn = at.button(key="btn_extract_data")
        btn.click().run()

        assert not at.exception
        assert mock_svc.called, "IngestionService.process_sources was not called"


def test_chunk_generation_calls_chunking_service() -> None:
    """5. Test that the chunk generation button invokes the chunking service."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Pre-populate session state with an extracted document
    sample_doc = Document(page_content="Sample long text " * 30, metadata={"source": "test.pdf"})
    at.session_state["extracted_documents"] = [sample_doc]
    at.session_state["chunk_size"] = 200
    at.session_state["chunk_overlap"] = 40
    at.run()

    mock_chunks = [
        Document(page_content="Chunk 1 content", metadata={"chunk_id": "c1"}),
        Document(page_content="Chunk 2 content", metadata={"chunk_id": "c2"}),
    ]

    with patch("services.ingestion_service.IngestionService.chunk_documents", return_value=mock_chunks) as mock_chunk:
        btn = at.button(key="btn_generate_chunks")
        btn.click().run()

        assert not at.exception
        assert mock_chunk.called, "IngestionService.chunk_documents was not called"
        assert len(at.session_state["generated_chunks"]) == 2


def test_chunk_settings_are_validated() -> None:
    """6. Test that invalid chunk settings are caught and rejected."""
    # Test valid configuration
    validate_chunking_config(1000, 200)

    # Test invalid cases
    with pytest.raises(ValueError):
        validate_chunking_config(0, 0)

    with pytest.raises(ValueError):
        validate_chunking_config(500, -1)

    with pytest.raises(ValueError):
        validate_chunking_config(500, 500)

    with pytest.raises(ValueError):
        validate_chunking_config(500, 600)


def test_session_state_retains_extracted_documents() -> None:
    """7. Test that session state preserves extracted documents across reruns."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    test_docs = [
        Document(page_content="Doc 1 content", metadata={"source": "first.pdf"}),
        Document(page_content="Doc 2 content", metadata={"source": "second.txt"}),
    ]
    at.session_state["extracted_documents"] = test_docs
    at.run()

    assert len(at.session_state["extracted_documents"]) == 2
    assert at.session_state["extracted_documents"][0].metadata["source"] == "first.pdf"


def test_session_state_retains_generated_chunks() -> None:
    """8. Test that session state preserves generated chunks across reruns."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    chunks = [
        Document(page_content="Chunk alpha", metadata={"chunk_id": "chunk_0"}),
        Document(page_content="Chunk beta", metadata={"chunk_id": "chunk_1"}),
    ]
    at.session_state["generated_chunks"] = chunks
    at.run()

    assert len(at.session_state["generated_chunks"]) == 2
    assert at.session_state["generated_chunks"][0].metadata["chunk_id"] == "chunk_0"


def test_changing_chunk_settings_invalidates_old_chunks() -> None:
    """9. Test that changing chunk size or overlap invalidates previously generated chunks."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Preload chunks
    at.session_state["extracted_documents"] = [Document(page_content="Text", metadata={})]
    at.session_state["generated_chunks"] = [Document(page_content="Old Chunk", metadata={})]
    at.session_state["chunk_size"] = 1000
    at.session_state["chunk_overlap"] = 200
    at.run()

    assert len(at.session_state["generated_chunks"]) == 1

    # Modify chunk_size via number input
    size_input = at.number_input(key="input_chunk_size")
    size_input.set_value(800).run()

    # Previous chunks should be invalidated (cleared)
    assert at.session_state["chunk_size"] == 800
    assert len(at.session_state["generated_chunks"]) == 0


def test_json_download_produces_correct_content() -> None:
    """10. Test that JSON download payload contains valid structured chunk data."""
    chunks = [
        Document(
            page_content="First test chunk",
            metadata={"source": "audit.pdf", "chunk_id": "audit_c0", "page": 1},
        ),
        Document(
            page_content="Second test chunk",
            metadata={"source": "audit.pdf", "chunk_id": "audit_c1", "page": 1},
        ),
    ]

    json_str = export_chunks_to_json(chunks)
    import json
    parsed = json.loads(json_str)

    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["chunk_id"] == "audit_c0"
    assert parsed[0]["content"] == "First test chunk"
    assert parsed[0]["metadata"]["source"] == "audit.pdf"
