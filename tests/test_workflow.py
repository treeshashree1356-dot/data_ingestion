"""Phase 6 End-to-End Workflow and Fault Isolation Tests."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
import pytest
from streamlit.testing.v1 import AppTest

from services.ingestion_service import IngestionResult, IngestionService

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_failed_extraction_displays_appropriate_error() -> None:
    """11. Test that extraction failures display descriptive errors in the UI."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Simulate an extraction result with an error
    error_result = IngestionResult(
        documents=[],
        successful_count=0,
        failed_count=1,
        errors=[{"source": "corrupted.docx", "error": "BadZipFile: File is not a zip file"}],
    )

    at.session_state["extraction_result"] = error_result
    at.session_state["extracted_documents"] = []
    at.run()

    assert not at.exception
    # Verify error metrics and notification messages are rendered
    assert len(at.error) > 0
    error_text = " ".join([e.value for e in at.error])
    assert "corrupted.docx" in error_text
    assert "BadZipFile" in error_text


def test_one_failed_file_does_not_prevent_successful_processing(tmp_path: Path) -> None:
    """12. Test that a failure on one file does not halt processing of other valid files."""
    valid_file = tmp_path / "valid_document.txt"
    valid_file.write_text("Valid text content for pipeline extraction.", encoding="utf-8")

    corrupted_file = tmp_path / "corrupted_archive.pdf"
    corrupted_file.write_bytes(b"Corrupted binary data that fails pypdf")

    upload_dir = tmp_path / "uploads"
    service = IngestionService()
    result = service.process_sources(
        uploaded_files=[corrupted_file, valid_file],
        destination_dir=upload_dir,
    )

    # 1 failure and 1 success recorded
    assert result.successful_count == 1
    assert result.failed_count == 1
    assert len(result.errors) == 1
    assert "corrupted_archive" in result.errors[0]["source"]

    # Valid file content successfully extracted and normalized
    assert len(result.documents) == 1
    assert "Valid text content" in result.documents[0].page_content
    assert "valid_document" in result.documents[0].metadata["filename"]


def test_full_end_to_end_pipeline_workflow(tmp_path: Path) -> None:
    """Test full integrated pipeline: multi-source -> extract -> normalize -> chunk -> export."""
    # 1. Setup multi-source inputs
    txt_file = tmp_path / "article.txt"
    txt_file.write_text(
        "Modern retrieval architectures require disciplined chunking and normalization. "
        "Information retrieval models depend heavily on chunk coherence and boundary overlaps. " * 10,
        encoding="utf-8",
    )
    manual_note = "User manual meeting notes regarding project scope and timelines. " * 8

    upload_dir = tmp_path / "e2e_uploads"
    service = IngestionService()

    # 2. Process sources
    result = service.process_sources(
        uploaded_files=[txt_file],
        manual_text=manual_note,
        manual_title="meeting_summary",
        destination_dir=upload_dir,
    )

    assert result.successful_count == 2
    assert result.failed_count == 0
    assert len(result.documents) == 2

    # 3. Chunk documents
    chunks = service.chunk_documents(result.documents, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 2

    # 4. Verify unique IDs and metadata integrity
    chunk_ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))

    sources = {c.metadata["source"] for c in chunks}
    assert any("article.txt" in s for s in sources)
    assert any("meeting_summary" in s for s in sources)

    # 5. Export to JSON
    export_file = tmp_path / "e2e_export.json"
    json_output = service.export_chunks_json(chunks, file_path=export_file)
    assert export_file.exists()

    parsed = json.loads(json_output)
    assert len(parsed) == len(chunks)
    assert "chunk_id" in parsed[0]
    assert "content" in parsed[0]
    assert "metadata" in parsed[0]


def test_ui_workflow_invalidation_on_new_extraction() -> None:
    """Test that performing a new extraction in the UI invalidates existing chunks."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Pre-populate state with old chunks
    at.session_state["extracted_documents"] = [Document(page_content="Old doc", metadata={})]
    at.session_state["generated_chunks"] = [Document(page_content="Old chunk", metadata={})]
    at.run()

    assert len(at.session_state["generated_chunks"]) == 1

    # Enter new manual text and click Extract Data
    at.text_area(key="source_manual_text").input("Brand new text input").run()
    at.button(key="btn_extract_data").click().run()

    # Generated chunks should now be emptied / invalidated
    assert len(at.session_state["generated_chunks"]) == 0
    assert len(at.session_state["extracted_documents"]) > 0
