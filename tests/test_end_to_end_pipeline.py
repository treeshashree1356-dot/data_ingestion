"""Phase 7 Comprehensive End-to-End Pipeline Workflow Tests.

Covers the full progression from Knowledge Sources -> Data Extraction ->
Standardized LangChain Documents -> Chunking -> JSON Export.
"""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import zipfile

from langchain_core.documents import Document
from pypdf import PdfReader, PdfWriter
import pytest
from streamlit.testing.v1 import AppTest

from chunking.text_splitter import ChunkingManager, deserialize_chunks_from_json
from ingestion.pdf_loader import PDFDocumentLoader
from services.ingestion_service import IngestionService

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

MINIMAL_PDF = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 44 >> stream
BT
/F1 12 Tf
72 712 Td
(PDF Paragraph for End-to-End Workflow Testing) Tj
ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000338 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
417
%%EOF"""


def make_test_pdf(path: Path, page_count: int = 1) -> Path:
    """Helper to generate valid test PDF."""
    reader = PdfReader(io.BytesIO(MINIMAL_PDF))
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_page(reader.pages[0])
    with open(path, "wb") as f:
        writer.write(f)
    return path


def make_test_docx(path: Path, text: str = "DOCX Content Paragraph") -> Path:
    """Helper to generate valid test DOCX."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/></Types>',
        )
        zf.writestr(
            "word/document.xml",
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f'<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>',
        )
    return path


def test_1_single_pdf_upload_to_chunk_generation(tmp_path: Path) -> None:
    """1. Test single PDF upload progressing to chunk generation."""
    pdf_path = make_test_pdf(tmp_path / "single_test.pdf", page_count=2)
    service = IngestionService()

    # Ingest
    result = service.process_sources(
        uploaded_files=[pdf_path],
        destination_dir=tmp_path / "uploads",
    )
    assert result.successful_count == 1
    assert len(result.documents) == 2

    # Chunk
    chunks = service.chunk_documents(result.documents, chunk_size=200, chunk_overlap=40)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert "single_test" in chunk.metadata["filename"]
        assert "chunk_id" in chunk.metadata


def test_2_multiple_pdf_uploads_to_chunks(tmp_path: Path) -> None:
    """2. Test multiple PDF uploads progressing to chunk generation."""
    pdf1 = make_test_pdf(tmp_path / "multi_a.pdf", page_count=1)
    pdf2 = make_test_pdf(tmp_path / "multi_b.pdf", page_count=1)
    service = IngestionService()

    result = service.process_sources(
        uploaded_files=[pdf1, pdf2],
        destination_dir=tmp_path / "uploads",
    )
    assert result.successful_count == 2
    assert len(result.documents) == 2

    chunks = service.chunk_documents(result.documents, chunk_size=200, chunk_overlap=30)
    filenames = {c.metadata["filename"] for c in chunks}
    assert any("multi_a" in fn for fn in filenames)
    assert any("multi_b" in fn for fn in filenames)


def test_3_docx_upload_to_chunk_generation(tmp_path: Path) -> None:
    """3. Test DOCX upload progressing to chunk generation."""
    docx_path = make_test_docx(tmp_path / "document.docx", text="Corporate policy guidelines and protocols. " * 15)
    service = IngestionService()

    result = service.process_sources(
        uploaded_files=[docx_path],
        destination_dir=tmp_path / "uploads",
    )
    assert result.successful_count == 1
    assert result.documents[0].metadata["source_type"] == "docx"

    chunks = service.chunk_documents(result.documents, chunk_size=250, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["source_type"] == "docx"


def test_4_txt_and_markdown_processing(tmp_path: Path) -> None:
    """4. Test TXT and Markdown files progressing to chunk generation."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Plain text documentation notes. " * 20, encoding="utf-8")

    md_file = tmp_path / "guide.md"
    md_file.write_text("# Markdown Title\n\nDetailed guidance sections. " * 20, encoding="utf-8")

    service = IngestionService()
    result = service.process_sources(
        uploaded_files=[txt_file, md_file],
        destination_dir=tmp_path / "uploads",
    )
    assert result.successful_count == 2
    types = {d.metadata["source_type"] for d in result.documents}
    assert "text" in types
    assert "markdown" in types

    chunks = service.chunk_documents(result.documents, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 2


def test_5_website_url_extraction_to_chunk_generation() -> None:
    """5. Test website URL extraction progressing to chunk generation."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><head><title>Web Guide</title></head><body><p>" + ("Online article content. " * 30) + "</p></body></html>"
    mock_resp.content = mock_resp.text.encode("utf-8")

    service = IngestionService()
    with patch("requests.Session.get", return_value=mock_resp):
        result = service.process_sources(urls=["https://example.com/article"])

    assert result.successful_count == 1
    assert len(result.documents) == 1
    assert result.documents[0].metadata["source_type"] == "web"

    chunks = service.chunk_documents(result.documents, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "https://example.com/article"


def test_6_image_ocr_to_chunk_generation(tmp_path: Path) -> None:
    """6. Test image OCR extraction progressing to chunk generation."""
    from PIL import Image, ImageDraw
    img_path = tmp_path / "receipt.png"
    img = Image.new("RGB", (200, 80), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 70], outline="black")
    draw.text((20, 25), "Line item details and totals", fill="black")
    img.save(img_path)

    service = IngestionService()
    mock_ocr = "Invoice Item 1: $40.00\nInvoice Item 2: $60.00\nTotal Due: $100.00"

    with patch("pytesseract.image_to_string", return_value=mock_ocr):
        result = service.process_sources(
            uploaded_files=[img_path],
            destination_dir=tmp_path / "uploads",
        )

    assert result.successful_count == 1
    assert result.documents[0].metadata["source_type"] == "image"

    chunks = service.chunk_documents(result.documents, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    assert "Invoice Item" in chunks[0].page_content


def test_7_scanned_pdf_ocr_fallback_to_chunk_generation(tmp_path: Path) -> None:
    """7. Test scanned PDF OCR fallback triggering and chunk generation."""
    pdf_path = make_test_pdf(tmp_path / "scanned_invoice.pdf")
    service = IngestionService()

    # Simulate PyPDF returning empty page text
    empty_pypdf = [Document(page_content="", metadata={"page": 0, "source": str(pdf_path)})]
    mock_ocr_text = "Scanned Form Table Data: Account #987654321 Balance: $1,250.00"

    with patch("langchain_community.document_loaders.PyPDFLoader.load", return_value=empty_pypdf):
        with patch.object(PDFDocumentLoader, "_ocr_page", return_value=mock_ocr_text):
            result = service.process_sources(
                uploaded_files=[pdf_path],
                destination_dir=tmp_path / "uploads",
            )

    assert result.successful_count == 1
    assert result.documents[0].metadata["ocr_applied"] is True

    chunks = service.chunk_documents(result.documents, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    assert "Scanned Form Table Data" in chunks[0].page_content


def test_8_manual_text_to_chunk_generation() -> None:
    """8. Test manual text input progressing to chunk generation."""
    service = IngestionService()
    raw_notes = "Executive meeting decisions and action items regarding pipeline deployment. " * 15

    result = service.process_sources(
        manual_text=raw_notes,
        manual_title="q4_decisions",
    )
    assert result.successful_count == 1
    assert result.documents[0].metadata["source_type"] == "manual_text"

    chunks = service.chunk_documents(result.documents, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["source"] == "q4_decisions"


def test_9_mixed_source_processing_pipeline(tmp_path: Path) -> None:
    """9. Test comprehensive mixed-source processing in a single unified operation."""
    pdf_file = make_test_pdf(tmp_path / "mix.pdf")
    docx_file = make_test_docx(tmp_path / "mix.docx", text="DOCX mixed source paragraph.")
    txt_file = tmp_path / "mix.txt"
    txt_file.write_text("TXT mixed source paragraph.", encoding="utf-8")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body>Web mixed source paragraph.</body></html>"
    mock_resp.content = mock_resp.text.encode("utf-8")

    service = IngestionService()
    with patch("requests.Session.get", return_value=mock_resp):
        result = service.process_sources(
            uploaded_files=[pdf_file, docx_file, txt_file],
            urls=["https://example.com/mix"],
            manual_text="Manual mixed source paragraph.",
            manual_title="mix_manual",
            destination_dir=tmp_path / "uploads",
        )

    assert result.successful_count == 5
    assert result.failed_count == 0
    assert len(result.documents) == 5

    # Chunk all mixed sources together
    chunks = service.chunk_documents(result.documents, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 5

    types = {c.metadata["source_type"] for c in chunks}
    assert {"pdf", "docx", "text", "web", "manual_text"}.issubset(types)


def test_10_partial_extraction_failure_handling(tmp_path: Path) -> None:
    """10. Test that partial extraction failures are isolated and do not block valid sources."""
    valid_txt = tmp_path / "good.txt"
    valid_txt.write_text("Good text file content.", encoding="utf-8")

    corrupted_pdf = tmp_path / "corrupt.pdf"
    corrupted_pdf.write_bytes(b"Broken non-PDF data")

    service = IngestionService()
    # Mock bad URL throwing 500 error
    with patch("requests.Session.get", side_effect=Exception("500 Server Error")):
        result = service.process_sources(
            uploaded_files=[corrupted_pdf, valid_txt],
            urls=["https://broken.server.example"],
            destination_dir=tmp_path / "uploads",
        )

    # 1 valid file succeeded, 2 sources failed
    assert result.successful_count == 1
    assert result.failed_count == 2
    assert len(result.errors) == 2
    assert len(result.documents) == 1

    # Chunking functions normally on remaining valid documents
    chunks = service.chunk_documents(result.documents, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    assert "Good text file" in chunks[0].page_content


def test_11_chunk_size_and_overlap_configuration() -> None:
    """11. Test that chunk size and overlap configurations are strictly enforced."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six."
    doc = Document(page_content=text, metadata={"source": "test"})

    manager = ChunkingManager(chunk_size=40, chunk_overlap=10)
    chunks = manager.split_documents([doc])

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.page_content) <= 40


def test_12_json_export_and_roundtrip_parsing(tmp_path: Path) -> None:
    """12. Test JSON export and deserialization roundtrip parity."""
    doc = Document(
        page_content="Content for JSON export and deserialization testing.",
        metadata={"source": "test.txt", "author": "Alice", "page": 1},
    )
    service = IngestionService()
    chunks = service.chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    json_file = tmp_path / "export.json"
    json_str = service.export_chunks_json(chunks, file_path=json_file)

    # Parse back from file
    restored = deserialize_chunks_from_json(json_file)
    assert len(restored) == len(chunks)
    assert restored[0].page_content == chunks[0].page_content
    assert restored[0].metadata["source"] == "test.txt"
    assert restored[0].metadata["chunk_id"] == chunks[0].metadata["chunk_id"]


def test_13_streamlit_session_state_lifecycle() -> None:
    """13. Test full Streamlit session state lifecycle through UI interactions."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    # Enter manual text and extract
    at.text_area(key="source_manual_text").input("Session state lifecycle text test").run()
    at.button(key="btn_extract_data").click().run()

    assert not at.exception
    assert len(at.session_state["extracted_documents"]) == 1
    assert len(at.session_state["generated_chunks"]) == 0

    # Generate chunks
    at.button(key="btn_generate_chunks").click().run()
    assert len(at.session_state["generated_chunks"]) == 1

    # Invalidate chunks by modifying chunk size
    at.number_input(key="input_chunk_size").set_value(600).run()
    assert len(at.session_state["generated_chunks"]) == 0


def test_14_complete_application_execution() -> None:
    """14. Test complete application execution renders without exceptions."""
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert len(at.title) >= 1
    assert "Data Ingestion Pipeline" in at.title[0].value
    assert len(at.tabs) == 6
    assert len(at.metric) >= 2
