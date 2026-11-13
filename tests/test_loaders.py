"""Phase 3 Knowledge Source Document Loaders and Normalization Tests."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import zipfile

import pytest
from langchain_core.documents import Document
from pypdf import PdfReader, PdfWriter

from ingestion.document_loader import (
    DocumentLoader,
    DocumentLoadingError,
    document_loader,
)
from ingestion.document_processor import (
    DocumentProcessor,
    document_processor,
)
from ingestion.docx_loader import DOCXDocumentLoader
from ingestion.pdf_loader import PDFDocumentLoader
from ingestion.text_loader import ManualTextLoader, TextDocumentLoader
from ingestion.web_loader import WebDocumentLoader


# ===========================================================================
# FIXTURES & HELPERS
# ===========================================================================

MINIMAL_PDF_BYTES = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 44 >> stream
BT
/F1 12 Tf
72 712 Td
(Sample PDF Page Content) Tj
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


def make_pdf(path: Path, page_count: int = 1) -> Path:
    """Helper creating a syntactically valid PDF file with page_count pages."""
    reader = PdfReader(io.BytesIO(MINIMAL_PDF_BYTES))
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_page(reader.pages[0])
    with open(path, "wb") as f:
        writer.write(f)
    return path


def make_docx(path: Path, text: str = "Sample DOCX Content") -> Path:
    """Helper creating a minimal valid DOCX file containing paragraph text."""
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


# ===========================================================================
# 1-4. PDF LOADER TESTS
# ===========================================================================

def test_extract_text_from_valid_pdf(tmp_path: Path) -> None:
    """1. PDF: Extract text from a valid PDF."""
    pdf_path = make_pdf(tmp_path / "valid_document.pdf", page_count=1)

    loader = PDFDocumentLoader()
    docs = loader.load(pdf_path)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Sample PDF Page Content" in docs[0].page_content
    assert docs[0].metadata["filename"] == "valid_document.pdf"
    assert docs[0].metadata["source"] == str(pdf_path)


def test_preserve_pdf_page_numbers(tmp_path: Path) -> None:
    """2. PDF: Preserve PDF page numbers across multiple pages."""
    multi_page_pdf = make_pdf(tmp_path / "multi_page.pdf", page_count=3)

    loader = PDFDocumentLoader()
    docs = loader.load(multi_page_pdf)

    assert len(docs) == 3
    for i, doc in enumerate(docs):
        assert doc.metadata["page"] == i
        assert doc.metadata["page_number"] == i + 1
        assert doc.metadata["filename"] == "multi_page.pdf"


def test_handle_corrupted_pdf(tmp_path: Path) -> None:
    """3. PDF: Handle corrupted PDF gracefully."""
    corrupted_pdf = tmp_path / "corrupted.pdf"
    corrupted_pdf.write_bytes(b"%PDF-1.4 Corrupted non-PDF binary payload \x00\xff\xfe")

    loader = PDFDocumentLoader()
    with pytest.raises(DocumentLoadingError) as exc_info:
        loader.load(corrupted_pdf)

    assert "Failed to extract text from PDF" in str(exc_info.value)


def test_handle_empty_pdf(tmp_path: Path) -> None:
    """4. PDF: Handle empty (0-byte) PDF."""
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.touch()

    loader = PDFDocumentLoader()
    with pytest.raises(DocumentLoadingError) as exc_info:
        loader.load(empty_pdf)

    assert "empty (0 bytes)" in str(exc_info.value)


# ===========================================================================
# 5-6. DOCX LOADER TESTS
# ===========================================================================

def test_extract_text_from_valid_docx(tmp_path: Path) -> None:
    """5. DOCX: Extract text from a valid DOCX."""
    docx_path = make_docx(tmp_path / "document.docx", text="Quarterly Ingestion Findings")

    loader = DOCXDocumentLoader()
    docs = loader.load(docx_path)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Quarterly Ingestion Findings" in docs[0].page_content
    assert docs[0].metadata["filename"] == "document.docx"
    assert docs[0].metadata["source"] == str(docx_path)
    assert docs[0].metadata["source_type"] == "docx"


def test_handle_invalid_docx(tmp_path: Path) -> None:
    """6. DOCX: Handle invalid or non-DOCX file."""
    invalid_docx = tmp_path / "corrupt.docx"
    invalid_docx.write_bytes(b"This is completely invalid DOCX data.")

    loader = DOCXDocumentLoader()
    with pytest.raises(DocumentLoadingError) as exc_info:
        loader.load(invalid_docx)

    assert "Failed to extract text from DOCX" in str(exc_info.value)


# ===========================================================================
# 7-9. TXT AND MARKDOWN LOADER TESTS
# ===========================================================================

def test_extract_text_from_txt_and_markdown(tmp_path: Path) -> None:
    """7. TXT & MD: Extract text correctly."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Standard plain text content line 1.\nLine 2.", encoding="utf-8")

    md_file = tmp_path / "guide.md"
    md_file.write_text("# Markdown Heading\n\n- Bullet item A\n- Bullet item B", encoding="utf-8")

    loader = TextDocumentLoader()

    txt_docs = loader.load(txt_file)
    assert len(txt_docs) == 1
    assert "Standard plain text content line 1." in txt_docs[0].page_content
    assert txt_docs[0].metadata["source_type"] == "text"

    md_docs = loader.load(md_file)
    assert len(md_docs) == 1
    assert "# Markdown Heading" in md_docs[0].page_content
    assert md_docs[0].metadata["source_type"] == "markdown"


def test_handle_empty_text_files(tmp_path: Path) -> None:
    """8. TXT & MD: Handle empty files (0-byte and whitespace-only)."""
    empty_txt = tmp_path / "empty.txt"
    empty_txt.touch()

    whitespace_md = tmp_path / "whitespace.md"
    whitespace_md.write_text("   \n\t\n   ", encoding="utf-8")

    loader = TextDocumentLoader()

    with pytest.raises(DocumentLoadingError) as exc_0:
        loader.load(empty_txt)
    assert "empty" in str(exc_0.value).lower()

    with pytest.raises(DocumentLoadingError) as exc_ws:
        loader.load(whitespace_md)
    assert "empty" in str(exc_ws.value).lower() or "whitespace" in str(exc_ws.value).lower()


def test_handle_unicode_text_content(tmp_path: Path) -> None:
    """9. TXT & MD: Handle diverse Unicode content properly."""
    unicode_file = tmp_path / "multilingual.txt"
    unicode_content = (
        "English text with emojis: 🚀 🌟 📊\n"
        "French: Résumé de la situation financière avec des accents aigus et graves.\n"
        "German: Übergrößenträger müssen vorsichtig sein.\n"
        "Chinese: 统一数据摄取管道系统\n"
        "Hindi: डेटा अंतर्ग्रहण पाइपलाइन"
    )
    unicode_file.write_text(unicode_content, encoding="utf-8")

    loader = TextDocumentLoader()
    docs = loader.load(unicode_file)

    assert len(docs) == 1
    assert "🚀 🌟 📊" in docs[0].page_content
    assert "Résumé" in docs[0].page_content
    assert "Übergrößenträger" in docs[0].page_content
    assert "统一数据摄取管道系统" in docs[0].page_content
    assert "डेटा अंतर्ग्रहण पाइपलाइन" in docs[0].page_content


# ===========================================================================
# 10-13. WEBSITE LOADER TESTS
# ===========================================================================

def test_extract_webpage_with_mocked_response() -> None:
    """10. Website: Extract content from a valid webpage using a mocked HTTP response."""
    html_content = (
        "<!DOCTYPE html>"
        "<html>"
        "<head><title>Mocked Documentation</title></head>"
        "<body>"
        "<header><h1>Data Pipeline Architecture</h1></header>"
        "<main><p>This is extracted article body content.</p></main>"
        "</body>"
        "</html>"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html_content
    mock_resp.content = html_content.encode("utf-8")

    loader = WebDocumentLoader()
    with patch("requests.Session.get", return_value=mock_resp):
        docs = loader.load("https://example.com/docs")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Data Pipeline Architecture" in docs[0].page_content
    assert "This is extracted article body content." in docs[0].page_content
    assert docs[0].metadata["source"] == "https://example.com/docs"
    assert docs[0].metadata["source_type"] == "web"


def test_handle_inaccessible_urls() -> None:
    """11. Website: Handle inaccessible URLs."""
    import requests

    loader = WebDocumentLoader()
    with patch("requests.Session.get", side_effect=requests.exceptions.ConnectionError("Connection failed")):
        with pytest.raises(DocumentLoadingError) as exc_info:
            loader.load("https://unreachable-server.example.org")

    assert "Failed to access URL" in str(exc_info.value)


def test_handle_empty_webpage_content() -> None:
    """12. Website: Handle empty webpage content."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body></body></html>"
    mock_resp.content = b"<html><body></body></html>"

    loader = WebDocumentLoader()
    with patch("requests.Session.get", return_value=mock_resp):
        with pytest.raises(DocumentLoadingError) as exc_info:
            loader.load("https://empty-page.example.com")

    assert "has no extractable text content" in str(exc_info.value)


def test_handle_malformed_url() -> None:
    """13. Website: Handle malformed URLs."""
    loader = WebDocumentLoader()
    malformed_urls = [
        "not-a-valid-url",
        "http://",
        "ftp://example.com/file.txt",
    ]

    for bad_url in malformed_urls:
        with pytest.raises(DocumentLoadingError) as exc_info:
            loader.load(bad_url)
        assert "Malformed URL" in str(exc_info.value) or "Invalid URL" in str(exc_info.value)


# ===========================================================================
# 14-15. MANUAL TEXT LOADER TESTS
# ===========================================================================

def test_convert_valid_manual_text_to_document() -> None:
    """14. Manual Text: Convert valid manual text into Document objects."""
    loader = ManualTextLoader()
    raw_text = "This is a user-entered note for ingestion."

    docs = loader.load(raw_text, title="my_custom_note")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == raw_text
    assert docs[0].metadata["source"] == "my_custom_note"
    assert docs[0].metadata["source_type"] == "manual_text"
    assert docs[0].metadata["char_count"] == len(raw_text)


def test_reject_empty_manual_text() -> None:
    """15. Manual Text: Reject empty text."""
    loader = ManualTextLoader()
    empty_samples = ["", "   ", "\n\t  \r\n"]

    for sample in empty_samples:
        with pytest.raises(DocumentLoadingError) as exc_info:
            loader.load(sample)
        assert "Manual text validation failed" in str(exc_info.value)


# ===========================================================================
# 16-18. DOCUMENT STANDARDIZATION TESTS
# ===========================================================================

def test_verify_all_loaders_return_langchain_documents(tmp_path: Path) -> None:
    """16. Standardization: Verify all loaders return LangChain Document objects."""
    pdf_path = make_pdf(tmp_path / "std.pdf")
    docx_path = make_docx(tmp_path / "std.docx", text="Standard DOCX")
    txt_path = tmp_path / "std.txt"
    txt_path.write_text("Standard TXT", encoding="utf-8")

    pdf_docs = PDFDocumentLoader().load(pdf_path)
    docx_docs = DOCXDocumentLoader().load(docx_path)
    txt_docs = TextDocumentLoader().load(txt_path)
    manual_docs = ManualTextLoader().load("Standard Manual")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body>Standard Web</body></html>"
    mock_resp.content = b"<html><body>Standard Web</body></html>"
    with patch("requests.Session.get", return_value=mock_resp):
        web_docs = WebDocumentLoader().load("https://example.com/std")

    all_loader_docs = pdf_docs + docx_docs + txt_docs + manual_docs + web_docs
    assert len(all_loader_docs) >= 5
    for doc in all_loader_docs:
        assert isinstance(doc, Document), f"Object {type(doc)} is not a LangChain Document"
        assert isinstance(doc.page_content, str)
        assert isinstance(doc.metadata, dict)


def test_verify_source_metadata_preserved(tmp_path: Path) -> None:
    """17. Standardization: Verify source metadata is preserved through DocumentProcessor."""
    processor = DocumentProcessor()

    doc = Document(
        page_content="Sample content with \r\n Windows breaks and \x00 null bytes.",
        metadata={
            "source": str(tmp_path / "audit.pdf"),
            "filename": "audit.pdf",
            "source_type": "pdf",
            "page": 2,
            "page_number": 3,
        },
    )

    processed_docs = processor.process(doc)
    assert len(processed_docs) == 1
    norm_doc = processed_docs[0]

    # Verify normalization cleaned text
    assert "\x00" not in norm_doc.page_content
    assert "\r\n" not in norm_doc.page_content
    assert "Sample content with" in norm_doc.page_content

    # Verify metadata preservation
    assert norm_doc.metadata["source"] == str(tmp_path / "audit.pdf")
    assert norm_doc.metadata["filename"] == "audit.pdf"
    assert norm_doc.metadata["source_type"] == "pdf"
    assert norm_doc.metadata["page"] == 2
    assert norm_doc.metadata["page_number"] == 3
    assert norm_doc.metadata["char_count"] == len(norm_doc.page_content)


def test_verify_multiple_documents_processed_independently(tmp_path: Path) -> None:
    """18. Standardization: Verify multiple documents can be processed independently via coordinator."""
    pdf_path = make_pdf(tmp_path / "batch1.pdf", page_count=1)
    docx_path = make_docx(tmp_path / "batch2.docx", text="Batch DOCX Paragraph")
    txt_path = tmp_path / "batch3.txt"
    txt_path.write_text("Batch TXT Paragraph", encoding="utf-8")

    coordinator = DocumentLoader()
    processor = DocumentProcessor()

    # Load from heterogeneous sources in one batch
    raw_docs = coordinator.load_multiple([pdf_path, docx_path, txt_path])
    assert len(raw_docs) == 3

    # Normalize through DocumentProcessor
    processed_docs = processor.process(raw_docs)
    assert len(processed_docs) == 3

    sources = [doc.metadata["source"] for doc in processed_docs]
    types = [doc.metadata["source_type"] for doc in processed_docs]

    assert str(pdf_path) in sources
    assert str(docx_path) in sources
    assert str(txt_path) in sources

    assert "pdf" in types
    assert "docx" in types
    assert "text" in types

    # Each document maintains its own distinct text and metadata
    for doc in processed_docs:
        assert len(doc.page_content) > 0
        assert doc.metadata["char_count"] == len(doc.page_content)
