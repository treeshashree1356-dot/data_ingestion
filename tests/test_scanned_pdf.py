"""Phase 4 Scanned PDF OCR Fallback Tests."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from pypdf import PdfReader, PdfWriter
import pytest

from ingestion.document_loader import DocumentLoader
from ingestion.pdf_loader import PDFDocumentLoader


MINIMAL_PDF_BYTES = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 44 >> stream
BT
/F1 12 Tf
72 712 Td
(Digital PDF Page Content Here) Tj
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
    """Helper creating a valid PDF file."""
    reader = PdfReader(io.BytesIO(MINIMAL_PDF_BYTES))
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_page(reader.pages[0])
    with open(path, "wb") as f:
        writer.write(f)
    return path


def test_trigger_scanned_pdf_fallback_when_extracted_text_is_empty(tmp_path: Path) -> None:
    """9. Trigger scanned PDF fallback when extracted text is empty or insufficient."""
    pdf_path = make_pdf(tmp_path / "scanned_doc.pdf")

    loader = PDFDocumentLoader(enable_ocr_fallback=True, min_text_chars=10)

    # Simulate PyPDF returning empty page_content (as occurs with scanned image PDFs)
    empty_pypdf_doc = [
        Document(
            page_content="   ",
            metadata={"source": str(pdf_path), "page": 0},
        )
    ]

    with patch("langchain_community.document_loaders.PyPDFLoader.load", return_value=empty_pypdf_doc):
        with patch.object(loader, "_ocr_page", return_value="Scanned Invoice Total: $500"):
            docs = loader.load(pdf_path)

    assert len(docs) == 1
    assert docs[0].page_content == "Scanned Invoice Total: $500"
    assert docs[0].metadata["ocr_applied"] is True
    assert docs[0].metadata["page"] == 0
    assert docs[0].metadata["page_number"] == 1


def test_avoid_ocr_when_pdf_text_extraction_succeeds(tmp_path: Path) -> None:
    """10. Avoid OCR processing when PDF text extraction succeeds with usable text."""
    pdf_path = make_pdf(tmp_path / "digital_doc.pdf")

    loader = PDFDocumentLoader(enable_ocr_fallback=True, min_text_chars=5)

    with patch.object(loader, "_ocr_page") as mock_ocr:
        docs = loader.load(pdf_path)
        # Verify OCR was never invoked because native text was found
        mock_ocr.assert_not_called()

    assert len(docs) == 1
    assert "Digital PDF Page Content" in docs[0].page_content
    assert docs[0].metadata["ocr_applied"] is False


def test_preserve_pdf_page_numbers_across_mixed_extraction(tmp_path: Path) -> None:
    """11. Preserve PDF page numbers across mixed digital and OCR pages."""
    pdf_path = make_pdf(tmp_path / "mixed_doc.pdf", page_count=2)

    loader = PDFDocumentLoader(enable_ocr_fallback=True, min_text_chars=10)

    mixed_pypdf_docs = [
        Document(page_content="Digital content page 1", metadata={"page": 0}),
        Document(page_content="", metadata={"page": 1}),  # Scanned page
    ]

    def mock_ocr_dispatcher(path, page_idx):
        if page_idx == 1:
            return "OCR text page 2"
        return None

    with patch("langchain_community.document_loaders.PyPDFLoader.load", return_value=mixed_pypdf_docs):
        with patch.object(loader, "_ocr_page", side_effect=mock_ocr_dispatcher):
            docs = loader.load(pdf_path)

    assert len(docs) == 2
    # Page 0 (digital)
    assert docs[0].metadata["page"] == 0
    assert docs[0].metadata["page_number"] == 1
    assert docs[0].metadata["ocr_applied"] is False
    assert docs[0].page_content == "Digital content page 1"

    # Page 1 (OCR fallback)
    assert docs[1].metadata["page"] == 1
    assert docs[1].metadata["page_number"] == 2
    assert docs[1].metadata["ocr_applied"] is True
    assert docs[1].page_content == "OCR text page 2"


def test_handle_ocr_failure_without_crashing_other_documents(tmp_path: Path) -> None:
    """12. Handle OCR failure gracefully without crashing the processing of other documents."""
    scanned_pdf = make_pdf(tmp_path / "scanned_broken_ocr.pdf")
    text_file = tmp_path / "standard.txt"
    text_file.write_text("Regular plain text document", encoding="utf-8")

    coordinator = DocumentLoader()

    # Mock PyPDF returning empty text for the PDF
    empty_pypdf_doc = [
        Document(page_content="", metadata={"source": str(scanned_pdf), "page": 0})
    ]

    with patch("langchain_community.document_loaders.PyPDFLoader.load", return_value=empty_pypdf_doc):
        # Force OCR to fail with an exception
        with patch("pypdfium2.PdfDocument", side_effect=RuntimeError("OCR engine crashed")):
            docs = coordinator.load_multiple([scanned_pdf, text_file])

    # Pipeline did not crash; both documents are returned
    assert len(docs) == 2

    # Scanned PDF document handled gracefully
    pdf_doc = [d for d in docs if d.metadata.get("source_type") == "pdf"][0]
    assert pdf_doc.metadata["ocr_applied"] is False

    # Plain text document succeeded completely
    txt_doc = [d for d in docs if d.metadata.get("source_type") == "text"][0]
    assert "Regular plain text document" in txt_doc.page_content
