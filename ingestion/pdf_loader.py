"""PDF document loader with intelligent scanned PDF OCR fallback."""

import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import pypdfium2 as pdfium
import pytesseract

from config.settings import settings
from ingestion.document_loader import BaseDocumentLoader, DocumentLoadingError
from ingestion.image_loader import (
    get_tesseract_cmd,
    preprocess_image,
)

logger = logging.getLogger(__name__)


class PDFDocumentLoader(BaseDocumentLoader):
    """PDF loader supporting both digital PDFs and scanned PDF fallback via OCR."""

    def __init__(
        self,
        enable_ocr_fallback: bool = True,
        min_text_chars: int = 15,
        tesseract_cmd: str | None = None,
    ) -> None:
        """Initialize the PDF loader.

        Args:
            enable_ocr_fallback: Whether to run OCR on pages with insufficient extracted text.
            min_text_chars: Character count threshold below which a page is treated as scanned.
            tesseract_cmd: Optional path to tesseract binary.
        """
        self.enable_ocr_fallback = enable_ocr_fallback
        self.min_text_chars = min_text_chars
        cmd = tesseract_cmd or get_tesseract_cmd()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def _ocr_page(self, pdf_path: Path, page_index: int) -> str | None:
        """Render a PDF page to image and perform OCR extraction.

        Args:
            pdf_path: Path to the PDF file.
            page_index: 0-indexed page number.

        Returns:
            Extracted text string, or None if OCR failed or yielded no text.
        """
        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
            if page_index >= len(pdf):
                return None
            page = pdf[page_index]
            # Render page at 2x resolution for clear OCR
            page_img = page.render(scale=2.0).to_pil()
            proc_img = preprocess_image(page_img)
            text = pytesseract.image_to_string(proc_img).strip()
            return text if text else None
        except Exception as exc:
            logger.warning(
                "OCR fallback failed on page %d of '%s': %s",
                page_index + 1,
                pdf_path.name,
                exc,
            )
            return None

    def load(
        self,
        source: str | Path,
        enable_ocr_fallback: bool | None = None,
        min_text_chars: int | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Extract text from a PDF, falling back to OCR only for scanned/empty pages.

        Args:
            source: Path to the PDF file.
            enable_ocr_fallback: Override fallback behavior for this load.
            min_text_chars: Override character threshold for text usability.
            **kwargs: Extra parameters passed to PyPDFLoader.

        Returns:
            List of LangChain Document objects corresponding to pages.

        Raises:
            DocumentLoadingError: If the PDF is corrupted, empty, or unreadable.
        """
        path = Path(source)
        if not path.exists():
            raise DocumentLoadingError(f"PDF file not found: {source}")

        if path.stat().st_size == 0:
            raise DocumentLoadingError(f"PDF file '{path.name}' is empty (0 bytes).")

        do_ocr = (
            enable_ocr_fallback
            if enable_ocr_fallback is not None
            else self.enable_ocr_fallback
        )
        min_chars = (
            min_text_chars
            if min_text_chars is not None
            else self.min_text_chars
        )

        try:
            loader = PyPDFLoader(file_path=str(path), **kwargs)
            docs = loader.load()
        except Exception as exc:
            raise DocumentLoadingError(
                f"Failed to extract text from PDF '{path.name}': {exc}"
            ) from exc

        if not docs:
            raise DocumentLoadingError(f"PDF file '{path.name}' contains no readable pages.")

        # Process each page and apply OCR fallback only when text is insufficient
        for idx, doc in enumerate(docs):
            page_idx = doc.metadata.get("page", idx)
            doc.metadata["source"] = str(path)
            doc.metadata["filename"] = path.name
            doc.metadata["source_type"] = "pdf"
            doc.metadata["page"] = page_idx
            doc.metadata["page_number"] = page_idx + 1

            has_usable_text = len(doc.page_content.strip()) >= min_chars

            if has_usable_text:
                # Usable text already extracted by PyPDF; avoid OCR
                doc.metadata["ocr_applied"] = False
            elif do_ocr:
                # Text extraction yielded insufficient text; trigger OCR fallback
                ocr_text = self._ocr_page(path, page_idx)
                if ocr_text:
                    doc.page_content = ocr_text
                    doc.metadata["ocr_applied"] = True
                else:
                    doc.metadata["ocr_applied"] = False
                    doc.metadata["ocr_note"] = "OCR returned no text or encountered an error"
            else:
                doc.metadata["ocr_applied"] = False

        return docs
