"""Ingestion package for loading, parsing, and normalizing diverse knowledge sources."""

from ingestion import (
    document_loader,
    document_processor,
    docx_loader,
    image_loader,
    pdf_loader,
    text_loader,
    web_loader,
)
from ingestion.document_loader import (
    BaseDocumentLoader,
    DocumentLoader,
    DocumentLoadingError,
    document_loader as default_document_loader,
)
from ingestion.document_processor import (
    DocumentProcessor,
    document_processor as default_document_processor,
)
from ingestion.docx_loader import DOCXDocumentLoader
from ingestion.pdf_loader import PDFDocumentLoader
from ingestion.text_loader import ManualTextLoader, TextDocumentLoader
from ingestion.web_loader import WebDocumentLoader

__all__ = [
    "document_loader",
    "document_processor",
    "docx_loader",
    "image_loader",
    "pdf_loader",
    "text_loader",
    "web_loader",
    "BaseDocumentLoader",
    "DocumentLoader",
    "DocumentLoadingError",
    "DocumentProcessor",
    "DOCXDocumentLoader",
    "PDFDocumentLoader",
    "TextDocumentLoader",
    "ManualTextLoader",
    "WebDocumentLoader",
    "default_document_loader",
    "default_document_processor",
]
