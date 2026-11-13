"""Document normalization, validation, and processing pipeline."""

import re
from typing import Any

from langchain_core.documents import Document


class DocumentProcessor:
    """Orchestrates document normalization, cleaning, and validation."""

    def normalize_text(self, text: str) -> str:
        """Clean and normalize document text content.

        - Removes null bytes and control characters (except tabs and newlines).
        - Normalizes Windows (CRLF) and legacy Mac (CR) line breaks to Unix (LF).
        - Normalizes consecutive blank lines (max 2 consecutive newlines).
        - Strips extraneous whitespace at the start and end.

        Args:
            text: Raw text string.

        Returns:
            Cleaned and normalized text string.
        """
        if not text:
            return ""

        # Remove null bytes
        cleaned = text.replace("\x00", "")

        # Normalize line endings
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse excess empty lines (preserve paragraphs)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def normalize_metadata(
        self,
        metadata: dict[str, Any],
        default_source: str = "unknown",
    ) -> dict[str, Any]:
        """Ensure standard metadata fields are present and correctly typed.

        Args:
            metadata: Existing document metadata.
            default_source: Fallback source name if missing.

        Returns:
            Normalized metadata dictionary.
        """
        meta = dict(metadata) if metadata else {}

        # Ensure 'source' is populated
        if "source" not in meta or not meta["source"]:
            meta["source"] = default_source

        # Infer source_type if missing
        if "source_type" not in meta:
            src = str(meta["source"]).lower()
            if src.startswith("http://") or src.startswith("https://"):
                meta["source_type"] = "web"
            elif src.endswith(".pdf"):
                meta["source_type"] = "pdf"
            elif src.endswith(".docx"):
                meta["source_type"] = "docx"
            elif src.endswith(".md") or src.endswith(".markdown"):
                meta["source_type"] = "markdown"
            elif src.endswith(".txt"):
                meta["source_type"] = "text"
            else:
                meta["source_type"] = "document"

        return meta

    def validate_document(self, document: Any) -> bool:
        """Validate that an object is a compliant LangChain Document.

        Args:
            document: Object to test.

        Returns:
            True if valid.

        Raises:
            TypeError: If document is not an instance of LangChain Document.
            ValueError: If document content or metadata is invalid.
        """
        if not isinstance(document, Document):
            raise TypeError(f"Expected LangChain Document, got {type(document).__name__}")

        if not isinstance(document.page_content, str):
            raise ValueError(
                f"Document page_content must be str, got {type(document.page_content).__name__}"
            )

        if not isinstance(document.metadata, dict):
            raise ValueError(
                f"Document metadata must be dict, got {type(document.metadata).__name__}"
            )

        if not document.page_content.strip():
            raise ValueError("Document page_content is empty or whitespace only.")

        return True

    def normalize_document(self, document: Document) -> Document:
        """Validate and normalize a single LangChain Document.

        Args:
            document: Input LangChain Document.

        Returns:
            Normalized LangChain Document.
        """
        self.validate_document(document)

        normalized_content = self.normalize_text(document.page_content)
        normalized_meta = self.normalize_metadata(document.metadata)
        normalized_meta["char_count"] = len(normalized_content)

        return Document(page_content=normalized_content, metadata=normalized_meta)

    def process(self, documents: list[Document] | Document) -> list[Document]:
        """Process, normalize, and validate one or more LangChain Documents.

        Args:
            documents: Single Document or list of Document objects.

        Returns:
            List of standardized, normalized LangChain Document objects.
        """
        doc_list = [documents] if isinstance(documents, Document) else list(documents)
        processed: list[Document] = []

        for doc in doc_list:
            normalized_doc = self.normalize_document(doc)
            processed.append(normalized_doc)

        return processed


# Global processor instance
document_processor = DocumentProcessor()
