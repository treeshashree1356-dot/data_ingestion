"""Base document loader interface, errors, and central coordinator."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


class DocumentLoadingError(Exception):
    """Raised when document loading or parsing fails."""
    pass


class BaseDocumentLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    def load(self, source: Any, **kwargs: Any) -> list[Document]:
        """Load and parse documents from the specified source.

        Args:
            source: Source to load from (path, URL, raw string, etc.).
            **kwargs: Loader-specific arguments.

        Returns:
            List of LangChain Document objects.

        Raises:
            DocumentLoadingError: If document extraction or loading fails.
        """
        raise NotImplementedError


class DocumentLoader:
    """Coordinator that routes input sources to the appropriate specialized loader."""

    @staticmethod
    def get_loader_for_source(
        source: Any,
        source_type: str | None = None,
    ) -> BaseDocumentLoader:
        """Resolve the appropriate loader based on source format or explicit type.

        Args:
            source: Path, URL, or raw text.
            source_type: Optional explicit type indicator.

        Returns:
            An instance of BaseDocumentLoader.

        Raises:
            DocumentLoadingError: If the source format is unsupported.
        """
        # Lazy imports to avoid circular dependency
        from ingestion.pdf_loader import PDFDocumentLoader
        from ingestion.docx_loader import DOCXDocumentLoader
        from ingestion.text_loader import TextDocumentLoader, ManualTextLoader
        from ingestion.web_loader import WebDocumentLoader
        from ingestion.image_loader import ImageDocumentLoader

        type_lower = source_type.lower() if source_type else None

        # Check explicit source_type
        if type_lower == "pdf":
            return PDFDocumentLoader()
        if type_lower == "docx":
            return DOCXDocumentLoader()
        if type_lower in ("txt", "text", "md", "markdown"):
            return TextDocumentLoader()
        if type_lower in ("web", "url", "urls"):
            return WebDocumentLoader()
        if type_lower in ("manual", "manual_text", "raw_text"):
            return ManualTextLoader()
        if type_lower in ("image", "jpg", "jpeg", "png", "ocr"):
            return ImageDocumentLoader()

        # Infer from source structure
        if isinstance(source, (str, Path)):
            str_source = str(source).strip()
            # Check for URL
            if str_source.startswith("http://") or str_source.startswith("https://"):
                return WebDocumentLoader()

            # Check file extension
            suffix = Path(str_source).suffix.lower().lstrip(".")
            if suffix == "pdf":
                return PDFDocumentLoader()
            if suffix == "docx":
                return DOCXDocumentLoader()
            if suffix in ("txt", "md", "markdown"):
                return TextDocumentLoader()
            if suffix in ("jpg", "jpeg", "png"):
                return ImageDocumentLoader()

        # List of URLs
        if isinstance(source, (list, tuple)) and all(
            isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))
            for s in source
        ):
            return WebDocumentLoader()

        raise DocumentLoadingError(
            f"Cannot determine appropriate loader for source: {source} (type={source_type})"
        )

    def load(
        self,
        source: Any,
        source_type: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Load documents from a single source using the appropriate loader.

        Args:
            source: Source location, URL, or content.
            source_type: Optional explicit type indicator.
            **kwargs: Extra parameters passed to the loader.

        Returns:
            List of LangChain Document objects.
        """
        loader = self.get_loader_for_source(source, source_type=source_type)
        return loader.load(source, **kwargs)

    def load_multiple(
        self,
        sources: list[Any],
        source_types: list[str | None] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Load documents from multiple independent sources.

        Args:
            sources: List of source items (paths, URLs, or manual texts).
            source_types: Optional list of corresponding source types.
            **kwargs: Additional options.

        Returns:
            Combined list of LangChain Document objects.
        """
        all_documents: list[Document] = []
        types = source_types if source_types else [None] * len(sources)

        for src, stype in zip(sources, types):
            docs = self.load(src, source_type=stype, **kwargs)
            all_documents.extend(docs)

        return all_documents


# Global coordinator instance
document_loader = DocumentLoader()
