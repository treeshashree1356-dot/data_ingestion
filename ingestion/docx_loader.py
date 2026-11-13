"""DOCX document loader using LangChain Docx2txtLoader."""

from pathlib import Path
from typing import Any
from zipfile import BadZipFile

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

from ingestion.document_loader import BaseDocumentLoader, DocumentLoadingError


class DOCXDocumentLoader(BaseDocumentLoader):
    """Loader for Microsoft Word DOCX documents."""

    def load(self, source: str | Path, **kwargs: Any) -> list[Document]:
        """Extract text from a DOCX file and convert to LangChain Documents.

        Args:
            source: File path to the DOCX file.
            **kwargs: Additional options.

        Returns:
            List of LangChain Document objects.

        Raises:
            DocumentLoadingError: If the file is missing, empty, or not a valid DOCX.
        """
        path = Path(source)
        if not path.exists():
            raise DocumentLoadingError(f"DOCX file not found: {source}")

        if path.stat().st_size == 0:
            raise DocumentLoadingError(f"DOCX file '{path.name}' is empty (0 bytes).")

        try:
            loader = Docx2txtLoader(str(path))
            docs = loader.load()
        except (BadZipFile, Exception) as exc:
            raise DocumentLoadingError(
                f"Failed to extract text from DOCX '{path.name}': {exc}"
            ) from exc

        # Standardize metadata
        for doc in docs:
            doc.metadata["source"] = str(path)
            doc.metadata["filename"] = path.name
            doc.metadata["source_type"] = "docx"

        return docs
