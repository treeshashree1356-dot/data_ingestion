"""Text, Markdown, and Manual Text document loaders."""

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ingestion.document_loader import BaseDocumentLoader, DocumentLoadingError
from utils.validators import EmptyManualTextError, validate_manual_text


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for Plain Text (.txt) and Markdown (.md) documents."""

    def load(
        self,
        source: str | Path,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> list[Document]:
        """Extract text from a TXT or Markdown file.

        Args:
            source: Path to the text or markdown file.
            encoding: Text encoding (default: 'utf-8').
            **kwargs: Additional parameters.

        Returns:
            List containing a single LangChain Document object.

        Raises:
            DocumentLoadingError: If the file is missing, empty, or cannot be read.
        """
        path = Path(source)
        if not path.exists():
            raise DocumentLoadingError(f"File not found: {source}")

        if path.stat().st_size == 0:
            raise DocumentLoadingError(f"File '{path.name}' is empty (0 bytes).")

        content: str
        try:
            content = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                content = path.read_text(encoding="latin-1")
        except Exception as exc:
            raise DocumentLoadingError(f"Failed to read file '{path.name}': {exc}") from exc

        # Also reject files containing only whitespace
        if not content.strip():
            raise DocumentLoadingError(f"File '{path.name}' contains only whitespace or is empty.")

        ext = path.suffix.lower().lstrip(".")
        source_type = "markdown" if ext in ("md", "markdown") else "text"

        metadata = {
            "source": str(path),
            "filename": path.name,
            "source_type": source_type,
            "char_count": len(content),
        }

        return [Document(page_content=content, metadata=metadata)]


class ManualTextLoader(BaseDocumentLoader):
    """Loader for user-entered manual text inputs."""

    def load(
        self,
        source: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Convert manual raw text into a LangChain Document.

        Args:
            source: Raw string content.
            title: Optional title/identifier for the text snippet.
            metadata: Optional additional metadata fields.
            **kwargs: Extra arguments.

        Returns:
            List containing a single LangChain Document object.

        Raises:
            DocumentLoadingError: If text is empty or invalid.
        """
        try:
            cleaned_text = validate_manual_text(source)
        except (EmptyManualTextError, Exception) as exc:
            raise DocumentLoadingError(f"Manual text validation failed: {exc}") from exc

        doc_title = title or "manual_text"
        doc_metadata: dict[str, Any] = {
            "source": doc_title,
            "filename": doc_title,
            "source_type": "manual_text",
            "char_count": len(cleaned_text),
        }
        if metadata:
            doc_metadata.update(metadata)

        return [Document(page_content=cleaned_text, metadata=doc_metadata)]
