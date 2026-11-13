"""High-level ingestion service coordinating extraction, processing, and chunking.

Encapsulates all business logic so UI layers remain purely presentation-focused.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from chunking.text_splitter import (
    ChunkingManager,
    export_chunks_to_json,
    validate_chunking_config,
)
from config.settings import settings
from ingestion.document_loader import DocumentLoader, DocumentLoadingError
from ingestion.document_processor import DocumentProcessor
from utils.file_utils import save_uploaded_file
from utils.validators import SourceValidationError, validate_file

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Encapsulates the aggregated outcome of a multi-source extraction operation."""

    documents: list[Document] = field(default_factory=list)
    successful_count: int = 0
    failed_count: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    sources_processed: list[str] = field(default_factory=list)


class IngestionService:
    """Service orchestrating document validation, loading, normalization, and chunking."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        processor: DocumentProcessor | None = None,
        chunking_manager: ChunkingManager | None = None,
    ) -> None:
        """Initialize IngestionService with dependencies."""
        self.loader = loader or DocumentLoader()
        self.processor = processor or DocumentProcessor()
        self.chunking_manager = chunking_manager or ChunkingManager()

    def process_sources(
        self,
        uploaded_files: list[Any] | None = None,
        urls: list[str] | None = None,
        manual_text: str | None = None,
        manual_title: str | None = None,
        destination_dir: Path | str | None = None,
    ) -> IngestionResult:
        """Process and extract documents from all supplied sources with fault isolation.

        Ensures that failures on individual sources do not stop successful ones.

        Args:
            uploaded_files: List of file objects, Streamlit UploadedFiles, or Paths.
            urls: List of website URL strings.
            manual_text: Optional raw text snippet.
            manual_title: Optional label for the manual text.
            destination_dir: Optional custom directory to store uploaded files.

        Returns:
            IngestionResult with extracted documents and error diagnostics.
        """
        result = IngestionResult()
        target_upload_dir = Path(destination_dir) if destination_dir else settings.UPLOAD_DIR

        # 1. Process Uploaded Files
        if uploaded_files:
            for file_item in uploaded_files:
                fname = getattr(file_item, "name", str(file_item))
                try:
                    # Validate and save file safely into upload directory
                    saved_path = save_uploaded_file(
                        file_data=file_item,
                        original_filename=fname,
                        destination_dir=target_upload_dir,
                        validate=True,
                    )
                    # Load documents from saved file
                    raw_docs = self.loader.load(saved_path)
                    normalized_docs = self.processor.process(raw_docs)

                    result.documents.extend(normalized_docs)
                    result.successful_count += 1
                    result.sources_processed.append(fname)
                except (SourceValidationError, DocumentLoadingError, Exception) as exc:
                    logger.warning("Extraction failed for file '%s': %s", fname, exc)
                    result.failed_count += 1
                    result.errors.append({"source": fname, "error": str(exc)})

        # 2. Process Website URLs
        if urls:
            for raw_url in urls:
                clean_url = raw_url.strip()
                if not clean_url:
                    continue
                try:
                    raw_docs = self.loader.load(clean_url, source_type="web")
                    normalized_docs = self.processor.process(raw_docs)

                    result.documents.extend(normalized_docs)
                    result.successful_count += 1
                    result.sources_processed.append(clean_url)
                except (SourceValidationError, DocumentLoadingError, Exception) as exc:
                    logger.warning("Extraction failed for URL '%s': %s", clean_url, exc)
                    result.failed_count += 1
                    result.errors.append({"source": clean_url, "error": str(exc)})

        # 3. Process Manual Text
        if manual_text and manual_text.strip():
            title = manual_title or "manual_notes"
            try:
                raw_docs = self.loader.load(
                    manual_text,
                    source_type="manual_text",
                    title=title,
                )
                normalized_docs = self.processor.process(raw_docs)

                result.documents.extend(normalized_docs)
                result.successful_count += 1
                result.sources_processed.append(f"Manual text ({title})")
            except (SourceValidationError, DocumentLoadingError, Exception) as exc:
                logger.warning("Extraction failed for manual text: %s", exc)
                result.failed_count += 1
                result.errors.append({"source": title, "error": str(exc)})

        return result

    def chunk_documents(
        self,
        documents: list[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[Document]:
        """Validate settings and split documents into compliant chunks.

        Args:
            documents: List of LangChain Document objects.
            chunk_size: Maximum characters per chunk (> 0).
            chunk_overlap: Overlap characters between chunks (>= 0 and < chunk_size).

        Returns:
            List of generated chunk Document objects.

        Raises:
            ValueError: If chunk configuration is invalid.
        """
        validate_chunking_config(chunk_size, chunk_overlap)
        return self.chunking_manager.split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def export_chunks_json(
        self,
        chunks: list[Document],
        file_path: Path | str | None = None,
    ) -> str:
        """Export chunks into JSON string representation.

        Args:
            chunks: List of chunk Document objects.
            file_path: Optional destination file path.

        Returns:
            Formatted JSON string.
        """
        return export_chunks_to_json(chunks, file_path=file_path)


# Singleton ingestion service instance
ingestion_service = IngestionService()
