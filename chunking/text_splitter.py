"""Document chunking and text splitting using LangChain RecursiveCharacterTextSplitter.

Provides configurable splitting, unique chunk ID generation, metadata preservation,
and bidirectional JSON serialization.
"""

import json
from pathlib import Path
import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


class ChunkingConfigurationError(ValueError):
    """Raised when chunk size or chunk overlap configurations are invalid."""
    pass


def validate_chunking_config(chunk_size: int, chunk_overlap: int) -> None:
    """Validate that chunk size and overlap parameters adhere to pipeline constraints.

    Args:
        chunk_size: Maximum characters per chunk. Must be > 0.
        chunk_overlap: Overlapping characters between consecutive chunks. Must be >= 0 and < chunk_size.

    Raises:
        ChunkingConfigurationError: If any condition is violated.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ChunkingConfigurationError(
            f"Chunk size must be an integer greater than zero. Received: {chunk_size}"
        )

    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool) or chunk_overlap < 0:
        raise ChunkingConfigurationError(
            f"Chunk overlap must be a non-negative integer. Received: {chunk_overlap}"
        )

    if chunk_overlap >= chunk_size:
        raise ChunkingConfigurationError(
            f"Chunk overlap ({chunk_overlap}) must be strictly smaller than chunk size ({chunk_size})."
        )


def generate_chunk_id(source_name: str, doc_index: int, chunk_index: int) -> str:
    """Generate a clean, deterministic, unique chunk ID.

    Args:
        source_name: Name or path of the source file/document.
        doc_index: Index of the document within the batch.
        chunk_index: Index of the chunk within the document.

    Returns:
        Formatted unique chunk identifier string.
    """
    stem = Path(str(source_name)).stem
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", stem).strip("_")
    prefix = sanitized if sanitized else f"doc_{doc_index}"
    return f"{prefix}_d{doc_index}_c{chunk_index}"


class ChunkingManager:
    """Manages document splitting into compliant LangChain Document chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ) -> None:
        """Initialize ChunkingManager with validated parameters.

        Args:
            chunk_size: Target chunk size (default from settings or 1000).
            chunk_overlap: Overlap size (default from settings or 200).
            separators: Optional list of splitting separators.
            keep_separator: Whether to preserve separators in chunk content.
        """
        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else getattr(settings, "DEFAULT_CHUNK_SIZE", 1000)
        )
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else getattr(settings, "DEFAULT_CHUNK_OVERLAP", 200)
        )
        self.separators = separators
        self.keep_separator = keep_separator

        # Validate initial configuration
        validate_chunking_config(self.chunk_size, self.chunk_overlap)

    def _get_splitter(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> RecursiveCharacterTextSplitter:
        """Create and return a validated RecursiveCharacterTextSplitter instance."""
        size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

        validate_chunking_config(size, overlap)

        kwargs: dict[str, Any] = {
            "chunk_size": size,
            "chunk_overlap": overlap,
            "keep_separator": self.keep_separator,
        }
        if self.separators is not None:
            kwargs["separators"] = self.separators

        return RecursiveCharacterTextSplitter(**kwargs)

    def split_documents(
        self,
        documents: list[Document] | Document,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Document]:
        """Split one or more LangChain Document objects into smaller chunks.

        - Preserves all original metadata.
        - Generates unique chunk IDs.
        - Filters out empty or whitespace-only chunks.
        - Returns a list of standardized LangChain Document objects.

        Args:
            documents: Single Document or list of Document objects.
            chunk_size: Optional runtime override for chunk size.
            chunk_overlap: Optional runtime override for chunk overlap.

        Returns:
            List of resulting chunk Document objects.
        """
        doc_list = [documents] if isinstance(documents, Document) else list(documents)
        splitter = self._get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        all_chunks: list[Document] = []

        for doc_idx, original_doc in enumerate(doc_list):
            if not isinstance(original_doc, Document):
                raise TypeError(
                    f"Expected LangChain Document, got {type(original_doc).__name__}"
                )

            # Skip empty or whitespace-only documents to avoid empty chunks
            content = original_doc.page_content
            if not content or not content.strip():
                continue

            # Split raw text into chunks
            raw_splits = splitter.split_text(content)

            chunk_counter = 0
            for split_text in raw_splits:
                cleaned_chunk = split_text.strip()
                # Avoid empty chunks
                if not cleaned_chunk:
                    continue

                # Copy and preserve all original metadata
                meta = dict(original_doc.metadata) if original_doc.metadata else {}

                source_name = meta.get("filename") or meta.get("source") or f"doc_{doc_idx}"
                chunk_id = generate_chunk_id(str(source_name), doc_idx, chunk_counter)

                # Assign chunk tracking metadata
                meta["chunk_id"] = chunk_id
                meta["chunk_index"] = chunk_counter
                meta["chunk_char_count"] = len(cleaned_chunk)

                chunk_doc = Document(page_content=cleaned_chunk, metadata=meta)
                all_chunks.append(chunk_doc)
                chunk_counter += 1

        return all_chunks

    def split_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Document]:
        """Convenience method to split raw text into Document chunks.

        Args:
            text: Text to split.
            metadata: Optional metadata to attach to all chunks.
            chunk_size: Optional runtime chunk size.
            chunk_overlap: Optional runtime chunk overlap.

        Returns:
            List of chunk Document objects.
        """
        doc = Document(page_content=text, metadata=metadata or {})
        return self.split_documents([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# ---------------------------------------------------------------------------
# Reusable JSON Serialization & Deserialization
# ---------------------------------------------------------------------------

def serialize_chunk(chunk: Document) -> dict[str, Any]:
    """Serialize a single chunk Document into a structured dictionary.

    Args:
        chunk: LangChain Document chunk.

    Returns:
        Dictionary containing chunk_id, content, and metadata.
    """
    metadata = dict(chunk.metadata) if chunk.metadata else {}
    chunk_id = metadata.get("chunk_id", "unknown_chunk")

    return {
        "chunk_id": chunk_id,
        "content": chunk.page_content,
        "metadata": metadata,
    }


def serialize_chunks(chunks: list[Document]) -> list[dict[str, Any]]:
    """Serialize a list of Document chunks into a list of structured dictionaries.

    Args:
        chunks: List of Document chunks.

    Returns:
        List of serialized chunk dictionaries.
    """
    return [serialize_chunk(c) for c in chunks]


def export_chunks_to_json(
    chunks: list[Document],
    file_path: str | Path | None = None,
    indent: int = 2,
) -> str:
    """Export chunks to a JSON string, optionally writing to disk.

    Args:
        chunks: List of LangChain Document chunks.
        file_path: Optional path to save the JSON file.
        indent: JSON indentation formatting (default: 2).

    Returns:
        The JSON string representation.
    """
    data = serialize_chunks(chunks)
    json_str = json.dumps(data, indent=indent, ensure_ascii=False)

    if file_path:
        out_path = Path(file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")

    return json_str


def deserialize_chunks_from_json(json_data: str | Path) -> list[Document]:
    """Parse JSON back into LangChain Document objects.

    Args:
        json_data: JSON formatted string or Path to a JSON file.

    Returns:
        List of LangChain Document objects reconstructed from JSON.
    """
    raw_str: str
    if isinstance(json_data, Path) or (
        isinstance(json_data, str) and not json_data.strip().startswith(("{", "[")) and Path(json_data).exists()
    ):
        raw_str = Path(json_data).read_text(encoding="utf-8")
    else:
        raw_str = str(json_data)

    parsed = json.loads(raw_str)
    if not isinstance(parsed, list):
        raise ValueError("Invalid chunks JSON format: Expected a JSON array of chunk objects.")

    reconstructed_documents: list[Document] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        meta = item.get("metadata", {})
        # Ensure chunk_id is recorded in metadata if not already present
        if "chunk_id" in item and "chunk_id" not in meta:
            meta["chunk_id"] = item["chunk_id"]

        doc = Document(page_content=content, metadata=meta)
        reconstructed_documents.append(doc)

    return reconstructed_documents
