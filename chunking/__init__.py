"""Chunking package for splitting documents into configurable chunks."""

from chunking import text_splitter
from chunking.text_splitter import (
    ChunkingConfigurationError,
    ChunkingManager,
    deserialize_chunks_from_json,
    export_chunks_to_json,
    generate_chunk_id,
    serialize_chunk,
    serialize_chunks,
    validate_chunking_config,
)

__all__ = [
    "text_splitter",
    "ChunkingManager",
    "ChunkingConfigurationError",
    "validate_chunking_config",
    "generate_chunk_id",
    "serialize_chunk",
    "serialize_chunks",
    "export_chunks_to_json",
    "deserialize_chunks_from_json",
]
