"""Vector store management and indexing using ChromaDB and LangChain.

Provides persistent collection management, idempotent batch upserts, duplicate detection,
metadata filtering, and LangChain Chroma handoff integration.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

import chromadb
from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from embedding.embedding_factory import get_embedding_model
from embedding.embedding_manager import EmbeddingResult
from enrichment.metadata_enricher import sanitize_metadata

logger = logging.getLogger(__name__)


class IndexingError(RuntimeError):
    """Raised when an error occurs during vector indexing or collection operations."""
    pass


@dataclass
class IndexingResult:
    """Encapsulates the outcome of a vector database indexing operation.

    Attributes:
        indexed_count: Number of vector chunks successfully indexed.
        skipped_duplicates: Number of duplicate chunks skipped in batch.
        replaced_sources: List of source names whose existing records were deleted.
        collection_name: Target vector store collection name.
        total_in_collection: Total vectors currently stored in the collection.
        duration_seconds: Wall-clock time spent executing indexing.
    """

    indexed_count: int
    skipped_duplicates: int
    replaced_sources: list[str]
    collection_name: str
    total_in_collection: int
    duration_seconds: float


def _build_chroma_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Format user metadata filters into valid ChromaDB query filter syntax.

    Combines multiple filters using the '$and' operator.

    Args:
        filters: Mapping of metadata keys to target values.

    Returns:
        ChromaDB compatible filter dictionary, or None if filters is empty.
    """
    if not filters:
        return None

    clean_filters = {k: v for k, v in filters.items() if v is not None}
    if not clean_filters:
        return None

    if len(clean_filters) == 1:
        return clean_filters

    return {"$and": [{k: v} for k, v in clean_filters.items()]}


class VectorStoreManager:
    """Manages persistent ChromaDB vector storage and LangChain Chroma integration."""

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        collection_name: str | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        """Initialize VectorStoreManager without opening storage or initializing models.

        Args:
            persist_directory: Destination directory for persistent storage.
            collection_name: Target Chroma collection name.
            embeddings: Optional pre-injected LangChain Embeddings instance.
        """
        self.persist_directory = Path(
            persist_directory
            if persist_directory is not None
            else getattr(settings, "VECTOR_STORE_DIR", "data/vector_store")
        )
        self.collection_name = str(
            collection_name
            if collection_name is not None
            else getattr(settings, "VECTOR_COLLECTION_NAME", "canonix_knowledge_base")
        )
        self._embeddings = embeddings
        self._client: chromadb.PersistentClient | None = None
        self._collection: Any = None
        self._langchain_vs: Chroma | None = None

    def _get_embeddings(self) -> Embeddings:
        """Resolve or lazily load the configured embeddings engine."""
        if self._embeddings is None:
            self._embeddings = get_embedding_model()
        return self._embeddings

    def _get_client(self) -> chromadb.PersistentClient:
        """Lazily initialize and return the persistent Chroma client."""
        if self._client is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=chromadb.config.Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self) -> Any:
        """Lazily retrieve or create the target collection configured with cosine space."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index(
        self,
        result: EmbeddingResult,
        replace_existing_sources: bool = False,
    ) -> IndexingResult:
        """Index precomputed vectors into the persistent ChromaDB collection.

        Deduplicates chunks with identical content_hash in the batch, optionally
        purges existing records for incoming sources, and verifies vector dimensionality.

        Args:
            result: EmbeddingResult containing document chunks and vectors.
            replace_existing_sources: Whether to delete existing vectors for incoming sources.

        Returns:
            IndexingResult with indexing counts, stats, and timings.

        Raises:
            IndexingError: If vector dimension mismatches existing collection or upsert fails.
        """
        start_time = time.perf_counter()
        collection = self._get_collection()

        if not result.chunks or not result.vectors:
            return IndexingResult(
                indexed_count=0,
                skipped_duplicates=0,
                replaced_sources=[],
                collection_name=self.collection_name,
                total_in_collection=collection.count(),
                duration_seconds=time.perf_counter() - start_time,
            )

        incoming_dim = result.dimension
        if incoming_dim == 0 and result.vectors:
            incoming_dim = len(result.vectors[0])

        # Validate vector dimensionality against existing collection content
        current_count = collection.count()
        if current_count > 0:
            sample = collection.get(limit=1, include=["embeddings"])
            existing_embeddings = sample.get("embeddings")
            if existing_embeddings is not None and len(existing_embeddings) > 0 and len(existing_embeddings[0]) > 0:
                existing_dim = len(existing_embeddings[0])
                if existing_dim != incoming_dim:
                    raise IndexingError(
                        f"Vector dimension mismatch: collection '{self.collection_name}' has "
                        f"dimension {existing_dim}, but incoming embeddings have dimension {incoming_dim}. "
                        "Please reset the collection or switch back to the original embedding model."
                    )

        replaced_sources: list[str] = []
        if replace_existing_sources:
            distinct_sources = {
                str(c.metadata.get("source"))
                for c in result.chunks
                if c.metadata and c.metadata.get("source")
            }
            for src in distinct_sources:
                self.delete_by_source(src)
                replaced_sources.append(src)

        # In-batch deduplication by content_hash
        seen_hashes: set[str] = set()
        ids_to_index: list[str] = []
        vectors_to_index: list[list[float]] = []
        documents_to_index: list[str] = []
        metadatas_to_index: list[dict[str, str | int | float | bool]] = []
        skipped_duplicates = 0

        for chunk, vector in zip(result.chunks, result.vectors):
            meta = dict(chunk.metadata) if chunk.metadata else {}
            content_hash = meta.get("content_hash")

            if content_hash:
                if content_hash in seen_hashes:
                    skipped_duplicates += 1
                    continue
                seen_hashes.add(content_hash)

            vector_id = meta.get("vector_id") or meta.get("chunk_id")
            if not vector_id:
                raise IndexingError("Chunk is missing vector_id and chunk_id in metadata.")

            sanitized_meta = sanitize_metadata(meta)

            ids_to_index.append(str(vector_id))
            vectors_to_index.append(vector)
            documents_to_index.append(chunk.page_content)
            metadatas_to_index.append(sanitized_meta)

        # Batch upsert in chunks of 500 or less
        batch_size = 500
        total_items = len(ids_to_index)
        try:
            for i in range(0, total_items, batch_size):
                end_i = i + batch_size
                collection.upsert(
                    ids=ids_to_index[i:end_i],
                    embeddings=vectors_to_index[i:end_i],
                    documents=documents_to_index[i:end_i],
                    metadatas=metadatas_to_index[i:end_i],
                )
        except Exception as exc:
            raise IndexingError(f"Failed to upsert vectors into collection: {exc}") from exc

        duration = time.perf_counter() - start_time
        total_in_collection = collection.count()

        return IndexingResult(
            indexed_count=total_items,
            skipped_duplicates=skipped_duplicates,
            replaced_sources=replaced_sources,
            collection_name=self.collection_name,
            total_in_collection=total_in_collection,
            duration_seconds=duration,
        )

    def as_langchain_vectorstore(self) -> Chroma:
        """Create and return a LangChain Chroma wrapper bound to the active collection.

        Returns:
            langchain_chroma.Chroma instance configured for similarity search and QA chains.
        """
        if self._langchain_vs is None:
            client = self._get_client()
            embeddings = self._get_embeddings()
            self._langchain_vs = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._langchain_vs

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Perform semantic similarity search with metadata filtering and relevance scoring.

        Args:
            query: Natural language search string.
            k: Top-k result limit. Defaults to settings.RETRIEVAL_TOP_K.
            filters: Optional metadata filters combined via $and.

        Returns:
            List of (Document, score) tuples sorted by relevance.

        Raises:
            ValueError: If query is empty.
            IndexingError: If search query fails.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty or whitespace.")

        top_k = k if k is not None else int(getattr(settings, "RETRIEVAL_TOP_K", 4))
        chroma_filter = _build_chroma_filter(filters)
        vs = self.as_langchain_vectorstore()

        try:
            results = vs.similarity_search_with_score(
                query=query.strip(),
                k=top_k,
                filter=chroma_filter,
            )
            return results
        except Exception as exc:
            raise IndexingError(f"Similarity search failed: {exc}") from exc

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Perform semantic similarity search using a precomputed embedding vector.

        Args:
            embedding: Query embedding vector (list of floats).
            k: Top-k maximum results to return.
            filters: Optional metadata filters combined via $and.

        Returns:
            List of (Document, score) tuples sorted by relevance score.

        Raises:
            ValueError: If embedding vector is empty or invalid.
            IndexingError: If query execution against the vector store fails.
        """
        if not embedding or not isinstance(embedding, (list, tuple)):
            raise ValueError("Embedding vector must be a non-empty list of floats.")

        if self.count() == 0:
            return []

        top_k = k if k is not None else int(getattr(settings, "RETRIEVAL_TOP_K", 4))
        chroma_filter = _build_chroma_filter(filters)
        collection = self._get_collection()

        n_results = min(top_k, max(1, collection.count()))
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [list(embedding)],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if chroma_filter:
            query_kwargs["where"] = chroma_filter

        try:
            res = collection.query(**query_kwargs)
            results: list[tuple[Document, float]] = []
            if res and res.get("documents") and res["documents"] and len(res["documents"][0]) > 0:
                docs = res["documents"][0]
                metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
                dists = res["distances"][0] if res.get("distances") else [0.0] * len(docs)

                for doc_text, meta, dist in zip(docs, metas, dists):
                    # For cosine distance, similarity is max(0.0, 1.0 - distance)
                    score = max(0.0, 1.0 - float(dist)) if dist is not None else 1.0
                    doc = Document(page_content=doc_text, metadata=dict(meta) if meta else {})
                    results.append((doc, score))

            # Ensure sorted descending by similarity score
            results.sort(key=lambda item: item[1], reverse=True)
            return results
        except Exception as exc:
            raise IndexingError(f"Vector similarity search failed: {exc}") from exc

    def count(self) -> int:
        """Return the current number of vectors stored in the collection."""
        return self._get_collection().count()

    def list_sources(self) -> list[str]:
        """List distinct source names currently present in the collection.

        Returns:
            Sorted list of unique source identifiers.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        data = collection.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in data.get("metadatas", []) or []:
            if meta and meta.get("source"):
                sources.add(str(meta["source"]))
        return sorted(sources)

    def delete_by_source(self, source: str) -> int:
        """Delete all vectors associated with a specific source identifier.

        Args:
            source: Source string to purge.
            Returns:
                Number of vectors deleted.
        """
        collection = self._get_collection()
        matching = collection.get(where={"source": source})
        ids_to_delete = matching.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def reset_collection(self) -> None:
        """Purge and recreate the active Chroma collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass

        self._collection = None
        self._langchain_vs = None
        # Re-create empty collection
        self._get_collection()

    def close(self) -> None:
        """Release Chroma system cache and reset active client handles."""
        self._collection = None
        self._langchain_vs = None
        self._client = None
        try:
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
