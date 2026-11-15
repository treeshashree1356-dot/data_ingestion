"""High-level service coordinating metadata enrichment, embedding, and indexing.

Encapsulates all downstream RAG ingestion business logic, ensuring UI components
remain purely presentational.
"""

from collections.abc import Callable
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from chunking.text_splitter import export_chunks_to_json
from config.settings import settings
from embedding.embedding_manager import EmbeddingManager, EmbeddingResult
from enrichment.metadata_enricher import EnrichmentOptions, MetadataEnricher
from indexing.vector_store import IndexingResult, VectorStoreManager

logger = logging.getLogger(__name__)


class IndexingService:
    """Service orchestrating document metadata enrichment, embedding, and vector indexing."""

    def __init__(
        self,
        enricher: MetadataEnricher | None = None,
        embedding_manager: EmbeddingManager | None = None,
        vector_store: VectorStoreManager | None = None,
    ) -> None:
        """Initialize IndexingService with optional injected dependencies.

        Defaults are lazily created upon first access to prevent premature model
        downloads or database handle initialization.

        Args:
            enricher: Optional MetadataEnricher instance.
            embedding_manager: Optional EmbeddingManager instance.
            vector_store: Optional VectorStoreManager instance.
        """
        self._enricher = enricher
        self._embedding_manager = embedding_manager
        self._vector_store = vector_store

    @property
    def enricher(self) -> MetadataEnricher:
        """Return or lazily instantiate the MetadataEnricher."""
        if self._enricher is None:
            self._enricher = MetadataEnricher()
        return self._enricher

    @property
    def embedding_manager(self) -> EmbeddingManager:
        """Return or lazily instantiate the EmbeddingManager."""
        if self._embedding_manager is None:
            self._embedding_manager = EmbeddingManager()
        return self._embedding_manager

    @property
    def vector_store(self) -> VectorStoreManager:
        """Return or lazily instantiate the VectorStoreManager."""
        if self._vector_store is None:
            self._vector_store = VectorStoreManager()
        return self._vector_store

    def enrich_chunks(
        self,
        chunks: list[Document],
        options: EnrichmentOptions | None = None,
        per_source_options: dict[str, EnrichmentOptions] | None = None,
    ) -> list[Document]:
        """Enrich chunks with metadata, access permissions, timestamps, and hashes.

        Args:
            chunks: List of raw LangChain Document chunks.
            options: Optional global EnrichmentOptions.
            per_source_options: Optional per-source option overrides.

        Returns:
            List of enriched LangChain Document chunks.
        """
        logger.info("Enriching %d chunks with metadata.", len(chunks))
        enriched = self.enricher.enrich_chunks(
            chunks,
            options=options,
            per_source_options=per_source_options,
        )
        logger.info("Enrichment complete. %d valid chunks enriched.", len(enriched))
        return enriched

    def embed_chunks(
        self,
        enriched_chunks: list[Document],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EmbeddingResult:
        """Generate vector embeddings for enriched chunks.

        Args:
            enriched_chunks: List of enriched LangChain Document chunks.
            progress_callback: Optional callable for batch progress updates (done, total).

        Returns:
            EmbeddingResult containing precomputed vectors and metadata.
        """
        logger.info("Generating embeddings for %d enriched chunks.", len(enriched_chunks))
        result = self.embedding_manager.embed_chunks(
            enriched_chunks,
            progress_callback=progress_callback,
        )
        logger.info(
            "Embedding complete. Generated %d vectors of dimension %d in %.2fs.",
            len(result.vectors),
            result.dimension,
            result.duration_seconds,
        )
        return result

    def index_embeddings(
        self,
        result: EmbeddingResult,
        replace_existing_sources: bool = False,
    ) -> IndexingResult:
        """Store precomputed embeddings and documents in the vector database.

        Args:
            result: EmbeddingResult with documents and precomputed vectors.
            replace_existing_sources: Whether to remove existing vectors for incoming sources.

        Returns:
            IndexingResult containing vector count, duplicate stats, and timings.
        """
        logger.info(
            "Indexing %d vector embeddings into collection '%s'.",
            len(result.vectors),
            self.vector_store.collection_name,
        )
        idx_result = self.vector_store.index(
            result,
            replace_existing_sources=replace_existing_sources,
        )
        logger.info(
            "Indexing complete. Indexed: %d, Skipped duplicates: %d, Total in collection: %d.",
            idx_result.indexed_count,
            idx_result.skipped_duplicates,
            idx_result.total_in_collection,
        )
        return idx_result

    def run_full_pipeline(
        self,
        chunks: list[Document],
        options: EnrichmentOptions | None = None,
        replace_existing_sources: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> IndexingResult:
        """Execute enrichment, embedding, and vector indexing in an automated end-to-end pipeline.

        Args:
            chunks: List of Document chunks to process.
            options: Optional metadata enrichment configuration.
            replace_existing_sources: Whether to overwrite existing vectors for incoming sources.
            progress_callback: Optional progress reporter for the embedding phase.

        Returns:
            IndexingResult with final storage and indexing statistics.
        """
        logger.info("Executing full RAG downstream pipeline for %d chunks.", len(chunks))
        enriched_chunks = self.enrich_chunks(chunks, options=options)
        embedding_result = self.embed_chunks(
            enriched_chunks,
            progress_callback=progress_callback,
        )
        indexing_result = self.index_embeddings(
            embedding_result,
            replace_existing_sources=replace_existing_sources,
        )
        return indexing_result

    def search(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Perform similarity search on the indexed vector store.

        Delegates retrieval through QueryService bound to this service's vector_store
        to ensure query normalization, intent detection, and filter validation.

        Args:
            query: Natural language query string.
            k: Top-k maximum results to return.
            filters: Optional metadata filtering criteria.

        Returns:
            List of (Document, score) tuples matching query.
        """
        from services.query_service import QueryService

        logger.info("Executing similarity search for query: '%s' (k=%s, filters=%s)", query, k, filters)
        qs = QueryService(vector_store=self.vector_store)
        results = qs.search(query=query, k=k, filters=filters)
        logger.info("Search returned %d results.", len(results))
        return results

    def get_collection_stats(self) -> dict[str, Any]:
        """Retrieve overview statistics for the active vector store collection.

        Returns:
            Dictionary with collection_name, count, sources, embedding_model, persist_directory.
        """
        collection_name = self.vector_store.collection_name
        count = self.vector_store.count()
        sources = self.vector_store.list_sources()
        model_name = self.embedding_manager.model_name
        persist_dir = str(self.vector_store.persist_directory)

        return {
            "collection_name": collection_name,
            "count": count,
            "sources": sources,
            "embedding_model": model_name,
            "persist_directory": persist_dir,
        }

    def export_enriched_json(self, chunks: list[Document]) -> str:
        """Serialize enriched chunks into a structured JSON string.

        Args:
            chunks: List of enriched Document chunks.

        Returns:
            Formatted JSON string representation.
        """
        return export_chunks_to_json(chunks)


# Singleton instance
indexing_service = IndexingService()
