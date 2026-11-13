"""Embedding generation and batch orchestration.

Wraps LangChain embedding models with batching, progress reporting,
vector dimension verification, and robust error handling.
"""

from collections.abc import Callable
from dataclasses import dataclass
import math
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from embedding.embedding_factory import get_embedding_model


class EmbeddingError(RuntimeError):
    """Raised when an error occurs during chunk or query embedding generation."""
    pass


@dataclass
class EmbeddingResult:
    """Encapsulates the outcome of a batch embedding generation operation.

    Attributes:
        chunks: List of Document chunks with added embedding metadata.
        vectors: Precomputed floating point embedding vectors.
        model_name: Name of the embedding model utilized.
        provider: Embedding provider (e.g., 'huggingface', 'fake').
        dimension: Dimensionality of the resulting embedding vectors.
        duration_seconds: Total wall-clock time spent generating embeddings.
        skipped_count: Number of chunks skipped (e.g. empty content).
    """

    chunks: list[Document]
    vectors: list[list[float]]
    model_name: str
    provider: str
    dimension: int
    duration_seconds: float
    skipped_count: int = 0


class EmbeddingManager:
    """Orchestrates batch vector embedding generation for LangChain Document chunks."""

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        batch_size: int | None = None,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize EmbeddingManager with optional pre-injected embeddings.

        Embeddings model is resolved lazily upon first use if not provided.

        Args:
            embeddings: Optional pre-instantiated LangChain Embeddings instance.
            batch_size: Batch size for chunk embedding processing.
            provider: Provider name ('huggingface', 'fake').
            model_name: Name/path of the embedding model.
        """
        self._embeddings = embeddings
        self.batch_size = (
            batch_size
            if batch_size is not None
            else int(getattr(settings, "EMBEDDING_BATCH_SIZE", 32))
        )
        self.provider = provider or getattr(settings, "EMBEDDING_PROVIDER", "huggingface")
        self.model_name = model_name or getattr(
            settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )

    def _get_embeddings(self) -> Embeddings:
        """Resolve or lazily instantiate the Embeddings engine."""
        if self._embeddings is None:
            self._embeddings = get_embedding_model(
                provider=self.provider,
                model_name=self.model_name,
            )
        return self._embeddings

    def embed_chunks(
        self,
        chunks: list[Document],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EmbeddingResult:
        """Embed document chunk texts in configurable batches with progress reporting.

        Args:
            chunks: List of LangChain Document chunks to embed.
            progress_callback: Optional callable receiving (done_count, total_count).

        Returns:
            EmbeddingResult containing chunks, vectors, dimension, and timing stats.

        Raises:
            EmbeddingError: If an error occurs during batch embedding or vector validation.
        """
        if not chunks:
            return EmbeddingResult(
                chunks=[],
                vectors=[],
                model_name=self.model_name,
                provider=self.provider,
                dimension=0,
                duration_seconds=0.0,
                skipped_count=0,
            )

        start_time = time.perf_counter()
        engine = self._get_embeddings()

        valid_chunks: list[Document] = []
        skipped_count = 0
        for c in chunks:
            if c and c.page_content and c.page_content.strip():
                valid_chunks.append(c)
            else:
                skipped_count += 1

        if not valid_chunks:
            return EmbeddingResult(
                chunks=[],
                vectors=[],
                model_name=self.model_name,
                provider=self.provider,
                dimension=0,
                duration_seconds=time.perf_counter() - start_time,
                skipped_count=skipped_count,
            )

        total_chunks = len(valid_chunks)
        all_vectors: list[list[float]] = []
        expected_dim: int | None = None
        processed_count = 0

        # Process in batches
        step = max(1, self.batch_size)
        for batch_idx, start_idx in enumerate(range(0, total_chunks, step)):
            batch_slice = valid_chunks[start_idx : start_idx + step]
            batch_texts = [c.page_content for c in batch_slice]

            try:
                batch_vectors = engine.embed_documents(batch_texts)
            except Exception as exc:
                raise EmbeddingError(
                    f"Embedding failed at batch {batch_idx + 1} (items {start_idx}..{start_idx + len(batch_texts)}): {exc}"
                ) from exc

            # Validate generated vectors
            for vec in batch_vectors:
                if not isinstance(vec, list):
                    vec = list(vec)

                vec_len = len(vec)
                if expected_dim is None:
                    expected_dim = vec_len
                elif vec_len != expected_dim:
                    raise EmbeddingError(
                        f"Inconsistent vector dimension at batch {batch_idx + 1}: "
                        f"expected {expected_dim}, received {vec_len}."
                    )

                # Check for NaN values
                for val in vec:
                    if math.isnan(val):
                        raise EmbeddingError(
                            f"NaN detected in generated embedding vector at batch {batch_idx + 1}."
                        )

                all_vectors.append(vec)

            processed_count += len(batch_slice)
            if progress_callback:
                progress_callback(processed_count, total_chunks)

        final_dim = expected_dim or 0
        duration = time.perf_counter() - start_time

        # Update metadata on chunks with embedding details
        enriched_chunks: list[Document] = []
        for c in valid_chunks:
            new_meta = dict(c.metadata) if c.metadata else {}
            new_meta["embedding_model"] = self.model_name
            new_meta["embedding_dim"] = final_dim
            enriched_chunks.append(Document(page_content=c.page_content, metadata=new_meta))

        return EmbeddingResult(
            chunks=enriched_chunks,
            vectors=all_vectors,
            model_name=self.model_name,
            provider=self.provider,
            dimension=final_dim,
            duration_seconds=duration,
            skipped_count=skipped_count,
        )

    def embed_query(self, text: str) -> list[float]:
        """Generate vector embedding for a single search or query text.

        Args:
            text: Query string.

        Returns:
            List of floats representing the query embedding.

        Raises:
            ValueError: If query text is empty or whitespace.
            EmbeddingError: If embedding generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty or whitespace.")

        engine = self._get_embeddings()
        try:
            vector = engine.embed_query(text.strip())
            return list(vector)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc
