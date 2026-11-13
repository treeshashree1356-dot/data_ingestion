"""Factory and caching mechanism for LangChain embedding models.

Supports local HuggingFace sentence-transformers and DeterministicFakeEmbedding for testing.
Guarantees lazy initialization and process-level caching.
"""

from typing import Any
from langchain_core.embeddings import Embeddings

from config.settings import settings


class EmbeddingConfigurationError(ValueError):
    """Raised when an unsupported or invalid embedding provider is configured."""
    pass


# Module-level cache for embedding model instances: key is (provider, model_name)
_EMBEDDING_CACHE: dict[tuple[str, str], Embeddings] = {}


def clear_embedding_cache() -> None:
    """Clear cached embedding model instances (primarily for testing)."""
    _EMBEDDING_CACHE.clear()


def get_embedding_model(
    provider: str | None = None,
    model_name: str | None = None,
) -> Embeddings:
    """Retrieve or lazily instantiate an embedding model instance.

    Supported providers:
    - 'huggingface': HuggingFaceEmbeddings using sentence-transformers
    - 'fake': DeterministicFakeEmbedding for fast, offline testing

    Args:
        provider: Embedding provider name ('huggingface' or 'fake'). Defaults to settings.
        model_name: Model identifier string. Defaults to settings.

    Returns:
        Embeddings instance adhering to LangChain core interface.

    Raises:
        EmbeddingConfigurationError: If an unsupported provider is requested.
    """
    selected_provider = (
        provider or getattr(settings, "EMBEDDING_PROVIDER", "huggingface")
    ).lower().strip()

    selected_model = (
        model_name or getattr(settings, "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    ).strip()

    cache_key = (selected_provider, selected_model)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]

    instance: Embeddings
    if selected_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        device = getattr(settings, "EMBEDDING_DEVICE", "cpu")
        normalize = getattr(settings, "NORMALIZE_EMBEDDINGS", True)

        instance = HuggingFaceEmbeddings(
            model_name=selected_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize},
        )
    elif selected_provider == "fake":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        dim = int(getattr(settings, "FAKE_EMBEDDING_DIM", 384))
        instance = DeterministicFakeEmbedding(size=dim)
    else:
        raise EmbeddingConfigurationError(
            f"Unsupported embedding provider '{selected_provider}'. "
            f"Allowed providers: ('huggingface', 'fake')."
        )

    _EMBEDDING_CACHE[cache_key] = instance
    return instance
