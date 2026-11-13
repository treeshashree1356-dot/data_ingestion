"""Metadata enrichment for chunked LangChain documents.

Enriches document chunks with categorical classification, identity hashes,
access control permissions, UTC timestamps, content statistics, and tags
while strictly sanitizing metadata for vector database storage.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from langchain_core.documents import Document

from config.settings import settings
from enrichment.category_rules import CATEGORY_RULES, detect_category


class MetadataEnrichmentError(ValueError):
    """Raised when metadata enrichment fails or configuration is invalid."""
    pass


ALLOWED_ACCESS_LEVELS: tuple[str, ...] = ("public", "internal", "restricted")

REQUIRED_ENRICHED_FIELDS: tuple[str, ...] = (
    "source",
    "source_name",
    "source_type",
    "category",
    "access_level",
    "allowed_roles",
    "ingested_at",
    "enriched_at",
    "doc_id",
    "content_hash",
    "vector_id",
    "word_count",
    "char_count",
    "tags",
    "enrichment_version",
)


@dataclass
class EnrichmentOptions:
    """Configuration options governing metadata enrichment behavior.

    Attributes:
        category_override: Optional explicit category override.
        access_level: Access permission level (public, internal, restricted).
        allowed_roles: List or comma-separated roles permitted access.
        extra_tags: Additional user or system tags.
    """

    category_override: str | None = None
    access_level: str = field(
        default_factory=lambda: getattr(settings, "DEFAULT_ACCESS_LEVEL", "internal")
    )
    allowed_roles: list[str] | str = field(
        default_factory=lambda: getattr(settings, "DEFAULT_ALLOWED_ROLES", "all")
    )
    extra_tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate options upon instantiation."""
        # Validate access level
        if not isinstance(self.access_level, str):
            raise MetadataEnrichmentError(
                f"access_level must be a string. Received: {type(self.access_level)}"
            )
        self.access_level = self.access_level.strip().lower()
        if self.access_level not in ALLOWED_ACCESS_LEVELS:
            raise MetadataEnrichmentError(
                f"Invalid access_level '{self.access_level}'. "
                f"Allowed levels: {ALLOWED_ACCESS_LEVELS}"
            )

        # Validate category override if provided
        if self.category_override is not None:
            if not isinstance(self.category_override, str):
                raise MetadataEnrichmentError(
                    f"category_override must be a string. Received: {type(self.category_override)}"
                )
            self.category_override = self.category_override.strip().lower()
            if self.category_override not in CATEGORY_RULES:
                raise MetadataEnrichmentError(
                    f"Unknown category_override '{self.category_override}'. "
                    f"Allowed categories: {list(CATEGORY_RULES.keys())}"
                )

        # Standardize allowed_roles into a lowercase comma-separated string
        if isinstance(self.allowed_roles, list):
            roles = [r.strip().lower() for r in self.allowed_roles if str(r).strip()]
            self.allowed_roles = ",".join(roles) if roles else "all"
        elif isinstance(self.allowed_roles, str):
            roles = [r.strip().lower() for r in self.allowed_roles.split(",") if r.strip()]
            self.allowed_roles = ",".join(roles) if roles else "all"
        else:
            raise MetadataEnrichmentError(
                f"allowed_roles must be a list or comma-separated string. Received: {type(self.allowed_roles)}"
            )

        # Standardize extra_tags
        if isinstance(self.extra_tags, list):
            self.extra_tags = [str(t).strip().lower() for t in self.extra_tags if str(t).strip()]
        elif isinstance(self.extra_tags, str):
            self.extra_tags = [t.strip().lower() for t in self.extra_tags.split(",") if t.strip()]
        else:
            self.extra_tags = []


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Sanitize metadata dictionary to adhere to ChromaDB scalar value requirements.

    Converts lists, tuples, and sets to comma-separated strings, Path objects
    and other non-scalars to strings, drops None values, and preserves booleans,
    integers, floats, and strings.

    Args:
        meta: Raw metadata dictionary.

    Returns:
        Sanitized metadata dictionary containing only str, int, float, or bool.
    """
    sanitized: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if v is None:
            continue
        # Check bool first because bool subclasses int in Python
        if isinstance(v, bool):
            sanitized[k] = v
        elif isinstance(v, (int, float, str)):
            sanitized[k] = v
        elif isinstance(v, (list, tuple, set)):
            sanitized[k] = ",".join(str(elem).strip() for elem in v if str(elem).strip())
        elif isinstance(v, Path):
            sanitized[k] = str(v)
        else:
            sanitized[k] = str(v)
    return sanitized


def validate_enriched_metadata(meta: dict[str, Any]) -> None:
    """Validate that enriched metadata contains all required fields and scalar types.

    Args:
        meta: Enriched metadata dictionary to validate.

    Raises:
        MetadataEnrichmentError: If a required field is missing or contains non-scalar value.
    """
    for required_key in REQUIRED_ENRICHED_FIELDS:
        if required_key not in meta:
            raise MetadataEnrichmentError(
                f"Required enriched metadata field '{required_key}' is missing."
            )
        val = meta[required_key]
        if val is None:
            raise MetadataEnrichmentError(
                f"Required enriched metadata field '{required_key}' cannot be None."
            )
        if not isinstance(val, (str, int, float, bool)):
            raise MetadataEnrichmentError(
                f"Metadata field '{required_key}' must be scalar (str, int, float, bool). "
                f"Found {type(val)}."
            )

    for k, v in meta.items():
        if not isinstance(v, (str, int, float, bool)):
            raise MetadataEnrichmentError(
                f"Enriched metadata contains non-scalar value for key '{k}': {type(v)}"
            )


class MetadataEnricher:
    """Enriches Document chunks with standardized governance, classification, and identity metadata."""

    def __init__(self, default_options: EnrichmentOptions | None = None) -> None:
        """Initialize MetadataEnricher.

        Args:
            default_options: Optional default enrichment options.
        """
        self.default_options = default_options or EnrichmentOptions()

    def enrich_chunk(
        self,
        chunk: Document,
        options: EnrichmentOptions | None = None,
        ingested_at: str | None = None,
    ) -> Document:
        """Enrich a single document chunk without mutating the original input.

        Args:
            chunk: Input LangChain Document chunk.
            options: Optional enrichment options.
            ingested_at: Optional ISO 8601 UTC timestamp shared across batch.

        Returns:
            New LangChain Document with enriched metadata.

        Raises:
            MetadataEnrichmentError: If chunk content is empty or validation fails.
        """
        if not chunk.page_content or not chunk.page_content.strip():
            raise MetadataEnrichmentError("Cannot enrich empty or whitespace document chunk.")

        opts = options or self.default_options
        content = chunk.page_content

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ingestion_ts = ingested_at or now_utc
        enrichment_ts = now_utc

        # Clone original metadata to guarantee immutability
        enriched_meta: dict[str, Any] = dict(chunk.metadata) if chunk.metadata else {}

        # 1. Source normalization
        source_val = str(enriched_meta.get("source", ""))
        source_type = str(enriched_meta.get("source_type") or "document")

        filename = enriched_meta.get("filename")
        if filename:
            source_name = str(filename)
        elif source_val.startswith(("http://", "https://")):
            parsed = urlparse(source_val)
            source_name = f"{parsed.netloc}{parsed.path}".rstrip("/") or source_val
        elif source_val:
            source_name = Path(source_val).name or source_val
        else:
            source_name = "unknown_source"

        enriched_meta["source"] = source_val
        enriched_meta["source_name"] = source_name
        enriched_meta["source_type"] = source_type

        # 2. Categorization
        if opts.category_override:
            category = opts.category_override
        else:
            detect_src = str(filename or source_name or source_val)
            category = detect_category(content, source=detect_src, source_type=source_type)
        enriched_meta["category"] = category

        # 3. Access control & permissions
        enriched_meta["access_level"] = opts.access_level
        enriched_meta["allowed_roles"] = str(opts.allowed_roles)

        # 4. Timestamps
        enriched_meta["ingested_at"] = ingestion_ts
        enriched_meta["enriched_at"] = enrichment_ts

        # 5. Identity & Hashes
        doc_id = hashlib.sha256(source_val.encode("utf-8")).hexdigest()[:16]
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        page_val = (
            enriched_meta.get("page_number")
            or enriched_meta.get("page")
            or 0
        )
        chunk_idx = enriched_meta.get("chunk_index", 0)
        raw_key = f"{source_val}|{page_val}|{chunk_idx}|{content_hash}"
        vector_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:32]

        enriched_meta["doc_id"] = doc_id
        enriched_meta["content_hash"] = content_hash
        enriched_meta["vector_id"] = vector_id

        # 6. Statistics
        enriched_meta["char_count"] = len(content)
        enriched_meta["word_count"] = len(content.split())

        # 7. Tags & versioning
        existing_tags = str(enriched_meta.get("tags", "")).split(",") if enriched_meta.get("tags") else []
        all_tags = [t.strip().lower() for t in existing_tags if t.strip()]
        if opts.extra_tags:
            for t in opts.extra_tags:
                if t.strip() and t.strip().lower() not in all_tags:
                    all_tags.append(t.strip().lower())

        enriched_meta["tags"] = ",".join(all_tags)
        enriched_meta["enrichment_version"] = "1.0"

        # Sanitize for vector database compatibility
        sanitized = sanitize_metadata(enriched_meta)
        validate_enriched_metadata(sanitized)

        return Document(page_content=content, metadata=sanitized)

    def enrich_chunks(
        self,
        chunks: list[Document],
        options: EnrichmentOptions | None = None,
        per_source_options: dict[str, EnrichmentOptions] | None = None,
    ) -> list[Document]:
        """Enrich a batch of document chunks.

        All chunks processed within a single batch share an identical ingested_at timestamp.
        Chunks with empty or whitespace-only page content are skipped.

        Args:
            chunks: List of LangChain Document chunks.
            options: Fallback enrichment options for chunks without source overrides.
            per_source_options: Optional mapping of source name to dedicated EnrichmentOptions.

        Returns:
            List of new enriched Document chunks.
        """
        shared_ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        base_opts = options or self.default_options
        enriched_docs: list[Document] = []

        for chunk in chunks:
            # Skip empty or whitespace chunks
            if not chunk or not chunk.page_content or not chunk.page_content.strip():
                continue

            src = str(chunk.metadata.get("source", "")) if chunk.metadata else ""
            fname = str(chunk.metadata.get("filename", "")) if chunk.metadata else ""

            # Check per-source option overrides
            chunk_opts = base_opts
            if per_source_options:
                if src and src in per_source_options:
                    chunk_opts = per_source_options[src]
                elif fname and fname in per_source_options:
                    chunk_opts = per_source_options[fname]

            enriched = self.enrich_chunk(
                chunk=chunk,
                options=chunk_opts,
                ingested_at=shared_ingested_at,
            )
            enriched_docs.append(enriched)

        return enriched_docs
