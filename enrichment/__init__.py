"""Metadata enrichment package for document chunks.

Exports core enricher, configuration dataclass, and categorization rules.
"""

from enrichment.category_rules import CATEGORY_RULES, detect_category
from enrichment.metadata_enricher import (
    ALLOWED_ACCESS_LEVELS,
    REQUIRED_ENRICHED_FIELDS,
    EnrichmentOptions,
    MetadataEnricher,
    MetadataEnrichmentError,
    sanitize_metadata,
    validate_enriched_metadata,
)

__all__ = [
    "CATEGORY_RULES",
    "detect_category",
    "ALLOWED_ACCESS_LEVELS",
    "REQUIRED_ENRICHED_FIELDS",
    "EnrichmentOptions",
    "MetadataEnricher",
    "MetadataEnrichmentError",
    "sanitize_metadata",
    "validate_enriched_metadata",
]
