"""Data schemas and types for user query processing and retrieval.

Defines context, intent, processed query outcomes, and response structures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.documents import Document


class IntentType(str, Enum):
    """Supported user query intent types."""

    SEARCH = "search"                  # General factual or semantic question
    POLICY_CHECK = "policy_check"      # Compliance, code of conduct, or rules query
    HOW_TO = "how_to"                  # Procedural, tutorial, or setup instructions
    FAQ = "faq"                        # Frequently asked questions or troubleshooting
    INTERNAL_INQUIRY = "internal"      # Team meetings, memos, or internal notes
    UNKNOWN = "unknown"                # Ambiguous or unrecognizable intent


class QueryValidationError(ValueError):
    """Raised when a user query fails sanitization or structural validation."""
    pass


class QueryProcessingError(RuntimeError):
    """Raised when an unexpected error occurs during query processing."""
    pass


@dataclass
class QueryIntent:
    """Classified intent of a user query with associated confidence and metadata."""

    intent_type: IntentType
    confidence: float
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryContext:
    """User context and configuration parameters governing query execution.

    Attributes:
        user_role: Caller's security role (e.g. 'all', 'admin', 'engineering', 'guest').
        target_category: Optional manual category override (e.g. 'policy', 'guide').
        access_level: Optional explicit access level filter ('public', 'internal', 'restricted').
        top_k: Maximum number of search results to retrieve.
        auto_detect_category: Whether to infer document category from query keywords.
        enable_expansion: Whether to expand the query with domain synonyms/acronyms.
        extra_filters: Additional metadata filters to apply during retrieval.
    """

    user_role: str = "all"
    target_category: str | None = None
    access_level: str | None = None
    top_k: int = 4
    auto_detect_category: bool = True
    enable_expansion: bool = False
    extra_filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize query context fields."""
        if not isinstance(self.user_role, str):
            self.user_role = str(self.user_role)
        self.user_role = self.user_role.strip().lower() or "all"

        if self.target_category is not None:
            self.target_category = str(self.target_category).strip().lower()
            if self.target_category in ("all", "", "none"):
                self.target_category = None

        if self.access_level is not None:
            self.access_level = str(self.access_level).strip().lower()
            if self.access_level in ("all", "", "none"):
                self.access_level = None

        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k <= 0:
            self.top_k = 4


@dataclass
class ProcessedQuery:
    """Outcome of query preprocessing, intent classification, and filter synthesis.

    Attributes:
        original_query: Raw input string as received from user.
        cleaned_query: Normalized and sanitized query string.
        intent: Classified intent and confidence score.
        detected_category: Category inferred from keywords or set by override.
        filters: ChromaDB-compatible metadata query filter dictionary.
        expanded_terms: List of synonym / acronym expansions generated.
        diagnostics: Additional debug details (e.g., keyword hit counts, matched rules).
    """

    original_query: str
    cleaned_query: str
    intent: QueryIntent
    detected_category: str | None
    filters: dict[str, Any] = field(default_factory=dict)
    expanded_terms: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResponse:
    """End-to-end outcome of a processed query execution and retrieval.

    Attributes:
        results: List of (Document, score) tuples returned from vector store.
        processed_query: Full diagnostic details of how the query was processed.
        total_found: Number of documents retrieved.
        duration_seconds: Elapsed wall-clock time for the complete query pipeline.
    """

    results: list[tuple[Document, float]]
    processed_query: ProcessedQuery
    total_found: int
    duration_seconds: float
