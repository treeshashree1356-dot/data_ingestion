"""Core user query processor coordinating validation, intent detection, category inference, and RBAC filter generation.

Adheres strictly to the principle of reusing existing components (e.g. CATEGORY_RULES,
metadata constants, settings) without duplicating logic.
"""

import logging
import re
from typing import Any

from config.settings import settings
from enrichment.category_rules import CATEGORY_RULES, _count_keyword_matches
from enrichment.metadata_enricher import ALLOWED_ACCESS_LEVELS
from query.intent_detector import BaseIntentDetector, RuleBasedIntentDetector
from query.query_expander import BaseQueryExpander, RuleBasedQueryExpander
from query.query_schemas import (
    ProcessedQuery,
    QueryContext,
    QueryIntent,
    QueryProcessingError,
    QueryValidationError,
)

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Processes, cleans, validates, classifies, and prepares user queries for retrieval."""

    MIN_QUERY_LENGTH: int = 2
    MAX_QUERY_LENGTH: int = 1000

    def __init__(
        self,
        intent_detector: BaseIntentDetector | None = None,
        query_expander: BaseQueryExpander | None = None,
    ) -> None:
        """Initialize QueryProcessor with extensible dependencies.

        Args:
            intent_detector: Optional custom intent classifier (defaults to RuleBasedIntentDetector).
            query_expander: Optional custom query expander (defaults to RuleBasedQueryExpander).
        """
        self.intent_detector = intent_detector or RuleBasedIntentDetector()
        self.query_expander = query_expander or RuleBasedQueryExpander()

    def clean_query(self, query: Any) -> str:
        """Normalize whitespace, strip control characters, and validate query boundaries.

        Args:
            query: Raw query input.

        Returns:
            Sanitized, clean query string.

        Raises:
            QueryValidationError: If query is not a string, empty, or outside length limits.
        """
        if query is None or not isinstance(query, str):
            raise QueryValidationError(
                f"Query must be a non-empty string. Received type: {type(query).__name__}"
            )

        # Remove null bytes and non-printable control characters (except common punctuation)
        cleaned = query.replace("\x00", "")

        # Normalize carriage returns and line feeds to single spaces
        cleaned = cleaned.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

        # Collapse repeated whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if len(cleaned) < self.MIN_QUERY_LENGTH:
            raise QueryValidationError(
                f"Query is too short. Minimum length is {self.MIN_QUERY_LENGTH} characters."
            )

        if len(cleaned) > self.MAX_QUERY_LENGTH:
            raise QueryValidationError(
                f"Query exceeds maximum allowed length of {self.MAX_QUERY_LENGTH} characters."
            )

        return cleaned

    def detect_category_from_query(self, cleaned_query: str) -> tuple[str | None, dict[str, int]]:
        """Infer document category from query text by reusing existing CATEGORY_RULES.

        Args:
            cleaned_query: Sanitized user query text.

        Returns:
            Tuple of (detected_category or None, keyword_hit_counts_dict).
        """
        counts = _count_keyword_matches(cleaned_query)

        # Filter out categories with 0 hits or non-domain categories
        candidates = {cat: hit for cat, hit in counts.items() if hit > 0 and cat not in ("general", "web_page")}

        if not candidates:
            return None, counts

        # Select highest hit count, breaking ties by order of appearance in CATEGORY_RULES
        best_category = max(
            candidates.keys(),
            key=lambda cat: (candidates[cat], -list(CATEGORY_RULES.keys()).index(cat)),
        )

        return best_category, counts

    def build_rbac_filters(
        self,
        context: QueryContext,
        detected_category: str | None,
    ) -> dict[str, Any]:
        """Synthesize a ChromaDB-compliant metadata filter dictionary.

        Combines category, access_level, and role-based permissions into a unified
        filter adhering to the metadata schemas generated during ingestion enrichment.

        Args:
            context: User query context containing role and overrides.
            detected_category: Inferred or user-specified category.

        Returns:
            Dictionary formatted for ChromaDB metadata filtering.

        Raises:
            QueryValidationError: If an invalid access level is requested.
        """
        filters: dict[str, Any] = {}

        # 1. Category filter
        if detected_category:
            filters["category"] = detected_category

        # 2. Access Level filter
        if context.access_level:
            level = context.access_level.strip().lower()
            if level not in ALLOWED_ACCESS_LEVELS:
                raise QueryValidationError(
                    f"Invalid access_level '{level}'. Allowed levels: {ALLOWED_ACCESS_LEVELS}"
                )
            filters["access_level"] = level
        else:
            # Derive access level filter from user role
            role = context.user_role.strip().lower()
            if role in ("guest", "public", "anonymous"):
                filters["access_level"] = "public"
            elif role in ("internal", "employee", "staff"):
                filters["access_level"] = {"$in": ["public", "internal"]}
            # 'admin' and 'all' roles have unrestricted access by default

        # 3. Incorporate extra custom filters if provided
        if context.extra_filters:
            for k, v in context.extra_filters.items():
                if v is not None:
                    filters[k] = v

        return filters

    def process(
        self,
        query: str,
        context: QueryContext | None = None,
    ) -> ProcessedQuery:
        """Execute end-to-end query preprocessing, classification, and filter synthesis.

        Args:
            query: Raw user query string.
            context: Optional QueryContext configuring roles, overrides, and flags.

        Returns:
            ProcessedQuery containing normalized text, intent, category, and filters.

        Raises:
            QueryValidationError: If query text or parameters are invalid.
            QueryProcessingError: If processing encounters an unexpected runtime error.
        """
        ctx = context or QueryContext()

        try:
            # 1. Preprocessing and validation
            cleaned = self.clean_query(query)

            # 2. Intent detection
            intent: QueryIntent = self.intent_detector.detect_intent(cleaned, context=ctx)

            # 3. Category detection (explicit override takes precedence, else auto-detection)
            diagnostics: dict[str, Any] = {"raw_query": query}
            if ctx.target_category:
                cat_override = ctx.target_category.strip().lower()
                if cat_override in CATEGORY_RULES:
                    detected_category = cat_override
                else:
                    detected_category = cat_override
                diagnostics["category_source"] = "manual_override"
            elif ctx.auto_detect_category:
                detected_category, counts = self.detect_category_from_query(cleaned)
                diagnostics["category_counts"] = counts
                diagnostics["category_source"] = "auto_detected" if detected_category else "none"
            else:
                detected_category = None
                diagnostics["category_source"] = "disabled"

            # 4. RBAC & Metadata filter synthesis
            filters = self.build_rbac_filters(ctx, detected_category)

            # 5. Optional query expansion
            expanded_terms: list[str] = []
            if ctx.enable_expansion:
                expanded_terms = self.query_expander.expand(cleaned, context=ctx)
                diagnostics["expansion_enabled"] = True
            else:
                diagnostics["expansion_enabled"] = False

            diagnostics["intent_type"] = intent.intent_type.value
            diagnostics["intent_confidence"] = intent.confidence

            return ProcessedQuery(
                original_query=query,
                cleaned_query=cleaned,
                intent=intent,
                detected_category=detected_category,
                filters=filters,
                expanded_terms=expanded_terms,
                diagnostics=diagnostics,
            )
        except QueryValidationError:
            raise
        except Exception as exc:
            logger.error("Failed to process query '%s': %s", query, exc)
            raise QueryProcessingError(f"Query processing failed: {exc}") from exc
