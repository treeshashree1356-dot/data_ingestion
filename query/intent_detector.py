"""Extensible intent detection interfaces and rule-based classifier for user queries.

Allows pluggable intent detection (e.g. rule-based regex today, LLM-based tomorrow).
"""

from abc import ABC, abstractmethod
import logging
import re
from typing import Any

from query.query_schemas import IntentType, QueryIntent

logger = logging.getLogger(__name__)


class BaseIntentDetector(ABC):
    """Abstract base class for user query intent classification."""

    @abstractmethod
    def detect_intent(self, query: str, context: Any = None) -> QueryIntent:
        """Classify user query into an intent category.

        Args:
            query: Normalized user query string.
            context: Optional contextual parameters.

        Returns:
            QueryIntent containing intent type, confidence score, and descriptive metadata.
        """
        raise NotImplementedError


class RuleBasedIntentDetector(BaseIntentDetector):
    """Pattern- and keyword-based intent classifier.

    Evaluates user queries against pre-compiled regex patterns to identify
    governance checks, how-to requests, FAQs, or standard semantic searches.
    """

    # Intent regex patterns ordered by evaluation precedence
    DEFAULT_PATTERNS: list[tuple[IntentType, list[str], float, str]] = [
        (
            IntentType.POLICY_CHECK,
            [
                r"\b(policy|policies|compliance|compliant|terms|code of conduct|guidelines?)\b",
                r"\b(is it allowed|can (we|i|employees)|prohibited|mandatory|violation|retention period)\b",
                r"\b(rules? regarding|legal requirements?)\b",
            ],
            0.90,
            "Query matches governance, compliance, or policy keywords",
        ),
        (
            IntentType.HOW_TO,
            [
                r"\b(how (to|do i|can i|should we)|steps? to|guide to|tutorial)\b",
                r"\b(how (does|do) .* work|instructions? for|setup|configure|installation)\b",
                r"\b(walkthrough|getting started)\b",
            ],
            0.88,
            "Query seeks procedural steps, instructions, or tutorials",
        ),
        (
            IntentType.FAQ,
            [
                r"\b(faq|frequently asked|what (is|are)|why (does|is|are)|what does .* mean)\b",
                r"\b(troubleshoot|troubleshooting|error|issue|problem|fix|fails? to)\b",
                r"\b(q&a|common questions?)\b",
            ],
            0.85,
            "Query represents a common question, definition, or troubleshooting inquiry",
        ),
        (
            IntentType.INTERNAL_INQUIRY,
            [
                r"\b(internal|confidential|team notes?|meeting minutes?|memo|standup)\b",
                r"\b(all hands|sprint planning|retro|roadmap)\b",
            ],
            0.82,
            "Query targets internal operational or team documentation",
        ),
    ]

    def __init__(
        self,
        custom_patterns: list[tuple[IntentType, list[str], float, str]] | None = None,
    ) -> None:
        """Initialize RuleBasedIntentDetector with default or custom patterns."""
        self.patterns = custom_patterns if custom_patterns is not None else self.DEFAULT_PATTERNS

    def detect_intent(self, query: str, context: Any = None) -> QueryIntent:
        """Classify the user query using compiled regex patterns.

        Args:
            query: Sanitized user query text.
            context: Optional contextual information.

        Returns:
            QueryIntent instance.
        """
        normalized = query.lower().strip()

        for intent_type, regex_list, confidence, description in self.patterns:
            for pattern in regex_list:
                match = re.search(pattern, normalized)
                if match:
                    logger.debug(
                        "Intent '%s' matched pattern '%s' for query: '%s'",
                        intent_type.value,
                        pattern,
                        query,
                    )
                    return QueryIntent(
                        intent_type=intent_type,
                        confidence=confidence,
                        description=description,
                        metadata={"matched_pattern": pattern, "matched_text": match.group(0)},
                    )

        # Fallback default: General semantic search
        return QueryIntent(
            intent_type=IntentType.SEARCH,
            confidence=0.50,
            description="General semantic search inquiry",
            metadata={"fallback": True},
        )
