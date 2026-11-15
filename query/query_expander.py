"""Extensible query expansion interfaces and domain acronym/synonym expander.

Enables expanding abbreviations, technical jargon, or acronyms to improve
vector similarity retrieval recall.
"""

from abc import ABC, abstractmethod
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class BaseQueryExpander(ABC):
    """Abstract base class for query expansion strategies."""

    @abstractmethod
    def expand(self, query: str, context: Any = None) -> list[str]:
        """Generate expanded variations or synonym terms for a user query.

        Args:
            query: Sanitized user query text.
            context: Optional contextual information.

        Returns:
            List of unique expansion terms or augmented query variants.
        """
        raise NotImplementedError


class RuleBasedQueryExpander(BaseQueryExpander):
    """Rule-based query expander using a predefined technical acronym and synonym map."""

    DEFAULT_SYNONYMS: dict[str, list[str]] = {
        "auth": ["authentication", "authorization"],
        "k8s": ["kubernetes"],
        "rpc": ["remote procedure call", "gRPC"],
        "api": ["application programming interface", "endpoints"],
        "db": ["database", "datastore"],
        "tls": ["transport layer security", "ssl"],
        "ocr": ["optical character recognition", "text recognition"],
        "rbac": ["role based access control", "permissions"],
        "rag": ["retrieval augmented generation"],
        "sso": ["single sign on"],
        "mfa": ["multi factor authentication", "2fa"],
        "sdk": ["software development kit"],
        "ci": ["continuous integration"],
        "cd": ["continuous deployment"],
    }

    def __init__(
        self,
        custom_synonyms: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize RuleBasedQueryExpander with default or custom synonym map."""
        self.synonyms = dict(self.DEFAULT_SYNONYMS)
        if custom_synonyms:
            self.synonyms.update(custom_synonyms)

    def expand(self, query: str, context: Any = None) -> list[str]:
        """Identify matching acronyms or jargon in the query and return expanded terms.

        Args:
            query: Sanitized user query text.
            context: Optional contextual information.

        Returns:
            List of detected synonym expansions.
        """
        normalized = query.lower()
        expanded_terms: list[str] = []

        for acronym, expansions in self.synonyms.items():
            pattern = rf"\b{re.escape(acronym)}\b"
            if re.search(pattern, normalized):
                for term in expansions:
                    if term not in expanded_terms and term.lower() not in normalized:
                        expanded_terms.append(term)

        logger.debug("Query '%s' expanded with terms: %s", query, expanded_terms)
        return expanded_terms
