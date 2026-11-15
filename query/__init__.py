"""User Query Processing and Intent Classification package.

Exposes QueryProcessor, schemas, intent detectors, and query expanders.
"""

from query.intent_detector import BaseIntentDetector, RuleBasedIntentDetector
from query.query_expander import BaseQueryExpander, RuleBasedQueryExpander
from query.query_processor import QueryProcessor
from query.query_schemas import (
    IntentType,
    ProcessedQuery,
    QueryContext,
    QueryIntent,
    QueryProcessingError,
    QueryResponse,
    QueryValidationError,
)

__all__ = [
    "BaseIntentDetector",
    "BaseQueryExpander",
    "IntentType",
    "ProcessedQuery",
    "QueryContext",
    "QueryIntent",
    "QueryProcessingError",
    "QueryProcessor",
    "QueryResponse",
    "QueryValidationError",
    "RuleBasedIntentDetector",
    "RuleBasedQueryExpander",
]
