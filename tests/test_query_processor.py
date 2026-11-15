"""Unit tests for QueryProcessor, intent detection, category inference, and RBAC filter synthesis."""

import pytest

from enrichment.category_rules import CATEGORY_RULES
from query.intent_detector import BaseIntentDetector, RuleBasedIntentDetector
from query.query_expander import BaseQueryExpander, RuleBasedQueryExpander
from query.query_processor import QueryProcessor
from query.query_schemas import (
    IntentType,
    ProcessedQuery,
    QueryContext,
    QueryIntent,
    QueryValidationError,
)


@pytest.fixture
def processor() -> QueryProcessor:
    """Fixture providing a default QueryProcessor instance."""
    return QueryProcessor()


# ==============================================================================
# 1. Preprocessing and Validation Tests
# ==============================================================================

def test_query_cleaning_normalizes_whitespace(processor: QueryProcessor) -> None:
    """Test that multiple spaces, tabs, and newlines are collapsed into single spaces."""
    raw = "   What   is \t the  \n\n  retention    policy?   "
    cleaned = processor.clean_query(raw)
    assert cleaned == "What is the retention policy?"


def test_query_cleaning_removes_null_bytes(processor: QueryProcessor) -> None:
    """Test that null bytes are stripped during preprocessing."""
    raw = "What is the\x00 policy\x00 on data security?"
    cleaned = processor.clean_query(raw)
    assert cleaned == "What is the policy on data security?"


def test_query_validation_rejects_empty_and_whitespace(processor: QueryProcessor) -> None:
    """Test that empty or whitespace-only queries raise QueryValidationError."""
    with pytest.raises(QueryValidationError, match="too short"):
        processor.clean_query("")

    with pytest.raises(QueryValidationError, match="too short"):
        processor.clean_query("     ")


def test_query_validation_rejects_short_queries(processor: QueryProcessor) -> None:
    """Test that queries shorter than minimum length raise QueryValidationError."""
    with pytest.raises(QueryValidationError, match="too short"):
        processor.clean_query("a")


def test_query_validation_rejects_non_string_inputs(processor: QueryProcessor) -> None:
    """Test that non-string inputs raise QueryValidationError."""
    with pytest.raises(QueryValidationError, match="must be a non-empty string"):
        processor.clean_query(12345)

    with pytest.raises(QueryValidationError, match="must be a non-empty string"):
        processor.clean_query(None)


def test_query_validation_rejects_excessively_long_queries(processor: QueryProcessor) -> None:
    """Test that queries exceeding MAX_QUERY_LENGTH raise QueryValidationError."""
    huge_query = "word " * 300  # 1500 chars
    with pytest.raises(QueryValidationError, match="exceeds maximum allowed length"):
        processor.clean_query(huge_query)


# ==============================================================================
# 2. Intent Detection Tests
# ==============================================================================

def test_intent_detection_policy_check(processor: QueryProcessor) -> None:
    """Test classification of governance and policy compliance questions."""
    queries = [
        "What is our data retention policy?",
        "Is password sharing compliant with internal guidelines?",
        "What are the mandatory rules regarding remote work?",
    ]
    for q in queries:
        pq = processor.process(q)
        assert pq.intent.intent_type == IntentType.POLICY_CHECK
        assert pq.intent.confidence >= 0.85


def test_intent_detection_how_to(processor: QueryProcessor) -> None:
    """Test classification of procedural how-to questions."""
    queries = [
        "How do I configure mutual TLS for microservices?",
        "Steps to setup the local vector database",
        "Tutorial on deploying the data ingestion pipeline",
    ]
    for q in queries:
        pq = processor.process(q)
        assert pq.intent.intent_type == IntentType.HOW_TO
        assert pq.intent.confidence >= 0.85


def test_intent_detection_faq(processor: QueryProcessor) -> None:
    """Test classification of FAQs and troubleshooting queries."""
    queries = [
        "Frequently asked questions about user onboarding",
        "Why does ChromaDB throw an indexing error?",
        "What does the vector_id hash mean?",
    ]
    for q in queries:
        pq = processor.process(q)
        assert pq.intent.intent_type == IntentType.FAQ
        assert pq.intent.confidence >= 0.80


def test_intent_detection_internal_inquiry(processor: QueryProcessor) -> None:
    """Test classification of internal memos and team documentation queries."""
    queries = [
        "Internal team notes from the quarterly all hands",
        "Confidential sprint planning roadmap",
    ]
    for q in queries:
        pq = processor.process(q)
        assert pq.intent.intent_type == IntentType.INTERNAL_INQUIRY


def test_intent_detection_general_search_fallback(processor: QueryProcessor) -> None:
    """Test that general semantic questions fall back to SEARCH intent."""
    q = "Quantum mechanics wave particle duality"
    pq = processor.process(q)
    assert pq.intent.intent_type == IntentType.SEARCH
    assert pq.intent.confidence == 0.50


def test_custom_intent_detector_extensibility() -> None:
    """Test that a custom BaseIntentDetector implementation can be seamlessly injected."""
    class MockLLMIntentDetector(BaseIntentDetector):
        def detect_intent(self, query: str, context: object = None) -> QueryIntent:
            return QueryIntent(
                intent_type=IntentType.POLICY_CHECK,
                confidence=0.99,
                description="LLM detected high-priority compliance audit",
                metadata={"model": "mock-llm-v1"},
            )

    custom_proc = QueryProcessor(intent_detector=MockLLMIntentDetector())
    pq = custom_proc.process("What are the system specifications?")
    assert pq.intent.intent_type == IntentType.POLICY_CHECK
    assert pq.intent.confidence == 0.99
    assert pq.intent.metadata["model"] == "mock-llm-v1"


# ==============================================================================
# 3. Automatic Category Detection Tests
# ==============================================================================

def test_auto_category_detection_uses_category_rules(processor: QueryProcessor) -> None:
    """Test that queries matching category rules automatically set category filter."""
    pq_policy = processor.process("What is the company compliance policy?")
    assert pq_policy.detected_category == "policy"
    assert pq_policy.filters.get("category") == "policy"

    pq_guide = processor.process("A step-by-step tutorial and handbook for engineers")
    assert pq_guide.detected_category == "guide"
    assert pq_guide.filters.get("category") == "guide"

    pq_faq = processor.process("FAQ and questions regarding user permissions")
    assert pq_faq.detected_category == "faq"
    assert pq_faq.filters.get("category") == "faq"


def test_auto_category_detection_fallback_none_when_no_match(processor: QueryProcessor) -> None:
    """Test that queries with no category keywords leave detected_category as None."""
    pq = processor.process("Semantic vector embeddings with ChromaDB")
    assert pq.detected_category is None
    assert "category" not in pq.filters


def test_manual_category_override_takes_precedence(processor: QueryProcessor) -> None:
    """Test that target_category in QueryContext overrides automatic detection."""
    ctx = QueryContext(target_category="faq")
    # Query text contains "policy", but explicit override is "faq"
    pq = processor.process("What is the policy for expense reports?", context=ctx)
    assert pq.detected_category == "faq"
    assert pq.filters.get("category") == "faq"


def test_category_detection_disabled_flag(processor: QueryProcessor) -> None:
    """Test that setting auto_detect_category=False disables automatic categorization."""
    ctx = QueryContext(auto_detect_category=False)
    pq = processor.process("What is the compliance policy?", context=ctx)
    assert pq.detected_category is None
    assert "category" not in pq.filters


# ==============================================================================
# 4. RBAC Role and Access Level Filtering Tests
# ==============================================================================

def test_rbac_filtering_for_guest_user(processor: QueryProcessor) -> None:
    """Test that guest / public role restricts retrieval to public access level."""
    ctx = QueryContext(user_role="guest")
    pq = processor.process("General product overview", context=ctx)
    assert pq.filters.get("access_level") == "public"


def test_rbac_filtering_for_internal_user(processor: QueryProcessor) -> None:
    """Test that internal role restricts retrieval to public and internal levels."""
    ctx = QueryContext(user_role="internal")
    pq = processor.process("General product overview", context=ctx)
    assert pq.filters.get("access_level") == {"$in": ["public", "internal"]}


def test_rbac_filtering_for_admin_user(processor: QueryProcessor) -> None:
    """Test that admin role imposes no access level restrictions by default."""
    ctx = QueryContext(user_role="admin")
    pq = processor.process("General product overview", context=ctx)
    assert "access_level" not in pq.filters


def test_explicit_access_level_override(processor: QueryProcessor) -> None:
    """Test explicit access_level in QueryContext sets the filter directly."""
    ctx = QueryContext(access_level="restricted")
    pq = processor.process("Executive compensation notes", context=ctx)
    assert pq.filters.get("access_level") == "restricted"


def test_invalid_access_level_raises_error(processor: QueryProcessor) -> None:
    """Test that an invalid access_level value raises QueryValidationError."""
    ctx = QueryContext(access_level="super_secret_invalid")
    with pytest.raises(QueryValidationError, match="Invalid access_level"):
        processor.process("Some query", context=ctx)


def test_combined_category_and_rbac_filters(processor: QueryProcessor) -> None:
    """Test that category and RBAC filters combine cleanly into ChromaDB filters."""
    ctx = QueryContext(user_role="guest", target_category="policy")
    pq = processor.process("Code of conduct rules", context=ctx)
    assert pq.filters == {
        "category": "policy",
        "access_level": "public",
    }


# ==============================================================================
# 5. Query Expansion Tests
# ==============================================================================

def test_query_expansion_disabled_by_default(processor: QueryProcessor) -> None:
    """Test that query expansion is disabled by default."""
    pq = processor.process("Configure auth with k8s and rpc")
    assert pq.expanded_terms == []


def test_query_expansion_when_enabled(processor: QueryProcessor) -> None:
    """Test expansion of technical acronyms when enabled."""
    ctx = QueryContext(enable_expansion=True)
    pq = processor.process("Configure auth with k8s and rpc", context=ctx)
    assert "authentication" in pq.expanded_terms or "authorization" in pq.expanded_terms
    assert "kubernetes" in pq.expanded_terms
    assert "remote procedure call" in pq.expanded_terms or "gRPC" in pq.expanded_terms


def test_custom_query_expander_extensibility() -> None:
    """Test that a custom BaseQueryExpander can be injected."""
    class CustomExpander(BaseQueryExpander):
        def expand(self, query: str, context: object = None) -> list[str]:
            return ["custom_synonym_1", "custom_synonym_2"]

    custom_proc = QueryProcessor(query_expander=CustomExpander())
    ctx = QueryContext(enable_expansion=True)
    pq = custom_proc.process("Test query", context=ctx)
    assert pq.expanded_terms == ["custom_synonym_1", "custom_synonym_2"]
