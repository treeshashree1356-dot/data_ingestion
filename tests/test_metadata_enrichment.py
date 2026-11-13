"""Unit tests for metadata enrichment and category detection.

Verifies schema compliance, deterministic identity hashes, access level enforcement,
ChromaDB sanitization, and immutability.
"""

from datetime import datetime
import hashlib
from pathlib import Path

from langchain_core.documents import Document
import pytest

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


def test_1_all_required_fields_present_after_enrichment() -> None:
    """1. Test that all required metadata fields are present after enrichment."""
    enricher = MetadataEnricher()
    doc = Document(
        page_content="Policy guidelines regarding corporate code of conduct.",
        metadata={"source": "corporate_policy.txt", "chunk_id": "c1"},
    )
    enriched = enricher.enrich_chunks([doc])[0]

    for req_field in REQUIRED_ENRICHED_FIELDS:
        assert req_field in enriched.metadata, f"Missing required field: {req_field}"
        assert enriched.metadata[req_field] is not None


def test_2_existing_metadata_preserved_unchanged() -> None:
    """2. Test that existing metadata keys are preserved exactly without alteration."""
    enricher = MetadataEnricher()
    original_meta = {
        "source": "manual.pdf",
        "chunk_id": "manual_d0_c3",
        "chunk_index": 3,
        "page_number": 12,
        "ocr_applied": True,
        "source_type": "pdf",
        "chunk_char_count": 55,
        "custom_tracker": 99,
    }
    doc = Document(page_content="Operational instructions for the manual.", metadata=original_meta)
    enriched = enricher.enrich_chunk(doc)

    for k, v in original_meta.items():
        assert enriched.metadata[k] == v, f"Metadata value changed for key {k}: expected {v}, got {enriched.metadata[k]}"


def test_3_input_chunks_are_not_mutated() -> None:
    """3. Test that enrichment produces new Document objects and does not mutate input."""
    enricher = MetadataEnricher()
    original_meta = {"source": "doc.txt", "chunk_id": "c1"}
    doc = Document(page_content="Sample document content.", metadata=original_meta)

    enriched = enricher.enrich_chunk(doc)
    assert enriched is not doc
    assert doc.metadata == {"source": "doc.txt", "chunk_id": "c1"}
    assert "vector_id" not in doc.metadata


def test_4_detect_category_precedence_and_mapping() -> None:
    """4. Test that detect_category identifies policy, faq, guide, internal, web_page, and general."""
    assert detect_category("Employee code of conduct and compliance terms.", source="doc.pdf") == "policy"
    assert detect_category("Frequently asked questions and q&a list.", source="doc.pdf") == "faq"
    assert detect_category("How to setup your account in 3 steps with this tutorial.", source="doc.pdf") == "guide"
    assert detect_category("Confidential internal team meeting memo.", source="doc.pdf") == "internal"
    assert detect_category("Random content with no keywords.", source="https://news.com/page", source_type="web") == "web_page"
    assert detect_category("Just some plain text without keywords.", source="random.txt", source_type="document") == "general"


def test_5_filename_keywords_win_over_content_keywords() -> None:
    """5. Test that keyword matches in filename take precedence over content keywords."""
    # Filename contains 'policy', while content contains multiple guide keywords ('tutorial', 'how to', 'steps')
    cat = detect_category(
        text="This is a tutorial on how to complete the steps.",
        source="security_policy.pdf",
        source_type="pdf",
    )
    assert cat == "policy"


def test_6_category_override_applied_and_unknown_raises_error() -> None:
    """6. Test that category_override is applied and invalid overrides raise MetadataEnrichmentError."""
    enricher = MetadataEnricher()
    doc = Document(page_content="Policy guidelines text.", metadata={"source": "test.txt", "chunk_id": "c1"})

    # Valid override
    opts = EnrichmentOptions(category_override="guide")
    enriched = enricher.enrich_chunk(doc, options=opts)
    assert enriched.metadata["category"] == "guide"

    # Invalid override
    with pytest.raises(MetadataEnrichmentError, match="Unknown category_override"):
        EnrichmentOptions(category_override="nonexistent_category")


def test_7_invalid_access_level_raises_error() -> None:
    """7. Test that invalid access level raises MetadataEnrichmentError."""
    for valid in ALLOWED_ACCESS_LEVELS:
        opts = EnrichmentOptions(access_level=valid)
        assert opts.access_level == valid

    with pytest.raises(MetadataEnrichmentError, match="Invalid access_level"):
        EnrichmentOptions(access_level="top_secret_forbidden")


def test_8_allowed_roles_standardized_to_lowercase_string() -> None:
    """8. Test that allowed_roles list or string becomes a lowercase comma-separated string."""
    opts1 = EnrichmentOptions(allowed_roles=["Admin", "HR ", "Security"])
    assert opts1.allowed_roles == "admin,hr,security"

    opts2 = EnrichmentOptions(allowed_roles=" Engineering, DevOps ")
    assert opts2.allowed_roles == "engineering,devops"


def test_9_timestamps_valid_iso_utc_and_shared_across_batch() -> None:
    """9. Test that timestamps are valid ISO 8601 UTC ending in Z and shared across batch."""
    enricher = MetadataEnricher()
    doc1 = Document(page_content="Chunk 1 content.", metadata={"source": "s1.txt", "chunk_id": "c1"})
    doc2 = Document(page_content="Chunk 2 content.", metadata={"source": "s2.txt", "chunk_id": "c2"})

    results = enricher.enrich_chunks([doc1, doc2])
    assert len(results) == 2

    for r in results:
        ing_ts = r.metadata["ingested_at"]
        enr_ts = r.metadata["enriched_at"]
        assert ing_ts.endswith("Z")
        assert enr_ts.endswith("Z")
        # Validate format
        datetime.strptime(ing_ts, "%Y-%m-%dT%H:%M:%SZ")
        datetime.strptime(enr_ts, "%Y-%m-%dT%H:%M:%SZ")

    # Shared ingested_at
    assert results[0].metadata["ingested_at"] == results[1].metadata["ingested_at"]


def test_10_deterministic_vector_id_and_content_hash() -> None:
    """10. Test that vector_id is deterministic across runs and content_hash matches SHA-256."""
    enricher = MetadataEnricher()
    content = "Unique content text for deterministic testing."
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    doc_a = Document(page_content=content, metadata={"source": "file.txt", "chunk_id": "c0", "chunk_index": 0})
    doc_b = Document(page_content=content, metadata={"source": "file.txt", "chunk_id": "c0", "chunk_index": 0})
    doc_diff = Document(page_content="Different text.", metadata={"source": "file.txt", "chunk_id": "c1", "chunk_index": 1})

    res_a = enricher.enrich_chunk(doc_a)
    res_b = enricher.enrich_chunk(doc_b)
    res_diff = enricher.enrich_chunk(doc_diff)

    assert res_a.metadata["content_hash"] == expected_hash
    assert res_a.metadata["vector_id"] == res_b.metadata["vector_id"]
    assert res_a.metadata["vector_id"] != res_diff.metadata["vector_id"]
    assert len(res_a.metadata["vector_id"]) == 32


def test_11_sanitize_metadata_types_and_handling() -> None:
    """11. Test that sanitize_metadata converts lists and Paths, drops None, and preserves bools/ints."""
    raw = {
        "str_val": "hello",
        "int_val": 42,
        "float_val": 3.14,
        "bool_val": True,
        "none_val": None,
        "list_val": ["alpha", "beta"],
        "path_val": Path("data/uploads/sample.pdf"),
    }
    sanitized = sanitize_metadata(raw)

    assert "none_val" not in sanitized
    assert sanitized["str_val"] == "hello"
    assert sanitized["int_val"] == 42
    assert sanitized["float_val"] == 3.14
    assert sanitized["bool_val"] is True
    assert isinstance(sanitized["bool_val"], bool)
    assert sanitized["list_val"] == "alpha,beta"
    assert isinstance(sanitized["path_val"], str)


def test_12_all_enriched_metadata_values_are_scalar() -> None:
    """12. Test that all enriched metadata values pass scalar validation without error."""
    enricher = MetadataEnricher()
    doc = Document(
        page_content="Validated scalar content.",
        metadata={"source": Path("test.txt"), "tags": ["tag1", "tag2"], "chunk_id": "c1"},
    )
    enriched = enricher.enrich_chunk(doc)
    validate_enriched_metadata(enriched.metadata)

    for k, v in enriched.metadata.items():
        assert isinstance(v, (str, int, float, bool)), f"Non-scalar key {k}: {type(v)}"


def test_13_empty_and_whitespace_chunks_are_skipped() -> None:
    """13. Test that empty and whitespace-only chunks are skipped in batch enrichment."""
    enricher = MetadataEnricher()
    chunks = [
        Document(page_content="Valid text.", metadata={"source": "s1.txt"}),
        Document(page_content="", metadata={"source": "s2.txt"}),
        Document(page_content="   \n\t  ", metadata={"source": "s3.txt"}),
        Document(page_content="Another valid text.", metadata={"source": "s4.txt"}),
    ]
    enriched = enricher.enrich_chunks(chunks)
    assert len(enriched) == 2
    assert enriched[0].page_content == "Valid text."
    assert enriched[1].page_content == "Another valid text."


def test_14_per_source_options_applied_selectively() -> None:
    """14. Test that per_source_options applies different access levels to distinct sources."""
    enricher = MetadataEnricher()
    chunks = [
        Document(page_content="Public info.", metadata={"source": "public_doc.txt", "chunk_id": "c1"}),
        Document(page_content="Confidential info.", metadata={"source": "confidential_doc.txt", "chunk_id": "c2"}),
    ]
    per_source = {
        "public_doc.txt": EnrichmentOptions(access_level="public"),
        "confidential_doc.txt": EnrichmentOptions(access_level="restricted", allowed_roles=["execs"]),
    }
    enriched = enricher.enrich_chunks(chunks, per_source_options=per_source)

    assert enriched[0].metadata["access_level"] == "public"
    assert enriched[1].metadata["access_level"] == "restricted"
    assert enriched[1].metadata["allowed_roles"] == "execs"
