"""Phase 5 Document Chunking, Splitter Configuration, and Serialization Tests."""

import json
from pathlib import Path
import pytest

from langchain_core.documents import Document

from chunking.text_splitter import (
    ChunkingConfigurationError,
    ChunkingManager,
    deserialize_chunks_from_json,
    export_chunks_to_json,
    serialize_chunks,
    validate_chunking_config,
)


# Sample paragraphs used to generate realistic document content
PARAGRAPHS = [
    (
        "Data ingestion is the foundational first step of modern data and RAG pipelines. "
        "It involves fetching raw assets from diverse locations, validating file types and sizes, "
        "and converting disparate sources into a common structured representation."
    ),
    (
        "Document parsing extracts clean textual content and structural attributes from proprietary formats "
        "such as Adobe PDF, Microsoft Word DOCX, plain text, Markdown, live websites, and image files. "
        "Care must be taken to preserve page numbers, headings, and metadata."
    ),
    (
        "Chunking breaks large narrative corpora into bite-sized segments optimized for downstream token limits. "
        "RecursiveCharacterTextSplitter uses an intelligent hierarchy of separators including paragraphs, "
        "newlines, sentences, and words to maintain contextual cohesion."
    ),
    (
        "Overlap between consecutive chunks guarantees that semantic meaning spanning boundary cutoffs "
        "is not lost. This overlap enables dense retrieval models and embeddings to locate complete propositions."
    ),
    (
        "Finally, serialization exports structured chunks alongside persistent chunk IDs and enriched metadata "
        "into portable JSON format, establishing readiness for vector indexing and semantic search."
    ),
]


def create_sample_text(repetitions: int = 4) -> str:
    """Helper generating a realistic document text of configurable length."""
    return "\n\n".join(PARAGRAPHS * repetitions)


# ===========================================================================
# 1-4. BASIC CHUNKING & CONFIGURATION
# ===========================================================================

def test_split_long_document() -> None:
    """1. Split a long document into multiple LangChain Document chunks."""
    long_text = create_sample_text(repetitions=5)  # ~3,500 characters
    doc = Document(page_content=long_text, metadata={"source": "manual.pdf", "author": "Engineer"})

    manager = ChunkingManager(chunk_size=1000, chunk_overlap=200)
    chunks = manager.split_documents([doc])

    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert len(chunk.page_content) <= 1000
        assert chunk.metadata["source"] == "manual.pdf"
        assert chunk.metadata["author"] == "Engineer"
        assert "chunk_id" in chunk.metadata


def test_process_multiple_documents() -> None:
    """2. Process multiple documents in a single batch split."""
    docs = [
        Document(page_content=create_sample_text(2), metadata={"source": "doc1.pdf", "page": 1}),
        Document(page_content=create_sample_text(3), metadata={"source": "doc2.docx", "author": "Alice"}),
        Document(page_content=create_sample_text(1), metadata={"source": "doc3.txt", "department": "AI"}),
    ]

    manager = ChunkingManager(chunk_size=800, chunk_overlap=150)
    chunks = manager.split_documents(docs)

    assert len(chunks) >= 3

    # Verify all 3 sources are represented in the chunks
    chunk_sources = {c.metadata["source"] for c in chunks}
    assert chunk_sources == {"doc1.pdf", "doc2.docx", "doc3.txt"}


def test_verify_chunk_size_configuration() -> None:
    """3. Verify chunk size configuration enforces character limits."""
    text = create_sample_text(3)
    doc = Document(page_content=text, metadata={"source": "sized_test.txt"})

    target_size = 400
    target_overlap = 50
    manager = ChunkingManager(chunk_size=target_size, chunk_overlap=target_overlap)
    chunks = manager.split_documents([doc])

    assert len(chunks) > 0
    for chunk in chunks:
        # Each chunk should strictly respect the maximum size constraint
        assert len(chunk.page_content) <= target_size


def test_verify_chunk_overlap_behavior() -> None:
    """4. Verify chunk overlap behavior preserves boundary text across consecutive chunks."""
    text = (
        "Alpha beta gamma delta epsilon. Zeta eta theta iota kappa. "
        "Lambda mu nu xi omicron. Pi rho sigma tau upsilon. Phi chi psi omega."
    )
    doc = Document(page_content=text, metadata={"source": "overlap_test.txt"})

    # Split with small size and explicit overlap
    manager = ChunkingManager(chunk_size=60, chunk_overlap=25)
    chunks = manager.split_documents([doc])

    assert len(chunks) >= 2
    # Check that consecutive chunks have shared text segment
    for i in range(len(chunks) - 1):
        chunk_a = chunks[i].page_content
        chunk_b = chunks[i + 1].page_content

        # Find words in chunk_a that also appear in chunk_b due to overlap
        words_a = set(chunk_a.split()[-3:])  # Tail of chunk A
        words_b = set(chunk_b.split()[:5])   # Head of chunk B
        shared = words_a.intersection(words_b)
        assert len(shared) > 0, f"Expected overlapping words between chunk {i} and {i+1}"


# ===========================================================================
# 5-8. VALIDATION & ERROR REJECTION
# ===========================================================================

def test_reject_zero_or_negative_chunk_size() -> None:
    """5. Reject zero or negative chunk size configurations."""
    for invalid_size in [0, -1, -500]:
        with pytest.raises((ValueError, ChunkingConfigurationError)) as exc_info:
            ChunkingManager(chunk_size=invalid_size, chunk_overlap=0)
        assert "greater than zero" in str(exc_info.value)


def test_reject_negative_overlap() -> None:
    """6. Reject negative overlap configurations."""
    for invalid_overlap in [-1, -50]:
        with pytest.raises((ValueError, ChunkingConfigurationError)) as exc_info:
            ChunkingManager(chunk_size=1000, chunk_overlap=invalid_overlap)
        assert "non-negative" in str(exc_info.value) or "greater than or equal to zero" in str(exc_info.value)


def test_reject_overlap_equal_to_chunk_size() -> None:
    """7. Reject overlap equal to chunk size."""
    with pytest.raises((ValueError, ChunkingConfigurationError)) as exc_info:
        ChunkingManager(chunk_size=500, chunk_overlap=500)
    assert "smaller than chunk size" in str(exc_info.value)


def test_reject_overlap_greater_than_chunk_size() -> None:
    """8. Reject overlap greater than chunk size."""
    with pytest.raises((ValueError, ChunkingConfigurationError)) as exc_info:
        ChunkingManager(chunk_size=400, chunk_overlap=600)
    assert "smaller than chunk size" in str(exc_info.value)


# ===========================================================================
# 9-13. METADATA & EDGE CASES
# ===========================================================================

def test_preserve_original_metadata() -> None:
    """9. Preserve original document metadata across all generated chunks."""
    doc = Document(
        page_content=create_sample_text(2),
        metadata={
            "source": "/data/contracts/agreement.pdf",
            "filename": "agreement.pdf",
            "source_type": "pdf",
            "page": 3,
            "page_number": 4,
            "confidential": True,
        },
    )

    manager = ChunkingManager(chunk_size=600, chunk_overlap=100)
    chunks = manager.split_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "/data/contracts/agreement.pdf"
        assert chunk.metadata["filename"] == "agreement.pdf"
        assert chunk.metadata["source_type"] == "pdf"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["page_number"] == 4
        assert chunk.metadata["confidential"] is True
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata


def test_generate_unique_chunk_ids() -> None:
    """10. Generate unique chunk IDs across all documents in a batch."""
    docs = [
        Document(page_content=create_sample_text(3), metadata={"source": "report_a.pdf"}),
        Document(page_content=create_sample_text(3), metadata={"source": "report_b.pdf"}),
    ]

    manager = ChunkingManager(chunk_size=500, chunk_overlap=100)
    chunks = manager.split_documents(docs)

    assert len(chunks) >= 4
    chunk_ids = [c.metadata["chunk_id"] for c in chunks]

    # Every chunk ID must be distinct
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs detected!"
    for cid in chunk_ids:
        assert len(cid) > 0


def test_handle_empty_documents() -> None:
    """11. Handle empty documents gracefully without emitting empty chunks."""
    empty_docs = [
        Document(page_content="", metadata={"source": "empty1.txt"}),
        Document(page_content="   \n\t  \r\n", metadata={"source": "empty2.txt"}),
    ]

    manager = ChunkingManager(chunk_size=1000, chunk_overlap=200)
    chunks = manager.split_documents(empty_docs)

    # Empty documents produce no chunks (avoids empty chunk pollution)
    assert chunks == []


def test_handle_very_short_documents() -> None:
    """12. Handle very short documents smaller than chunk size."""
    short_text = "Brief summary note."
    doc = Document(page_content=short_text, metadata={"source": "short.txt"})

    manager = ChunkingManager(chunk_size=1000, chunk_overlap=200)
    chunks = manager.split_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].page_content == short_text
    assert chunks[0].metadata["source"] == "short.txt"
    assert chunks[0].metadata["chunk_index"] == 0


def test_handle_unicode_text() -> None:
    """13. Handle diverse Unicode text and symbols without corruption."""
    unicode_text = (
        "Deep Learning Pipeline 🚀 🌟\n\n"
        "Japanese: 自然言語処理システムとテキスト分割\n\n"
        "German: Übergrößenträger müssen vorsichtig sein, um Schäden zu vermeiden.\n\n"
        "Arabic: معالجة اللغات الطبيعية وتقسيم النصوص إلى أجزاء\n\n"
        "Hindi: डेटा पाइपलाइन और टेक्स्ट चंकिंग प्रक्रिया"
    )
    doc = Document(page_content=unicode_text, metadata={"source": "unicode.txt"})

    manager = ChunkingManager(chunk_size=150, chunk_overlap=30)
    chunks = manager.split_documents([doc])

    assert len(chunks) > 1
    recombined = " ".join(c.page_content for c in chunks)
    assert "🚀" in recombined
    assert "自然言語処理" in recombined
    assert "Übergrößenträger" in recombined
    assert "معالجة اللغات" in recombined
    assert "डेटा पाइपलाइन" in recombined


# ===========================================================================
# 14-16. JSON EXPORT & SERIALIZATION
# ===========================================================================

def test_verify_json_serialization(tmp_path: Path) -> None:
    """14. Verify JSON serialization structure includes chunk_id, content, and metadata."""
    doc = Document(
        page_content="Data ingestion chunking unit test content.",
        metadata={"source": "export_test.pdf", "page": 1},
    )

    manager = ChunkingManager(chunk_size=500, chunk_overlap=50)
    chunks = manager.split_documents([doc])
    assert len(chunks) == 1

    out_file = tmp_path / "exported_chunks.json"
    json_output = export_chunks_to_json(chunks, file_path=out_file)

    # Verify JSON string is valid and file was written
    assert out_file.exists()
    parsed = json.loads(json_output)
    assert isinstance(parsed, list)
    assert len(parsed) == 1

    item = parsed[0]
    assert "chunk_id" in item
    assert item["content"] == "Data ingestion chunking unit test content."
    assert "metadata" in item
    assert item["metadata"]["source"] == "export_test.pdf"
    assert item["metadata"]["page"] == 1


def test_verify_json_output_can_be_parsed_back(tmp_path: Path) -> None:
    """15. Verify JSON output can be parsed back into LangChain Document objects."""
    docs = [
        Document(page_content=create_sample_text(2), metadata={"source": "roundtrip.pdf", "author": "Tester"}),
    ]
    manager = ChunkingManager(chunk_size=500, chunk_overlap=100)
    original_chunks = manager.split_documents(docs)

    json_path = tmp_path / "roundtrip.json"
    json_str = export_chunks_to_json(original_chunks, file_path=json_path)

    # Deserialize from JSON string
    reconstructed_from_str = deserialize_chunks_from_json(json_str)
    assert len(reconstructed_from_str) == len(original_chunks)

    # Deserialize from JSON file path
    reconstructed_from_file = deserialize_chunks_from_json(json_path)
    assert len(reconstructed_from_file) == len(original_chunks)

    # Verify exact fidelity of content and metadata
    for orig, recon in zip(original_chunks, reconstructed_from_file):
        assert isinstance(recon, Document)
        assert recon.page_content == orig.page_content
        assert recon.metadata["source"] == orig.metadata["source"]
        assert recon.metadata["chunk_id"] == orig.metadata["chunk_id"]
        assert recon.metadata["chunk_index"] == orig.metadata["chunk_index"]


def test_verify_chunk_source_association() -> None:
    """16. Verify each chunk is strictly associated with its originating source."""
    doc_a = Document(page_content="Content from file Alpha. " * 30, metadata={"source": "alpha.pdf"})
    doc_b = Document(page_content="Content from file Beta. " * 30, metadata={"source": "beta.docx"})

    manager = ChunkingManager(chunk_size=200, chunk_overlap=40)
    chunks = manager.split_documents([doc_a, doc_b])

    alpha_chunks = [c for c in chunks if c.metadata["source"] == "alpha.pdf"]
    beta_chunks = [c for c in chunks if c.metadata["source"] == "beta.docx"]

    assert len(alpha_chunks) > 0
    assert len(beta_chunks) > 0
    assert len(alpha_chunks) + len(beta_chunks) == len(chunks)

    for c in alpha_chunks:
        assert "Alpha" in c.page_content
        assert "Beta" not in c.page_content

    for c in beta_chunks:
        assert "Beta" in c.page_content
        assert "Alpha" not in c.page_content
