"""Pytest configuration and global fixtures for the RAG pipeline test suite.

Configures test isolation before any project module import by redirecting
the vector store to a temporary directory and selecting the fake embedding provider.
"""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

# Pre-import environment overrides to ensure tests run offline without downloading models
_SESSION_TEMP_DIR = tempfile.mkdtemp(prefix="canonix_test_vstore_")
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["VECTOR_STORE_DIR"] = _SESSION_TEMP_DIR
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Graceful fallback for environments where optional dependencies are not installed
for mod_name in ("pypdfium2", "docx2txt", "requests", "bs4"):
    try:
        __import__(mod_name)
    except ImportError:
        sys.modules[mod_name] = types.ModuleType(mod_name)

try:
    import langchain_core
    from langchain_core.documents import Document
    from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
except ImportError:
    lc = types.ModuleType("langchain_core")
    lcdoc = types.ModuleType("langchain_core.documents")

    class Document:  # type: ignore
        def __init__(self, page_content: str = "", metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

        def __repr__(self) -> str:
            return f"Document(page_content={self.page_content!r}, metadata={self.metadata!r})"

    lcdoc.Document = Document
    lc.documents = lcdoc

    lcemb = types.ModuleType("langchain_core.embeddings")

    class Embeddings:  # type: ignore
        pass

    class DeterministicFakeEmbedding(Embeddings):  # type: ignore
        def __init__(self, size: int = 384) -> None:
            self.size = size

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * self.size for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [0.1] * self.size

    lcemb.Embeddings = Embeddings
    lcemb.DeterministicFakeEmbedding = DeterministicFakeEmbedding
    lc.embeddings = lcemb

    sys.modules["langchain_core"] = lc
    sys.modules["langchain_core.documents"] = lcdoc
    sys.modules["langchain_core.embeddings"] = lcemb

# Graceful fallback for langchain_text_splitters
try:
    import langchain_text_splitters
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    lcts = types.ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:  # type: ignore
        def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs: object) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str) -> list[str]:
            if not text:
                return []
            chunks = []
            start = 0
            stride = max(1, self.chunk_size - self.chunk_overlap)
            while start < len(text):
                chunks.append(text[start : start + self.chunk_size])
                start += stride
            return chunks

    lcts.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain_text_splitters"] = lcts

# Graceful fallback for langchain_community
try:
    import langchain_community
except ImportError:
    lc_comm = types.ModuleType("langchain_community")

    class DummyLoader:  # type: ignore
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def load(self) -> list[object]:
            return []

    class LoaderModule(types.ModuleType):
        def __getattr__(self, name: str) -> type:
            return DummyLoader

    lc_doc_loaders = LoaderModule("langchain_community.document_loaders")
    lc_comm.document_loaders = lc_doc_loaders
    sys.modules["langchain_community"] = lc_comm
    sys.modules["langchain_community.document_loaders"] = lc_doc_loaders

# Graceful fallback for chromadb and langchain_chroma
try:
    import chromadb
    from chromadb.api.client import SharedSystemClient
except ImportError:
    chroma_mod = types.ModuleType("chromadb")
    chroma_config = types.ModuleType("chromadb.config")

    class DummySettings:  # type: ignore
        def __init__(self, **kwargs: object) -> None:
            pass

    chroma_config.Settings = DummySettings
    chroma_mod.config = chroma_config

    class DummyClient:  # type: ignore
        pass

    chroma_mod.PersistentClient = DummyClient
    sys.modules["chromadb"] = chroma_mod
    sys.modules["chromadb.config"] = chroma_config

    class SharedSystemClient:  # type: ignore
        @staticmethod
        def clear_system_cache() -> None:
            pass

    chroma_api_client = types.ModuleType("chromadb.api.client")
    chroma_api_client.SharedSystemClient = SharedSystemClient
    sys.modules["chromadb.api.client"] = chroma_api_client
    chromadb = chroma_mod

try:
    import langchain_chroma
    from langchain_chroma import Chroma
except ImportError:
    lc_chroma = types.ModuleType("langchain_chroma")

    class Chroma:  # type: ignore
        def __init__(self, **kwargs: object) -> None:
            pass

    lc_chroma.Chroma = Chroma
    sys.modules["langchain_chroma"] = lc_chroma

import pytest

try:
    from chunking.text_splitter import ChunkingManager
    from indexing.vector_store import VectorStoreManager
except ImportError:
    ChunkingManager = None
    VectorStoreManager = None


@pytest.fixture(scope="session", autouse=True)
def clean_session_temp_dir():
    """Ensure session temporary storage is cleaned up after entire test suite completes."""
    yield
    if SharedSystemClient is not None:
        try:
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
    shutil.rmtree(_SESSION_TEMP_DIR, ignore_errors=True)


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """Fixture providing a deterministic fake embedding model with 384 dimensions."""
    return DeterministicFakeEmbedding(size=384)


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Fixture providing 3-5 Document chunks generated by ChunkingManager from sample texts."""
    if ChunkingManager is None:
        pytest.skip("ChunkingManager is unavailable")

    policy_text = (
        "Canonix Security Policy: All employee passwords must be rotated every 90 days. "
        "Compliance with internal data handling standards is mandatory for all team members."
    )
    faq_text = (
        "Frequently Asked Questions (FAQ): How do I reset my company VPN credentials? "
        "Submit a request to the IT helpdesk portal or contact internal support."
    )
    guide_text = (
        "Deployment Guide: Follow these steps to deploy microservices to the cluster. "
        "Step 1: Check environment configuration. Step 2: Run deployment pipeline."
    )

    doc_policy = Document(page_content=policy_text, metadata={"source": "security_policy.pdf", "filename": "security_policy.pdf"})
    doc_faq = Document(page_content=faq_text, metadata={"source": "system_faq.md", "filename": "system_faq.md"})
    doc_guide = Document(page_content=guide_text, metadata={"source": "deploy_guide.docx", "filename": "deploy_guide.docx"})

    chunker = ChunkingManager(chunk_size=120, chunk_overlap=20)
    chunks = chunker.split_documents([doc_policy, doc_faq, doc_guide])
    assert 3 <= len(chunks) <= 10
    return chunks


@pytest.fixture
def vector_store(tmp_path: Path, fake_embeddings: DeterministicFakeEmbedding):
    """Fixture providing an isolated VectorStoreManager instance on tmp_path."""
    if VectorStoreManager is None or fake_embeddings is None:
        pytest.skip("VectorStoreManager or fake_embeddings is unavailable")

    manager = VectorStoreManager(
        persist_directory=tmp_path / "vstore",
        collection_name="test_collection",
        embeddings=fake_embeddings,
    )
    yield manager
    manager.close()
    if SharedSystemClient is not None:
        try:
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
    shutil.rmtree(tmp_path / "vstore", ignore_errors=True)
