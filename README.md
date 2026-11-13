# Data Ingestion Pipeline

An extensible, production-ready document ingestion, extraction, chunking, enrichment, embedding, and vector indexing pipeline built with **Python**, **LangChain**, **HuggingFace**, **ChromaDB**, and **Streamlit**.

---

## 📖 Project Description

The **Data Ingestion Pipeline** provides a unified platform to ingest diverse document formats (PDFs, DOCX, Plain Text/Markdown, Web pages, and Images via OCR), process and clean their textual content, segment them into configurable chunks, enrich them with governance metadata and identity hashes, generate high-dimensional semantic embeddings locally, and index them into a persistent ChromaDB vector store ready for downstream Retrieval-Augmented Generation (RAG) and LLM applications.

### Key Capabilities

- **Multi-Source Ingestion**: Ingest PDFs, Word documents, text/markdown files, live web URLs, and images (OCR).
- **Document Processing**: Unified document extraction, normalization, and page tracking.
- **Configurable Chunking**: Recursive character splitting with dynamic chunk size and overlap adjustments, plus bidirectional JSON export/import.
- **Metadata Enrichment & Governance (Stage 4)**:
  - Automatic rule-based category detection (`policy`, `faq`, `guide`, `internal`, `web_page`, `general`) with word-boundary pattern matching and strict filename-over-content precedence.
  - Multi-tier access control levels (`public`, `internal`, `restricted`) and role-based permissions (`allowed_roles`).
  - Deterministic SHA-256 identity hashes: `doc_id` (16 chars), `content_hash` (64 chars), and collision-resistant `vector_id` (32 chars).
  - UTC ISO 8601 timestamps (`ingested_at`, `enriched_at`) ending in `Z`.
  - Strict ChromaDB scalar type sanitization (`str`, `int`, `float`, `bool`).
- **Local Vector Embeddings (Stage 5)**:
  - Local semantic vector embeddings via `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) or `DeterministicFakeEmbedding` for fast, offline testing.
  - Chunk batching with progress tracking callbacks, vector dimension verification, and NaN checks.
  - Process-level instance caching with lazy initialization on demand.
- **Persistent Vector Indexing & ChromaDB (Stage 6)**:
  - Persistent vector storage using ChromaDB configured with cosine similarity distance (`hnsw:space = "cosine"`).
  - Idempotent upserts preventing duplicate embeddings on repeat runs.
  - Batch duplicate elimination by `content_hash` and optional source purging (`replace_existing_sources=True`).
  - Metadata-filtered similarity search combining multiple criteria with `$and`.
  - LangChain `Chroma` integration handoff object for retrieval teams.
- **Interactive Streamlit UI**: Intuitive 6-tab web dashboard for managing the entire RAG ingestion lifecycle with downstream state reset protection.

---

## 🛠️ Technology Stack

- **Core Language**: Python 3.10+ (Tested with Python 3.13)
- **UI Framework**: [Streamlit](https://streamlit.io/)
- **Orchestration & Splitting**: [LangChain](https://www.langchain.com/), `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-classic`
- **Embeddings & Vector Store**:
  - `langchain-huggingface>=0.1.0`
  - `sentence-transformers>=3.0.0`
  - `langchain-chroma>=0.1.4`
  - `chromadb>=0.5.0`
- **Parsers & Loaders**:
  - `pypdf` (PDF processing)
  - `docx2txt` (DOCX extraction)
  - `beautifulsoup4` & `requests` (Web scraping & HTML extraction)
  - `Pillow` & `pytesseract` (Image processing and OCR)
- **Configuration & Environment**: `python-dotenv`
- **Testing**: `pytest`

---

## 📁 Folder Structure

```text
data_ingestion_pipeline/
├── app.py                         # Main Streamlit application entrypoint (6 tabs)
├── requirements.txt               # Project dependencies and version pins
├── .env                           # Local environment variables (git-ignored)
├── .env.example                   # Template for environment configuration
├── .gitignore                     # Git ignore rules for venv, vector store, and caches
├── README.md                      # Comprehensive project documentation
├── config/                        # Configuration and environment management
│   ├── __init__.py
│   └── settings.py                # Centralized settings and directory verification
├── ingestion/                     # Document ingestion and extraction modules
│   ├── __init__.py
│   ├── document_loader.py         # Base loader interface and coordinator
│   ├── docx_loader.py             # DOCX document loader
│   ├── pdf_loader.py              # PDF loader
│   ├── web_loader.py              # Web/URL content loader
│   ├── text_loader.py             # Plain text, Markdown, and manual text loaders
│   ├── image_loader.py            # Image & OCR loader
│   └── document_processor.py      # Document validation and normalization
├── chunking/                      # Text splitting and chunking strategies
│   ├── __init__.py
│   └── text_splitter.py           # RecursiveCharacterTextSplitter and JSON serialization
├── enrichment/                    # Metadata enrichment & categorization (Stage 4)
│   ├── __init__.py
│   ├── category_rules.py          # Category keyword rules and detect_category()
│   └── metadata_enricher.py       # MetadataEnricher, sanitize_metadata, EnrichmentOptions
├── embedding/                     # Vector embedding generation (Stage 5)
│   ├── __init__.py
│   ├── embedding_factory.py       # get_embedding_model factory and caching
│   └── embedding_manager.py       # EmbeddingManager, batching, and EmbeddingResult
├── indexing/                      # Vector store indexing & ChromaDB (Stage 6)
│   ├── __init__.py
│   └── vector_store.py            # VectorStoreManager, Chroma client, and search
├── services/                      # High-level business logic & pipeline services
│   ├── __init__.py
│   ├── ingestion_service.py       # Extraction & chunking orchestration
│   └── indexing_service.py        # Enrichment, embedding, and indexing orchestration
├── ui/                            # Streamlit UI modular components
│   ├── __init__.py
│   ├── state.py                   # Downstream state invalidation helper
│   ├── source_selection.py        # 1. Source selection & upload UI
│   ├── extraction_view.py         # 2. Extracted text & preview UI
│   ├── chunking_view.py           # 3. Chunk preview & export UI
│   ├── metadata_view.py           # 4. Metadata enrichment UI
│   ├── embedding_view.py          # 5. Embedding generation UI
│   └── indexing_view.py           # 6. Vector store indexing & test retrieval UI
├── utils/                         # Helper and file management utilities
│   ├── __init__.py
│   ├── file_utils.py              # Upload, unique pathing, and temp file utilities
│   └── validators.py              # Input validation for files, URLs, and text
├── data/                          # Local data storage directories
│   ├── uploads/                   # Uploaded raw files
│   ├── processed/                 # Processed outputs
│   └── vector_store/              # Persistent ChromaDB vector database storage
└── tests/                         # Automated test suite (156 tests)
    ├── __init__.py
    ├── conftest.py                # Test fixtures, temp vector store, fake embeddings
    ├── test_project_setup.py      # Base setup verification tests
    ├── test_phase8_setup.py       # Stages 4-6 architecture and setup tests
    ├── test_source_validation.py  # Input validation test suite
    ├── test_loaders.py            # Document extraction and loader tests
    ├── test_image_loader.py       # Image OCR extraction tests
    ├── test_scanned_pdf.py        # Scanned PDF fallback tests
    ├── test_chunking.py           # Document chunking and serialization tests
    ├── test_metadata_enrichment.py# Stage 4 metadata enrichment & classification tests
    ├── test_embedding.py          # Stage 5 embedding factory and batching tests
    ├── test_indexing.py           # Stage 6 vector store and ChromaDB tests
    ├── test_indexing_service.py   # Indexing service coordination tests
    ├── test_ui.py                 # Initial Streamlit UI tests
    ├── test_ui_indexing.py        # Tabs 4-6 UI integration tests
    ├── test_workflow.py           # Workflow and fault isolation tests
    ├── test_end_to_end_pipeline.py# End-to-end extraction and chunking tests
    └── test_full_rag_ingestion.py # Full stages 1-6 end-to-end integration tests
```

---

## ⚙️ Environment Configuration

Copy `.env.example` to `.env` to customize settings:

```dotenv
# Storage Directories
UPLOAD_DIR=data/uploads
PROCESSED_DIR=data/processed
VECTOR_STORE_DIR=data/vector_store

# Ingestion Limits
MAX_FILE_SIZE_MB=50

# Default Chunking Settings
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200

# Embedding Configuration
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32
NORMALIZE_EMBEDDINGS=True
FAKE_EMBEDDING_DIM=384

# Vector Store & Access Control
VECTOR_COLLECTION_NAME=canonix_knowledge_base
DEFAULT_ACCESS_LEVEL=internal
DEFAULT_ALLOWED_ROLES=all
RETRIEVAL_TOP_K=4

# Optional OCR / External Services
TESSERACT_CMD=
```

---

## 🔌 Using the Vector Store from the Retrieval Pipeline

Downstream retrieval pipelines and agentic search modules can directly connect to the persistent ChromaDB collection using the `VectorStoreManager` hand-off method:

```python
from indexing.vector_store import VectorStoreManager

# Connect to the persistent ChromaDB vector store
manager = VectorStoreManager()
vectorstore = manager.as_langchain_vectorstore()

# Query the vector store using LangChain's standard similarity search
query = "What is the policy for data retention and audit compliance?"
results = vectorstore.similarity_search(query, k=4)

for doc in results:
    print(f"[{doc.metadata.get('category')}] {doc.metadata.get('chunk_id')}")
    print(f"Source: {doc.metadata.get('source_name')} | Access Level: {doc.metadata.get('access_level')}")
    print(f"Content: {doc.page_content}\n")
```

> [!IMPORTANT]
> The retrieval team **must use the same embedding model** (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) as configured in the ingestion pipeline. Vector comparisons across disparate embedding dimensions or model architectures will produce invalid results.

---

## 🚀 Running the Streamlit App

With the virtual environment activated:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Access the web interface at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the full automated test suite (offline, deterministic, using `DeterministicFakeEmbedding`):

```powershell
.\.venv\Scripts\Activate.ps1
pytest -v
```

### Running Real Embedding Model Verification

To verify that the live HuggingFace model (`sentence-transformers/all-MiniLM-L6-v2`) downloads and executes properly:

- **Windows PowerShell**:
  ```powershell
  $env:RUN_MODEL_TESTS="1"; pytest -v tests/test_embedding.py
  ```
- **macOS / Linux**:
  ```bash
  RUN_MODEL_TESTS=1 pytest -v tests/test_embedding.py
  ```

---

## 🔒 Security & Best Practices

- **Never Commit Secrets**: The `.env` file, `.venv/`, and `data/vector_store/` files are excluded by `.gitignore`.
- **Offline Test Isolation**: Pytest fixtures automatically configure `EMBEDDING_PROVIDER=fake` and isolated temporary directories so tests never trigger network downloads or touch production vector databases.
- **Downstream Invalidation**: The Streamlit interface automatically resets downstream artifacts whenever upstream sources or chunking parameters change, preventing stale data contamination in the vector store.
