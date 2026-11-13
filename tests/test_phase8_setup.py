"""Phase 8 Setup and Verification Tests for Stages 4 to 6.

Validates that:
- New folders (enrichment, embedding, indexing) exist
- New Python modules exist
- New packages are in requirements.txt
- New settings are registered in Settings
- data/vector_store is properly tracked/ignored in .gitignore
"""

from pathlib import Path
import pytest

from config.settings import settings


@pytest.fixture
def project_root() -> Path:
    """Fixture providing root directory of the project."""
    return Path(__file__).resolve().parent.parent


def test_1_new_directories_exist(project_root: Path) -> None:
    """1. Test that all new package directories and storage folders exist."""
    required_dirs = [
        "enrichment",
        "embedding",
        "indexing",
        "data/vector_store",
    ]
    for d in required_dirs:
        target = project_root / d
        assert target.exists(), f"Required directory missing: {d}"
        assert target.is_dir(), f"Path is not a directory: {d}"


def test_2_new_python_modules_exist(project_root: Path) -> None:
    """2. Test that all new Python modules exist in the codebase."""
    required_modules = [
        "enrichment/__init__.py",
        "enrichment/category_rules.py",
        "enrichment/metadata_enricher.py",
        "embedding/__init__.py",
        "embedding/embedding_factory.py",
        "embedding/embedding_manager.py",
        "indexing/__init__.py",
        "indexing/vector_store.py",
        "services/indexing_service.py",
        "ui/state.py",
        "ui/metadata_view.py",
        "ui/embedding_view.py",
        "ui/indexing_view.py",
    ]
    for m in required_modules:
        target = project_root / m
        assert target.exists(), f"Required module missing: {m}"
        assert target.is_file(), f"Path is not a file: {m}"


def test_3_new_packages_in_requirements_txt(project_root: Path) -> None:
    """3. Test that requirements.txt includes all 4 required embedding and vector store packages."""
    req_file = project_root / "requirements.txt"
    assert req_file.exists()
    content = req_file.read_text(encoding="utf-8")

    expected_packages = [
        "langchain-huggingface",
        "sentence-transformers",
        "langchain-chroma",
        "chromadb",
    ]
    for pkg in expected_packages:
        assert pkg in content, f"Package {pkg} missing from requirements.txt"


def test_4_new_settings_registered() -> None:
    """4. Test that all new settings attributes are accessible on the settings singleton."""
    expected_settings = [
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL_NAME",
        "EMBEDDING_DEVICE",
        "EMBEDDING_BATCH_SIZE",
        "NORMALIZE_EMBEDDINGS",
        "FAKE_EMBEDDING_DIM",
        "VECTOR_STORE_DIR",
        "VECTOR_COLLECTION_NAME",
        "DEFAULT_ACCESS_LEVEL",
        "DEFAULT_ALLOWED_ROLES",
        "RETRIEVAL_TOP_K",
    ]
    for s in expected_settings:
        assert hasattr(settings, s), f"Settings is missing attribute {s}"


def test_5_vector_store_in_gitignore(project_root: Path) -> None:
    """5. Test that data/vector_store/* is ignored and .gitkeep is kept in .gitignore."""
    gitignore = project_root / ".gitignore"
    assert gitignore.exists()
    content = gitignore.read_text(encoding="utf-8")

    assert "data/vector_store/*" in content
    assert "!data/vector_store/.gitkeep" in content
