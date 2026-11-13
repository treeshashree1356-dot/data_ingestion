"""Phase 1 Setup and Architecture Verification Tests."""

from pathlib import Path
import pytest


@pytest.fixture
def project_root() -> Path:
    """Fixture providing the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def test_required_directories_exist(project_root: Path) -> None:
    """1. Test that all required project directories exist."""
    required_directories = [
        "config",
        "ingestion",
        "chunking",
        "services",
        "ui",
        "utils",
        "data",
        "data/uploads",
        "data/processed",
        "tests",
    ]

    for directory in required_directories:
        dir_path = project_root / directory
        assert dir_path.exists(), f"Required directory missing: {directory}"
        assert dir_path.is_dir(), f"Path is not a directory: {directory}"


def test_required_python_modules_exist(project_root: Path) -> None:
    """2. Test that all required Python modules and source files exist."""
    required_modules = [
        "app.py",
        "config/__init__.py",
        "config/settings.py",
        "ingestion/__init__.py",
        "ingestion/document_loader.py",
        "ingestion/pdf_loader.py",
        "ingestion/web_loader.py",
        "ingestion/text_loader.py",
        "ingestion/image_loader.py",
        "ingestion/document_processor.py",
        "chunking/__init__.py",
        "chunking/text_splitter.py",
        "services/__init__.py",
        "services/ingestion_service.py",
        "ui/__init__.py",
        "ui/source_selection.py",
        "ui/extraction_view.py",
        "ui/chunking_view.py",
        "utils/__init__.py",
        "utils/file_utils.py",
        "tests/__init__.py",
        "tests/test_project_setup.py",
    ]

    for module in required_modules:
        file_path = project_root / module
        assert file_path.exists(), f"Required module missing: {module}"
        assert file_path.is_file(), f"Path is not a file: {module}"


def test_requirements_file_exists_and_readable(project_root: Path) -> None:
    """3. Test that requirements.txt exists, is readable, and specifies key packages."""
    requirements_file = project_root / "requirements.txt"
    assert requirements_file.exists(), "requirements.txt does not exist"
    assert requirements_file.is_file(), "requirements.txt is not a file"

    content = requirements_file.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, "requirements.txt is empty"

    required_packages = [
        "streamlit",
        "langchain",
        "langchain-community",
        "langchain-text-splitters",
        "pypdf",
        "docx2txt",
        "beautifulsoup4",
        "requests",
        "pillow",
        "pytesseract",
        "python-dotenv",
        "pytest",
    ]

    content_lower = content.lower()
    for package in required_packages:
        assert package in content_lower, f"Package {package} missing from requirements.txt"


def test_configuration_module_importable(project_root: Path) -> None:
    """4. Test that the configuration module can be imported and initialized."""
    import sys

    # Ensure project root is on sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from config.settings import Settings, settings

    assert settings is not None, "settings instance is None"
    assert isinstance(settings, Settings), "settings is not an instance of Settings"
    assert hasattr(settings, "BASE_DIR"), "Settings missing BASE_DIR"
    assert hasattr(settings, "UPLOAD_DIR"), "Settings missing UPLOAD_DIR"
    assert hasattr(settings, "PROCESSED_DIR"), "Settings missing PROCESSED_DIR"
    assert hasattr(settings, "APP_TITLE"), "Settings missing APP_TITLE"
    assert hasattr(settings, "MAX_FILE_SIZE_MB"), "Settings missing MAX_FILE_SIZE_MB"
    assert hasattr(settings, "DEFAULT_CHUNK_SIZE"), "Settings missing DEFAULT_CHUNK_SIZE"
    assert hasattr(settings, "DEFAULT_CHUNK_OVERLAP"), "Settings missing DEFAULT_CHUNK_OVERLAP"
    assert settings.BASE_DIR.exists(), f"BASE_DIR does not exist: {settings.BASE_DIR}"


def test_environment_configuration_no_exposed_secrets(project_root: Path) -> None:
    """5. Test that environment configuration loads properly without exposing hardcoded secrets."""
    env_example = project_root / ".env.example"
    assert env_example.exists(), ".env.example file is missing"

    content = env_example.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Verify no secret patterns are hardcoded in the example
    suspicious_patterns = ["sk-", "ghp_", "api_key=", "password=", "secret="]
    for line in lines:
        cleaned = line.strip().lower()
        if not cleaned or cleaned.startswith("#"):
            continue
        for pattern in suspicious_patterns:
            if pattern in cleaned:
                # Ensure it's not holding an actual value
                value = cleaned.split("=", 1)[-1].strip()
                assert value == "" or value.startswith("<") or value.startswith("your_"), (
                    f"Possible secret exposed in .env.example: {line}"
                )

    # Verify config loads cleanly
    from config.settings import settings
    assert settings.APP_TITLE != "", "APP_TITLE should have a valid default"


def test_gitignore_excludes_sensitive_and_virtualenv_files(project_root: Path) -> None:
    """6. Test that .gitignore excludes .venv, .env, pycache, and pytest cache."""
    gitignore_file = project_root / ".gitignore"
    assert gitignore_file.exists(), ".gitignore file is missing"

    content = gitignore_file.read_text(encoding="utf-8")
    patterns = [".venv", ".env", "__pycache__", "*.pyc", ".pytest_cache"]

    for pattern in patterns:
        assert (
            pattern in content or pattern.replace("*.pyc", "*.py[cod]") in content
        ), f"Missing ignore pattern: {pattern}"
