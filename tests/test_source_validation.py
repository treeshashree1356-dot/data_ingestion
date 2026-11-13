"""Phase 2 Input Handling and Source Validation Tests."""

from pathlib import Path
import pytest

from utils.file_utils import (
    cleanup_temp_file,
    create_temp_file,
    get_unique_destination,
    sanitize_filename,
    save_multiple_files,
    save_uploaded_file,
    temp_file_context,
)
from utils.validators import (
    EmptyFileError,
    EmptyManualTextError,
    FileSizeLimitExceededError,
    InvalidFileTypeError,
    InvalidURLError,
    SourceValidationError,
    validate_file,
    validate_file_extension,
    validate_file_size,
    validate_manual_text,
    validate_multiple_files,
    validate_url,
)


# ===========================================================================
# POSITIVE TESTS
# ===========================================================================

def test_accept_valid_pdf(tmp_path: Path) -> None:
    """1. Positive test: Accept valid PDF files."""
    pdf_file = tmp_path / "sample_document.pdf"
    pdf_file.write_bytes(b"%PDF-1.5 Sample PDF Content")

    # Extension validation
    ext = validate_file_extension(pdf_file)
    assert ext == "pdf"

    # Size validation
    size = validate_file_size(pdf_file)
    assert size > 0

    # Unified file validation
    result = validate_file(pdf_file)
    assert result.is_valid is True
    assert result.error_message is None
    assert result.metadata["extension"] == "pdf"


def test_accept_valid_docx(tmp_path: Path) -> None:
    """2. Positive test: Accept valid DOCX files."""
    docx_file = tmp_path / "sample_word.docx"
    docx_file.write_bytes(b"PK\x03\x04 Sample DOCX Content")

    ext = validate_file_extension(docx_file)
    assert ext == "docx"

    result = validate_file(docx_file)
    assert result.is_valid is True
    assert result.metadata["extension"] == "docx"


def test_accept_valid_txt_and_markdown(tmp_path: Path) -> None:
    """3. Positive test: Accept valid TXT and Markdown files."""
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Plain text notes content", encoding="utf-8")

    md_file = tmp_path / "README.md"
    md_file.write_text("# Markdown Title\nContent", encoding="utf-8")

    for f, expected_ext in [(txt_file, "txt"), (md_file, "md")]:
        ext = validate_file_extension(f)
        assert ext == expected_ext
        result = validate_file(f)
        assert result.is_valid is True


def test_accept_valid_image_files(tmp_path: Path) -> None:
    """4. Positive test: Accept valid JPG, JPEG, and PNG files."""
    images = {
        "photo.jpg": (b"\xff\xd8\xff Sample JPG", "jpg"),
        "picture.jpeg": (b"\xff\xd8\xff Sample JPEG", "jpeg"),
        "diagram.png": (b"\x89PNG\r\n\x1a\n Sample PNG", "png"),
    }

    for fname, (content, expected_ext) in images.items():
        img_path = tmp_path / fname
        img_path.write_bytes(content)

        ext = validate_file_extension(img_path)
        assert ext == expected_ext

        result = validate_file(img_path)
        assert result.is_valid is True
        assert result.metadata["extension"] == expected_ext


def test_accept_valid_http_and_https_urls() -> None:
    """5. Positive test: Accept valid HTTP and HTTPS URLs."""
    valid_urls = [
        "http://example.com",
        "https://example.org",
        "https://www.example.com/docs/guide?version=2.0#overview",
        "http://localhost:8501",
        "http://127.0.0.1:8000/api/v1/resource",
        "https://sub.domain.co.uk/path/file.html",
    ]

    for url in valid_urls:
        validated = validate_url(url)
        assert validated == url


def test_accept_non_empty_manual_text() -> None:
    """6. Positive test: Accept non-empty manual text."""
    samples = [
        "Simple manual text input.",
        "   Text with leading and trailing spaces   ",
        "Multi-line manual text\nLine 2\nLine 3",
        "Special symbols: @#$%^&*()_+ and emoji 🚀",
    ]

    for sample in samples:
        validated = validate_manual_text(sample)
        assert validated == sample.strip()
        assert len(validated) > 0


def test_accept_multiple_valid_files(tmp_path: Path) -> None:
    """7. Positive test: Accept multiple valid files in batch."""
    upload_dest = tmp_path / "uploads"
    files = [
        ("report.pdf", b"%PDF-1.4 Data Ingestion Report"),
        ("notes.txt", b"Notes text content"),
        ("spec.docx", b"PK\x03\x04 Word Specification"),
        ("chart.png", b"\x89PNG Chart Data"),
    ]

    batch_inputs = []
    for fname, content in files:
        fpath = tmp_path / fname
        fpath.write_bytes(content)
        batch_inputs.append(fpath)

    results = validate_multiple_files(batch_inputs)
    assert len(results) == 4
    for res in results:
        assert res.is_valid is True

    # Save multiple files and verify all exist
    saved = save_multiple_files(batch_inputs, destination_dir=upload_dest)
    assert len(saved) == 4
    for saved_file in saved:
        assert saved_file.exists()


# ===========================================================================
# NEGATIVE TESTS
# ===========================================================================

def test_reject_unsupported_extensions(tmp_path: Path) -> None:
    """8. Negative test: Reject unsupported extensions."""
    unsupported_files = [
        "malware.exe",
        "data.csv",
        "table.xlsx",
        "archive.zip",
        "script.py",
        "style.css",
        "video.mp4",
    ]

    for fname in unsupported_files:
        fpath = tmp_path / fname
        fpath.write_bytes(b"Unsupported content")

        with pytest.raises(InvalidFileTypeError):
            validate_file_extension(fpath)

        result = validate_file(fpath)
        assert result.is_valid is False
        assert "Unsupported file extension" in result.error_message


def test_reject_empty_files(tmp_path: Path) -> None:
    """9. Negative test: Reject empty (0-byte) files."""
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.touch()  # 0 bytes

    with pytest.raises(EmptyFileError):
        validate_file_size(empty_pdf)

    result = validate_file(empty_pdf)
    assert result.is_valid is False
    assert "empty" in result.error_message.lower()


def test_reject_invalid_urls() -> None:
    """10. Negative test: Reject invalid URLs."""
    invalid_urls = [
        "",
        "   ",
        "ftp://ftp.example.com/file.txt",
        "javascript:alert(1)",
        "file:///Users/harsh/secret.txt",
        "http://",
        "https://",
        "not_a_valid_url",
        "http:///missing-host",
        None,
    ]

    for url in invalid_urls:
        with pytest.raises(InvalidURLError):
            validate_url(url)


def test_reject_empty_manual_text() -> None:
    """11. Negative test: Reject empty manual text."""
    empty_inputs = [
        "",
        "   ",
        "\t\n  \r\n",
        None,
    ]

    for text_input in empty_inputs:
        with pytest.raises(EmptyManualTextError):
            validate_manual_text(text_input)


def test_reject_files_exceeding_size_limit(tmp_path: Path) -> None:
    """12. Negative test: Reject files exceeding configured size limits."""
    # Test with max_size_mb = 1 (1MB) and create a 2MB file
    large_file = tmp_path / "oversized.pdf"
    large_file.write_bytes(b"X" * (2 * 1024 * 1024))

    with pytest.raises(FileSizeLimitExceededError):
        validate_file_size(large_file, max_size_mb=1)

    result = validate_file(large_file, max_size_mb=1)
    assert result.is_valid is False
    assert "exceeds maximum allowed size" in result.error_message


# ===========================================================================
# EDGE CASES
# ===========================================================================

def test_handle_filenames_with_spaces(tmp_path: Path) -> None:
    """13. Edge case: Handle filenames containing spaces."""
    dest_dir = tmp_path / "upload_space"
    filename_with_spaces = "Quarterly Business Review 2026.pdf"

    saved_path = save_uploaded_file(
        file_data=b"%PDF-1.4 QBR Content",
        original_filename=filename_with_spaces,
        destination_dir=dest_dir,
    )

    assert saved_path.exists()
    assert "Quarterly Business Review 2026.pdf" in saved_path.name
    assert saved_path.read_bytes() == b"%PDF-1.4 QBR Content"


def test_handle_filenames_with_special_characters(tmp_path: Path) -> None:
    """14. Edge case: Handle filenames containing special characters and prevent path traversal."""
    dest_dir = tmp_path / "upload_special"

    test_cases = [
        ("report (v2) #1 & final [draft].pdf", True),
        ("résumé @ company! $2026.docx", True),
        ("../../etc/passwd.txt", False),  # Directory traversal attempt
    ]

    for raw_name, should_keep_name in test_cases:
        sanitized = sanitize_filename(raw_name)
        assert "/" not in sanitized
        assert "\\" not in sanitized
        assert ".." not in sanitized

        saved_path = save_uploaded_file(
            file_data=b"Special file content",
            original_filename=raw_name,
            destination_dir=dest_dir,
        )
        assert saved_path.exists()
        # Verify saved file resides strictly inside dest_dir
        assert saved_path.parent.resolve() == dest_dir.resolve()


def test_handle_duplicate_filenames_safely(tmp_path: Path) -> None:
    """15. Edge case: Handle duplicate filenames safely without overwriting."""
    dest_dir = tmp_path / "upload_dups"

    # Save first file
    file1 = save_uploaded_file(
        file_data=b"Version 1",
        original_filename="document.pdf",
        destination_dir=dest_dir,
    )
    assert file1.name == "document.pdf"
    assert file1.read_bytes() == b"Version 1"

    # Save duplicate file
    file2 = save_uploaded_file(
        file_data=b"Version 2",
        original_filename="document.pdf",
        destination_dir=dest_dir,
    )
    assert file2.name == "document_1.pdf"
    assert file2.read_bytes() == b"Version 2"

    # Save third duplicate file
    file3 = save_uploaded_file(
        file_data=b"Version 3",
        original_filename="document.pdf",
        destination_dir=dest_dir,
    )
    assert file3.name == "document_2.pdf"
    assert file3.read_bytes() == b"Version 3"

    # Ensure all three files still exist with their original distinct content
    assert file1.read_bytes() == b"Version 1"
    assert file2.read_bytes() == b"Version 2"
    assert file3.read_bytes() == b"Version 3"


def test_handle_multiple_mixed_file_types(tmp_path: Path) -> None:
    """16. Edge case: Handle multiple mixed file types simultaneously."""
    dest_dir = tmp_path / "upload_mixed"

    mixed_files = [
        ("contract.pdf", b"%PDF-1.4 Contract Content"),
        ("memo.docx", b"PK\x03\x04 Word Memo Content"),
        ("notes.txt", b"Plain text notes"),
        ("guide.md", b"# Markdown Guide"),
        ("scan.jpg", b"\xff\xd8\xff JPG Scan"),
        ("photo.jpeg", b"\xff\xd8\xff JPEG Photo"),
        ("screenshot.png", b"\x89PNG Screenshot"),
    ]

    saved_paths = save_multiple_files(mixed_files, destination_dir=dest_dir)
    assert len(saved_paths) == len(mixed_files)

    for saved_file, (orig_name, orig_content) in zip(saved_paths, mixed_files):
        assert saved_file.exists()
        assert saved_file.read_bytes() == orig_content


# ===========================================================================
# TEMPORARY FILE UTILITIES TEST
# ===========================================================================

def test_temporary_file_utilities(tmp_path: Path) -> None:
    """Verify temporary file creation, context manager auto-cleanup, and manual cleanup."""
    # Context manager test
    temp_target: Path
    with temp_file_context(suffix=".pdf", content=b"%PDF Temp Content", directory=tmp_path) as tf:
        temp_target = tf
        assert temp_target.exists()
        assert temp_target.read_bytes() == b"%PDF Temp Content"

    # Verify auto-cleanup upon context exit
    assert not temp_target.exists()

    # Manual creation and cleanup
    tf_manual = create_temp_file(suffix=".txt", content="Manual temp file", directory=tmp_path)
    assert tf_manual.exists()
    assert cleanup_temp_file(tf_manual) is True
    assert not tf_manual.exists()
    # Second cleanup call should return False safely without error
    assert cleanup_temp_file(tf_manual) is False
