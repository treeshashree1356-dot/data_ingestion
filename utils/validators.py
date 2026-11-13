"""Modular source and input validation utilities for Data Ingestion Pipeline.

Supports validation for files, URLs, manual text, and batch inputs.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

from config.settings import settings


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class SourceValidationError(Exception):
    """Base exception for all source validation errors."""
    pass


class InvalidFileTypeError(SourceValidationError):
    """Raised when an uploaded file extension is not supported."""
    pass


class FileSizeLimitExceededError(SourceValidationError):
    """Raised when a file exceeds the configured maximum file size."""
    pass


class EmptyFileError(SourceValidationError):
    """Raised when an uploaded file is empty (0 bytes)."""
    pass


class InvalidURLError(SourceValidationError):
    """Raised when a provided URL is malformed or invalid."""
    pass


class EmptyManualTextError(SourceValidationError):
    """Raised when manual text input is empty or contains only whitespace."""
    pass


# ---------------------------------------------------------------------------
# Validation Result Data Structure
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Represents the outcome of a source validation check."""
    is_valid: bool
    error_message: str | None = None
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.is_valid


# ---------------------------------------------------------------------------
# Supported Extensions
# ---------------------------------------------------------------------------

DEFAULT_ALLOWED_EXTENSIONS: list[str] = [
    "pdf",
    "docx",
    "txt",
    "md",
    "markdown",
    "jpg",
    "jpeg",
    "png",
]


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def validate_file_extension(
    filename: str | Path,
    allowed_extensions: list[str] | None = None,
) -> str:
    """Validate that the file extension is supported.

    Args:
        filename: Name or Path of the file.
        allowed_extensions: Optional list of allowed extensions. Defaults to settings.

    Returns:
        The normalized lowercase extension (without dot).

    Raises:
        InvalidFileTypeError: If the extension is not in the allowed list.
    """
    path = Path(filename)
    raw_ext = path.suffix.lower().lstrip(".")

    allowed = allowed_extensions or getattr(settings, "ALLOWED_EXTENSIONS", DEFAULT_ALLOWED_EXTENSIONS)
    normalized_allowed = [e.lower().lstrip(".") for e in allowed]

    # Also accept 'markdown' alongside 'md'
    if "md" in normalized_allowed and "markdown" not in normalized_allowed:
        normalized_allowed.append("markdown")

    if not raw_ext or raw_ext not in normalized_allowed:
        raise InvalidFileTypeError(
            f"Unsupported file extension '{raw_ext or '<none>'}'. "
            f"Allowed extensions: {sorted(list(set(normalized_allowed)))}"
        )

    return raw_ext


def validate_file_size(
    file_input: str | Path | int | bytes | BinaryIO,
    max_size_mb: int | None = None,
    min_size_bytes: int = 1,
    filename: str = "file",
) -> int:
    """Validate file size constraints (rejects empty and oversized files).

    Args:
        file_input: Path, size in bytes (int), raw bytes, or a file-like object.
        max_size_mb: Maximum allowed file size in megabytes. Defaults to settings.
        min_size_bytes: Minimum required size in bytes (default 1).
        filename: Optional filename used in error messages.

    Returns:
        The validated file size in bytes.

    Raises:
        EmptyFileError: If the file size is less than min_size_bytes.
        FileSizeLimitExceededError: If the file size exceeds max_size_mb.
    """
    max_mb = max_size_mb if max_size_mb is not None else getattr(settings, "MAX_FILE_SIZE_MB", 50)
    max_bytes = max_mb * 1024 * 1024

    size_bytes: int

    if isinstance(file_input, int):
        size_bytes = file_input
    elif isinstance(file_input, (bytes, bytearray)):
        size_bytes = len(file_input)
    elif isinstance(file_input, (str, Path)):
        p = Path(file_input)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {file_input}")
        size_bytes = p.stat().st_size
        filename = p.name
    elif hasattr(file_input, "seek") and hasattr(file_input, "tell"):
        current_pos = file_input.tell()
        file_input.seek(0, os.SEEK_END)
        size_bytes = file_input.tell()
        file_input.seek(current_pos)
    elif hasattr(file_input, "size"):
        # For Streamlit UploadedFile or similar objects
        size_bytes = int(file_input.size)
        if hasattr(file_input, "name"):
            filename = str(file_input.name)
    else:
        raise TypeError(f"Unsupported file_input type: {type(file_input)}")

    if size_bytes < min_size_bytes:
        raise EmptyFileError(f"File '{filename}' is empty ({size_bytes} bytes).")

    if size_bytes > max_bytes:
        raise FileSizeLimitExceededError(
            f"File '{filename}' exceeds maximum allowed size of {max_mb} MB "
            f"({size_bytes / (1024 * 1024):.2f} MB)."
        )

    return size_bytes


def validate_file(
    file_path_or_name: str | Path,
    content_or_size: bytes | int | BinaryIO | None = None,
    allowed_extensions: list[str] | None = None,
    max_size_mb: int | None = None,
) -> ValidationResult:
    """Validate both file type and file size.

    Args:
        file_path_or_name: File path or filename.
        content_or_size: Optional content bytes, size in bytes, or open file.
        allowed_extensions: Optional list of allowed extensions.
        max_size_mb: Optional maximum file size in MB.

    Returns:
        ValidationResult indicating success or failure with error details.
    """
    path = Path(file_path_or_name)
    filename = path.name

    try:
        ext = validate_file_extension(filename, allowed_extensions)

        if content_or_size is not None:
            size_bytes = validate_file_size(content_or_size, max_size_mb=max_size_mb, filename=filename)
        elif path.exists() and path.is_file():
            size_bytes = validate_file_size(path, max_size_mb=max_size_mb, filename=filename)
        else:
            size_bytes = -1

        return ValidationResult(
            is_valid=True,
            source_type="file",
            metadata={
                "filename": filename,
                "extension": ext,
                "size_bytes": size_bytes,
            },
        )
    except SourceValidationError as err:
        return ValidationResult(
            is_valid=False,
            error_message=str(err),
            source_type="file",
            metadata={"filename": filename},
        )


def validate_url(
    url: str | None,
    allowed_schemes: tuple[str, ...] = ("http", "https"),
) -> str:
    """Validate a website URL for protocol, domain, and format.

    Args:
        url: The URL string to validate.
        allowed_schemes: Tuple of permitted schemes (defaults to ('http', 'https')).

    Returns:
        The stripped, normalized URL string.

    Raises:
        InvalidURLError: If the URL is empty, malformed, or has an invalid scheme/domain.
    """
    if url is None or not isinstance(url, str):
        raise InvalidURLError("URL must be a non-empty string.")

    cleaned_url = url.strip()
    if not cleaned_url:
        raise InvalidURLError("URL cannot be empty or whitespace only.")

    try:
        parsed = urlparse(cleaned_url)
    except Exception as exc:
        raise InvalidURLError(f"Malformed URL '{cleaned_url}': {exc}") from exc

    if not parsed.scheme or parsed.scheme.lower() not in [s.lower() for s in allowed_schemes]:
        raise InvalidURLError(
            f"Invalid URL scheme '{parsed.scheme or '<none>'}'. "
            f"Allowed schemes: {list(allowed_schemes)}"
        )

    netloc = parsed.netloc.strip()
    if not netloc:
        raise InvalidURLError(f"Invalid URL '{cleaned_url}': Missing domain / host.")

    host = netloc.split(":")[0].strip()
    if not host:
        raise InvalidURLError(f"Invalid URL '{cleaned_url}': Hostname is empty.")

    # Validate that host is localhost or has a domain dot or is an IP address
    if host.lower() != "localhost" and "." not in host:
        raise InvalidURLError(
            f"Invalid domain '{host}' in URL '{cleaned_url}'. Must be a valid domain or localhost."
        )

    return cleaned_url


def validate_manual_text(
    text: str | None,
    min_length: int = 1,
    max_length: int | None = None,
) -> str:
    """Validate manual raw text input.

    Args:
        text: Input string.
        min_length: Minimum non-whitespace characters required (default 1).
        max_length: Optional maximum characters allowed.

    Returns:
        The stripped text string.

    Raises:
        EmptyManualTextError: If text is None, empty, or below min_length.
        SourceValidationError: If text exceeds max_length.
    """
    if text is None or not isinstance(text, str):
        raise EmptyManualTextError("Manual text cannot be empty or None.")

    stripped = text.strip()
    if len(stripped) < min_length:
        raise EmptyManualTextError(
            f"Manual text input cannot be empty (minimum {min_length} non-whitespace character(s) required)."
        )

    if max_length is not None and len(stripped) > max_length:
        raise SourceValidationError(
            f"Manual text exceeds maximum allowed length of {max_length} characters (got {len(stripped)})."
        )

    return stripped


def validate_multiple_files(
    files: list[Any],
    allowed_extensions: list[str] | None = None,
    max_size_mb: int | None = None,
) -> list[ValidationResult]:
    """Validate a batch of files (paths, UploadedFile objects, or (name, content) tuples).

    Args:
        files: List of file inputs.
        allowed_extensions: Optional list of allowed extensions.
        max_size_mb: Optional maximum file size in MB.

    Returns:
        List of ValidationResult objects for each file.
    """
    results: list[ValidationResult] = []

    for file_item in files:
        if isinstance(file_item, (str, Path)):
            res = validate_file(
                file_item,
                allowed_extensions=allowed_extensions,
                max_size_mb=max_size_mb,
            )
        elif isinstance(file_item, tuple) and len(file_item) == 2:
            filename, content = file_item
            res = validate_file(
                filename,
                content_or_size=content,
                allowed_extensions=allowed_extensions,
                max_size_mb=max_size_mb,
            )
        elif hasattr(file_item, "name"):
            res = validate_file(
                file_item.name,
                content_or_size=file_item,
                allowed_extensions=allowed_extensions,
                max_size_mb=max_size_mb,
            )
        else:
            res = ValidationResult(
                is_valid=False,
                error_message=f"Unsupported file item type: {type(file_item)}",
                source_type="file",
            )
        results.append(res)

    return results
