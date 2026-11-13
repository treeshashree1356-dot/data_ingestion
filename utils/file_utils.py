"""File management, sanitization, upload handling, and temporary file utilities."""

from contextlib import contextmanager
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, BinaryIO, Generator

from config.settings import settings
from utils.validators import (
    validate_file_extension,
    validate_file_size,
)


def get_file_extension(filename: str | Path) -> str:
    """Return the normalized lowercase file extension without dot."""
    path = Path(filename)
    return path.suffix.lower().lstrip(".")


def is_allowed_file(filename: str | Path, allowed_extensions: list[str]) -> bool:
    """Check if the given filename has an allowed extension."""
    ext = get_file_extension(filename)
    return ext in [e.lower().lstrip(".") for e in allowed_extensions]


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing path traversals and illegal filesystem characters.

    Preserves spaces, alphanumeric characters, standard punctuation, and accents.

    Args:
        filename: Raw filename (can include path or unsafe characters).

    Returns:
        A safe, sanitized filename.
    """
    # Extract only the base name (prevents directory traversal e.g. ../../)
    base_name = Path(filename).name.strip()

    # If completely empty after taking name
    if not base_name:
        return "unnamed_file"

    # Separate stem and extension
    dot_idx = base_name.rfind(".")
    if dot_idx > 0:
        stem = base_name[:dot_idx]
        ext = base_name[dot_idx:]
    else:
        stem = base_name
        ext = ""

    # Replace forbidden filesystem characters (< > : " / \ | ? * and control chars)
    # Note: Spaces and other safe characters are preserved
    sanitized_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip()

    # Disallow names that are just dots or empty
    if not sanitized_stem or all(c == "." for c in sanitized_stem):
        sanitized_stem = "file"

    # Sanitize extension
    sanitized_ext = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]', "", ext)

    return f"{sanitized_stem}{sanitized_ext}"


def get_unique_destination(destination_dir: str | Path, filename: str) -> Path:
    """Generate a collision-free file path in destination_dir.

    If 'report.pdf' exists, generates 'report_1.pdf', 'report_2.pdf', etc.

    Args:
        destination_dir: Target directory.
        filename: Proposed filename.

    Returns:
        A unique Path inside destination_dir.
    """
    dest_dir = Path(destination_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(filename)
    target_path = dest_dir / safe_name

    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix

    counter = 1
    while target_path.exists():
        new_name = f"{stem}_{counter}{suffix}"
        target_path = dest_dir / new_name
        counter += 1

    return target_path


def save_uploaded_file(
    file_data: bytes | BinaryIO | Path | Any,
    original_filename: str | None = None,
    destination_dir: str | Path | None = None,
    validate: bool = True,
    allowed_extensions: list[str] | None = None,
    max_size_mb: int | None = None,
) -> Path:
    """Validate, sanitize, and safely save an uploaded file without collisions.

    Args:
        file_data: Raw bytes, an open file-like object, Path to a source file, or Streamlit UploadedFile.
        original_filename: Name of the file. Required if file_data is bytes or unnamed stream.
        destination_dir: Directory where the file should be stored. Defaults to settings.UPLOAD_DIR.
        validate: Whether to validate extension and size before saving.
        allowed_extensions: Optional list of allowed extensions.
        max_size_mb: Optional max file size in MB.

    Returns:
        Path to the saved file.
    """
    # Resolve filename
    fname: str = ""
    if original_filename:
        fname = original_filename
    elif hasattr(file_data, "name"):
        fname = Path(file_data.name).name
    elif isinstance(file_data, (str, Path)):
        fname = Path(file_data).name
    else:
        fname = "uploaded_file"

    target_dir = Path(destination_dir) if destination_dir else settings.UPLOAD_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if validate:
        validate_file_extension(fname, allowed_extensions=allowed_extensions)
        validate_file_size(file_data, max_size_mb=max_size_mb, filename=fname)

    unique_target = get_unique_destination(target_dir, fname)

    # Write content
    if isinstance(file_data, (bytes, bytearray)):
        unique_target.write_bytes(file_data)
    elif isinstance(file_data, (str, Path)):
        src_path = Path(file_data)
        shutil.copy2(src_path, unique_target)
    elif hasattr(file_data, "read"):
        if hasattr(file_data, "seek"):
            file_data.seek(0)
        with open(unique_target, "wb") as out_f:
            shutil.copyfileobj(file_data, out_f)
    elif hasattr(file_data, "getvalue"):
        # For Streamlit UploadedFile
        unique_target.write_bytes(file_data.getvalue())
    else:
        raise TypeError(f"Unsupported file_data type: {type(file_data)}")

    return unique_target


def save_multiple_files(
    files: list[Any],
    destination_dir: str | Path | None = None,
    validate: bool = True,
    allowed_extensions: list[str] | None = None,
    max_size_mb: int | None = None,
) -> list[Path]:
    """Validate and safely save multiple files (supports mixed file types and duplicates).

    Args:
        files: List of file objects, paths, or (filename, data) tuples.
        destination_dir: Target directory.
        validate: Whether to validate each file before saving.
        allowed_extensions: Optional allowed extensions.
        max_size_mb: Optional max file size in MB.

    Returns:
        List of Paths to saved files.
    """
    saved_paths: list[Path] = []

    for item in files:
        if isinstance(item, tuple) and len(item) == 2:
            fname, data = item
            saved = save_uploaded_file(
                file_data=data,
                original_filename=fname,
                destination_dir=destination_dir,
                validate=validate,
                allowed_extensions=allowed_extensions,
                max_size_mb=max_size_mb,
            )
        elif isinstance(item, (str, Path)):
            p = Path(item)
            saved = save_uploaded_file(
                file_data=p,
                original_filename=p.name,
                destination_dir=destination_dir,
                validate=validate,
                allowed_extensions=allowed_extensions,
                max_size_mb=max_size_mb,
            )
        elif hasattr(item, "name"):
            saved = save_uploaded_file(
                file_data=item,
                original_filename=getattr(item, "name", None),
                destination_dir=destination_dir,
                validate=validate,
                allowed_extensions=allowed_extensions,
                max_size_mb=max_size_mb,
            )
        else:
            raise TypeError(f"Cannot save file item of type {type(item)}")

        saved_paths.append(saved)

    return saved_paths


# ---------------------------------------------------------------------------
# Temporary File Handling
# ---------------------------------------------------------------------------

def create_temp_file(
    suffix: str | None = None,
    prefix: str = "ingest_temp_",
    content: bytes | str | None = None,
    directory: str | Path | None = None,
) -> Path:
    """Create a temporary file on disk, optionally populating it with content.

    Args:
        suffix: File extension/suffix (e.g., '.pdf', '.txt').
        prefix: Filename prefix.
        content: Optional bytes or string content to write.
        directory: Optional directory for the temp file.

    Returns:
        Path to the created temporary file.
    """
    dir_str = str(directory) if directory else None
    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix=prefix,
        dir=dir_str,
    )
    temp_path = Path(temp.name)

    try:
        if content is not None:
            if isinstance(content, str):
                temp.write(content.encode("utf-8"))
            else:
                temp.write(content)
            temp.flush()
    finally:
        temp.close()

    return temp_path


@contextmanager
def temp_file_context(
    suffix: str | None = None,
    prefix: str = "ingest_temp_",
    content: bytes | str | None = None,
    directory: str | Path | None = None,
) -> Generator[Path, None, None]:
    """Context manager for temporary files that guarantees deletion upon exit.

    Yields:
        Path to the temporary file.
    """
    temp_path = create_temp_file(
        suffix=suffix,
        prefix=prefix,
        content=content,
        directory=directory,
    )
    try:
        yield temp_path
    finally:
        cleanup_temp_file(temp_path)


def cleanup_temp_file(path: str | Path) -> bool:
    """Safely delete a temporary file if it exists.

    Args:
        path: Path to the file to remove.

    Returns:
        True if file existed and was removed, False otherwise.
    """
    p = Path(path)
    try:
        if p.exists() and p.is_file():
            p.unlink()
            return True
    except OSError:
        pass
    return False
