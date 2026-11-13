"""Phase 4 Image OCR Loader Tests."""

from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from PIL import Image, ImageDraw
import pytest
import pytesseract

from ingestion.document_loader import DocumentLoadingError
from ingestion.image_loader import (
    ImageDocumentLoader,
    is_blank_image,
    preprocess_image,
)


def create_sample_image(
    path: Path,
    text: str = "Test",
    fmt: str = "PNG",
    size: tuple[int, int] = (200, 80),
) -> Path:
    """Helper creating an image with drawn text and background noise to simulate scanned/printed document."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    # Draw simple lines/shapes and text representation so it is not blank
    draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline="black")
    draw.text((20, 30), text, fill="black")
    img.save(path, format=fmt)
    return path


def test_extract_text_from_valid_image_using_mocked_ocr(tmp_path: Path) -> None:
    """1. Extract text from a valid image using mocked OCR."""
    img_path = create_sample_image(tmp_path / "receipt.png", text="Receipt Total $42.50")

    loader = ImageDocumentLoader()
    with patch("pytesseract.image_to_string", return_value="Receipt Total $42.50"):
        docs = loader.load(img_path)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Receipt Total $42.50"
    assert docs[0].metadata["filename"] == "receipt.png"


def test_process_jpg_jpeg_and_png(tmp_path: Path) -> None:
    """2. Process JPG, JPEG, and PNG images successfully."""
    formats = [
        ("photo.jpg", "JPEG"),
        ("scan.jpeg", "JPEG"),
        ("diagram.png", "PNG"),
    ]

    loader = ImageDocumentLoader()
    with patch("pytesseract.image_to_string", return_value="Extracted Format Content"):
        for fname, fmt in formats:
            img_path = create_sample_image(tmp_path / fname, text="Content", fmt=fmt)
            docs = loader.load(img_path)
            assert len(docs) == 1
            assert docs[0].page_content == "Extracted Format Content"
            assert docs[0].metadata["filename"] == fname


def test_handle_corrupted_images(tmp_path: Path) -> None:
    """3. Handle corrupted or non-image files."""
    corrupted_img = tmp_path / "corrupted.png"
    corrupted_img.write_bytes(b"\x89PNG Corrupted header payload that fails to open")

    loader = ImageDocumentLoader()
    with pytest.raises(DocumentLoadingError) as exc_info:
        loader.load(corrupted_img)

    assert "Corrupted or invalid image" in str(exc_info.value) or "cannot identify" in str(exc_info.value)


def test_handle_blank_images(tmp_path: Path) -> None:
    """4. Handle blank solid-color images."""
    blank_path = tmp_path / "solid_white.png"
    blank_img = Image.new("RGB", (100, 100), color="white")
    blank_img.save(blank_path)

    assert is_blank_image(blank_img) is True

    loader = ImageDocumentLoader()
    with pytest.raises(DocumentLoadingError) as exc_info:
        loader.load(blank_path)

    assert "is blank" in str(exc_info.value)


def test_handle_images_containing_no_text(tmp_path: Path) -> None:
    """5. Handle images containing shapes/noise but no recognizable text."""
    noisy_path = create_sample_image(tmp_path / "abstract.png", text="")

    loader = ImageDocumentLoader()
    # Mock OCR returning whitespace / empty string
    with patch("pytesseract.image_to_string", return_value="   \n\t  "):
        with pytest.raises(DocumentLoadingError) as exc_info:
            loader.load(noisy_path)

    assert "No detectable text found" in str(exc_info.value)


def test_handle_missing_tesseract_installation(tmp_path: Path) -> None:
    """6. Handle missing Tesseract installation gracefully with clear message."""
    img_path = create_sample_image(tmp_path / "doc.png", text="Some Text")

    loader = ImageDocumentLoader()
    with patch("pytesseract.image_to_string", side_effect=pytesseract.TesseractNotFoundError()):
        with pytest.raises(DocumentLoadingError) as exc_info:
            loader.load(img_path)

    assert "Tesseract OCR executable not found" in str(exc_info.value)


def test_preserve_image_source_metadata(tmp_path: Path) -> None:
    """7. Preserve image source metadata (filename, dimensions, format, char count)."""
    img_path = create_sample_image(
        tmp_path / "scanned_doc.jpg",
        text="Scanned Page 1",
        fmt="JPEG",
        size=(300, 150),
    )

    loader = ImageDocumentLoader()
    with patch("pytesseract.image_to_string", return_value="Scanned Page 1"):
        docs = loader.load(img_path)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["source"] == str(img_path)
    assert doc.metadata["filename"] == "scanned_doc.jpg"
    assert doc.metadata["source_type"] == "image"
    assert doc.metadata["format"] == "jpg"
    assert doc.metadata["image_size"] == "300x150"
    assert doc.metadata["char_count"] == len("Scanned Page 1")


def test_convert_ocr_output_into_langchain_documents(tmp_path: Path) -> None:
    """8. Convert OCR output into compliant LangChain Document objects."""
    img_path = create_sample_image(tmp_path / "card.png", text="Business Card")

    loader = ImageDocumentLoader()
    with patch("pytesseract.image_to_string", return_value="CEO John Doe\nAcme Corp"):
        docs = loader.load(img_path)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "CEO John Doe\nAcme Corp"
    assert isinstance(docs[0].metadata, dict)
