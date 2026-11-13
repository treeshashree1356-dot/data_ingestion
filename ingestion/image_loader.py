"""Image OCR loader using Pillow and pytesseract."""

from pathlib import Path
import shutil
from typing import Any

from langchain_core.documents import Document
from PIL import Image, ImageEnhance, UnidentifiedImageError
import pytesseract

from config.settings import settings
from ingestion.document_loader import BaseDocumentLoader, DocumentLoadingError

# Supported formats
SUPPORTED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}


def get_tesseract_cmd() -> str | None:
    """Resolve the path to the Tesseract executable."""
    configured = getattr(settings, "TESSERACT_CMD", None)
    if configured:
        return configured
    return shutil.which("tesseract")


def is_tesseract_installed() -> bool:
    """Check if Tesseract OCR executable is available."""
    cmd = get_tesseract_cmd()
    if cmd:
        return Path(cmd).exists() if Path(cmd).is_absolute() else bool(shutil.which(cmd))
    return False


def is_blank_image(image: Image.Image) -> bool:
    """Detect if an image is completely blank / solid color with no detail."""
    gray = image.convert("L")
    extrema = gray.getextrema()
    # All pixels have identical intensity (e.g. solid white or black)
    return extrema[0] == extrema[1]


def preprocess_image(
    image: Image.Image,
    to_grayscale: bool = True,
    enhance_contrast: bool = True,
    contrast_factor: float = 1.5,
    apply_threshold: bool = False,
    threshold_value: int = 150,
) -> Image.Image:
    """Apply standard OCR image preprocessing.

    1. Grayscale conversion.
    2. Contrast enhancement.
    3. Optional binarization / thresholding.

    Args:
        image: Source PIL Image.
        to_grayscale: Whether to convert image to grayscale.
        enhance_contrast: Whether to boost image contrast.
        contrast_factor: Multiplier for contrast enhancement.
        apply_threshold: Whether to apply binary thresholding.
        threshold_value: Pixel cutoff threshold (0-255).

    Returns:
        Preprocessed PIL Image.
    """
    img = image.copy()

    # 1. Grayscale conversion
    if to_grayscale and img.mode != "L":
        img = img.convert("L")

    # 2. Contrast improvement
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

    # 3. Thresholding when useful
    if apply_threshold:
        img = img.point(lambda p: 255 if p > threshold_value else 0, mode="L")

    return img


class ImageDocumentLoader(BaseDocumentLoader):
    """Loader that performs OCR on JPG, JPEG, and PNG images to produce LangChain Documents."""

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        """Initialize the ImageDocumentLoader."""
        cmd = tesseract_cmd or get_tesseract_cmd()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def load(
        self,
        source: str | Path,
        preprocess: bool = True,
        apply_threshold: bool = False,
        lang: str = "eng",
        psm: int = 3,
        **kwargs: Any,
    ) -> list[Document]:
        """Extract text from an image file using OCR and return LangChain Documents.

        Args:
            source: Path to the image file.
            preprocess: Whether to apply grayscale and contrast preprocessing.
            apply_threshold: Whether to apply binary thresholding.
            lang: Tesseract language string (default: 'eng').
            psm: Page segmentation mode.
            **kwargs: Extra parameters.

        Returns:
            List containing a single LangChain Document with extracted text.

        Raises:
            DocumentLoadingError: For missing file, unsupported format, corrupted image,
                                blank image, missing Tesseract, or OCR failure.
        """
        path = Path(source)
        if not path.exists():
            raise DocumentLoadingError(f"Image file not found: {source}")

        # Check unsupported format
        ext = path.suffix.lower().lstrip(".")
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise DocumentLoadingError(
                f"Unsupported image format '.{ext}'. Supported: {sorted(list(SUPPORTED_IMAGE_EXTENSIONS))}"
            )

        # Check empty file
        if path.stat().st_size == 0:
            raise DocumentLoadingError(f"Image file '{path.name}' is empty (0 bytes).")

        # Open and validate image
        try:
            with Image.open(path) as img:
                img.load()  # Ensure image bytes are read
                width, height = img.size

                if is_blank_image(img):
                    raise DocumentLoadingError(
                        f"Image '{path.name}' is blank (contains solid color / no visual content)."
                    )

                # Preprocessing
                proc_img = (
                    preprocess_image(img, apply_threshold=apply_threshold)
                    if preprocess
                    else img
                )

                # Check Tesseract availability
                try:
                    raw_text = pytesseract.image_to_string(
                        proc_img,
                        lang=lang,
                        config=f"--psm {psm}",
                    )
                except pytesseract.TesseractNotFoundError as err:
                    raise DocumentLoadingError(
                        "Tesseract OCR executable not found. Please install Tesseract on your system "
                        "and ensure it is in PATH or configure TESSERACT_CMD in .env."
                    ) from err
                except Exception as err:
                    raise DocumentLoadingError(
                        f"OCR processing failed for '{path.name}': {err}"
                    ) from err

        except UnidentifiedImageError as err:
            raise DocumentLoadingError(
                f"Corrupted or invalid image file '{path.name}': cannot identify image format."
            ) from err
        except OSError as err:
            if "not found" in str(err).lower() or "tesseract" in str(err).lower():
                raise DocumentLoadingError(str(err)) from err
            raise DocumentLoadingError(
                f"Corrupted or unreadable image file '{path.name}': {err}"
            ) from err

        cleaned_text = raw_text.strip()
        if not cleaned_text:
            raise DocumentLoadingError(
                f"No detectable text found in image '{path.name}'."
            )

        metadata = {
            "source": str(path),
            "filename": path.name,
            "source_type": "image",
            "format": ext,
            "image_size": f"{width}x{height}",
            "char_count": len(cleaned_text),
        }

        return [Document(page_content=cleaned_text, metadata=metadata)]
