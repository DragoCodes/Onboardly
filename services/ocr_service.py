"""OCR service module.

Provides functionality for performing OCR processing and extracting structured
information from images.
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Use Path methods instead of os.path.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import easyocr
from loguru import logger  # Unified logging with loguru
from pydantic import BaseModel, ValidationError

from unified_logging.logging_setup import setup_logging  # Centralized setup
from utils.config_loader import get_project_root, load_config


# Pydantic models for configuration validation
class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str
    format: str
    file_path: str


class ExtractionConfig(BaseModel):
    """Configuration for advanced text extraction."""

    date_pattern: str
    name_identifiers: list[str]
    id_identifiers: list[str]
    min_id_digits: int


class OCRConfig(BaseModel):
    """Configuration for OCR settings."""

    languages: list[str]
    use_gpu: bool
    extraction: ExtractionConfig


class PathsConfig(BaseModel):
    """Configuration for file paths."""

    uploads: str
    processed: str


class AppConfig(BaseModel):
    """Application configuration for OCR service."""

    logging: LoggingConfig
    ocr: OCRConfig
    paths: PathsConfig


# Set up unified logging.
setup_logging()


class OCRService:
    """Service for OCR processing and structured information extraction."""

    def __init__(self) -> None:
        """Initialize OCRService and load configuration."""
        try:
            raw_config = load_config("ocr_service")
            self.config = AppConfig.parse_obj(raw_config)
            logger.info("OCRService configuration loaded successfully")
        except ValidationError as e:
            error_msg = "Configuration validation error: " + str(e)
            logger.error("OCRService config validation error: %s", str(e))
            raise RuntimeError(error_msg) from e

        ocr_config = self.config.ocr

        # Initialize paths.
        self.path_config = self.config.paths
        self.uploads_dir = get_project_root() / self.path_config.uploads
        self.processed_dir = get_project_root() / self.path_config.processed
        self._create_directories()  # type: ignore[no-untyped-call]

        # OCR engine parameters.
        self.languages = ocr_config.languages
        self.use_gpu = ocr_config.use_gpu

        # Text extraction patterns.
        extraction_config = ocr_config.extraction
        self.date_pattern = re.compile(extraction_config.date_pattern)
        self.name_identifiers = extraction_config.name_identifiers
        self.id_identifiers = extraction_config.id_identifiers
        self.min_id_digits = extraction_config.min_id_digits

        # Lazy-loaded OCR reader.
        self.reader: easyocr.Reader | None = None
        logger.debug(
            "OCRService initialized with languages=%s, use_gpu=%s",
            self.languages,
            self.use_gpu,
        )

    def _create_directories(self) -> None:
        """Create required file directories from config."""
        try:
            self.uploads_dir.mkdir(parents=True, exist_ok=True)
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(
                "Directories created: uploads=%s, processed=%s",
                self.uploads_dir,
                self.processed_dir,
            )
        except Exception as e:
            logger.error("Failed to create directories: %s", str(e), exc_info=True)
            raise

    def get_reader(self) -> easyocr.Reader:
        """Initialize OCR reader with configured settings."""
        if self.reader is None:
            logger.info(
                "Initializing OCR reader with languages=%s, gpu=%s",
                self.languages,
                self.use_gpu,
            )
            try:
                self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
                logger.debug(
                    "OCR reader initialized successfully",
                )
            except Exception as e:
                logger.error(
                    "Failed to initialize OCR reader: %s",
                    str(e),
                    exc_info=True,
                )
                raise
        return self.reader

    def extract_text(self, image_path: str) -> dict[str, Any]:
        """End-to-end OCR processing with configured parameters."""
        logger.info("Starting OCR processing for image: %s", image_path)
        try:
            reader = self.get_reader()
            absolute_path = self._resolve_input_path(image_path)

            if not absolute_path.exists():
                error_msg = "OCR input file not found: " + str(absolute_path)
                logger.error(error_msg)
                self._raise_file_not_found(error_msg)

            # Perform OCR and process results.
            logger.debug("Running OCR on %s", str(absolute_path))
            results = reader.readtext(str(absolute_path))
            extracted_text = " ".join([text for _, text, _ in results])
            structured_data = self._extract_structured_info(results)

            # Save processed results.
            output_path = self._save_processed_result(
                structured_data, absolute_path.name,
            )
            logger.info("OCR processing completed, results saved to %s", output_path)

            return {
                "full_text": extracted_text,
                "structured_data": structured_data,
                "processed_path": str(output_path),
            }
        except Exception as e:  # noqa: BLE001
            logger.error(
                "OCR processing error for %s: %s", image_path, str(e), exc_info=True,
            )
            return {"error": str(e)}

    def _resolve_input_path(self, image_path: str) -> Path:
        """Resolve input path relative to configured uploads directory."""
        path = Path(image_path)
        resolved_path = path if path.is_absolute() else self.uploads_dir / path
        logger.debug("Resolved input path: %s -> %s", image_path, resolved_path)
        return resolved_path

    def _save_processed_result(self, data: dict, source_filename: str) -> Path:
        """Save structured data to processed directory."""
        try:
            timestamp = int(time.time())
            output_filename = f"{Path(source_filename).stem}_{timestamp}.json"
            output_path = self.processed_dir / output_filename

            with output_path.open("w") as f:  # PTH123
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(
                "Failed to save processed result: %s", str(e), exc_info=True,
            )
            raise
        else:
            logger.debug("Structured data saved to %s", output_path)
            return output_path

    def _raise_file_not_found(self, message: str) -> None:
        """Raise FileNotFoundError using an inner function."""
        raise FileNotFoundError(message)

    def _extract_structured_info(
        self, ocr_results: list[list[Any]],
    ) -> dict[str, Any]:
        """Advanced information extraction using configured patterns."""
        logger.debug("Extracting structured information from OCR results")
        id_info: dict[str, Any] = {}
        full_text = " ".join([text for _, text, _ in ocr_results])

        # Name extraction.
        for identifier in self.name_identifiers:
            if match := re.search(
                rf"{identifier}[:]?\s*([A-Za-z ]+)",
                full_text,
                re.IGNORECASE,
            ):
                id_info["name"] = match.group(1).strip()
                logger.debug("Extracted name: %s", id_info["name"])
                break

        # Date extraction.
        if date_match := self.date_pattern.search(full_text):
            id_info["dob"] = date_match.group()
            logger.debug("Extracted DOB: %s", id_info["dob"])

        # ID number extraction.
        for identifier in self.id_identifiers:
            if match := re.search(
                rf"{identifier}[:]?\s*(\d{{{self.min_id_digits},}})",
                full_text,
                re.IGNORECASE,
            ):
                id_info["id_number"] = match.group(1)
                logger.debug("Extracted ID number: %s", id_info["id_number"])
                break

        if not id_info:
            logger.warning("No structured information extracted from OCR results")
        return id_info


if __name__ == "__main__":
    ocr_service = OCRService()
    result = ocr_service.extract_text("aadhar.jpg")
    logger.info("OCR result: %s", result)
