import easyocr
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

from pydantic import BaseModel, ValidationError
from utils.config_loader import load_config, get_project_root

logger = logging.getLogger(__name__)

# Pydantic models for configuration validation

class LoggingConfig(BaseModel):
    level: str
    format: str
    file_path: str

class ExtractionConfig(BaseModel):
    date_pattern: str
    name_identifiers: List[str]
    id_identifiers: List[str]
    min_id_digits: int

class OCRConfig(BaseModel):
    languages: List[str]
    use_gpu: bool
    extraction: ExtractionConfig

class PathsConfig(BaseModel):
    uploads: str
    processed: str

class AppConfig(BaseModel):
    logging: LoggingConfig
    ocr: OCRConfig
    paths: PathsConfig

class OCRService:
    def __init__(self):
        # Load and validate configuration
        try:
            raw_config = load_config("ocr_service")
            self.config = AppConfig.parse_obj(raw_config)
        except ValidationError as e:
            raise RuntimeError(f"Configuration validation error: {e}")

        ocr_config = self.config.ocr
        
        # Initialize paths
        self.path_config = self.config.paths
        self.uploads_dir = get_project_root() / self.path_config.uploads
        self.processed_dir = get_project_root() / self.path_config.processed
        self._create_directories()
        
        # OCR engine parameters
        self.languages = ocr_config.languages
        self.use_gpu = ocr_config.use_gpu
        
        # Text extraction patterns
        extraction_config = ocr_config.extraction
        self.date_pattern = re.compile(extraction_config.date_pattern)
        self.name_identifiers = extraction_config.name_identifiers
        self.id_identifiers = extraction_config.id_identifiers
        self.min_id_digits = extraction_config.min_id_digits
        
        # Lazy-loaded OCR reader
        self.reader = None

    def _create_directories(self):
        """Create required file directories from config"""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def get_reader(self):
        """Initialize OCR reader with configured settings"""
        if self.reader is None:
            logger.info("Initializing OCR reader with languages: %s (GPU: %s)",
                        self.languages, self.use_gpu)
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu
            )
        return self.reader

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """End-to-end OCR processing with configured parameters"""
        try:
            reader = self.get_reader()
            absolute_path = self._resolve_input_path(image_path)
            
            if not absolute_path.exists():
                raise FileNotFoundError(f"OCR input file not found: {absolute_path}")
            
            # Perform OCR and process results
            results = reader.readtext(str(absolute_path))
            extracted_text = ' '.join([text for _, text, _ in results])
            structured_data = self._extract_structured_info(results)
            
            # Save processed results
            output_path = self._save_processed_result(structured_data, absolute_path.name)
            
            return {
                "full_text": extracted_text,
                "structured_data": structured_data,
                "processed_path": str(output_path)
            }
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            return {"error": str(e)}

    def _resolve_input_path(self, image_path: str) -> Path:
        """Resolve input path relative to configured uploads directory"""
        path = Path(image_path)
        return path if path.is_absolute() else self.uploads_dir / path

    def _save_processed_result(self, data: Dict, source_filename: str) -> Path:
        """Save structured data to processed directory"""
        timestamp = int(time.time())
        output_filename = f"{Path(source_filename).stem}_{timestamp}.json"
        output_path = self.processed_dir / output_filename
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return output_path

    def _extract_structured_info(self, ocr_results):
        """Advanced information extraction using configured patterns"""
        id_info = {}
        full_text = ' '.join([text for _, text, _ in ocr_results])
        
        # Name extraction
        for identifier in self.name_identifiers:
            if match := re.search(rf"{identifier}[:]?\s*([A-Za-z ]+)", full_text, re.I):
                id_info['name'] = match.group(1).strip()
                break
                
        # Date extraction
        if date_match := self.date_pattern.search(full_text):
            id_info['dob'] = date_match.group()
            
        # ID number extraction
        for identifier in self.id_identifiers:
            if match := re.search(rf"{identifier}[:]?\s*(\d{{{self.min_id_digits},}})", full_text, re.I):
                id_info['id_number'] = match.group(1)
                break
                
        return id_info
