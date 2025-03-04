import easyocr
from typing import Dict, Any
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        self.reader = None
        self.date_pattern = re.compile(r'\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/(19|20)\d{2}\b')
        
    def get_reader(self):
        if self.reader is None:
            logger.info("Initializing OCR reader")
            self.reader = easyocr.Reader(['en'])
        return self.reader
    
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        try:
            reader = self.get_reader()
            absolute_path = Path(image_path).absolute()
            
            if not absolute_path.exists():
                raise FileNotFoundError(f"OCR input file not found: {absolute_path}")
            
            results = reader.readtext(str(absolute_path))
            extracted_text = ' '.join([text for _, text, _ in results])
            
            id_info = self._extract_id_info(results)
            
            return {
                "full_text": extracted_text,
                "structured_data": id_info
            }
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {str(e)}")
            return {"error": str(e)}
    
    def _extract_id_info(self, ocr_results):
        id_info = {}
        for _, text, _ in ocr_results:
            text_lower = text.lower()
            
            # Extract name
            if 'name' in text_lower and ':' in text:
                id_info['name'] = text.split(':', 1)[1].strip()
            
            # Extract dates
            date_match = self.date_pattern.search(text)
            if date_match:
                id_info['dob'] = date_match.group()
            
            # Extract ID numbers
            if any(key in text_lower for key in ['id', 'number', 'no.']) \
                and sum(c.isdigit() for c in text) > 4:
                id_info['id_number'] = ''.join(filter(str.isdigit, text))
                
        return id_info