import os
from pathlib import Path

class Settings:
    APP_NAME = "Customer Onboarding"
    UPLOAD_FOLDER = Path(__file__).parent / "uploads"  # Absolute path
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}
    MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
    
    def __init__(self):
        self.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

settings = Settings()