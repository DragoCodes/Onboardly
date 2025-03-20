"""Configuration module for Customer Onboarding settings.

This module defines settings for the application.
"""

from pathlib import Path
from typing import ClassVar


class Settings:
    """Settings for the Customer Onboarding application."""

    APP_NAME: ClassVar[str] = "Customer Onboarding"
    UPLOAD_FOLDER: ClassVar[Path] = Path(__file__).parent / "uploads"  # Absolute path
    ALLOWED_EXTENSIONS: ClassVar[set[str]] = {"png", "jpg", "jpeg", "pdf"}
    MAX_FILE_SIZE: ClassVar[int] = 16 * 1024 * 1024  # 16MB

    def __init__(self) -> None:
        """Initialize settings and ensure the upload folder exists."""
        self.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


settings = Settings()
