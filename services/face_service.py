"""Module for face service operations.

Provides functionality for extracting faces, comparing face encodings,
and capturing images via webcam.
"""

import base64
import io
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel, ValidationError

from unified_logging.logging_setup import setup_logging  # Centralized setup
from utils.config_loader import get_project_root, load_config

# Update sys.path using pathlib instead of os.path
sys.path.append(str(Path(__file__).resolve().parent.parent))


# Pydantic models for configuration validation
class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str
    format: str
    file_path: str


class ImageConfig(BaseModel):
    """Configuration for image processing."""

    output_format: str
    jpeg_quality: int


class WebcamConfig(BaseModel):
    """Configuration for webcam settings."""

    camera_index: int
    countdown_seconds: int
    capture_dir: str
    font_scale: float
    font_thickness: int
    display_time_ms: int


class FaceConfig(BaseModel):
    """Configuration for face processing."""

    similarity_threshold: float
    detection_model: str
    encoding_model: str
    image: ImageConfig
    webcam: WebcamConfig


class PathsConfig(BaseModel):
    """Configuration for file paths."""

    uploads: str
    models: str


class AppConfig(BaseModel):
    """Application configuration."""

    logging: LoggingConfig
    face: FaceConfig
    paths: PathsConfig


# Load and validate configuration
try:
    raw_config = load_config("face_service")
    config = AppConfig.parse_obj(raw_config)
except ValidationError as e:
    msg = f"Configuration validation error: {e}"
    raise RuntimeError(msg) from e  # B904: use raise ... from e


# Set up unified logging (uses default configs.toml from project_root/config/)
setup_logging()


def _raise_file_not_found(message: str) -> None:
    """Raise FileNotFoundError."""
    raise FileNotFoundError(message)


def _raise_runtime_error(message: str) -> None:
    """Raise RuntimeError."""
    raise RuntimeError(message)


class FaceService:
    """Service for face detection, extraction, comparison, and webcam capture."""

    def __init__(self) -> None:
        """Initialize FaceService with configuration."""
        face_config = config.face
        self.similarity_threshold = face_config.similarity_threshold
        self.detection_model = face_config.detection_model
        self.encoding_model = face_config.encoding_model
        self.image_config = face_config.image
        self.webcam_config = face_config.webcam
        self.path_config = config.paths

    def extract_face(self, image_path: str) -> dict[str, Any] | None:
        """Extract the face from an image and return its details.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing face encoding,
            face location, base64-encoded face image, and mime type,
            or None if no face is detected.

        """
        try:
            absolute_path = (
                Path(image_path)
                if Path(image_path).is_absolute()
                else get_project_root() / image_path
            )
            if not absolute_path.exists():
                msg = f"Face image not found: {absolute_path}"
                _raise_file_not_found(msg)
            image = face_recognition.load_image_file(absolute_path)
            face_locations = face_recognition.face_locations(
                image, model=self.detection_model,
            )
            if not face_locations:
                logger.warning(f"No faces detected in {image_path}")
                return None
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            buffered = io.BytesIO()
            if self.image_config.output_format.upper() == "JPEG":
                pil_image.save(
                    buffered,
                    format=self.image_config.output_format,
                    quality=self.image_config.jpeg_quality,
                )
            else:
                pil_image.save(buffered, format=self.image_config.output_format)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            face_encoding = face_recognition.face_encodings(
                image,
                known_face_locations=[face_locations[0]],
                model=self.encoding_model,
            )[0]
            logger.debug(
                "Extracted face from %s with encoding length %s",
                image_path,
                len(face_encoding),
            )
            return {
                "face_encoding": face_encoding.tolist(),
                "face_location": face_locations[0],
                "face_image": img_str,
                "mime_type": f"image/{self.image_config.output_format.lower()}",
            }
        except Exception as error:  # noqa: BLE001
            logger.error(
                "Face processing error in "
                "%s: %s", image_path, str(error), exc_info=True,
            )
            return {"error": str(error)}

    def compare_faces(
        self, known_enc: list[float], unknown_enc: list[float],
    ) -> dict[str, Any]:
        """Compare two face encodings and return the similarity and match result.

        Args:
            known_enc (List[float]): Known face encoding.
            unknown_enc (List[float]): Unknown face encoding.

        Returns:
            Dict[str, Any]: Dictionary containing similarity percentage and
            match boolean.

        """
        try:
            known_array = np.array(known_enc)
            unknown_array = np.array(unknown_enc)
            face_distance = face_recognition.face_distance(
                [known_array], unknown_array,
            )[0]
            similarity = (1.0 - face_distance) * 100
            logger.info("Compared faces with similarity %.2f%%", similarity)
            return {
                "similarity": round(float(similarity), 2),
                "match": bool(similarity >= self.similarity_threshold),
            }
        except Exception as error:  # noqa: BLE001
            logger.error("Comparison Error: %s", str(error), exc_info=True)
            return {"error": str(error)}

    def capture_and_compare(self, session_id: str, sessions: dict) -> dict[str, Any]:
        """Capture an image from the webcam, extract the face, and compare
        with a known face.

        Args:
            session_id (str): Session identifier.
            sessions (dict): Dictionary containing session data.

        Returns:
            Dict[str, Any]: Dictionary with similarity, match result,
            id face image, and webcam image path.

        """
        try:
            cap = cv2.VideoCapture(self.webcam_config.camera_index)
            if not cap.isOpened():
                msg = "Could not open webcam"
                _raise_runtime_error(msg)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self.webcam_config.font_scale
            font_thickness = self.webcam_config.font_thickness
            display_time = self.webcam_config.display_time_ms
            start_time = time.time()
            while (time.time() - start_time) < self.webcam_config.countdown_seconds:
                ret, frame = cap.read()
                if not ret:
                    msg = "Could not capture frame"
                    _raise_runtime_error(msg)
                remaining = (
                    self.webcam_config.countdown_seconds - int(time.time() - start_time)
                )
                cv2.putText(
                    frame,
                    f"Capturing in {remaining}...",
                    (50, 50),
                    font,
                    font_scale,
                    (0, 0, 255),
                    font_thickness,
                )
                cv2.imshow("Webcam", frame)
                cv2.waitKey(display_time)
            ret, frame = cap.read()
            if not ret:
                msg = "Could not capture final frame"
                _raise_runtime_error(msg)
            capture_dir = get_project_root() / self.webcam_config.capture_dir
            capture_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            image_path = capture_dir / f"capture_{timestamp}.jpg"
            cv2.imwrite(str(image_path), frame)
            cv2.putText(
                frame,
                "Captured!",
                (50, 50),
                font,
                font_scale,
                (0, 255, 0),
                font_thickness,
            )
            cv2.imshow("Webcam", frame)
            cv2.waitKey(display_time)
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam image captured at %s", image_path)
            webcam_face = self.extract_face(str(image_path))
            if webcam_face is None or "face_encoding" not in webcam_face:
                logger.error("No face detected in the webcam image")
                return {
                    "similarity": 0.0,
                    "match": False,
                    "id_face_image":
                    sessions[session_id]["data"]["id"]["face"]["face_image"],
                    "webcam_image_path": str(image_path),
                    "error": "No face detected in the webcam image",
                }
            id_face = sessions[session_id]["data"]["id"]["face"]
            comparison = self.compare_faces(
                id_face["face_encoding"], webcam_face["face_encoding"],
            )
            logger.success(
                "Face comparison completed: match=%s", comparison.get("match"),
            )
            return {
                "similarity": comparison.get("similarity"),
                "match": comparison.get("match"),
                "id_face_image": id_face["face_image"],
                "webcam_image_path": str(image_path),
            }
        except Exception as error:
            logger.error("Webcam capture failed: %s", str(error), exc_info=True)
            if "cap" in locals():
                cap.release()
            cv2.destroyAllWindows()
            raise  # Use bare raise per TRY201


if __name__ == "__main__":
    face_service = FaceService()
    face_service.capture_and_compare(
        "test_session",
        {
            "test_session": {
                "data": {"id": {"face": {"face_encoding": [], "face_image": ""}}},
            },
        },
    )
