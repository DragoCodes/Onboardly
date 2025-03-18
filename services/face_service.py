# services/face_service.py

import base64
import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel, ValidationError
from unified_logging.logging_setup import setup_logging  # Centralized setup
from utils.config_loader import get_project_root, load_config


# Pydantic models for configuration validation
class LoggingConfig(BaseModel):
    level: str
    format: str
    file_path: str


class ImageConfig(BaseModel):
    output_format: str
    jpeg_quality: int


class WebcamConfig(BaseModel):
    camera_index: int
    countdown_seconds: int
    capture_dir: str
    font_scale: float
    font_thickness: int
    display_time_ms: int


class FaceConfig(BaseModel):
    similarity_threshold: float
    detection_model: str
    encoding_model: str
    image: ImageConfig
    webcam: WebcamConfig


class PathsConfig(BaseModel):
    uploads: str
    models: str


class AppConfig(BaseModel):
    logging: LoggingConfig
    face: FaceConfig
    paths: PathsConfig


# Load and validate configuration
try:
    raw_config = load_config("face_service")
    config = AppConfig.parse_obj(raw_config)
except ValidationError as e:
    raise RuntimeError(f"Configuration validation error: {e}")

# Set up unified logging (uses default configs.toml from project_root/config/)
setup_logging()


class FaceService:
    def __init__(self):
        face_config = config.face
        self.similarity_threshold = face_config.similarity_threshold
        self.detection_model = face_config.detection_model
        self.encoding_model = face_config.encoding_model
        self.image_config = face_config.image
        self.webcam_config = face_config.webcam
        self.path_config = config.paths

    def extract_face(self, image_path: str):
        try:
            absolute_path = (
                Path(image_path)
                if Path(image_path).is_absolute()
                else get_project_root() / image_path
            )
            if not absolute_path.exists():
                raise FileNotFoundError(f"Face image not found: {absolute_path}")
            image = face_recognition.load_image_file(absolute_path)
            face_locations = face_recognition.face_locations(
                image, model=self.detection_model
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
                f"Extracted face from {image_path} with encoding length {len(face_encoding)}"
            )
            return {
                "face_encoding": face_encoding.tolist(),
                "face_location": face_locations[0],
                "face_image": img_str,
                "mime_type": f"image/{self.image_config.output_format.lower()}",
            }
        except Exception as e:
            logger.error(
                f"Face processing error in {image_path}: {str(e)}", exc_info=True
            )
            return {"error": str(e)}

    def compare_faces(self, known_enc, unknown_enc):
        try:
            known_array = np.array(known_enc)
            unknown_array = np.array(unknown_enc)
            face_distance = face_recognition.face_distance(
                [known_array], unknown_array
            )[0]
            similarity = (1.0 - face_distance) * 100
<<<<<<< HEAD


=======
            logger.info(f"Compared faces with similarity {similarity:.2f}%")
>>>>>>> develop
            return {
                "similarity": round(float(similarity), 2),
                "match": bool(similarity >= self.similarity_threshold),
            }
        except Exception as e:
            logger.error(f"Comparison Error: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def capture_and_compare(self, session_id: str, sessions: dict):
        try:
            cap = cv2.VideoCapture(self.webcam_config.camera_index)
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam")
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self.webcam_config.font_scale
            font_thickness = self.webcam_config.font_thickness
            display_time = self.webcam_config.display_time_ms
            start_time = time.time()
            while (time.time() - start_time) < self.webcam_config.countdown_seconds:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Could not capture frame")
                remaining = self.webcam_config.countdown_seconds - int(
                    time.time() - start_time
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
                raise RuntimeError("Could not capture final frame")
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
            logger.info(f"Webcam image captured at {image_path}")
            webcam_face = self.extract_face(str(image_path))
            if webcam_face is None or "face_encoding" not in webcam_face:
                logger.error("No face detected in the webcam image")
                return {
                    "similarity": 0.0,
                    "match": False,
                    "id_face_image": sessions[session_id]["data"]["id"]["face"][
                        "face_image"
                    ],
                    "webcam_image_path": str(image_path),
                    "error": "No face detected in the webcam image",
                }
            id_face = sessions[session_id]["data"]["id"]["face"]
            comparison = self.compare_faces(
                id_face["face_encoding"], webcam_face["face_encoding"]
            )
            logger.success(f"Face comparison completed: match={comparison['match']}")
            return {
                "similarity": comparison["similarity"],
                "match": comparison["match"],
                "id_face_image": id_face["face_image"],
                "webcam_image_path": str(image_path),
            }
        except Exception as e:
            logger.error(f"Webcam capture failed: {str(e)}", exc_info=True)
            if "cap" in locals():
                cap.release()
            cv2.destroyAllWindows()
            raise e


if __name__ == "__main__":
    face_service = FaceService()
    face_service.capture_and_compare(
        "test_session",
        {
            "test_session": {
                "data": {"id": {"face": {"face_encoding": [], "face_image": ""}}}
            }
        },
    )
