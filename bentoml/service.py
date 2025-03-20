"""Integrated verification service module."""

import sys
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

import bentoml

# Update sys.path using pathlib
sys.path.append(str(Path(__file__).resolve().parent.parent))

from unified_logging.logging_setup import setup_logging

# Set up unified logging
setup_logging()

with bentoml.importing():
    from services.face_service import FaceService
    from services.gesture_service import GestureService
    from services.ocr_service import OCRService


@bentoml.service
class VerificationSession:
    """Class for managing verification sessions."""

    def __init__(self) -> None:
        """Initialize the verification session with services and session storage."""
        self.session_id = str(uuid.uuid4())
        self.sessions: dict[str, dict[str, Any]] = {}
        self.ocr_service = OCRService()
        self.face_service = FaceService()
        self.gesture_service = GestureService()
        logger.info(
            "New verification session created with ID: {session_id}",
            session_id=self.session_id,
        )
        logger.debug(
            "Initialized services: OCR, Face, Gesture for session {session_id}",
            session_id=self.session_id,
        )

    @bentoml.api
    def process_id_file(self, id_file_path: str) -> dict[str, Any]:
        """Process the ID file using OCR and face extraction."""
        logger.info(
            "Starting ID file processing for session {session_id}, file: {file_path}",
            session_id=self.session_id,
            file_path=id_file_path,
        )
        result = None
        try:
            logger.debug("Running OCR on {file_path}", file_path=id_file_path)
            ocr_result = self.ocr_service.extract_text(id_file_path)
            if "error" in ocr_result:
                logger.error(
                    "OCR processing failed for {file_path}: {error}",
                    file_path=id_file_path,
                    error=ocr_result["error"],
                )
                result = {"error": ocr_result["error"]}
            else:
                logger.debug("Extracting face from {file_path}", file_path=id_file_path)
                face_result = self.face_service.extract_face(id_file_path)
                if face_result is None or "error" in face_result:
                    error_msg = (
                        face_result.get("error", "No face detected in ID image")
                        if face_result
                        else "No face detected"
                    )
                    logger.error(
                        "Face extraction failed for {file_path}: {error_msg}",
                        file_path=id_file_path,
                        error_msg=error_msg,
                    )
                    result = {"error": error_msg}
                else:
                    self.sessions[self.session_id] = {
                        "data": {"id": {"ocr": ocr_result, "face": face_result}},
                    }
                    logger.debug(
                        "Stored OCR and face results in session {session_id}",
                        session_id=self.session_id,
                    )
                    logger.info(
                        "ID file processed successfully for session {session_id}",
                        session_id=self.session_id,
                    )
                    result = {
                        "session_id": self.session_id,
                        "ocr_data": ocr_result,
                        "face_data": face_result,
                    }
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Error processing ID file {file_path} for session {session_id}: {err}",
                file_path=id_file_path,
                session_id=self.session_id,
                err=str(e),
                exc_info=True,
            )
            result = {"error": str(e)}
        return result

    @bentoml.api
    def capture_and_compare(self) -> dict[str, Any]:
        """Capture webcam image and compare with ID face."""
        logger.info(
            "Starting capture and compare for session {session_id}",
            session_id=self.session_id,
        )
        result = None
        try:
            if self.session_id not in self.sessions:
                logger.warning(
                    "No session found for {session_id}. Please process ID file first.",
                    session_id=self.session_id,
                )
                result = {"error": "No active session. Process ID file first."}
            else:
                logger.debug(
                    "Running capture and compare for session {session_id}",
                    session_id=self.session_id,
                )
                result = self.face_service.capture_and_compare(
                    self.session_id,
                    self.sessions,
                )
                if "error" in result:
                    logger.error(
                        "Capture and compare failed for session {session_id}: {error}",
                        session_id=self.session_id,
                        error=result["error"],
                    )
                else:
                    self.sessions[self.session_id]["data"]["webcam"] = result
                    logger.debug(
                        "Stored webcam comparison result in session {session_id}",
                        session_id=self.session_id,
                    )
                    logger.info(
                        "Capture and compare completed for session {session_id}",
                        session_id=self.session_id,
                    )
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Error in capture_and_compare for session {session_id}: {error}",
                session_id=self.session_id,
                error=str(e),
                exc_info=True,
            )
            result = {"error": str(e)}
        return result

    @bentoml.api
    def verify_gesture(self) -> dict[str, Any]:
        """Verify liveness through gesture recognition."""
        logger.info(
            "Starting gesture verification for session {session_id}",
            session_id=self.session_id,
        )
        result = None
        try:
            if self.session_id not in self.sessions:
                logger.warning(
                    "No session found for {session_id}. Please process ID file first.",
                    session_id=self.session_id,
                )
                result = {"error": "No active session. Process ID file first."}
            elif "face" not in self.sessions[self.session_id]["data"]["id"]:
                logger.warning(
                    "No face data found in session {session_id}",
                    session_id=self.session_id,
                )
                result = {"error": "No face data available. Process ID file first."}
            else:
                face_data = self.sessions[self.session_id]["data"]["id"]["face"]
                reference_face_encoding = face_data["face_encoding"]
                logger.debug(
                    "Using ref face encoding for gesture verification in session {sid}",
                    sid=self.session_id,
                )
                result = self.gesture_service.verify_liveness(reference_face_encoding)
                self.sessions[self.session_id]["data"]["gesture"] = result
                logger.debug(
                    "Stored gesture verification result in session {session_id}",
                    session_id=self.session_id,
                )
                logger.info(
                    "Gesture verification completed for session {session_id}",
                    session_id=self.session_id,
                )
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Error in verify_gesture for session {session_id}: {error}",
                session_id=self.session_id,
                error=str(e),
                exc_info=True,
            )
            result = {"error": str(e)}
        return result

    @bentoml.api
    def get_session_id(self) -> str:
        """Return the current session ID."""
        logger.debug("Retrieved session ID: {session_id}", session_id=self.session_id)
        return self.session_id

    def run_verification(self, id_file_path: str) -> dict[str, Any]:
        """Run the complete verification process."""
        logger.info(
            "Starting full verification process for session {sid}, file: {file_path}",
            sid=self.session_id,
            file_path=id_file_path,
        )
        result = None
        try:
            # Step 1: Process ID file
            logger.debug(
                "Step 1: Processing ID file {file_path}",
                file_path=id_file_path,
            )
            id_result = self.process_id_file(id_file_path)
            if "error" in id_result:
                logger.warning(
                    "ID processing failed: {error}",
                    error=id_result["error"],
                )
                result = id_result
            else:
                # Step 2: Capture and compare face
                logger.debug(
                    "Step 2: Running capture and compare for session {session_id}",
                    session_id=self.session_id,
                )
                face_result = self.capture_and_compare()
                if "error" in face_result:
                    logger.warning(
                        "Capture and compare failed: {error}",
                        error=face_result["error"],
                    )
                    result = face_result
                else:
                    # Step 3: Verify gesture
                    logger.debug(
                        "Step 3: Running gesture verification for session {session_id}",
                        session_id=self.session_id,
                    )
                    gesture_result = self.verify_gesture()
                    if "error" in gesture_result:
                        logger.warning(
                            "Gesture verification failed: {error}",
                            error=gesture_result["error"],
                        )
                        result = gesture_result
                    else:
                        final_result = {
                            "session_id": self.session_id,
                            "id_processing": id_result,
                            "face_verification": face_result,
                            "gesture_verification": gesture_result,
                            "overall_success": (
                                face_result["match"]
                                and gesture_result["success"]
                                and gesture_result["overall_face_match"]
                            ),
                        }
                        logger.info(
                            "Verification process done successfully for session {sid}",
                            sid=self.session_id,
                        )
                        logger.debug(
                            "Final result: overall_success={success}",
                            success=final_result["overall_success"],
                        )
                        result = final_result
        except Exception as e:  # noqa: BLE001
            logger.error(
                "Error in run_verification for session {session_id}: {error}",
                session_id=self.session_id,
                error=str(e),
                exc_info=True,
            )
            result = {"error": str(e)}
        return result


if __name__ == "__main__":
    # Example usage
    verifier = VerificationSession()
    result = verifier.run_verification("path/to/your/id_image.jpg")
    logger.info("Verification result: {result}", result=result)
