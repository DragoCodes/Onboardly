"""ID router module.

Provides API endpoints for processing ID uploads, including OCR, face extraction,
and gesture verification.
"""

import base64
import tempfile
from pathlib import Path
from typing import Annotated, Any

import aiofiles  # Added for async file operations
import fitz  # PyMuPDF
import magic
from fastapi import APIRouter, Cookie, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

from config import settings
from routers.session_router import sessions
from services.face_service import FaceService
from services.gesture_service import GestureService
from services.ocr_service import OCRService
from unified_logging.logging_setup import setup_logging

# Set up unified logging.
setup_logging()

router = APIRouter()
ocr = OCRService()
face = FaceService()
gesture_service = GestureService()

STORED_VIDEO_PATH = Path(__file__).parent / "video" / "devansh1.mp4"
IMAGE_PATH = Path(__file__).parent / "images" / "d1.jpeg"
UPLOAD_PATH = Path(__file__).parent / "upload" / "sample_upload.jpeg"


class Base64Video(BaseModel):
    """Model for base64 encoded video data."""

    video: str  # Base64 encoded video string.


def convert_pdf_to_images(pdf_path: Path) -> list[Path]:
    """Convert PDF to a list of image paths."""
    logger.debug(f"Converting PDF to images: {pdf_path}")
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap()
            temp_dir = tempfile.mkdtemp()
            img_path = Path(temp_dir) / f"page_{page.number}.png"
            pix.save(img_path)
            images.append(img_path)
        logger.debug(f"Converted PDF to {len(images)} images")
        return images # noqa: TRY300
    except Exception as e:
        logger.error(f"PDF conversion failed for {pdf_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=400, detail=f"PDF conversion failed: {e!s}",
        ) from e


async def validate_file(file: UploadFile) -> None:
    """Validate file type using python-magic."""
    allowed_types = ["image/png", "image/jpeg", "application/pdf"]
    logger.debug(f"Validating file: {file.filename}")

    # Read first 2048 bytes for MIME detection.
    header = await file.read(2048)
    await file.seek(0)
    mime = magic.from_buffer(header, mime=True)
    if mime not in allowed_types:
        logger.warning(f"Unsupported file type: {mime} for {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {mime}. "
                f"Allowed types: {', '.join(allowed_types)}"
            ),
        )
    logger.debug(f"File type validated: {mime}")


async def save_uploaded_file(file: UploadFile, file_path: Path) -> None:
    """Save the uploaded file asynchronously."""
    try:
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(await file.read())
        logger.debug(f"File saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file {file_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to save file: {e!s}",
        ) from e


async def process_file(
    process_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Process the file for OCR and face extraction."""
    logger.debug(f"Processing file: {process_path}")
    ocr_result = ocr.extract_text(process_path)
    face_result = face.extract_face(process_path)

    if face_result and "error" in face_result:
        logger.error(f"Face extraction failed: {face_result['error']}")
        raise HTTPException(
            status_code=400,
            detail=f"Face extraction failed: {face_result['error']}",
        )

    if not face_result or "face_encoding" not in face_result:
        logger.warning(f"No face detected in {process_path}")
        raise HTTPException(
            status_code=400, detail="No face detected in the uploaded ID",
        )

    return ocr_result, face_result


async def save_face_image(
    face_result: dict[str, Any], session_id: str, file: UploadFile,
) -> Path:
    """Save the extracted face image."""
    face_filename = f"face_{session_id}_{file.filename}.jpg"
    face_path = settings.UPLOAD_FOLDER.absolute() / face_filename
    try:
        async with aiofiles.open(face_path, "wb") as f:
            await f.write(base64.b64decode(face_result["face_image"]))
        logger.debug(f"Saved face image to {face_path}")
        return face_path # noqa: TRY300
    except Exception as e:
        logger.error(f"Failed to save face image {face_path}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to save face image: {e!s}",
        ) from e


@router.post("/upload-id")
async def upload_id(
    file: Annotated[UploadFile, File(...)],
    session_id: Annotated[str | None, Cookie()] = None,

) -> dict[str, Any]:
    """Upload an ID image and process OCR and face extraction.

    Args:
        session_id (str): Session ID from cookie.
        file (UploadFile): The uploaded file.

    Returns:
        dict[str, any]: A dictionary containing OCR results, face detection status,
                        extracted face image (if any), and image format.

    """
    logger.info(f"Starting ID upload for session: {session_id}, file: {file.filename}")
    if not session_id or session_id not in sessions:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Invalid session")

    await validate_file(file)

    file_path = settings.UPLOAD_FOLDER.absolute() / f"{session_id}_{file.filename}"
    await save_uploaded_file(file, file_path)

    try:
        images = []
        if file.content_type == "application/pdf":
            images = convert_pdf_to_images(file_path)
            if not images:
                logger.warning(f"No pages found in PDF: {file_path}")
                raise HTTPException(status_code=400, detail="No pages found in PDF") # noqa: TRY301
            process_path = images[0]
        else:
            process_path = file_path

        ocr_result, face_result = await process_file(process_path)

        if "data" not in sessions[session_id]:
            sessions[session_id]["data"] = {}
            logger.debug(f"Initialized data dictionary for session: {session_id}")

        sessions[session_id]["data"]["id"] = {"ocr": ocr_result, "face": face_result}
        logger.debug(f"Stored OCR and face results in session: {session_id}")

        if face_result and "face_image" in face_result:
            face_path = await save_face_image(face_result, session_id, file)
            sessions[session_id]["data"]["id"]["face_image_path"] = str(face_path)

        logger.info(f"ID upload completed successfully for session: {session_id}")
        return {
            "ocr_result": ocr_result,
            "face_detected": face_result is not None,
            "face_image": face_result.get("face_image") if face_result else None,
            "image_format": face_result.get("mime_type") if face_result else None,
        }

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Cleaned up file: {file_path}")
        for img in images:
            if img.exists():
                img.unlink()
                logger.debug(f"Cleaned up temp image: {img}")
        logger.error(
            f"Error during ID upload for session {session_id}: {e}", exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/capture-and-compare")
async def capture_and_compare(
    session_id: Annotated[str | None, Cookie()] = None,
) -> dict[str, Any]:
    """Capture and compare face from webcam with the reference face.

    Args:
        session_id (str): Session ID from cookie.

    Returns:
        dict[str, any]: Results from the face comparison process.

    """
    logger.info(f"Starting capture and compare for session: {session_id}")

    if not session_id or session_id not in sessions:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Invalid session")

    try:
        result = face.capture_and_compare(session_id, sessions)
        logger.info(f"Capture and compare completed for session: {session_id}")
        return result  # noqa: TRY300
    except ValueError as e:
        logger.warning(f"ValueError during capture and compare: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        logger.error(f"RuntimeError during capture and compare: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error during capture and compare: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e!s}",
        ) from e


@router.post("/verify-gesture")
async def verify_gesture(
    session_id: Annotated[str | None, Cookie()] = None,
) -> dict[str, Any]:
    """Verify gesture using the uploaded ID's face encoding.

    Args:
        session_id (str): Session ID from cookie.

    Returns:
        dict[str, any]: The result of the gesture verification.

    """
    logger.info(f"Starting gesture verification for session: {session_id}")
    if not session_id or session_id not in sessions:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Invalid session")

    if "data" not in sessions[session_id]:
        logger.warning(f"No data found in session: {session_id}")
        raise HTTPException(
            status_code=400,
            detail="No data found in session. Please complete ID upload first.",
        )

    if "id" not in sessions[session_id]["data"]:
        logger.warning(f"ID document not uploaded for session: {session_id}")
        raise HTTPException(
            status_code=400,
            detail=(
                "ID document not uploaded. "
                "Please upload ID before verifying gesture."
            ),
        )

    if "face" not in sessions[session_id]["data"]["id"]:
        logger.warning(f"No face data in session: {session_id}")
        raise HTTPException(
            status_code=400,
            detail=(
                "No face data found in session. "
                "Please upload a valid ID with a face."
            ),
        )

    reference_face_encoding = sessions[session_id]["data"]["id"]["face"].get(
        "face_encoding",
    )
    if not reference_face_encoding:
        logger.warning(f"No face encoding found in session: {session_id}")
        raise HTTPException(
            status_code=400,
            detail=(
                "No face encoding found in session data. "
                "Please upload a valid ID with a recognizable face."
            ),
        )

    try:
        result = gesture_service.verify_liveness(reference_face_encoding)
        logger.info(f"Gesture verification completed for session: {session_id}")
        return result  # noqa: TRY300
    except Exception as e:
        logger.error(
            f"Error during gesture verification for session {session_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/test-upload-id")
async def test_upload_id(
    session_id: Annotated[str | None, Cookie()] = None,
) -> dict[str, Any]:
    """Test API to process an ID image (OCR & face extraction) from a stored test
    image.
    """
    logger.info(f"Starting test upload ID for session: {session_id}")
    if not session_id or session_id not in sessions:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Invalid session")

    if not UPLOAD_PATH.exists():
        logger.error("Test ID image not found")
        raise HTTPException(status_code=400, detail="Test ID image not found")

    try:
        ocr_result = ocr.extract_text(str(UPLOAD_PATH))
        face_result = face.extract_face(str(UPLOAD_PATH))

        if face_result and "error" in face_result:
            logger.error(f"Face extraction failed: {face_result['error']}")
            raise HTTPException(   # noqa: TRY301
                status_code=400,
                detail=f"Face extraction failed: {face_result['error']}",
            )

        if not face_result or "face_encoding" not in face_result:
            logger.warning("No face detected in the test ID image")
            raise HTTPException(  # noqa: TRY301
                status_code=400, detail="No face detected in the test ID image",
            )

        if "data" not in sessions[session_id]:
            sessions[session_id]["data"] = {}
            logger.debug(f"Initialized data dictionary for session: {session_id}")

        sessions[session_id]["data"]["id"] = {
            "ocr": ocr_result,
            "face": face_result,
        }
        logger.info(f"Test ID processed and stored for session: {session_id}")

        if face_result and "face_image" in face_result:
            face_filename = f"face_{session_id}_test.jpg"
            face_path = settings.UPLOAD_FOLDER.absolute() / face_filename
            async with aiofiles.open(face_path, "wb") as f:
                await f.write(base64.b64decode(face_result["face_image"]))
            sessions[session_id]["data"]["id"]["face_image_path"] = str(face_path)
            logger.debug(f"Saved test face image to {face_path}")

        return {
            "ocr_result": ocr_result,
            "face_detected": face_result is not None,
            "face_image": face_result.get("face_image") if face_result else None,
            "image_format": face_result.get("mime_type") if face_result else None,
        }

    except Exception as e:
        logger.error(
            f"Error in test_upload_id for session {session_id}: {e}", exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/test-capture-and-compare")
async def test_capture_and_compare(
    session_id: Annotated[str | None, Cookie()] = None,
) -> dict[str, Any]:
    """API to compare a face extracted from a pre-stored image with the reference
    face.
    """
    logger.info(f"Starting test capture and compare for session: {session_id}")
    if not session_id or session_id not in sessions:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Invalid session")

    if not IMAGE_PATH.exists():
        logger.error("Stored image not found")
        raise HTTPException(status_code=400, detail="Stored image not found")

    try:
        image_face = face.extract_face(str(IMAGE_PATH))
        if image_face is None or "face_encoding" not in image_face:
            logger.warning("No face detected in the stored image")
            raise HTTPException(  # noqa: TRY301
                status_code=400, detail="No face detected in the stored image",
            )

        id_face = sessions[session_id]["data"]["id"]["face"]
        comparison = face.compare_faces(
            id_face["face_encoding"], image_face["face_encoding"],
        )
        logger.info(f"Test capture and compare completed for session: {session_id}")

        return {
            "similarity": comparison.get("similarity"),
            "match": comparison.get("match"),
            "id_face_image": id_face.get("face_image"),
            "image_face_image": image_face.get("face_image"),
        }

    except Exception as e:
        logger.error(
            f"Error in test_capture_and_compare for session {session_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/verify-gesture-from-stored-video")
async def verify_gesture_from_stored_video(
    session_id: Annotated[str | None, Cookie()] = None,
) -> dict[str, Any]:
    """Process a pre-stored video to verify gestures.

    This endpoint processes a pre-stored video to verify a fixed gesture
    sequence [1, 2, 3, 4]
    and performs face matching.
    """
    logger.info(
        f"Starting gesture verificatn from stored video for session: {session_id}",
    )
    if not session_id or session_id not in sessions:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Invalid session")

    if "data" not in sessions[session_id] or "id" not in sessions[session_id]["data"]:
        logger.warning("ID document not uploaded")
        raise HTTPException(status_code=400, detail="ID document not uploaded")

    reference_face_encoding = sessions[session_id]["data"]["id"].get("face", {}).get(
        "face_encoding",
    )
    if not reference_face_encoding:
        logger.warning("No face encoding available in session")
        raise HTTPException(
            status_code=400, detail="No face encoding available in session",
        )

    video_file = Path(STORED_VIDEO_PATH)
    if not video_file.exists():
        logger.error("Stored video not found")
        raise HTTPException(status_code=404, detail="Stored video not found")

    try:
        logger.debug(f"Processing stored video: {video_file}")
        results = gesture_service.verify_gesture_from_video(
            reference_face_encoding,
            str(video_file),
            expected_sequence=[1, 2, 3, 4],
        )
        logger.info(
            f"Gesture verification from stored video completed "
            f"for session: {session_id}",
        )
        return results  # noqa: TRY300
    except Exception as e:
        logger.error(
            f"Error during gesture verification from stored video "
            f"for session {session_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e
