# routers/api_router.py
import base64
import os
import sys
import tempfile
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fitz  # PyMuPDF
import magic
from config import settings
from fastapi import APIRouter, Cookie, File, HTTPException, UploadFile
from loguru import logger  # Unified logging with loguru

# Import sessions from session_router
from routers.session_router import sessions
from services.face_service import FaceService
from services.gesture_service import GestureService
from services.ocr_service import OCRService
from unified_logging.logging_setup import setup_logging  # Centralized setup

# Set up unified logging
setup_logging()

router = APIRouter()
ocr = OCRService()
face = FaceService()
gesture_service = GestureService()


def convert_pdf_to_images(pdf_path: Path) -> List[Path]:
    """Convert PDF to list of image paths"""
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
        return images
    except Exception as e:
        logger.error(f"PDF conversion failed for {pdf_path}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"PDF conversion failed: {str(e)}")


async def validate_file(file: UploadFile):
    """Validate file type using python-magic"""
    allowed_types = ["image/png", "image/jpeg", "application/pdf"]
    logger.debug(f"Validating file: {file.filename}")

    # Read first 2048 bytes for MIME detection
    header = await file.read(2048)
    await file.seek(0)

    mime = magic.from_buffer(header, mime=True)
    if mime not in allowed_types:
        logger.warning(f"Unsupported file type: {mime} for {file.filename}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {mime}. Allowed types: {', '.join(allowed_types)}",
        )
    logger.debug(f"File type validated: {mime}")


@router.post("/upload-id")
async def upload_id(session_id: str = Cookie(None), file: UploadFile = File(...)):
    logger.info(f"Starting ID upload for session: {session_id}, file: {file.filename}")
    if not session_id or session_id not in sessions:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Invalid session")

    await validate_file(file)

    # Save file
    file_path = settings.UPLOAD_FOLDER.absolute() / f"{session_id}_{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        logger.debug(f"File saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file {file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        images = []
        # Handle PDF files
        if file.content_type == "application/pdf":
            images = convert_pdf_to_images(file_path)
            if not images:
                logger.warning(f"No pages found in PDF: {file_path}")
                raise HTTPException(status_code=400, detail="No pages found in PDF")
            process_path = images[0]
        else:
            process_path = file_path

        # Process files
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
                status_code=400, detail="No face detected in the uploaded ID"
            )

        # Initialize id dictionary if it doesn't exist
        if "data" not in sessions[session_id]:
            sessions[session_id]["data"] = {}
            logger.debug(f"Initialized data dictionary for session: {session_id}")

        # Store results
        sessions[session_id]["data"]["id"] = {"ocr": ocr_result, "face": face_result}
        logger.debug(f"Stored OCR and face results in session: {session_id}")

        if face_result and "face_image" in face_result:
            face_filename = f"face_{session_id}_{file.filename}.jpg"
            face_path = settings.UPLOAD_FOLDER.absolute() / face_filename
            try:
                with open(face_path, "wb") as f:
                    f.write(base64.b64decode(face_result["face_image"]))
                sessions[session_id]["data"]["id"]["face_image_path"] = str(face_path)
                logger.debug(f"Saved face image to {face_path}")
            except Exception as e:
                logger.error(
                    f"Failed to save face image {face_path}: {e}", exc_info=True
                )
                raise HTTPException(
                    status_code=500, detail=f"Failed to save face image: {str(e)}"
                )

        logger.info(f"ID upload completed successfully for session: {session_id}")
        return {
            "ocr_result": ocr_result,
            "face_detected": face_result is not None,
            "face_image": face_result.get("face_image") if face_result else None,
            "image_format": face_result.get("mime_type") if face_result else None,
        }

    except Exception as e:
        # Cleanup on error
        if file_path.exists():
            os.remove(file_path)
            logger.debug(f"Cleaned up file: {file_path}")
        for img in images:
            if img.exists():
                os.remove(img)
                logger.debug(f"Cleaned up temp image: {img}")
        logger.error(
            f"Error during ID upload for session {session_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capture-and-compare")
async def capture_and_compare(session_id: str = Cookie(None)):
    logger.info(f"Starting capture and compare for session: {session_id}")

    if not session_id or session_id not in sessions:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Invalid session")

    try:
        result = face.capture_and_compare(session_id, sessions)
        logger.info(f"Capture and compare completed for session: {session_id}")
        return result
    except ValueError as e:
        logger.warning(f"ValueError during capture and compare: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"RuntimeError during capture and compare: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during capture and compare: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@router.post("/verify-gesture")
async def verify_gesture(session_id: str = Cookie(None)):
    logger.info(f"Starting gesture verification for session: {session_id}")
    if not session_id or session_id not in sessions:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=404, detail="Invalid session")

    # Check if data dict exists
    if "data" not in sessions[session_id]:
        logger.warning(f"No data found in session: {session_id}")
        raise HTTPException(
            status_code=400,
            detail="No data found in session. Please complete ID upload first.",
        )

    # Get the reference face encoding from the session data
    if "id" not in sessions[session_id]["data"]:
        logger.warning(f"ID document not uploaded for session: {session_id}")
        raise HTTPException(
            status_code=400,
            detail="ID document not uploaded. Please upload ID before verifying gesture.",
        )

    if "face" not in sessions[session_id]["data"]["id"]:
        logger.warning(f"No face data in session: {session_id}")
        raise HTTPException(
            status_code=400,
            detail="No face data found in session. Please upload a valid ID with a face.",
        )

    reference_face_encoding = sessions[session_id]["data"]["id"]["face"].get(
        "face_encoding"
    )
    if not reference_face_encoding:
        logger.warning(f"No face encoding found in session: {session_id}")
        raise HTTPException(
            status_code=400,
            detail="No face encoding found in session data. Please upload a valid ID with a recognizable face.",
        )

    try:
        # Pass the reference face encoding to verify_liveness
        result = gesture_service.verify_liveness(reference_face_encoding)
        logger.info(f"Gesture verification completed for session: {session_id}")
        return result
    except Exception as e:
        logger.error(
            f"Error during gesture verification for session {session_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup-gesture/{session_id}")
async def cleanup_gesture_session(session_id: str):
    logger.info(f"Starting cleanup for session: {session_id}")
    if session_id not in sessions:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    logger.info(f"Session cleaned up: {session_id}")
    return {"status": "cleaned"}
