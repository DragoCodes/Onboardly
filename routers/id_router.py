from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Cookie, Response
from services.ocr_service import OCRService
from services.face_service import FaceService
from services.gesture_service import GestureService
from config import settings
from pathlib import Path
import fitz  # PyMuPDF
import tempfile
import os
import magic
from typing import List
import base64
from PIL import Image
import io
import time
import random

# Import sessions from session_router
from routers.session_router import sessions

router = APIRouter()
ocr = OCRService()
face = FaceService()
gesture_service = GestureService()

def convert_pdf_to_images(pdf_path: Path) -> List[Path]:
    """Convert PDF to list of image paths"""
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap()
            temp_dir = tempfile.mkdtemp()
            img_path = Path(temp_dir) / f"page_{page.number}.png"
            pix.save(img_path)
            images.append(img_path)
        return images
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF conversion failed: {str(e)}")

async def validate_file(file: UploadFile):
    """Validate file type using python-magic"""
    allowed_types = ["image/png", "image/jpeg", "application/pdf"]
    
    # Read first 2048 bytes for MIME detection
    header = await file.read(2048)
    await file.seek(0)
    
    mime = magic.from_buffer(header, mime=True)
    if mime not in allowed_types:
        raise HTTPException(status_code=400, 
                          detail=f"Unsupported file type: {mime}. Allowed types: {', '.join(allowed_types)}")

@router.post("/upload-id")
async def upload_id(
    session_id: str = Cookie(None),
    file: UploadFile = File(...)
):
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid session")
    
    await validate_file(file)
    
    # Save file
    file_path = settings.UPLOAD_FOLDER.absolute() / f"{session_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        images = []
        # Handle PDF files
        if file.content_type == "application/pdf":
            images = convert_pdf_to_images(file_path)
            if not images:
                raise HTTPException(status_code=400, detail="No pages found in PDF")
            process_path = images[0]
        else:
            process_path = file_path
        
        # Process files
        ocr_result = ocr.extract_text(process_path)
        face_result = face.extract_face(process_path)
        
        if face_result and "error" in face_result:
            raise HTTPException(status_code=400, detail=f"Face extraction failed: {face_result['error']}")

        if not face_result or "face_encoding" not in face_result:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded ID")

        # Initialize id dictionary if it doesn't exist
        if "data" not in sessions[session_id]:
            sessions[session_id]["data"] = {}
            
        # Store results
        sessions[session_id]["data"]["id"] = {
            "ocr": ocr_result,
            "face": face_result  
        }

        if face_result and "face_image" in face_result:
            face_filename = f"face_{session_id}_{file.filename}.jpg"
            face_path = settings.UPLOAD_FOLDER.absolute() / face_filename
            with open(face_path, "wb") as f:
                f.write(base64.b64decode(face_result["face_image"]))
            sessions[session_id]["data"]["id"]["face_image_path"] = str(face_path)
        
        return {
            "ocr_result": ocr_result,
            "face_detected": face_result is not None,
            "face_image": face_result.get("face_image") if face_result else None,
            "image_format": face_result.get("mime_type") if face_result else None
        }
        
    except Exception as e:
        # Cleanup on error
        if file_path.exists():
            os.remove(file_path)
        for img in images:
            if img.exists():
                os.remove(img)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/capture-and-compare")
async def capture_and_compare(session_id: str = Cookie(None)):
    
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid session")
    
    try:
        result = face.capture_and_compare(session_id, sessions)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Remove the start-gesture-session endpoint since we'll use the session_router's create-session instead

@router.post("/verify-gesture")
async def verify_gesture(session_id: str = Cookie(None)):
    if not session_id or session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid session")
    
    # Check if data dict exists
    if "data" not in sessions[session_id]:
        raise HTTPException(status_code=400, detail="No data found in session. Please complete ID upload first.")
    
    # Get the reference face encoding from the session data
    if "id" not in sessions[session_id]["data"]:
        raise HTTPException(status_code=400, detail="ID document not uploaded. Please upload ID before verifying gesture.")
    
    if "face" not in sessions[session_id]["data"]["id"]:
        raise HTTPException(status_code=400, detail="No face data found in session. Please upload a valid ID with a face.")

    reference_face_encoding = sessions[session_id]["data"]["id"]["face"].get("face_encoding")
    if not reference_face_encoding:
        raise HTTPException(status_code=400, detail="No face encoding found in session data. Please upload a valid ID with a recognizable face.")
    
    try:
        # Pass the reference face encoding to verify_liveness
        result = gesture_service.verify_liveness(reference_face_encoding)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup-gesture/{session_id}")
async def cleanup_gesture_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"status": "cleaned"}