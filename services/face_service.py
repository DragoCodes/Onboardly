import face_recognition
import numpy as np
import logging
from pathlib import Path
import base64
from PIL import Image
import io
import cv2 
import tempfile
import os

logger = logging.getLogger(__name__)

class FaceService:
    def extract_face(self, image_path: str):
        try:
            absolute_path = Path(image_path).absolute()
            
            if not absolute_path.exists():
                raise FileNotFoundError(f"Face image not found: {absolute_path}")
            
            image = face_recognition.load_image_file(absolute_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return None
                
            top, right, bottom, left = face_locations[0]
            
            # Extract face image
            face_image = image[top:bottom, left:right]
            
            # Convert to base64
            pil_image = Image.fromarray(face_image)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
            
            return {
                "face_encoding": face_encoding.tolist(),
                "face_location": face_locations[0],
                "face_image": img_str,
                "mime_type": "image/jpeg"
            }
        except Exception as e:
            logger.error(f"Face processing error in {image_path}: {str(e)}")
            return {"error": str(e)}

    def compare_faces(self, known_enc, unknown_enc):
        try:
            known_array = np.array(known_enc)
            unknown_array = np.array(unknown_enc)
            face_distance = face_recognition.face_distance([known_array], unknown_array)[0]
            similarity = (1.0 - face_distance) * 100

            # Convert numpy.bool_ to native Python bool
            match = bool(similarity > 70)

            return {
                "similarity": round(float(similarity), 2),  # Convert to native Python float
                "match": match  # Convert to native Python bool
            }
        except Exception as e:
            logger.error(f"Comparison Error: {str(e)}")
            return {"error": str(e)}
        
    def capture_and_compare(self, session_id: str, sessions: dict):
        """
        Capture an image from the webcam, extract the face, and compare it with the face from the ID card.
        """
        if session_id not in sessions:
            raise ValueError("Invalid session")
        if "id" not in sessions[session_id]["data"] or "face" not in sessions[session_id]["data"]["id"]:
            raise ValueError("ID not uploaded or face not detected in ID")

        # Capture image from webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Could not capture image from webcam")
        cap.release()

        # Create a directory to save webcam images
        webcam_images_dir = Path("uploads/webcam_images")
        webcam_images_dir.mkdir(parents=True, exist_ok=True)

        # Save the captured image
        webcam_image_path = webcam_images_dir / f"webcam_capture_{session_id}.jpg"
        cv2.imwrite(str(webcam_image_path), frame)

        try:
            # Extract face from webcam image
            webcam_face_result = self.extract_face(str(webcam_image_path))
            if not webcam_face_result or "face_encoding" not in webcam_face_result:
                raise ValueError("No face detected in webcam image")

            # Get face encoding from ID card
            id_face_result = sessions[session_id]["data"]["id"]["face"]
            if not id_face_result or "face_encoding" not in id_face_result:
                raise ValueError("No face encoding found in ID card data")

            # Compare faces
            comparison_result = self.compare_faces(id_face_result["face_encoding"], webcam_face_result["face_encoding"])

            return {
                "similarity": comparison_result.get("similarity"),
                "match": comparison_result.get("match"),
                "webcam_face_image": webcam_face_result.get("face_image"),
                "id_face_image": id_face_result.get("face_image"),
                "webcam_image_path": str(webcam_image_path)  # Return the path of the saved webcam image
            }
        except Exception as e:
            # Cleanup on error
            if webcam_image_path.exists():
                os.remove(webcam_image_path)
            raise e