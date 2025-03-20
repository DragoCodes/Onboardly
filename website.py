"""Frontend module for the Onboardly ID Verification System using FastAPI backend.

This module provides a Streamlit-based interface for interacting with FastAPI backend
to perform ID verification, face comparison, and gesture verification.
"""

import time
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# Constants
BACKEND_URL: str = "http://localhost:8000"  # FastAPI backend URL
HTTP_OK: int = 200  # HTTP status code for success
GESTURE_STEP: int = 2  # Step number for gesture verification
REQUEST_TIMEOUT: float = 10.0  # Timeout for HTTP requests in seconds
MAX_RETRIES: int = 5  # Maximum number of retries for session creation
RETRY_DELAY: float = 2.0  # Delay between retries in seconds


def create_session() -> tuple[str, str]:
    """Create a new session by calling the backend API.

    Returns:
        Tuple[str, str]: A tuple containing the session ID and session cookie.

    Raises:
        ConnectionError: If the connection to the backend fails after all retries.

    """
    retries = MAX_RETRIES
    while retries > 0:
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/sessions/create-session",
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return (
                response.json()["session_id"],
                response.cookies.get("session_cookie"),
            )
        except requests.exceptions.ConnectionError:
            st.warning("Connection refused. Retrying...")
            retries -= 1
            time.sleep(RETRY_DELAY)  # Wait before retrying
    error_message = "Failed to establish a connection to the backend service."
    raise ConnectionError(error_message)


def upload_id(session_id: str, file: BytesIO) -> dict | None:
    """Upload an ID card image or PDF to the backend for processing.

    Args:
        session_id (str): The session ID for authentication.
        file (BytesIO): The file object containing the ID image or PDF.

    Returns:
        Optional[dict]: The response from the backend, or None if the request fails.

    """
    files = {"file": file}
    cookies = {"session_id": session_id}
    response = requests.post(
        f"{BACKEND_URL}/api/id/upload-id",
        files=files,
        cookies=cookies,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == HTTP_OK:
        return response.json()
    st.error("Failed to upload ID")
    return None


def capture_and_compare(session_id: str) -> dict | None:
    """Capture a selfie and compare it with the ID face using the backend API.

    Args:
        session_id (str): The session ID for authentication.

    Returns:
        Optional[dict]: The response from the backend, or None if the request fails.

    """
    cookies = {"session_id": session_id}
    response = requests.post(
        f"{BACKEND_URL}/api/id/capture-and-compare",
        cookies=cookies,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == HTTP_OK:
        return response.json()
    st.error("Failed to capture and compare")
    return None


def verify_gesture(session_id: str) -> dict | None:
    """Verify the user's gesture using the backend API.

    Args:
        session_id (str): The session ID for authentication.

    Returns:
        Optional[dict]: The response from the backend, or None if the request fails.

    """
    cookies = {"session_id": session_id}
    response = requests.post(
        f"{BACKEND_URL}/api/id/verify-gesture",
        cookies=cookies,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == HTTP_OK:
        return response.json()
    st.error("Failed to verify gesture")
    return None


# Create a session if not already created
if "session_id" not in st.session_state:
    session_id, session_cookie = create_session()
    if session_id:
        st.session_state.session_id = session_id
        st.session_state.session_cookie = session_cookie
        st.success(f"Session created: {session_id}")
    else:
        st.error("Session creation failed")
else:
    session_id = st.session_state.session_id
    session_cookie = st.session_state.session_cookie

# Initialize step completion flags
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

# Initialize important information storage
if "important_info" not in st.session_state:
    st.session_state.important_info = {}

# Steps
steps = ["Upload ID Card", "Capture Selfie", "Verify Gesture"]

# Ensure current_step is within bounds
if st.session_state.current_step < 0 or st.session_state.current_step >= len(steps):
    st.session_state.current_step = 0

# Progress bar
progress = st.progress(st.session_state.current_step / len(steps))

# Sidebar for important information
st.sidebar.title("Important Information")
for key, value in st.session_state.important_info.items():
    st.sidebar.write(f"**{key}:** {value}")

# Display current step
st.header(steps[st.session_state.current_step])

if st.session_state.current_step == 0:
    # Upload ID Card
    id_file = st.file_uploader(
        "Choose an ID card image or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
    )
    if id_file:
        id_result = upload_id(session_id, id_file)
        if id_result:
            st.success("ID uploaded successfully")

            # Extract and format OCR Result
            ocr_result = id_result["ocr_result"]
            full_text = ocr_result.get("full_text", "")
            structured_data = ocr_result.get("structured_data", {})
            name = structured_data.get("name", "N/A")
            dob = structured_data.get("dob", "N/A")

            formatted_ocr = f"Name: {name}\nDate of Birth: {dob}"
            st.write("OCR Result:", formatted_ocr)
            st.session_state.important_info["OCR Result"] = formatted_ocr

            if id_result["face_detected"]:
                st.write("Face detected in ID")
                st.session_state.important_info["Face Detected"] = "Yes"
            else:
                st.session_state.important_info["Face Detected"] = "No"
            st.session_state.current_step += 1
            st.rerun()

elif st.session_state.current_step == 1:
    # Capture Selfie
    selfie = st.camera_input("Take a selfie")
    if selfie:
        selfie_image = Image.open(selfie)
        buffered = BytesIO()
        selfie_image.save(buffered, format="JPEG")
        selfie_bytes = buffered.getvalue()

        # Send selfie for comparison
        compare_result = capture_and_compare(session_id)
        if compare_result:
            st.success("Selfie captured and compared successfully")
            st.write("Comparison Result:", compare_result)
            del compare_result["id_face_image"]
            st.session_state.important_info["Comparison Result"] = compare_result
            st.session_state.current_step += 1
            st.rerun()

elif st.session_state.current_step == GESTURE_STEP:
    # Verify Gesture
    if st.button("Verify Gesture"):
        gesture_result = verify_gesture(session_id)
        if gesture_result:
            st.success("Gesture verified successfully")
            st.write("Verification Result:", gesture_result)
            st.session_state.important_info["Verification Result"] = gesture_result
            st.session_state.current_step += 1
            st.rerun()

# End Session button on all pages
if st.button("End Session"):
    cookies = {"session_id": session_id}
    response = requests.delete(
        f"{BACKEND_URL}/api/sessions/cleanup",
        cookies=cookies,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == HTTP_OK:
        st.success("Session cleaned up")
        del st.session_state.session_id
        del st.session_state.session_cookie
        st.session_state.current_step = 0
        st.session_state.important_info = {}  # Clear important info
        st.rerun()
    else:
        st.error("Failed to clean up session")
