"""Frontend module for the Onboardly ID Verification System.

This module provides a Streamlit-based interface for interacting with the 
BentoML backend to perform ID
verification, face comparison, and gesture verification.
"""

import tempfile
from pathlib import Path

import requests
import streamlit as st

# Constants
HTTP_OK: int = 200  # HTTP status code for success
GESTURE_STEP: int = 2  # Step number for gesture verification
BACKEND_URL: str = "http://localhost:3000"  # Default BentoML serving port
REQUEST_TIMEOUT: float = 10.0  # Timeout for HTTP requests in seconds


def create_session() -> str | None:
    """Create a new session by calling the backend API.

    Returns:
        str | None: The session ID if successful, None otherwise.

    """
    result = None
    try:
        response = requests.post(
            f"{BACKEND_URL}/get_session_id", timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == HTTP_OK:
            result = response.text.strip('"')  # Remove quotes from the response
        else:
            st.error(
                f"Failed to create session: {response.status_code} - {response.text}",
            )
            result = None
    except requests.RequestException as e:
        st.error(f"Error connecting to backend: {e!s}")
        result = None
    return result


def process_id_file(file_content: bytes) -> dict:
    """Upload and process an ID card image using the BentoML service.

    Args:
        file_content (bytes): The binary content of the uploaded ID file.

    Returns:
        dict: Response from the backend or an error dict if the request fails.

    """
    result = None
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name

        # Send the file path to the BentoML service
        response = requests.post(
            f"{BACKEND_URL}/process_id_file",
            json={"id_file_path": tmp_file_path},
            timeout=REQUEST_TIMEOUT,
        )

        # Remove temporary file
        Path(tmp_file_path).unlink()

        if response.status_code == HTTP_OK:
            result = response.json()
        else:
            st.error(f"Failed to process ID: {response.status_code} - {response.text}")
            result = {"error": f"HTTP {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        st.error(f"Error processing ID: {e!s}")
        result = {"error": str(e)}
    return result


def capture_and_compare() -> dict:
    """Capture a selfie and compare it with the ID face using the backend API.

    Returns:
        dict: Response from the backend or an error dict if the request fails.

    """
    result = None
    try:
        response = requests.post(
            f"{BACKEND_URL}/capture_and_compare", timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == HTTP_OK:
            result = response.json()
        else:
            st.error(
                f"Failed to capture and compare: {response.status_code} - "
                f"{response.text}",
            )
            result = {"error": f"HTTP {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        st.error(f"Error in capture and compare: {e!s}")
        result = {"error": str(e)}
    return result


def verify_gesture() -> dict:
    """Verify the user's gesture using the backend API.

    Returns:
        dict: Response from the backend or an error dict if the request fails.

    """
    result = None
    try:
        response = requests.post(
            f"{BACKEND_URL}/verify_gesture", timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == HTTP_OK:
            result = response.json()
        else:
            st.error(
                f"Failed to verify gesture: {response.status_code} - {response.text}",
            )
            result = {"error": f"HTTP {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        st.error(f"Error in gesture verification: {e!s}")
        result = {"error": str(e)}
    return result


# App title
st.title("ID Verification System")

# Create a session if not already created
if "session_id" not in st.session_state:
    # Show BentoML service information
    st.info("Connecting to BentoML service at " + BACKEND_URL)
    st.info(
        "If you encounter errors, make sure the BentoML service is running correctly",
    )

    # For now, set a placeholder session ID
    # Since your service seems to maintain its own session internally
    st.session_state.session_id = "bentoml-internal-session"
    st.success("Connected to verification service")

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
st.sidebar.title("Session Information")
st.sidebar.markdown(f"**Session ID:** `{st.session_state.session_id}`")
st.sidebar.markdown(f"**Service URL:** `{BACKEND_URL}`")
st.sidebar.markdown(f"**Current Step:** {steps[st.session_state.current_step]}")

# Additional important information
st.sidebar.title("Verification Data")
for key, value in st.session_state.important_info.items():
    if isinstance(value, dict):
        st.sidebar.write(f"**{key}:**")
        for sub_key, sub_value in value.items():
            st.sidebar.write(f"- **{sub_key.title()}:** {sub_value}")
    else:
        st.sidebar.write(f"**{key}:** {value}")

# Display current step
st.header(steps[st.session_state.current_step])

if st.session_state.current_step == 0:
    # Upload ID Card
    id_file = st.file_uploader("Choose an ID card image", type=["png", "jpg", "jpeg"])
    if id_file:
        file_content = id_file.getvalue()

        with st.spinner("Processing ID card..."):
            id_result = process_id_file(file_content)

        if id_result and "error" not in id_result:
            st.success("ID processed successfully")

            # Extract and format OCR Result
            ocr_data = id_result.get("ocr_data", {})
            structured_data = ocr_data.get("structured_data", {})

            # Display structured data nicely
            if structured_data:
                st.subheader("Extracted Information")
                for key, value in structured_data.items():
                    st.write(f"**{key.title()}:** {value}")

                st.session_state.important_info["ID Data"] = structured_data

            # Display face detection result
            face_data = id_result.get("face_data", {})
            if face_data.get("face_detected", False):
                st.write("✅ Face detected in ID")
                st.session_state.important_info["Face Detected"] = "Yes"
            else:
                st.write("❌ No face detected in ID")
                st.session_state.important_info["Face Detected"] = "No"

            # Store the session ID from the response if available
            if "session_id" in id_result:
                st.session_state.session_id = id_result["session_id"]
                st.sidebar.success("Session ID updated")

            st.session_state.current_step += 1  # Move to next step
        elif id_result and "error" in id_result:
            st.error(f"Error processing ID: {id_result['error']}")

if st.session_state.current_step == 1:
    # Capture Selfie
    st.info(
        "This step will use your webcam to capture a selfie and compare it with the "
        "ID photo",
    )
    st.write("Please position your face clearly in front of the camera")

    if st.button("Start Face Verification"):
        with st.spinner("Capturing and comparing faces..."):
            compare_result = capture_and_compare()

        if compare_result and "error" not in compare_result:
            match_score = compare_result.get("match_score", 0)
            is_match = compare_result.get("match", False)

            if is_match:
                st.success(f"Face match successful! Match score: {match_score:.2f}")
            else:
                st.error(f"Face match failed. Match score: {match_score:.2f}")

            # Store results in session state
            st.session_state.important_info["Face Match"] = "Yes" if is_match else "No"
            st.session_state.important_info["Match Score"] = f"{match_score:.2f}"
            st.session_state.important_info["Face Comparison Results"] = compare_result

            # Move to next step if match is successful
            if is_match:
                st.session_state.current_step += 1
        elif compare_result and "error" in compare_result:
            st.error(f"Error comparing faces: {compare_result['error']}")

if st.session_state.current_step == GESTURE_STEP:
    # Verify Gesture
    st.info("This step will verify your identity through gesture recognition")
    st.write(
        "You will be prompted to perform specific gestures to confirm that you are a "
        "real person",
    )

    if st.button("Start Gesture Verification"):
        with st.spinner("Verifying gesture..."):
            gesture_result = verify_gesture()

        if gesture_result and "error" not in gesture_result:
            success = gesture_result.get("success", False)
            overall_face_match = gesture_result.get("overall_face_match", False)

            if success and overall_face_match:
                st.success("✅ Gesture verification successful!")
                st.session_state.important_info["Gesture Verification"] = "Passed"
                st.session_state.important_info["Gesture Results"] = gesture_result
                st.session_state.current_step += 1
                st.rerun()
            else:
                st.error("❌ Gesture verification failed")
                st.session_state.important_info["Gesture Verification"] = "Failed"
                if not success:
                    st.warning("The required gesture was not detected")
                if not overall_face_match:
                    st.warning("Face in gesture video does not match ID face")
        elif gesture_result and "error" in gesture_result:
            st.error(f"Error in gesture verification: {gesture_result['error']}")

# If all steps are completed
if st.session_state.current_step >= len(steps):
    st.header("Verification Complete")
    st.balloons()

    # Calculate overall success
    ocr_success = "ID Data" in st.session_state.important_info
    face_success = st.session_state.important_info.get("Face Match") == "Yes"
    gesture_success = (
        st.session_state.important_info.get("Gesture Verification") == "Passed"
    )

    overall_success = ocr_success and face_success and gesture_success

    # Display verification status
    if overall_success:
        st.success("✅ All verification steps completed successfully!")
    else:
        st.warning("⚠️ Verification process completed with some issues")

    # Summary
    st.subheader("Verification Summary")
    summary = {
        "session_id": st.session_state.session_id,
        "ocr_processing": ocr_success,
        "face_verification": face_success,
        "gesture_verification": gesture_success,
        "overall_success": overall_success,
    }
    st.write(f"**Session ID:** {summary['session_id']}")
    st.write(
        f"**OCR Processing:** {'Success' if summary['ocr_processing'] else 'Failed'}",
    )
    st.write(
        f"**Face Verification:** "
        f"{'Success' if summary['face_verification'] else 'Failed'}",
    )
    st.write(
        f"**Gesture Verification:** "
        f"{'Success' if summary['gesture_verification'] else 'Failed'}",
    )
    st.write(f"**Overall Success:** {'Yes' if summary['overall_success'] else 'No'}")

    # Reset button
    if st.button("Start New Verification"):
        # Reset the session state
        for key in list(st.session_state.keys()):
            if key != "session_id":  # Keep the session ID
                del st.session_state[key]
        st.session_state.current_step = 0
        st.session_state.important_info = {}
        st.rerun()

# Reset Session button
if st.sidebar.button("Reset Session"):
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
