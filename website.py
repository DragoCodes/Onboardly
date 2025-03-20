from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

# Function to create a session
def create_session():
    response = requests.post(f"{BACKEND_URL}/api/sessions/create-session")
    if response.status_code == 200:
        return response.json()["session_id"], response.cookies.get("session_id")
    st.error("Failed to create session")
    return None, None

# Function to upload ID card
def upload_id(session_id, file):
    files = {"file": file}
    cookies = {"session_id": session_id}
    response = requests.post(f"{BACKEND_URL}/api/id/upload-id", files=files, cookies=cookies)
    if response.status_code == 200:
        return response.json()
    st.error("Failed to upload ID")
    return None

# Function to capture and compare selfie
def capture_and_compare(session_id):
    cookies = {"session_id": session_id}
    response = requests.post(f"{BACKEND_URL}/api/id/capture-and-compare", cookies=cookies)
    if response.status_code == 200:
        return response.json()
    st.error("Failed to capture and compare")
    return None

# Function to verify gesture
def verify_gesture(session_id):
    cookies = {"session_id": session_id}
    response = requests.post(f"{BACKEND_URL}/api/id/verify-gesture", cookies=cookies)
    if response.status_code == 200:
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
    id_file = st.file_uploader("Choose an ID card image or PDF", type=["png", "jpg", "jpeg", "pdf"])
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

elif st.session_state.current_step == 2:
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
    response = requests.delete(f"{BACKEND_URL}/api/sessions/cleanup", cookies=cookies)
    if response.status_code == 200:
        st.success("Session cleaned up")
        del st.session_state.session_id
        del st.session_state.session_cookie
        st.session_state.current_step = 0
        st.session_state.important_info = {}  # Clear important info
        # st.rerun()
    else:
        st.error("Failed to clean up session")
