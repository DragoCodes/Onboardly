# Customer Onboardly Installation Guide and Overview

Welcome to the **Onboardly** project documentation. This guide covers how to run the project, access the API endpoints, and evaluate the system. The project automates complex banking operations, allowing customers to onboard remotely with robust fraud prevention checks including OCR, face extraction, and gesture-based liveness verification.

---

## How to Run the Project

### Prerequisites

1. **Install Python 3.12**  
   Ensure that Python 3.12 is installed on your system. Download it from the official [Python website](https://www.python.org/downloads/).

2. **Install `just`**  
   `just` is a command runner that simplifies task execution. Install it using:
   ```bash
   sudo apt install just
   ```

3. **Install `uv`**  
   `uv` is a package installer and environment manager. Install it by running:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Step 1: Set Up the Environment

Initialize your environment and install all dependencies with:

```bash
just setup
```

This command will:
- Set up your Python environment.
- Install the required dependencies.

### Step 2: Start the Servers

To start all servers simultaneously, execute:

```bash
just start-all
```

This command will launch all servers in the background and log their outputs to:
- `logging_server.log`
- `backend_server.log`
- `frontend_server.log`

To stop all servers, use:

```bash
just stop-all
```

Alternatively, you can start each server individually in separate terminals:

```bash
# Start the Logging Server
just start-logging-server

# Start the FastAPI Application
just start-backend-server

# Start the Frontend Server
just start-frontend-server
```

### Accessing the Frontend

The frontend can be accessed at:
http://localhost:7860/

### Documentation

View the project's documentation at:
http://127.0.0.1:8000/

### Step 3: Access the API

Open your browser and navigate to:

http://127.0.0.1:3000/docs

This URL opens the Swagger UI where you can interact with all the API endpoints.

## API Endpoints

The following endpoints are available for testing:

### Session Management

- `/api/sessions/create-session`
  Creates a new session. Returns a session_id and a randomly generated sequence for verification. The session ID is set as a cookie.

- `/api/sessions/cleanup`
  Cleans up a session. The session ID is automatically retrieved from the cookie.

### ID Processing and Verification

- `/api/id/upload-id`
  Uploads an ID card (image or PDF) for OCR processing and face extraction. The extracted data (OCR result and face) are stored in the session.

- `/api/id/test-upload-id`
  A test endpoint that processes a pre-stored test ID image for OCR and face extraction.

- `/api/id/capture-and-compare`
  Compares a face extracted from a stored test image (simulating a webcam capture) with the reference face from the uploaded ID.
  Note: This endpoint requires that an ID has been successfully uploaded.

- `/api/id/verify-gesture`
  Verifies the user's gesture (for a liveness check) by comparing hand gesture data against the expected sequence. It uses the reference face from the uploaded ID.

- `/api/id/verify-gesture-from-stored-video`
  Processes a pre-stored video to verify that the user performed the correct gesture sequence and conducts a liveness check through face matching.


### Step 4: Test the Endpoints

Use the Swagger UI (at http://127.0.0.1:3000/docs) to interact with the API:

1. **Create a Session**
   - Call `/api/sessions/create-session` to generate a new session.
   - The response will include a session_id and a random sequence which will be stored via a cookie.

2. **Upload an ID Document**
   - Use `/api/id/upload-id` (or `/api/id/test-upload-id` for a test image) to upload your ID card.
   - This step performs OCR, extracts the face, and stores the result in the session.

3. **Capture and Compare**
   - Call `/api/id/capture-and-compare` to simulate capturing a selfie and comparing it with the ID's face.
   - Ensure that the ID has been successfully uploaded before invoking this endpoint.

4. **Verify Gesture**
   - Use `/api/id/verify-gesture` or `/api/id/verify-gesture-from-stored-video` to verify the user's gesture and perform a liveness check.
   - Make sure the ID upload step is complete before calling these endpoints.

5. **Cleanup**
   - Finally, call `/api/sessions/cleanup` to delete the session when testing is complete.


## How to Evaluate the Project

### 1. Test with Sample Inputs

- **Upload a valid ID**: Verify that OCR and face extraction work correctly.
- **Capture and Compare**: Ensure that the similarity score is reasonable when comparing the ID face with a test image.
- **Verify Gesture**: Confirm that the gesture recognition and liveness check pass based on the expected gesture sequence.

### 2. Check Logs

- Review the log files (`logging_server.log`, `backend_server.log`, and `frontend_server.log`) for any errors or warnings.

### 3. Evaluate Performance

- Measure processing times for each endpoint.
- Verify that error handling is robust (e.g., for missing or invalid files).

### 4. Test Error Handling

- Attempt to call endpoints out of sequence (e.g., capture and compare before uploading an ID) and verify that appropriate error messages are returned.