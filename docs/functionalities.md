# Onboardly Installation Guide

Welcome to the **Onboardly** project. This guide will help you set up and run this banking automation system that enables remote customer onboarding with advanced fraud prevention features.

## Prerequisites

Before getting started, ensure you have:

1. **Python 3.12** installed
   - Download from the [official Python website](https://www.python.org/downloads/)

2. **`just`** command runner
   - Install via:
     ```bash
     sudo apt install just
     ```

3. **`uv`** package installer and environment manager
   - Install via:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

## Installation Steps

### Step 1: Set Up the Environment

Initialize your environment and install all required dependencies:

```bash
just setup
```

This command will set up your Python environment and install all necessary dependencies.

### Step 2: Start the Servers

#### Start All Servers Simultaneously

```bash
just start-all
```

This launches all servers in the background with logs written to:
- `logging_server.log`
- `backend_server.log`
- `backend_server_bento.log`
- `frontend_server.log`

To stop all servers:
```bash
just stop-all
```

#### Start Individual Servers

You can also start each server separately in different terminals:

```bash
# Start the Logging Server
just start-logging-server

# Start the FastAPI Application
just start-backend-server-fastapi

# Start the BentoML Server
just start-backend-server-bentoml

# Start the Frontend Server
just start-frontend-server
```

### Running the BentoML Server Independently

To start just the BentoML server:

```bash
just start-backend-server-bentoml
```

Or serve the BentoML project directly:

```bash
just serve_bm
```

## Accessing the System

- **Frontend**: http://localhost:7860/
- **API Documentation**: http://127.0.0.1:8000/
- **API Endpoints (Swagger UI)**: http://127.0.0.1:3000/docs

## API Endpoints Overview

### Session Management
- `/api/sessions/create-session`: Creates a new session
- `/api/sessions/cleanup`: Cleans up a session

### ID Processing and Verification
- `/api/id/upload-id`: Uploads and processes an ID card
- `/api/id/test-upload-id`: Test endpoint with pre-stored ID image
- `/api/id/capture-and-compare`: Compares captured face with ID face
- `/api/id/verify-gesture`: Verifies user's gesture for liveness check
- `/api/id/verify-gesture-from-stored-video`: Processes pre-stored video for gesture verification

## Testing the System

1. Create a session using `/api/sessions/create-session`
2. Upload an ID document via `/api/id/upload-id` or `/api/id/test-upload-id`
3. Capture and compare faces using `/api/id/capture-and-compare`
4. Verify gesture with `/api/id/verify-gesture` or `/api/id/verify-gesture-from-stored-video`
5. Clean up the session when finished with `/api/sessions/cleanup`

## Evaluation Methods

### 1. Sample Input Testing
- Test with valid IDs to verify OCR and face extraction
- Ensure reasonable similarity scores during face comparison
- Confirm proper gesture recognition and liveness checks

### 2. Log Review
- Check log files for errors or warnings:
  - `logging_server.log`
  - `backend_server.log`
  - `backend_server_bento.log`
  - `frontend_server.log`

### 3. Performance Evaluation
- Measure processing times for endpoints
- Verify robust error handling

### 4. Error Handling Tests
- Test calling endpoints out of sequence
- Verify appropriate error messages

### 5. Load Testing
Ensure the backend server is running, then execute:

```bash
just load_testing
```

This runs a Locust load test targeting the API. Access the Locust web interface at http://localhost:8089 to configure users and spawn rate.