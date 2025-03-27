"""Load testing module using Locust.

This module defines the user behavior for load testing the API.
"""

from locust import HttpUser, between, task

# Define a constant for HTTP 200 OK status
HTTP_STATUS_OK = 200


class LoadTestUser(HttpUser):
    """User behavior for load testing the API endpoints."""

    wait_time = between(1, 3)  # Simulate user wait time between requests

    def on_start(self) -> None:
        """Create a session when a user starts the test and upload the ID."""
        # Create a session
        response = self.client.post("/api/sessions/create-session")
        if response.status_code == HTTP_STATUS_OK:
            self.session_id = response.json()["session_id"]
            self.client.cookies.set("session_id", self.session_id)
        else:
            self.session_id = None

        # Ensure the ID is uploaded for this session
        if self.session_id:
            upload_response = self.client.post("/api/id/test-upload-id")
            if upload_response.status_code == HTTP_STATUS_OK:
                self.id_uploaded = True
            else:
                # You may decide to retry or mark as failed
                self.id_uploaded = False
        else:
            self.id_uploaded = False

    @task(3)
    def test_upload_id(self) -> None:
        """Test uploading an ID using a stored test image."""
        # Optionally, re-upload if needed
        if self.session_id:
            response = self.client.post("/api/id/test-upload-id")
            if response.status_code == HTTP_STATUS_OK:
                self.id_uploaded = True

    @task(2)
    def test_capture_and_compare(self) -> None:
        """Test capturing and comparing face with stored image."""
        # Run this only if the ID has been successfully uploaded
        if self.session_id and self.id_uploaded:
            self.client.post("/api/id/test-capture-and-compare")
        elif self.session_id:
            # Re-upload ID if not done yet
            self.client.post("/api/id/test-upload-id")
            self.id_uploaded = True

    @task(2)
    def verify_gesture(self) -> None:
        """Test verifying gestures from a stored video."""
        # Run this only if the ID has been successfully uploaded
        if self.session_id and self.id_uploaded:
            self.client.post("/api/id/verify-gesture-from-stored-video")
        elif self.session_id:
            # Re-upload ID if not done yet
            self.client.post("/api/id/test-upload-id")
            self.id_uploaded = True

    @task(1)
    def cleanup_session(self) -> None:
        """Test session cleanup."""
        if self.session_id:
            # Now calls the cleanup endpoint that uses the cookie for session ID
            self.client.delete("/api/sessions/cleanup")
            self.session_id = None
            self.id_uploaded = False
