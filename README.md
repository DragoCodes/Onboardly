# Face Authentication System

A biometric verification system that authenticates users through facial recognition, document verification, and liveness checks.

## Overview

This system provides a comprehensive solution for remote user verification, particularly useful for banking operations like account opening and pension verification. It combines:

- Document verification (ID cards, licenses, etc.)
- Facial recognition
- Liveness detection through hand gesture verification

## Project Structure

```
face_auth_system/
├── .github/
│   ├── workflows/      # CI/CD (testing & linting)
│   └── CODEOWNERS
├── app/
│   ├── api/            # FastAPI endpoints
│   │   ├── routers/    # auth_router.py, verification_router.py
│   │   └── client.py   # API client for service switching
│   ├── gui/            # Gradio/PyQt UI
│   │   ├── webcam.py   # Webcam capture logic
│   │   ├── upload.py   # File upload handlers
│   │   └── video_analysis.py
│   ├── processing/     # Core CV logic
│   │   ├── face/       # face_comparison.py, face_extraction.py
│   │   ├── ocr/        # document_ocr.py
│   │   ├── gestures/   # mediapipe_hands.py
│   │   └── video/      # frame_extraction.py, liveness_check.py
│   ├── services/       # Model serving
│   │   ├── bento_service.py
│   │   ├── fastapi_service.py
│   │   └── service_base.py  # Abstract base class
│   ├── core/
│   │   ├── config.py   # Pydantic settings
│   │   └── logging.py  # Unified logging setup
│   └── schemas/        # Pydantic models
│       ├── auth.py     # Input validation
│       └── responses.py  # Output formats
├── configs/
│   ├── base.yaml       # Common settings
│   ├── fastapi.yaml    # API-specific config
│   └── local.yaml      # Local development
├── data/
│   ├── raw/            # User uploads
│   ├── processed/      # Extracted faces/text
│   └── temp/           # Video recordings
├── docs/
│   ├── docs/           # Mkdocs content
│   └── load_testing/   # Locust results/analysis
├── load_testing/
│   └── locustfile.py   # Load test scenarios
├── logs/               # Rotating log files
├── models/             # Serialized models
│   └── bentoml/        # BentoML artifacts
├── scripts/
│   ├── setup_env.sh    # Environment setup
│   └── start_service.sh  # Service launcher
├── tests/
│   ├── unit/           # Component tests
│   └── integration/    # End-to-end flows
├── .env                # Environment vars
├── .gitignore
├── .ruff.toml          # Linting rules
├── Dockerfile
├── Justfile            # Task runner
├── mkdocs.yml          # Documentation config
├── pyproject.toml      # PEP 621 config
└── README.md
```

