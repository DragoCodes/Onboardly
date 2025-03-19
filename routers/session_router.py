# routers/session_router.py
import os
import random
import sys
import time
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import APIRouter, HTTPException, Response
from loguru import logger  # Unified logging with loguru
from unified_logging.logging_setup import setup_logging  # Centralized setup

# Set up unified logging
setup_logging()

router = APIRouter()
sessions: Dict[str, dict] = {}


@router.post("/create-session")
async def create_session(response: Response):
    logger.info("Starting session creation")
    session_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
    sequence = [random.randint(1, 5) for _ in range(4)]
    sessions[session_id] = {
        "expected_sequence": sequence,
        "created_at": time.time(),
        "data": {},
    }

    # Set the session_id as a cookie
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    logger.info(f"Session created: {session_id} with sequence {sequence}")

    return {"session_id": session_id, "sequence": sequence}


@router.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    logger.info(f"Starting cleanup for session: {session_id}")
    if session_id not in sessions:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    logger.info(f"Session cleaned up: {session_id}")
    return {"status": "cleaned"}
