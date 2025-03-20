"""Session router module for managing user sessions via API endpoints."""

import random
import sys
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Cookie, HTTPException, Response
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from unified_logging.logging_setup import setup_logging

# Set up unified logging
setup_logging()

router = APIRouter()
sessions: dict[str, dict] = {}


@router.post("/create-session")
async def create_session(response: Response) -> dict[str, str]:
    """Create a new session and set its ID as a cookie.

    Args:
        response (Response): The FastAPI response object to set the cookie.

    Returns:
        dict[str, str]: A dictionary containing the new session ID.

    """
    logger.info("Starting session creation")
    session_id = f"{int(time.time())}-{random.randint(1000, 9999)}"  # noqa: S311
    sessions[session_id] = {
        "created_at": time.time(),
        "data": {},
    }

    # Set the session_id as a cookie
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    logger.info(f"Session created: {session_id}")

    return {"session_id": session_id}


@router.delete("/cleanup")
async def cleanup_session(session_id: Annotated[str, Cookie()]=None) -> dict[str, str]:
    """Delete a session identified by its session ID.

    Args:
        session_id (str, optional): The session ID from the cookie. Defaults to None.

    Returns:
        dict[str, str]: A status message confirming the session was cleaned.

    """
    logger.info(f"Starting cleanup for session: {session_id}")
    if not session_id or session_id not in sessions:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    logger.info(f"Session cleaned up: {session_id}")
    return {"status": "cleaned"}
