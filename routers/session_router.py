from fastapi import APIRouter, HTTPException, Response
from typing import Dict
import time
import random

router = APIRouter()
sessions: Dict[str, dict] = {}

@router.post("/create-session")
async def create_session(response: Response):
    session_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
    sequence = [random.randint(1, 5) for _ in range(4)]
    sessions[session_id] = {
        "expected_sequence": sequence,
        "created_at": time.time(),
        "data": {}
    }
    
    # Set the session_id as a cookie
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    
    return {
        "session_id": session_id,
        "sequence": sequence
    }


@router.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"status": "cleaned"}