# main.py
from fastapi import FastAPI
from loguru import logger
from routers.id_router import router as api_router
from routers.session_router import router as session_router
from unified_logging.logging_setup import setup_logging

# Set up unified logging
setup_logging()

app = FastAPI(
    title="Onboardly API",
    description="API for ID processing, face comparison, and gesture verification",
    version="1.0",
)

# Include routers
app.include_router(api_router, prefix="/api", tags=["api"])
app.include_router(session_router, prefix="/session", tags=["session"])


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Auto-reload for development
        log_level="info",
    )
