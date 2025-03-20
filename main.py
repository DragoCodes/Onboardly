import os
import toml
from fastapi import FastAPI
from loguru import logger
import uvicorn

from routers.id_router import router as api_router
from routers.session_router import router as session_router
from unified_logging.logging_setup import setup_logging

# Set up unified logging
setup_logging()

# Load configuration from config.toml
config_path = os.path.join(os.path.dirname(__file__), "config.toml")
config = toml.load(config_path)
fastapi_config = config.get("fastapi", {})

# Create FastAPI instance with settings from config
app = FastAPI(
    title=fastapi_config.get("title", "Onboardly API"),
    description=fastapi_config.get("description", "API for ID processing, face comparison, and gesture verification"),
    version=fastapi_config.get("version", "1.0"),
)

# Include routers with their respective prefixes
app.include_router(api_router, prefix="/api/id")
app.include_router(session_router, prefix="/api/sessions")


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=fastapi_config.get("host", "127.0.0.1"),
        port=fastapi_config.get("port", 8000),
        reload=fastapi_config.get("reload", True),
        log_level=fastapi_config.get("log_level", "info"),
    )
