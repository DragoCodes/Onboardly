from fastapi import FastAPI
from routers import session_router, id_router
from config import settings

app = FastAPI(title=settings.APP_NAME)

# Include routers
app.include_router(session_router.router, prefix="/api/sessions")
app.include_router(id_router.router, prefix="/api/id")

@app.get("/")
async def serve_ui():
    return FileResponse("app/templates/index.html")