import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Guarantee 'backend' directory is in sys.path regardless of execution CWD
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from app.core.config import settings, BASE_DIR
from app.db.session import engine, Base
from app.api import auth, recommend, history, watchlist

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize Database Tables
    print("[Startup] Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    print("[Startup] Moofy Backend is ready and listening on port!")
    yield
    print("[Shutdown] Cleaning up resources...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Intelligent emotion-aware movie recommendation platform backend",
    lifespan=lifespan
)

# Configure CORS
cors_env = os.getenv("CORS_ORIGINS", "*")
cors_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(recommend.router, prefix=settings.API_V1_STR)
app.include_router(history.router, prefix=settings.API_V1_STR)
app.include_router(watchlist.router, prefix=settings.API_V1_STR)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Moofy API", "version": settings.VERSION}

@app.get("/.streamlit/secrets.toml")
@app.get("/_stcore/health")
@app.get("/_stcore/host-config")
def streamlit_health_check():
    return {"status": "ok"}

# Optional: Serve production frontend build if dist folder exists
frontend_dist = BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
