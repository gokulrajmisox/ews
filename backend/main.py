"""Supported FastAPI application entry point for SilentWindow."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.router import router as api_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_VERSION = os.getenv("SILENTWINDOW_VERSION", "1.1.0")


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = FastAPI(
    title="SilentWindow - Clinical Decision Support Prototype",
    description="Retrospective, trust-aware early-warning research system.",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(api_router)


@app.get("/health", tags=["system"])
def health() -> dict:
    """Return a cheap liveness/readiness signal without loading model artifacts."""
    return {
        "status": "ok",
        "service": "silentwindow",
        "version": APP_VERSION,
        "mode": "retrospective-with-incremental-simulation",
    }


@app.get("/version", tags=["system"])
def version() -> dict:
    """Return runtime and artifact metadata for reproducible demos."""
    return {
        "service": "silentwindow",
        "version": APP_VERSION,
        "model_artifact": "models/xgboost_model.joblib",
        "calibrator_artifact": "models/calibrator.joblib",
        "config": "configs/config.yaml",
        "disclaimer": "Research prototype; not for clinical decision-making.",
    }


frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "SilentWindow API is running; frontend index.html not found."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
