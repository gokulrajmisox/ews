"""Vercel entrypoint for the existing FastAPI application."""
try:
    from backend.main import app
except Exception as exc:
    from fastapi import FastAPI
    app = FastAPI()
    _startup_error = f"{type(exc).__name__}: {exc}"

    @app.get("/{path:path}")
    def startup_error(path: str):
        return {"startup_error": _startup_error}

# Vercel discovers this ASGI app as the Python function handler.
