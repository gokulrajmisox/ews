"""Compatibility wrapper; use ``backend.main:app`` for the supported app."""

from backend.main import app

__all__ = ["app"]


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
