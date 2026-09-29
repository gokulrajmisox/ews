"""Vercel entrypoint for the existing FastAPI application."""
from backend.main import app

# Vercel discovers this ASGI app as the Python function handler.
