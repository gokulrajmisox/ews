"""Vercel entrypoint for the existing FastAPI application."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app

# Vercel discovers this ASGI app as the Python function handler.
