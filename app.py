"""
Agentic AI eBook RAG - FastAPI Application & Swagger UI Entrypoint.

Starts the FastAPI server with Swagger documentation (/docs).
Usage:
  python app.py
  uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import uvicorn
from api import app

__all__ = ["app"]

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Starting Agentic AI eBook RAG API Server...")
    print("Swagger Interactive Docs: http://127.0.0.1:8000/docs")
    print("Alternative Redoc Docs:   http://127.0.0.1:8000/redoc")
    print("=" * 70 + "\n")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
