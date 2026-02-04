"""Entry point for Render Python service.

This thin wrapper exposes the FastAPI `app` from api/main.py
as `app` at the repository root so that a start command like
`uvicorn main:app` works regardless of the working directory.
"""

from api.main import app  # re-export for uvicorn
