"""
Main entry point for semantic-search-service.

Creates the FastAPI application instance for uvicorn.
"""

from src.api.app import create_app

# Create application instance
app = create_app()
