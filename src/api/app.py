"""
FastAPI application factory for semantic search service.

Creates and configures the FastAPI application with routes and dependencies.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import (
    ServiceConfig,
    ServiceContainer,
)


def create_app(
    config: ServiceConfig | None = None,
    services: ServiceContainer | None = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Optional service configuration
        services: Optional pre-configured service container (required for production)

    Returns:
        Configured FastAPI application

    Raises:
        ValueError: If services is None in production (no default fakes)
    """
    app = FastAPI(
        title="Semantic Search Service",
        description="Hybrid vector + graph search API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Set up service container - require explicit services (no default fakes)
    if services is None:
        raise ValueError(
            "ServiceContainer is required. "
            "For testing, use fixtures from tests/fakes.py. "
            "For production, provide real service implementations."
        )

    # Store services in app state for dependency injection
    app.state.services = services

    # Import routes here to avoid circular imports
    from src.api.routes import get_services, router

    # Override the dependency to return our services
    def _get_services() -> ServiceContainer:
        return app.state.services

    app.dependency_overrides[get_services] = _get_services

    # Include routes
    app.include_router(router)

    return app

    return app


def configure_app_services(app: FastAPI, services: ServiceContainer) -> None:
    """
    Configure services for an existing app.

    This allows reconfiguring services after app creation,
    useful for testing.

    Args:
        app: FastAPI application instance
        services: Service container to use
    """
    from src.api.routes import get_services

    app.state.services = services

    def _get_services() -> ServiceContainer:
        return app.state.services

    app.dependency_overrides[get_services] = _get_services
