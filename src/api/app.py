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
        services: Optional pre-configured service container. If None, services
                  must be injected later via app.state.services before handling
                  requests (the lifespan handler does this automatically).

    Returns:
        Configured FastAPI application

    Note:
        For testing, use fixtures from tests/fakes.py.
        For production, the lifespan handler in main.py injects real services.
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

    # Store services in app state for dependency injection (may be None initially)
    # The lifespan handler will inject real services before requests are processed
    if services is not None:
        app.state.services = services
    else:
        # Initialize as None - lifespan will populate before first request
        app.state.services = None

    # Import routes here to avoid circular imports
    from src.api.routes import get_services, router

    # Override the dependency to return our services
    def _get_services() -> ServiceContainer:
        if app.state.services is None:
            raise RuntimeError(
                "Services not initialized. This should not happen in production - "
                "the lifespan handler should have injected services at startup."
            )
        return app.state.services

    app.dependency_overrides[get_services] = _get_services

    # Include routes
    app.include_router(router)

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
