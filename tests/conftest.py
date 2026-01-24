"""
Pytest configuration and fixtures for semantic-search-service tests.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import ServiceConfig, ServiceContainer
from src.core.config import Settings
from tests.fakes import FakeEmbeddingService, FakeGraphClient, FakeVectorClient


@pytest.fixture
def settings() -> Settings:
    """Provide test settings with feature flags enabled."""
    return Settings(
        neo4j_url="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="testpassword",
        qdrant_url="http://localhost:6333",
        enable_graph_search=True,
        enable_hybrid_search=True,
    )


@pytest.fixture
def fake_vector_client() -> FakeVectorClient:
    """Provide fake vector client for testing."""
    return FakeVectorClient()


@pytest.fixture
def fake_graph_client() -> FakeGraphClient:
    """Provide fake graph client for testing."""
    return FakeGraphClient()


@pytest.fixture
def fake_embedding_service() -> FakeEmbeddingService:
    """Provide fake embedding service for testing."""
    return FakeEmbeddingService()


@pytest.fixture
def service_container(
    fake_vector_client: FakeVectorClient,
    fake_graph_client: FakeGraphClient,
    fake_embedding_service: FakeEmbeddingService,
) -> ServiceContainer:
    """Provide a fully configured service container with fakes."""
    return ServiceContainer(
        config=ServiceConfig(),
        vector_client=fake_vector_client,
        graph_client=fake_graph_client,
        embedding_service=fake_embedding_service,
    )


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for unit tests."""
    driver = MagicMock()
    driver.verify_connectivity = MagicMock(return_value=None)
    session = MagicMock()
    session.run = MagicMock(return_value=MagicMock(single=MagicMock(return_value=[1])))
    driver.session = MagicMock(return_value=session)
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=None)
    return driver


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for unit tests."""
    client = MagicMock()
    # Mock the health check response
    client.get_collections = MagicMock(return_value=MagicMock(collections=[]))
    return client


@pytest.fixture
def test_app(service_container: ServiceContainer) -> FastAPI:
    """
    Provide a fully configured test application with fake services.
    
    Use this instead of calling create_app() directly in tests.
    """
    return create_app(services=service_container)


@pytest.fixture
def test_client(test_app: FastAPI) -> TestClient:
    """Provide a test client for the configured test app."""
    return TestClient(test_app)


def create_test_app(
    vector_client: FakeVectorClient | None = None,
    graph_client: FakeGraphClient | None = None,
    embedding_service: FakeEmbeddingService | None = None,
) -> FastAPI:
    """
    Helper function to create a test app with specific fake services.
    
    Use this when you need custom fake configurations.
    All parameters default to new fake instances.
    """
    container = ServiceContainer(
        config=ServiceConfig(),
        vector_client=vector_client or FakeVectorClient(),
        graph_client=graph_client or FakeGraphClient(),
        embedding_service=embedding_service or FakeEmbeddingService(),
    )
    return create_app(services=container)
