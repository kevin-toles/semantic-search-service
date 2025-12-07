"""
Pytest configuration and fixtures for semantic-search-service tests.
"""

from unittest.mock import MagicMock

import pytest

from src.core.config import Settings


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
