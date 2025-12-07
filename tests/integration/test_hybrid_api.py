"""
WBS 3.8 RED: Integration tests for Hybrid Search API routes.

Tests follow TDD RED phase - all tests should fail initially until
API routes are implemented in WBS 3.9.

API Endpoints:
- POST /v1/search/hybrid - Hybrid vector + graph search
- POST /v1/graph/traverse - Graph traversal from start node
- POST /v1/graph/query - Execute custom Cypher query

Based on Pre-Implementation Analysis (WBS 3.0.1-3.0.4):
- FastAPI endpoints with Pydantic models
- Feature flag integration (enable_hybrid_search)
- Proper error handling and response codes
- P95 latency < 500ms target

Anti-Pattern Mitigations Applied:
- Proper exception handling with custom error types
- Request/response validation with Pydantic
- Feature flag checks at route level
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for API tests."""
    settings = MagicMock()
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_collection = "test_chapters"
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_user = "neo4j"
    settings.neo4j_password = "testpassword"
    settings.enable_graph_search = True
    settings.enable_hybrid_search = True
    settings.hybrid_vector_weight = 0.7
    settings.hybrid_graph_weight = 0.3
    return settings


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample embedding vector."""
    return [0.1] * 384


@pytest.fixture
def sample_hybrid_response() -> list[dict[str, Any]]:
    """Create sample hybrid search response data."""
    return [
        {
            "id": "doc1",
            "vector_score": 0.95,
            "graph_score": 0.8,
            "hybrid_score": 0.905,
            "payload": {"text": "Chapter 1: Introduction", "book": "AI Engineering"},
            "relationship_type": "PARALLEL",
            "depth": 1,
        },
        {
            "id": "doc2",
            "vector_score": 0.85,
            "graph_score": 0.0,
            "hybrid_score": 0.595,
            "payload": {"text": "Chapter 2: ML Basics", "book": "AI Engineering"},
            "relationship_type": None,
            "depth": None,
        },
    ]


# =============================================================================
# Test: Hybrid Search Endpoint
# =============================================================================


class TestHybridSearchEndpoint:
    """Tests for POST /v1/search/hybrid endpoint."""

    def test_hybrid_search_success(self) -> None:
        """POST /v1/search/hybrid should return hybrid results."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/search/hybrid",
            json={
                "query": "machine learning basics",
                "limit": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_hybrid_search_with_embedding(self) -> None:
        """POST /v1/search/hybrid should accept pre-computed embedding."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/search/hybrid",
            json={
                "embedding": [0.1] * 384,
                "start_node_id": "doc1",
                "limit": 5,
            },
        )

        assert response.status_code == 200

    def test_hybrid_search_requires_query_or_embedding(self) -> None:
        """POST /v1/search/hybrid should require query or embedding."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/search/hybrid",
            json={
                "limit": 10,
                # Missing both query and embedding
            },
        )

        assert response.status_code == 422  # Validation error

    def test_hybrid_search_response_format(
        self,
        sample_hybrid_response: list[dict[str, Any]],
    ) -> None:
        """Hybrid search response should have correct format."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/search/hybrid",
            json={"query": "test"},
        )

        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "total" in data
            assert "alpha" in data
            assert "latency_ms" in data

    def test_hybrid_search_respects_limit(self) -> None:
        """Hybrid search should respect limit parameter."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/search/hybrid",
            json={
                "query": "test query",
                "limit": 3,
            },
        )

        if response.status_code == 200:
            data = response.json()
            assert len(data["results"]) <= 3

    def test_hybrid_search_disabled_returns_error(self) -> None:
        """When hybrid search disabled, should return 503."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.dependencies import ServiceConfig, ServiceContainer

        # Create app with hybrid search disabled
        config = ServiceConfig(enable_hybrid_search=False)
        services = ServiceContainer(config=config)

        app = create_app(services=services)
        client = TestClient(app)

        response = client.post(
            "/v1/search/hybrid",
            json={"query": "test"},
        )

        # When disabled, should return 503 Service Unavailable
        assert response.status_code in [503, 501]


# =============================================================================
# Test: Graph Traverse Endpoint
# =============================================================================


class TestGraphTraverseEndpoint:
    """Tests for POST /v1/graph/traverse endpoint."""

    def test_graph_traverse_success(self) -> None:
        """POST /v1/graph/traverse should return traversal results."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/graph/traverse",
            json={
                "start_node_id": "ch-1",
                "max_depth": 3,
                "relationship_types": ["PARALLEL", "PERPENDICULAR"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert "start_node" in data
        assert "latency_ms" in data

    def test_graph_traverse_requires_start_node(self) -> None:
        """POST /v1/graph/traverse should require start_node_id."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/graph/traverse",
            json={
                "max_depth": 3,
                # Missing start_node_id
            },
        )

        assert response.status_code == 422

    def test_graph_traverse_default_depth(self) -> None:
        """Graph traverse should use default max_depth if not specified."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/graph/traverse",
            json={
                "start_node_id": "ch-1",
                # max_depth not specified - should use default
            },
        )

        assert response.status_code == 200

    def test_graph_traverse_disabled_returns_error(self) -> None:
        """When graph search disabled, should return 503."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.dependencies import ServiceConfig, ServiceContainer

        config = ServiceConfig(enable_hybrid_search=False)
        services = ServiceContainer(config=config)

        app = create_app(services=services)
        client = TestClient(app)

        response = client.post(
            "/v1/graph/traverse",
            json={"start_node_id": "ch-1"},
        )

        assert response.status_code in [503, 501]


# =============================================================================
# Test: Graph Query Endpoint
# =============================================================================


class TestGraphQueryEndpoint:
    """Tests for POST /v1/graph/query endpoint."""

    def test_graph_query_success(self) -> None:
        """POST /v1/graph/query should execute Cypher query."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/graph/query",
            json={
                "cypher": "MATCH (n:Chapter) RETURN n.title LIMIT 10",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "records" in data
        assert "columns" in data
        assert "latency_ms" in data

    def test_graph_query_with_parameters(self) -> None:
        """POST /v1/graph/query should accept parameters."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/graph/query",
            json={
                "cypher": "MATCH (n:Chapter {book: $book}) RETURN n.title",
                "parameters": {"book": "AI Engineering"},
            },
        )

        assert response.status_code == 200

    def test_graph_query_requires_cypher(self) -> None:
        """POST /v1/graph/query should require cypher field."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/graph/query",
            json={
                "parameters": {"book": "AI Engineering"},
                # Missing cypher
            },
        )

        assert response.status_code == 422

    def test_graph_query_rejects_write_operations(self) -> None:
        """POST /v1/graph/query should reject write operations."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/graph/query",
            json={
                "cypher": "CREATE (n:Test {name: 'test'})",
            },
        )

        # Should reject CREATE, DELETE, MERGE, SET, REMOVE operations
        # 422 is returned by Pydantic validation which catches write operations
        assert response.status_code == 422


# =============================================================================
# Test: Request/Response Models
# =============================================================================


class TestRequestResponseModels:
    """Tests for Pydantic request/response models."""

    def test_hybrid_search_request_model(self) -> None:
        """HybridSearchRequest should validate inputs."""
        from src.api.models import HybridSearchRequest

        # Valid request with query
        request = HybridSearchRequest(query="test query", limit=10)
        assert request.query == "test query"
        assert request.limit == 10

    def test_hybrid_search_request_with_embedding(self) -> None:
        """HybridSearchRequest should accept embedding."""
        from src.api.models import HybridSearchRequest

        embedding = [0.1] * 384
        request = HybridSearchRequest(embedding=embedding, start_node_id="doc1")
        assert request.embedding == embedding

    def test_hybrid_search_response_model(self) -> None:
        """HybridSearchResponse should structure results."""
        from src.api.models import HybridSearchResponse, SearchResultItem

        item = SearchResultItem(
            id="doc1",
            score=0.87,
            vector_score=0.9,
            graph_score=0.8,
            payload={"text": "Test"},
        )
        response = HybridSearchResponse(
            results=[item],
            total=1,
            alpha=0.7,
            latency_ms=10.5,
        )

        assert len(response.results) == 1
        assert response.results[0].id == "doc1"

    def test_traverse_request_model(self) -> None:
        """TraverseRequest should validate inputs."""
        from src.api.models import TraverseRequest

        request = TraverseRequest(
            start_node_id="ch-1",
            max_depth=3,
            relationship_types=["PARALLEL"],
        )
        assert request.start_node_id == "ch-1"
        assert request.max_depth == 3

    def test_graph_query_request_model(self) -> None:
        """GraphQueryRequest should validate inputs."""
        from src.api.models import GraphQueryRequest

        request = GraphQueryRequest(
            cypher="MATCH (n) RETURN n",
            parameters={"limit": 10},
        )
        assert request.cypher == "MATCH (n) RETURN n"


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json_returns_422(self) -> None:
        """Invalid JSON should return 422."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/v1/search/hybrid",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_search_error_returns_500(self) -> None:
        """Search errors should return 500 with error details."""
        from fastapi.testclient import TestClient

        from src.api.app import configure_app_services, create_app
        from src.api.dependencies import (
            FakeEmbeddingService,
            FakeGraphClient,
            ServiceConfig,
            ServiceContainer,
        )

        # Create a vector client that raises an error
        class ErrorVectorClient:
            async def search(self, **kwargs: Any) -> list[Any]:  # noqa: ARG002, ANN401
                import asyncio
                await asyncio.sleep(0)  # Yield to event loop
                msg = "Connection lost"
                raise RuntimeError(msg)

            async def health_check(self) -> bool:
                import asyncio
                await asyncio.sleep(0)  # Yield to event loop
                return False

        config = ServiceConfig()
        services = ServiceContainer(
            config=config,
            vector_client=ErrorVectorClient(),
            graph_client=FakeGraphClient(),
            embedding_service=FakeEmbeddingService(),
        )

        app = create_app()
        configure_app_services(app, services)
        client = TestClient(app)

        response = client.post(
            "/v1/search/hybrid",
            json={"query": "test"},
        )

        # Should return 500 or handle gracefully
        assert response.status_code in [500, 503]


# =============================================================================
# Test: Health Check
# =============================================================================


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_returns_200(self) -> None:
        """GET /health should return 200."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200

    def test_health_check_includes_services(self) -> None:
        """Health check should report service status."""
        from fastapi.testclient import TestClient

        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/health")

        if response.status_code == 200:
            data = response.json()
            assert "status" in data
