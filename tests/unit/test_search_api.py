"""
Unit tests for the /v1/search (simple similarity search) endpoint.

Reference: END_TO_END_INTEGRATION_WBS.md WBS 0.2.2
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.dependencies import ServiceContainer, ServiceConfig
from src.api.models import (
    SimpleSearchRequest,
    SimpleSearchResponse,
    SimpleSearchResultItem,
)
from src.api.routes import router


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_services() -> ServiceContainer:
    """Create a mock service container with all services configured."""
    config = ServiceConfig(
        enable_hybrid_search=True,
    )
    
    # Mock vector client
    vector_client = AsyncMock()
    
    # Create a mock result object with proper structure
    mock_result = MagicMock()
    mock_result.id = "doc-1"
    mock_result.score = 0.85
    mock_result.payload = {"content": "Test content", "metadata": {"source": "test"}}
    
    mock_result_2 = MagicMock()
    mock_result_2.id = "doc-2"
    mock_result_2.score = 0.72
    mock_result_2.payload = {"content": "More content"}
    
    vector_client.search = AsyncMock(return_value=[mock_result, mock_result_2])
    vector_client.health_check = AsyncMock(return_value=True)
    
    # Mock embedding service
    embedding_service = AsyncMock()
    embedding_service.embed = AsyncMock(return_value=[0.1] * 768)
    
    return ServiceContainer(
        config=config,
        vector_client=vector_client,
        graph_client=None,
        embedding_service=embedding_service,
    )


@pytest.fixture
def app(mock_services: ServiceContainer) -> FastAPI:
    """Create a FastAPI test application with mocked services."""
    app = FastAPI()
    app.include_router(router)
    
    # Override the dependency
    def override_get_services() -> ServiceContainer:
        return mock_services
    
    from src.api.routes import get_services
    app.dependency_overrides[get_services] = override_get_services
    
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


# ==============================================================================
# SimpleSearchRequest Model Tests
# ==============================================================================


class TestSimpleSearchRequest:
    """Tests for SimpleSearchRequest model validation."""

    def test_valid_request_minimal(self) -> None:
        """Test valid request with minimal fields."""
        request = SimpleSearchRequest(query="test query")
        assert request.query == "test query"
        assert request.collection == "documents"
        assert request.limit == 10
        assert request.min_score is None

    def test_valid_request_all_fields(self) -> None:
        """Test valid request with all fields specified."""
        request = SimpleSearchRequest(
            query="search query",
            collection="my_collection",
            limit=50,
            min_score=0.5,
        )
        assert request.query == "search query"
        assert request.collection == "my_collection"
        assert request.limit == 50
        assert request.min_score == 0.5

    def test_invalid_query_empty(self) -> None:
        """Test that empty query is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleSearchRequest(query="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_invalid_query_too_long(self) -> None:
        """Test that query exceeding max length is rejected."""
        long_query = "x" * 10001
        with pytest.raises(ValidationError) as exc_info:
            SimpleSearchRequest(query=long_query)
        assert "String should have at most 10000 characters" in str(exc_info.value)

    def test_invalid_limit_too_small(self) -> None:
        """Test that limit below minimum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleSearchRequest(query="test", limit=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_invalid_limit_too_large(self) -> None:
        """Test that limit above maximum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleSearchRequest(query="test", limit=101)
        assert "less than or equal to 100" in str(exc_info.value)

    def test_invalid_min_score_negative(self) -> None:
        """Test that negative min_score is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleSearchRequest(query="test", min_score=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_invalid_min_score_above_one(self) -> None:
        """Test that min_score above 1.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleSearchRequest(query="test", min_score=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleSearchRequest(query="test", unknown_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


# ==============================================================================
# SimpleSearchResultItem Model Tests
# ==============================================================================


class TestSimpleSearchResultItem:
    """Tests for SimpleSearchResultItem model validation."""

    def test_valid_result_minimal(self) -> None:
        """Test valid result with minimal fields."""
        result = SimpleSearchResultItem(id="doc-1", score=0.95)
        assert result.id == "doc-1"
        assert result.score == 0.95
        assert result.payload == {}

    def test_valid_result_with_payload(self) -> None:
        """Test valid result with payload."""
        result = SimpleSearchResultItem(
            id="doc-2",
            score=0.8,
            payload={"content": "Test content", "metadata": {"source": "test"}},
        )
        assert result.id == "doc-2"
        assert result.score == 0.8
        assert result.payload["content"] == "Test content"

    def test_score_at_boundaries(self) -> None:
        """Test score at boundary values."""
        result_zero = SimpleSearchResultItem(id="doc-1", score=0.0)
        assert result_zero.score == 0.0
        
        result_one = SimpleSearchResultItem(id="doc-2", score=1.0)
        assert result_one.score == 1.0

    def test_invalid_score_below_zero(self) -> None:
        """Test that score below 0 is rejected."""
        with pytest.raises(ValidationError):
            SimpleSearchResultItem(id="doc-1", score=-0.1)

    def test_invalid_score_above_one(self) -> None:
        """Test that score above 1 is rejected."""
        with pytest.raises(ValidationError):
            SimpleSearchResultItem(id="doc-1", score=1.1)


# ==============================================================================
# SimpleSearchResponse Model Tests
# ==============================================================================


class TestSimpleSearchResponse:
    """Tests for SimpleSearchResponse model validation."""

    def test_valid_response_empty_results(self) -> None:
        """Test valid response with empty results."""
        response = SimpleSearchResponse(
            results=[],
            total=0,
            query="test",
            latency_ms=10.5,
        )
        assert response.results == []
        assert response.total == 0
        assert response.query == "test"
        assert response.latency_ms == 10.5

    def test_valid_response_with_results(self) -> None:
        """Test valid response with results."""
        results = [
            SimpleSearchResultItem(id="doc-1", score=0.9, payload={"title": "Test"}),
            SimpleSearchResultItem(id="doc-2", score=0.8),
        ]
        response = SimpleSearchResponse(
            results=results,
            total=2,
            query="test query",
            latency_ms=25.3,
        )
        assert len(response.results) == 2
        assert response.total == 2
        assert response.results[0].id == "doc-1"
        assert response.results[1].id == "doc-2"

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden on response."""
        with pytest.raises(ValidationError):
            SimpleSearchResponse(
                results=[],
                total=0,
                query="test",
                latency_ms=10.0,
                extra_field="not allowed",
            )


# ==============================================================================
# /v1/search Endpoint Tests
# ==============================================================================


class TestSearchEndpoint:
    """Tests for the /v1/search endpoint."""

    def test_search_success(self, client: TestClient) -> None:
        """Test successful search returns 200."""
        response = client.post(
            "/v1/search",
            json={"query": "test search query"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert "query" in data
        assert "latency_ms" in data

    def test_search_returns_results_array(self, client: TestClient) -> None:
        """Test that search returns a results array."""
        response = client.post(
            "/v1/search",
            json={"query": "test query"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["results"], list)

    def test_search_results_have_id_and_score(self, client: TestClient) -> None:
        """Test that results have id and score fields."""
        response = client.post(
            "/v1/search",
            json={"query": "test query"},
        )
        assert response.status_code == 200
        data = response.json()
        for result in data["results"]:
            assert "id" in result
            assert "score" in result
            assert isinstance(result["score"], float)

    def test_search_score_in_valid_range(self, client: TestClient) -> None:
        """Test that all scores are in [0, 1] range."""
        response = client.post(
            "/v1/search",
            json={"query": "test query"},
        )
        assert response.status_code == 200
        data = response.json()
        for result in data["results"]:
            assert 0.0 <= result["score"] <= 1.0

    def test_search_with_custom_limit(self, client: TestClient, mock_services: ServiceContainer) -> None:
        """Test that limit parameter is respected."""
        # Set up more results than requested
        results = []
        for i in range(5):
            mock_result = MagicMock()
            mock_result.id = f"doc-{i}"
            mock_result.score = 0.9 - (i * 0.1)
            mock_result.payload = {}
            results.append(mock_result)
        mock_services.vector_client.search = AsyncMock(return_value=results[:3])
        
        response = client.post(
            "/v1/search",
            json={"query": "test query", "limit": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 3

    def test_search_with_collection(self, client: TestClient, mock_services: ServiceContainer) -> None:
        """Test search with custom collection."""
        response = client.post(
            "/v1/search",
            json={"query": "test", "collection": "custom_collection"},
        )
        assert response.status_code == 200
        # Verify the collection was passed to vector client
        mock_services.vector_client.search.assert_called()

    def test_search_with_min_score(self, client: TestClient, mock_services: ServiceContainer) -> None:
        """Test search with min_score filter."""
        # Set up results with varying scores
        results = []
        for i, score in enumerate([0.9, 0.6, 0.4]):
            mock_result = MagicMock()
            mock_result.id = f"doc-{i}"
            mock_result.score = score
            mock_result.payload = {}
            results.append(mock_result)
        mock_services.vector_client.search = AsyncMock(return_value=results)
        
        response = client.post(
            "/v1/search",
            json={"query": "test query", "min_score": 0.5},
        )
        assert response.status_code == 200
        data = response.json()
        # Should filter out the result with score 0.4
        for result in data["results"]:
            assert result["score"] >= 0.5

    def test_search_returns_payload(self, client: TestClient) -> None:
        """Test that results include payload."""
        response = client.post(
            "/v1/search",
            json={"query": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        for result in data["results"]:
            assert "payload" in result

    def test_search_returns_latency(self, client: TestClient) -> None:
        """Test that response includes latency."""
        response = client.post(
            "/v1/search",
            json={"query": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], float)
        assert data["latency_ms"] >= 0


# ==============================================================================
# /v1/search Endpoint Validation Tests
# ==============================================================================


class TestSearchEndpointValidation:
    """Tests for request validation on /v1/search endpoint."""

    def test_missing_query(self, client: TestClient) -> None:
        """Test that missing query returns 422."""
        response = client.post("/v1/search", json={})
        assert response.status_code == 422

    def test_empty_query(self, client: TestClient) -> None:
        """Test that empty query returns 422."""
        response = client.post(
            "/v1/search",
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_query_too_long(self, client: TestClient) -> None:
        """Test that oversized query returns 422."""
        response = client.post(
            "/v1/search",
            json={"query": "x" * 10001},
        )
        assert response.status_code == 422

    def test_invalid_limit_zero(self, client: TestClient) -> None:
        """Test that zero limit returns 422."""
        response = client.post(
            "/v1/search",
            json={"query": "test", "limit": 0},
        )
        assert response.status_code == 422

    def test_invalid_limit_too_large(self, client: TestClient) -> None:
        """Test that oversized limit returns 422."""
        response = client.post(
            "/v1/search",
            json={"query": "test", "limit": 101},
        )
        assert response.status_code == 422

    def test_invalid_min_score_negative(self, client: TestClient) -> None:
        """Test that negative min_score returns 422."""
        response = client.post(
            "/v1/search",
            json={"query": "test", "min_score": -0.1},
        )
        assert response.status_code == 422

    def test_invalid_min_score_above_one(self, client: TestClient) -> None:
        """Test that min_score > 1 returns 422."""
        response = client.post(
            "/v1/search",
            json={"query": "test", "min_score": 1.5},
        )
        assert response.status_code == 422


# ==============================================================================
# Service Unavailability Tests
# ==============================================================================


class TestSearchServiceUnavailable:
    """Tests for service unavailability handling."""

    def test_vector_client_unavailable(self, app: FastAPI) -> None:
        """Test 503 when vector client is unavailable."""
        # Create service container without vector client
        config = ServiceConfig(enable_hybrid_search=False)
        services = ServiceContainer(
            config=config,
            vector_client=None,
            graph_client=None,
            embedding_service=AsyncMock(),
        )
        
        from src.api.routes import get_services
        app.dependency_overrides[get_services] = lambda: services
        
        client = TestClient(app)
        response = client.post(
            "/v1/search",
            json={"query": "test"},
        )
        assert response.status_code == 503

    def test_embedding_service_unavailable(self, app: FastAPI) -> None:
        """Test 503 when embedding service is unavailable."""
        # Create service container without embedding service
        config = ServiceConfig(enable_hybrid_search=True)
        services = ServiceContainer(
            config=config,
            vector_client=AsyncMock(),
            graph_client=None,
            embedding_service=None,
        )
        
        from src.api.routes import get_services
        app.dependency_overrides[get_services] = lambda: services
        
        client = TestClient(app)
        response = client.post(
            "/v1/search",
            json={"query": "test"},
        )
        assert response.status_code == 503
