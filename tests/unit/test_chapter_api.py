"""
Unit tests for the /v1/chapters/{book_id}/{chapter_number} endpoint.

TDD Phase: RED - These tests should FAIL until implementation is complete.

Reference: 
- Kitchen Brigade: ai-agents (Expeditor) → semantic-search (Cookbook) → Neo4j (Pantry)
- CODING_PATTERNS_ANALYSIS.md: Anti-Pattern #12 (Connection Pooling), #7/#13 (Exception Shadowing)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.dependencies import ServiceContainer, ServiceConfig


# ==============================================================================
# ChapterContentResponse Model Tests - RED Phase
# ==============================================================================


class TestChapterContentResponseModel:
    """Tests for ChapterContentResponse model validation."""

    def test_model_exists_and_importable(self) -> None:
        """Test that ChapterContentResponse can be imported from models."""
        from src.api.models import ChapterContentResponse
        
        assert ChapterContentResponse is not None

    def test_valid_response_all_fields(self) -> None:
        """Test valid response with all fields populated."""
        from src.api.models import ChapterContentResponse
        
        response = ChapterContentResponse(
            book_id="Architecture_Patterns_with_Python",
            chapter_number=5,
            title="TDD, DDD, and Event-Driven Architecture",
            summary="This chapter covers test-driven development patterns...",
            keywords=["TDD", "DDD", "event-driven", "CQRS"],
            concepts=["hexagonal architecture", "ports and adapters", "domain model"],
            page_range="89-134",
            found=True,
        )
        
        assert response.book_id == "Architecture_Patterns_with_Python"
        assert response.chapter_number == 5
        assert response.title == "TDD, DDD, and Event-Driven Architecture"
        assert response.found is True
        assert len(response.keywords) == 4
        assert len(response.concepts) == 3

    def test_not_found_response(self) -> None:
        """Test response when chapter is not found."""
        from src.api.models import ChapterContentResponse
        
        response = ChapterContentResponse(
            book_id="NonExistent_Book",
            chapter_number=999,
            title="",
            summary="",
            keywords=[],
            concepts=[],
            page_range="",
            found=False,
        )
        
        assert response.found is False
        assert response.title == ""

    def test_required_fields(self) -> None:
        """Test that required fields raise ValidationError when missing."""
        from src.api.models import ChapterContentResponse
        
        with pytest.raises(ValidationError) as exc_info:
            ChapterContentResponse()  # type: ignore[call-arg]
        
        errors = exc_info.value.errors()
        # book_id, chapter_number, and title are required (no defaults)
        required_fields = {"book_id", "chapter_number", "title"}
        missing_fields = {e["loc"][0] for e in errors}
        assert required_fields.issubset(missing_fields)


# ==============================================================================
# API Endpoint Tests - RED Phase
# ==============================================================================


@pytest.fixture
def mock_services_with_graph() -> ServiceContainer:
    """Create a mock service container with graph client configured."""
    config = ServiceConfig(
        enable_hybrid_search=True,
    )
    
    # Mock graph client with execute_query method
    graph_client = AsyncMock()
    
    # Mock Neo4j response for a found chapter
    graph_client.execute_query = AsyncMock(return_value={
        "records": [
            {
                "book_id": "Architecture_Patterns_with_Python",
                "chapter_number": 5,
                "title": "TDD, DDD, and Event-Driven Architecture",
                "summary": "This chapter explores the relationship between TDD and domain-driven design...",
                "keywords": ["TDD", "DDD", "event-driven", "CQRS", "aggregate"],
                "concepts": ["hexagonal architecture", "ports and adapters"],
                "page_range": "89-134",
            }
        ]
    })
    
    return ServiceContainer(
        config=config,
        vector_client=AsyncMock(),
        graph_client=graph_client,
        embedding_service=AsyncMock(),
    )


@pytest.fixture
def mock_services_no_graph() -> ServiceContainer:
    """Create a mock service container without graph client."""
    config = ServiceConfig(
        enable_hybrid_search=True,
    )
    
    return ServiceContainer(
        config=config,
        vector_client=AsyncMock(),
        graph_client=None,  # No graph client
        embedding_service=AsyncMock(),
    )


@pytest.fixture
def app_with_graph(mock_services_with_graph: ServiceContainer) -> FastAPI:
    """Create a FastAPI test application with graph client."""
    from src.api.routes import router, get_services
    
    app = FastAPI()
    app.include_router(router)
    
    def override_get_services() -> ServiceContainer:
        return mock_services_with_graph
    
    app.dependency_overrides[get_services] = override_get_services
    
    return app


@pytest.fixture
def app_no_graph(mock_services_no_graph: ServiceContainer) -> FastAPI:
    """Create a FastAPI test application without graph client."""
    from src.api.routes import router, get_services
    
    app = FastAPI()
    app.include_router(router)
    
    def override_get_services() -> ServiceContainer:
        return mock_services_no_graph
    
    app.dependency_overrides[get_services] = override_get_services
    
    return app


@pytest.fixture
def client_with_graph(app_with_graph: FastAPI) -> TestClient:
    """Create a test client with graph client available."""
    return TestClient(app_with_graph)


@pytest.fixture
def client_no_graph(app_no_graph: FastAPI) -> TestClient:
    """Create a test client without graph client."""
    return TestClient(app_no_graph)


class TestChapterContentEndpoint:
    """Tests for GET /v1/chapters/{book_id}/{chapter_number} endpoint."""

    def test_get_chapter_found(self, client_with_graph: TestClient) -> None:
        """Test successful retrieval of an existing chapter."""
        response = client_with_graph.get(
            "/v1/chapters/Architecture_Patterns_with_Python/5"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["book_id"] == "Architecture_Patterns_with_Python"
        assert data["chapter_number"] == 5
        assert data["title"] == "TDD, DDD, and Event-Driven Architecture"
        assert data["found"] is True
        assert "TDD" in data["keywords"]
        assert len(data["concepts"]) > 0

    def test_get_chapter_not_found(
        self, 
        mock_services_with_graph: ServiceContainer,
        app_with_graph: FastAPI,
    ) -> None:
        """Test response when chapter does not exist."""
        # Override to return empty records
        mock_services_with_graph.graph_client.execute_query = AsyncMock(
            return_value={"records": []}
        )
        
        client = TestClient(app_with_graph)
        response = client.get("/v1/chapters/NonExistent_Book/999")
        
        assert response.status_code == 200  # Not 404 - we return found=False
        data = response.json()
        
        assert data["found"] is False
        assert data["book_id"] == "NonExistent_Book"
        assert data["chapter_number"] == 999
        assert data["title"] == ""

    def test_get_chapter_service_unavailable(self, client_no_graph: TestClient) -> None:
        """Test 503 response when graph client is not configured."""
        response = client_no_graph.get(
            "/v1/chapters/Architecture_Patterns_with_Python/5"
        )
        
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["error"] == "graph_unavailable"

    def test_get_chapter_neo4j_error(
        self,
        mock_services_with_graph: ServiceContainer,
        app_with_graph: FastAPI,
    ) -> None:
        """Test 500 response when Neo4j query fails."""
        # Override to raise an exception
        mock_services_with_graph.graph_client.execute_query = AsyncMock(
            side_effect=Exception("Neo4j connection lost")
        )
        
        client = TestClient(app_with_graph)
        response = client.get("/v1/chapters/Architecture_Patterns_with_Python/5")
        
        assert response.status_code == 500
        data = response.json()
        assert data["detail"]["error"] == "chapter_retrieval_failed"


class TestChapterContentEndpointValidation:
    """Tests for endpoint input validation."""

    def test_invalid_chapter_number_negative(self, client_with_graph: TestClient) -> None:
        """Test that negative chapter numbers are handled gracefully."""
        response = client_with_graph.get(
            "/v1/chapters/Architecture_Patterns_with_Python/-1"
        )
        # Should either be 422 validation error or handle gracefully
        assert response.status_code in (200, 404, 422)

    def test_book_id_with_spaces(self, client_with_graph: TestClient) -> None:
        """Test book_id with URL-encoded spaces."""
        response = client_with_graph.get(
            "/v1/chapters/Architecture%20Patterns%20with%20Python/5"
        )
        # Should handle URL-encoded book IDs
        assert response.status_code in (200, 404)
