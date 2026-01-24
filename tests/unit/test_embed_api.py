"""
Unit tests for /v1/embed API endpoint.

Reference: END_TO_END_INTEGRATION_WBS.md WBS 0.2.1.5
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import ServiceConfig, ServiceContainer
from src.api.models import EmbedRequest, EmbedResponse
from src.api.routes import get_services, router
from tests.fakes import FakeEmbeddingService


@pytest.fixture
def app_with_embedding_service() -> FastAPI:
    """Create FastAPI app with configured embedding service."""
    app = FastAPI()
    app.include_router(router)

    # Configure services with embedding service
    embedding_service = FakeEmbeddingService(dimension=768)
    container = ServiceContainer(
        config=ServiceConfig(embedding_dimension=768),
        embedding_service=embedding_service,
    )

    def override_get_services() -> ServiceContainer:
        return container

    app.dependency_overrides[get_services] = override_get_services
    return app


@pytest.fixture
def app_without_embedding_service() -> FastAPI:
    """Create FastAPI app without embedding service."""
    app = FastAPI()
    app.include_router(router)

    container = ServiceContainer(
        config=ServiceConfig(),
        embedding_service=None,
    )

    def override_get_services() -> ServiceContainer:
        return container

    app.dependency_overrides[get_services] = override_get_services
    return app


@pytest.fixture
def client_with_service(app_with_embedding_service: FastAPI) -> TestClient:
    """Test client with embedding service configured."""
    return TestClient(app_with_embedding_service)


@pytest.fixture
def client_without_service(app_without_embedding_service: FastAPI) -> TestClient:
    """Test client without embedding service."""
    return TestClient(app_without_embedding_service)


class TestEmbedRequest:
    """Tests for EmbedRequest Pydantic model."""

    def test_single_text_valid(self):
        """Test valid single text input."""
        request = EmbedRequest(text="Hello, world!")
        assert request.text == "Hello, world!"

    def test_list_text_valid(self):
        """Test valid list of texts input."""
        request = EmbedRequest(text=["Hello", "World"])
        assert request.text == ["Hello", "World"]

    def test_empty_string_invalid(self):
        """Test that empty string is rejected."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            EmbedRequest(text="")

    def test_empty_list_invalid(self):
        """Test that empty list is rejected."""
        with pytest.raises(ValueError, match="Text list cannot be empty"):
            EmbedRequest(text=[])

    def test_whitespace_only_invalid(self):
        """Test that whitespace-only text is rejected."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            EmbedRequest(text="   ")

    def test_list_with_empty_string_invalid(self):
        """Test that list containing empty string is rejected."""
        with pytest.raises(ValueError, match="Text at index 1 is empty"):
            EmbedRequest(text=["valid", ""])


class TestEmbedResponse:
    """Tests for EmbedResponse Pydantic model."""

    def test_valid_response(self):
        """Test valid response construction."""
        response = EmbedResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model="all-mpnet-base-v2",
            dimensions=3,
        )
        assert response.embeddings == [[0.1, 0.2, 0.3]]
        assert response.model == "all-mpnet-base-v2"
        assert response.dimensions == 3

    def test_response_with_usage(self):
        """Test response with usage statistics."""
        response = EmbedResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model="all-mpnet-base-v2",
            dimensions=3,
            usage={"total_tokens": 10},
        )
        assert response.usage == {"total_tokens": 10}

    def test_multiple_embeddings(self):
        """Test response with multiple embeddings."""
        response = EmbedResponse(
            embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            model="test-model",
            dimensions=2,
        )
        assert len(response.embeddings) == 3
        assert response.dimensions == 2


class TestEmbedEndpoint:
    """Tests for POST /v1/embed endpoint."""

    def test_embed_single_text_success(self, client_with_service: TestClient):
        """Test successful embedding of single text."""
        response = client_with_service.post(
            "/v1/embed",
            json={"text": "Hello, world!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) == 768  # Default dimension
        assert data["dimensions"] == 768
        assert "model" in data

    def test_embed_multiple_texts_success(self, client_with_service: TestClient):
        """Test successful embedding of multiple texts."""
        response = client_with_service.post(
            "/v1/embed",
            json={"text": ["Hello", "World", "Test"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == 3
        assert all(len(emb) == 768 for emb in data["embeddings"])

    def test_embed_returns_usage(self, client_with_service: TestClient):
        """Test that response includes usage statistics."""
        response = client_with_service.post(
            "/v1/embed",
            json={"text": "Hello world test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "usage" in data
        assert "total_tokens" in data["usage"]

    def test_embed_empty_text_rejected(self, client_with_service: TestClient):
        """Test that empty text is rejected with 422."""
        response = client_with_service.post(
            "/v1/embed",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_embed_service_unavailable(self, client_without_service: TestClient):
        """Test 503 when embedding service is not configured."""
        response = client_without_service.post(
            "/v1/embed",
            json={"text": "Hello, world!"},
        )
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["error"] == "embedding_service_unavailable"

    def test_embed_with_model_parameter(self, client_with_service: TestClient):
        """Test embedding with explicit model parameter."""
        response = client_with_service.post(
            "/v1/embed",
            json={"text": "Hello", "model": "custom-model"},
        )
        assert response.status_code == 200
        data = response.json()
        # Model should be returned (may or may not match request depending on service)
        assert "model" in data

    def test_embed_deterministic_for_same_input(self, client_with_service: TestClient):
        """Test that same input produces same embedding (deterministic fake)."""
        response1 = client_with_service.post(
            "/v1/embed",
            json={"text": "Consistent input"},
        )
        response2 = client_with_service.post(
            "/v1/embed",
            json={"text": "Consistent input"},
        )
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["embeddings"] == response2.json()["embeddings"]

    def test_embed_different_for_different_input(self, client_with_service: TestClient):
        """Test that different inputs produce different embeddings."""
        response1 = client_with_service.post(
            "/v1/embed",
            json={"text": "First text"},
        )
        response2 = client_with_service.post(
            "/v1/embed",
            json={"text": "Second text"},
        )
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["embeddings"] != response2.json()["embeddings"]


class TestEmbedEndpointValidation:
    """Tests for /v1/embed request validation."""

    def test_missing_text_field(self, client_with_service: TestClient):
        """Test that missing text field returns 422."""
        response = client_with_service.post(
            "/v1/embed",
            json={},
        )
        assert response.status_code == 422

    def test_invalid_text_type(self, client_with_service: TestClient):
        """Test that invalid text type returns 422."""
        response = client_with_service.post(
            "/v1/embed",
            json={"text": 123},
        )
        assert response.status_code == 422

    def test_extra_fields_rejected(self, client_with_service: TestClient):
        """Test that extra fields are rejected (strict mode)."""
        response = client_with_service.post(
            "/v1/embed",
            json={"text": "Hello", "unknown_field": "value"},
        )
        assert response.status_code == 422
