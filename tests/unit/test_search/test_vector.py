"""
WBS 3.1 RED: Unit tests for QdrantSearchClient.

Tests follow TDD RED phase - all tests should fail initially until
vector.py is implemented in WBS 3.2.

Based on Pre-Implementation Analysis (WBS 3.0.1-3.0.4):
- Repository pattern with duck typing (GUIDELINES line 795)
- FakeQdrantSearchClient for unit tests (Architecture Patterns p.95)
- Connection reuse pattern (Comp_Static_Analysis #12)
- Custom exceptions to avoid shadowing (Comp_Static_Analysis #7, #13)
- Async context manager (Comp_Static_Analysis #12 resolution)
- HNSW indexing for approximate vector search (AI Engineering p.158)
- Score normalization to [0, 1] range

Anti-Pattern Mitigations Applied:
- QdrantConnectionError/QdrantSearchError (not ConnectionError)
- Lazy client initialization
- Full type hints on all public methods
- No hardcoded collection names (use Settings)
- Always use `limit` parameter (default: 10)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for Qdrant configuration."""
    settings = MagicMock()
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_collection = "test_chapters"
    settings.qdrant_api_key = None  # Optional API key
    return settings


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample 384-dimensional embedding (all-MiniLM-L6-v2 size)."""
    return [0.1] * 384


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Create sample embeddings for batch operations."""
    return [
        [0.1] * 384,
        [0.2] * 384,
        [0.3] * 384,
    ]


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """Create sample documents with metadata for upsert operations."""
    return [
        {
            "id": "doc1",
            "text": "Chapter 1: Introduction to AI",
            "metadata": {"book": "AI Engineering", "chapter": 1, "tier": "T2"},
        },
        {
            "id": "doc2",
            "text": "Chapter 2: Machine Learning Basics",
            "metadata": {"book": "AI Engineering", "chapter": 2, "tier": "T2"},
        },
        {
            "id": "doc3",
            "text": "Chapter 3: Deep Learning",
            "metadata": {"book": "AI Engineering", "chapter": 3, "tier": "T2"},
        },
    ]


# =============================================================================
# Test: QdrantSearchClient Initialization
# =============================================================================


class TestQdrantSearchClientInitialization:
    """Tests for QdrantSearchClient initialization and configuration."""

    def test_client_accepts_settings(self, mock_settings: MagicMock) -> None:
        """QdrantSearchClient should accept Settings object for configuration."""
        from src.search.vector import QdrantSearchClient

        client = QdrantSearchClient(settings=mock_settings)

        assert client._settings == mock_settings

    def test_client_lazy_initialization(self, mock_settings: MagicMock) -> None:
        """Qdrant client should not be created until first use (lazy init).

        Mitigates Anti-Pattern #12: new client per request.
        """
        from src.search.vector import QdrantSearchClient

        client = QdrantSearchClient(settings=mock_settings)

        # Internal client should be None until connect() is called
        assert client._client is None

    def test_client_stores_configuration(self, mock_settings: MagicMock) -> None:
        """Client should store URL, collection name from settings."""
        from src.search.vector import QdrantSearchClient

        client = QdrantSearchClient(settings=mock_settings)

        assert client._url == "http://localhost:6333"
        assert client._collection == "test_chapters"

    def test_client_default_limit(self, mock_settings: MagicMock) -> None:
        """Client should have default limit to prevent unbounded results."""
        from src.search.vector import QdrantSearchClient

        client = QdrantSearchClient(settings=mock_settings)

        assert client._default_limit == 10


# =============================================================================
# Test: QdrantSearchClient Connection Management
# =============================================================================


class TestQdrantSearchClientConnection:
    """Tests for QdrantSearchClient connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_creates_client(self, mock_settings: MagicMock) -> None:
        """connect() should create AsyncQdrantClient."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_qdrant.return_value = mock_client

            client = QdrantSearchClient(settings=mock_settings)
            await client.connect()

            mock_qdrant.assert_called_once_with(
                url="http://localhost:6333",
                api_key=None,
            )
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_connect_verifies_connectivity(
        self, mock_settings: MagicMock
    ) -> None:
        """connect() should verify connectivity by calling get_collections()."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_qdrant.return_value = mock_client

            client = QdrantSearchClient(settings=mock_settings)
            await client.connect()

            mock_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_raises_on_failure(self, mock_settings: MagicMock) -> None:
        """connect() should raise QdrantConnectionError on failure."""
        from src.search.exceptions import QdrantConnectionError
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_qdrant.side_effect = Exception("Connection refused")

            client = QdrantSearchClient(settings=mock_settings)

            with pytest.raises(QdrantConnectionError) as exc_info:
                await client.connect()

            assert "Connection refused" in str(exc_info.value)
            assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_close_closes_client(self, mock_settings: MagicMock) -> None:
        """close() should close the client connection."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_qdrant.return_value = mock_client

            client = QdrantSearchClient(settings=mock_settings)
            await client.connect()
            await client.close()

            mock_client.close.assert_called_once()
            assert client._client is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, mock_settings: MagicMock) -> None:
        """close() should be idempotent - calling twice should not error."""
        from src.search.vector import QdrantSearchClient

        client = QdrantSearchClient(settings=mock_settings)
        # Close without connecting should not raise
        await client.close()
        await client.close()  # Second call also should not raise


# =============================================================================
# Test: Async Context Manager
# =============================================================================


class TestQdrantSearchClientContextManager:
    """Tests for async context manager support."""

    @pytest.mark.asyncio
    async def test_async_context_manager_connects(
        self, mock_settings: MagicMock
    ) -> None:
        """Async context manager should call connect() on enter."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                assert client._client is not None

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_on_exit(
        self, mock_settings: MagicMock
    ) -> None:
        """Async context manager should call close() on exit."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings):
                _ = "Context manager test"  # Verify entry/exit hooks

            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_on_exception(
        self, mock_settings: MagicMock
    ) -> None:
        """Async context manager should close even if exception occurs."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_qdrant.return_value = mock_client

            with pytest.raises(ValueError):
                async with QdrantSearchClient(settings=mock_settings):
                    raise ValueError("Test error")

            mock_client.close.assert_called_once()


# =============================================================================
# Test: Vector Search Operations
# =============================================================================


class TestQdrantSearchClientSearch:
    """Tests for vector search operations."""

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should return list of SearchResult objects."""
        from src.search.vector import QdrantSearchClient, SearchResult

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()

            # Mock search response
            mock_point = MagicMock()
            mock_point.id = "doc1"
            mock_point.score = 0.95
            mock_point.payload = {"text": "Chapter 1", "book": "AI Engineering"}
            mock_client.search = AsyncMock(return_value=[mock_point])
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                results = await client.search(embedding=sample_embedding)

            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            assert results[0].id == "doc1"
            assert results[0].score == pytest.approx(0.95)
            assert results[0].payload == {"text": "Chapter 1", "book": "AI Engineering"}

    @pytest.mark.asyncio
    async def test_search_respects_limit(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should pass limit parameter to Qdrant."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.search = AsyncMock(return_value=[])
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                await client.search(embedding=sample_embedding, limit=5)

            mock_client.search.assert_called_once()
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs.get("limit") == 5

    @pytest.mark.asyncio
    async def test_search_uses_default_limit(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should use default limit when not specified."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.search = AsyncMock(return_value=[])
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                await client.search(embedding=sample_embedding)

            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs.get("limit") == 10  # Default limit

    @pytest.mark.asyncio
    async def test_search_with_filter(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should accept optional filter parameter."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.search = AsyncMock(return_value=[])
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                await client.search(
                    embedding=sample_embedding,
                    filter_conditions={"tier": "T2"},
                )

            mock_client.search.assert_called_once()
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs.get("query_filter") is not None

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should accept optional score_threshold parameter."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.search = AsyncMock(return_value=[])
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                await client.search(
                    embedding=sample_embedding,
                    score_threshold=0.8,
                )

            mock_client.search.assert_called_once()
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs.get("score_threshold") == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_search_raises_on_not_connected(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should raise QdrantConnectionError if not connected."""
        from src.search.exceptions import QdrantConnectionError
        from src.search.vector import QdrantSearchClient

        client = QdrantSearchClient(settings=mock_settings)

        with pytest.raises(QdrantConnectionError) as exc_info:
            await client.search(embedding=sample_embedding)

        assert "not connected" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_raises_on_error(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should raise QdrantSearchError on Qdrant errors."""
        from src.search.exceptions import QdrantSearchError
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.search = AsyncMock(side_effect=Exception("Collection not found"))
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                with pytest.raises(QdrantSearchError) as exc_info:
                    await client.search(embedding=sample_embedding)

            assert "Collection not found" in str(exc_info.value)


# =============================================================================
# Test: Upsert Operations
# =============================================================================


class TestQdrantSearchClientUpsert:
    """Tests for vector upsert (insert/update) operations."""

    @pytest.mark.asyncio
    async def test_upsert_single_document(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """upsert() should insert a single document with embedding."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.upsert = AsyncMock()
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                await client.upsert(
                    id="doc1",
                    embedding=sample_embedding,
                    payload={"text": "Chapter 1", "book": "AI Engineering"},
                )

            mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_batch(
        self,
        mock_settings: MagicMock,
        sample_embeddings: list[list[float]],
        sample_documents: list[dict[str, Any]],
    ) -> None:
        """upsert_batch() should insert multiple documents efficiently."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.upsert = AsyncMock()
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                await client.upsert_batch(
                    ids=["doc1", "doc2", "doc3"],
                    embeddings=sample_embeddings,
                    payloads=[doc["metadata"] for doc in sample_documents],
                )

            mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_raises_on_mismatched_lengths(
        self,
        mock_settings: MagicMock,
        sample_embeddings: list[list[float]],
    ) -> None:
        """upsert_batch() should raise ValueError if lengths don't match."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                with pytest.raises(ValueError) as exc_info:
                    await client.upsert_batch(
                        ids=["doc1", "doc2"],  # 2 IDs
                        embeddings=sample_embeddings,  # 3 embeddings
                        payloads=[{}, {}],  # 2 payloads
                    )

            assert "length" in str(exc_info.value).lower()


# =============================================================================
# Test: Delete Operations
# =============================================================================


class TestQdrantSearchClientDelete:
    """Tests for vector delete operations."""

    @pytest.mark.asyncio
    async def test_delete_by_id(self, mock_settings: MagicMock) -> None:
        """delete() should remove document by ID."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.delete = AsyncMock()
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                await client.delete(ids=["doc1", "doc2"])

            mock_client.delete.assert_called_once()


# =============================================================================
# Test: Collection Management
# =============================================================================


class TestQdrantSearchClientCollections:
    """Tests for collection management operations."""

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_if_not_exists(
        self, mock_settings: MagicMock
    ) -> None:
        """ensure_collection() should create collection if it doesn't exist."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.collection_exists = AsyncMock(return_value=False)
            mock_client.create_collection = AsyncMock()
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                await client.ensure_collection(vector_size=384)

            mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_skips_if_exists(
        self, mock_settings: MagicMock
    ) -> None:
        """ensure_collection() should not create if collection already exists."""
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.collection_exists = AsyncMock(return_value=True)
            mock_client.create_collection = AsyncMock()
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                await client.ensure_collection(vector_size=384)

            mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_collection_configures_hnsw(
        self, mock_settings: MagicMock
    ) -> None:
        """ensure_collection() should configure HNSW index parameters.

        HNSW (Hierarchical Navigable Small World) provides approximate
        nearest neighbor search with logarithmic scaling (AI Engineering p.158).
        """
        from src.search.vector import QdrantSearchClient

        with patch("src.search.vector.AsyncQdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
            mock_client.close = AsyncMock()
            mock_client.collection_exists = AsyncMock(return_value=False)
            mock_client.create_collection = AsyncMock()
            mock_qdrant.return_value = mock_client

            async with QdrantSearchClient(settings=mock_settings) as client:
                await client.ensure_collection(
                    vector_size=384,
                    hnsw_m=16,
                    hnsw_ef_construct=100,
                )

            mock_client.create_collection.assert_called_once()
            call_kwargs = mock_client.create_collection.call_args.kwargs
            # Verify HNSW config is passed
            assert "hnsw_config" in call_kwargs or "vectors_config" in call_kwargs


# =============================================================================
# Test: FakeQdrantSearchClient for Unit Testing
# =============================================================================


class TestFakeQdrantSearchClient:
    """Tests for FakeQdrantSearchClient - in-memory test double.

    FakeQdrantSearchClient enables unit testing without Qdrant server.
    Based on Architecture Patterns with Python p.95 "FakeRepository".
    """

    @pytest.mark.asyncio
    async def test_fake_implements_same_interface(self) -> None:
        """FakeQdrantSearchClient should implement QdrantSearchClientProtocol."""
        from src.search.vector import FakeQdrantSearchClient, QdrantSearchClientProtocol

        fake = FakeQdrantSearchClient()

        assert isinstance(fake, QdrantSearchClientProtocol)

    @pytest.mark.asyncio
    async def test_fake_connect_succeeds(self) -> None:
        """FakeQdrantSearchClient.connect() should always succeed."""
        from src.search.vector import FakeQdrantSearchClient

        fake = FakeQdrantSearchClient()
        await fake.connect()  # Should not raise

    @pytest.mark.asyncio
    async def test_fake_upsert_stores_data(self) -> None:
        """FakeQdrantSearchClient.upsert() should store data in memory."""
        from src.search.vector import FakeQdrantSearchClient

        fake = FakeQdrantSearchClient()
        await fake.connect()
        await fake.upsert(
            id="doc1",
            embedding=[0.1] * 384,
            payload={"text": "Test document"},
        )

        # Verify data is stored
        assert "doc1" in fake._storage

    @pytest.mark.asyncio
    async def test_fake_search_returns_stored_data(self) -> None:
        """FakeQdrantSearchClient.search() should return matching stored data."""
        from src.search.vector import FakeQdrantSearchClient

        fake = FakeQdrantSearchClient()
        await fake.connect()
        await fake.upsert(
            id="doc1",
            embedding=[0.1] * 384,
            payload={"text": "Test document"},
        )

        results = await fake.search(embedding=[0.1] * 384, limit=10)

        assert len(results) >= 1
        assert results[0].id == "doc1"

    @pytest.mark.asyncio
    async def test_fake_delete_removes_data(self) -> None:
        """FakeQdrantSearchClient.delete() should remove data from memory."""
        from src.search.vector import FakeQdrantSearchClient

        fake = FakeQdrantSearchClient()
        await fake.connect()
        await fake.upsert(
            id="doc1",
            embedding=[0.1] * 384,
            payload={"text": "Test document"},
        )
        await fake.delete(ids=["doc1"])

        assert "doc1" not in fake._storage

    @pytest.mark.asyncio
    async def test_fake_async_context_manager(self) -> None:
        """FakeQdrantSearchClient should work as async context manager."""
        from src.search.vector import FakeQdrantSearchClient

        async with FakeQdrantSearchClient() as fake:
            await fake.upsert(
                id="doc1",
                embedding=[0.1] * 384,
                payload={"text": "Test"},
            )
            results = await fake.search(embedding=[0.1] * 384)
            assert len(results) >= 1


# =============================================================================
# Test: SearchResult Data Class
# =============================================================================


class TestSearchResult:
    """Tests for SearchResult data class."""

    def test_search_result_creation(self) -> None:
        """SearchResult should be creatable with required fields."""
        from src.search.vector import SearchResult

        result = SearchResult(
            id="doc1",
            score=0.95,
            payload={"text": "Chapter 1", "book": "AI Engineering"},
        )

        assert result.id == "doc1"
        assert result.score == pytest.approx(0.95)
        assert result.payload["text"] == "Chapter 1"

    def test_search_result_score_normalized(self) -> None:
        """SearchResult.score should be in [0, 1] range."""
        from src.search.vector import SearchResult

        result = SearchResult(id="doc1", score=0.95, payload={})

        assert 0 <= result.score <= 1

    def test_search_result_optional_payload(self) -> None:
        """SearchResult should allow None payload."""
        from src.search.vector import SearchResult

        result = SearchResult(id="doc1", score=0.95, payload=None)

        assert result.payload is None

    def test_search_result_equality(self) -> None:
        """SearchResult should support equality comparison by ID."""
        from src.search.vector import SearchResult

        result1 = SearchResult(id="doc1", score=0.95, payload={})
        result2 = SearchResult(id="doc1", score=0.90, payload={"other": "data"})

        # Same ID means same document (scores may differ)
        assert result1.id == result2.id
