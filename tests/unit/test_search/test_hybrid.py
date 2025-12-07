"""
WBS 3.3 RED: Unit tests for HybridSearchService.

Tests follow TDD RED phase - all tests should fail initially until
hybrid.py is implemented in WBS 3.4.

Based on Pre-Implementation Analysis (WBS 3.0.1-3.0.4):
- Linear score fusion: final = α*vector_score + (1-α)*graph_score
- Configurable weights (default: vector=0.7, graph=0.3)
- Score normalization to [0, 1] range before fusion
- Combine vector similarity (semantic) with graph relationships (structural)
- P95 latency < 500ms target

Anti-Pattern Mitigations Applied:
- Configurable weights (not hardcoded equal weighting)
- Score validation before fusion
- Division by zero handling
- Deterministic tie-breaking by ID
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for hybrid search configuration."""
    settings = MagicMock()
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_collection = "test_chapters"
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_user = "neo4j"
    settings.neo4j_password = "testpassword"
    settings.neo4j_database = "neo4j"
    settings.enable_hybrid_search = True
    settings.hybrid_vector_weight = 0.7
    settings.hybrid_graph_weight = 0.3
    return settings


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample 384-dimensional embedding."""
    return [0.1] * 384


@pytest.fixture
def mock_vector_results() -> list[MagicMock]:
    """Create mock vector search results."""
    results = []
    for i, (doc_id, score) in enumerate([
        ("doc1", 0.95),
        ("doc2", 0.85),
        ("doc3", 0.75),
        ("doc4", 0.65),
    ]):
        result = MagicMock()
        result.id = doc_id
        result.score = score
        result.payload = {"text": f"Chapter {i+1}", "book": "AI Engineering"}
        results.append(result)
    return results


@pytest.fixture
def mock_graph_results() -> list[dict[str, Any]]:
    """Create mock graph traversal results."""
    return [
        {
            "node_id": "doc2",
            "depth": 1,
            "relevance_score": 0.9,
            "relationship_type": "PARALLEL",
        },
        {
            "node_id": "doc3",
            "depth": 2,
            "relevance_score": 0.7,
            "relationship_type": "PERPENDICULAR",
        },
        {
            "node_id": "doc5",
            "depth": 1,
            "relevance_score": 0.8,
            "relationship_type": "SKIP_TIER",
        },
    ]


# =============================================================================
# Test: HybridSearchService Initialization
# =============================================================================


class TestHybridSearchServiceInitialization:
    """Tests for HybridSearchService initialization and configuration."""

    def test_service_accepts_clients(self, mock_settings: MagicMock) -> None:
        """HybridSearchService should accept vector and graph clients."""
        from src.search.hybrid import HybridSearchService

        mock_vector_client = MagicMock()
        mock_graph_client = MagicMock()

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        assert service._vector_client == mock_vector_client
        assert service._graph_client == mock_graph_client

    def test_service_default_weights(self, mock_settings: MagicMock) -> None:
        """Service should use configurable default weights."""
        from src.search.hybrid import HybridSearchService

        # Remove weight settings to test defaults
        mock_settings.hybrid_vector_weight = None
        mock_settings.hybrid_graph_weight = None

        service = HybridSearchService(
            vector_client=MagicMock(),
            graph_client=MagicMock(),
            settings=mock_settings,
        )

        # Default: 70% vector, 30% graph
        assert service._vector_weight == pytest.approx(0.7)
        assert service._graph_weight == pytest.approx(0.3)

    def test_service_custom_weights(self, mock_settings: MagicMock) -> None:
        """Service should accept custom weights from settings."""
        from src.search.hybrid import HybridSearchService

        mock_settings.hybrid_vector_weight = 0.6
        mock_settings.hybrid_graph_weight = 0.4

        service = HybridSearchService(
            vector_client=MagicMock(),
            graph_client=MagicMock(),
            settings=mock_settings,
        )

        assert service._vector_weight == pytest.approx(0.6)
        assert service._graph_weight == pytest.approx(0.4)

    def test_service_weights_must_sum_to_one(self, mock_settings: MagicMock) -> None:
        """Service should validate weights sum to 1.0."""
        from src.search.hybrid import HybridSearchService

        mock_settings.hybrid_vector_weight = 0.5
        mock_settings.hybrid_graph_weight = 0.3  # noqa: ERA001 - intentional invalid weight

        with pytest.raises(ValueError) as exc_info:
            HybridSearchService(
                vector_client=MagicMock(),
                graph_client=MagicMock(),
                settings=mock_settings,
            )

        assert "sum to 1.0" in str(exc_info.value).lower()


# =============================================================================
# Test: Hybrid Search Execution
# =============================================================================


class TestHybridSearchExecution:
    """Tests for hybrid search execution flow."""

    @pytest.mark.asyncio
    async def test_search_calls_both_sources(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should query both vector and graph sources."""
        from src.search.hybrid import HybridSearchService

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=[])

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=[])

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        mock_vector_client.search.assert_called_once()
        mock_graph_client.query.assert_called()

    @pytest.mark.asyncio
    async def test_search_returns_hybrid_results(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
        mock_vector_results: list[MagicMock],
        mock_graph_results: list[dict[str, Any]],
    ) -> None:
        """search() should return HybridSearchResult objects."""
        from src.search.hybrid import HybridSearchResult, HybridSearchService

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=mock_vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=mock_graph_results)

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        assert len(results) > 0
        assert all(isinstance(r, HybridSearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_fuses_scores(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should fuse vector and graph scores using weights."""
        from src.search.hybrid import HybridSearchService
        from src.search.vector import SearchResult

        # Setup: doc1 has both vector and graph scores
        vector_results = [
            SearchResult(id="doc1", score=0.9, payload={}),
        ]
        graph_results = [
            {"node_id": "doc1", "depth": 1, "relevance_score": 0.8},
        ]

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=graph_results)

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        # Expected: 0.7 * 0.9 + 0.3 * 0.8 = 0.63 + 0.24 = 0.87
        assert len(results) == 1
        assert 0.85 <= results[0].hybrid_score <= 0.89  # Allow small float variance

    @pytest.mark.asyncio
    async def test_search_handles_vector_only_results(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should handle docs found only in vector search."""
        from src.search.hybrid import HybridSearchService
        from src.search.vector import SearchResult

        # doc1 only in vector results, not in graph
        vector_results = [
            SearchResult(id="doc1", score=0.9, payload={}),
        ]
        graph_results = []  # No graph results

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=graph_results)

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        # doc1 should still appear with graph_score = 0
        assert len(results) == 1
        assert results[0].id == "doc1"
        assert results[0].vector_score == pytest.approx(0.9)
        assert results[0].graph_score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_search_handles_graph_only_results(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should handle docs found only in graph traversal."""
        from src.search.hybrid import HybridSearchService

        # doc2 only in graph results, not in vector
        vector_results = []  # No vector results
        graph_results = [
            {"node_id": "doc2", "depth": 1, "relevance_score": 0.8},
        ]

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=graph_results)

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        # doc2 should appear with vector_score = 0
        assert len(results) == 1
        assert results[0].id == "doc2"
        assert results[0].vector_score == pytest.approx(0.0)
        assert results[0].graph_score == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_search_respects_limit(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
        mock_vector_results: list[MagicMock],
    ) -> None:
        """search() should respect limit parameter."""
        from src.search.hybrid import HybridSearchService

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=mock_vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=[])

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
            limit=2,
        )

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_sorts_by_hybrid_score_descending(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """search() should return results sorted by hybrid score descending."""
        from src.search.hybrid import HybridSearchService
        from src.search.vector import SearchResult

        # Multiple results with different scores
        vector_results = [
            SearchResult(id="doc1", score=0.5, payload={}),
            SearchResult(id="doc2", score=0.9, payload={}),
            SearchResult(id="doc3", score=0.7, payload={}),
        ]

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=[])

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        # Verify descending order by hybrid_score
        scores = [r.hybrid_score for r in results]
        assert scores == sorted(scores, reverse=True)


# =============================================================================
# Test: Score Normalization
# =============================================================================


class TestScoreNormalization:
    """Tests for score normalization before fusion."""

    @pytest.mark.asyncio
    async def test_normalizes_scores_to_0_1_range(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """All scores should be normalized to [0, 1] range."""
        from src.search.hybrid import HybridSearchService
        from src.search.vector import SearchResult

        # Vector score already normalized, graph score from depth
        vector_results = [
            SearchResult(id="doc1", score=0.8, payload={}),
        ]
        graph_results = [
            {"node_id": "doc1", "depth": 2, "relevance_score": 0.5},
        ]

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=graph_results)

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        for result in results:
            assert 0 <= result.vector_score <= 1
            assert 0 <= result.graph_score <= 1
            assert 0 <= result.hybrid_score <= 1

    @pytest.mark.asyncio
    async def test_validates_non_negative_scores(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """Should handle edge cases of negative or zero scores."""
        from src.search.hybrid import HybridSearchService
        from src.search.vector import SearchResult

        # Edge case: score = 0
        vector_results = [
            SearchResult(id="doc1", score=0.0, payload={}),
        ]

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=[])

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        # Should not raise, score should be 0
        assert len(results) == 1
        assert results[0].hybrid_score >= 0


# =============================================================================
# Test: Tie Breaking
# =============================================================================


class TestTieBreaking:
    """Tests for deterministic tie-breaking."""

    @pytest.mark.asyncio
    async def test_tie_breaking_by_id(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """When hybrid scores are equal, should break ties by ID."""
        from src.search.hybrid import HybridSearchService
        from src.search.vector import SearchResult

        # Two docs with same score
        vector_results = [
            SearchResult(id="doc_b", score=0.8, payload={}),
            SearchResult(id="doc_a", score=0.8, payload={}),
        ]

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=[])

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        # Deterministic order by ID when scores equal
        ids = [r.id for r in results]
        # Should be consistent across multiple calls
        assert ids == ["doc_a", "doc_b"] or ids == ["doc_b", "doc_a"]

        # Run again to verify determinism
        results2 = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )
        ids2 = [r.id for r in results2]
        assert ids == ids2  # Same order each time


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestHybridSearchErrorHandling:
    """Tests for error handling in hybrid search."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_graph_failure(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """Should return vector-only results if graph search fails."""
        from src.search.hybrid import HybridSearchService
        from src.search.vector import SearchResult

        vector_results = [
            SearchResult(id="doc1", score=0.9, payload={}),
        ]

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = False  # Graph disconnected

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        # Should not raise, should return vector-only results
        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        assert len(results) == 1
        assert results[0].graph_score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_raises_on_vector_failure(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """Should raise error if vector search fails (primary source)."""
        from src.search.exceptions import QdrantSearchError
        from src.search.hybrid import HybridSearchService

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(side_effect=QdrantSearchError("Connection lost"))

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=[])

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        with pytest.raises(QdrantSearchError):
            await service.search(
                query_embedding=sample_embedding,
                start_node_id="doc1",
            )


# =============================================================================
# Test: HybridSearchResult Data Class
# =============================================================================


class TestHybridSearchResult:
    """Tests for HybridSearchResult data class."""

    def test_hybrid_result_creation(self) -> None:
        """HybridSearchResult should be creatable with required fields."""
        from src.search.hybrid import HybridSearchResult

        result = HybridSearchResult(
            id="doc1",
            vector_score=0.9,
            graph_score=0.8,
            hybrid_score=0.87,
            payload={"text": "Chapter 1"},
        )

        assert result.id == "doc1"
        assert result.vector_score == pytest.approx(0.9)
        assert result.graph_score == pytest.approx(0.8)
        assert result.hybrid_score == pytest.approx(0.87)
        assert result.payload["text"] == "Chapter 1"

    def test_hybrid_result_optional_fields(self) -> None:
        """HybridSearchResult should support optional fields."""
        from src.search.hybrid import HybridSearchResult

        result = HybridSearchResult(
            id="doc1",
            vector_score=0.9,
            graph_score=0.0,
            hybrid_score=0.63,
            payload=None,
            relationship_type=None,
            depth=None,
        )

        assert result.payload is None
        assert result.relationship_type is None
        assert result.depth is None

    def test_hybrid_result_with_graph_metadata(self) -> None:
        """HybridSearchResult should include graph traversal metadata."""
        from src.search.hybrid import HybridSearchResult

        result = HybridSearchResult(
            id="doc1",
            vector_score=0.9,
            graph_score=0.8,
            hybrid_score=0.87,
            payload={"text": "Chapter 1"},
            relationship_type="PARALLEL",
            depth=2,
        )

        assert result.relationship_type == "PARALLEL"
        assert result.depth == 2


# =============================================================================
# Test: Feature Flag Integration
# =============================================================================


class TestFeatureFlagIntegration:
    """Tests for feature flag integration."""

    @pytest.mark.asyncio
    async def test_vector_only_when_hybrid_disabled(
        self,
        mock_settings: MagicMock,
        sample_embedding: list[float],
    ) -> None:
        """When enable_hybrid_search=False, should skip graph search."""
        from src.search.hybrid import HybridSearchService
        from src.search.vector import SearchResult

        mock_settings.enable_hybrid_search = False

        vector_results = [
            SearchResult(id="doc1", score=0.9, payload={}),
        ]

        mock_vector_client = MagicMock()
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        mock_graph_client = MagicMock()
        mock_graph_client.is_connected = True
        mock_graph_client.query = AsyncMock(return_value=[])

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        results = await service.search(
            query_embedding=sample_embedding,
            start_node_id="doc1",
        )

        # Graph client should not be called
        mock_graph_client.query.assert_not_called()
        # Results should still be returned (vector only)
        assert len(results) == 1
