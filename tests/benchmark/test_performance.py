"""
WBS 6.1 RED: Performance benchmarking tests.

Tests for validating performance targets from GRAPH_RAG_POC_PLAN.md:
- Hybrid search latency < 500ms p95
- Graph traversal within acceptable bounds
- Score fusion computation efficient

TDD Phase: RED - Tests written first, implementation to follow.

Anti-Pattern Mitigations:
- pytest.approx() for float comparisons (S5727)
- No hardcoded timeouts (configurable)
- Proper async test patterns
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.traversal import GraphTraversal, TraversalResult, RelationshipType
from src.search.hybrid import HybridSearchService, HybridSearchResult


# =============================================================================
# Constants for Performance Targets
# =============================================================================

P95_LATENCY_TARGET_MS = 500  # Phase 6 acceptance criteria
MAX_GRAPH_TRAVERSAL_DEPTH = 5  # From WBS 2.4
BENCHMARK_ITERATIONS = 100  # Number of iterations for statistical significance


# =============================================================================
# Benchmark Result Data Class
# =============================================================================


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark run.

    Attributes:
        operation: Name of the operation benchmarked
        iterations: Number of test iterations
        mean_ms: Mean latency in milliseconds
        p50_ms: 50th percentile (median) latency
        p95_ms: 95th percentile latency
        p99_ms: 99th percentile latency
        min_ms: Minimum latency observed
        max_ms: Maximum latency observed
    """

    operation: str
    iterations: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def passes_target(self, target_p95_ms: float) -> bool:
        """Check if benchmark passes the P95 latency target."""
        return self.p95_ms <= target_p95_ms


# =============================================================================
# Benchmark Utility Functions
# =============================================================================


def calculate_percentile(latencies: list[float], percentile: float) -> float:
    """Calculate a percentile from a list of latencies.

    Args:
        latencies: List of latency measurements
        percentile: Percentile to calculate (0-100)

    Returns:
        The percentile value

    Raises:
        ValueError: If latencies is empty
    """
    if not latencies:
        raise ValueError("Cannot calculate percentile of empty list")

    sorted_latencies = sorted(latencies)
    index = int((percentile / 100) * (len(sorted_latencies) - 1))
    return sorted_latencies[index]


def run_benchmark(latencies_ms: list[float], operation: str) -> BenchmarkResult:
    """Calculate benchmark statistics from latency measurements.

    Args:
        latencies_ms: List of latency measurements in milliseconds
        operation: Name of the operation being benchmarked

    Returns:
        BenchmarkResult with calculated statistics
    """
    return BenchmarkResult(
        operation=operation,
        iterations=len(latencies_ms),
        mean_ms=sum(latencies_ms) / len(latencies_ms),
        p50_ms=calculate_percentile(latencies_ms, 50),
        p95_ms=calculate_percentile(latencies_ms, 95),
        p99_ms=calculate_percentile(latencies_ms, 99),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for hybrid search."""
    settings = MagicMock()
    settings.hybrid_vector_weight = 0.7
    settings.hybrid_graph_weight = 0.3
    settings.enable_hybrid_search = True
    return settings


@pytest.fixture
def mock_vector_client() -> AsyncMock:
    """Create a mock vector search client with realistic latency."""
    client = AsyncMock()

    async def mock_search(*args: Any, **kwargs: Any) -> list:
        # Simulate realistic vector search latency (50-100ms)
        await asyncio.sleep(0.05 + 0.05 * (time.time() % 1))
        from src.search.vector import SearchResult

        return [
            SearchResult(id=f"doc_{i}", score=0.9 - i * 0.1, payload={"title": f"Doc {i}"})
            for i in range(min(kwargs.get("limit", 10), 5))
        ]

    client.search = mock_search
    return client


@pytest.fixture
def mock_graph_client() -> AsyncMock:
    """Create a mock graph client with realistic latency."""
    client = AsyncMock()
    client.is_connected = True

    async def mock_query(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        # Simulate realistic graph query latency (20-50ms)
        await asyncio.sleep(0.02 + 0.03 * (time.time() % 1))
        return [
            {
                "n.id": f"node_{i}",
                "r.type": "PARALLEL",
                "length(p)": i + 1,
                "n.tier": f"T{i % 5 + 1}",
            }
            for i in range(3)
        ]

    client.query = mock_query
    return client


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create a mock embedder."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1] * 384  # Standard embedding dimension
    return embedder


# =============================================================================
# Benchmark Tests
# =============================================================================


class TestHybridSearchPerformance:
    """Performance benchmarks for hybrid search."""

    @pytest.mark.asyncio
    async def test_hybrid_search_p95_under_500ms(
        self,
        mock_vector_client: AsyncMock,
        mock_graph_client: AsyncMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test that hybrid search P95 latency is under 500ms target.

        Acceptance Criteria from GRAPH_RAG_POC_PLAN.md:
        - Hybrid search latency < 500ms p95
        """
        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=mock_graph_client,
            settings=mock_settings,
        )

        latencies_ms: list[float] = []
        test_embedding = [0.1] * 384

        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            await service.search(
                query_embedding=test_embedding,
                start_node_id="test_node",
                limit=10,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        result = run_benchmark(latencies_ms, "hybrid_search")

        assert result.passes_target(
            P95_LATENCY_TARGET_MS
        ), f"P95 latency {result.p95_ms:.2f}ms exceeds target {P95_LATENCY_TARGET_MS}ms"

    @pytest.mark.asyncio
    async def test_hybrid_search_graceful_degradation_latency(
        self,
        mock_vector_client: AsyncMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test latency when graph client fails (graceful degradation).

        Should still meet targets with vector-only search.
        """
        # Create failing graph client
        failing_graph_client = AsyncMock()
        failing_graph_client.is_connected = False

        service = HybridSearchService(
            vector_client=mock_vector_client,
            graph_client=failing_graph_client,
            settings=mock_settings,
        )

        latencies_ms: list[float] = []
        test_embedding = [0.1] * 384

        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            await service.search(
                query_embedding=test_embedding,
                start_node_id="test_node",
                limit=10,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        result = run_benchmark(latencies_ms, "hybrid_search_degraded")

        # Should still be fast without graph overhead
        assert result.p95_ms < P95_LATENCY_TARGET_MS


class TestGraphTraversalPerformance:
    """Performance benchmarks for graph traversal."""

    @pytest.mark.asyncio
    async def test_bfs_traversal_performance(
        self,
        mock_graph_client: AsyncMock,
    ) -> None:
        """Test BFS traversal performance within acceptable bounds."""
        traversal = GraphTraversal(client=mock_graph_client)

        latencies_ms: list[float] = []

        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            await traversal.bfs_traverse(
                start_node_id="test_node",
                max_depth=MAX_GRAPH_TRAVERSAL_DEPTH,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        result = run_benchmark(latencies_ms, "bfs_traversal")

        # BFS should complete within 200ms p95 for max depth 5
        assert result.p95_ms < 200, f"BFS P95 {result.p95_ms:.2f}ms exceeds 200ms"

    @pytest.mark.asyncio
    async def test_dfs_traversal_performance(
        self,
        mock_graph_client: AsyncMock,
    ) -> None:
        """Test DFS traversal performance within acceptable bounds."""
        traversal = GraphTraversal(client=mock_graph_client)

        latencies_ms: list[float] = []

        for _ in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            await traversal.dfs_traverse(
                start_node_id="test_node",
                max_depth=MAX_GRAPH_TRAVERSAL_DEPTH,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        result = run_benchmark(latencies_ms, "dfs_traversal")

        # DFS should complete within 200ms p95 for max depth 5
        assert result.p95_ms < 200, f"DFS P95 {result.p95_ms:.2f}ms exceeds 200ms"


class TestScoreFusionPerformance:
    """Performance benchmarks for score fusion computation."""

    def test_score_fusion_computational_overhead(self) -> None:
        """Test that score fusion computation is negligible (<1ms).

        Score fusion is CPU-bound, should not add significant latency.
        """
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()

        # Create test data - using dictionaries for the ranker input format
        vector_scores = {f"doc_{i}": 0.9 - i * 0.01 for i in range(100)}
        graph_scores = {f"doc_{i}": 0.8 - i * 0.008 for i in range(100)}

        latencies_ms: list[float] = []

        for _ in range(1000):  # More iterations for CPU-bound ops
            start = time.perf_counter()
            ranker.fuse(
                vector_scores=vector_scores,
                graph_scores=graph_scores,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        result = run_benchmark(latencies_ms, "score_fusion")

        # Score fusion should be sub-millisecond
        assert result.p95_ms < 1, f"Score fusion P95 {result.p95_ms:.4f}ms exceeds 1ms"


class TestBenchmarkUtilities:
    """Tests for benchmark utility functions."""

    def test_calculate_percentile_p50(self) -> None:
        """Test median calculation."""
        latencies = [100, 200, 300, 400, 500]
        p50 = calculate_percentile(latencies, 50)
        assert p50 == pytest.approx(300, rel=0.1)

    def test_calculate_percentile_p95(self) -> None:
        """Test P95 calculation."""
        latencies = list(range(1, 101))  # 1 to 100
        p95 = calculate_percentile(latencies, 95)
        assert p95 == pytest.approx(95, rel=0.1)

    def test_calculate_percentile_empty_raises(self) -> None:
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate percentile"):
            calculate_percentile([], 50)

    def test_benchmark_result_passes_target(self) -> None:
        """Test BenchmarkResult.passes_target method."""
        result = BenchmarkResult(
            operation="test",
            iterations=100,
            mean_ms=100,
            p50_ms=90,
            p95_ms=450,  # Under 500ms target
            p99_ms=480,
            min_ms=50,
            max_ms=490,
        )
        assert result.passes_target(500) is True
        assert result.passes_target(400) is False

    def test_run_benchmark_statistics(self) -> None:
        """Test run_benchmark calculates correct statistics."""
        latencies = [100, 150, 200, 250, 300]
        result = run_benchmark(latencies, "test_op")

        assert result.operation == "test_op"
        assert result.iterations == 5
        assert result.mean_ms == pytest.approx(200, rel=0.01)
        assert result.min_ms == pytest.approx(100, rel=0.01)
        assert result.max_ms == pytest.approx(300, rel=0.01)
