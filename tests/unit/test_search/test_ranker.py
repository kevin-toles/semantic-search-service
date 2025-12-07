"""
WBS 3.5 RED: Unit tests for ResultRanker.

Tests follow TDD RED phase - all tests should fail initially until
ranker.py is implemented in WBS 3.6.

Based on Pre-Implementation Analysis (WBS 3.0.1-3.0.4):
- Multiple score fusion strategies (linear, harmonic, max)
- Configurable weights for each strategy
- Min-max score normalization
- Reciprocal Rank Fusion (RRF) for combining ranked lists
- Boost factors for relationship types

Anti-Pattern Mitigations Applied:
- Configurable weights (not hardcoded)
- Score validation before fusion
- Division by zero handling
- Extract constants for repeated patterns (_DEFAULT_K, _DEFAULT_ALPHA)
"""

from __future__ import annotations

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_vector_scores() -> list[tuple[str, float]]:
    """Sample (doc_id, vector_score) pairs."""
    return [
        ("doc1", 0.95),
        ("doc2", 0.85),
        ("doc3", 0.75),
        ("doc4", 0.60),
    ]


@pytest.fixture
def sample_graph_scores() -> list[tuple[str, float]]:
    """Sample (doc_id, graph_score) pairs."""
    return [
        ("doc2", 0.90),  # High graph score, also in vector
        ("doc3", 0.70),
        ("doc5", 0.85),  # Only in graph results
    ]


# =============================================================================
# Test: ResultRanker Initialization
# =============================================================================


class TestResultRankerInitialization:
    """Tests for ResultRanker initialization."""

    def test_ranker_default_strategy(self) -> None:
        """ResultRanker should default to linear fusion strategy."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()

        assert ranker.strategy == "linear"

    def test_ranker_accepts_linear_strategy(self) -> None:
        """ResultRanker should accept 'linear' strategy."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="linear")

        assert ranker.strategy == "linear"

    def test_ranker_accepts_rrf_strategy(self) -> None:
        """ResultRanker should accept 'rrf' (Reciprocal Rank Fusion) strategy."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="rrf")

        assert ranker.strategy == "rrf"

    def test_ranker_accepts_max_strategy(self) -> None:
        """ResultRanker should accept 'max' strategy."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="max")

        assert ranker.strategy == "max"

    def test_ranker_rejects_invalid_strategy(self) -> None:
        """ResultRanker should reject invalid strategy names."""
        from src.search.ranker import ResultRanker

        with pytest.raises(ValueError) as exc_info:
            ResultRanker(strategy="invalid")

        assert "invalid" in str(exc_info.value).lower()

    def test_ranker_default_weights(self) -> None:
        """ResultRanker should have configurable default weights."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()

        assert ranker.vector_weight == pytest.approx(0.7)
        assert ranker.graph_weight == pytest.approx(0.3)

    def test_ranker_custom_weights(self) -> None:
        """ResultRanker should accept custom weights."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(vector_weight=0.6, graph_weight=0.4)

        assert ranker.vector_weight == pytest.approx(0.6)
        assert ranker.graph_weight == pytest.approx(0.4)


# =============================================================================
# Test: Linear Fusion
# =============================================================================


class TestLinearFusion:
    """Tests for linear score fusion."""

    def test_linear_fusion_basic(
        self,
        sample_vector_scores: list[tuple[str, float]],
        sample_graph_scores: list[tuple[str, float]],
    ) -> None:
        """Linear fusion should compute α*vector + (1-α)*graph."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(
            strategy="linear",
            vector_weight=0.7,
            graph_weight=0.3,
        )

        results = ranker.fuse(
            vector_scores=dict(sample_vector_scores),
            graph_scores=dict(sample_graph_scores),
        )

        # doc2: 0.7 * 0.85 + 0.3 * 0.90 = 0.595 + 0.27 = 0.865
        assert "doc2" in results
        assert 0.86 <= results["doc2"] <= 0.87

    def test_linear_fusion_vector_only(
        self,
        sample_vector_scores: list[tuple[str, float]],
    ) -> None:
        """Linear fusion should handle docs only in vector results."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="linear", vector_weight=0.7, graph_weight=0.3)

        results = ranker.fuse(
            vector_scores=dict(sample_vector_scores),
            graph_scores={},  # Empty graph scores
        )

        # doc1: 0.7 * 0.95 + 0.3 * 0.0 = 0.665
        assert "doc1" in results
        assert 0.66 <= results["doc1"] <= 0.67

    def test_linear_fusion_graph_only(
        self,
        sample_graph_scores: list[tuple[str, float]],
    ) -> None:
        """Linear fusion should handle docs only in graph results."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="linear", vector_weight=0.7, graph_weight=0.3)

        results = ranker.fuse(
            vector_scores={},  # Empty vector scores
            graph_scores=dict(sample_graph_scores),
        )

        # doc5: 0.7 * 0.0 + 0.3 * 0.85 = 0.255
        assert "doc5" in results
        assert 0.25 <= results["doc5"] <= 0.26


# =============================================================================
# Test: Reciprocal Rank Fusion (RRF)
# =============================================================================


class TestReciprocalRankFusion:
    """Tests for Reciprocal Rank Fusion (RRF)."""

    def test_rrf_basic(
        self,
        sample_vector_scores: list[tuple[str, float]],
        sample_graph_scores: list[tuple[str, float]],
    ) -> None:
        """RRF should combine rankings using 1/(k + rank) formula."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="rrf", rrf_k=60)

        results = ranker.fuse(
            vector_scores=dict(sample_vector_scores),
            graph_scores=dict(sample_graph_scores),
        )

        # All docs should have RRF scores
        assert len(results) > 0
        # Scores should be positive
        assert all(score > 0 for score in results.values())

    def test_rrf_k_parameter(self) -> None:
        """RRF should accept configurable k parameter."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="rrf", rrf_k=10)

        assert ranker.rrf_k == 10

    def test_rrf_default_k(self) -> None:
        """RRF should have default k=60."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="rrf")

        assert ranker.rrf_k == 60

    def test_rrf_higher_k_smooths_ranking(
        self,
        sample_vector_scores: list[tuple[str, float]],
        sample_graph_scores: list[tuple[str, float]],
    ) -> None:
        """Higher k values should produce more similar scores."""
        from src.search.ranker import ResultRanker

        ranker_low_k = ResultRanker(strategy="rrf", rrf_k=10)
        ranker_high_k = ResultRanker(strategy="rrf", rrf_k=100)

        results_low = ranker_low_k.fuse(
            vector_scores=dict(sample_vector_scores),
            graph_scores=dict(sample_graph_scores),
        )
        results_high = ranker_high_k.fuse(
            vector_scores=dict(sample_vector_scores),
            graph_scores=dict(sample_graph_scores),
        )

        # Higher k should have smaller score variance
        low_variance = max(results_low.values()) - min(results_low.values())
        high_variance = max(results_high.values()) - min(results_high.values())

        assert high_variance < low_variance


# =============================================================================
# Test: Max Score Fusion
# =============================================================================


class TestMaxFusion:
    """Tests for max score fusion."""

    def test_max_fusion_takes_highest(
        self,
        sample_vector_scores: list[tuple[str, float]],
        sample_graph_scores: list[tuple[str, float]],
    ) -> None:
        """Max fusion should take the highest score from any source."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="max")

        results = ranker.fuse(
            vector_scores=dict(sample_vector_scores),
            graph_scores=dict(sample_graph_scores),
        )

        # doc2: max(0.85, 0.90) = 0.90
        assert "doc2" in results
        assert results["doc2"] == pytest.approx(0.90)

    def test_max_fusion_vector_only(
        self,
        sample_vector_scores: list[tuple[str, float]],
    ) -> None:
        """Max fusion should work with vector-only results."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="max")

        results = ranker.fuse(
            vector_scores=dict(sample_vector_scores),
            graph_scores={},
        )

        # doc1: max(0.95, 0) = 0.95
        assert results["doc1"] == pytest.approx(0.95)


# =============================================================================
# Test: Score Normalization
# =============================================================================


class TestScoreNormalization:
    """Tests for score normalization methods."""

    def test_min_max_normalization(self) -> None:
        """min_max_normalize should scale scores to [0, 1]."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()
        scores = {"doc1": 10.0, "doc2": 50.0, "doc3": 100.0}

        normalized = ranker.min_max_normalize(scores)

        assert normalized["doc1"] == pytest.approx(0.0)  # min
        assert normalized["doc3"] == pytest.approx(1.0)  # max
        # doc2: (50 - 10) / (100 - 10) = 40/90 ≈ 0.444
        assert 0.44 <= normalized["doc2"] <= 0.45

    def test_min_max_handles_single_value(self) -> None:
        """min_max_normalize should handle single-value dict."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()
        scores = {"doc1": 0.5}

        normalized = ranker.min_max_normalize(scores)

        # Single value should normalize to 1.0
        assert normalized["doc1"] == pytest.approx(1.0)

    def test_min_max_handles_empty(self) -> None:
        """min_max_normalize should handle empty dict."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()
        scores: dict[str, float] = {}

        normalized = ranker.min_max_normalize(scores)

        assert normalized == {}

    def test_min_max_handles_same_values(self) -> None:
        """min_max_normalize should handle all same values."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()
        scores = {"doc1": 0.5, "doc2": 0.5, "doc3": 0.5}

        normalized = ranker.min_max_normalize(scores)

        # All same values should normalize to 1.0
        assert all(v == pytest.approx(1.0) for v in normalized.values())


# =============================================================================
# Test: Relationship Type Boost
# =============================================================================


class TestRelationshipTypeBoost:
    """Tests for relationship type boosting."""

    def test_parallel_relationship_boost(self) -> None:
        """PARALLEL relationships should receive configurable boost."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(
            relationship_boosts={
                "PARALLEL": 1.2,
                "PERPENDICULAR": 1.0,
                "SKIP_TIER": 0.8,
            }
        )

        score = ranker.apply_relationship_boost(
            base_score=0.5,
            relationship_type="PARALLEL",
        )

        # Base score times boost factor
        assert score == pytest.approx(0.6)

    def test_skip_tier_relationship_penalty(self) -> None:
        """SKIP_TIER relationships can have penalty (boost < 1)."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(
            relationship_boosts={
                "PARALLEL": 1.2,
                "PERPENDICULAR": 1.0,
                "SKIP_TIER": 0.8,
            }
        )

        score = ranker.apply_relationship_boost(
            base_score=0.5,
            relationship_type="SKIP_TIER",
        )

        # Base score times penalty factor
        assert score == pytest.approx(0.4)

    def test_unknown_relationship_no_boost(self) -> None:
        """Unknown relationship types should get no boost (1.0)."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(
            relationship_boosts={
                "PARALLEL": 1.2,
            }
        )

        score = ranker.apply_relationship_boost(
            base_score=0.5,
            relationship_type="UNKNOWN",
        )

        # No boost, score unchanged
        assert score == pytest.approx(0.5)

    def test_none_relationship_no_boost(self) -> None:
        """None relationship type should get no boost."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()

        score = ranker.apply_relationship_boost(
            base_score=0.5,
            relationship_type=None,
        )

        assert score == pytest.approx(0.5)


# =============================================================================
# Test: Rank Method
# =============================================================================


class TestRankMethod:
    """Tests for the rank() convenience method."""

    def test_rank_returns_sorted_list(
        self,
        sample_vector_scores: list[tuple[str, float]],
        sample_graph_scores: list[tuple[str, float]],
    ) -> None:
        """rank() should return results sorted by score descending."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="linear")

        ranked = ranker.rank(
            vector_scores=dict(sample_vector_scores),
            graph_scores=dict(sample_graph_scores),
        )

        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_respects_limit(
        self,
        sample_vector_scores: list[tuple[str, float]],
        sample_graph_scores: list[tuple[str, float]],
    ) -> None:
        """rank() should respect limit parameter."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()

        ranked = ranker.rank(
            vector_scores=dict(sample_vector_scores),
            graph_scores=dict(sample_graph_scores),
            limit=3,
        )

        assert len(ranked) == 3

    def test_rank_returns_tuples(
        self,
        sample_vector_scores: list[tuple[str, float]],
        sample_graph_scores: list[tuple[str, float]],
    ) -> None:
        """rank() should return list of (doc_id, score) tuples."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()

        ranked = ranker.rank(
            vector_scores=dict(sample_vector_scores),
            graph_scores=dict(sample_graph_scores),
        )

        assert all(isinstance(item, tuple) for item in ranked)
        assert all(len(item) == 2 for item in ranked)
        assert all(isinstance(item[0], str) for item in ranked)
        assert all(isinstance(item[1], float) for item in ranked)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_inputs(self) -> None:
        """Should handle empty inputs gracefully."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker()

        results = ranker.fuse(
            vector_scores={},
            graph_scores={},
        )

        assert results == {}

    def test_negative_scores_clamped(self) -> None:
        """Negative scores should be clamped to 0."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="linear")

        results = ranker.fuse(
            vector_scores={"doc1": -0.5},
            graph_scores={"doc1": 0.5},
        )

        # -0.5 should be treated as 0
        # 0.7 * 0 + 0.3 * 0.5 = 0.15
        assert results["doc1"] >= 0

    def test_scores_above_one_clamped(self) -> None:
        """Scores above 1.0 should be clamped to 1.0."""
        from src.search.ranker import ResultRanker

        ranker = ResultRanker(strategy="max")

        results = ranker.fuse(
            vector_scores={"doc1": 1.5},  # Above 1.0
            graph_scores={},
        )

        assert results["doc1"] == pytest.approx(1.0)
