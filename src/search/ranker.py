"""
WBS 3.6 GREEN: Result ranker implementation.

Provides multiple score fusion strategies for combining vector
and graph search results.

Strategies:
- linear: Weighted linear combination (α*vector + (1-α)*graph)
- rrf: Reciprocal Rank Fusion - combines rankings
- max: Takes maximum score from any source

Design follows Pre-Implementation Analysis (WBS 3.0.1-3.0.4):
- Multiple fusion strategies with configurable selection
- Min-max normalization for score comparability
- Relationship type boosting for graph results
- Extract constants for repeated patterns

Anti-Pattern Mitigations Applied:
- Configurable weights (not hardcoded)
- Score validation and clamping
- Division by zero handling
- Constants for repeated values (_DEFAULT_K, _DEFAULT_WEIGHTS)
"""

from __future__ import annotations

# =============================================================================
# Constants (Anti-Pattern #50: Duplicated Literals)
# =============================================================================

_DEFAULT_STRATEGY = "linear"
_DEFAULT_VECTOR_WEIGHT = 0.7
_DEFAULT_GRAPH_WEIGHT = 0.3
_DEFAULT_RRF_K = 60
_VALID_STRATEGIES = {"linear", "rrf", "max"}

_DEFAULT_RELATIONSHIP_BOOSTS: dict[str, float] = {
    "PARALLEL": 1.0,
    "PERPENDICULAR": 1.0,
    "SKIP_TIER": 1.0,
}


# =============================================================================
# ResultRanker Class
# =============================================================================


class ResultRanker:
    """Configurable result ranker with multiple fusion strategies.

    Supports:
    - Linear fusion: α*vector + (1-α)*graph
    - RRF (Reciprocal Rank Fusion): Combines rankings, not raw scores
    - Max: Takes the maximum score from any source

    Usage:
        ranker = ResultRanker(strategy="linear", vector_weight=0.7)
        fused = ranker.fuse(vector_scores, graph_scores)
        ranked = ranker.rank(vector_scores, graph_scores, limit=10)
    """

    def __init__(
        self,
        strategy: str = _DEFAULT_STRATEGY,
        vector_weight: float = _DEFAULT_VECTOR_WEIGHT,
        graph_weight: float = _DEFAULT_GRAPH_WEIGHT,
        rrf_k: int = _DEFAULT_RRF_K,
        relationship_boosts: dict[str, float] | None = None,
    ) -> None:
        """Initialize the ranker with configuration.

        Args:
            strategy: Fusion strategy ("linear", "rrf", "max")
            vector_weight: Weight for vector scores (default: 0.7)
            graph_weight: Weight for graph scores (default: 0.3)
            rrf_k: K parameter for RRF (default: 60)
            relationship_boosts: Boost multipliers by relationship type

        Raises:
            ValueError: If strategy is invalid
        """
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid options: {', '.join(sorted(_VALID_STRATEGIES))}"
            )

        self._strategy = strategy
        self._vector_weight = vector_weight
        self._graph_weight = graph_weight
        self._rrf_k = rrf_k
        self._relationship_boosts = (
            relationship_boosts
            if relationship_boosts is not None
            else _DEFAULT_RELATIONSHIP_BOOSTS.copy()
        )

    @property
    def strategy(self) -> str:
        """Get the current fusion strategy."""
        return self._strategy

    @property
    def vector_weight(self) -> float:
        """Get the vector score weight."""
        return self._vector_weight

    @property
    def graph_weight(self) -> float:
        """Get the graph score weight."""
        return self._graph_weight

    @property
    def rrf_k(self) -> int:
        """Get the RRF k parameter."""
        return self._rrf_k

    def fuse(
        self,
        vector_scores: dict[str, float],
        graph_scores: dict[str, float],
    ) -> dict[str, float]:
        """Fuse vector and graph scores using the configured strategy.

        Args:
            vector_scores: Dictionary of {doc_id: score} from vector search
            graph_scores: Dictionary of {doc_id: score} from graph search

        Returns:
            Dictionary of {doc_id: fused_score}
        """
        if self._strategy == "linear":
            return self._linear_fusion(vector_scores, graph_scores)
        elif self._strategy == "rrf":
            return self._rrf_fusion(vector_scores, graph_scores)
        elif self._strategy == "max":
            return self._max_fusion(vector_scores, graph_scores)
        else:
            # Should not reach here due to validation in __init__
            return self._linear_fusion(vector_scores, graph_scores)

    def rank(
        self,
        vector_scores: dict[str, float],
        graph_scores: dict[str, float],
        limit: int | None = None,
    ) -> list[tuple[str, float]]:
        """Fuse and rank results by score descending.

        Args:
            vector_scores: Dictionary of {doc_id: score} from vector search
            graph_scores: Dictionary of {doc_id: score} from graph search
            limit: Maximum number of results to return

        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        fused = self.fuse(vector_scores, graph_scores)

        # Sort by score descending, then by ID for tie-breaking
        ranked = sorted(
            fused.items(),
            key=lambda x: (-x[1], x[0]),
        )

        if limit is not None:
            ranked = ranked[:limit]

        return ranked

    def min_max_normalize(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Normalize scores to [0, 1] using min-max scaling.

        Args:
            scores: Dictionary of {doc_id: raw_score}

        Returns:
            Dictionary of {doc_id: normalized_score}
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)

        # Handle case where all scores are the same
        if max_score == min_score:
            return dict.fromkeys(scores, 1.0)

        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in scores.items()
        }

    def apply_relationship_boost(
        self,
        base_score: float,
        relationship_type: str | None,
    ) -> float:
        """Apply boost multiplier based on relationship type.

        Args:
            base_score: Original score
            relationship_type: Type of graph relationship (or None)

        Returns:
            Boosted score
        """
        if relationship_type is None:
            return base_score

        boost = self._relationship_boosts.get(relationship_type, 1.0)
        return base_score * boost

    def _clamp_score(self, score: float) -> float:
        """Clamp score to [0, 1] range.

        Args:
            score: Raw score

        Returns:
            Score clamped to [0, 1]
        """
        return max(0.0, min(1.0, score))

    def _linear_fusion(
        self,
        vector_scores: dict[str, float],
        graph_scores: dict[str, float],
    ) -> dict[str, float]:
        """Linear weighted fusion.

        Formula: α*vector_score + (1-α)*graph_score
        """
        # Get all unique doc IDs
        all_doc_ids = set(vector_scores.keys()) | set(graph_scores.keys())

        result = {}
        for doc_id in all_doc_ids:
            v_score = self._clamp_score(vector_scores.get(doc_id, 0.0))
            g_score = self._clamp_score(graph_scores.get(doc_id, 0.0))

            fused = self._vector_weight * v_score + self._graph_weight * g_score
            result[doc_id] = self._clamp_score(fused)

        return result

    def _rrf_fusion(
        self,
        vector_scores: dict[str, float],
        graph_scores: dict[str, float],
    ) -> dict[str, float]:
        """Reciprocal Rank Fusion.

        Formula: sum(1 / (k + rank)) for each source where doc appears

        Higher k values smooth the ranking differences.
        """
        # Convert scores to rankings (1-indexed)
        vector_ranking = self._scores_to_ranking(vector_scores)
        graph_ranking = self._scores_to_ranking(graph_scores)

        # Get all unique doc IDs
        all_doc_ids = set(vector_scores.keys()) | set(graph_scores.keys())

        result = {}
        for doc_id in all_doc_ids:
            rrf_score = 0.0

            if doc_id in vector_ranking:
                rrf_score += 1.0 / (self._rrf_k + vector_ranking[doc_id])

            if doc_id in graph_ranking:
                rrf_score += 1.0 / (self._rrf_k + graph_ranking[doc_id])

            result[doc_id] = rrf_score

        return result

    def _scores_to_ranking(
        self,
        scores: dict[str, float],
    ) -> dict[str, int]:
        """Convert scores to 1-indexed rankings.

        Args:
            scores: Dictionary of {doc_id: score}

        Returns:
            Dictionary of {doc_id: rank} where rank=1 is best
        """
        if not scores:
            return {}

        # Sort by score descending
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])

        return {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_items)}

    def _max_fusion(
        self,
        vector_scores: dict[str, float],
        graph_scores: dict[str, float],
    ) -> dict[str, float]:
        """Max score fusion - takes the maximum from any source."""
        # Get all unique doc IDs
        all_doc_ids = set(vector_scores.keys()) | set(graph_scores.keys())

        result = {}
        for doc_id in all_doc_ids:
            v_score = self._clamp_score(vector_scores.get(doc_id, 0.0))
            g_score = self._clamp_score(graph_scores.get(doc_id, 0.0))

            result[doc_id] = max(v_score, g_score)

        return result
