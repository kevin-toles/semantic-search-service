"""
WBS 3.4 GREEN: Hybrid search implementation.

Combines vector similarity search (Qdrant) with graph relationship
traversal (Neo4j) using configurable score fusion.

Design follows Pre-Implementation Analysis (WBS 3.0.1-3.0.4):
- Linear score fusion: final = α*vector_score + (1-α)*graph_score
- Default weights: vector=0.7, graph=0.3
- Score normalization to [0, 1] range before fusion
- Graceful degradation if graph search fails
- Deterministic tie-breaking by ID

Anti-Pattern Mitigations Applied:
- Configurable weights (not hardcoded equal weighting)
- Score validation before fusion
- Division by zero handling
- Deterministic tie-breaking by ID
- No empty f-strings (use regular strings for static messages)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.search.vector import SearchResult

# =============================================================================
# Constants
# =============================================================================

_DEFAULT_VECTOR_WEIGHT = 0.7
_DEFAULT_GRAPH_WEIGHT = 0.3
_DEFAULT_LIMIT = 10
_WEIGHT_TOLERANCE = 0.001  # Allow small float variance when checking sum


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class HybridSearchResult:
    """Represents a single hybrid search result.

    Combines vector similarity score with graph relationship score
    into a fused hybrid score.

    Attributes:
        id: Unique identifier for the document
        vector_score: Score from vector similarity search [0, 1]
        graph_score: Score from graph relationship traversal [0, 1]
        hybrid_score: Fused score using configured weights [0, 1]
        payload: Document metadata and content
        relationship_type: Type of graph relationship (if found via graph)
        depth: Graph traversal depth (if found via graph)
    """

    id: str
    vector_score: float
    graph_score: float
    hybrid_score: float
    payload: dict[str, Any] | None = None
    relationship_type: str | None = None
    depth: int | None = None


# =============================================================================
# Protocols for Duck Typing
# =============================================================================


@runtime_checkable
class VectorSearchClientProtocol(Protocol):
    """Protocol for vector search client interface."""

    async def search(
        self,
        embedding: list[float],
        limit: int | None = None,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Execute vector similarity search."""
        ...


@runtime_checkable
class GraphClientProtocol(Protocol):
    """Protocol for graph client interface."""

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        ...

    async def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute graph query."""
        ...


# =============================================================================
# HybridSearchService
# =============================================================================


class HybridSearchService:
    """Service for hybrid vector + graph search.

    Combines semantic similarity (vector search) with structural
    relationships (graph traversal) using configurable score fusion.

    Score fusion formula: hybrid = α*vector + (1-α)*graph
    where α is the vector weight (default: 0.7)

    Usage:
        service = HybridSearchService(
            vector_client=qdrant_client,
            graph_client=neo4j_client,
            settings=settings,
        )

        results = await service.search(
            query_embedding=[0.1] * 384,
            start_node_id="doc1",
            limit=10,
        )
    """

    def __init__(
        self,
        vector_client: Any,  # VectorSearchClientProtocol
        graph_client: Any,  # GraphClientProtocol
        settings: Any,
    ) -> None:
        """Initialize hybrid search service.

        Args:
            vector_client: Client for vector similarity search (Qdrant)
            graph_client: Client for graph traversal (Neo4j)
            settings: Application settings with hybrid search configuration

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        self._vector_client = vector_client
        self._graph_client = graph_client
        self._settings = settings

        # Get weights from settings or use defaults
        vector_weight = getattr(settings, "hybrid_vector_weight", None)
        graph_weight = getattr(settings, "hybrid_graph_weight", None)

        if vector_weight is None:
            self._vector_weight = _DEFAULT_VECTOR_WEIGHT
        else:
            self._vector_weight = vector_weight

        if graph_weight is None:
            self._graph_weight = _DEFAULT_GRAPH_WEIGHT
        else:
            self._graph_weight = graph_weight

        # Validate weights sum to 1.0 (with tolerance for float precision)
        weight_sum = self._vector_weight + self._graph_weight
        if abs(weight_sum - 1.0) > _WEIGHT_TOLERANCE:
            raise ValueError(
                f"Weights must sum to 1.0, got {self._vector_weight} + "
                f"{self._graph_weight} = {weight_sum}"
            )

        # Check if hybrid search is enabled
        self._hybrid_enabled = getattr(settings, "enable_hybrid_search", True)

    async def search(
        self,
        query_embedding: list[float],
        start_node_id: str,
        limit: int | None = None,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[HybridSearchResult]:
        """Execute hybrid search combining vector and graph sources.

        Args:
            query_embedding: Query vector for similarity search
            start_node_id: Starting node ID for graph traversal
            limit: Maximum number of results (default: 10)
            filter_conditions: Optional metadata filters for vector search
            score_threshold: Optional minimum score threshold

        Returns:
            List of HybridSearchResult objects sorted by hybrid score

        Raises:
            QdrantSearchError: If vector search fails
        """
        actual_limit = limit if limit is not None else _DEFAULT_LIMIT

        # Execute vector search (primary source)
        vector_results = await self._vector_client.search(
            embedding=query_embedding,
            limit=actual_limit * 2,  # Fetch more to allow for merging
            filter_conditions=filter_conditions,
            score_threshold=score_threshold,
        )

        # Execute graph search if enabled and connected
        graph_results: list[dict[str, Any]] = []
        if self._hybrid_enabled and self._is_graph_available():
            try:
                graph_results = await self._execute_graph_search(
                    start_node_id=start_node_id,
                    max_depth=3,
                )
            except Exception:
                # Graceful degradation - continue with vector only
                graph_results = []

        # Merge and fuse scores
        merged = self._merge_results(vector_results, graph_results)

        # Calculate hybrid scores
        hybrid_results = self._calculate_hybrid_scores(merged)

        # Sort by hybrid score descending, then by ID for tie-breaking
        hybrid_results.sort(key=lambda r: (-r.hybrid_score, r.id))

        return hybrid_results[:actual_limit]

    def _is_graph_available(self) -> bool:
        """Check if graph client is available and connected."""
        if self._graph_client is None:
            return False
        return getattr(self._graph_client, "is_connected", False)

    async def _execute_graph_search(
        self,
        start_node_id: str,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """Execute graph traversal to find related nodes.

        Args:
            start_node_id: Starting node for traversal
            max_depth: Maximum traversal depth

        Returns:
            List of graph traversal results
        """
        # Cypher query to find related nodes within max_depth
        cypher = """
        MATCH path = (start)-[r*1..{max_depth}]-(end)
        WHERE start.id = $start_node_id
        RETURN
            end.id AS node_id,
            length(path) AS depth,
            type(r[-1]) AS relationship_type,
            1.0 / (length(path) + 1) AS relevance_score
        ORDER BY relevance_score DESC
        LIMIT 20
        """.replace("{max_depth}", str(max_depth))

        try:
            results = await self._graph_client.query(
                cypher=cypher,
                parameters={"start_node_id": start_node_id},
            )
            return results
        except Exception:
            return []

    def _merge_results(
        self,
        vector_results: list[SearchResult],
        graph_results: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Merge vector and graph results by document ID.

        Args:
            vector_results: Results from vector search
            graph_results: Results from graph traversal

        Returns:
            Dictionary mapping doc ID to merged scores and metadata
        """
        merged: dict[str, dict[str, Any]] = {}

        # Add vector results
        for result in vector_results:
            merged[result.id] = {
                "id": result.id,
                "vector_score": self._normalize_score(result.score),
                "graph_score": 0.0,
                "payload": result.payload,
                "relationship_type": None,
                "depth": None,
            }

        # Add/merge graph results
        for result in graph_results:
            node_id = result.get("node_id")
            if node_id is None:
                continue

            graph_score = self._normalize_score(
                result.get("relevance_score", 0.0)
            )
            relationship_type = result.get("relationship_type")
            depth = result.get("depth")

            if node_id in merged:
                # Merge with existing vector result
                merged[node_id]["graph_score"] = graph_score
                merged[node_id]["relationship_type"] = relationship_type
                merged[node_id]["depth"] = depth
            else:
                # New result from graph only
                merged[node_id] = {
                    "id": node_id,
                    "vector_score": 0.0,
                    "graph_score": graph_score,
                    "payload": None,  # No payload from graph
                    "relationship_type": relationship_type,
                    "depth": depth,
                }

        return merged

    def _normalize_score(self, score: float | None) -> float:
        """Normalize score to [0, 1] range.

        Args:
            score: Raw score (may be negative or > 1)

        Returns:
            Score clamped to [0, 1]
        """
        if score is None:
            return 0.0

        # Clamp to [0, 1]
        return max(0.0, min(1.0, float(score)))

    def _calculate_hybrid_scores(
        self,
        merged: dict[str, dict[str, Any]],
    ) -> list[HybridSearchResult]:
        """Calculate hybrid scores using linear fusion.

        hybrid = α*vector_score + (1-α)*graph_score

        Args:
            merged: Dictionary of merged results

        Returns:
            List of HybridSearchResult objects
        """
        results = []

        for doc_data in merged.values():
            vector_score = doc_data["vector_score"]
            graph_score = doc_data["graph_score"]

            # Linear score fusion
            hybrid_score = (
                self._vector_weight * vector_score +
                self._graph_weight * graph_score
            )

            results.append(
                HybridSearchResult(
                    id=doc_data["id"],
                    vector_score=vector_score,
                    graph_score=graph_score,
                    hybrid_score=hybrid_score,
                    payload=doc_data.get("payload"),
                    relationship_type=doc_data.get("relationship_type"),
                    depth=doc_data.get("depth"),
                )
            )

        return results
