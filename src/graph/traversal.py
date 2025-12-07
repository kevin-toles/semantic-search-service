"""
Graph traversal module for spider web model.

Implements traversal algorithms for the taxonomy graph based on
TIER_RELATIONSHIP_DIAGRAM.md:
- PARALLEL: Same tier traversal (e.g., Tier 1 ↔ Tier 1)
- PERPENDICULAR: Adjacent tier traversal (e.g., Tier 1 → Tier 2)
- SKIP_TIER: Non-adjacent tier traversal (e.g., Tier 1 ↔ Tier 3)

All traversals are BIDIRECTIONAL - concepts flow in ANY direction.

Design follows:
- Repository pattern (GUIDELINES line 795)
- Duck typing for client (works with Neo4jClient or FakeNeo4jClient)
- BFS for tier-priority traversal
- DFS for deep path exploration
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# =============================================================================
# Enums for Relationship Types and Directions
# =============================================================================


class RelationshipType(Enum):
    """Types of relationships between nodes in the spider web model.

    Based on TIER_RELATIONSHIP_DIAGRAM.md:
    - PARALLEL: Same tier (bidirectional within a tier)
    - PERPENDICULAR: Adjacent tiers (bidirectional between Tier N and N+1)
    - SKIP_TIER: Non-adjacent tiers (bidirectional between Tier N and N+2+)
    """

    PARALLEL = "PARALLEL"
    PERPENDICULAR = "PERPENDICULAR"
    SKIP_TIER = "SKIP_TIER"


class TraversalDirection(Enum):
    """Direction of traversal through tiers.

    - UP: Toward Tier 1 (higher priority)
    - DOWN: Toward higher tier numbers (lower priority)
    - LATERAL: Within same tier
    """

    UP = "UP"
    DOWN = "DOWN"
    LATERAL = "LATERAL"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TraversalResult:
    """Result of a graph traversal step.

    Attributes:
        node_id: The ID of the node reached
        depth: How many hops from the start node
        path: List of node IDs from start to this node
        relationship_type: Type of relationship used to reach this node
        relevance_score: Optional relevance score (0.0 to 1.0)
        tier_id: Optional tier ID this node belongs to
    """

    node_id: str
    depth: int
    path: list[str]
    relationship_type: str
    relevance_score: float | None = None
    tier_id: str | None = None


# =============================================================================
# Protocol for Neo4j Client (Duck Typing)
# =============================================================================


@runtime_checkable
class Neo4jClientLike(Protocol):
    """Protocol for Neo4j client interface.

    Allows GraphTraversal to work with Neo4jClient, FakeNeo4jClient,
    or any other implementation that provides these methods.
    """

    async def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read query."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        ...


# =============================================================================
# GraphTraversal Class
# =============================================================================


class GraphTraversal:
    """Graph traversal engine for the spider web model.

    Provides BFS and DFS traversal algorithms for finding related
    concepts, chapters, and books across the taxonomy.

    Usage:
        traversal = GraphTraversal(client=neo4j_client, max_depth=3)

        # BFS for breadth-first exploration
        results = await traversal.bfs_traverse(start_node_id="ch-1")

        # DFS for depth-first exploration
        results = await traversal.dfs_traverse(start_node_id="ch-1")

        # Find path between concepts
        path = await traversal.find_cross_reference_path(
            from_concept="concept-a",
            to_concept="concept-b",
        )
    """

    DEFAULT_MAX_DEPTH = 3

    def __init__(
        self,
        client: Any,  # Neo4jClientLike but using Any for flexibility
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        """Initialize traversal engine.

        Args:
            client: Neo4j client (Neo4jClient or FakeNeo4jClient)
            max_depth: Maximum traversal depth (default: 3)
        """
        self._client = client
        self._max_depth = max_depth

    # =========================================================================
    # Relationship Type Detection
    # =========================================================================

    def get_relationship_type(
        self,
        tier_from: int,
        tier_to: int,
    ) -> RelationshipType:
        """Determine the relationship type between two tiers.

        Args:
            tier_from: Source tier level (1, 2, 3, etc.)
            tier_to: Target tier level (1, 2, 3, etc.)

        Returns:
            RelationshipType: PARALLEL, PERPENDICULAR, or SKIP_TIER
        """
        tier_diff = abs(tier_from - tier_to)

        if tier_diff == 0:
            return RelationshipType.PARALLEL
        elif tier_diff == 1:
            return RelationshipType.PERPENDICULAR
        else:
            return RelationshipType.SKIP_TIER

    def get_traversal_direction(
        self,
        tier_from: int,
        tier_to: int,
    ) -> TraversalDirection:
        """Determine the direction of traversal between tiers.

        Args:
            tier_from: Source tier level
            tier_to: Target tier level

        Returns:
            TraversalDirection: UP, DOWN, or LATERAL
        """
        if tier_from == tier_to:
            return TraversalDirection.LATERAL
        elif tier_from > tier_to:
            return TraversalDirection.UP
        else:
            return TraversalDirection.DOWN

    # =========================================================================
    # BFS Traversal
    # =========================================================================

    async def bfs_traverse(
        self,
        start_node_id: str,
        max_depth: int | None = None,
    ) -> list[dict[str, Any]]:
        """Breadth-first search traversal from a starting node.

        BFS prioritizes finding all nodes at each depth level before
        going deeper. This is ideal for finding PARALLEL connections
        first (same tier), then PERPENDICULAR (adjacent tier).

        Args:
            start_node_id: ID of the starting node
            max_depth: Override max depth (uses instance default if None)

        Returns:
            List of traversal results with node_id, depth, path, etc.
        """
        depth_limit = max_depth if max_depth is not None else self._max_depth
        results: list[dict[str, Any]] = []
        visited: set[str] = {start_node_id}
        queue: deque[tuple[str, int, list[str]]] = deque()

        # Initialize queue with start node
        queue.append((start_node_id, 0, [start_node_id]))

        while queue:
            current_id, current_depth, path = queue.popleft()

            if current_depth > 0:  # Don't include start node in results
                results.append({
                    "node_id": current_id,
                    "depth": current_depth,
                    "path": path,
                    "relationship_type": self._infer_relationship_type(path),
                })

            if current_depth >= depth_limit:
                continue

            # Get neighbors from graph
            neighbors = await self._get_neighbors(current_id)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [neighbor_id]
                    queue.append((neighbor_id, current_depth + 1, new_path))

        return results

    # =========================================================================
    # DFS Traversal
    # =========================================================================

    async def dfs_traverse(
        self,
        start_node_id: str,
        max_depth: int | None = None,
    ) -> list[dict[str, Any]]:
        """Depth-first search traversal from a starting node.

        DFS explores as deep as possible before backtracking.
        This is useful for finding complete paths through the graph.

        Args:
            start_node_id: ID of the starting node
            max_depth: Override max depth (uses instance default if None)

        Returns:
            List of traversal results with node_id, depth, path, etc.
        """
        depth_limit = max_depth if max_depth is not None else self._max_depth
        results: list[dict[str, Any]] = []
        visited: set[str] = set()

        async def _dfs_recursive(
            node_id: str,
            depth: int,
            path: list[str],
        ) -> None:
            if node_id in visited:
                return
            if depth > depth_limit:
                return

            visited.add(node_id)

            if depth > 0:  # Don't include start node in results
                results.append({
                    "node_id": node_id,
                    "depth": depth,
                    "path": path,
                    "relationship_type": self._infer_relationship_type(path),
                })

            # Get neighbors and recurse
            neighbors = await self._get_neighbors(node_id)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    await _dfs_recursive(
                        neighbor_id,
                        depth + 1,
                        path + [neighbor_id],
                    )

        await _dfs_recursive(start_node_id, 0, [start_node_id])
        return results

    # =========================================================================
    # Cross-Reference Path Finding
    # =========================================================================

    async def find_cross_reference_path(
        self,
        from_concept: str,
        to_concept: str,
        prefer_higher_priority: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Find a path between two concepts in the graph.

        Uses BFS to find the shortest path, optionally preferring
        paths through higher-priority tiers.

        Args:
            from_concept: Starting concept ID
            to_concept: Target concept ID
            prefer_higher_priority: If True, prefer Tier 1 over Tier 2, etc.

        Returns:
            List of path steps with relationship types, or None if no path
        """
        visited: set[str] = {from_concept}
        queue: deque[tuple[str, list[dict[str, Any]]]] = deque()

        # Initialize with start concept
        queue.append((from_concept, []))

        while queue:
            current_id, path_so_far = queue.popleft()

            if current_id == to_concept:
                return path_so_far

            # Get neighbors
            neighbors = await self._get_neighbors(current_id)

            # Sort by tier priority if requested
            if prefer_higher_priority:
                neighbors = await self._sort_by_tier_priority(neighbors)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path_so_far + [{
                        "from_node": current_id,
                        "to_node": neighbor_id,
                        "relationship_type": "CONNECTED",
                    }]
                    queue.append((neighbor_id, new_path))

        return None  # No path found

    # =========================================================================
    # Related Chapter Finding (Spider Web Traversal)
    # =========================================================================

    async def find_related_chapters(
        self,
        source_chapter_id: str,
        max_depth: int | None = None,
        group_by_tier: bool = False,
        include_relevance: bool = False,
    ) -> list[dict[str, Any]]:
        """Find all chapters related to a source chapter.

        This is the main entry point for spider web model traversal.
        Finds chapters across all tiers with various relationship types.

        Args:
            source_chapter_id: ID of the source chapter
            max_depth: Maximum traversal depth
            group_by_tier: If True, include tier information
            include_relevance: If True, include relevance scores

        Returns:
            List of related chapters with metadata
        """
        depth_limit = max_depth if max_depth is not None else self._max_depth

        # Use BFS to find all related chapters
        results = await self.bfs_traverse(
            start_node_id=source_chapter_id,
            max_depth=depth_limit,
        )

        # Enrich with tier information if requested
        if group_by_tier:
            for result in results:
                result["tier"] = await self._get_tier_for_node(result["node_id"])
                result["tier_id"] = result["tier"]

        # Add relevance scores if requested
        if include_relevance:
            for result in results:
                result["relevance"] = self._calculate_relevance(
                    depth=result["depth"],
                    relationship_type=result.get("relationship_type", ""),
                )

        return results

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_neighbors(self, node_id: str) -> list[str]:
        """Get neighboring node IDs from the graph.

        Args:
            node_id: ID of the current node

        Returns:
            List of neighbor node IDs
        """
        cypher = """
        MATCH (n {id: $node_id})-[:RELATED_TO|CONNECTS_TO|HAS_CHAPTER|REFERENCES]-(m)
        RETURN m.id AS neighbor_id
        """
        try:
            results = await self._client.query(
                cypher,
                parameters={"node_id": node_id},
            )
            return [r.get("neighbor_id") for r in results if r.get("neighbor_id")]
        except Exception:
            # Return empty list if query fails
            return []

    async def _get_tier_for_node(self, node_id: str) -> str | None:
        """Get the tier ID for a given node.

        Args:
            node_id: ID of the node

        Returns:
            Tier ID or None if not found
        """
        cypher = """
        MATCH (n {id: $node_id})-[:BELONGS_TO*1..3]-(t:Tier)
        RETURN t.id AS tier_id
        LIMIT 1
        """
        try:
            results = await self._client.query(
                cypher,
                parameters={"node_id": node_id},
            )
            if results:
                return results[0].get("tier_id")
        except Exception:
            pass
        return None

    async def _sort_by_tier_priority(
        self,
        node_ids: list[str],
    ) -> list[str]:
        """Sort node IDs by tier priority (Tier 1 first).

        Args:
            node_ids: List of node IDs to sort

        Returns:
            Sorted list with higher-priority tiers first
        """
        # Query tier for each node and sort
        # For now, return as-is; actual implementation would query tiers
        # Using await to satisfy async requirement
        tier_priorities: dict[str, int] = {}
        for node_id in node_ids:
            tier_id = await self._get_tier_for_node(node_id)
            # Extract tier number (tier-1 -> 1, tier-2 -> 2)
            if tier_id:
                try:
                    tier_num = int(tier_id.split("-")[-1])
                except (ValueError, IndexError):
                    tier_num = 99
            else:
                tier_num = 99
            tier_priorities[node_id] = tier_num

        return sorted(node_ids, key=lambda x: tier_priorities.get(x, 99))

    def _infer_relationship_type(self, path: list[str]) -> str:
        """Infer relationship type from path length.

        This is a simplified implementation; actual implementation
        would query tier information from the graph.

        Args:
            path: List of node IDs in the path

        Returns:
            Relationship type string
        """
        # Simplified logic - actual impl would check tiers
        if len(path) <= 2:
            return RelationshipType.PARALLEL.value
        elif len(path) == 3:
            return RelationshipType.PERPENDICULAR.value
        else:
            return RelationshipType.SKIP_TIER.value

    def _calculate_relevance(
        self,
        depth: int,
        relationship_type: str,
    ) -> float:
        """Calculate relevance score based on depth and relationship type.

        Closer nodes and PARALLEL relationships have higher relevance.

        Args:
            depth: Distance from source node
            relationship_type: Type of relationship

        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Base score from depth (closer = higher score)
        depth_score = max(0.0, 1.0 - (depth * 0.2))

        # Bonus for relationship type
        type_bonus = {
            RelationshipType.PARALLEL.value: 0.2,
            RelationshipType.PERPENDICULAR.value: 0.1,
            RelationshipType.SKIP_TIER.value: 0.0,
        }.get(relationship_type, 0.0)

        return min(1.0, depth_score + type_bonus)
