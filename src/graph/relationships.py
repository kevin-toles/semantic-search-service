"""EEP-4 Graph Relationships Module.

Implements relationship edge creation for the spider web model.

WBS: EEP-4.1 - Create relationship edges in Neo4j
- AC-4.1.1: PARALLEL edges (same tier)
- AC-4.1.2: PERPENDICULAR edges (adjacent tiers)
- AC-4.1.3: SKIP_TIER edges (non-adjacent tiers)
- AC-4.1.4: All edges bidirectional per TIER_RELATIONSHIP_DIAGRAM.md

Design follows:
- Repository Pattern (GUIDELINES line 795)
- FakeClient for testing (GUIDELINES line 276)
- Anti-Pattern #7: Custom exception names (no shadowing builtins)
- Anti-Pattern #12: Connection pooling (reuse client)
- Anti-Pattern S1192: Use constants for repeated strings
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


# =============================================================================
# Constants (Anti-Pattern S1192: No duplicate string literals)
# =============================================================================

RELATIONSHIP_PARALLEL = "PARALLEL"
RELATIONSHIP_PERPENDICULAR = "PERPENDICULAR"
RELATIONSHIP_SKIP_TIER = "SKIP_TIER"


# =============================================================================
# Enums and Data Classes
# =============================================================================


class RelationshipEdgeType(Enum):
    """Types of relationship edges in the spider web model.

    Based on TIER_RELATIONSHIP_DIAGRAM.md:
    - PARALLEL: Same tier (bidirectional within a tier)
    - PERPENDICULAR: Adjacent tiers (bidirectional between Tier N and N+1)
    - SKIP_TIER: Non-adjacent tiers (bidirectional between Tier N and N+2+)
    """

    PARALLEL = RELATIONSHIP_PARALLEL
    PERPENDICULAR = RELATIONSHIP_PERPENDICULAR
    SKIP_TIER = RELATIONSHIP_SKIP_TIER


@dataclass
class RelationshipEdge:
    """Represents a relationship edge between two nodes.

    Attributes:
        from_id: Source chapter/node ID
        to_id: Target chapter/node ID
        edge_type: Type of relationship (PARALLEL, PERPENDICULAR, SKIP_TIER)
        from_tier: Tier of the source node
        to_tier: Tier of the target node
    """

    from_id: str
    to_id: str
    edge_type: RelationshipEdgeType
    from_tier: int
    to_tier: int


@dataclass
class RelatedChapter:
    """A related chapter from traversal."""

    chapter_id: str
    tier: int
    relationship_type: str
    path: list[str] = field(default_factory=list)
    target_tier: int | None = None

    def __post_init__(self) -> None:
        """Set target_tier from tier if not provided."""
        if self.target_tier is None:
            self.target_tier = self.tier


@dataclass
class RelationshipResult:
    """Result of relationship query for a chapter.

    Attributes:
        chapter_id: The chapter ID queried
        relationships: List of related chapters with their relationships
        total_count: Total number of relationships found
    """

    chapter_id: str
    relationships: list[RelatedChapter] = field(default_factory=list)
    total_count: int = 0


# =============================================================================
# Protocol for Neo4j Client (Duck Typing)
# =============================================================================


@runtime_checkable
class Neo4jClientLike(Protocol):
    """Protocol for Neo4j client interface."""

    async def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute read query."""
        ...

    async def execute_write(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute write query."""
        ...


# =============================================================================
# Edge Creation Functions (AC-4.1.1 to AC-4.1.3) - Synchronous
# =============================================================================


def create_parallel_edges(
    books: list[dict[str, Any]],
) -> list[RelationshipEdge]:
    """Create PARALLEL edges between chapters within the same tier.

    AC-4.1.1: PARALLEL edges connect chapters in books of the same tier.
    AC-4.1.4: All edges are bidirectional.

    Per TIER_RELATIONSHIP_DIAGRAM.md:
    - Tier 1 ↔ Tier 1 (between books in same tier)
    - Example: Python Distilled ↔ Philosophy of Software Design

    Args:
        books: List of book dicts with 'id', 'title', 'tier' keys

    Returns:
        List of RelationshipEdge objects created (bidirectional = 2 edges per pair)
    """
    edges: list[RelationshipEdge] = []

    # Group books by tier
    books_by_tier: dict[int, list[dict[str, Any]]] = {}
    for book in books:
        tier = book.get("tier", 0)
        if tier not in books_by_tier:
            books_by_tier[tier] = []
        books_by_tier[tier].append(book)

    # Create bidirectional edges between all pairs within each tier
    for tier, tier_books in books_by_tier.items():
        if len(tier_books) < 2:
            continue  # Need at least 2 books for parallel edges

        # Create bidirectional edges for all pairs
        for i, source_book in enumerate(tier_books):
            for target_book in tier_books[i + 1:]:
                source_id = source_book.get("id", "")
                target_id = target_book.get("id", "")

                if not source_id or not target_id:
                    continue

                # Forward edge: source -> target
                edges.append(
                    RelationshipEdge(
                        from_id=source_id,
                        to_id=target_id,
                        edge_type=RelationshipEdgeType.PARALLEL,
                        from_tier=tier,
                        to_tier=tier,
                    )
                )

                # Reverse edge: target -> source (bidirectional)
                edges.append(
                    RelationshipEdge(
                        from_id=target_id,
                        to_id=source_id,
                        edge_type=RelationshipEdgeType.PARALLEL,
                        from_tier=tier,
                        to_tier=tier,
                    )
                )

    return edges


def create_perpendicular_edges(
    books: list[dict[str, Any]],
) -> list[RelationshipEdge]:
    """Create PERPENDICULAR edges between chapters in adjacent tiers.

    AC-4.1.2: PERPENDICULAR edges connect chapters in adjacent tiers.
    AC-4.1.4: All edges are bidirectional.

    Per TIER_RELATIONSHIP_DIAGRAM.md:
    - Tier 1 ↔ Tier 2 (adjacent tiers)
    - Tier 2 ↔ Tier 3 (adjacent tiers)

    Args:
        books: List of book dicts with 'id', 'title', 'tier' keys

    Returns:
        List of RelationshipEdge objects created (bidirectional = 2 edges per pair)
    """
    edges: list[RelationshipEdge] = []

    # Group books by tier
    books_by_tier: dict[int, list[dict[str, Any]]] = {}
    for book in books:
        tier = book.get("tier", 0)
        if tier not in books_by_tier:
            books_by_tier[tier] = []
        books_by_tier[tier].append(book)

    # Get sorted list of tiers
    sorted_tiers = sorted(books_by_tier.keys())

    # Create bidirectional edges between adjacent tiers
    for i, tier in enumerate(sorted_tiers[:-1]):
        next_tier = sorted_tiers[i + 1]

        # Only create edges if tiers are truly adjacent (diff of 1)
        if next_tier - tier != 1:
            continue

        # Create bidirectional edges between all books in adjacent tiers
        for source_book in books_by_tier[tier]:
            for target_book in books_by_tier[next_tier]:
                source_id = source_book.get("id", "")
                target_id = target_book.get("id", "")

                if not source_id or not target_id:
                    continue

                # Forward edge: tier N -> tier N+1
                edges.append(
                    RelationshipEdge(
                        from_id=source_id,
                        to_id=target_id,
                        edge_type=RelationshipEdgeType.PERPENDICULAR,
                        from_tier=tier,
                        to_tier=next_tier,
                    )
                )

                # Reverse edge: tier N+1 -> tier N (bidirectional)
                edges.append(
                    RelationshipEdge(
                        from_id=target_id,
                        to_id=source_id,
                        edge_type=RelationshipEdgeType.PERPENDICULAR,
                        from_tier=next_tier,
                        to_tier=tier,
                    )
                )

    return edges


def create_skip_tier_edges(
    books: list[dict[str, Any]],
) -> list[RelationshipEdge]:
    """Create SKIP_TIER edges between chapters in non-adjacent tiers.

    AC-4.1.3: SKIP_TIER edges connect chapters in non-adjacent tiers.
    AC-4.1.4: All edges are bidirectional.

    Per TIER_RELATIONSHIP_DIAGRAM.md:
    - Tier 1 ↔ Tier 3 (skips Tier 2)
    - Only created when tier difference >= 2

    Args:
        books: List of book dicts with 'id', 'title', 'tier' keys

    Returns:
        List of RelationshipEdge objects created (bidirectional = 2 edges per pair)
    """
    edges: list[RelationshipEdge] = []

    # Group books by tier
    books_by_tier: dict[int, list[dict[str, Any]]] = {}
    for book in books:
        tier = book.get("tier", 0)
        if tier not in books_by_tier:
            books_by_tier[tier] = []
        books_by_tier[tier].append(book)

    # Get sorted list of tiers
    sorted_tiers = sorted(books_by_tier.keys())

    # Create bidirectional edges between non-adjacent tiers (tier_diff >= 2)
    for i, tier in enumerate(sorted_tiers):
        # Look at ALL tiers after this one in the sorted list
        for other_tier in sorted_tiers[i + 1:]:
            tier_diff = abs(other_tier - tier)
            # Only create SKIP_TIER for tier differences >= 2
            if tier_diff < 2:
                continue

            # Create bidirectional edges between all books in non-adjacent tiers
            for source_book in books_by_tier[tier]:
                for target_book in books_by_tier[other_tier]:
                    source_id = source_book.get("id", "")
                    target_id = target_book.get("id", "")

                    if not source_id or not target_id:
                        continue

                    # Forward edge: tier N -> tier N+2+
                    edges.append(
                        RelationshipEdge(
                            from_id=source_id,
                            to_id=target_id,
                            edge_type=RelationshipEdgeType.SKIP_TIER,
                            from_tier=tier,
                            to_tier=other_tier,
                        )
                    )

                    # Reverse edge: tier N+2+ -> tier N (bidirectional)
                    edges.append(
                        RelationshipEdge(
                            from_id=target_id,
                            to_id=source_id,
                            edge_type=RelationshipEdgeType.SKIP_TIER,
                            from_tier=other_tier,
                            to_tier=tier,
                        )
                    )

    return edges


# =============================================================================
# Chapter Relationship Finder (AC-4.2.1 to AC-4.2.3)
# =============================================================================


class ChapterRelationshipFinder:
    """Find related chapters using BFS, DFS, and bidirectional traversal.

    Implements AC-4.2.1 (BFS), AC-4.2.2 (DFS), AC-4.2.3 (bidirectional).

    Usage:
        finder = ChapterRelationshipFinder(client=neo4j_client)
        results = await finder.find_related_bfs(chapter_id="ch-1a-3", max_depth=2)
    """

    def __init__(self, client: Neo4jClientLike | None = None) -> None:
        """Initialize with optional Neo4j client.

        Args:
            client: Neo4j client (real or fake). If None, uses mock data.
        """
        self._client = client
        # Mock graph data for testing when client is None
        self._mock_graph: dict[str, list[tuple[str, int, str]]] = {}

    def set_mock_graph(
        self, graph: dict[str, list[tuple[str, int, str]]]
    ) -> None:
        """Set mock graph data for testing.

        Args:
            graph: Dict mapping chapter_id to list of (related_id, tier, rel_type)
        """
        self._mock_graph = graph

    async def find_related_bfs(
        self,
        chapter_id: str,
        max_depth: int = 2,
    ) -> list[RelatedChapter]:
        """Find related chapters using BFS traversal.

        AC-4.2.1: BFS traversal for finding related chapters.

        Args:
            chapter_id: Starting chapter ID
            max_depth: Maximum depth to traverse (default 2)

        Returns:
            List of RelatedChapter objects found
        """
        visited: set[str] = {chapter_id}
        results: list[RelatedChapter] = []
        queue: deque[tuple[str, int, list[str]]] = deque(
            [(chapter_id, 0, [chapter_id])]
        )

        while queue:
            current_id, depth, path = queue.popleft()

            if depth >= max_depth:
                continue

            # Get neighbors
            neighbors = await self._get_neighbors(current_id)

            for neighbor_id, tier, rel_type in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [neighbor_id]
                    results.append(
                        RelatedChapter(
                            chapter_id=neighbor_id,
                            tier=tier,
                            relationship_type=rel_type,
                            path=new_path,
                        )
                    )
                    queue.append((neighbor_id, depth + 1, new_path))

        return results

    async def find_related_dfs(
        self,
        chapter_id: str,
        max_depth: int = 3,
    ) -> list[RelatedChapter]:
        """Find related chapters using DFS traversal.

        AC-4.2.2: DFS traversal for deep concept chains.

        Args:
            chapter_id: Starting chapter ID
            max_depth: Maximum depth to traverse (default 3)

        Returns:
            List of RelatedChapter objects found
        """
        visited: set[str] = {chapter_id}
        results: list[RelatedChapter] = []

        async def dfs_recursive(
            current_id: str, depth: int, path: list[str]
        ) -> None:
            if depth >= max_depth:
                return

            neighbors = await self._get_neighbors(current_id)

            for neighbor_id, tier, rel_type in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [neighbor_id]
                    results.append(
                        RelatedChapter(
                            chapter_id=neighbor_id,
                            tier=tier,
                            relationship_type=rel_type,
                            path=new_path,
                        )
                    )
                    await dfs_recursive(neighbor_id, depth + 1, new_path)

        await dfs_recursive(chapter_id, 0, [chapter_id])
        return results

    async def find_bidirectional(
        self,
        chapter_id: str,
        direction: str = "both",
    ) -> list[RelatedChapter]:
        """Find related chapters with directional traversal.

        AC-4.2.3: Bidirectional traversal.

        Args:
            chapter_id: Starting chapter ID
            direction: "up" (toward Tier 1), "down" (toward higher tiers),
                      or "both" (any direction)

        Returns:
            List of RelatedChapter objects found
        """
        neighbors = await self._get_neighbors(chapter_id)

        # Get source tier for filtering
        source_tier = await self._get_tier(chapter_id)

        results: list[RelatedChapter] = []
        for neighbor_id, tier, rel_type in neighbors:
            # Filter by direction
            if direction == "up" and tier >= source_tier:
                continue
            if direction == "down" and tier <= source_tier:
                continue

            results.append(
                RelatedChapter(
                    chapter_id=neighbor_id,
                    tier=tier,
                    relationship_type=rel_type,
                    path=[chapter_id, neighbor_id],
                )
            )

        return results

    async def _get_neighbors(
        self, chapter_id: str
    ) -> list[tuple[str, int, str]]:
        """Get neighboring chapters.

        Returns:
            List of (chapter_id, tier, relationship_type) tuples
        """
        if self._client is None:
            # Return mock data for testing
            return self._mock_graph.get(chapter_id, [])

        # Query Neo4j for neighbors
        cypher = """
        MATCH (c {id: $chapter_id})-[r:PARALLEL|PERPENDICULAR|SKIP_TIER]-(related)
        RETURN related.id AS id, related.tier AS tier, type(r) AS rel_type
        """
        results = await self._client.query(cypher, {"chapter_id": chapter_id})
        return [
            (r["id"], r.get("tier", 0), r["rel_type"])
            for r in results
        ]

    async def _get_tier(self, chapter_id: str) -> int:
        """Get tier for a chapter."""
        if self._client is None:
            # Extract tier from mock data or ID
            for cid, neighbors in self._mock_graph.items():
                if cid == chapter_id:
                    # Try to get tier from first neighbor's reverse lookup
                    return 1  # Default for mock
            return 1

        cypher = "MATCH (c {id: $chapter_id}) RETURN c.tier AS tier"
        results = await self._client.query(cypher, {"chapter_id": chapter_id})
        return results[0].get("tier", 1) if results else 1


# =============================================================================
# Module-Level Async Query Functions (used by API endpoints)
# =============================================================================


# Global client reference (set via dependency injection in FastAPI)
_global_client: Neo4jClientLike | None = None


def set_global_client(client: Neo4jClientLike | None) -> None:
    """Set the global Neo4j client for module-level functions."""
    global _global_client
    _global_client = client


async def get_chapter_relationships(
    chapter_id: str,
) -> RelationshipResult:
    """Get all relationships for a specific chapter.

    AC-4.3.1: GET /v1/graph/relationships/{chapter_id}

    Args:
        chapter_id: ID of the chapter to query

    Returns:
        RelationshipResult with all relationships for the chapter
    """
    if _global_client is None:
        # Return empty result if no client configured
        return RelationshipResult(
            chapter_id=chapter_id,
            relationships=[],
            total_count=0,
        )

    # Query all relationship types
    cypher = """
    MATCH (c {id: $chapter_id})-[r:PARALLEL|PERPENDICULAR|SKIP_TIER]-(related)
    RETURN 
        related.id AS related_id,
        related.title AS related_title,
        related.tier AS related_tier,
        type(r) AS relationship_type
    """

    results = await _global_client.query(cypher, {"chapter_id": chapter_id})

    relationships: list[RelatedChapter] = []
    for record in results:
        relationships.append(
            RelatedChapter(
                chapter_id=record.get("related_id", ""),
                tier=record.get("related_tier", 0),
                relationship_type=record.get("relationship_type", ""),
            )
        )

    return RelationshipResult(
        chapter_id=chapter_id,
        relationships=relationships,
        total_count=len(relationships),
    )


async def get_batch_relationships(
    chapter_ids: list[str],
) -> list[RelationshipResult]:
    """Get relationships for multiple chapters in a batch.

    AC-4.3.2: POST /v1/graph/relationships/batch

    Args:
        chapter_ids: List of chapter IDs to query

    Returns:
        List of RelationshipResult objects
    """
    results: list[RelationshipResult] = []

    for chapter_id in chapter_ids:
        result = await get_chapter_relationships(chapter_id)
        results.append(result)

    return results
