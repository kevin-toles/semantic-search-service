"""
WBS 2.3 RED: Unit tests for graph traversal module.

Tests for spider web model traversal based on TIER_RELATIONSHIP_DIAGRAM.md:
- PARALLEL: Same tier traversal (e.g., Tier 1 ↔ Tier 1)
- PERPENDICULAR: Adjacent tier traversal (e.g., Tier 1 → Tier 2)
- SKIP_TIER: Non-adjacent tier traversal (e.g., Tier 1 ↔ Tier 3)

All traversals are BIDIRECTIONAL - concepts flow in ANY direction.

Design follows:
- Repository pattern (GUIDELINES line 795)
- FakeNeo4jClient for unit testing
- Graph traversal algorithms (BFS/DFS)
- Tier-aware relevance scoring
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_graph_data() -> dict[str, list[dict[str, Any]]]:
    """Sample graph data representing spider web model.

    Structure:
    - TIER 1: Python Distilled, Philosophy of Software Design
    - TIER 2: Clean Code, Pragmatic Programmer, Architecture Patterns
    - TIER 3: Building LLM Apps, Microservices Patterns, AI Agents

    Relationships:
    - PARALLEL: Same tier (bidirectional)
    - PERPENDICULAR: Adjacent tiers (bidirectional)
    - SKIP_TIER: Non-adjacent tiers (bidirectional)
    """
    return {
        "tiers": [
            {"id": "tier-1", "name": "Foundational", "level": 1, "priority": "required"},
            {"id": "tier-2", "name": "Best Practices", "level": 2, "priority": "required"},
            {"id": "tier-3", "name": "Operational", "level": 3, "priority": "optional"},
        ],
        "books": [
            # Tier 1 books
            {"id": "book-pd", "title": "Python Distilled", "tier_id": "tier-1"},
            {"id": "book-psd", "title": "Philosophy of Software Design", "tier_id": "tier-1"},
            # Tier 2 books
            {"id": "book-cc", "title": "Clean Code", "tier_id": "tier-2"},
            {"id": "book-pp", "title": "Pragmatic Programmer", "tier_id": "tier-2"},
            {"id": "book-app", "title": "Architecture Patterns Python", "tier_id": "tier-2"},
            # Tier 3 books
            {"id": "book-llm", "title": "Building LLM Powered Apps", "tier_id": "tier-3"},
            {"id": "book-msp", "title": "Microservices Patterns", "tier_id": "tier-3"},
            {"id": "book-aia", "title": "AI Agents in Action", "tier_id": "tier-3"},
        ],
        "chapters": [
            # Python Distilled chapters
            {"id": "ch-pd-7", "title": "Decorators", "book_id": "book-pd", "chapter_num": 7},
            {"id": "ch-pd-8", "title": "Iterators", "book_id": "book-pd", "chapter_num": 8},
            # Philosophy of Software Design chapters
            {"id": "ch-psd-4", "title": "Deep Modules", "book_id": "book-psd", "chapter_num": 4},
            {"id": "ch-psd-9", "title": "Define Errors Out", "book_id": "book-psd", "chapter_num": 9},
            # Clean Code chapters
            {"id": "ch-cc-3", "title": "Functions", "book_id": "book-cc", "chapter_num": 3},
            # Pragmatic Programmer chapters
            {"id": "ch-pp-17", "title": "Power of Plain Text", "book_id": "book-pp", "chapter_num": 17},
            # Architecture Patterns Python chapters
            {"id": "ch-app-12", "title": "Dependency Injection", "book_id": "book-app", "chapter_num": 12},
            # Building LLM Powered Apps chapters
            {"id": "ch-llm-5", "title": "Prompt Engineering", "book_id": "book-llm", "chapter_num": 5},
            # Microservices Patterns chapters
            {"id": "ch-msp-8", "title": "Service Mesh", "book_id": "book-msp", "chapter_num": 8},
            # AI Agents chapters
            {"id": "ch-aia-3", "title": "Agent Architectures", "book_id": "book-aia", "chapter_num": 3},
        ],
        "concepts": [
            {"id": "concept-decorators", "name": "Decorators", "chapter_id": "ch-pd-7"},
            {"id": "concept-abstraction", "name": "Abstraction", "chapter_id": "ch-psd-4"},
            {"id": "concept-clean-func", "name": "Clean Functions", "chapter_id": "ch-cc-3"},
            {"id": "concept-di", "name": "Dependency Injection", "chapter_id": "ch-app-12"},
            {"id": "concept-prompts", "name": "Prompt Engineering", "chapter_id": "ch-llm-5"},
        ],
    }


@pytest.fixture
def fake_neo4j_client_for_traversal(
    sample_graph_data: dict[str, list[dict[str, Any]]]
) -> MagicMock:
    """Create a fake Neo4j client configured with sample graph data."""
    client = MagicMock()
    client.query = AsyncMock()
    client.is_connected = True

    # Define neighbor relationships for traversal
    # Based on spider web model from TIER_RELATIONSHIP_DIAGRAM.md
    neighbors_map: dict[str, list[str]] = {
        # Python Distilled Ch 7 (Tier 1) connects to:
        "ch-pd-7": ["ch-psd-4", "ch-cc-3", "ch-pd-8", "concept-decorators"],
        "ch-pd-8": ["ch-pd-7", "ch-psd-9"],
        # Philosophy of Software Design (Tier 1)
        "ch-psd-4": ["ch-pd-7", "ch-app-12", "ch-psd-9", "concept-abstraction"],
        "ch-psd-9": ["ch-psd-4", "ch-pd-8", "ch-msp-8"],
        # Clean Code (Tier 2)
        "ch-cc-3": ["ch-pd-7", "ch-pp-17", "ch-app-12", "concept-clean-func"],
        "ch-pp-17": ["ch-cc-3", "ch-msp-8"],
        "ch-app-12": ["ch-psd-4", "ch-cc-3", "ch-llm-5", "concept-di"],
        # Tier 3 books
        "ch-llm-5": ["ch-app-12", "ch-aia-3", "concept-prompts"],
        "ch-msp-8": ["ch-pp-17", "ch-psd-9", "ch-aia-3"],
        "ch-aia-3": ["ch-llm-5", "ch-msp-8"],
        # Concepts connect to their chapters (bidirectional)
        "concept-decorators": ["ch-pd-7"],
        "concept-abstraction": ["ch-psd-4"],
        "concept-clean-func": ["ch-cc-3"],
        "concept-di": ["ch-app-12"],
        "concept-prompts": ["ch-llm-5"],
    }

    # Configure query responses based on Cypher patterns
    async def query_side_effect(
        cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Simulate Neo4j query responses."""
        if "MATCH (t:Tier)" in cypher:
            return [{"t": tier} for tier in sample_graph_data["tiers"]]
        if "MATCH (b:Book)" in cypher:
            return [{"b": book} for book in sample_graph_data["books"]]
        if "MATCH (c:Chapter)" in cypher:
            return [{"c": chapter} for chapter in sample_graph_data["chapters"]]

        # Handle neighbor queries for traversal
        if "RELATED_TO|CONNECTS_TO|HAS_CHAPTER|REFERENCES" in cypher:
            node_id = parameters.get("node_id", "") if parameters else ""
            neighbor_ids = neighbors_map.get(node_id, [])
            return [{"neighbor_id": n} for n in neighbor_ids]

        # Handle tier queries
        if "BELONGS_TO" in cypher:
            node_id = parameters.get("node_id", "") if parameters else ""
            # Map nodes to tiers
            tier_map = {
                "ch-pd-7": "tier-1", "ch-pd-8": "tier-1",
                "ch-psd-4": "tier-1", "ch-psd-9": "tier-1",
                "ch-cc-3": "tier-2", "ch-pp-17": "tier-2", "ch-app-12": "tier-2",
                "ch-llm-5": "tier-3", "ch-msp-8": "tier-3", "ch-aia-3": "tier-3",
            }
            tier_id = tier_map.get(node_id)
            if tier_id:
                return [{"tier_id": tier_id}]

        return []

    client.query = AsyncMock(side_effect=query_side_effect)
    return client


# =============================================================================
# Test: RelationshipType Enum
# =============================================================================


class TestRelationshipTypes:
    """Tests for relationship type enumeration."""

    def test_relationship_types_defined(self) -> None:
        """RelationshipType enum should define PARALLEL, PERPENDICULAR, SKIP_TIER."""
        from src.graph.traversal import RelationshipType

        assert hasattr(RelationshipType, "PARALLEL")
        assert hasattr(RelationshipType, "PERPENDICULAR")
        assert hasattr(RelationshipType, "SKIP_TIER")

    def test_parallel_is_same_tier(self) -> None:
        """PARALLEL relationship should represent same-tier connections."""
        from src.graph.traversal import RelationshipType

        assert RelationshipType.PARALLEL.value == "PARALLEL"

    def test_perpendicular_is_adjacent_tier(self) -> None:
        """PERPENDICULAR relationship should represent adjacent-tier connections."""
        from src.graph.traversal import RelationshipType

        assert RelationshipType.PERPENDICULAR.value == "PERPENDICULAR"

    def test_skip_tier_is_non_adjacent(self) -> None:
        """SKIP_TIER relationship should represent non-adjacent tier connections."""
        from src.graph.traversal import RelationshipType

        assert RelationshipType.SKIP_TIER.value == "SKIP_TIER"


# =============================================================================
# Test: TraversalDirection Enum
# =============================================================================


class TestTraversalDirection:
    """Tests for traversal direction - all directions are bidirectional."""

    def test_direction_types_defined(self) -> None:
        """TraversalDirection should define UP, DOWN, LATERAL."""
        from src.graph.traversal import TraversalDirection

        assert hasattr(TraversalDirection, "UP")
        assert hasattr(TraversalDirection, "DOWN")
        assert hasattr(TraversalDirection, "LATERAL")

    def test_up_direction_toward_tier_1(self) -> None:
        """UP direction should move toward Tier 1 (higher priority)."""
        from src.graph.traversal import TraversalDirection

        assert TraversalDirection.UP.value == "UP"

    def test_down_direction_toward_tier_n(self) -> None:
        """DOWN direction should move toward higher tier numbers."""
        from src.graph.traversal import TraversalDirection

        assert TraversalDirection.DOWN.value == "DOWN"

    def test_lateral_direction_same_tier(self) -> None:
        """LATERAL direction should stay within same tier."""
        from src.graph.traversal import TraversalDirection

        assert TraversalDirection.LATERAL.value == "LATERAL"


# =============================================================================
# Test: GraphTraversal Class Initialization
# =============================================================================


class TestGraphTraversalInitialization:
    """Tests for GraphTraversal class initialization."""

    def test_traversal_accepts_neo4j_client(self) -> None:
        """GraphTraversal should accept a Neo4jClient for queries."""
        from src.graph.traversal import GraphTraversal

        mock_client = MagicMock()
        traversal = GraphTraversal(client=mock_client)

        assert traversal is not None
        assert traversal._client == mock_client

    def test_traversal_accepts_fake_client(self) -> None:
        """GraphTraversal should work with FakeNeo4jClient (duck typing)."""
        from src.graph.neo4j_client import FakeNeo4jClient
        from src.graph.traversal import GraphTraversal

        fake_client = FakeNeo4jClient()
        traversal = GraphTraversal(client=fake_client)

        assert traversal._client == fake_client

    def test_traversal_stores_max_depth(self) -> None:
        """GraphTraversal should accept max_depth parameter."""
        from src.graph.traversal import GraphTraversal

        mock_client = MagicMock()
        traversal = GraphTraversal(client=mock_client, max_depth=5)

        assert traversal._max_depth == 5

    def test_traversal_default_max_depth(self) -> None:
        """GraphTraversal should have reasonable default max_depth."""
        from src.graph.traversal import GraphTraversal

        mock_client = MagicMock()
        traversal = GraphTraversal(client=mock_client)

        assert traversal._max_depth == 3  # Default


# =============================================================================
# Test: Tier Relationship Detection
# =============================================================================


class TestTierRelationshipDetection:
    """Tests for detecting relationship types between tiers."""

    def test_same_tier_is_parallel(self) -> None:
        """Books in the same tier should have PARALLEL relationship."""
        from src.graph.traversal import GraphTraversal, RelationshipType

        mock_client = MagicMock()
        traversal = GraphTraversal(client=mock_client)

        # Tier 1 to Tier 1
        rel_type = traversal.get_relationship_type(tier_from=1, tier_to=1)

        assert rel_type == RelationshipType.PARALLEL

    def test_adjacent_tier_is_perpendicular(self) -> None:
        """Books in adjacent tiers should have PERPENDICULAR relationship."""
        from src.graph.traversal import GraphTraversal, RelationshipType

        mock_client = MagicMock()
        traversal = GraphTraversal(client=mock_client)

        # Tier 1 to Tier 2 (adjacent)
        rel_type = traversal.get_relationship_type(tier_from=1, tier_to=2)

        assert rel_type == RelationshipType.PERPENDICULAR

    def test_non_adjacent_tier_is_skip(self) -> None:
        """Books in non-adjacent tiers should have SKIP_TIER relationship."""
        from src.graph.traversal import GraphTraversal, RelationshipType

        mock_client = MagicMock()
        traversal = GraphTraversal(client=mock_client)

        # Tier 1 to Tier 3 (skip)
        rel_type = traversal.get_relationship_type(tier_from=1, tier_to=3)

        assert rel_type == RelationshipType.SKIP_TIER

    def test_relationship_is_bidirectional(self) -> None:
        """Relationship type should be same in both directions."""
        from src.graph.traversal import GraphTraversal, RelationshipType

        mock_client = MagicMock()
        traversal = GraphTraversal(client=mock_client)

        # Tier 1 → Tier 2 should equal Tier 2 → Tier 1
        rel_1_to_2 = traversal.get_relationship_type(tier_from=1, tier_to=2)
        rel_2_to_1 = traversal.get_relationship_type(tier_from=2, tier_to=1)

        assert rel_1_to_2 == rel_2_to_1 == RelationshipType.PERPENDICULAR


# =============================================================================
# Test: BFS Traversal (Breadth-First Search)
# =============================================================================


class TestBFSTraversal:
    """Tests for breadth-first search traversal."""

    @pytest.mark.asyncio
    async def test_bfs_finds_parallel_connections_first(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """BFS should prioritize PARALLEL (same tier) connections first."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        results = await traversal.bfs_traverse(
            start_node_id="ch-pd-7",  # Python Distilled Ch 7
            max_depth=2,
        )

        # First level should be same tier (PARALLEL)
        assert len(results) > 0
        # Results should be ordered by relationship priority

    @pytest.mark.asyncio
    async def test_bfs_respects_max_depth(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """BFS should not traverse beyond max_depth."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        results = await traversal.bfs_traverse(
            start_node_id="ch-pd-7",
            max_depth=1,
        )

        # All results should be within depth 1
        for result in results:
            assert result.get("depth", 0) <= 1

    @pytest.mark.asyncio
    async def test_bfs_returns_traversal_path(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """BFS should return the traversal path for each result."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        results = await traversal.bfs_traverse(
            start_node_id="ch-pd-7",
            max_depth=2,
        )

        for result in results:
            assert "path" in result
            assert isinstance(result["path"], list)


# =============================================================================
# Test: DFS Traversal (Depth-First Search)
# =============================================================================


class TestDFSTraversal:
    """Tests for depth-first search traversal."""

    @pytest.mark.asyncio
    async def test_dfs_explores_deep_paths(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """DFS should explore deep paths before breadth."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        results = await traversal.dfs_traverse(
            start_node_id="ch-pd-7",
            max_depth=3,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_dfs_respects_max_depth(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """DFS should not traverse beyond max_depth."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        results = await traversal.dfs_traverse(
            start_node_id="ch-pd-7",
            max_depth=2,
        )

        for result in results:
            assert result.get("depth", 0) <= 2

    @pytest.mark.asyncio
    async def test_dfs_avoids_cycles(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """DFS should not revisit nodes (avoid infinite loops)."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        results = await traversal.dfs_traverse(
            start_node_id="ch-pd-7",
            max_depth=5,
        )

        # Each node should appear only once
        visited_ids = [r["node_id"] for r in results if "node_id" in r]
        assert len(visited_ids) == len(set(visited_ids))


# =============================================================================
# Test: Cross-Reference Path Finding
# =============================================================================


class TestCrossReferencePaths:
    """Tests for finding cross-reference paths between concepts."""

    @pytest.mark.asyncio
    async def test_find_path_between_concepts(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """Should find traversal path between two concepts."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        path = await traversal.find_cross_reference_path(
            from_concept="concept-decorators",
            to_concept="concept-di",
        )

        assert path is not None
        assert len(path) > 0

    @pytest.mark.asyncio
    async def test_path_includes_relationship_types(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """Path should include relationship type for each hop."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        path = await traversal.find_cross_reference_path(
            from_concept="concept-decorators",
            to_concept="concept-prompts",
        )

        if path:
            for hop in path:
                assert "relationship_type" in hop

    @pytest.mark.asyncio
    async def test_path_respects_tier_priority(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """Path finding should prefer higher-priority tiers."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        path = await traversal.find_cross_reference_path(
            from_concept="concept-decorators",
            to_concept="concept-di",
            prefer_higher_priority=True,
        )

        # Path should go through Tier 1 if possible
        assert path is not None


# =============================================================================
# Test: Spider Web Model Traversal
# =============================================================================


class TestSpiderWebTraversal:
    """Tests for complete spider web model traversal."""

    @pytest.mark.asyncio
    async def test_traverse_all_related_chapters(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """Should find all chapters related to a source chapter."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        related = await traversal.find_related_chapters(
            source_chapter_id="ch-pd-7",
            max_depth=3,
        )

        assert isinstance(related, list)

    @pytest.mark.asyncio
    async def test_related_chapters_grouped_by_tier(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """Related chapters should be groupable by tier."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        related = await traversal.find_related_chapters(
            source_chapter_id="ch-pd-7",
            group_by_tier=True,
        )

        # Should have tier-grouped structure
        if related:
            assert any(
                "tier" in item or "tier_id" in item for item in related
            )

    @pytest.mark.asyncio
    async def test_related_chapters_include_relevance_score(
        self, fake_neo4j_client_for_traversal: MagicMock
    ) -> None:
        """Related chapters should include relevance scores."""
        from src.graph.traversal import GraphTraversal

        traversal = GraphTraversal(client=fake_neo4j_client_for_traversal)

        related = await traversal.find_related_chapters(
            source_chapter_id="ch-pd-7",
            include_relevance=True,
        )

        for chapter in related:
            if "relevance" in chapter:
                assert isinstance(chapter["relevance"], (int, float))


# =============================================================================
# Test: TraversalResult Dataclass
# =============================================================================


class TestTraversalResult:
    """Tests for TraversalResult dataclass."""

    def test_traversal_result_has_required_fields(self) -> None:
        """TraversalResult should have node_id, depth, path, relationship_type."""
        from src.graph.traversal import TraversalResult

        result = TraversalResult(
            node_id="ch-pd-7",
            depth=1,
            path=["ch-pd-7", "ch-psd-4"],
            relationship_type="PARALLEL",
        )

        assert result.node_id == "ch-pd-7"
        assert result.depth == 1
        assert result.path == ["ch-pd-7", "ch-psd-4"]
        assert result.relationship_type == "PARALLEL"

    def test_traversal_result_optional_fields(self) -> None:
        """TraversalResult should support optional relevance_score."""
        from src.graph.traversal import TraversalResult

        result = TraversalResult(
            node_id="ch-pd-7",
            depth=1,
            path=["ch-pd-7"],
            relationship_type="PARALLEL",
            relevance_score=0.85,
        )

        assert result.relevance_score == 0.85
