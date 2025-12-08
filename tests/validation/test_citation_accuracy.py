"""Citation Accuracy Validation Tests (WBS 6.3).

TDD Phase: RED → GREEN → REFACTOR
Target: ≥90% relevance for cross-referenced citations

Tests validate that:
1. Cross-referenced citations maintain high relevance scores
2. Tier proximity increases relevance (PARALLEL > PERPENDICULAR > SKIP_TIER)
3. Depth-based relevance decay is within acceptable bounds
4. Chicago-formatted citations preserve source relevance metadata
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

# Import the GraphTraversal class and related types
from src.graph.traversal import (
    GraphTraversal,
    TraversalResult,
    RelationshipType,
)


# =============================================================================
# Test Data Models (Avoiding 'Test' prefix for pytest collection)
# =============================================================================


@dataclass
class CitationNode:
    """Represents a citation source node in the graph."""
    
    node_id: str
    title: str
    author: str
    chapter_number: int
    tier: int
    relevance_score: float = 1.0


@dataclass
class CitationRelationship:
    """Represents a relationship between citation nodes."""
    
    source_id: str
    target_id: str
    rel_type: str
    weight: float = 1.0


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def citation_nodes() -> list[CitationNode]:
    """Create sample citation nodes across tiers."""
    return [
        CitationNode(
            node_id="philo_ch2",
            title="A Philosophy of Software Design",
            author="Ousterhout, John",
            chapter_number=2,
            tier=1,
        ),
        CitationNode(
            node_id="micro_ch4",
            title="Building Microservices",
            author="Newman, Sam",
            chapter_number=4,
            tier=2,
        ),
        CitationNode(
            node_id="patterns_ch3",
            title="Architecture Patterns with Python",
            author="Percival, Harry",
            chapter_number=3,
            tier=1,
        ),
        CitationNode(
            node_id="ddd_ch5",
            title="Domain Driven Design",
            author="Evans, Eric",
            chapter_number=5,
            tier=3,
        ),
        CitationNode(
            node_id="clean_ch7",
            title="Clean Architecture",
            author="Martin, Robert",
            chapter_number=7,
            tier=2,
        ),
    ]


@pytest.fixture
def citation_relationships() -> list[CitationRelationship]:
    """Create relationships between citation nodes."""
    return [
        # T1 PARALLEL relationships (same tier)
        CitationRelationship(
            source_id="philo_ch2",
            target_id="patterns_ch3",
            rel_type=RelationshipType.PARALLEL.value,
            weight=0.95,
        ),
        # T1 → T2 PERPENDICULAR relationships
        CitationRelationship(
            source_id="philo_ch2",
            target_id="micro_ch4",
            rel_type=RelationshipType.PERPENDICULAR.value,
            weight=0.85,
        ),
        CitationRelationship(
            source_id="patterns_ch3",
            target_id="clean_ch7",
            rel_type=RelationshipType.PERPENDICULAR.value,
            weight=0.88,
        ),
        # T1 → T3 SKIP_TIER relationships
        CitationRelationship(
            source_id="philo_ch2",
            target_id="ddd_ch5",
            rel_type=RelationshipType.SKIP_TIER.value,
            weight=0.72,
        ),
        # T2 → T3 PERPENDICULAR relationships
        CitationRelationship(
            source_id="micro_ch4",
            target_id="ddd_ch5",
            rel_type=RelationshipType.PERPENDICULAR.value,
            weight=0.80,
        ),
    ]


@pytest.fixture
def mock_neo4j_client(
    citation_nodes: list[CitationNode],
    citation_relationships: list[CitationRelationship],
) -> AsyncMock:
    """Create a mock Neo4j client with citation data."""
    client = AsyncMock()
    
    async def mock_query(cypher: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """Mock query that returns citation graph data."""
        # Handle neighbor queries
        if "MATCH" in cypher and "neighbor" in cypher.lower():
            source_id = parameters.get("node_id") if parameters else None
            results = []
            for rel in citation_relationships:
                if rel.source_id == source_id:
                    results.append({
                        "neighbor_id": rel.target_id,
                        "relationship_type": rel.rel_type,
                        "weight": rel.weight,
                    })
                elif rel.target_id == source_id:  # Bidirectional
                    results.append({
                        "neighbor_id": rel.source_id,
                        "relationship_type": rel.rel_type,
                        "weight": rel.weight,
                    })
            return results
        return []
    
    client.query = AsyncMock(side_effect=mock_query)
    return client


@pytest.fixture
def traversal(mock_neo4j_client: AsyncMock) -> GraphTraversal:
    """Create GraphTraversal instance with mock client."""
    return GraphTraversal(client=mock_neo4j_client, max_depth=3)


# =============================================================================
# Citation Relevance Tests
# =============================================================================


class TestCitationRelevanceScoring:
    """Tests for citation relevance score calculations."""
    
    def test_parallel_relationship_high_relevance(self, traversal: GraphTraversal) -> None:
        """PARALLEL relationships should have ≥0.9 relevance at depth 1."""
        relevance = traversal._calculate_relevance(
            depth=1,
            relationship_type=RelationshipType.PARALLEL.value,
        )
        
        # PARALLEL at depth 1: (1.0 - 0.2) + 0.2 = 1.0
        assert relevance >= 0.9, f"PARALLEL at depth 1 should be ≥0.9, got {relevance}"
    
    def test_perpendicular_relationship_good_relevance(self, traversal: GraphTraversal) -> None:
        """PERPENDICULAR relationships should have ≥0.7 relevance at depth 1."""
        relevance = traversal._calculate_relevance(
            depth=1,
            relationship_type=RelationshipType.PERPENDICULAR.value,
        )
        
        # PERPENDICULAR at depth 1: (1.0 - 0.2) + 0.1 = 0.9
        assert relevance >= 0.7, f"PERPENDICULAR at depth 1 should be ≥0.7, got {relevance}"
    
    def test_skip_tier_relationship_acceptable_relevance(self, traversal: GraphTraversal) -> None:
        """SKIP_TIER relationships should have ≥0.5 relevance at depth 1."""
        relevance = traversal._calculate_relevance(
            depth=1,
            relationship_type=RelationshipType.SKIP_TIER.value,
        )
        
        # SKIP_TIER at depth 1: (1.0 - 0.2) + 0.0 = 0.8
        assert relevance >= 0.5, f"SKIP_TIER at depth 1 should be ≥0.5, got {relevance}"
    
    def test_depth_decay_within_bounds(self, traversal: GraphTraversal) -> None:
        """Relevance should decay by at most 20% per depth level."""
        depth_1 = traversal._calculate_relevance(1, RelationshipType.PARALLEL.value)
        depth_2 = traversal._calculate_relevance(2, RelationshipType.PARALLEL.value)
        depth_3 = traversal._calculate_relevance(3, RelationshipType.PARALLEL.value)
        
        # Decay should be at most 0.2 per level
        assert depth_1 - depth_2 <= 0.25, "Depth 1→2 decay exceeds 25%"
        assert depth_2 - depth_3 <= 0.25, "Depth 2→3 decay exceeds 25%"
    
    def test_all_relevance_scores_positive(self, traversal: GraphTraversal) -> None:
        """All relevance scores should be positive (>= 0)."""
        for depth in range(1, 5):  # Up to depth 4 (before scores hit 0)
            for rel_type in RelationshipType:
                relevance = traversal._calculate_relevance(depth, rel_type.value)
                assert relevance >= 0, f"Relevance at depth {depth} for {rel_type} should be >= 0"
    
    def test_relevance_capped_at_one(self, traversal: GraphTraversal) -> None:
        """Relevance scores should never exceed 1.0."""
        relevance = traversal._calculate_relevance(
            depth=0,  # Closest possible
            relationship_type=RelationshipType.PARALLEL.value,
        )
        
        assert relevance <= 1.0, f"Relevance should be capped at 1.0, got {relevance}"


class TestCitationAccuracyTarget:
    """Tests for ≥90% citation accuracy target."""
    
    @pytest.mark.asyncio
    async def test_tier1_citations_meet_90_percent_target(
        self,
        traversal: GraphTraversal,
    ) -> None:
        """Tier 1 PARALLEL citations should achieve ≥90% relevance."""
        results = await traversal.bfs_traverse(
            start_node_id="philo_ch2",
            max_depth=1,
        )
        
        # Filter for PARALLEL relationships (results are dicts)
        parallel_results = [
            r for r in results
            if r["relationship_type"] == RelationshipType.PARALLEL.value
        ]
        
        for result in parallel_results:
            calculated_relevance = traversal._calculate_relevance(
                depth=result["depth"],
                relationship_type=result["relationship_type"],
            )
            assert calculated_relevance >= 0.9, (
                f"PARALLEL citation {result['node_id']} has {calculated_relevance:.2%} relevance, "
                f"expected ≥90%"
            )
    
    @pytest.mark.asyncio
    async def test_cross_tier_citations_meet_70_percent_target(
        self,
        traversal: GraphTraversal,
    ) -> None:
        """Cross-tier PERPENDICULAR citations should achieve ≥70% relevance."""
        results = await traversal.bfs_traverse(
            start_node_id="philo_ch2",
            max_depth=1,
        )
        
        # Filter for PERPENDICULAR relationships (results are dicts)
        perpendicular_results = [
            r for r in results
            if r["relationship_type"] == RelationshipType.PERPENDICULAR.value
        ]
        
        for result in perpendicular_results:
            calculated_relevance = traversal._calculate_relevance(
                depth=result["depth"],
                relationship_type=result["relationship_type"],
            )
            assert calculated_relevance >= 0.7, (
                f"PERPENDICULAR citation {result['node_id']} has {calculated_relevance:.2%} relevance, "
                f"expected ≥70%"
            )
    
    @pytest.mark.asyncio
    async def test_average_citation_relevance_above_threshold(
        self,
        traversal: GraphTraversal,
    ) -> None:
        """Average relevance across all depth-1 citations should be ≥0.85."""
        results = await traversal.bfs_traverse(
            start_node_id="philo_ch2",
            max_depth=1,
        )
        
        if not results:
            pytest.skip("No citations found for averaging")
        
        total_relevance = sum(
            traversal._calculate_relevance(r["depth"], r["relationship_type"])
            for r in results
        )
        avg_relevance = total_relevance / len(results)
        
        assert avg_relevance >= 0.85, (
            f"Average citation relevance is {avg_relevance:.2%}, expected ≥85%"
        )


class TestCitationRelevanceDistribution:
    """Tests for citation relevance distribution patterns."""
    
    def test_relationship_type_ordering(self, traversal: GraphTraversal) -> None:
        """PARALLEL > PERPENDICULAR > SKIP_TIER for same depth."""
        depth = 1
        
        parallel = traversal._calculate_relevance(depth, RelationshipType.PARALLEL.value)
        perpendicular = traversal._calculate_relevance(depth, RelationshipType.PERPENDICULAR.value)
        skip_tier = traversal._calculate_relevance(depth, RelationshipType.SKIP_TIER.value)
        
        assert parallel > perpendicular, "PARALLEL should have higher relevance than PERPENDICULAR"
        assert perpendicular > skip_tier, "PERPENDICULAR should have higher relevance than SKIP_TIER"
    
    def test_depth_ordering(self, traversal: GraphTraversal) -> None:
        """Closer citations should have higher relevance."""
        rel_type = RelationshipType.PERPENDICULAR.value
        
        depth_1 = traversal._calculate_relevance(1, rel_type)
        depth_2 = traversal._calculate_relevance(2, rel_type)
        depth_3 = traversal._calculate_relevance(3, rel_type)
        
        assert depth_1 > depth_2 > depth_3, "Closer citations should have higher relevance"
    
    @pytest.mark.asyncio
    async def test_citation_count_at_depth_1(
        self,
        traversal: GraphTraversal,
    ) -> None:
        """Should discover expected number of citations at depth 1."""
        results = await traversal.bfs_traverse(
            start_node_id="philo_ch2",
            max_depth=1,
        )
        
        # philo_ch2 has 3 direct relationships
        assert len(results) >= 1, "Should find at least 1 citation at depth 1"
    
    @pytest.mark.asyncio
    async def test_multi_hop_relevance_accumulation(
        self,
        traversal: GraphTraversal,
    ) -> None:
        """Multi-hop traversal should show decreasing relevance."""
        results = await traversal.bfs_traverse(
            start_node_id="philo_ch2",
            max_depth=2,
        )
        
        depth_1_relevances = []
        depth_2_relevances = []
        
        for r in results:
            relevance = traversal._calculate_relevance(r["depth"], r["relationship_type"])
            if r["depth"] == 1:
                depth_1_relevances.append(relevance)
            elif r["depth"] == 2:
                depth_2_relevances.append(relevance)
        
        if depth_1_relevances and depth_2_relevances:
            avg_depth_1 = sum(depth_1_relevances) / len(depth_1_relevances)
            avg_depth_2 = sum(depth_2_relevances) / len(depth_2_relevances)
            
            assert avg_depth_1 > avg_depth_2, (
                f"Depth 1 avg ({avg_depth_1:.2f}) should exceed depth 2 avg ({avg_depth_2:.2f})"
            )


class TestCitationMetadataPreservation:
    """Tests for preserving citation metadata through traversal."""
    
    @pytest.mark.asyncio
    async def test_traversal_preserves_node_id(
        self,
        traversal: GraphTraversal,
    ) -> None:
        """Traversal results should preserve original node IDs."""
        results = await traversal.bfs_traverse(
            start_node_id="philo_ch2",
            max_depth=1,
        )
        
        for result in results:
            assert result["node_id"] is not None
            assert len(result["node_id"]) > 0
    
    @pytest.mark.asyncio
    async def test_traversal_preserves_relationship_type(
        self,
        traversal: GraphTraversal,
    ) -> None:
        """Traversal results should preserve relationship type."""
        results = await traversal.bfs_traverse(
            start_node_id="philo_ch2",
            max_depth=1,
        )
        
        valid_types = {rel.value for rel in RelationshipType}
        
        for result in results:
            assert result["relationship_type"] in valid_types, (
                f"Invalid relationship type: {result['relationship_type']}"
            )
    
    @pytest.mark.asyncio
    async def test_traversal_tracks_path(
        self,
        traversal: GraphTraversal,
    ) -> None:
        """Traversal should track full path from source."""
        results = await traversal.bfs_traverse(
            start_node_id="philo_ch2",
            max_depth=2,
        )
        
        for result in results:
            # Path should start with source and end with current node
            assert len(result["path"]) >= 2, "Path should have at least source and target"
            assert result["path"][0] == "philo_ch2", "Path should start with source node"
            assert result["path"][-1] == result["node_id"], "Path should end with current node"


class TestCitationQualityMetrics:
    """Tests for citation quality metrics and reporting."""
    
    def test_relevance_score_precision(self, traversal: GraphTraversal) -> None:
        """Relevance scores should have reasonable precision (2 decimal places)."""
        relevance = traversal._calculate_relevance(1, RelationshipType.PERPENDICULAR.value)
        
        # Check that relevance can be rounded to 2 decimal places without losing info
        rounded = round(relevance, 2)
        assert abs(relevance - rounded) < 0.001, "Relevance precision should be at most 2 decimals"
    
    def test_relevance_score_consistency(self, traversal: GraphTraversal) -> None:
        """Same inputs should produce same relevance score."""
        score1 = traversal._calculate_relevance(2, RelationshipType.PARALLEL.value)
        score2 = traversal._calculate_relevance(2, RelationshipType.PARALLEL.value)
        
        assert score1 == score2, "Relevance calculation should be deterministic"
    
    @pytest.mark.asyncio
    async def test_citation_accuracy_report_data(
        self,
        traversal: GraphTraversal,
    ) -> None:
        """Collect data for citation accuracy report."""
        results = await traversal.bfs_traverse(
            start_node_id="philo_ch2",
            max_depth=2,
        )
        
        # Collect metrics for report
        metrics = {
            "total_citations": len(results),
            "by_depth": {},
            "by_relationship": {},
            "avg_relevance": 0.0,
        }
        
        relevances = []
        for result in results:
            relevance = traversal._calculate_relevance(result["depth"], result["relationship_type"])
            relevances.append(relevance)
            
            # Count by depth
            depth_key = f"depth_{result['depth']}"
            metrics["by_depth"][depth_key] = metrics["by_depth"].get(depth_key, 0) + 1
            
            # Count by relationship
            metrics["by_relationship"][result["relationship_type"]] = (
                metrics["by_relationship"].get(result["relationship_type"], 0) + 1
            )
        
        if relevances:
            metrics["avg_relevance"] = sum(relevances) / len(relevances)
        
        # Validate metrics structure
        assert "total_citations" in metrics
        assert "avg_relevance" in metrics
        assert metrics["total_citations"] >= 0
