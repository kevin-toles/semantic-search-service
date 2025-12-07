"""
WBS 2.5 RED: Unit tests for graph schema helpers.

Tests for schema management utilities including:
- Node label definitions (Tier, Book, Chapter, Concept)
- Relationship type definitions (BELONGS_TO, HAS_CHAPTER, REFERENCES)
- Index creation helpers
- Constraint helpers
- Schema migration utilities

Design follows:
- Clean separation of schema from data operations
- Cypher generation for schema operations
- Neo4j 5.x schema best practices
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_neo4j_client() -> MagicMock:
    """Create a mock Neo4j client for schema operations."""
    client = MagicMock()
    client.execute_write = AsyncMock(return_value=[])
    client.query = AsyncMock(return_value=[])
    client.is_connected = True
    return client


# =============================================================================
# Test: Node Labels
# =============================================================================


class TestNodeLabels:
    """Tests for node label definitions."""

    def test_tier_label_defined(self) -> None:
        """Tier node label should be defined."""
        from src.graph.schema import NodeLabels

        assert NodeLabels.TIER == "Tier"

    def test_book_label_defined(self) -> None:
        """Book node label should be defined."""
        from src.graph.schema import NodeLabels

        assert NodeLabels.BOOK == "Book"

    def test_chapter_label_defined(self) -> None:
        """Chapter node label should be defined."""
        from src.graph.schema import NodeLabels

        assert NodeLabels.CHAPTER == "Chapter"

    def test_concept_label_defined(self) -> None:
        """Concept node label should be defined."""
        from src.graph.schema import NodeLabels

        assert NodeLabels.CONCEPT == "Concept"

    def test_taxonomy_label_defined(self) -> None:
        """Taxonomy node label should be defined."""
        from src.graph.schema import NodeLabels

        assert NodeLabels.TAXONOMY == "Taxonomy"


# =============================================================================
# Test: Relationship Types
# =============================================================================


class TestRelationshipLabels:
    """Tests for relationship type definitions."""

    def test_belongs_to_relationship(self) -> None:
        """BELONGS_TO relationship should be defined."""
        from src.graph.schema import RelationshipLabels

        assert RelationshipLabels.BELONGS_TO == "BELONGS_TO"

    def test_has_chapter_relationship(self) -> None:
        """HAS_CHAPTER relationship should be defined."""
        from src.graph.schema import RelationshipLabels

        assert RelationshipLabels.HAS_CHAPTER == "HAS_CHAPTER"

    def test_has_concept_relationship(self) -> None:
        """HAS_CONCEPT relationship should be defined."""
        from src.graph.schema import RelationshipLabels

        assert RelationshipLabels.HAS_CONCEPT == "HAS_CONCEPT"

    def test_references_relationship(self) -> None:
        """REFERENCES relationship should be defined."""
        from src.graph.schema import RelationshipLabels

        assert RelationshipLabels.REFERENCES == "REFERENCES"

    def test_related_to_relationship(self) -> None:
        """RELATED_TO relationship should be defined."""
        from src.graph.schema import RelationshipLabels

        assert RelationshipLabels.RELATED_TO == "RELATED_TO"


# =============================================================================
# Test: Schema Manager Initialization
# =============================================================================


class TestSchemaManagerInit:
    """Tests for SchemaManager initialization."""

    def test_schema_manager_accepts_client(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """SchemaManager should accept a Neo4jClient."""
        from src.graph.schema import SchemaManager

        manager = SchemaManager(client=mock_neo4j_client)

        assert manager._client == mock_neo4j_client

    def test_schema_manager_works_with_fake_client(self) -> None:
        """SchemaManager should work with FakeNeo4jClient."""
        from src.graph.neo4j_client import FakeNeo4jClient
        from src.graph.schema import SchemaManager

        fake = FakeNeo4jClient()
        manager = SchemaManager(client=fake)

        assert manager._client == fake


# =============================================================================
# Test: Index Creation
# =============================================================================


class TestIndexCreation:
    """Tests for creating indexes on node properties."""

    @pytest.mark.asyncio
    async def test_create_unique_constraint_tier_id(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """Should create unique constraint on Tier.id."""
        from src.graph.schema import SchemaManager

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.create_tier_constraints()

        # Verify execute_write was called with constraint Cypher
        mock_neo4j_client.execute_write.assert_called()
        call_args = mock_neo4j_client.execute_write.call_args_list
        cypher_calls = [str(call) for call in call_args]
        assert any("Tier" in str(c) for c in cypher_calls)

    @pytest.mark.asyncio
    async def test_create_unique_constraint_book_id(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """Should create unique constraint on Book.id."""
        from src.graph.schema import SchemaManager

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.create_book_constraints()

        mock_neo4j_client.execute_write.assert_called()

    @pytest.mark.asyncio
    async def test_create_unique_constraint_chapter_id(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """Should create unique constraint on Chapter.id."""
        from src.graph.schema import SchemaManager

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.create_chapter_constraints()

        mock_neo4j_client.execute_write.assert_called()

    @pytest.mark.asyncio
    async def test_create_index_book_title(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """Should create index on Book.title for faster lookups."""
        from src.graph.schema import SchemaManager

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.create_book_indexes()

        mock_neo4j_client.execute_write.assert_called()

    @pytest.mark.asyncio
    async def test_create_index_chapter_number(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """Should create index on Chapter.chapter_num."""
        from src.graph.schema import SchemaManager

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.create_chapter_indexes()

        mock_neo4j_client.execute_write.assert_called()


# =============================================================================
# Test: Full Schema Creation
# =============================================================================


class TestFullSchemaCreation:
    """Tests for creating the complete schema."""

    @pytest.mark.asyncio
    async def test_create_all_constraints(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """create_all_constraints should create all required constraints."""
        from src.graph.schema import SchemaManager

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.create_all_constraints()

        # Should have multiple execute_write calls
        assert mock_neo4j_client.execute_write.call_count >= 3

    @pytest.mark.asyncio
    async def test_create_all_indexes(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """create_all_indexes should create all required indexes."""
        from src.graph.schema import SchemaManager

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.create_all_indexes()

        # Should have multiple execute_write calls
        assert mock_neo4j_client.execute_write.call_count >= 2

    @pytest.mark.asyncio
    async def test_init_schema(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """init_schema should create all constraints and indexes."""
        from src.graph.schema import SchemaManager

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.init_schema()

        # Should have calls for constraints + indexes
        assert mock_neo4j_client.execute_write.call_count >= 5


# =============================================================================
# Test: Cypher Generation
# =============================================================================


class TestCypherGeneration:
    """Tests for Cypher query generation."""

    def test_generate_create_node_cypher(self) -> None:
        """Should generate valid CREATE node Cypher."""
        from src.graph.schema import generate_create_node_cypher

        cypher = generate_create_node_cypher(
            label="Book",
            properties={"id": "book-1", "title": "Test Book"},
        )

        assert "CREATE" in cypher
        assert "Book" in cypher
        assert "id" in cypher
        assert "title" in cypher

    def test_generate_merge_node_cypher(self) -> None:
        """Should generate valid MERGE node Cypher."""
        from src.graph.schema import generate_merge_node_cypher

        cypher = generate_merge_node_cypher(
            label="Chapter",
            match_property="id",
            match_value="ch-1",
            set_properties={"title": "Updated Title"},
        )

        assert "MERGE" in cypher
        assert "Chapter" in cypher

    def test_generate_create_relationship_cypher(self) -> None:
        """Should generate valid CREATE relationship Cypher."""
        from src.graph.schema import generate_create_relationship_cypher

        cypher = generate_create_relationship_cypher(
            from_label="Book",
            from_id="book-1",
            to_label="Tier",
            to_id="tier-1",
            relationship="BELONGS_TO",
        )

        assert "MATCH" in cypher
        assert "CREATE" in cypher or "MERGE" in cypher
        assert "BELONGS_TO" in cypher


# =============================================================================
# Test: Schema Validation
# =============================================================================


class TestSchemaValidation:
    """Tests for schema validation utilities."""

    @pytest.mark.asyncio
    async def test_validate_schema_returns_status(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """validate_schema should return validation status."""
        from src.graph.schema import SchemaManager

        # Mock query to return existing constraints
        mock_neo4j_client.query = AsyncMock(return_value=[
            {"name": "constraint_tier_id"},
            {"name": "constraint_book_id"},
        ])

        manager = SchemaManager(client=mock_neo4j_client)
        result = await manager.validate_schema()

        assert isinstance(result, dict)
        assert "is_valid" in result

    @pytest.mark.asyncio
    async def test_get_existing_constraints(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """Should retrieve existing constraints from database."""
        from src.graph.schema import SchemaManager

        mock_neo4j_client.query = AsyncMock(return_value=[
            {"name": "constraint_1", "type": "UNIQUENESS"},
        ])

        manager = SchemaManager(client=mock_neo4j_client)
        constraints = await manager.get_existing_constraints()

        assert isinstance(constraints, list)
        mock_neo4j_client.query.assert_called()

    @pytest.mark.asyncio
    async def test_get_existing_indexes(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """Should retrieve existing indexes from database."""
        from src.graph.schema import SchemaManager

        mock_neo4j_client.query = AsyncMock(return_value=[
            {"name": "index_1", "labelsOrTypes": ["Book"]},
        ])

        manager = SchemaManager(client=mock_neo4j_client)
        indexes = await manager.get_existing_indexes()

        assert isinstance(indexes, list)
        mock_neo4j_client.query.assert_called()


# =============================================================================
# Test: Schema Drop/Reset
# =============================================================================


class TestSchemaReset:
    """Tests for schema reset/drop functionality."""

    @pytest.mark.asyncio
    async def test_drop_all_constraints(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """drop_all_constraints should remove all constraints."""
        from src.graph.schema import SchemaManager

        # Mock existing constraints
        mock_neo4j_client.query = AsyncMock(return_value=[
            {"name": "constraint_1"},
            {"name": "constraint_2"},
        ])

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.drop_all_constraints()

        # Should have queried for constraints, then dropped each
        assert mock_neo4j_client.execute_write.called

    @pytest.mark.asyncio
    async def test_drop_all_indexes(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """drop_all_indexes should remove all indexes."""
        from src.graph.schema import SchemaManager

        mock_neo4j_client.query = AsyncMock(return_value=[
            {"name": "index_1", "type": "BTREE"},
        ])

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.drop_all_indexes()

        assert mock_neo4j_client.execute_write.called

    @pytest.mark.asyncio
    async def test_reset_schema(
        self, mock_neo4j_client: MagicMock
    ) -> None:
        """reset_schema should drop all and recreate schema."""
        from src.graph.schema import SchemaManager

        mock_neo4j_client.query = AsyncMock(return_value=[])

        manager = SchemaManager(client=mock_neo4j_client)
        await manager.reset_schema()

        # Should have multiple calls for drop + create
        assert mock_neo4j_client.execute_write.call_count >= 2
