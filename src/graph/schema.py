"""
Graph schema helpers for Neo4j.

Provides utilities for:
- Node label definitions (Tier, Book, Chapter, Concept)
- Relationship type definitions (BELONGS_TO, HAS_CHAPTER, REFERENCES)
- Index creation helpers
- Constraint helpers
- Schema migration utilities
- Cypher generation for schema operations

Design follows:
- Clean separation of schema from data operations
- Neo4j 5.x schema best practices
- Repository pattern for client interaction
"""

from __future__ import annotations

from typing import Any

# =============================================================================
# Node Label Definitions
# =============================================================================


class NodeLabels:
    """Node label constants for the graph schema.

    These labels represent the node types in the spider web model:
    - TAXONOMY: Root node for a taxonomy
    - TIER: Organizational tier (Foundational, Best Practices, Operational)
    - BOOK: A book in the taxonomy
    - CHAPTER: A chapter within a book
    - CONCEPT: A concept mentioned in chapters
    """

    TAXONOMY = "Taxonomy"
    TIER = "Tier"
    BOOK = "Book"
    CHAPTER = "Chapter"
    CONCEPT = "Concept"


# =============================================================================
# Relationship Label Definitions
# =============================================================================


class RelationshipLabels:
    """Relationship type constants for the graph schema.

    These relationships represent connections in the spider web model:
    - BELONGS_TO: Book belongs to a Tier
    - HAS_CHAPTER: Book has chapters
    - HAS_CONCEPT: Chapter contains concepts
    - REFERENCES: Chapter references another chapter
    - RELATED_TO: Generic relationship for concept connections
    """

    BELONGS_TO = "BELONGS_TO"
    HAS_CHAPTER = "HAS_CHAPTER"
    HAS_CONCEPT = "HAS_CONCEPT"
    REFERENCES = "REFERENCES"
    RELATED_TO = "RELATED_TO"


# =============================================================================
# Cypher Generation Functions
# =============================================================================


def generate_create_node_cypher(
    label: str,
    properties: dict[str, Any],
) -> str:
    """Generate Cypher to CREATE a node.

    Args:
        label: Node label (e.g., "Book", "Chapter")
        properties: Node properties as dict

    Returns:
        Cypher CREATE statement
    """
    props_list = [f"{k}: ${k}" for k in properties]
    props_str = ", ".join(props_list)
    return f"CREATE (n:{label} {{{props_str}}}) RETURN n"


def generate_merge_node_cypher(
    label: str,
    match_property: str,
    match_value: str,
    set_properties: dict[str, Any] | None = None,
) -> str:
    """Generate Cypher to MERGE a node.

    MERGE creates if not exists, otherwise matches existing.

    Args:
        label: Node label
        match_property: Property to match on (e.g., "id")
        match_value: Value to match (passed as parameter at runtime)
        set_properties: Additional properties to set on merge

    Returns:
        Cypher MERGE statement
    """
    # match_value is used as parameter at runtime, not in the query template
    _ = match_value
    cypher = f"MERGE (n:{label} {{{match_property}: ${match_property}}})"

    if set_properties:
        set_clauses = [f"n.{k} = ${k}" for k in set_properties]
        cypher += f" ON CREATE SET {', '.join(set_clauses)}"
        cypher += f" ON MATCH SET {', '.join(set_clauses)}"

    cypher += " RETURN n"
    return cypher


def generate_create_relationship_cypher(
    from_label: str,
    from_id: str,
    to_label: str,
    to_id: str,
    relationship: str,
    properties: dict[str, Any] | None = None,
) -> str:
    """Generate Cypher to create a relationship between nodes.

    Args:
        from_label: Source node label
        from_id: Source node id (passed as parameter at runtime)
        to_label: Target node label
        to_id: Target node id (passed as parameter at runtime)
        relationship: Relationship type
        properties: Optional relationship properties

    Returns:
        Cypher statement for relationship creation
    """
    # from_id and to_id are used as parameters at runtime, not in the query template
    _ = (from_id, to_id)
    cypher = f"""
    MATCH (a:{from_label} {{id: $from_id}})
    MATCH (b:{to_label} {{id: $to_id}})
    MERGE (a)-[r:{relationship}]->(b)
    """

    if properties:
        set_clauses = [f"r.{k} = ${k}" for k in properties]
        cypher += f" SET {', '.join(set_clauses)}"

    cypher += " RETURN r"
    return cypher.strip()


# =============================================================================
# Schema Manager Class
# =============================================================================


class SchemaManager:
    """Manages Neo4j schema operations.

    Provides methods for:
    - Creating constraints (uniqueness, existence)
    - Creating indexes for performance
    - Validating schema
    - Resetting schema for testing

    Usage:
        manager = SchemaManager(client=neo4j_client)
        await manager.init_schema()  # Create all constraints and indexes
    """

    def __init__(self, client: Any) -> None:
        """Initialize schema manager.

        Args:
            client: Neo4j client (Neo4jClient or FakeNeo4jClient)
        """
        self._client = client

    # =========================================================================
    # Constraint Creation
    # =========================================================================

    async def create_tier_constraints(self) -> None:
        """Create unique constraint on Tier.id."""
        cypher = """
        CREATE CONSTRAINT constraint_tier_id IF NOT EXISTS
        FOR (t:Tier) REQUIRE t.id IS UNIQUE
        """
        await self._client.execute_write(cypher)

    async def create_book_constraints(self) -> None:
        """Create unique constraint on Book.id."""
        cypher = """
        CREATE CONSTRAINT constraint_book_id IF NOT EXISTS
        FOR (b:Book) REQUIRE b.id IS UNIQUE
        """
        await self._client.execute_write(cypher)

    async def create_chapter_constraints(self) -> None:
        """Create unique constraint on Chapter.id."""
        cypher = """
        CREATE CONSTRAINT constraint_chapter_id IF NOT EXISTS
        FOR (c:Chapter) REQUIRE c.id IS UNIQUE
        """
        await self._client.execute_write(cypher)

    async def create_concept_constraints(self) -> None:
        """Create unique constraint on Concept.id."""
        cypher = """
        CREATE CONSTRAINT constraint_concept_id IF NOT EXISTS
        FOR (c:Concept) REQUIRE c.id IS UNIQUE
        """
        await self._client.execute_write(cypher)

    async def create_taxonomy_constraints(self) -> None:
        """Create unique constraint on Taxonomy.id."""
        cypher = """
        CREATE CONSTRAINT constraint_taxonomy_id IF NOT EXISTS
        FOR (t:Taxonomy) REQUIRE t.id IS UNIQUE
        """
        await self._client.execute_write(cypher)

    async def create_all_constraints(self) -> None:
        """Create all required constraints."""
        await self.create_taxonomy_constraints()
        await self.create_tier_constraints()
        await self.create_book_constraints()
        await self.create_chapter_constraints()
        await self.create_concept_constraints()

    # =========================================================================
    # Index Creation
    # =========================================================================

    async def create_book_indexes(self) -> None:
        """Create indexes on Book properties for faster lookups."""
        cypher = """
        CREATE INDEX index_book_title IF NOT EXISTS
        FOR (b:Book) ON (b.title)
        """
        await self._client.execute_write(cypher)

    async def create_chapter_indexes(self) -> None:
        """Create indexes on Chapter properties."""
        cypher = """
        CREATE INDEX index_chapter_number IF NOT EXISTS
        FOR (c:Chapter) ON (c.chapter_num)
        """
        await self._client.execute_write(cypher)

    async def create_concept_indexes(self) -> None:
        """Create indexes on Concept properties."""
        cypher = """
        CREATE INDEX index_concept_name IF NOT EXISTS
        FOR (c:Concept) ON (c.name)
        """
        await self._client.execute_write(cypher)

    async def create_all_indexes(self) -> None:
        """Create all required indexes."""
        await self.create_book_indexes()
        await self.create_chapter_indexes()
        await self.create_concept_indexes()

    # =========================================================================
    # Full Schema Initialization
    # =========================================================================

    async def init_schema(self) -> None:
        """Initialize the complete schema.

        Creates all constraints and indexes needed for the spider web model.
        Safe to run multiple times (uses IF NOT EXISTS).
        """
        await self.create_all_constraints()
        await self.create_all_indexes()

    # =========================================================================
    # Schema Validation
    # =========================================================================

    async def validate_schema(self) -> dict[str, Any]:
        """Validate the current schema.

        Returns:
            dict with validation status and details
        """
        constraints = await self.get_existing_constraints()
        indexes = await self.get_existing_indexes()

        expected_constraints = {
            "constraint_taxonomy_id",
            "constraint_tier_id",
            "constraint_book_id",
            "constraint_chapter_id",
            "constraint_concept_id",
        }

        existing_constraint_names = {c.get("name", "") for c in constraints}

        missing_constraints = expected_constraints - existing_constraint_names

        return {
            "is_valid": len(missing_constraints) == 0,
            "constraints": constraints,
            "indexes": indexes,
            "missing_constraints": list(missing_constraints),
        }

    async def get_existing_constraints(self) -> list[dict[str, Any]]:
        """Get list of existing constraints.

        Returns:
            List of constraint definitions
        """
        cypher = "SHOW CONSTRAINTS"
        return await self._client.query(cypher)

    async def get_existing_indexes(self) -> list[dict[str, Any]]:
        """Get list of existing indexes.

        Returns:
            List of index definitions
        """
        cypher = "SHOW INDEXES"
        return await self._client.query(cypher)

    # =========================================================================
    # Schema Reset (for testing)
    # =========================================================================

    async def drop_all_constraints(self) -> None:
        """Drop all constraints.

        Warning: This is destructive! Use only for testing.
        """
        constraints = await self.get_existing_constraints()

        for constraint in constraints:
            name = constraint.get("name")
            if name:
                cypher = f"DROP CONSTRAINT {name} IF EXISTS"
                await self._client.execute_write(cypher)

    async def drop_all_indexes(self) -> None:
        """Drop all indexes.

        Warning: This is destructive! Use only for testing.
        """
        indexes = await self.get_existing_indexes()

        for index in indexes:
            name = index.get("name")
            index_type = index.get("type", "")
            # Skip constraint-backing indexes
            if name and "CONSTRAINT" not in index_type.upper():
                cypher = f"DROP INDEX {name} IF EXISTS"
                await self._client.execute_write(cypher)

    async def reset_schema(self) -> None:
        """Drop and recreate the entire schema.

        Warning: This is destructive! Use only for testing.
        """
        await self.drop_all_constraints()
        await self.drop_all_indexes()
        await self.init_schema()
