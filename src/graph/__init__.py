# Graph module for Neo4j integration
"""
Graph layer for Neo4j operations including:
- Neo4jClient: Repository pattern client with connection pooling
- GraphTraversal: BFS/DFS algorithms for spider web model
- SchemaManager: Neo4j schema management utilities
"""

from src.graph.exceptions import (
    Neo4jConnectionError,
    Neo4jQueryError,
    Neo4jTransactionError,
)
from src.graph.neo4j_client import (
    FakeNeo4jClient,
    Neo4jClient,
    Neo4jClientProtocol,
)
from src.graph.schema import (
    NodeLabels,
    RelationshipLabels,
    SchemaManager,
)
from src.graph.traversal import (
    GraphTraversal,
    RelationshipType,
    TraversalDirection,
    TraversalResult,
)

__all__ = [
    # Exceptions
    "Neo4jConnectionError",
    "Neo4jQueryError",
    "Neo4jTransactionError",
    # Client
    "Neo4jClient",
    "Neo4jClientProtocol",
    "FakeNeo4jClient",
    # Traversal
    "GraphTraversal",
    "RelationshipType",
    "TraversalDirection",
    "TraversalResult",
    # Schema
    "NodeLabels",
    "RelationshipLabels",
    "SchemaManager",
]
