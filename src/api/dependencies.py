"""
Dependency injection for API services.

Provides protocols and container for service dependencies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorClientProtocol(Protocol):
    """Protocol for vector search client."""

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        """Search for similar vectors."""
        ...

    async def health_check(self) -> bool:
        """Check if service is healthy."""
        ...


@runtime_checkable
class GraphClientProtocol(Protocol):
    """Protocol for graph database client."""

    async def traverse(
        self,
        start_node_id: str,
        relationship_types: list[str] | None = None,
        max_depth: int = 3,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Traverse graph from a starting node."""
        ...

    async def execute_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        timeout: float = 30.0,  # NOSONAR - Protocol matches standard database client patterns
    ) -> dict[str, Any]:
        """Execute a Cypher query."""
        ...

    async def get_relationship_scores(
        self,
        node_ids: list[str],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get relationship-based scores for nodes."""
        ...

    async def health_check(self) -> bool:
        """Check if service is healthy."""
        ...


@runtime_checkable
class EmbeddingServiceProtocol(Protocol):
    """Protocol for embedding service."""

    @property
    def model_name(self) -> str:
        """Get the embedding model name."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


@dataclass
class ServiceConfig:
    """Configuration for services."""

    enable_hybrid_search: bool = True
    vector_collection: str = "chapters"  # Default to chapters (textbook data)
    default_alpha: float = 0.7
    max_results: int = 100
    embedding_dimension: int = 384  # all-MiniLM-L6-v2 dimension


@dataclass
class ServiceContainer:
    """Container for all service dependencies."""

    config: ServiceConfig = field(default_factory=ServiceConfig)
    vector_client: VectorClientProtocol | None = None
    graph_client: GraphClientProtocol | None = None
    embedding_service: EmbeddingServiceProtocol | None = None


# Note: Fake implementations for testing have been moved to tests/fakes.py
# Import from there for test fixtures:
#   from tests.fakes import FakeVectorClient, FakeGraphClient, FakeEmbeddingService
