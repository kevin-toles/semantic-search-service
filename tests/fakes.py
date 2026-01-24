"""
Fake implementations for testing.

These test doubles implement the service protocols for unit/integration testing
without requiring real infrastructure (Qdrant, Neo4j, embedding models).
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any


class FakeVectorClient:
    """Fake vector client for testing."""

    def __init__(self, results: list[Any] | None = None) -> None:
        """Initialize with optional fake results."""
        self._results = results or []
        self._healthy = True

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        """Return fake search results."""
        # Consume parameters to satisfy protocol interface
        _ = (collection, vector, kwargs)
        await asyncio.sleep(0)  # Yield to event loop
        return self._results[:limit]

    async def health_check(self) -> bool:
        """Return health status."""
        await asyncio.sleep(0)  # Yield to event loop
        return self._healthy

    def set_healthy(self, healthy: bool) -> None:
        """Set health status for testing."""
        self._healthy = healthy

    def set_results(self, results: list[Any]) -> None:
        """Set results for testing."""
        self._results = results


class FakeGraphClient:
    """Fake graph client for testing."""

    def __init__(self) -> None:
        """Initialize fake client."""
        self._healthy = True
        self._traverse_result: dict[str, Any] = {"nodes": [], "edges": []}
        self._query_result: dict[str, Any] = {"records": [], "columns": []}
        self._relationship_scores: dict[str, Any] = {"scores": {}, "metadata": {}}

    async def traverse(
        self,
        start_node_id: str,
        relationship_types: list[str] | None = None,
        max_depth: int = 3,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return fake traversal results."""
        # Consume parameters to satisfy protocol interface
        _ = (start_node_id, relationship_types, max_depth, limit)
        await asyncio.sleep(0)  # Yield to event loop
        return self._traverse_result

    async def execute_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Return fake query results."""
        # Consume parameters to satisfy protocol interface
        _ = (cypher, parameters, timeout)
        await asyncio.sleep(0)  # Yield to event loop
        return self._query_result

    async def get_relationship_scores(
        self,
        node_ids: list[str],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return fake relationship scores."""
        # Consume parameters to satisfy protocol interface
        _ = (node_ids, context)
        await asyncio.sleep(0)  # Yield to event loop
        return self._relationship_scores

    async def health_check(self) -> bool:
        """Return health status."""
        await asyncio.sleep(0)  # Yield to event loop
        return self._healthy

    def set_healthy(self, healthy: bool) -> None:
        """Set health status for testing."""
        self._healthy = healthy

    def set_traverse_result(self, result: dict[str, Any]) -> None:
        """Set traversal result for testing."""
        self._traverse_result = result

    def set_query_result(self, result: dict[str, Any]) -> None:
        """Set query result for testing."""
        self._query_result = result

    def set_relationship_scores(self, scores: dict[str, Any]) -> None:
        """Set relationship scores for testing."""
        self._relationship_scores = scores


class FakeEmbeddingService:
    """Fake embedding service for testing."""

    def __init__(self, dimension: int = 768, model_name: str = "fake-model") -> None:
        """Initialize with embedding dimension."""
        self._dimension = dimension
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    async def embed(self, text: str) -> list[float]:
        """Return fake embedding."""
        await asyncio.sleep(0)  # Yield to event loop
        # Generate deterministic embedding based on text hash
        # SECURITY: MD5 used only for test double determinism, not for security.
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)  # noqa: S324
        return [(hash_value >> i) % 256 / 255.0 for i in range(self._dimension)]
