"""
WBS 3.2 GREEN: Qdrant vector search client implementation.

Design follows Pre-Implementation Analysis (WBS 3.0.1-3.0.4):
- Repository Pattern: Abstraction over vector storage (GUIDELINES line 795)
- FakeClient for testing: Building fakes for abstractions (GUIDELINES line 276)
- Connection pooling: Reuse client instance (Anti-Pattern #12)
- Custom exceptions: Avoid shadowing builtins (Anti-Pattern #7, #13)
- Async context manager: Proper resource management (Anti-Pattern #42, #43)
- HNSW indexing: Approximate NN search with logarithmic scaling (AI Engineering p.158)
- Score normalization: Normalize to [0, 1] range before fusion

Anti-Pattern Mitigations Applied:
- QdrantConnectionError/QdrantSearchError (not ConnectionError)
- Lazy client initialization
- Full type hints on all public methods
- No hardcoded collection names (use Settings)
- Always use `limit` parameter (default: 10)
- No empty f-strings (use regular strings for static messages)
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.search.exceptions import QdrantConnectionError, QdrantSearchError

# =============================================================================
# Constants (Anti-Pattern #50: Duplicated Literals)
# =============================================================================

_DEFAULT_LIMIT = 10
_DEFAULT_HNSW_M = 16
_DEFAULT_HNSW_EF_CONSTRUCT = 100


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SearchResult:
    """Represents a single search result from Qdrant.

    Attributes:
        id: Unique identifier for the document
        score: Similarity score in [0, 1] range (normalized)
        payload: Document metadata and content
    """

    id: str
    score: float
    payload: dict[str, Any] | None

    def __post_init__(self) -> None:
        """Validate score is in normalized range."""
        # Qdrant cosine similarity returns [-1, 1], normalize to [0, 1]
        # For other metrics, score may already be in [0, 1]
        if self.score < 0:
            self.score = (self.score + 1) / 2
        elif self.score > 1:
            self.score = 1.0


# =============================================================================
# Protocol for Duck Typing
# =============================================================================


@runtime_checkable
class QdrantSearchClientProtocol(Protocol):
    """Protocol defining the QdrantSearchClient interface.

    Enables duck typing - any class implementing these methods
    can be used interchangeably (Repository pattern).
    """

    async def connect(self) -> None:
        """Connect to Qdrant."""
        ...

    async def close(self) -> None:
        """Close connection."""
        ...

    async def search(
        self,
        embedding: list[float],
        limit: int | None = None,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Execute vector similarity search."""
        ...

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a single document."""
        ...

    async def upsert_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update multiple documents."""
        ...

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs."""
        ...

    async def ensure_collection(
        self,
        vector_size: int,
        hnsw_m: int | None = None,
        hnsw_ef_construct: int | None = None,
    ) -> None:
        """Ensure collection exists with proper configuration."""
        ...


# =============================================================================
# Real Implementation
# =============================================================================


class QdrantSearchClient:
    """Qdrant client implementing Repository pattern.

    Provides connection pooling by reusing a single client instance
    (addresses Anti-Pattern #12: new client per request).

    Usage:
        # As async context manager (recommended)
        async with QdrantSearchClient(settings=settings) as client:
            results = await client.search(embedding=[0.1] * 384)

        # Manual connection management
        client = QdrantSearchClient(settings=settings)
        await client.connect()
        results = await client.search(embedding=[0.1] * 384)
        await client.close()
    """

    def __init__(self, settings: Any) -> None:
        """Initialize client with Settings object.

        Args:
            settings: Settings object with qdrant_url, qdrant_collection,
                      and optional qdrant_api_key attributes

        Note:
            Client is NOT created here - uses lazy initialization.
            Call connect() or use as async context manager.
        """
        self._settings = settings
        self._url = settings.qdrant_url
        self._collection = getattr(settings, "qdrant_collection", "chapters")
        self._api_key = getattr(settings, "qdrant_api_key", None)
        self._client: AsyncQdrantClient | None = None
        self._default_limit = _DEFAULT_LIMIT

    async def connect(self) -> None:
        """Connect to Qdrant server.

        Creates AsyncQdrantClient and verifies connectivity.

        Raises:
            QdrantConnectionError: If connection fails
        """
        try:
            self._client = AsyncQdrantClient(
                url=self._url,
                api_key=self._api_key,
            )
            # Verify connectivity
            await self._client.get_collections()
        except Exception as e:
            self._client = None
            raise QdrantConnectionError(
                f"Failed to connect to Qdrant at {self._url}: {e}",
                cause=e,
            ) from e

    async def close(self) -> None:
        """Close the Qdrant client connection.

        Idempotent - safe to call multiple times.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def __aenter__(self) -> QdrantSearchClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - always closes connection."""
        await self.close()

    def _ensure_connected(self) -> None:
        """Verify client is connected.

        Raises:
            QdrantConnectionError: If not connected
        """
        if self._client is None:
            raise QdrantConnectionError("Client is not connected. Call connect() first.")

    async def search(
        self,
        embedding: list[float],
        limit: int | None = None,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Execute vector similarity search.

        Args:
            embedding: Query vector (must match collection vector size)
            limit: Maximum number of results (default: 10)
            filter_conditions: Optional metadata filters
            score_threshold: Minimum score threshold

        Returns:
            List of SearchResult objects sorted by score descending

        Raises:
            QdrantConnectionError: If not connected
            QdrantSearchError: If search fails
        """
        self._ensure_connected()

        actual_limit = limit if limit is not None else self._default_limit

        # Build filter if conditions provided
        query_filter = None
        if filter_conditions:
            must_conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_conditions.items()
            ]
            query_filter = Filter(must=must_conditions)

        try:
            points = await self._client.search(  # type: ignore[union-attr]
                collection_name=self._collection,
                query_vector=embedding,
                limit=actual_limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )

            return [
                SearchResult(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload,
                )
                for point in points
            ]
        except Exception as e:
            raise QdrantSearchError(
                f"Search failed in collection '{self._collection}': {e}",
                cause=e,
            ) from e

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update a single document.

        Args:
            id: Unique document identifier
            embedding: Document vector
            payload: Optional metadata

        Raises:
            QdrantConnectionError: If not connected
            QdrantSearchError: If upsert fails
        """
        self._ensure_connected()

        point = PointStruct(
            id=id,
            vector=embedding,
            payload=payload or {},
        )

        try:
            await self._client.upsert(  # type: ignore[union-attr]
                collection_name=self._collection,
                points=[point],
            )
        except Exception as e:
            raise QdrantSearchError(
                f"Upsert failed for document '{id}': {e}",
                cause=e,
            ) from e

    async def upsert_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update multiple documents efficiently.

        Args:
            ids: List of unique document identifiers
            embeddings: List of document vectors
            payloads: Optional list of metadata dicts

        Raises:
            ValueError: If list lengths don't match
            QdrantConnectionError: If not connected
            QdrantSearchError: If upsert fails
        """
        self._ensure_connected()

        # Validate lengths match
        if len(ids) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(ids)} IDs but {len(embeddings)} embeddings"
            )

        actual_payloads = payloads if payloads else [{} for _ in ids]
        if len(actual_payloads) != len(ids):
            raise ValueError(
                f"Length mismatch: {len(ids)} IDs but {len(actual_payloads)} payloads"
            )

        points = [
            PointStruct(
                id=doc_id,
                vector=embedding,
                payload=payload,
            )
            for doc_id, embedding, payload in zip(ids, embeddings, actual_payloads, strict=False)
        ]

        try:
            await self._client.upsert(  # type: ignore[union-attr]
                collection_name=self._collection,
                points=points,
            )
        except Exception as e:
            raise QdrantSearchError(
                f"Batch upsert failed for {len(ids)} documents: {e}",
                cause=e,
            ) from e

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Raises:
            QdrantConnectionError: If not connected
            QdrantSearchError: If delete fails
        """
        self._ensure_connected()

        try:
            await self._client.delete(  # type: ignore[union-attr]
                collection_name=self._collection,
                points_selector=ids,
            )
        except Exception as e:
            raise QdrantSearchError(
                f"Delete failed for {len(ids)} documents: {e}",
                cause=e,
            ) from e

    async def ensure_collection(
        self,
        vector_size: int,
        hnsw_m: int | None = None,
        hnsw_ef_construct: int | None = None,
    ) -> None:
        """Ensure collection exists with proper configuration.

        Creates collection if it doesn't exist, configuring HNSW index
        for approximate nearest neighbor search (AI Engineering p.158).

        Args:
            vector_size: Dimension of embedding vectors
            hnsw_m: HNSW M parameter (default: 16)
            hnsw_ef_construct: HNSW ef_construct parameter (default: 100)

        Raises:
            QdrantConnectionError: If not connected
            QdrantSearchError: If collection creation fails
        """
        self._ensure_connected()

        actual_m = hnsw_m if hnsw_m is not None else _DEFAULT_HNSW_M
        actual_ef = hnsw_ef_construct if hnsw_ef_construct is not None else _DEFAULT_HNSW_EF_CONSTRUCT

        try:
            exists = await self._client.collection_exists(  # type: ignore[union-attr]
                collection_name=self._collection
            )

            if not exists:
                await self._client.create_collection(  # type: ignore[union-attr]
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=actual_m,
                        ef_construct=actual_ef,
                    ),
                )
        except Exception as e:
            raise QdrantSearchError(
                f"Failed to ensure collection '{self._collection}': {e}",
                cause=e,
            ) from e


# =============================================================================
# Fake Implementation for Testing
# =============================================================================


class FakeQdrantSearchClient:
    """In-memory fake Qdrant client for unit testing.

    Based on Architecture Patterns with Python p.95 "FakeRepository".
    Implements the same interface as QdrantSearchClient (duck typing).
    """

    def __init__(self) -> None:
        """Initialize with empty in-memory storage."""
        self._storage: dict[str, dict[str, Any]] = {}
        self._connected = False

    async def connect(self) -> None:
        """Simulate connection (always succeeds)."""
        await asyncio.sleep(0)  # Yield to event loop
        self._connected = True

    async def close(self) -> None:
        """Simulate disconnection."""
        await asyncio.sleep(0)  # Yield to event loop
        self._connected = False

    async def __aenter__(self) -> FakeQdrantSearchClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def search(
        self,
        embedding: list[float],
        limit: int | None = None,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Simulate vector search with cosine similarity.

        Args:
            embedding: Query vector
            limit: Maximum results (default: 10)
            filter_conditions: Optional metadata filters
            score_threshold: Minimum score threshold

        Returns:
            List of SearchResult objects sorted by score descending
        """
        await asyncio.sleep(0)  # Yield to event loop
        actual_limit = limit if limit is not None else _DEFAULT_LIMIT

        results = []
        for doc_id, data in self._storage.items():
            # Calculate cosine similarity
            stored_embedding = data["embedding"]
            score = self._cosine_similarity(embedding, stored_embedding)

            # Apply score threshold
            if score_threshold is not None and score < score_threshold:
                continue

            # Apply filters
            if filter_conditions:
                payload = data.get("payload", {})
                if not all(
                    payload.get(k) == v for k, v in filter_conditions.items()
                ):
                    continue

            results.append(
                SearchResult(
                    id=doc_id,
                    score=score,
                    payload=data.get("payload"),
                )
            )

        # Sort by score descending and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:actual_limit]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        # Normalize from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Store document in memory."""
        await asyncio.sleep(0)  # Yield to event loop
        self._storage[id] = {
            "embedding": embedding,
            "payload": payload or {},
        }

    async def upsert_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store multiple documents in memory."""
        if len(ids) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(ids)} IDs but {len(embeddings)} embeddings"
            )

        actual_payloads = payloads if payloads else [{} for _ in ids]

        for doc_id, embedding, payload in zip(ids, embeddings, actual_payloads, strict=False):
            await self.upsert(id=doc_id, embedding=embedding, payload=payload)

    async def delete(self, ids: list[str]) -> None:
        """Remove documents from memory."""
        await asyncio.sleep(0)  # Yield to event loop
        for doc_id in ids:
            self._storage.pop(doc_id, None)

    async def ensure_collection(
        self,
        _vector_size: int,
        _hnsw_m: int | None = None,
        _hnsw_ef_construct: int | None = None,
    ) -> None:
        """No-op for fake client (collection always "exists")."""
        await asyncio.sleep(0)  # Yield to event loop
        # Fake client doesn't need actual collection setup
