"""
Neo4j client module implementing Repository pattern.

Design follows:
- Repository Pattern: Abstraction over persistent storage (GUIDELINES line 795)
- FakeClient for testing: Building fakes for abstractions (GUIDELINES line 276)
- Connection pooling: Reuse driver instance (Anti-Pattern #12)
- Custom exceptions: Avoid shadowing builtins (Anti-Pattern #7, #13)
- Async context manager: Proper resource management (Anti-Pattern #42, #43)

This module provides:
- Neo4jClient: Real client for production use
- FakeNeo4jClient: In-memory fake for testing
- Both share the same interface (duck typing)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ClientError, ServiceUnavailable

from src.graph.exceptions import (
    Neo4jConnectionError,
    Neo4jQueryError,
    Neo4jTransactionError,
)

if TYPE_CHECKING:
    from neo4j import AsyncDriver


@runtime_checkable
class Neo4jClientProtocol(Protocol):
    """Protocol defining the Neo4jClient interface.

    Enables duck typing - any class implementing these methods
    can be used interchangeably (Repository pattern).
    """

    async def connect(self) -> None:
        """Connect to Neo4j."""
        ...

    async def close(self) -> None:
        """Close connection."""
        ...

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


class Neo4jClient:
    """Neo4j client implementing Repository pattern.

    Provides connection pooling by reusing a single driver instance
    (addresses Anti-Pattern #12: new client per request).

    Usage:
        # As async context manager (recommended)
        async with Neo4jClient(settings=settings) as client:
            results = await client.query("MATCH (n) RETURN n LIMIT 10")

        # Manual connection management
        client = Neo4jClient(settings=settings)
        await client.connect()
        results = await client.query("MATCH (n) RETURN n")
        await client.close()
    """

    def __init__(self, settings: Any) -> None:
        """Initialize client with Settings object.

        Args:
            settings: Settings object with neo4j_uri, neo4j_user,
                      neo4j_password, neo4j_database attributes

        Note:
            Driver is NOT created here - uses lazy initialization.
            Call connect() or use as async context manager.
        """
        self._settings = settings
        self._uri = settings.neo4j_uri
        self._user = settings.neo4j_user
        self._password = settings.neo4j_password
        self._database = settings.neo4j_database
        self._driver: AsyncDriver | None = None

    @property
    def uri(self) -> str:
        """Get the connection URI."""
        return self._uri

    @property
    def database(self) -> str:
        """Get the database name."""
        return self._database

    @property
    def is_connected(self) -> bool:
        """Check if driver is initialized."""
        return self._driver is not None

    async def connect(self) -> None:
        """Create driver and verify connectivity.

        Raises:
            Neo4jConnectionError: If connection fails.
                Uses custom exception name to avoid shadowing
                Python's built-in ConnectionError (Anti-Pattern #7, #13).
        """
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
            # Verify connectivity immediately
            await self._driver.verify_connectivity()
        except ServiceUnavailable as e:
            self._driver = None
            raise Neo4jConnectionError(
                f"Failed to connect to Neo4j at {self._uri}",
                cause=e,
            ) from e
        except Exception as e:
            self._driver = None
            raise Neo4jConnectionError(
                f"Unexpected error connecting to Neo4j: {e}",
                cause=e,
            ) from e

    async def close(self) -> None:
        """Close the driver connection.

        Safe to call even if not connected (no-op).
        """
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def __aenter__(self) -> Neo4jClient:
        """Async context manager entry - connect to Neo4j."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - close connection."""
        await self.close()

    def _ensure_connected(self) -> None:
        """Raise if not connected.

        Raises:
            Neo4jConnectionError: If driver is not initialized.
        """
        if self._driver is None:
            raise Neo4jConnectionError(
                "Not connected to Neo4j. Call connect() first or use async context manager."
            )

    async def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a read query and return results.

        Args:
            cypher: Cypher query string
            parameters: Optional query parameters

        Returns:
            List of records as dictionaries

        Raises:
            Neo4jConnectionError: If not connected
            Neo4jQueryError: If query execution fails
        """
        self._ensure_connected()
        assert self._driver is not None  # For type checker

        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(cypher, parameters or {})
                records = await result.data()
                return records
        except ClientError as e:
            raise Neo4jQueryError(
                f"Query failed: {e}",
                query=cypher,
                cause=e,
            ) from e
        except Exception as e:
            raise Neo4jQueryError(
                f"Unexpected error executing query: {e}",
                query=cypher,
                cause=e,
            ) from e

    async def execute_write(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a write query in a transaction.

        Args:
            cypher: Cypher query string (CREATE, MERGE, DELETE, etc.)
            parameters: Optional query parameters

        Returns:
            List of records as dictionaries (if any)

        Raises:
            Neo4jConnectionError: If not connected
            Neo4jTransactionError: If transaction fails
        """
        self._ensure_connected()
        assert self._driver is not None  # For type checker

        try:
            async with self._driver.session(database=self._database) as session:

                async def _transaction_work(tx: Any) -> list[dict[str, Any]]:
                    result = await tx.run(cypher, parameters or {})
                    return await result.data()

                return await session.execute_write(_transaction_work)
        except ClientError as e:
            raise Neo4jTransactionError(
                f"Transaction failed: {e}",
            ) from e
        except Exception as e:
            raise Neo4jTransactionError(
                f"Unexpected error in transaction: {e}",
            ) from e


class FakeNeo4jClient:
    """In-memory fake Neo4j client for testing.

    Implements the same interface as Neo4jClient but stores
    data in memory. Useful for unit tests that don't need
    a real database.

    Design follows GUIDELINES line 276:
    "Building fakes for your abstractions is an excellent way
    to get design feedback: if it's hard to fake, it's probably
    hard to use."

    Usage:
        fake = FakeNeo4jClient()
        async with fake:
            await fake.execute_write(
                "CREATE (n:Person {name: $name})",
                {"name": "Alice"}
            )
            results = await fake.query("MATCH (n:Person) RETURN n")
    """

    def __init__(self) -> None:
        """Initialize fake client with empty storage."""
        self._connected = False
        self._nodes: list[dict[str, Any]] = []
        self._query_results: list[dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        """Check if fake is 'connected'."""
        return self._connected

    async def connect(self) -> None:
        """Simulate connecting (always succeeds)."""
        await asyncio.sleep(0)  # Yield to event loop for true async
        self._connected = True

    async def close(self) -> None:
        """Simulate closing connection."""
        await asyncio.sleep(0)  # Yield to event loop for true async
        self._connected = False

    async def __aenter__(self) -> FakeNeo4jClient:
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

    def _ensure_connected(self) -> None:
        """Raise if not 'connected'."""
        if not self._connected:
            raise Neo4jConnectionError("Fake client not connected")

    async def query(
        self,
        cypher: str,  # noqa: ARG002 - Required for interface compatibility
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return stored query results matching parameters.

        For more sophisticated testing, use set_query_results()
        to configure responses.

        This fake implementation returns nodes wrapped in {"n": node_data}
        to match Neo4j's return format for RETURN n queries.
        """
        await asyncio.sleep(0)  # Yield to event loop for true async
        self._ensure_connected()
        # Return any pre-configured results first
        if self._query_results:
            return self._query_results

        # Filter stored nodes based on parameters if provided
        if parameters and self._nodes:
            # Return matching nodes wrapped in {"n": ...} format
            matching = []
            for node in self._nodes:
                # Check if all parameters match the node
                matches = all(
                    node.get(key) == value
                    for key, value in parameters.items()
                )
                if matches:
                    matching.append({"n": node})
            return matching

        # Return all nodes wrapped in {"n": ...} format
        return [{"n": node} for node in self._nodes]

    async def execute_write(
        self,
        cypher: str,  # noqa: ARG002 - Required for interface compatibility
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Store node data from parameters.

        This is a simplified implementation that extracts
        data from parameters for basic testing scenarios.
        """
        await asyncio.sleep(0)  # Yield to event loop for true async
        self._ensure_connected()
        if parameters:
            self._nodes.append(parameters.copy())
        return []

    def set_query_results(self, results: list[dict[str, Any]]) -> None:
        """Configure results to return from query().

        Args:
            results: List of dicts to return from next query call
        """
        self._query_results = results

    def get_stored_nodes(self) -> list[dict[str, Any]]:
        """Get all nodes stored via execute_write().

        Returns:
            List of node data dictionaries
        """
        return self._nodes.copy()

    def clear(self) -> None:
        """Clear all stored data."""
        self._nodes.clear()
        self._query_results.clear()
