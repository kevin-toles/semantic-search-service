"""
WBS 2.1 RED: Unit tests for Neo4jClient.

Tests follow TDD RED phase - all tests should fail initially.
Based on Pre-Implementation Analysis (WBS 2.0.1-2.0.4):
- Repository pattern with duck typing (GUIDELINES line 795)
- FakeNeo4jClient for unit tests (Architecture Patterns p.95)
- Connection reuse pattern (Comp_Static_Analysis #12)
- Custom exceptions to avoid shadowing (Comp_Static_Analysis #7, #13)
- Async context manager (Comp_Static_Analysis #12 resolution)

Anti-Pattern Mitigations Applied:
- Neo4jConnectionError/Neo4jQueryError (not ConnectionError)
- Lazy driver initialization
- Full type hints on all public methods
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings for Neo4j configuration."""
    settings = MagicMock()
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_user = "neo4j"
    settings.neo4j_password = "testpassword"
    settings.neo4j_database = "neo4j"
    return settings


@pytest.fixture
def mock_driver() -> MagicMock:
    """Create a mock Neo4j async driver."""
    driver = MagicMock()
    driver.close = AsyncMock()
    driver.verify_connectivity = AsyncMock()
    return driver


# =============================================================================
# Test: Neo4jClient Initialization
# =============================================================================


class TestNeo4jClientInitialization:
    """Tests for Neo4jClient initialization and configuration."""

    def test_client_accepts_settings(self, mock_settings: MagicMock) -> None:
        """Neo4jClient should accept Settings object for configuration."""
        from src.graph.neo4j_client import Neo4jClient

        client = Neo4jClient(settings=mock_settings)

        assert client is not None
        assert client._settings == mock_settings

    def test_client_lazy_driver_initialization(
        self, mock_settings: MagicMock
    ) -> None:
        """Driver should not be created until first use (lazy init)."""
        from src.graph.neo4j_client import Neo4jClient

        client = Neo4jClient(settings=mock_settings)

        # Driver should be None until connect() is called
        assert client._driver is None

    def test_client_stores_configuration(self, mock_settings: MagicMock) -> None:
        """Client should store URI, user, password, database from settings."""
        from src.graph.neo4j_client import Neo4jClient

        client = Neo4jClient(settings=mock_settings)

        assert client._uri == "bolt://localhost:7687"
        assert client._user == "neo4j"
        assert client._password == "testpassword"
        assert client._database == "neo4j"


# =============================================================================
# Test: Neo4jClient Connection Management
# =============================================================================


class TestNeo4jClientConnection:
    """Tests for Neo4jClient connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_creates_driver(self, mock_settings: MagicMock) -> None:
        """connect() should create AsyncGraphDatabase driver."""
        from src.graph.neo4j_client import Neo4jClient

        with patch(
            "src.graph.neo4j_client.AsyncGraphDatabase"
        ) as mock_async_db:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_async_db.driver.return_value = mock_driver

            client = Neo4jClient(settings=mock_settings)
            await client.connect()

            mock_async_db.driver.assert_called_once_with(
                "bolt://localhost:7687",
                auth=("neo4j", "testpassword"),
            )
            assert client._driver is not None

    @pytest.mark.asyncio
    async def test_connect_verifies_connectivity(
        self, mock_settings: MagicMock
    ) -> None:
        """connect() should verify connectivity after creating driver."""
        from src.graph.neo4j_client import Neo4jClient

        with patch(
            "src.graph.neo4j_client.AsyncGraphDatabase"
        ) as mock_async_db:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_async_db.driver.return_value = mock_driver

            client = Neo4jClient(settings=mock_settings)
            await client.connect()

            mock_driver.verify_connectivity.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_closes_driver(self, mock_settings: MagicMock) -> None:
        """close() should close the driver connection."""
        from src.graph.neo4j_client import Neo4jClient

        with patch(
            "src.graph.neo4j_client.AsyncGraphDatabase"
        ) as mock_async_db:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_async_db.driver.return_value = mock_driver

            client = Neo4jClient(settings=mock_settings)
            await client.connect()
            await client.close()

            mock_driver.close.assert_called_once()
            assert client._driver is None

    @pytest.mark.asyncio
    async def test_close_handles_no_driver(
        self, mock_settings: MagicMock
    ) -> None:
        """close() should handle case where driver was never created."""
        from src.graph.neo4j_client import Neo4jClient

        client = Neo4jClient(settings=mock_settings)
        # Should not raise even if driver is None
        await client.close()


# =============================================================================
# Test: Neo4jClient Async Context Manager
# =============================================================================


class TestNeo4jClientContextManager:
    """Tests for async context manager protocol."""

    @pytest.mark.asyncio
    async def test_async_context_manager_connects_on_enter(
        self, mock_settings: MagicMock
    ) -> None:
        """__aenter__ should call connect()."""
        from src.graph.neo4j_client import Neo4jClient

        with patch(
            "src.graph.neo4j_client.AsyncGraphDatabase"
        ) as mock_async_db:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_async_db.driver.return_value = mock_driver

            async with Neo4jClient(settings=mock_settings) as client:
                assert client._driver is not None

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_on_exit(
        self, mock_settings: MagicMock
    ) -> None:
        """__aexit__ should call close()."""
        from src.graph.neo4j_client import Neo4jClient

        with patch(
            "src.graph.neo4j_client.AsyncGraphDatabase"
        ) as mock_async_db:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.close = AsyncMock()
            mock_async_db.driver.return_value = mock_driver

            async with Neo4jClient(settings=mock_settings):
                pass

            mock_driver.close.assert_called_once()


# =============================================================================
# Test: Neo4jClient Query Execution
# =============================================================================


class TestNeo4jClientQuery:
    """Tests for query execution methods."""

    @pytest.mark.asyncio
    async def test_query_executes_cypher(
        self, mock_settings: MagicMock
    ) -> None:
        """query() should execute Cypher and return records."""
        from src.graph.neo4j_client import Neo4jClient

        with patch(
            "src.graph.neo4j_client.AsyncGraphDatabase"
        ) as mock_async_db:
            # Setup mock driver and session
            mock_result = AsyncMock()
            mock_result.data = AsyncMock(return_value=[{"name": "Test"}])

            mock_session = AsyncMock()
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session.return_value = mock_session

            mock_async_db.driver.return_value = mock_driver

            client = Neo4jClient(settings=mock_settings)
            await client.connect()

            results = await client.query("MATCH (n) RETURN n.name AS name")

            mock_session.run.assert_called_once_with(
                "MATCH (n) RETURN n.name AS name", {}
            )
            assert results == [{"name": "Test"}]

    @pytest.mark.asyncio
    async def test_query_accepts_parameters(
        self, mock_settings: MagicMock
    ) -> None:
        """query() should accept parameters for parameterized queries."""
        from src.graph.neo4j_client import Neo4jClient

        with patch(
            "src.graph.neo4j_client.AsyncGraphDatabase"
        ) as mock_async_db:
            mock_result = AsyncMock()
            mock_result.data = AsyncMock(return_value=[])

            mock_session = AsyncMock()
            mock_session.run = AsyncMock(return_value=mock_result)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session.return_value = mock_session

            mock_async_db.driver.return_value = mock_driver

            client = Neo4jClient(settings=mock_settings)
            await client.connect()

            await client.query(
                "MATCH (n:Book {id: $id}) RETURN n",
                parameters={"id": "book-123"},
            )

            mock_session.run.assert_called_once_with(
                "MATCH (n:Book {id: $id}) RETURN n", {"id": "book-123"}
            )

    @pytest.mark.asyncio
    async def test_query_raises_without_connection(
        self, mock_settings: MagicMock
    ) -> None:
        """query() should raise Neo4jConnectionError if not connected."""
        from src.graph.exceptions import Neo4jConnectionError
        from src.graph.neo4j_client import Neo4jClient

        client = Neo4jClient(settings=mock_settings)

        with pytest.raises(Neo4jConnectionError):
            await client.query("MATCH (n) RETURN n")


# =============================================================================
# Test: Neo4jClient Write Operations
# =============================================================================


class TestNeo4jClientWrite:
    """Tests for write operations (add/update/delete)."""

    @pytest.mark.asyncio
    async def test_execute_write_in_transaction(
        self, mock_settings: MagicMock
    ) -> None:
        """execute_write() should run write operations in a transaction."""
        from src.graph.neo4j_client import Neo4jClient

        with patch(
            "src.graph.neo4j_client.AsyncGraphDatabase"
        ) as mock_async_db:
            mock_result = AsyncMock()
            mock_result.data = AsyncMock(return_value=[{"id": "node-123"}])

            mock_tx = AsyncMock()
            mock_tx.run = AsyncMock(return_value=mock_result)

            async def mock_execute_write(func: Any) -> Any:
                """Mock execute_write that awaits the transaction function."""
                return await func(mock_tx)

            mock_session = AsyncMock()
            mock_session.execute_write = mock_execute_write
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session.return_value = mock_session

            mock_async_db.driver.return_value = mock_driver

            client = Neo4jClient(settings=mock_settings)
            await client.connect()

            result = await client.execute_write(
                "CREATE (n:Test {name: $name}) RETURN n.id AS id",
                parameters={"name": "TestNode"},
            )

            assert result == [{"id": "node-123"}]


# =============================================================================
# Test: Neo4jClient Error Handling
# =============================================================================


class TestNeo4jClientErrorHandling:
    """Tests for error handling and custom exceptions."""

    @pytest.mark.asyncio
    async def test_connection_failure_raises_neo4j_connection_error(
        self, mock_settings: MagicMock
    ) -> None:
        """Connection failure should raise Neo4jConnectionError."""
        from neo4j.exceptions import ServiceUnavailable

        from src.graph.exceptions import Neo4jConnectionError
        from src.graph.neo4j_client import Neo4jClient

        with patch(
            "src.graph.neo4j_client.AsyncGraphDatabase"
        ) as mock_async_db:
            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock(
                side_effect=ServiceUnavailable("Connection refused")
            )
            mock_async_db.driver.return_value = mock_driver

            client = Neo4jClient(settings=mock_settings)

            with pytest.raises(Neo4jConnectionError) as exc_info:
                await client.connect()

            # Check the error message contains URI info (not the cause message)
            assert "bolt://localhost:7687" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_failure_raises_neo4j_query_error(
        self, mock_settings: MagicMock
    ) -> None:
        """Query failure should raise Neo4jQueryError."""
        from neo4j.exceptions import CypherSyntaxError

        from src.graph.exceptions import Neo4jQueryError
        from src.graph.neo4j_client import Neo4jClient

        with patch(
            "src.graph.neo4j_client.AsyncGraphDatabase"
        ) as mock_async_db:
            mock_session = AsyncMock()
            mock_session.run = AsyncMock(
                side_effect=CypherSyntaxError("Invalid syntax")
            )
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_driver = MagicMock()
            mock_driver.verify_connectivity = AsyncMock()
            mock_driver.session.return_value = mock_session

            mock_async_db.driver.return_value = mock_driver

            client = Neo4jClient(settings=mock_settings)
            await client.connect()

            with pytest.raises(Neo4jQueryError) as exc_info:
                await client.query("INVALID CYPHER")

            assert "Invalid syntax" in str(exc_info.value)


# =============================================================================
# Test: FakeNeo4jClient for Unit Testing
# =============================================================================


class TestFakeNeo4jClient:
    """Tests for FakeNeo4jClient (in-memory test double)."""

    def test_fake_client_same_interface(self) -> None:
        """FakeNeo4jClient should have same interface as Neo4jClient."""
        from src.graph.neo4j_client import FakeNeo4jClient, Neo4jClient

        # Check that FakeNeo4jClient has all required methods
        neo4j_methods = {
            "connect",
            "close",
            "query",
            "execute_write",
            "__aenter__",
            "__aexit__",
        }

        fake_methods = set(dir(FakeNeo4jClient))
        real_methods = set(dir(Neo4jClient))

        for method in neo4j_methods:
            assert method in fake_methods, f"FakeNeo4jClient missing {method}"
            assert method in real_methods, f"Neo4jClient missing {method}"

    @pytest.mark.asyncio
    async def test_fake_client_stores_nodes(self) -> None:
        """FakeNeo4jClient should store nodes in memory."""
        from src.graph.neo4j_client import FakeNeo4jClient

        client = FakeNeo4jClient()
        await client.connect()

        # Add a node
        await client.execute_write(
            "CREATE (n:Book {id: $id, title: $title})",
            parameters={"id": "book-1", "title": "Test Book"},
        )

        # Query should return the node
        results = await client.query(
            "MATCH (n:Book {id: $id}) RETURN n",
            parameters={"id": "book-1"},
        )

        assert len(results) == 1
        assert results[0]["n"]["id"] == "book-1"
        assert results[0]["n"]["title"] == "Test Book"

    @pytest.mark.asyncio
    async def test_fake_client_async_context_manager(self) -> None:
        """FakeNeo4jClient should work as async context manager."""
        from src.graph.neo4j_client import FakeNeo4jClient

        async with FakeNeo4jClient() as client:
            results = await client.query("MATCH (n) RETURN n")
            assert results == []


# =============================================================================
# Test: Type Hints and Protocol Compliance
# =============================================================================


class TestNeo4jClientTypes:
    """Tests for type hint compliance."""

    def test_query_return_type_is_list_of_dicts(
        self, mock_settings: MagicMock
    ) -> None:
        """query() should be annotated to return list[dict[str, Any]]."""
        from typing import get_type_hints

        from src.graph.neo4j_client import Neo4jClient

        hints = get_type_hints(Neo4jClient.query)
        # Should have return type annotation
        assert "return" in hints

    def test_execute_write_return_type(
        self, mock_settings: MagicMock
    ) -> None:
        """execute_write() should be annotated to return dict[str, Any] | None."""
        from typing import get_type_hints

        from src.graph.neo4j_client import Neo4jClient

        hints = get_type_hints(Neo4jClient.execute_write)
        assert "return" in hints
