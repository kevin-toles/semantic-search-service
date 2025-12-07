"""
Custom exceptions for the graph module.

Exception naming follows Comp_Static_Analysis #7, #13:
- Avoid shadowing Python builtins (ConnectionError, TimeoutError)
- Use descriptive names: Neo4jConnectionError, Neo4jQueryError
"""

from __future__ import annotations


class Neo4jError(Exception):
    """Base exception for all Neo4j-related errors."""

    pass


class Neo4jConnectionError(Neo4jError):
    """Raised when connection to Neo4j fails.

    Named Neo4jConnectionError to avoid shadowing Python's
    built-in ConnectionError (Comp_Static_Analysis #7, #13).
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize with message and optional cause.

        Args:
            message: Human-readable error description
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.cause = cause


class Neo4jQueryError(Neo4jError):
    """Raised when a Cypher query fails.

    This includes syntax errors, constraint violations,
    and other query execution failures.
    """

    def __init__(
        self,
        message: str,
        query: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize with message, failed query, and optional cause.

        Args:
            message: Human-readable error description
            query: The Cypher query that failed
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.query = query
        self.cause = cause


class Neo4jTransactionError(Neo4jError):
    """Raised when a transaction fails to commit or rollback."""

    pass
