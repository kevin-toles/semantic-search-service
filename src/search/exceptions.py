"""
Custom exceptions for the search module.

Exception naming follows Comp_Static_Analysis #7, #13:
- Avoid shadowing Python builtins (ConnectionError, TimeoutError)
- Use descriptive names: QdrantConnectionError, QdrantSearchError
"""

from __future__ import annotations


class QdrantError(Exception):
    """Base exception for all Qdrant-related errors."""

    pass


class QdrantConnectionError(QdrantError):
    """Raised when connection to Qdrant fails.

    Named QdrantConnectionError to avoid shadowing Python's
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


class QdrantSearchError(QdrantError):
    """Raised when a search operation fails.

    This includes empty queries, invalid embeddings,
    and other search execution failures.
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
            query: The search query that failed
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.query = query
        self.cause = cause


class ScoreFusionError(QdrantError):
    """Raised when score fusion calculation fails."""

    pass
