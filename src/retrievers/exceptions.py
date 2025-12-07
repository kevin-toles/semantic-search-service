"""
Retriever exceptions module.

Custom exceptions for retrievers to avoid shadowing Python builtins
(Anti-Pattern #7, #13 from Comp_Static_Analysis_Report).
"""

from __future__ import annotations


class RetrieverError(Exception):
    """Base exception for all retriever errors."""

    pass


class RetrieverConnectionError(RetrieverError):
    """Raised when the underlying client is disconnected."""

    pass


class RetrieverQueryError(RetrieverError):
    """Raised when a query execution fails."""

    pass


class DocumentNotFoundError(RetrieverError):
    """Raised when expected documents are not found."""

    pass


class EmbedderError(RetrieverError):
    """Raised when embedding generation fails."""

    pass
