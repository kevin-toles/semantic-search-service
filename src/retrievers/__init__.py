"""
LangChain-compatible retrievers for semantic-search-service.

This module provides retrievers that wrap the existing search components
(Neo4j, Qdrant, Hybrid) with LangChain's BaseRetriever interface.

Design follows:
- Repository Pattern: Retrievers as adapters over existing clients (GUIDELINES line 795)
- Duck Typing: Protocol-based interface for testing (GUIDELINES line 276)
- LCEL Compatible: Works with LangChain pipe operator (GUIDELINES line 103)

Modules:
- neo4j_retriever: Graph-based retrieval via Neo4j traversal
- qdrant_retriever: Vector similarity search via Qdrant
- hybrid_retriever: Ensemble retriever combining both
- exceptions: Custom exceptions for retriever errors
"""

from src.retrievers.neo4j_retriever import Neo4jRetriever
from src.retrievers.qdrant_retriever import QdrantRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.exceptions import (
    RetrieverError,
    RetrieverConnectionError,
    DocumentNotFoundError,
    EmbedderError,
)

__all__ = [
    "Neo4jRetriever",
    "QdrantRetriever",
    "HybridRetriever",
    "RetrieverError",
    "RetrieverConnectionError",
    "DocumentNotFoundError",
    "EmbedderError",
]
