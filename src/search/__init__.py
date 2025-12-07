"""
Search module for semantic-search-service.

Provides vector search (Qdrant), hybrid search (vector + graph),
and result ranking with configurable score fusion.

Phase 3 Implementation (TDD):
- vector.py: Qdrant vector search client
- hybrid.py: Combined vector + graph search
- ranker.py: Score fusion logic
"""

from __future__ import annotations

from src.search.hybrid import HybridSearchResult, HybridSearchService
from src.search.ranker import ResultRanker
from src.search.vector import FakeQdrantSearchClient, QdrantSearchClient, SearchResult

__all__ = [
    "QdrantSearchClient",
    "FakeQdrantSearchClient",
    "HybridSearchService",
    "ResultRanker",
    "SearchResult",
    "HybridSearchResult",
]
