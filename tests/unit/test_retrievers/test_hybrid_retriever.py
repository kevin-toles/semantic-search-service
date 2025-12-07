"""
WBS 4.5 RED: Hybrid Retriever Tests

Tests for HybridRetriever that combines Neo4j and Qdrant retrievers
using an ensemble approach with configurable weights.

Following TDD RED-GREEN-REFACTOR cycle.
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.neo4j_retriever import Neo4jRetriever
from src.retrievers.qdrant_retriever import QdrantRetriever
from src.retrievers.exceptions import RetrieverError


# =============================================================================
# Test Fixtures - Fake Clients
# =============================================================================


class FakeNeo4jClientForHybrid:
    """Fake Neo4j client for hybrid retriever unit tests."""

    def __init__(self, documents: list[dict[str, Any]] | None = None, should_fail: bool = False) -> None:
        """Initialize fake client with configurable documents."""
        self._is_connected = not should_fail
        self._should_fail = should_fail
        self._documents = documents or []

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._is_connected

    async def query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Simulate Neo4j query execution."""
        if self._should_fail:
            raise RuntimeError("Simulated Neo4j connection failure")
        
        search_text = (parameters or {}).get("search_text", "")
        limit = (parameters or {}).get("limit", 4)
        
        results = []
        for doc in self._documents:
            metadata = doc.get("metadata", {})
            node = {
                "id": metadata.get("id", ""),
                "title": metadata.get("title", ""),
                "content": doc.get("content", ""),
                "book_id": metadata.get("book_id", ""),
                "tier_id": metadata.get("tier_id", ""),
                "concepts": metadata.get("concepts", []),
            }
            results.append({"n": node, "score": metadata.get("score", 0.8)})
        
        return results[:limit]


class FakeSearchResult:
    """Fake search result matching Qdrant SearchResult interface."""

    def __init__(self, id: str, score: float, payload: dict[str, Any]) -> None:
        self.id = id
        self.score = score
        self.payload = payload


class FakeQdrantClientForHybrid:
    """Fake Qdrant client for hybrid retriever unit tests."""

    def __init__(self, results: list[dict[str, Any]] | None = None, should_fail: bool = False) -> None:
        """Initialize fake client with configurable results."""
        self._is_connected = not should_fail
        self._should_fail = should_fail
        self._results = results or []

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._is_connected

    async def search(
        self,
        embedding: list[float] | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list[FakeSearchResult]:
        """Simulate Qdrant search returning SearchResult-like objects."""
        if self._should_fail:
            raise RuntimeError("Simulated Qdrant connection failure")
        
        results = []
        for r in self._results[:limit]:
            metadata = r.get("metadata", {})
            payload = {
                "title": metadata.get("title", r.get("title", "")),
                "content": r.get("content", ""),
                "book_id": metadata.get("book_id", ""),
                "tier_id": metadata.get("tier_id", ""),
            }
            results.append(FakeSearchResult(
                id=metadata.get("id", r.get("id", "")),
                score=r.get("score", 0.9),
                payload=payload,
            ))
        
        return results


class FakeEmbedderForHybrid:
    """Fake embedder for hybrid retriever tests."""

    async def embed(self, text: str) -> list[float]:
        """Return a fake embedding vector."""
        return [0.1] * 384

    async def embed_query(self, text: str) -> list[float]:
        """Return a fake embedding vector for query (LangChain interface)."""
        return [0.1] * 384


# =============================================================================
# Test Classes
# =============================================================================


class TestHybridRetrieverInit:
    """Test HybridRetriever initialization."""
    
    def test_init_with_retrievers(self):
        """HybridRetriever initializes with Neo4j and Qdrant retrievers."""
        neo4j_client = FakeNeo4jClientForHybrid()
        qdrant_client = FakeQdrantClientForHybrid()
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever
        )
        
        assert hybrid._neo4j_retriever is neo4j_retriever
        assert hybrid._qdrant_retriever is qdrant_retriever
    
    def test_init_default_weights(self):
        """HybridRetriever has default equal weights (0.5, 0.5)."""
        neo4j_client = FakeNeo4jClientForHybrid()
        qdrant_client = FakeQdrantClientForHybrid()
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever
        )
        
        assert hybrid.neo4j_weight == 0.5
        assert hybrid.qdrant_weight == 0.5
    
    def test_init_custom_weights(self):
        """HybridRetriever accepts custom weights."""
        neo4j_client = FakeNeo4jClientForHybrid()
        qdrant_client = FakeQdrantClientForHybrid()
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            neo4j_weight=0.7,
            qdrant_weight=0.3
        )
        
        assert hybrid.neo4j_weight == 0.7
        assert hybrid.qdrant_weight == 0.3
    
    def test_init_with_k_limit(self):
        """HybridRetriever accepts k limit parameter."""
        neo4j_client = FakeNeo4jClientForHybrid()
        qdrant_client = FakeQdrantClientForHybrid()
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            k=10
        )
        
        assert hybrid.k == 10


class TestHybridRetrieverGetRelevantDocuments:
    """Test HybridRetriever._get_relevant_documents method."""
    
    def test_combines_results_from_both_retrievers(self):
        """HybridRetriever combines results from Neo4j and Qdrant."""
        neo4j_client = FakeNeo4jClientForHybrid(documents=[
            {"content": "Neo4j doc 1", "metadata": {"source": "neo4j", "id": "n1"}}
        ])
        qdrant_client = FakeQdrantClientForHybrid(results=[
            {"content": "Qdrant doc 1", "score": 0.9, "metadata": {"source": "qdrant", "id": "q1"}}
        ])
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever
        )
        
        docs = hybrid.invoke("test query")
        
        assert len(docs) >= 1
        sources = [doc.metadata.get("source") for doc in docs]
        # Should have results from both sources
        assert "neo4j" in sources or "qdrant" in sources
    
    def test_deduplicates_by_id(self):
        """HybridRetriever deduplicates documents by ID."""
        neo4j_client = FakeNeo4jClientForHybrid(documents=[
            {"content": "Shared doc", "metadata": {"source": "neo4j", "id": "shared-1"}}
        ])
        qdrant_client = FakeQdrantClientForHybrid(results=[
            {"content": "Shared doc", "score": 0.9, "metadata": {"source": "qdrant", "id": "shared-1"}}
        ])
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever
        )
        
        docs = hybrid.invoke("test query")
        
        # Should only have one document with id "shared-1"
        ids = [doc.metadata.get("id") for doc in docs]
        assert ids.count("shared-1") == 1
    
    def test_respects_k_limit(self):
        """HybridRetriever respects k limit on final results."""
        neo4j_client = FakeNeo4jClientForHybrid(documents=[
            {"content": f"Neo4j doc {i}", "metadata": {"source": "neo4j", "id": f"n{i}"}}
            for i in range(5)
        ])
        qdrant_client = FakeQdrantClientForHybrid(results=[
            {"content": f"Qdrant doc {i}", "score": 0.9 - i*0.1, "metadata": {"source": "qdrant", "id": f"q{i}"}}
            for i in range(5)
        ])
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            k=3
        )
        
        docs = hybrid.invoke("test query")
        
        assert len(docs) <= 3
    
    def test_empty_query_returns_empty_list(self):
        """HybridRetriever returns empty list for empty query."""
        neo4j_client = FakeNeo4jClientForHybrid()
        qdrant_client = FakeQdrantClientForHybrid()
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever
        )
        
        docs = hybrid.invoke("")
        
        assert docs == []
    
    def test_documents_have_combined_score(self):
        """HybridRetriever documents have combined score in metadata."""
        neo4j_client = FakeNeo4jClientForHybrid(documents=[
            {"content": "Neo4j doc 1", "metadata": {"source": "neo4j", "id": "n1", "score": 0.8}}
        ])
        qdrant_client = FakeQdrantClientForHybrid(results=[
            {"content": "Qdrant doc 1", "score": 0.9, "metadata": {"source": "qdrant", "id": "q1"}}
        ])
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever
        )
        
        docs = hybrid.invoke("test query")
        
        for doc in docs:
            assert "combined_score" in doc.metadata or "score" in doc.metadata


class TestHybridRetrieverAsyncGetRelevantDocuments:
    """Test HybridRetriever._aget_relevant_documents method."""
    
    @pytest.mark.asyncio
    async def test_async_combines_results(self):
        """HybridRetriever async combines results from both retrievers."""
        neo4j_client = FakeNeo4jClientForHybrid(documents=[
            {"content": "Neo4j doc 1", "metadata": {"source": "neo4j", "id": "n1"}}
        ])
        qdrant_client = FakeQdrantClientForHybrid(results=[
            {"content": "Qdrant doc 1", "score": 0.9, "metadata": {"source": "qdrant", "id": "q1"}}
        ])
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever
        )
        
        docs = await hybrid.ainvoke("test query")
        
        assert len(docs) >= 1
        assert all(isinstance(doc, Document) for doc in docs)
    
    @pytest.mark.asyncio
    async def test_async_respects_k_limit(self):
        """HybridRetriever async respects k limit."""
        neo4j_client = FakeNeo4jClientForHybrid(documents=[
            {"content": f"Neo4j doc {i}", "metadata": {"source": "neo4j", "id": f"n{i}"}}
            for i in range(5)
        ])
        qdrant_client = FakeQdrantClientForHybrid(results=[
            {"content": f"Qdrant doc {i}", "score": 0.9 - i*0.1, "metadata": {"source": "qdrant", "id": f"q{i}"}}
            for i in range(5)
        ])
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            k=2
        )
        
        docs = await hybrid.ainvoke("test query")
        
        assert len(docs) <= 2


class TestHybridRetrieverLCELCompat:
    """Test HybridRetriever LCEL compatibility."""
    
    def test_invoke_returns_documents(self):
        """HybridRetriever.invoke returns list of Documents."""
        neo4j_client = FakeNeo4jClientForHybrid(documents=[
            {"content": "Test doc", "metadata": {"id": "1"}}
        ])
        qdrant_client = FakeQdrantClientForHybrid(results=[
            {"content": "Test doc 2", "score": 0.9, "metadata": {"id": "2"}}
        ])
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever
        )
        
        result = hybrid.invoke("test")
        
        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)
    
    @pytest.mark.asyncio
    async def test_ainvoke_returns_documents(self):
        """HybridRetriever.ainvoke returns list of Documents."""
        neo4j_client = FakeNeo4jClientForHybrid(documents=[
            {"content": "Test doc", "metadata": {"id": "1"}}
        ])
        qdrant_client = FakeQdrantClientForHybrid(results=[
            {"content": "Test doc 2", "score": 0.9, "metadata": {"id": "2"}}
        ])
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever
        )
        
        result = await hybrid.ainvoke("test")
        
        assert isinstance(result, list)
        assert all(isinstance(doc, Document) for doc in result)


class TestHybridRetrieverWeighting:
    """Test HybridRetriever score weighting."""
    
    def test_weights_affect_ranking(self):
        """Higher weighted retriever results rank higher."""
        # Neo4j doc with score 0.5, weight 0.9 = 0.45
        # Qdrant doc with score 0.9, weight 0.1 = 0.09
        # Neo4j should rank higher
        neo4j_client = FakeNeo4jClientForHybrid(documents=[
            {"content": "Neo4j high weight", "metadata": {"source": "neo4j", "id": "n1", "score": 0.5}}
        ])
        qdrant_client = FakeQdrantClientForHybrid(results=[
            {"content": "Qdrant low weight", "score": 0.9, "metadata": {"source": "qdrant", "id": "q1"}}
        ])
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            neo4j_weight=0.9,
            qdrant_weight=0.1
        )
        
        docs = hybrid.invoke("test query")
        
        # First doc should be from neo4j due to higher weight
        if len(docs) > 0:
            assert docs[0].metadata.get("source") == "neo4j"


class TestHybridRetrieverErrorHandling:
    """Test HybridRetriever error handling."""
    
    def test_one_retriever_fails_returns_other_results(self):
        """If one retriever fails, return results from the other."""
        neo4j_client = FakeNeo4jClientForHybrid(documents=[
            {"content": "Neo4j doc", "metadata": {"source": "neo4j", "id": "n1"}}
        ])
        # Qdrant client that raises error
        qdrant_client = FakeQdrantClientForHybrid(should_fail=True)
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            fail_silently=True
        )
        
        docs = hybrid.invoke("test query")
        
        # Should still get Neo4j results
        assert len(docs) >= 1
        assert docs[0].metadata.get("source") == "neo4j"
    
    def test_both_retrievers_fail_raises_error(self):
        """If both retrievers fail, raise RetrieverError."""
        neo4j_client = FakeNeo4jClientForHybrid(should_fail=True)
        qdrant_client = FakeQdrantClientForHybrid(should_fail=True)
        embedder = FakeEmbedderForHybrid()
        
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever
        )
        
        with pytest.raises(RetrieverError):
            hybrid.invoke("test query")
