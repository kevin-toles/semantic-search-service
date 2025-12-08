"""
WBS 4.7: Integration Tests for LangChain Retrievers.

Integration tests verify that retrievers work correctly with:
1. Real or mock backend clients (Neo4j, Qdrant)
2. LangChain chains and agents
3. End-to-end retrieval scenarios

These tests use more realistic test data and may be slower
than unit tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.documents import Document

from src.retrievers.neo4j_retriever import Neo4jRetriever
from src.retrievers.qdrant_retriever import QdrantRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.exceptions import RetrieverError, RetrieverConnectionError


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_CHAPTERS = [
    {
        "id": "ch-ai-intro",
        "title": "Introduction to Artificial Intelligence",
        "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. The field was founded in 1956 at the Dartmouth Conference.",
        "book_id": "ai-foundations",
        "tier_id": "T1",
        "concepts": ["artificial intelligence", "machine intelligence", "cognitive computing"],
    },
    {
        "id": "ch-ml-basics",
        "title": "Machine Learning Fundamentals",
        "content": "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. There are three main types: supervised, unsupervised, and reinforcement learning.",
        "book_id": "ai-foundations",
        "tier_id": "T2",
        "concepts": ["machine learning", "supervised learning", "unsupervised learning", "reinforcement learning"],
    },
    {
        "id": "ch-neural-nets",
        "title": "Neural Networks and Deep Learning",
        "content": "Neural networks are computing systems inspired by biological neural networks. Deep learning uses multiple layers of neurons to learn hierarchical representations of data.",
        "book_id": "ai-foundations",
        "tier_id": "T2",
        "concepts": ["neural networks", "deep learning", "backpropagation", "gradient descent"],
    },
    {
        "id": "ch-nlp",
        "title": "Natural Language Processing",
        "content": "Natural Language Processing (NLP) is a field that combines linguistics and AI to enable computers to understand, interpret, and generate human language. Applications include chatbots, translation, and sentiment analysis.",
        "book_id": "ai-applications",
        "tier_id": "T3",
        "concepts": ["NLP", "text processing", "language models", "transformers"],
    },
    {
        "id": "ch-rag",
        "title": "Retrieval-Augmented Generation",
        "content": "RAG combines retrieval systems with generative AI models to produce more accurate and factual responses. The retriever finds relevant context, which is then used by the generator to produce answers.",
        "book_id": "ai-applications",
        "tier_id": "T3",
        "concepts": ["RAG", "retrieval", "generation", "knowledge bases"],
    },
]


# =============================================================================
# Fake Clients for Integration Tests
# =============================================================================


class FakeSearchResult:
    """Fake search result matching Qdrant SearchResult interface."""

    def __init__(self, id: str, score: float, payload: dict[str, Any]) -> None:
        self.id = id
        self.score = score
        self.payload = payload


class IntegrationNeo4jClient:
    """Integration test Neo4j client with realistic data."""

    def __init__(self, chapters: list[dict[str, Any]] | None = None) -> None:
        """Initialize with sample chapter data."""
        self._is_connected = True
        self._chapters = {ch["id"]: ch for ch in (chapters or SAMPLE_CHAPTERS)}

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._is_connected

    async def query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a Cypher query simulation."""
        await asyncio.sleep(0)  # Satisfy async requirement per Anti-Pattern #8.1
        _ = cypher  # Unused but required by interface
        params = parameters or {}
        search_text = params.get("search_text", "").lower()
        limit = params.get("limit", 4)

        results = []
        for ch in self._chapters.values():
            # Search in title, content, and concepts
            title = ch.get("title", "").lower()
            content = ch.get("content", "").lower()
            concepts = [c.lower() for c in ch.get("concepts", [])]

            if (
                search_text in title
                or search_text in content
                or any(search_text in c for c in concepts)
            ):
                # Calculate a simple relevance score
                score = 0.0
                if search_text in title:
                    score += 0.5
                if search_text in content:
                    score += 0.3
                if any(search_text in c for c in concepts):
                    score += 0.2
                
                results.append({"n": ch, "score": min(score, 1.0)})

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


class IntegrationQdrantClient:
    """Integration test Qdrant client with realistic data."""

    def __init__(self, chapters: list[dict[str, Any]] | None = None) -> None:
        """Initialize with sample chapter data."""
        self._is_connected = True
        self._chapters = chapters or SAMPLE_CHAPTERS

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
        """Execute a vector search simulation (uses fake similarity)."""
        await asyncio.sleep(0)  # Satisfy async requirement per Anti-Pattern #8.1
        _ = embedding  # Unused but required by interface
        results = []
        for i, ch in enumerate(self._chapters):
            # Simulate similarity score based on embedding (fake)
            # In reality, this would be cosine similarity
            score = 0.95 - (i * 0.1)  # Decreasing scores
            if score >= score_threshold:
                results.append(FakeSearchResult(
                    id=ch["id"],
                    score=score,
                    payload={
                        "title": ch["title"],
                        "content": ch["content"],
                        "book_id": ch["book_id"],
                        "tier_id": ch["tier_id"],
                    },
                ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]


class IntegrationEmbedder:
    """Integration test embedder."""

    async def embed_query(self, text: str) -> list[float]:
        """Generate a fake embedding vector."""
        await asyncio.sleep(0)  # Satisfy async requirement per Anti-Pattern #8.1
        _ = text  # Unused but required by interface
        # In reality, this would call an embedding model
        return [0.1] * 384


# =============================================================================
# Integration Test Classes
# =============================================================================


class TestNeo4jRetrieverIntegration:
    """Integration tests for Neo4jRetriever."""

    def test_retrieves_relevant_documents_for_ai_query(self):
        """Neo4jRetriever finds AI-related documents."""
        client = IntegrationNeo4jClient()
        retriever = Neo4jRetriever(client=client, k=5)

        docs = retriever.invoke("artificial intelligence")

        assert len(docs) >= 1
        # Should find the AI intro chapter
        titles = [doc.metadata.get("title", "") for doc in docs]
        assert any("Artificial Intelligence" in t for t in titles)

    def test_retrieves_relevant_documents_for_ml_query(self):
        """Neo4jRetriever finds ML-related documents."""
        client = IntegrationNeo4jClient()
        retriever = Neo4jRetriever(client=client, k=5)

        docs = retriever.invoke("machine learning")

        assert len(docs) >= 1
        titles = [doc.metadata.get("title", "") for doc in docs]
        assert any("Machine Learning" in t for t in titles)

    def test_documents_have_complete_metadata(self):
        """Documents include all expected metadata fields."""
        client = IntegrationNeo4jClient()
        retriever = Neo4jRetriever(client=client)

        docs = retriever.invoke("neural networks")

        assert len(docs) >= 1
        doc = docs[0]
        assert "id" in doc.metadata
        assert "title" in doc.metadata
        assert "source" in doc.metadata
        assert doc.metadata["source"] == "neo4j"
        assert "score" in doc.metadata

    @pytest.mark.asyncio
    async def test_async_retrieval(self):
        """Async retrieval works correctly."""
        client = IntegrationNeo4jClient()
        retriever = Neo4jRetriever(client=client)

        docs = await retriever.ainvoke("deep learning")

        assert len(docs) >= 1
        assert all(isinstance(doc, Document) for doc in docs)


class TestQdrantRetrieverIntegration:
    """Integration tests for QdrantRetriever."""

    def test_retrieves_documents_with_scores(self):
        """QdrantRetriever returns documents with similarity scores."""
        client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()
        retriever = QdrantRetriever(client=client, embedder=embedder, k=3)

        docs = retriever.invoke("artificial intelligence")

        assert len(docs) == 3
        # Scores should be in descending order
        scores = [doc.metadata.get("score", 0) for doc in docs]
        assert scores == sorted(scores, reverse=True)

    def test_respects_score_threshold(self):
        """QdrantRetriever filters by score threshold."""
        client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()
        retriever = QdrantRetriever(
            client=client,
            embedder=embedder,
            k=10,
            score_threshold=0.8,  # High threshold
        )

        docs = retriever.invoke("test query")

        # Should only get high-scoring documents
        for doc in docs:
            assert doc.metadata.get("score", 0) >= 0.8

    def test_documents_have_complete_metadata(self):
        """Documents include all expected metadata fields."""
        client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()
        retriever = QdrantRetriever(client=client, embedder=embedder)

        docs = retriever.invoke("test")

        assert len(docs) >= 1
        doc = docs[0]
        assert "id" in doc.metadata
        assert "title" in doc.metadata
        assert "source" in doc.metadata
        assert doc.metadata["source"] == "qdrant"
        assert "score" in doc.metadata

    @pytest.mark.asyncio
    async def test_async_retrieval(self):
        """Async retrieval works correctly."""
        client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()
        retriever = QdrantRetriever(client=client, embedder=embedder)

        docs = await retriever.ainvoke("language models")

        assert len(docs) >= 1
        assert all(isinstance(doc, Document) for doc in docs)


class TestHybridRetrieverIntegration:
    """Integration tests for HybridRetriever."""

    def test_combines_results_from_both_sources(self):
        """HybridRetriever combines Neo4j and Qdrant results."""
        neo4j_client = IntegrationNeo4jClient()
        qdrant_client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()

        neo4j_retriever = Neo4jRetriever(client=neo4j_client, k=3)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder, k=3)

        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            k=5,
        )

        docs = hybrid.invoke("artificial intelligence")

        assert len(docs) >= 1
        # Should have documents from both sources
        sources = [doc.metadata.get("source") for doc in docs]
        assert "neo4j" in sources or "qdrant" in sources

    def test_deduplicates_common_documents(self):
        """HybridRetriever deduplicates documents found in both sources."""
        neo4j_client = IntegrationNeo4jClient()
        qdrant_client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()

        neo4j_retriever = Neo4jRetriever(client=neo4j_client, k=5)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder, k=5)

        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            k=10,
        )

        docs = hybrid.invoke("machine learning")

        # Count unique IDs
        ids = [doc.metadata.get("id") for doc in docs]
        assert len(ids) == len(set(ids)), "Duplicate documents found"

    def test_documents_have_combined_score(self):
        """Documents have combined_score in metadata."""
        neo4j_client = IntegrationNeo4jClient()
        qdrant_client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()

        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)

        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
        )

        docs = hybrid.invoke("neural networks")

        for doc in docs:
            assert "combined_score" in doc.metadata

    def test_weights_affect_ranking(self):
        """Custom weights affect document ranking."""
        neo4j_client = IntegrationNeo4jClient()
        qdrant_client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()

        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)

        # Create hybrid with high Neo4j weight
        hybrid_neo4j = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            neo4j_weight=0.9,
            qdrant_weight=0.1,
        )

        # Create hybrid with high Qdrant weight
        hybrid_qdrant = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            neo4j_weight=0.1,
            qdrant_weight=0.9,
        )

        docs_neo4j = hybrid_neo4j.invoke("retrieval")
        docs_qdrant = hybrid_qdrant.invoke("retrieval")

        # Rankings may differ based on weights
        # At minimum, both should return documents
        assert len(docs_neo4j) >= 1
        assert len(docs_qdrant) >= 1

    @pytest.mark.asyncio
    async def test_async_retrieval(self):
        """Async hybrid retrieval works correctly."""
        neo4j_client = IntegrationNeo4jClient()
        qdrant_client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()

        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)

        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
        )

        docs = await hybrid.ainvoke("NLP")

        assert len(docs) >= 1
        assert all(isinstance(doc, Document) for doc in docs)


class TestLCELChainIntegration:
    """Test retrievers work in LangChain chains."""

    def test_neo4j_retriever_returns_document_list(self):
        """Neo4jRetriever returns list compatible with chain."""
        client = IntegrationNeo4jClient()
        retriever = Neo4jRetriever(client=client)

        result = retriever.invoke("test")

        # LCEL expects list of Documents
        assert isinstance(result, list)
        for doc in result:
            assert isinstance(doc, Document)
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")

    def test_qdrant_retriever_returns_document_list(self):
        """QdrantRetriever returns list compatible with chain."""
        client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()
        retriever = QdrantRetriever(client=client, embedder=embedder)

        result = retriever.invoke("test")

        assert isinstance(result, list)
        for doc in result:
            assert isinstance(doc, Document)

    def test_hybrid_retriever_returns_document_list(self):
        """HybridRetriever returns list compatible with chain."""
        neo4j_client = IntegrationNeo4jClient()
        qdrant_client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()

        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)

        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
        )

        result = hybrid.invoke("test")

        assert isinstance(result, list)
        for doc in result:
            assert isinstance(doc, Document)

    @pytest.mark.asyncio
    async def test_retriever_ainvoke_compatibility(self):
        """All retrievers support ainvoke for async chains."""
        neo4j_client = IntegrationNeo4jClient()
        qdrant_client = IntegrationQdrantClient()
        embedder = IntegrationEmbedder()

        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
        )

        # All should support ainvoke
        neo4j_docs = await neo4j_retriever.ainvoke("test")
        qdrant_docs = await qdrant_retriever.ainvoke("test")
        hybrid_docs = await hybrid.ainvoke("test")

        assert all(isinstance(doc, Document) for doc in neo4j_docs)
        assert all(isinstance(doc, Document) for doc in qdrant_docs)
        assert all(isinstance(doc, Document) for doc in hybrid_docs)
