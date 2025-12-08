"""
WBS 4.3 RED: Unit tests for QdrantRetriever.

Tests follow TDD methodology - these tests should FAIL initially
until WBS 4.4 GREEN implementation is complete.

Design follows Pre-Implementation Analysis (WBS 4.0.1-4.0.4):
- Repository Pattern: FakeClient for unit testing (GUIDELINES line 276)
- Duck typing: Tests work with FakeQdrantClient (GUIDELINES line 795)
- Document objects: Return LangChain Document with metadata
- Async support: Tests use pytest-asyncio

Anti-Patterns Mitigated:
- No new client connections (inject existing client)
- No empty f-strings (use regular strings)
- Full type hints on test functions
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.documents import Document

from src.retrievers.qdrant_retriever import QdrantRetriever
from src.search.vector import SearchResult


# =============================================================================
# Test Fixtures
# =============================================================================


class FakeQdrantClientForRetriever:
    """Fake Qdrant client for retriever unit tests.

    Simulates Qdrant search results without real database connection.
    Implements the same interface as QdrantVectorSearch via duck typing.
    """

    def __init__(self) -> None:
        """Initialize fake client with sample data."""
        self._is_connected = True
        self._documents: list[dict[str, Any]] = [
            {
                "id": "doc-1",
                "content": "Machine learning is a branch of artificial intelligence that enables systems to learn from data.",
                "title": "Introduction to Machine Learning",
                "book_id": "book-ml-101",
                "tier_id": "T1",
            },
            {
                "id": "doc-2",
                "content": "Neural networks are computing systems inspired by biological neural networks in animal brains.",
                "title": "Neural Network Fundamentals",
                "book_id": "book-nn-201",
                "tier_id": "T2",
            },
            {
                "id": "doc-3",
                "content": "Deep learning uses multi-layered neural networks to progressively extract features from raw input.",
                "title": "Deep Learning Architecture",
                "book_id": "book-dl-301",
                "tier_id": "T2",
            },
            {
                "id": "doc-4",
                "content": "Transformers are a type of deep learning model that uses self-attention mechanisms.",
                "title": "Transformer Models",
                "book_id": "book-tf-401",
                "tier_id": "T3",
            },
            {
                "id": "doc-5",
                "content": "Natural language processing enables computers to understand and generate human language.",
                "title": "NLP Overview",
                "book_id": "book-nlp-501",
                "tier_id": "T1",
            },
        ]

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._is_connected

    async def search(
        self,
        embedding: list[float],
        limit: int | None = None,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Execute fake vector search.

        In a real implementation, this would compute cosine similarity.
        Here we simulate results based on simple keyword matching.
        """
        # Simulate async behavior
        await asyncio.sleep(0)

        # Consume unused parameters per Anti-Pattern #51, #52
        _ = (embedding, filter_conditions, score_threshold)

        # Return simulated results (all documents with fake scores)
        results = []
        for i, doc in enumerate(self._documents):
            # Simulate decreasing relevance scores
            score = max(0.3, 0.95 - (i * 0.15))
            results.append(
                SearchResult(
                    id=doc["id"],
                    score=score,
                    payload=doc,
                )
            )

        # Apply limit
        limit = limit or 4
        return results[:limit]

    async def search_by_text(
        self,
        query: str,
        limit: int | None = None,
        filter_conditions: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Execute fake text search (uses embedder internally).

        Simulates search by matching query terms in content.
        """
        await asyncio.sleep(0)

        # Handle empty query
        if not query or not query.strip():
            return []

        query_lower = query.lower()
        results = []

        for doc in self._documents:
            content = doc.get("content", "").lower()
            title = doc.get("title", "").lower()

            # Simple keyword matching for simulation
            if query_lower in content or query_lower in title:
                score = 0.85 if query_lower in title else 0.7
                results.append(
                    SearchResult(
                        id=doc["id"],
                        score=score,
                        payload=doc,
                    )
                )

        # If no exact match, return some results anyway for testing
        if not results:
            for i, doc in enumerate(self._documents[:2]):
                results.append(
                    SearchResult(
                        id=doc["id"],
                        score=0.3 - (i * 0.05),
                        payload=doc,
                    )
                )

        # Apply limit and threshold
        limit = limit or 4
        threshold = score_threshold or 0.0
        results = [r for r in results if r.score >= threshold]

        return results[:limit]

    async def close(self) -> None:
        """Close the fake client."""
        await asyncio.sleep(0)  # Satisfy async requirement per Anti-Pattern #8.1
        self._is_connected = False


class FakeEmbedder:
    """Fake embedder for testing."""

    async def embed_query(self, text: str) -> list[float]:
        """Return fake embedding vector."""
        await asyncio.sleep(0)
        _ = text  # noqa: ARG002
        # Return a simple fake embedding
        return [0.1] * 384


@pytest.fixture
def fake_qdrant_client() -> FakeQdrantClientForRetriever:
    """Provide fake Qdrant client for tests."""
    return FakeQdrantClientForRetriever()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Provide fake embedder for tests."""
    return FakeEmbedder()


@pytest.fixture
def qdrant_retriever(
    fake_qdrant_client: FakeQdrantClientForRetriever,
    fake_embedder: FakeEmbedder,
) -> QdrantRetriever:
    """Provide Qdrant retriever with fake client and embedder."""
    return QdrantRetriever(client=fake_qdrant_client, embedder=fake_embedder)


# =============================================================================
# Test Cases: Initialization
# =============================================================================


class TestQdrantRetrieverInit:
    """Test QdrantRetriever initialization."""

    def test_init_with_client_and_embedder(
        self,
        fake_qdrant_client: FakeQdrantClientForRetriever,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """Test initialization with injected client and embedder."""
        retriever = QdrantRetriever(client=fake_qdrant_client, embedder=fake_embedder)
        assert retriever._client is fake_qdrant_client
        assert retriever._embedder is fake_embedder

    def test_init_with_custom_limit(
        self,
        fake_qdrant_client: FakeQdrantClientForRetriever,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """Test initialization with custom result limit."""
        retriever = QdrantRetriever(
            client=fake_qdrant_client, embedder=fake_embedder, k=10
        )
        assert retriever.k == 10

    def test_init_default_limit(
        self,
        fake_qdrant_client: FakeQdrantClientForRetriever,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """Test default result limit is 4."""
        retriever = QdrantRetriever(client=fake_qdrant_client, embedder=fake_embedder)
        assert retriever.k == 4

    def test_init_with_score_threshold(
        self,
        fake_qdrant_client: FakeQdrantClientForRetriever,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """Test initialization with score threshold."""
        retriever = QdrantRetriever(
            client=fake_qdrant_client, embedder=fake_embedder, score_threshold=0.5
        )
        assert retriever.score_threshold == pytest.approx(0.5)


# =============================================================================
# Test Cases: Document Retrieval
# =============================================================================


class TestQdrantRetrieverGetRelevantDocuments:
    """Test _get_relevant_documents method (sync)."""

    def test_retrieves_documents_for_query(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test retrieval returns Document objects."""
        docs = qdrant_retriever._get_relevant_documents("machine learning")

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(doc, Document) for doc in docs)

    def test_document_has_page_content(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test documents have page_content populated."""
        docs = qdrant_retriever._get_relevant_documents("neural networks")

        assert len(docs) > 0
        for doc in docs:
            assert doc.page_content is not None
            assert len(doc.page_content) > 0

    def test_document_has_metadata(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test documents have metadata with source info."""
        docs = qdrant_retriever._get_relevant_documents("deep learning")

        assert len(docs) > 0
        for doc in docs:
            assert "id" in doc.metadata
            assert "source" in doc.metadata
            assert doc.metadata["source"] == "qdrant"

    def test_empty_query_returns_empty_list(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test empty query returns empty list, not None."""
        docs = qdrant_retriever._get_relevant_documents("")

        assert isinstance(docs, list)
        assert len(docs) == 0

    def test_respects_k_limit(
        self,
        fake_qdrant_client: FakeQdrantClientForRetriever,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """Test result count respects k limit."""
        retriever = QdrantRetriever(
            client=fake_qdrant_client, embedder=fake_embedder, k=2
        )
        docs = retriever._get_relevant_documents("learning")

        assert len(docs) <= 2


# =============================================================================
# Test Cases: Async Document Retrieval
# =============================================================================


class TestQdrantRetrieverAsyncGetRelevantDocuments:
    """Test _aget_relevant_documents method (async)."""

    @pytest.mark.asyncio
    async def test_async_retrieves_documents(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test async retrieval returns Document objects."""
        docs = await qdrant_retriever._aget_relevant_documents("transformers")

        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(doc, Document) for doc in docs)

    @pytest.mark.asyncio
    async def test_async_document_has_content(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test async documents have page_content."""
        docs = await qdrant_retriever._aget_relevant_documents("natural language")

        for doc in docs:
            assert doc.page_content is not None

    @pytest.mark.asyncio
    async def test_async_document_has_metadata(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test async documents have metadata."""
        docs = await qdrant_retriever._aget_relevant_documents("NLP")

        for doc in docs:
            assert "id" in doc.metadata
            assert "source" in doc.metadata


# =============================================================================
# Test Cases: LCEL Compatibility
# =============================================================================


class TestQdrantRetrieverLCELCompat:
    """Test LCEL (LangChain Expression Language) compatibility."""

    def test_invoke_returns_documents(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test .invoke() method works (LCEL interface)."""
        docs = qdrant_retriever.invoke("AI")

        assert isinstance(docs, list)
        assert all(isinstance(doc, Document) for doc in docs)

    @pytest.mark.asyncio
    async def test_ainvoke_returns_documents(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test .ainvoke() method works (async LCEL)."""
        docs = await qdrant_retriever.ainvoke("machine learning")

        assert isinstance(docs, list)
        assert all(isinstance(doc, Document) for doc in docs)


# =============================================================================
# Test Cases: Vector-Specific Features
# =============================================================================


class TestQdrantRetrieverVectorFeatures:
    """Test vector search-specific retrieval features."""

    def test_metadata_includes_score(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test metadata includes similarity score."""
        docs = qdrant_retriever._get_relevant_documents("neural networks")

        for doc in docs:
            assert "score" in doc.metadata
            assert 0.0 <= doc.metadata["score"] <= 1.0

    def test_metadata_includes_title(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test metadata includes document title."""
        docs = qdrant_retriever._get_relevant_documents("deep learning")

        for doc in docs:
            assert "title" in doc.metadata

    def test_results_sorted_by_score(
        self, qdrant_retriever: QdrantRetriever
    ) -> None:
        """Test results are sorted by descending score."""
        docs = qdrant_retriever._get_relevant_documents("learning")

        if len(docs) > 1:
            scores = [doc.metadata.get("score", 0) for doc in docs]
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# Test Cases: Error Handling
# =============================================================================


class TestQdrantRetrieverErrorHandling:
    """Test error handling in retriever."""

    def test_disconnected_client_raises_error(
        self,
        fake_qdrant_client: FakeQdrantClientForRetriever,
        fake_embedder: FakeEmbedder,
    ) -> None:
        """Test behavior when client is disconnected."""
        # Simulate disconnection
        fake_qdrant_client._is_connected = False

        retriever = QdrantRetriever(client=fake_qdrant_client, embedder=fake_embedder)

        with pytest.raises(Exception):
            retriever._get_relevant_documents("test")
