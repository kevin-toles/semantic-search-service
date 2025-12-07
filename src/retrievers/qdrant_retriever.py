"""
WBS 4.4 GREEN: Qdrant retriever implementation.

LangChain-compatible retriever that wraps QdrantVectorSearch for
vector similarity-based document retrieval.

Design follows Pre-Implementation Analysis (WBS 4.0.1-4.0.4):
- Repository Pattern: Wraps existing QdrantVectorSearch (GUIDELINES line 795)
- Duck typing: Works with any client implementing search() method
- Document objects: Returns LangChain Document with metadata
- LCEL Compatible: Works with LangChain pipe operator

Anti-Patterns Mitigated:
- No new client connections (inject existing client per #12)
- No empty f-strings (use regular strings per #47, #48)
- Full type hints on all public methods (Batch 5)
- Custom exceptions, not builtins (#7, #13)
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from src.retrievers.exceptions import RetrieverConnectionError


class QdrantRetriever(BaseRetriever):
    """LangChain retriever that uses Qdrant for vector similarity search.

    Wraps an existing QdrantVectorSearch instance (injected via dependency injection)
    to perform semantic similarity search for document retrieval.

    Attributes:
        k: Number of documents to return (default: 4)
        score_threshold: Minimum score threshold (default: 0.0)

    Usage:
        # With dependency injection
        retriever = QdrantRetriever(client=qdrant_client, embedder=embedder, k=4)
        docs = retriever.invoke("machine learning")

        # In LCEL chain
        chain = retriever | prompt | llm
        result = chain.invoke({"question": "What is AI?"})
    """

    # Pydantic fields for LangChain BaseRetriever
    k: int = Field(default=4, description="Number of documents to return")
    score_threshold: float = Field(default=0.0, description="Minimum score threshold")

    # Private attributes (not Pydantic fields)
    _client: Any = None
    _embedder: Any = None

    def __init__(
        self,
        client: Any,
        embedder: Any | None = None,
        k: int = 4,
        score_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Initialize retriever with Qdrant client and embedder.

        Args:
            client: Qdrant client instance (QdrantVectorSearch or fake)
            embedder: Embedder for converting text to vectors (optional)
            k: Number of documents to return
            score_threshold: Minimum similarity score to include
            **kwargs: Additional arguments for BaseRetriever
        """
        super().__init__(k=k, score_threshold=score_threshold, **kwargs)
        self._client = client
        self._embedder = embedder

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents synchronously via vector search.

        Args:
            query: Search query string
            run_manager: Optional callback manager for tracing

        Returns:
            List of Document objects with page_content and metadata

        Raises:
            RetrieverConnectionError: If client is disconnected
        """
        # Use run_manager for tracing if provided
        _ = run_manager  # noqa: ARG002 - reserved for future tracing

        # Handle empty query
        if not query or not query.strip():
            return []

        # Check client connection
        if hasattr(self._client, "is_connected") and not self._client.is_connected:
            raise RetrieverConnectionError("Qdrant client is disconnected")

        # Run async query in sync context (Python 3.10+ compatible)
        return asyncio.run(self._aget_relevant_documents(query))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents asynchronously via vector search.

        Args:
            query: Search query string
            run_manager: Optional callback manager for tracing

        Returns:
            List of Document objects with page_content and metadata

        Raises:
            RetrieverConnectionError: If client is disconnected
        """
        # Use run_manager for tracing if provided
        _ = run_manager  # noqa: ARG002 - reserved for future tracing

        # Handle empty query
        if not query or not query.strip():
            return []

        # Check client connection
        if hasattr(self._client, "is_connected") and not self._client.is_connected:
            raise RetrieverConnectionError("Qdrant client is disconnected")

        # Execute search
        # Try text search first (if available), otherwise use embedding
        if hasattr(self._client, "search_by_text"):
            results = await self._client.search_by_text(
                query=query,
                limit=self.k,
                score_threshold=self.score_threshold,
            )
        elif self._embedder is not None:
            # Generate embedding
            embedding = await self._embedder.embed_query(query)
            results = await self._client.search(
                embedding=embedding,
                limit=self.k,
                score_threshold=self.score_threshold,
            )
        else:
            # Fallback: use client search with empty embedding
            results = await self._client.search(
                embedding=[0.0] * 384,  # Placeholder embedding
                limit=self.k,
            )

        # Convert SearchResults to LangChain Documents
        documents: list[Document] = []
        for result in results:
            payload = result.payload or {}

            # Build page_content from title and content
            title = payload.get("title", "")
            content = payload.get("content", "")
            page_content = f"{title}\n\n{content}" if title else content

            # Build metadata
            metadata = {
                "id": result.id,
                "title": title,
                "source": "qdrant",
                "book_id": payload.get("book_id", ""),
                "tier_id": payload.get("tier_id", ""),
                "score": result.score,
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

        # Sort by score descending (should already be sorted, but ensure)
        documents.sort(key=lambda d: d.metadata.get("score", 0), reverse=True)

        return documents
