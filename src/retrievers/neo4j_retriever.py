"""
WBS 4.2 GREEN: Neo4j retriever implementation.

LangChain-compatible retriever that wraps Neo4jClient for graph-based
document retrieval.

Design follows Pre-Implementation Analysis (WBS 4.0.1-4.0.4):
- Repository Pattern: Wraps existing Neo4jClient (GUIDELINES line 795)
- Duck typing: Works with any client implementing query() method
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


class Neo4jRetriever(BaseRetriever):
    """LangChain retriever that uses Neo4j for graph-based retrieval.

    Wraps an existing Neo4jClient instance (injected via dependency injection)
    to perform full-text search and graph traversal for document retrieval.

    Attributes:
        k: Number of documents to return (default: 4)

    Usage:
        # With dependency injection
        retriever = Neo4jRetriever(client=neo4j_client, k=4)
        docs = retriever.invoke("machine learning")

        # In LCEL chain
        chain = retriever | prompt | llm
        result = chain.invoke({"question": "What is AI?"})
    """

    # Pydantic fields for LangChain BaseRetriever
    k: int = Field(default=4, description="Number of documents to return")

    # Private attributes (not Pydantic fields)
    _client: Any = None

    def __init__(self, client: Any, k: int = 4, **kwargs: Any) -> None:
        """Initialize retriever with Neo4j client.

        Args:
            client: Neo4j client instance (Neo4jClient or FakeNeo4jClient)
            k: Number of documents to return
            **kwargs: Additional arguments for BaseRetriever
        """
        super().__init__(k=k, **kwargs)
        self._client = client

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents synchronously.

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
            raise RetrieverConnectionError("Neo4j client is disconnected")

        # Run async query in sync context (Python 3.10+ compatible)
        return asyncio.run(self._aget_relevant_documents(query))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents asynchronously.

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
            raise RetrieverConnectionError("Neo4j client is disconnected")

        # Execute search query
        cypher = """
        MATCH (n)
        WHERE n.title CONTAINS $search_text
           OR n.content CONTAINS $search_text
           OR ANY(c IN n.concepts WHERE c CONTAINS $search_text)
        RETURN n, 0.8 as score
        LIMIT $limit
        """

        results = await self._client.query(
            cypher,
            parameters={"search_text": query, "limit": self.k},
        )

        # Convert results to Document objects
        documents: list[Document] = []
        for result in results[: self.k]:
            node = result.get("n", {})
            score = result.get("score", 0.5)

            # Build page_content from title and content
            title = node.get("title", "")
            content = node.get("content", "")
            page_content = f"{title}\n\n{content}" if title else content

            # Build metadata
            metadata = {
                "id": node.get("id", ""),
                "title": title,
                "source": "neo4j",
                "book_id": node.get("book_id", ""),
                "tier_id": node.get("tier_id", ""),
                "score": score,
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents
