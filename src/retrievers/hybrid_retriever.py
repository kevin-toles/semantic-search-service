"""
WBS 4.6 GREEN: Hybrid retriever implementation.

LangChain-compatible ensemble retriever that combines Neo4j and Qdrant
retrievers using configurable score fusion.

Design follows Pre-Implementation Analysis (WBS 4.0.1-4.0.4):
- Repository Pattern: Wraps existing retrievers (GUIDELINES line 795)
- Reciprocal Rank Fusion (RRF) or weighted scoring
- Document objects: Returns LangChain Document with metadata
- LCEL Compatible: Works with LangChain pipe operator

Anti-Patterns Mitigated:
- No new client connections (inject existing retrievers per #12)
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

from src.retrievers.exceptions import RetrieverError


class HybridRetriever(BaseRetriever):
    """LangChain ensemble retriever combining vector + graph search.

    Combines results from Neo4j (graph) and Qdrant (vector) retrievers
    using configurable weights and score fusion.

    Attributes:
        k: Number of documents to return (default: 4)
        neo4j_weight: Weight for Neo4j graph results (default: 0.5)
        qdrant_weight: Weight for Qdrant vector results (default: 0.5)
        fail_silently: If True, continue if one retriever fails (default: False)

    Usage:
        # Create component retrievers
        neo4j_retriever = Neo4jRetriever(client=neo4j_client)
        qdrant_retriever = QdrantRetriever(client=qdrant_client, embedder=embedder)

        # Create hybrid retriever
        hybrid = HybridRetriever(
            neo4j_retriever=neo4j_retriever,
            qdrant_retriever=qdrant_retriever,
            neo4j_weight=0.3,
            qdrant_weight=0.7,
        )
        docs = hybrid.invoke("machine learning")

        # In LCEL chain
        chain = hybrid | prompt | llm
        result = chain.invoke({"question": "What is AI?"})
    """

    # Pydantic fields for LangChain BaseRetriever
    k: int = Field(default=4, description="Number of documents to return")
    neo4j_weight: float = Field(default=0.5, description="Weight for Neo4j graph results")
    qdrant_weight: float = Field(default=0.5, description="Weight for Qdrant vector results")
    fail_silently: bool = Field(default=False, description="Continue if one retriever fails")

    # Private attributes (not Pydantic fields)
    _neo4j_retriever: Any = None
    _qdrant_retriever: Any = None

    def __init__(
        self,
        neo4j_retriever: Any,
        qdrant_retriever: Any,
        k: int = 4,
        neo4j_weight: float = 0.5,
        qdrant_weight: float = 0.5,
        fail_silently: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize hybrid retriever with component retrievers.

        Args:
            neo4j_retriever: Neo4jRetriever for graph-based retrieval
            qdrant_retriever: QdrantRetriever for vector-based retrieval
            k: Number of documents to return (default: 4)
            neo4j_weight: Weight for Neo4j results (default: 0.5)
            qdrant_weight: Weight for Qdrant results (default: 0.5)
            fail_silently: If True, return results from working retriever if one fails
            **kwargs: Additional arguments for BaseRetriever
        """
        super().__init__(
            k=k,
            neo4j_weight=neo4j_weight,
            qdrant_weight=qdrant_weight,
            fail_silently=fail_silently,
            **kwargs,
        )
        self._neo4j_retriever = neo4j_retriever
        self._qdrant_retriever = qdrant_retriever

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve documents via hybrid search combining Neo4j and Qdrant.

        Args:
            query: Search query string
            run_manager: Optional callback manager

        Returns:
            List of Document objects with combined scores

        Raises:
            RetrieverError: If both retrievers fail
        """
        if not query or not query.strip():
            return []

        neo4j_docs: list[Document] = []
        qdrant_docs: list[Document] = []
        neo4j_error: Exception | None = None
        qdrant_error: Exception | None = None

        # Get Neo4j results
        try:
            neo4j_docs = self._neo4j_retriever.invoke(query)
        except Exception as e:
            neo4j_error = e
            if not self.fail_silently:
                # Check if Qdrant also fails before raising
                pass

        # Get Qdrant results
        try:
            qdrant_docs = self._qdrant_retriever.invoke(query)
        except Exception as e:
            qdrant_error = e

        # Handle failures
        if neo4j_error and qdrant_error:
            raise RetrieverError(
                f"Both retrievers failed: Neo4j={neo4j_error}, Qdrant={qdrant_error}"
            )
        if neo4j_error and not self.fail_silently:
            raise RetrieverError(f"Neo4j retriever failed: {neo4j_error}")
        if qdrant_error and not self.fail_silently:
            raise RetrieverError(f"Qdrant retriever failed: {qdrant_error}")

        # Combine and deduplicate results
        return self._combine_results(neo4j_docs, qdrant_docs)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve documents via hybrid search async.

        Args:
            query: Search query string
            run_manager: Optional callback manager

        Returns:
            List of Document objects with combined scores

        Raises:
            RetrieverError: If both retrievers fail
        """
        if not query or not query.strip():
            return []

        neo4j_error: Exception | None = None
        qdrant_error: Exception | None = None

        # Get results from both retrievers concurrently
        neo4j_task = self._neo4j_retriever.ainvoke(query)
        qdrant_task = self._qdrant_retriever.ainvoke(query)

        neo4j_docs: list[Document] = []
        qdrant_docs: list[Document] = []

        try:
            neo4j_docs = await neo4j_task
        except Exception as e:
            neo4j_error = e

        try:
            qdrant_docs = await qdrant_task
        except Exception as e:
            qdrant_error = e

        # Handle failures
        if neo4j_error and qdrant_error:
            raise RetrieverError(
                f"Both retrievers failed: Neo4j={neo4j_error}, Qdrant={qdrant_error}"
            )
        if neo4j_error and not self.fail_silently:
            raise RetrieverError(f"Neo4j retriever failed: {neo4j_error}")
        if qdrant_error and not self.fail_silently:
            raise RetrieverError(f"Qdrant retriever failed: {qdrant_error}")

        # Combine and deduplicate results
        return self._combine_results(neo4j_docs, qdrant_docs)

    def _combine_results(
        self,
        neo4j_docs: list[Document],
        qdrant_docs: list[Document],
    ) -> list[Document]:
        """Combine results from Neo4j and Qdrant with weighted scoring.

        Performs:
        1. Weighted score fusion
        2. Deduplication by document ID
        3. Sorting by combined score
        4. Limiting to k results

        Args:
            neo4j_docs: Documents from Neo4j retriever
            qdrant_docs: Documents from Qdrant retriever

        Returns:
            Combined, deduplicated, and sorted list of Documents
        """
        # Build document map for deduplication
        doc_map: dict[str, Document] = {}
        score_map: dict[str, float] = {}

        # Process Neo4j documents
        for doc in neo4j_docs:
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            raw_score = doc.metadata.get("score", 0.5)
            weighted_score = raw_score * self.neo4j_weight

            if doc_id not in doc_map:
                doc_map[doc_id] = doc
                score_map[doc_id] = weighted_score
            else:
                # Add weighted score if same doc found in both
                score_map[doc_id] += weighted_score

        # Process Qdrant documents
        for doc in qdrant_docs:
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            raw_score = doc.metadata.get("score", 0.5)
            weighted_score = raw_score * self.qdrant_weight

            if doc_id not in doc_map:
                doc_map[doc_id] = doc
                score_map[doc_id] = weighted_score
            else:
                # Add weighted score if same doc found in both
                score_map[doc_id] += weighted_score

        # Create final documents with combined scores
        result_docs: list[Document] = []
        for doc_id, doc in doc_map.items():
            combined_score = score_map[doc_id]
            # Update metadata with combined score
            new_metadata = dict(doc.metadata)
            new_metadata["combined_score"] = combined_score
            result_docs.append(
                Document(page_content=doc.page_content, metadata=new_metadata)
            )

        # Sort by combined score (descending) and limit to k
        result_docs.sort(key=lambda d: d.metadata.get("combined_score", 0), reverse=True)
        return result_docs[: self.k]
