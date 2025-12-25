"""
Pydantic models for API request/response validation.

These models define the contract for the hybrid search API endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(
        default=None,
        description="Text query to search for",
        min_length=1,
        max_length=10000,
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Pre-computed embedding vector",
    )
    start_node_id: str | None = Field(
        default=None,
        description="Starting node ID for graph context",
    )
    collection: str = Field(
        default="documents",
        description="Vector collection to search",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector score vs graph score (0-1)",
    )
    include_graph: bool = Field(
        default=True,
        description="Whether to include graph-based scoring",
    )
    graph_context: dict[str, Any] | None = Field(
        default=None,
        description="Additional context for graph traversal",
    )
    # Tier filtering (feature/semantic-tuning)
    tier_filter: list[int] | None = Field(
        default=None,
        description="Filter results to specific taxonomy tiers (1, 2, or 3)",
    )
    tier_boost: bool = Field(
        default=True,
        description="Apply tier-based score boosting (tier 1 = highest)",
    )
    min_term_matches: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum number of query terms that must match",
    )
    # Domain-aware filtering (RELEVANCE_TUNING_PLAN.md)
    focus_areas: list[str] | None = Field(
        default=None,
        description="Focus areas for domain-aware filtering (e.g., 'llm_rag', 'microservices_architecture')",
    )
    focus_keywords: list[str] | None = Field(
        default=None,
        description="Custom focus keywords for relevance scoring",
    )

    @model_validator(mode="after")
    def validate_query_or_embedding(self) -> HybridSearchRequest:
        """Ensure at least one of query or embedding is provided."""
        if self.query is None and self.embedding is None:
            msg = "Either 'query' or 'embedding' must be provided"
            raise ValueError(msg)
        return self
    
    @field_validator("tier_filter")
    @classmethod
    def validate_tier_filter(cls, v: list[int] | None) -> list[int] | None:
        """Ensure tier filter values are valid (1, 2, or 3)."""
        if v is not None:
            valid_tiers = {1, 2, 3}
            invalid = [t for t in v if t not in valid_tiers]
            if invalid:
                msg = f"Invalid tier values: {invalid}. Valid tiers are 1, 2, 3."
                raise ValueError(msg)
        return v


class SearchResultItem(BaseModel):
    """Individual search result item."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Unique identifier for the result")
    score: float = Field(description="Combined relevance score")
    vector_score: float | None = Field(
        default=None,
        description="Score from vector similarity",
    )
    graph_score: float | None = Field(
        default=None,
        description="Score from graph relationships",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata",
    )
    graph_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Graph relationship information",
    )
    # Tier information (feature/semantic-tuning)
    tier: int | None = Field(
        default=None,
        description="Taxonomy tier (1=Architecture, 2=Implementation, 3=Engineering)",
    )
    tier_boost_applied: float | None = Field(
        default=None,
        description="Boost factor applied based on tier (if tier_boost enabled)",
    )
    term_match_count: int | None = Field(
        default=None,
        description="Number of query terms that matched in this result",
    )
    # Domain filtering results (RELEVANCE_TUNING_PLAN.md)
    focus_area_applied: str | None = Field(
        default=None,
        description="Focus area filter that was applied (if any)",
    )
    focus_score: float | None = Field(
        default=None,
        description="Focus keyword overlap score (0-1)",
    )
    domain_filter_adjustment: float | None = Field(
        default=None,
        description="Score adjustment from domain filtering",
    )


class HybridSearchResponse(BaseModel):
    """Response model for hybrid search endpoint."""

    model_config = ConfigDict(extra="forbid")

    results: list[SearchResultItem] = Field(
        default_factory=list,
        description="List of search results",
    )
    total: int = Field(
        description="Total number of results found",
    )
    query: str | None = Field(
        default=None,
        description="Original query text",
    )
    alpha: float = Field(
        description="Alpha value used for scoring",
    )
    latency_ms: float = Field(
        description="Search latency in milliseconds",
    )


class TraverseRequest(BaseModel):
    """Request model for graph traversal endpoint."""

    model_config = ConfigDict(extra="forbid")

    start_node_id: str = Field(
        description="Starting node ID for traversal",
    )
    relationship_types: list[str] | None = Field(
        default=None,
        description="Types of relationships to follow",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum traversal depth",
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of nodes to return",
    )


class TraverseNode(BaseModel):
    """Node in traversal result."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Node identifier")
    labels: list[str] = Field(default_factory=list, description="Node labels")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Node properties",
    )
    depth: int = Field(description="Distance from start node")


class TraverseEdge(BaseModel):
    """Edge in traversal result."""

    model_config = ConfigDict(extra="allow")

    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    type: str = Field(description="Relationship type")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Edge properties",
    )


class TraverseResponse(BaseModel):
    """Response model for graph traversal endpoint."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[TraverseNode] = Field(
        default_factory=list,
        description="Traversed nodes",
    )
    edges: list[TraverseEdge] = Field(
        default_factory=list,
        description="Relationships between nodes",
    )
    start_node: str = Field(description="Starting node ID")
    depth: int = Field(description="Actual traversal depth")
    latency_ms: float = Field(description="Traversal latency in milliseconds")

    @property
    def results(self) -> list[TraverseNode]:
        """Alias for nodes to support expected API response format."""
        return self.nodes

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Include results alias in serialization."""
        data = super().model_dump(**kwargs)
        data["results"] = data.get("nodes", [])
        return data


class GraphQueryRequest(BaseModel):
    """Request model for graph query endpoint."""

    model_config = ConfigDict(extra="forbid")

    cypher: str = Field(
        description="Cypher query to execute",
        min_length=1,
        max_length=10000,
    )
    parameters: dict[str, Any] | None = Field(
        default=None,
        description="Query parameters",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Query timeout in seconds",
    )

    @field_validator("cypher")
    @classmethod
    def validate_cypher_read_only(cls, v: str) -> str:
        """Ensure cypher query is read-only for safety."""
        write_keywords = ["CREATE", "DELETE", "MERGE", "SET", "REMOVE", "DETACH"]
        upper_query = v.upper()
        for keyword in write_keywords:
            if keyword in upper_query:
                msg = f"Write operation '{keyword}' not allowed in query endpoint"
                raise ValueError(msg)
        return v


class GraphQueryResponse(BaseModel):
    """Response model for graph query endpoint."""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Query result records",
    )
    columns: list[str] = Field(
        default_factory=list,
        description="Column names in result",
    )
    latency_ms: float = Field(description="Query latency in milliseconds")

    @property
    def results(self) -> list[dict[str, Any]]:
        """Alias for records to support expected API response format."""
        return self.records

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Include results alias in serialization."""
        data = super().model_dump(**kwargs)
        data["results"] = data.get("records", [])
        return data


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(description="Overall health status")
    services: dict[str, str] = Field(
        default_factory=dict,
        description="Individual service statuses",
    )
    dependencies: dict[str, str] = Field(
        default_factory=dict,
        description="External dependency connection statuses (qdrant, neo4j, embedder)",
    )
    version: str = Field(description="API version")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    detail: dict[str, Any] | None = Field(
        default=None,
        description="Additional error details",
    )


# ==============================================================================
# Embedding Models (WBS 0.2.1)
# ==============================================================================


class EmbedRequest(BaseModel):
    """Request model for text embedding endpoint.
    
    Reference: END_TO_END_INTEGRATION_WBS.md WBS 0.2.1.1
    """

    model_config = ConfigDict(extra="forbid")

    text: str | list[str] = Field(
        description="Text or list of texts to embed",
    )
    model: str | None = Field(
        default=None,
        description="Model to use for embedding (uses default if not specified)",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str | list[str]) -> str | list[str]:
        """Ensure text is not empty."""
        if isinstance(v, str):
            if not v.strip():
                msg = "Text cannot be empty"
                raise ValueError(msg)
        elif isinstance(v, list):
            if not v:
                msg = "Text list cannot be empty"
                raise ValueError(msg)
            for i, t in enumerate(v):
                if not isinstance(t, str) or not t.strip():
                    msg = f"Text at index {i} is empty or invalid"
                    raise ValueError(msg)
        return v


class EmbedResponse(BaseModel):
    """Response model for text embedding endpoint.
    
    Reference: END_TO_END_INTEGRATION_WBS.md WBS 0.2.1.2
    """

    model_config = ConfigDict(extra="forbid")

    embeddings: list[list[float]] = Field(
        description="List of embedding vectors",
    )
    model: str = Field(
        description="Model used for embedding",
    )
    dimensions: int = Field(
        description="Dimension of each embedding vector",
    )
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage statistics",
    )


# ==============================================================================
# Simple Search Models (WBS 0.2.2)
# ==============================================================================


class SimpleSearchRequest(BaseModel):
    """Request model for simple similarity search endpoint.
    
    Reference: END_TO_END_INTEGRATION_WBS.md WBS 0.2.2
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description="Text query to search for",
        min_length=1,
        max_length=10000,
    )
    collection: str = Field(
        default="documents",
        description="Vector collection to search",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )


class SimpleSearchResultItem(BaseModel):
    """Individual search result for simple search."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Unique identifier for the result")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Similarity score (0-1)",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Associated metadata",
    )


class SimpleSearchResponse(BaseModel):
    """Response model for simple similarity search endpoint.
    
    Reference: END_TO_END_INTEGRATION_WBS.md WBS 0.2.2
    """

    model_config = ConfigDict(extra="forbid")

    results: list[SimpleSearchResultItem] = Field(
        default_factory=list,
        description="List of search results",
    )
    total: int = Field(
        description="Number of results returned",
    )
    query: str = Field(
        description="Original query text",
    )
    latency_ms: float = Field(
        description="Search latency in milliseconds",
    )


# =============================================================================
# Chapter Content Models - Kitchen Brigade: Cookbook (semantic-search)
# =============================================================================


class ChapterContentResponse(BaseModel):
    """Response model for chapter content retrieval.
    
    Used by ai-agents (Expeditor) to retrieve chapter content from
    the Cookbook (semantic-search) via Neo4j.
    
    Reference: Kitchen Brigade Architecture - Option A
    """

    model_config = ConfigDict(extra="forbid")

    book_id: str = Field(description="Book identifier")
    chapter_number: int = Field(description="Chapter number")
    title: str = Field(description="Chapter title")
    summary: str = Field(default="", description="Full chapter summary/content")
    keywords: list[str] = Field(default_factory=list, description="Chapter keywords")
    concepts: list[str] = Field(default_factory=list, description="Chapter concepts")
    page_range: str = Field(default="", description="Page range (e.g., '46-73')")
    found: bool = Field(default=True, description="Whether chapter was found")


# =============================================================================
# EEP-4: Graph Relationships Models (AC-4.3.1 to AC-4.3.3)
# =============================================================================


class RelatedChapterItem(BaseModel):
    """A related chapter from relationship traversal.
    
    AC-4.3.3: Return relationship type with each related chapter.
    """

    model_config = ConfigDict(extra="forbid")

    chapter_id: str = Field(description="Related chapter ID")
    relationship_type: str = Field(
        description="Type of relationship (PARALLEL, PERPENDICULAR, SKIP_TIER)"
    )
    target_tier: int = Field(
        ge=1,
        le=10,
        description="Tier of the related chapter",
    )
    title: str | None = Field(
        default=None,
        description="Chapter title (if available)",
    )


class ChapterRelationshipsResponse(BaseModel):
    """Response model for chapter relationships endpoint.
    
    AC-4.3.1: GET /v1/graph/relationships/{chapter_id}
    """

    model_config = ConfigDict(extra="forbid")

    chapter_id: str = Field(description="The queried chapter ID")
    relationships: list[RelatedChapterItem] = Field(
        default_factory=list,
        description="List of related chapters",
    )
    total_count: int = Field(
        ge=0,
        description="Total number of relationships found",
    )


class BatchRelationshipsRequest(BaseModel):
    """Request model for batch relationships endpoint.
    
    AC-4.3.2: POST /v1/graph/relationships/batch
    """

    model_config = ConfigDict(extra="forbid")

    chapter_ids: list[str] = Field(
        min_length=1,
        max_length=100,
        description="List of chapter IDs to query",
    )


class BatchRelationshipsResponse(BaseModel):
    """Response model for batch relationships endpoint.
    
    AC-4.3.2: POST /v1/graph/relationships/batch
    """

    model_config = ConfigDict(extra="forbid")

    results: list[ChapterRelationshipsResponse] = Field(
        default_factory=list,
        description="Relationships for each queried chapter",
    )
    total_chapters: int = Field(
        ge=0,
        description="Number of chapters processed",
    )
