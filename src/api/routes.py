"""
API routes for semantic search service.

Provides endpoints for hybrid search, graph traversal, and graph queries.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.models import (
    ErrorResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    HealthResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    SearchResultItem,
    TraverseEdge,
    TraverseNode,
    TraverseRequest,
    TraverseResponse,
)

if TYPE_CHECKING:
    from src.api.dependencies import ServiceContainer

router = APIRouter()


def get_services() -> ServiceContainer:
    """Get service container - injected at runtime."""
    # This is overridden by dependency injection in create_app
    msg = "Services not configured"
    raise RuntimeError(msg)


@router.post(
    "/v1/search/hybrid",
    response_model=HybridSearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["search"],
    summary="Perform hybrid vector + graph search",
)
async def hybrid_search(
    request: HybridSearchRequest,
    services: ServiceContainer = Depends(get_services),  # noqa: B008
) -> HybridSearchResponse:
    """
    Execute a hybrid search combining vector similarity and graph relationships.

    The search combines:
    - Vector similarity search using Qdrant
    - Graph relationship scoring using Neo4j

    The final score is computed as:
    `hybrid_score = α * vector_score + (1-α) * graph_score`

    Args:
        request: Search parameters including query/embedding and scoring weights
        services: Injected service container

    Returns:
        HybridSearchResponse with ranked results and metadata
    """
    if not services.config.enable_hybrid_search:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "hybrid_search_disabled", "message": "Hybrid search is currently disabled"},
        )

    start_time = time.perf_counter()

    try:
        # Get embedding for query if not provided
        embedding = request.embedding
        if embedding is None and request.query:
            embedding = await services.embedding_service.embed(request.query)

        # Perform vector search
        vector_results = await services.vector_client.search(
            collection=request.collection,
            vector=embedding,
            limit=request.limit * 2,  # Fetch extra for re-ranking
        )

        # Get graph scores if enabled
        graph_scores: dict[str, float] = {}
        graph_metadata: dict[str, dict[str, Any]] = {}

        if request.include_graph and services.graph_client:
            try:
                node_ids = [r.id for r in vector_results]
                graph_data = await services.graph_client.get_relationship_scores(
                    node_ids=node_ids,
                    context=request.graph_context,
                )
                graph_scores = graph_data.get("scores", {})
                graph_metadata = graph_data.get("metadata", {})
            except Exception:
                # Graceful degradation - continue without graph scores
                pass

        # Compute hybrid scores and build results
        results: list[SearchResultItem] = []
        for vr in vector_results:
            vector_score = vr.score
            graph_score = graph_scores.get(vr.id, 0.0)

            # Hybrid score formula
            hybrid_score = request.alpha * vector_score + (1 - request.alpha) * graph_score

            results.append(
                SearchResultItem(
                    id=vr.id,
                    score=hybrid_score,
                    vector_score=vector_score,
                    graph_score=graph_score if graph_score > 0 else None,
                    payload=vr.payload or {},
                    graph_metadata=graph_metadata.get(vr.id),
                )
            )

        # Sort by hybrid score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[: request.limit]

        latency_ms = (time.perf_counter() - start_time) * 1000

        return HybridSearchResponse(
            results=results,
            total=len(results),
            query=request.query,
            alpha=request.alpha,
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "search_error", "message": str(e)},
        ) from e


@router.post(
    "/v1/graph/traverse",
    response_model=TraverseResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["graph"],
    summary="Traverse graph from a starting node",
)
async def graph_traverse(
    request: TraverseRequest,
    services: ServiceContainer = Depends(get_services),  # noqa: B008
) -> TraverseResponse:
    """
    Traverse the knowledge graph starting from a specified node.

    Follows relationships up to the specified depth, returning
    all discovered nodes and edges.

    Args:
        request: Traversal parameters
        services: Injected service container

    Returns:
        TraverseResponse with nodes and edges
    """
    if not services.config.enable_hybrid_search:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "graph_disabled", "message": "Graph services are currently disabled"},
        )

    start_time = time.perf_counter()

    try:
        result = await services.graph_client.traverse(
            start_node_id=request.start_node_id,
            relationship_types=request.relationship_types,
            max_depth=request.max_depth,
            limit=request.limit,
        )

        nodes = [
            TraverseNode(
                id=n["id"],
                labels=n.get("labels", []),
                properties=n.get("properties", {}),
                depth=n.get("depth", 0),
            )
            for n in result.get("nodes", [])
        ]

        edges = [
            TraverseEdge(
                source=e["source"],
                target=e["target"],
                type=e["type"],
                properties=e.get("properties", {}),
            )
            for e in result.get("edges", [])
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000

        return TraverseResponse(
            nodes=nodes,
            edges=edges,
            start_node=request.start_node_id,
            depth=request.max_depth,
            latency_ms=latency_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "traversal_error", "message": str(e)},
        ) from e


@router.post(
    "/v1/graph/query",
    response_model=GraphQueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["graph"],
    summary="Execute a read-only Cypher query",
)
async def graph_query(
    request: GraphQueryRequest,
    services: ServiceContainer = Depends(get_services),  # noqa: B008
) -> GraphQueryResponse:
    """
    Execute a read-only Cypher query against the knowledge graph.

    Only SELECT/MATCH operations are allowed for security.
    Write operations (CREATE, DELETE, MERGE, SET, REMOVE) are rejected.

    Args:
        request: Query parameters including Cypher statement
        services: Injected service container

    Returns:
        GraphQueryResponse with query results
    """
    if not services.config.enable_hybrid_search:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "graph_disabled", "message": "Graph services are currently disabled"},
        )

    start_time = time.perf_counter()

    try:
        result = await services.graph_client.execute_query(
            cypher=request.cypher,
            parameters=request.parameters or {},
            timeout=request.timeout_seconds,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return GraphQueryResponse(
            records=result.get("records", []),
            columns=result.get("columns", []),
            latency_ms=latency_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "query_error", "message": str(e)},
        ) from e


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check endpoint",
)
async def health_check(
    services: ServiceContainer = Depends(get_services),  # noqa: B008
) -> HealthResponse:
    """
    Check the health of all services.

    Returns the status of:
    - Vector search (Qdrant)
    - Graph database (Neo4j)
    - Embedding service

    Args:
        services: Injected service container

    Returns:
        HealthResponse with service statuses
    """
    service_statuses: dict[str, str] = {}

    # Check vector service
    try:
        if services.vector_client:
            is_healthy = await services.vector_client.health_check()
            service_statuses["vector"] = "healthy" if is_healthy else "unhealthy"
        else:
            service_statuses["vector"] = "not_configured"
    except Exception:
        service_statuses["vector"] = "unhealthy"

    # Check graph service
    try:
        if services.graph_client:
            is_healthy = await services.graph_client.health_check()
            service_statuses["graph"] = "healthy" if is_healthy else "unhealthy"
        else:
            service_statuses["graph"] = "not_configured"
    except Exception:
        service_statuses["graph"] = "unhealthy"

    # Determine overall status
    all_healthy = all(s in ("healthy", "not_configured") for s in service_statuses.values())
    overall_status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        services=service_statuses,
        version="1.0.0",
    )
