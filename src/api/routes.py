"""
API routes for semantic search service.

Provides endpoints for hybrid search, graph traversal, and graph queries.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.dependencies import ServiceContainer
from src.api.models import (
    ChapterContentResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    GraphQueryRequest,
    GraphQueryResponse,
    HealthResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    SearchResultItem,
    SimpleSearchRequest,
    SimpleSearchResponse,
    SimpleSearchResultItem,
    TraverseEdge,
    TraverseNode,
    TraverseRequest,
    TraverseResponse,
)
from src.search.metadata_filter import create_filter as create_domain_filter

logger = logging.getLogger(__name__)

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

        # === Phase 1-2: Domain-aware filtering ===
        # Apply focus area domain filter if specified
        domain_filter = None
        domain_applied = None
        if request.focus_areas:
            try:
                domain_filter = create_domain_filter()
                # Use first focus area as domain (could extend to multiple)
                domain_applied = request.focus_areas[0] if request.focus_areas else None
                if domain_applied and domain_applied in domain_filter.available_domains:
                    # Convert vector results to filter-compatible format
                    passages = [
                        {
                            "id": r.id,
                            "content": r.payload.get("content", "") if r.payload else "",
                            "score": r.score,
                            "metadata": r.payload or {},
                        }
                        for r in vector_results
                    ]
                    # Apply filter (adjust scores, optionally remove low-relevance)
                    filtered_passages = domain_filter.apply(
                        passages=passages,
                        domain=domain_applied,
                        remove_filtered=False,  # Keep all, just adjust scores
                    )
                    # Update vector_results with adjusted scores and metadata
                    passage_map = {p["id"]: p for p in filtered_passages}
                    for vr in vector_results:
                        if vr.id in passage_map:
                            fp = passage_map[vr.id]
                            vr.score = fp["score"]  # Use adjusted score
                            if vr.payload is None:
                                vr.payload = {}
                            # Add domain filter metadata
                            vr.payload["domain_filter"] = fp.get("metadata", {}).get("domain_filter", {})
                    logger.debug(
                        "Applied domain filter '%s' to %d results",
                        domain_applied,
                        len(vector_results),
                    )
                else:
                    domain_applied = None  # Domain not recognized
            except Exception as e:
                logger.warning("Domain filter failed, continuing without: %s", e)
                domain_applied = None

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

            # Extract domain filter metadata if present
            domain_filter_data = (vr.payload or {}).get("domain_filter", {})
            focus_area_result = domain_applied if domain_filter_data else None
            focus_score_result = domain_filter_data.get("adjustment", 0.0) if domain_filter_data else None
            domain_adjustment = domain_filter_data.get("adjustment", 0.0) if domain_filter_data else None

            results.append(
                SearchResultItem(
                    id=vr.id,
                    score=hybrid_score,
                    vector_score=vector_score,
                    graph_score=graph_score if graph_score > 0 else None,
                    payload=vr.payload or {},
                    graph_metadata=graph_metadata.get(vr.id),
                    focus_area_applied=focus_area_result,
                    focus_score=focus_score_result,
                    domain_filter_adjustment=domain_adjustment,
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
    request: Request,
    services: ServiceContainer = Depends(get_services),  # noqa: B008
) -> HealthResponse:
    """
    Check the health of all services.

    Returns the status of:
    - Vector search (Qdrant)
    - Graph database (Neo4j)
    - Embedding service

    Args:
        request: FastAPI request (for app.state access)
        services: Injected service container

    Returns:
        HealthResponse with service statuses and dependencies
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

    # Get dependency status from app.state (set by lifespan handler)
    dependencies: dict[str, str] = {}
    if hasattr(request.app.state, "dependencies"):
        dependencies = request.app.state.dependencies
    else:
        # Default for backward compatibility
        dependencies = {
            "qdrant": service_statuses.get("vector", "unknown"),
            "neo4j": service_statuses.get("graph", "unknown"),
            "embedder": "loaded" if services.embedding_service else "not_configured",
        }

    # Determine overall status
    all_healthy = all(s in ("healthy", "not_configured") for s in service_statuses.values())
    overall_status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        services=service_statuses,
        dependencies=dependencies,
        version="1.0.0",
    )


# ==============================================================================
# Embedding Endpoint (WBS 0.2.1)
# ==============================================================================


@router.post(
    "/v1/embed",
    response_model=EmbedResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["embeddings"],
    summary="Generate embeddings for text",
)
async def embed_text(
    request: EmbedRequest,
    services: ServiceContainer = Depends(get_services),  # noqa: B008
) -> EmbedResponse:
    """
    Generate embedding vectors for text input.

    Accepts either a single text string or a list of texts.
    Returns embedding vectors of the configured dimension.

    Reference: END_TO_END_INTEGRATION_WBS.md WBS 0.2.1.3

    Args:
        request: Embedding request with text(s) to embed
        services: Injected service container

    Returns:
        EmbedResponse with embedding vectors and metadata
    """
    if services.embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "embedding_service_unavailable", "message": "Embedding service is not configured"},
        )

    start_time = time.perf_counter()

    try:
        # Normalize input to list
        texts = [request.text] if isinstance(request.text, str) else request.text

        # Generate embeddings for each text
        embeddings: list[list[float]] = []
        for text in texts:
            embedding = await services.embedding_service.embed(text)
            embeddings.append(embedding)

        # Determine dimensions from first embedding
        dimensions = len(embeddings[0]) if embeddings else 0

        # Model name (use default if not specified)
        model_name = request.model or services.config.embedding_model if hasattr(services.config, 'embedding_model') else "all-mpnet-base-v2"

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Generated %d embeddings in %.2fms",
            len(embeddings),
            latency_ms,
        )

        return EmbedResponse(
            embeddings=embeddings,
            model=model_name,
            dimensions=dimensions,
            usage={"total_tokens": sum(len(t.split()) for t in texts)},
        )

    except Exception as e:
        logger.exception("Embedding generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "embedding_failed", "message": str(e)},
        ) from e


# ==============================================================================
# Simple Search Endpoint (WBS 0.2.2)
# ==============================================================================


@router.post(
    "/v1/search",
    response_model=SimpleSearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["search"],
    summary="Perform simple similarity search",
)
async def simple_search(
    request: SimpleSearchRequest,
    services: ServiceContainer = Depends(get_services),  # noqa: B008
) -> SimpleSearchResponse:
    """
    Execute a simple vector similarity search.

    This is a streamlined search endpoint that performs pure vector
    similarity search without graph-based scoring.

    Reference: END_TO_END_INTEGRATION_WBS.md WBS 0.2.2

    Args:
        request: Search parameters including query and limit
        services: Injected service container

    Returns:
        SimpleSearchResponse with ranked results
    """
    if services.vector_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "vector_service_unavailable", "message": "Vector search service is not configured"},
        )

    start_time = time.perf_counter()

    try:
        # Get embedding for query
        if services.embedding_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "embedding_service_unavailable", "message": "Embedding service is not configured"},
            )

        embedding = await services.embedding_service.embed(request.query)

        # Perform vector search
        vector_results = await services.vector_client.search(
            collection=request.collection,
            vector=embedding,
            limit=request.limit,
        )

        # Build response results
        results: list[SimpleSearchResultItem] = []
        for vr in vector_results:
            # Apply min_score filter if specified
            if request.min_score is not None and vr.score < request.min_score:
                continue

            results.append(
                SimpleSearchResultItem(
                    id=vr.id,
                    score=vr.score,
                    payload=vr.payload or {},
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Simple search for '%s' returned %d results in %.2fms",
            request.query[:50],
            len(results),
            latency_ms,
        )

        return SimpleSearchResponse(
            results=results,
            total=len(results),
            query=request.query,
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Simple search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "search_failed", "message": str(e)},
        ) from e


# =============================================================================
# Chapter Content Retrieval - Kitchen Brigade: Cookbook (semantic-search)
# =============================================================================


@router.get(
    "/v1/chapters/{book_id}/{chapter_number}",
    response_model=ChapterContentResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Chapter not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    tags=["chapters"],
    summary="Get chapter content by book and chapter number",
)
async def get_chapter_content(
    book_id: str,
    chapter_number: int,
    services: ServiceContainer = Depends(get_services),  # noqa: B008
) -> ChapterContentResponse:
    """
    Retrieve chapter content from Neo4j by book ID and chapter number.
    
    This endpoint enables the Kitchen Brigade architecture where ai-agents
    (Expeditor) retrieves content through semantic-search (Cookbook) rather
    than directly accessing Neo4j.
    
    Args:
        book_id: Book identifier (e.g., "Architecture_Patterns_with_Python")
        chapter_number: Chapter number (1-indexed)
        services: Injected service container
        
    Returns:
        ChapterContentResponse with full chapter content and metadata
    """
    if services.graph_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "graph_unavailable", "message": "Graph database is not configured"},
        )
    
    try:
        # Query Neo4j for chapter content
        cypher = """
        MATCH (c:Chapter)
        WHERE c.book_id = $book_id AND c.number = $chapter_number
        RETURN c.book_id as book_id,
               c.number as chapter_number,
               c.title as title,
               c.summary as summary,
               c.keywords as keywords,
               c.concepts as concepts,
               c.page_range as page_range
        LIMIT 1
        """
        
        result = await services.graph_client.execute_query(
            cypher=cypher,
            parameters={"book_id": book_id, "chapter_number": chapter_number},
        )
        
        records = result.get("records", [])
        
        if not records:
            # Return not found response
            return ChapterContentResponse(
                book_id=book_id,
                chapter_number=chapter_number,
                title="",
                summary="",
                keywords=[],
                concepts=[],
                page_range="",
                found=False,
            )
        
        record = records[0]
        
        return ChapterContentResponse(
            book_id=record.get("book_id", book_id),
            chapter_number=record.get("chapter_number", chapter_number),
            title=record.get("title", ""),
            summary=record.get("summary", ""),
            keywords=record.get("keywords") or [],
            concepts=record.get("concepts") or [],
            page_range=record.get("page_range", ""),
            found=True,
        )
        
    except Exception as e:
        logger.exception("Failed to retrieve chapter content: %s/%d", book_id, chapter_number)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "chapter_retrieval_failed", "message": str(e)},
        ) from e
