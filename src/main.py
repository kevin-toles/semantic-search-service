"""
Main entry point for semantic-search-service.

Creates the FastAPI application instance for uvicorn.

WBS 0.2.3: Supports real database clients via USE_REAL_CLIENTS=true
PCON-7: Dynamic infrastructure URL resolution (no hardcoded values)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.api.app import create_app
from src.api.dependencies import (
    EmbeddingServiceProtocol,
    GraphClientProtocol,
    ServiceConfig,
    ServiceContainer,
    VectorClientProtocol,
)
from src.infrastructure_config import get_infrastructure_urls, get_infrastructure_mode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import FastAPI

logger = logging.getLogger(__name__)


# ==============================================================================
# Real Client Wrappers (WBS 0.2.3)
# ==============================================================================


class RealQdrantClient:
    """Wrapper around qdrant_client.QdrantClient implementing VectorClientProtocol."""

    def __init__(self, url: str) -> None:
        """Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL (e.g., http://localhost:6333)
        """
        from qdrant_client import QdrantClient
        
        self._client = QdrantClient(url=url, timeout=30)
        self._url = url
        logger.info("Initialized Qdrant client: %s", url)

    async def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        """Search for similar vectors in Qdrant."""
        try:
            # Run sync client in thread pool
            # Use query_points (new API) instead of search (deprecated)
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=limit,
                    **kwargs,
                ),
            )
            
            # Convert to format expected by routes
            # query_points returns QueryResponse with .points attribute
            return [
                _ScoredResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {},
                )
                for r in results.points
            ]
        except Exception as e:
            logger.warning("Qdrant search failed: %s", e)
            return []

    async def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.get_collections(),
            )
            return result is not None
        except Exception as e:
            logger.warning("Qdrant health check failed: %s", e)
            return False


@dataclass
class _ScoredResult:
    """Internal result object matching expected interface."""
    id: str
    score: float
    payload: dict[str, Any]


class RealNeo4jClient:
    """Wrapper around neo4j.AsyncGraphDatabase implementing GraphClientProtocol."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        """Initialize Neo4j driver.
        
        Args:
            uri: Neo4j Bolt URI (e.g., bolt://localhost:7687)
            user: Neo4j username
            password: Neo4j password
        """
        from neo4j import GraphDatabase
        
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._uri = uri
        logger.info("Initialized Neo4j driver: %s", uri)

    async def traverse(
        self,
        start_node_id: str,
        relationship_types: list[str] | None = None,
        max_depth: int = 3,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Traverse graph from a starting node."""
        rel_filter = ""
        if relationship_types:
            rel_filter = ":" + "|".join(relationship_types)
        
        cypher = f"""
        MATCH path = (start {{id: $start_id}})-[r{rel_filter}*1..{max_depth}]->(end)
        RETURN nodes(path) as nodes, relationships(path) as rels
        LIMIT $limit
        """
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._execute_read(cypher, {"start_id": start_node_id, "limit": limit}),
            )
            return result
        except Exception as e:
            logger.warning("Neo4j traverse failed: %s", e)
            return {"nodes": [], "edges": []}

    def _execute_read(self, cypher: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute read query synchronously."""
        nodes = []
        edges = []
        with self._driver.session() as session:
            result = session.run(cypher, parameters)
            for record in result:
                for node in record.get("nodes", []):
                    nodes.append({
                        "id": str(node.element_id),
                        "labels": list(node.labels),
                        "properties": dict(node),
                    })
                for rel in record.get("rels", []):
                    edges.append({
                        "source": str(rel.start_node.element_id),
                        "target": str(rel.end_node.element_id),
                        "type": rel.type,
                        "properties": dict(rel),
                    })
        return {"nodes": nodes, "edges": edges}

    async def execute_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a Cypher query."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._execute_query_sync(cypher, parameters or {}),
            )
            return result
        except Exception as e:
            logger.warning("Neo4j query failed: %s", e)
            return {"records": [], "columns": []}

    def _execute_query_sync(self, cypher: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute query synchronously."""
        with self._driver.session() as session:
            result = session.run(cypher, parameters)
            records = [dict(record) for record in result]
            columns = list(result.keys()) if records else []
            return {"records": records, "columns": columns}

    def get_relationship_scores(
        self,
        node_ids: list[str],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get relationship-based scores for nodes."""
        # Placeholder for future implementation - uses params to satisfy linter
        _ = node_ids, context
        return {"scores": {}, "metadata": {}}

    async def health_check(self) -> bool:
        """Check if Neo4j is reachable."""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._verify_connectivity(),
            )
            return result
        except Exception as e:
            logger.warning("Neo4j health check failed: %s", e)
            return False

    def _verify_connectivity(self) -> bool:
        """Verify connectivity synchronously."""
        self._driver.verify_connectivity()
        return True

    def close(self) -> None:
        """Close the driver."""
        self._driver.close()


class RealEmbeddingService:
    """Wrapper around SentenceTransformer implementing EmbeddingServiceProtocol."""

    def __init__(self, model_name: str) -> None:
        """Initialize SentenceTransformer model.
        
        Args:
            model_name: Name of the sentence-transformers model (REQUIRED)
        """
        from sentence_transformers import SentenceTransformer
        
        logger.info("Loading SentenceTransformer model: %s", model_name)
        start = time.time()
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        elapsed = time.time() - start
        logger.info("Model loaded in %.2fs", elapsed)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True).tolist(),
        )
        return result

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name


# ==============================================================================
# Service Factory (WBS 0.2.3, PCON-7)
# ==============================================================================


def create_real_services(config: ServiceConfig | None = None) -> ServiceContainer:
    """Create ServiceContainer with real database clients.
    
    Connection URLs are resolved dynamically based on INFRASTRUCTURE_MODE:
    - docker: Uses Docker DNS names (ai-platform-neo4j, ai-platform-qdrant)
    - hybrid/native: Uses localhost
    
    If explicit env vars are set, they override the dynamic values.
    
    Required env vars:
    - NEO4J_USER: Neo4j username
    - NEO4J_PASSWORD: Neo4j password
    - EMBEDDING_MODEL: SentenceTransformer model name
    
    Optional (auto-resolved from INFRASTRUCTURE_MODE if not set):
    - QDRANT_URL: Qdrant server URL
    - NEO4J_URI: Neo4j Bolt URI
    
    Args:
        config: Optional ServiceConfig override
        
    Returns:
        ServiceContainer with real clients initialized
    """
    cfg = config or ServiceConfig()
    
    # Get dynamic URLs based on infrastructure mode
    mode = get_infrastructure_mode()
    urls = get_infrastructure_urls(mode)
    
    # Use env var if explicitly set, otherwise use dynamic value
    qdrant_url = os.environ.get("QDRANT_URL") or urls["QDRANT_URL"]
    neo4j_uri = os.environ.get("NEO4J_URI") or urls["NEO4J_URI"]
    
    # Credentials - defaults match existing Neo4j volume (devpassword)
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "devpassword")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    logger.info("Creating real services (mode: %s)...", mode)
    logger.info("  Qdrant URL: %s", qdrant_url)
    logger.info("  Neo4j URI: %s", neo4j_uri)
    logger.info("  Embedding model: %s", embedding_model)
    
    # Initialize clients
    vector_client = RealQdrantClient(url=qdrant_url)
    graph_client = RealNeo4jClient(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    embedding_service = RealEmbeddingService(model_name=embedding_model)
    
    return ServiceContainer(
        config=cfg,
        vector_client=vector_client,
        graph_client=graph_client,
        embedding_service=embedding_service,
    )


# ==============================================================================
# Application Lifecycle (WBS 0.2.3)
# ==============================================================================


async def _check_qdrant_health(services: ServiceContainer, app: FastAPI) -> None:
    """Check Qdrant health and update app state."""
    try:
        if await services.vector_client.health_check():
            app.state.dependencies["qdrant"] = "connected"
            logger.info("Qdrant: connected")
        else:
            app.state.dependencies["qdrant"] = "disconnected"
            logger.warning("Qdrant: disconnected")
    except Exception as e:
        app.state.dependencies["qdrant"] = f"error: {e}"
        logger.error("Qdrant error: %s", e)


async def _check_neo4j_health(services: ServiceContainer, app: FastAPI) -> None:
    """Check Neo4j health and update app state."""
    try:
        if await services.graph_client.health_check():
            app.state.dependencies["neo4j"] = "connected"
            logger.info("Neo4j: connected")
        else:
            app.state.dependencies["neo4j"] = "disconnected"
            logger.warning("Neo4j: disconnected")
    except Exception as e:
        app.state.dependencies["neo4j"] = f"error: {e}"
        logger.error("Neo4j error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.
    
    On startup: Initialize real services (default behavior)
    On shutdown: Clean up resources
    
    Set USE_FAKE_CLIENTS=true to use mock clients for testing only.
    """
    use_fake_clients = os.getenv("USE_FAKE_CLIENTS", "false").lower() in ("true", "1", "yes")
    use_real_clients = not use_fake_clients
    
    if use_real_clients:
        logger.info("Starting with REAL database clients")
        services = create_real_services()
        
        # Store services in app state
        app.state.services = services
        
        # Store dependency status for health endpoint
        app.state.dependencies = {
            "qdrant": "checking",
            "neo4j": "checking", 
            "embedder": "loaded",
        }
        
        await _check_qdrant_health(services, app)
        await _check_neo4j_health(services, app)
    else:
        logger.info("Starting with FAKE clients (USE_FAKE_CLIENTS=true)")
        app.state.dependencies = {
            "qdrant": "fake",
            "neo4j": "fake",
            "embedder": "fake",
        }
    
    yield
    
    # Cleanup on shutdown
    if use_real_clients and hasattr(app.state, "services"):
        services = app.state.services
        if hasattr(services.graph_client, "close"):
            services.graph_client.close()
            logger.info("Neo4j driver closed")


# Create application with lifespan
app = create_app()
app.router.lifespan_context = lifespan
