# Semantic Search Service Inventory

**Service:** semantic-search-service  
**Port:** 8081  
**Role:** Cookbook (Retrieval) in Kitchen Brigade Architecture  
**Generated:** January 27, 2026

---

## 1. Module Inventory

### 1.1 API Layer (`src/api/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| [app.py](src/api/app.py) | FastAPI application factory | `create_app()`, `configure_app_services()` |
| [routes.py](src/api/routes.py#L1-L887) | All API endpoints | 10 endpoints (see Section 2) |
| [models.py](src/api/models.py#L1-L595) | Pydantic request/response schemas | 20+ models (see Section 2) |
| [dependencies.py](src/api/dependencies.py) | Dependency injection protocols | `VectorClientProtocol`, `GraphClientProtocol`, `EmbeddingServiceProtocol`, `ServiceContainer`, `ServiceConfig` |

### 1.2 Core Layer (`src/core/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| [config.py](src/core/config.py) | Pydantic Settings configuration | `Settings`, `get_settings()` |
| [logging.py](src/core/logging.py) | Logging configuration | (standard logging setup) |

### 1.3 Graph Layer (`src/graph/`) - Neo4j Integration

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| [neo4j_client.py](src/graph/neo4j_client.py#L1-L393) | Neo4j Repository client | `Neo4jClient`, `Neo4jClientProtocol` |
| [health.py](src/graph/health.py) | Neo4j health checks | `get_neo4j_driver()`, `check_neo4j_health()`, `check_neo4j_health_detailed()` |
| [traversal.py](src/graph/traversal.py#L1-L548) | BFS/DFS graph traversal | `RelationshipType`, `TraversalDirection`, `TraversalResult`, `Neo4jClientLike` |
| [relationships.py](src/graph/relationships.py#L1-L643) | EEP-4 Relationship edges | `RelationshipEdgeType`, `RelationshipEdge`, `RelatedChapter`, `RelationshipResult` |
| [schema.py](src/graph/schema.py#L1-L376) | Neo4j schema helpers | `NodeLabels`, `RelationshipLabels`, Cypher generators |
| [taxonomy_loader.py](src/graph/taxonomy_loader.py#L1-L259) | Taxonomy JSON → Neo4j loader | `Chapter`, `Book`, `Taxonomy`, `parse_chapter()`, `parse_book()` |
| [exceptions.py](src/graph/exceptions.py) | Custom Neo4j exceptions | `Neo4jError`, `Neo4jConnectionError`, `Neo4jQueryError`, `Neo4jTransactionError` |

### 1.4 Search Layer (`src/search/`) - Qdrant Integration

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| [vector.py](src/search/vector.py#L1-L606) | Qdrant vector search client | `QdrantSearchClient`, `QdrantSearchClientProtocol`, `SearchResult` |
| [hybrid.py](src/search/hybrid.py#L1-L390) | Vector + Graph hybrid search | `HybridSearchService`, `HybridSearchResult`, `VectorSearchClientProtocol`, `GraphClientProtocol` |
| [metadata_filter.py](src/search/metadata_filter.py#L1-L398) | Domain-aware filtering | `MetadataFilter`, `DomainConfig`, `FilterResult` |
| [ranker.py](src/search/ranker.py#L1-L393) | Score fusion strategies | `ResultRanker` (linear, RRF, max strategies) |
| [exceptions.py](src/search/exceptions.py) | Custom Qdrant exceptions | `QdrantError`, `QdrantConnectionError`, `QdrantSearchError` |

### 1.5 Vector Layer (`src/vector/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| [health.py](src/vector/health.py) | Qdrant health checks | `get_qdrant_client()`, `check_qdrant_health()`, `check_qdrant_health_detailed()` |

### 1.6 Retrievers Layer (`src/retrievers/`) - LangChain Integration

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| [hybrid_retriever.py](src/retrievers/hybrid_retriever.py#L1-L277) | LangChain ensemble retriever | `HybridRetriever` (extends `BaseRetriever`) |
| [neo4j_retriever.py](src/retrievers/neo4j_retriever.py#L1-L169) | LangChain Neo4j retriever | `Neo4jRetriever` (extends `BaseRetriever`) |
| [qdrant_retriever.py](src/retrievers/qdrant_retriever.py#L1-L192) | LangChain Qdrant retriever | `QdrantRetriever` (extends `BaseRetriever`) |
| [exceptions.py](src/retrievers/exceptions.py) | Custom retriever exceptions | `RetrieverError`, `RetrieverConnectionError`, `RetrieverQueryError`, `DocumentNotFoundError`, `EmbedderError` |

### 1.7 Infrastructure (`src/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| [main.py](src/main.py#L1-L432) | Application entry point | `RealQdrantClient`, `RealNeo4jClient`, `RealEmbeddingService`, `create_real_services()`, `lifespan()`, `app` |
| [infrastructure_config.py](src/infrastructure_config.py#L1-L172) | Dynamic URL resolution | `get_infrastructure_mode()`, `get_infrastructure_urls()` |

---

## 2. API Surface

### 2.1 Search Endpoints

#### `POST /v1/search/hybrid`
**File:** [routes.py#L53-L218](src/api/routes.py#L53)

| Aspect | Details |
|--------|---------|
| **Summary** | Perform hybrid vector + graph search |
| **Request Schema** | `HybridSearchRequest` |
| **Key Fields** | `query: str?`, `embedding: list[float]?`, `collection: str = "documents"`, `limit: int = 10`, `alpha: float = 0.7`, `include_graph: bool = True`, `focus_areas: list[str]?`, `tier_filter: list[int]?` |
| **Response Schema** | `HybridSearchResponse` |
| **Response Fields** | `results: list[SearchResultItem]`, `total: int`, `query: str?`, `alpha: float`, `latency_ms: float` |
| **Status Codes** | 200 OK, 400 Bad Request, 422 Validation Error, 500 Internal Error, 503 Service Unavailable |
| **Tags** | `search` |

#### `POST /v1/search`
**File:** [routes.py#L523-L621](src/api/routes.py#L523)

| Aspect | Details |
|--------|---------|
| **Summary** | Perform simple similarity search (no graph) |
| **Request Schema** | `SimpleSearchRequest` |
| **Key Fields** | `query: str` (required), `collection: str = "documents"`, `limit: int = 10`, `min_score: float?` |
| **Response Schema** | `SimpleSearchResponse` |
| **Response Fields** | `results: list[SimpleSearchResultItem]`, `total: int`, `query: str`, `latency_ms: float` |
| **Status Codes** | 200 OK, 400 Bad Request, 422 Validation Error, 500 Internal Error, 503 Service Unavailable |
| **Tags** | `search` |

---

### 2.2 Graph Endpoints

#### `POST /v1/graph/traverse`
**File:** [routes.py#L227-L298](src/api/routes.py#L227)

| Aspect | Details |
|--------|---------|
| **Summary** | Traverse graph from a starting node |
| **Request Schema** | `TraverseRequest` |
| **Key Fields** | `start_node_id: str` (required), `relationship_types: list[str]?`, `max_depth: int = 3`, `limit: int = 50` |
| **Response Schema** | `TraverseResponse` |
| **Response Fields** | `nodes: list[TraverseNode]`, `edges: list[TraverseEdge]`, `start_node: str`, `depth: int`, `latency_ms: float` |
| **Status Codes** | 200 OK, 400 Bad Request, 422 Validation Error, 500 Internal Error, 503 Service Unavailable |
| **Tags** | `graph` |

#### `POST /v1/graph/query`
**File:** [routes.py#L301-L367](src/api/routes.py#L301)

| Aspect | Details |
|--------|---------|
| **Summary** | Execute a read-only Cypher query |
| **Request Schema** | `GraphQueryRequest` |
| **Key Fields** | `cypher: str` (required, read-only enforced), `parameters: dict?`, `timeout_seconds: float = 30.0` |
| **Response Schema** | `GraphQueryResponse` |
| **Response Fields** | `records: list[dict]`, `columns: list[str]`, `latency_ms: float` |
| **Status Codes** | 200 OK, 400 Bad Request, 422 Validation Error, 500 Internal Error, 503 Service Unavailable |
| **Tags** | `graph` |
| **Security** | Write operations (CREATE, DELETE, MERGE, SET, REMOVE, DETACH) are **blocked** by validator |

#### `GET /v1/graph/relationships/{chapter_id}`
**File:** [routes.py#L729-L789](src/api/routes.py#L729)

| Aspect | Details |
|--------|---------|
| **Summary** | Get relationships for a chapter (EEP-4 AC-4.3.1) |
| **Path Params** | `chapter_id: str` |
| **Response Schema** | `ChapterRelationshipsResponse` |
| **Response Fields** | `chapter_id: str`, `relationships: list[RelatedChapterItem]`, `total_count: int` |
| **Status Codes** | 200 OK, 404 Not Found, 500 Internal Error, 503 Service Unavailable |
| **Tags** | `graph` |

#### `POST /v1/graph/relationships/batch`
**File:** [routes.py#L792-L867](src/api/routes.py#L792)

| Aspect | Details |
|--------|---------|
| **Summary** | Get relationships for multiple chapters (EEP-4 AC-4.3.2) |
| **Request Schema** | `BatchRelationshipsRequest` |
| **Key Fields** | `chapter_ids: list[str]` (1-100 items) |
| **Response Schema** | `BatchRelationshipsResponse` |
| **Response Fields** | `results: list[ChapterRelationshipsResponse]`, `total_chapters: int` |
| **Status Codes** | 200 OK, 400 Bad Request, 500 Internal Error, 503 Service Unavailable |
| **Tags** | `graph` |

---

### 2.3 Chapter Endpoints

#### `GET /v1/chapters/{book_id}/{chapter_number}`
**File:** [routes.py#L624-L700](src/api/routes.py#L624)

| Aspect | Details |
|--------|---------|
| **Summary** | Get chapter content by book and chapter number |
| **Path Params** | `book_id: str`, `chapter_number: int` |
| **Response Schema** | `ChapterContentResponse` |
| **Response Fields** | `book_id: str`, `chapter_number: int`, `title: str`, `summary: str`, `keywords: list[str]`, `concepts: list[str]`, `page_range: str`, `found: bool` |
| **Status Codes** | 200 OK, 404 Not Found, 500 Internal Error, 503 Service Unavailable |
| **Tags** | `chapters` |
| **Usage** | Kitchen Brigade: ai-agents (Expeditor) retrieves via semantic-search (Cookbook) |

---

### 2.4 Embedding Endpoint

#### `POST /v1/embed`
**File:** [routes.py#L453-L520](src/api/routes.py#L453)

| Aspect | Details |
|--------|---------|
| **Summary** | Generate embeddings for text |
| **Request Schema** | `EmbedRequest` |
| **Key Fields** | `text: str | list[str]` (required), `model: str?` |
| **Response Schema** | `EmbedResponse` |
| **Response Fields** | `embeddings: list[list[float]]`, `model: str`, `dimensions: int`, `usage: dict?` |
| **Status Codes** | 200 OK, 400 Bad Request, 422 Validation Error, 500 Internal Error, 503 Service Unavailable |
| **Tags** | `embeddings` |
| **WBS Reference** | WBS 0.2.1 |

---

### 2.5 Health Endpoint

#### `GET /health`
**File:** [routes.py#L370-L429](src/api/routes.py#L370)

| Aspect | Details |
|--------|---------|
| **Summary** | Health check endpoint |
| **Response Schema** | `HealthResponse` |
| **Response Fields** | `status: str` ("healthy"/"degraded"), `services: dict[str, str]`, `dependencies: dict[str, str]`, `version: str` |
| **Status Codes** | 200 OK |
| **Tags** | `health` |
| **Dependencies Checked** | `vector` (Qdrant), `graph` (Neo4j), `embedder` |

---

## 3. Interactions (What it Calls)

### 3.1 Database Connections

| Database | Client | Location | Connection Method |
|----------|--------|----------|-------------------|
| **Qdrant** | `qdrant_client.QdrantClient` | [main.py#L47](src/main.py#L47) | `query_points()` API (v1.7+) |
| **Neo4j** | `neo4j.GraphDatabase.driver` | [main.py#L119](src/main.py#L119) | Bolt protocol, sync driver wrapped in `run_in_executor` |

### 3.2 External HTTP Calls

**None.** This service does NOT make outbound HTTP calls. It only receives requests.

### 3.3 Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `QDRANT_URL` | Auto-resolved | `http://localhost:6333` (hybrid mode) | Qdrant server URL |
| `NEO4J_URI` | Auto-resolved | `bolt://localhost:7687` (hybrid mode) | Neo4j Bolt URI |
| `NEO4J_USER` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | No | `devpassword` | Neo4j password |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `USE_FAKE_CLIENTS` | No | `false` | Use mock clients for testing |
| `INFRASTRUCTURE_MODE` | No | `hybrid` (auto-detected) | `docker` / `hybrid` / `native` |
| `SBERT_MODEL` | Yes (in Settings) | N/A | Sentence-BERT model for embeddings |
| `NEO4J_URL` | Yes (in Settings) | N/A | Neo4j URL for Settings class |
| `ENABLE_GRAPH_SEARCH` | No | `true` | Feature flag for graph search |
| `ENABLE_HYBRID_SEARCH` | No | `true` | Feature flag for hybrid search |

### 3.4 Internal Module Dependencies

```
src/main.py
├── src/api/app.create_app()
├── src/api/dependencies.{ServiceContainer, ServiceConfig, ...}
├── src/infrastructure_config.{get_infrastructure_urls, get_infrastructure_mode}
└── (Runtime: qdrant_client, neo4j, sentence_transformers)

src/api/routes.py
├── src/api/dependencies.ServiceContainer
├── src/api/models.* (all request/response models)
└── src/search/metadata_filter.create_filter

src/search/hybrid.py
├── src/search/vector.SearchResult
└── (Protocols for clients)

src/retrievers/*.py
├── langchain_core.{BaseRetriever, Document}
└── src/retrievers/exceptions.*
```

---

## 4. Reverse References (Who Calls This Service)

### 4.1 ai-agents Service

| Caller | File | Usage |
|--------|------|-------|
| `KitchenBrigadeExecutor` | [kitchen_brigade_executor.py#L652](../ai-agents/src/protocols/kitchen_brigade_executor.py#L652) | `POST /v1/search/hybrid` |
| `KitchenBrigadeExecutor` | [kitchen_brigade_executor.py#L719](../ai-agents/src/protocols/kitchen_brigade_executor.py#L719) | `POST /v1/graph/query` |
| `SemanticSearchClient` | ai-agents/src/core/clients/semantic_search.py | Various endpoints |

### 4.2 search-orchestrator Service

| Caller | File | Usage |
|--------|------|-------|
| `VectorRetriever` | [vector_retriever.py](../search-orchestrator/src/retrievers/vector_retriever.py) | `POST /v1/search` (delegated semantic search) |

### 4.3 Platform Configuration

| File | Reference |
|------|-----------|
| [start_platform.sh](../ai-platform-data/start_platform.sh#L147) | Starts service on port 8081 |
| [start_hybrid.sh](../ai-platform-data/start_hybrid.sh) | Hybrid mode startup |

---

## 5. Issues/Observations

### 5.1 Missing Tests

| Area | Status | Notes |
|------|--------|-------|
| `src/graph/relationships.py` | ⚠️ Partial | Large file (643 lines), needs comprehensive edge case coverage |
| `src/search/ranker.py` | ⚠️ Partial | RRF and max strategies need more tests |
| `src/graph/traversal.py` | ⚠️ Partial | BFS/DFS traversal needs boundary testing |

### 5.2 Unclear Contracts

| Issue | Location | Recommendation |
|-------|----------|----------------|
| `get_relationship_scores()` returns empty | [main.py#L227-L232](src/main.py#L227) | Placeholder implementation - document or implement |
| `timeout` parameter naming | `execute_query(timeout=)` vs `query_timeout` | Standardize parameter naming |

### 5.3 Hardcoded Values

| Value | Location | Issue |
|-------|----------|-------|
| `devpassword` | [main.py#L329](src/main.py#L329) | Default Neo4j password in code |
| `"documents"` | [models.py#L32](src/api/models.py#L32) | Default collection name |
| `"chapters"` | [dependencies.py#L83](src/api/dependencies.py#L83) | Default collection in ServiceConfig |

### 5.4 Error Handling Gaps

| Issue | Location | Recommendation |
|-------|----------|----------------|
| Silent Qdrant failures | [main.py#L79](src/main.py#L79) | `search()` returns empty list on error - consider raising |
| Generic exception catching | Multiple | Some `except Exception` blocks could be more specific |
| No retry logic | Graph/Vector clients | Consider circuit breaker pattern |

### 5.5 Architecture Notes

| Observation | Impact |
|-------------|--------|
| **Sync Neo4j driver** wrapped in `run_in_executor` | Works but adds overhead; consider `neo4j.AsyncDriver` |
| **Embedding model loaded at startup** | ~80MB memory, 2-3 second cold start |
| **No connection pooling for Qdrant** | New client per health check in some paths |
| **Feature flags in Settings** | `enable_graph_search`, `enable_hybrid_search` - not hot-reloadable |

### 5.6 Security Considerations

| Item | Status |
|------|--------|
| Cypher injection | ✅ Blocked (parameterized queries + write keyword validation) |
| CORS | ⚠️ Wide open (`allow_origins=["*"]`) - tighten for production |
| Neo4j credentials | ⚠️ Defaults in code - use secrets management |

---

## 6. Dependencies (External Libraries)

### 6.1 Core Dependencies

```python
# requirements.txt
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Database clients
qdrant-client>=1.7.0
neo4j>=5.0.0

# ML/Embeddings
sentence-transformers>=2.2.0

# LangChain (for retrievers)
langchain-core>=0.1.0
```

### 6.2 Development Dependencies

```python
# requirements-dev.txt
pytest>=7.0.0
pytest-asyncio>=0.21.0
httpx>=0.24.0  # Test client
ruff>=0.1.0  # Linting
```

---

## 7. Configuration Files

| File | Purpose |
|------|---------|
| [.env.example](.env.example) | Environment variable template |
| [config/domain_taxonomy.json](config/domain_taxonomy.json) | Domain filtering configuration |
| [pyproject.toml](pyproject.toml) | Project metadata, ruff config |
| [sonar-project.properties](sonar-project.properties) | SonarQube settings |

---

## 8. Quick Reference

### Start Service

```bash
# Hybrid mode (recommended for development)
cd /Users/kevintoles/POC/semantic-search-service
source .venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8081
```

### Health Check

```bash
curl http://localhost:8081/health
```

### Sample Hybrid Search

```bash
curl -X POST http://localhost:8081/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning algorithms", "limit": 10}'
```

### Sample Graph Query

```bash
curl -X POST http://localhost:8081/v1/graph/query \
  -H "Content-Type: application/json" \
  -d '{"cypher": "MATCH (n:Chapter) RETURN n.title LIMIT 5"}'
```
