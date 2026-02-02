# Semantic Search Microservice

> **Version:** 2.1.0  
> **Updated:** 2026-02-01  
> **Status:** Active  
> **Role:** Kitchen Brigade "Cookbook" - Dumb Retrieval Service  
> **Git Reference:** `12634ed` - `910153d` (Jan 2026 updates)

## Overview

The Semantic Search Service is a **microservice** that provides embedding generation, vector search, and topic modeling capabilities. It exposes REST APIs consumed by other microservices and applications.

## Architecture Type

**Microservice** - Independently deployable, stateless (indices on disk/S3), horizontally scalable for search (single writer for indexing).

---

## ⚠️ Gateway-First Communication Pattern

**CRITICAL RULE**: External applications MUST access platform services through the Gateway.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SERVICE COMMUNICATION PATTERN                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXTERNAL → semantic-search: Via Gateway:8080 (REQUIRED)                    │
│  ─────────────────────────────────────────────────────                       │
│  Applications outside the AI Platform must route through Gateway.           │
│                                                                              │
│  ✅ llm-document-enhancer → Gateway:8080 → semantic-search:8081             │
│  ❌ llm-document-enhancer → semantic-search:8081 (VIOLATION!)               │
│                                                                              │
│  INTERNAL (Platform Services): Direct calls allowed                          │
│  ───────────────────────────────────────────────────                         │
│  Platform services (ai-agents, Code-Orchestrator) may call directly.        │
│                                                                              │
│  ✅ ai-agents:8082 → semantic-search:8081 (internal)                        │
│  ✅ Code-Orchestrator:8083 → semantic-search:8081 (internal)                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Kitchen Brigade Role: COOKBOOK (DUMB RETRIEVAL)

In the Kitchen Brigade architecture, **semantic-search-service** is the **Cookbook** - a dumb retrieval system:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     📖 COOKBOOK - INTENTIONALLY DUMB                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WHAT IT DOES:                                                               │
│  ─────────────                                                               │
│  ✓ Receives keywords/queries as INPUT (does NOT generate them)              │
│  ✓ Queries Qdrant vector DB and Neo4j graph DB                              │
│  ✓ Returns ALL matches without filtering or judgment                        │
│  ✓ Just looks up "recipes" in the "cookbook"                                │
│                                                                              │
│  WHAT IT DOES NOT DO:                                                        │
│  ────────────────────                                                        │
│  ✗ Generate search terms (that's Code-Orchestrator-Service)                 │
│  ✗ Filter or rank results (that's Code-Orchestrator-Service curation)       │
│  ✗ Make semantic judgments (e.g., "chunking" = LLM context)                 │
│  ✗ Host HuggingFace models (that's Code-Orchestrator-Service)               │
│                                                                              │
│  WHY DUMB IS GOOD:                                                           │
│  ─────────────────                                                           │
│  • Single responsibility (just retrieval)                                   │
│  • Easy to test (input → output, no complex logic)                          │
│  • Horizontally scalable (no state, no model loading)                       │
│  • Intelligence is centralized in Sous Chef                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Code-Orchestrator-Service (Sous Chef)
    │
    │ Extracted keywords: ["chunking", "RAG", "embedding", "overlap"]
    ▼
┌─────────────────────────────────────────────────────────────────┐
│             Semantic Search Service (This Service)              │
│                                                                 │
│  POST /v1/search                                                │
│  {                                                              │
│    "keywords": ["chunking", "RAG", "embedding"],               │
│    "top_k": 20                                                  │
│  }                                                              │
│                                                                 │
│  Internal:                                                      │
│  ├── Qdrant: Vector similarity search                          │
│  ├── Neo4j: Graph traversal (optional)                         │
│  └── Hybrid: Combine results                                   │
│                                                                 │
│  Returns: ALL matches (no filtering)                           │
│  [                                                              │
│    {book: "AI Engineering", chapter: 5, score: 0.91},         │
│    {book: "C++ Concurrency", chapter: 3, score: 0.45}, ← wrong│
│    {book: "Building LLM Apps", chapter: 8, score: 0.88},      │
│    ...                                                          │
│  ]                                                              │
└─────────────────────────────────────────────────────────────────┘
    │
    │ Raw results (may include false positives like C++ memory chunks)
    ▼
Code-Orchestrator-Service (Chef de Partie - Curation)
    │
    │ Filtered/ranked results (C++ filtered out)
    ▼
Consumer (ai-agents, llm-document-enhancer)
```

---

## Folder Structure

```
semantic-search-service/
├── src/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app entry point + lifespan handler
│   ├── infrastructure_config.py     # Dynamic infrastructure URL resolution
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                   # Application factory (create_app)
│   │   ├── dependencies.py          # ServiceContainer DI
│   │   ├── models.py                # Pydantic request/response models
│   │   └── routes.py                # All API endpoints (single file)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Pydantic settings (env-based)
│   │   └── logging.py               # Structured JSON logging (WBS-LOG0)
│   │
│   ├── graph/                       # Neo4j Graph Layer
│   │   ├── __init__.py
│   │   ├── exceptions.py            # Graph-specific exceptions
│   │   ├── health.py                # Neo4j health checks
│   │   ├── neo4j_client.py          # Async Neo4j driver wrapper
│   │   ├── relationships.py         # EEP-4 relationship types
│   │   ├── schema.py                # Graph schema definitions
│   │   ├── taxonomy_loader.py       # Taxonomy JSON loading
│   │   └── traversal.py             # Spider web traversal (BFS/DFS)
│   │
│   ├── retrievers/                  # Retriever Abstractions (Phase 4)
│   │   ├── __init__.py
│   │   ├── exceptions.py            # Retriever exceptions
│   │   ├── hybrid_retriever.py      # Combined vector + graph retriever
│   │   ├── neo4j_retriever.py       # Neo4j graph retriever
│   │   └── qdrant_retriever.py      # Qdrant vector retriever
│   │
│   ├── search/                      # Search Components
│   │   ├── __init__.py
│   │   ├── exceptions.py            # Search exceptions
│   │   ├── hybrid.py                # Hybrid search logic
│   │   ├── metadata_filter.py       # Domain-aware filtering
│   │   ├── ranker.py                # Score fusion (LINEAR, RRF, MAX)
│   │   └── vector.py                # Vector search utilities
│   │
│   └── vector/                      # Qdrant Vector Layer
│       ├── __init__.py
│       └── health.py                # Qdrant health checks
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── fakes.py                     # Test doubles (CL-017 refactor)
│   ├── benchmark/
│   │   └── test_performance.py      # WBS 6.1 performance tests
│   ├── integration/
│   │   ├── test_hybrid_api.py
│   │   └── test_retrievers_integration.py
│   ├── unit/
│   │   ├── test_chapter_api.py      # Kitchen Brigade chapter tests
│   │   ├── test_embed_api.py        # WBS 0.2.1 tests
│   │   ├── test_metadata_filter.py
│   │   ├── test_neo4j_health.py
│   │   ├── test_qdrant_health.py
│   │   ├── test_search_api.py       # WBS 0.2.2 tests
│   │   ├── test_taxonomy_loader.py
│   │   ├── test_graph/
│   │   │   ├── test_eep4_relationships.py
│   │   │   ├── test_schema.py
│   │   │   └── test_traversal.py
│   │   ├── test_retrievers/
│   │   │   ├── test_hybrid_retriever.py
│   │   │   ├── test_neo4j_retriever.py
│   │   │   └── test_qdrant_retriever.py
│   │   └── test_search/
│   │       └── test_ranker.py
│   └── validation/
│       ├── test_citation_accuracy.py   # WBS 6.3
│       └── test_spider_web_coverage.py # WBS 6.2
│
├── docs/
│   ├── ARCHITECTURE.md              # This file
│   ├── TECHNICAL_CHANGE_LOG.md      # Change history
│   ├── openapi.yaml                 # OpenAPI spec
│   └── reports/                     # Generated reports
│       ├── BENCHMARK_REPORT.md      # WBS 6.1 deliverable
│       ├── SPIDER_WEB_COVERAGE_REPORT.md
│       └── CITATION_ACCURACY_REPORT.md
│
├── scripts/
│   ├── demo_graph_rag.py            # Demo script
│   ├── generate_benchmark_report.py # WBS 6.1
│   ├── generate_citation_accuracy_report.py  # WBS 6.3
│   ├── validate_0.2.1_embed.sh      # Validation: embed endpoint
│   ├── validate_0.2.2_search.sh     # Validation: search endpoint
│   ├── validate_0.2.3_real_clients.sh # Validation: real backends
│   └── validate_domain_filter.py    # Domain filter validation
│
├── config/
│   └── domain_taxonomy.json         # Domain filtering config
│
├── Dockerfile
├── docker-compose.yml               # Tiered network (CL-015)
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── start_service.sh                 # Convenience startup script
└── SEMANTIC_SEARCH_SERVICE_INVENTORY.md
```

---

## System Context

```
                          ┌─────────────────────────────────────────┐
                          │            CONSUMERS                     │
                          │                                          │
                          │  ┌────────────┐  ┌────────────────────┐ │
                          │  │ llm-gateway│  │ ai-agents          │ │
                          │  │ (tools)    │  │ (Kitchen Brigade)  │ │
                          │  └─────┬──────┘  └─────────┬──────────┘ │
                          │        │                   │            │
                          │        │   ┌───────────────┘            │
                          │        │   │  ┌────────────────────┐   │
                          │        │   │  │ Code-Orchestrator  │   │
                          │        │   │  │ (Sous Chef)        │   │
                          │        │   │  └─────────┬──────────┘   │
                          └────────┼───┼────────────┼──────────────┘
                                   │   │            │
                                   ▼   ▼            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     SEMANTIC SEARCH MICROSERVICE                              │
│                           (Port 8081)                                         │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           API Layer (FastAPI)                            │ │
│  │  POST /v1/embed │ POST /v1/search │ POST /v1/search/hybrid │ GET /health│ │
│  │  POST /v1/graph/* │ GET /v1/chapters/* │ GET /v1/graph/relationships/*  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                        │
│  ┌──────────────┐  ┌──────────────┐  │  ┌──────────────┐                     │
│  │  Embedding   │  │   Vector     │  │  │   Graph      │                     │
│  │   Service    │  │   Client     │  │  │   Client     │                     │
│  │              │  │              │  │  │              │                     │
│  │ • SBERT      │  │ • Qdrant     │  │  │ • Neo4j      │                     │
│  │ • MiniLM     │  │ • Hybrid     │  │  │ • Traversal  │                     │
│  └──────┬───────┘  └──────┬───────┘  │  └──────┬───────┘                     │
│         │                 │          │         │                              │
└─────────┼─────────────────┼──────────┼─────────┼──────────────────────────────┘
          │                 │          │         │
          ▼                 ▼          │         ▼
┌──────────────────┐ ┌─────────────────┐│  ┌─────────────────┐
│ HuggingFace Hub  │ │ Qdrant          ││  │ Neo4j           │
│ (SBERT models)   │ │ (ai-platform-)  ││  │ (ai-platform-)  │
└──────────────────┘ └─────────────────┘│  └─────────────────┘
                                        │
```

---

## API Endpoints

| Method | Endpoint | Description | WBS Ref |
|--------|----------|-------------|---------|
| POST | `/v1/embed` | Generate embeddings for text(s) | WBS 0.2.1 |
| POST | `/v1/search` | Simple vector similarity search | WBS 0.2.2 |
| POST | `/v1/search/hybrid` | Combined vector + graph search (accepts `taxonomy`, `focus_areas`) | Phase 3 |
| POST | `/v1/graph/traverse` | Spider web graph traversal (BFS/DFS) | Phase 2 |
| POST | `/v1/graph/query` | Execute read-only Cypher queries | Phase 2 |
| GET | `/v1/chapters/{book_id}/{chapter_number}` | Get chapter content (Kitchen Brigade) | CL-017 |
| GET | `/v1/graph/relationships/{chapter_id}` | Get relationships for chapter | EEP-4 |
| POST | `/v1/graph/relationships/batch` | Batch relationship queries | EEP-4 |
| GET | `/health` | Health check with dependency status | WBS 0.2.3 |

### Removed/Not Implemented Endpoints

The following endpoints from the original design were **not implemented** (functionality absorbed elsewhere or deferred):

| Endpoint | Status | Reason |
|----------|--------|--------|
| `/v1/embed/batch` | Not implemented | Single `/v1/embed` accepts lists |
| `/v1/search/vector` | Not implemented | Use `/v1/search` with query |
| `/v1/topics/*` | Not implemented | Topic modeling deferred |
| `/v1/indices/*` | Not implemented | Index management via scripts |
| `/v1/chunks/*` | Not implemented | Chunk storage in Qdrant payloads |
| `/v1/taxonomies` | Not implemented | Taxonomies loaded from files |

---

## Taxonomy-Agnostic Architecture

> **Key Principle**: Taxonomies are query-time overlays, NOT baked into seeded data.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TAXONOMY AS QUERY-TIME OVERLAY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SEEDED DATA (One-time, taxonomy-agnostic):                                 │
│  • Qdrant vectors: content embeddings + enriched payloads (NO tier)         │
│  • Neo4j nodes: Book/Chapter structure (NO tier baked in)                   │
│                                                                              │
│  QUERY FLOW:                                                                 │
│  ───────────                                                                 │
│  POST /v1/search/hybrid                                                      │
│  {                                                                           │
│    "query": "rate limiting patterns",                                       │
│    "taxonomy": "AI-ML_taxonomy",    ← Optional: loaded at query time        │
│    "tier_filter": [1, 2]            ← Optional: filter by tier              │
│  }                                                                           │
│                                                                              │
│  1. Search Qdrant (taxonomy-agnostic vectors)                               │
│  2. Load taxonomy from ai-platform-data/taxonomies/ (if specified)          │
│  3. Apply tier mapping to results (query-time overlay)                      │
│  4. Filter by tier_filter (if specified)                                     │
│  5. Return results with tier/priority attached                               │
│                                                                              │
│  BENEFITS:                                                                   │
│  • Adding new taxonomy = just add JSON file (NO re-seeding!)                │
│  • Same book can have different tiers in different taxonomies               │
│  • Users specify taxonomy at runtime via prompt/API                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Search Response Examples

**Without taxonomy** (returns all results, no tier info):
```json
{
  "results": [
    {"book": "Building Microservices", "chapter": 5, "score": 0.91},
    {"book": "AI Engineering", "chapter": 3, "score": 0.88}
  ]
}
```

**With taxonomy** (tier/priority from specified taxonomy):
```json
{
  "results": [
    {"book": "Building Microservices", "chapter": 5, "score": 0.91, "tier": 1, "priority": 6},
    {"book": "AI Engineering", "chapter": 3, "score": 0.88, "tier": 1, "priority": 3}
  ]
}
```

---

## Enrichment Scalability Architecture

> **Key Principle**: Cross-book similarity (`similar_chapters`) is computed against the FULL corpus, then filtered at query-time by taxonomy.

### Problem (Pre-v1.4.0)

```
similar_chapters computed per taxonomy
    ↓
47 books × 1000 taxonomies = 47,000 enriched files (doesn't scale!)
    ↓
Adding new book = O(n² × t) re-enrichment
```

### Solution (v1.4.0)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            COMPUTE ONCE AGAINST FULL CORPUS, FILTER AT QUERY-TIME           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ENRICHMENT (One-time, O(n²)):                                              │
│  • `similar_chapters` computed against ALL 47+ books                        │
│  • Single enriched file per book, shared across all taxonomies              │
│  • Stored in Qdrant payload (no taxonomy info)                               │
│                                                                              │
│  QUERY-TIME FILTERING:                                                       │
│  ─────────────────────                                                       │
│  POST /v1/search/similar-chapters                                            │
│  {                                                                           │
│    "chapter_id": "arch_patterns_ch4",                                       │
│    "taxonomy": "AI-ML_taxonomy"    ← Filter by books in this taxonomy       │
│  }                                                                           │
│                                                                              │
│  1. Retrieve similar_chapters from Qdrant (all books)                       │
│  2. Load taxonomy from ai-platform-data/taxonomies/                          │
│  3. Filter similar_chapters to only books IN the taxonomy                   │
│  4. Attach tier/priority from taxonomy                                       │
│  5. Return filtered results                                                  │
│                                                                              │
│  INCREMENTAL UPDATE (Adding New Book):                                       │
│  ─────────────────────────────────────                                       │
│  1. Enrich new book against existing corpus (O(n))                          │
│  2. Append new book to existing books' similar_chapters                     │
│  3. Use Qdrant set_payload() for atomic updates                              │
│  4. NO full re-enrichment required!                                          │
│                                                                              │
│  COMPLEXITY:                                                                 │
│  │ Operation              │ Before        │ After          │                │
│  ├────────────────────────┼───────────────┼────────────────┤                │
│  │ Full enrichment        │ O(n² × t)     │ O(n²)          │                │
│  │ Add new taxonomy       │ O(n²)         │ O(1)           │                │
│  │ Add new book           │ O(n² × t)     │ O(n)           │                │
│  └────────────────────────┴───────────────┴────────────────┘                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Similar Chapters API

```python
# Endpoint: POST /v1/search/similar-chapters
{
    "chapter_id": "arch_patterns_ch4_abc123",
    "taxonomy": "AI-ML_taxonomy",    # Optional: filter by taxonomy
    "tier_filter": [1, 2],           # Optional: only certain tiers
    "limit": 10
}

# Response with taxonomy filter
{
    "similar_chapters": [
        {"chapter_id": "...", "book": "Building Microservices", "score": 0.91, "tier": 1},
        {"chapter_id": "...", "book": "Clean Architecture", "score": 0.88, "tier": 2}
    ],
    "total_unfiltered": 47,  # Total before taxonomy filter
    "filtered_by": "AI-ML_taxonomy"
}

# Response without taxonomy filter
{
    "similar_chapters": [
        {"chapter_id": "...", "book": "Building Microservices", "score": 0.91},
        {"chapter_id": "...", "book": "Random Book Not In Taxonomy", "score": 0.87}
    ],
    "total_unfiltered": 47
}
```

---

## Components

### Embedding Service (SBERT)
- Loads sentence-transformer models via `SentenceTransformer`
- Configured via `SBERT_MODEL` env var (default: `all-MiniLM-L6-v2`, 384 dims)
- Generates dense vector embeddings
- Batched inference via single `/v1/embed` endpoint accepting lists

### Vector Search (Qdrant)
- Cloud-native vector database (port 6333)
- Metadata filtering with search
- Top-k retrieval with scores
- Payload storage for chapter metadata
- **Atomic payload updates** via `set_payload()` for incremental enrichment
- Health checks via `src/vector/health.py`

### Graph Engine (Neo4j)
- Knowledge graph storage (port 7687)
- Cypher query execution
- Spider web traversal (PARALLEL, PERPENDICULAR, SKIP_TIER)
- Bidirectional relationship navigation
- EEP-4 relationship types: PARALLEL, PERPENDICULAR, SKIP_TIER, LATERAL
- Health checks via `src/graph/health.py`

### Hybrid Search Engine
- Query planner (vector vs graph vs both)
- Domain-aware filtering via `focus_areas` parameter
- Result merger with score fusion
- Re-ranking based on tier relationships
- Deduplication across sources

### Graph Traversal Engine (WBS 6.4)

The graph traversal system implements a "spider web" model for navigating the taxonomy graph:

#### Relationship Types
| Type | Description | Relevance Bonus |
|------|-------------|-----------------|
| PARALLEL | Same-tier relationships (horizontal) | +0.20 |
| PERPENDICULAR | Adjacent-tier relationships (vertical) | +0.10 |
| SKIP_TIER | Non-adjacent tier relationships | +0.00 |
| LATERAL | Cross-branch relationships | +0.05 |

#### Traversal Algorithms
- **BFS Traverse**: Breadth-first search for shortest paths
- **DFS Traverse**: Depth-first search for deep exploration
- **Cross-Reference Path**: Find connections between concepts

#### Relevance Scoring
```python
# Relevance = base_depth_score + relationship_bonus
depth_score = max(0.0, 1.0 - (depth * 0.2))  # Decay by 20% per hop
relevance = min(1.0, depth_score + type_bonus)
```

#### Performance Targets (WBS 6.1 Validated)
| Operation | P95 Target | P95 Actual | Status |
|-----------|------------|------------|--------|
| BFS Traversal | <200ms | 38.39ms | ✅ |
| DFS Traversal | <200ms | 38.27ms | ✅ |
| Hybrid Search | <500ms | 115.22ms | ✅ |
| Score Fusion | <1ms | 0.08ms | ✅ |

### Result Ranker (WBS 6.4)

Implements multiple score fusion strategies for combining vector and graph results:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| LINEAR | Weighted average | General purpose |
| RRF | Reciprocal Rank Fusion | Multi-source ranking |
| MAX | Maximum score wins | High-confidence matches |

```python
# Linear fusion (default)
final_score = (vector_score * vector_weight) + (graph_score * graph_weight)

# RRF fusion
rrf_score = 1 / (k + vector_rank) + 1 / (k + graph_rank)
```

### Citation Accuracy (WBS 6.3 Validated)

Cross-reference citations maintain high relevance:

| Relationship | Target | Achieved | Status |
|--------------|--------|----------|--------|
| PARALLEL (Tier 1) | ≥90% | 100% | ✅ |
| PERPENDICULAR | ≥70% | 90% | ✅ |
| Average Overall | ≥85% | 90% | ✅ |

### Service Container (Dependency Injection)

The `ServiceContainer` pattern provides centralized service management:

```python
# src/api/dependencies.py
class ServiceContainer:
    config: Settings
    embedding_service: EmbeddingService | None
    vector_client: QdrantRetriever | None
    graph_client: Neo4jClient | None
```

Injected at startup via `create_app()` factory with lifespan context manager.

---

## Dependencies

| Dependency | Type | Purpose |
|------------|------|---------|
| HuggingFace Hub | External | SBERT model downloads (startup) |
| Qdrant | Infrastructure | Vector database (`ai-platform-qdrant`) |
| Neo4j | Infrastructure | Knowledge graph (`ai-platform-neo4j`) |
| FastAPI | Framework | REST API framework |
| sentence-transformers | Library | Embedding generation |
| neo4j (driver) | Library | Async Neo4j client |
| qdrant-client | Library | Qdrant REST client |

**Note:** Topic modeling (Gensim) and index management (FAISS) were **not implemented** - functionality absorbed by Qdrant and Neo4j.

---

## Integration Points

> **Important for WBS Planning**: These are the integration points that require coordination with other services.

### Inbound (Services calling semantic-search)

| Consumer | Endpoint | Purpose | Priority |
|----------|----------|---------|----------|
| llm-gateway | `POST /v1/search` | Tool execution (search_corpus) | P0 |
| llm-gateway | `POST /v1/embed` | Generate embeddings for tools | P0 |
| ai-agents | `POST /v1/search/hybrid` | Cross-Reference Agent similarity search | P0 |
| ai-agents | `POST /v1/graph/traverse` | Spider web taxonomy traversal | P0 |
| llm-document-enhancer | `POST /v1/embed` | Pre-compute embeddings | P1 |
| llm-document-enhancer | `POST /v1/search` | Pre-compute matches | P1 |

### Outbound (semantic-search calling other services)

| Target | Protocol | Purpose | Priority |
|--------|----------|---------|----------|
| Qdrant | HTTP (6333) | Vector storage and search | P0 |
| Neo4j | Bolt (7687) | Graph queries and traversal | P0 |
| HuggingFace Hub | HTTPS | Model downloads (startup) | P1 |
| S3 | HTTPS | Index persistence (optional) | P2 |

### Data Dependencies

| Data | Source | Required For |
|------|--------|--------------|
| SBERT models | HuggingFace Hub | Embedding generation |
| Taxonomy graph | Neo4j | Hybrid search, traversal |
| Chapter vectors | Qdrant | Similarity search |
| Chunk metadata | Qdrant payloads | Result enrichment |

---

## Communication Matrix

| From | To | Protocol | Endpoint/Method |
|------|----|----------|-----------------|
| llm-gateway | semantic-search | HTTP | `POST /v1/search` |
| ai-agents | semantic-search | HTTP | `POST /v1/search/hybrid` |
| ai-agents | semantic-search | HTTP | `POST /v1/graph/traverse` |
| llm-doc-enhancer | semantic-search | HTTP | `POST /v1/embed`, `POST /v1/search` |
| semantic-search | Qdrant | HTTP | Qdrant REST API |
| semantic-search | Neo4j | Bolt | Cypher queries |

---

## Deployment

```yaml
# docker-compose.yml
services:
  semantic-search:
    build: .
    ports:
      - "8081:8081"
    volumes:
      - ./data/indices:/data/indices
      - ./data/models:/data/models
      - ./data/topics:/data/topics
    environment:
      - SBERT_MODEL=all-mpnet-base-v2
      - INDEX_STORAGE_PATH=/data/indices
      - MODEL_CACHE_PATH=/data/models
      - QDRANT_URL=http://qdrant:6333
      - NEO4J_URL=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
    depends_on:
      - qdrant
      - neo4j

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}

volumes:
  qdrant_data:
  neo4j_data:
```

---

## Configuration

```python
# src/core/config.py (actual implementation)
class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All database connections REQUIRED via env vars (no hardcoded defaults).
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Service
    semantic_search_port: int = Field(default=8081)
    
    # Embedding (REQUIRED)
    sbert_model: str = Field(description="REQUIRED via SBERT_MODEL env var")
    
    # Neo4j (REQUIRED - no defaults per PCON-7)
    neo4j_url: str = Field(description="REQUIRED via NEO4J_URL")
    neo4j_user: str = Field(description="REQUIRED via NEO4J_USER")
    neo4j_password: str = Field(description="REQUIRED via NEO4J_PASSWORD")
    
    # Qdrant (REQUIRED - no defaults per PCON-7)
    qdrant_url: str = Field(description="REQUIRED via QDRANT_URL")
    
    # Feature Flags (enabled by default after Phase 6 validation)
    enable_graph_search: bool = Field(default=True)
    enable_hybrid_search: bool = Field(default=True)
```

### Environment Variables (Required)

| Variable | Example | Description |
|----------|---------|-------------|
| `SBERT_MODEL` | `all-MiniLM-L6-v2` | Sentence-BERT model name |
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `devpassword` | Neo4j password |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST API URL |

**Note:** Per PCON-7 (Platform Consolidation), no hardcoded database URLs. All connections configured via environment variables. See `docker-compose.yml` for tiered network configuration.

---

## See Also

- [TECHNICAL_CHANGE_LOG.md](./TECHNICAL_CHANGE_LOG.md) - Detailed change history with git commit references
- [openapi.yaml](./openapi.yaml) - OpenAPI specification
- [ai-agents/docs/ARCHITECTURE.md](/ai-agents/docs/ARCHITECTURE.md) - AI Agents service (primary consumer)
- [ai-platform-data/docs/SCHEMA_REFERENCE.md](/ai-platform-data/docs/SCHEMA_REFERENCE.md) - Neo4j schema definitions

### Related Change Log Entries

| Entry | Date | Description |
|-------|------|-------------|
| CL-017 | 2026-01-07 | Neo4j ↔ Qdrant Bridge integration |
| CL-016 | 2026-01-01 | PCON-7 real backend configuration |
| CL-015 | 2025-12-31 | Tiered network docker-compose |
| CL-011 | 2025-12-19 | Gateway-First Communication Pattern |
| CL-009 | 2025-12-13 | Query-time taxonomy filtering |
