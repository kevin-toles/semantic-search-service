# Semantic Search Microservice

## Overview

The Semantic Search Service is a **microservice** that provides embedding generation, vector search, and topic modeling capabilities. It exposes REST APIs consumed by other microservices and applications.

## Architecture Type

**Microservice** - Independently deployable, stateless (indices on disk/S3), horizontally scalable for search (single writer for indexing).

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
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── embed.py             # POST /v1/embed
│   │   │   ├── search.py            # POST /v1/search
│   │   │   ├── topics.py            # /v1/topics/*
│   │   │   ├── indices.py           # /v1/indices/*
│   │   │   ├── chunks.py            # /v1/chunks/*
│   │   │   └── health.py            # /health, /ready
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   └── logging.py
│   │   └── deps.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # Pydantic settings
│   │   └── exceptions.py
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── engine.py                # SBERT embedding engine
│   │   ├── models.py                # Model registry
│   │   └── preprocessor.py          # Text preprocessing
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── faiss_index.py           # FAISS index wrapper
│   │   ├── metadata_filter.py       # Post-search filtering
│   │   └── ranker.py                # Result ranking
│   │
│   ├── graph/                       # NEW - Graph RAG components
│   │   ├── __init__.py
│   │   ├── traversal.py             # Spider web traversal (BFS/DFS)
│   │   └── hybrid_search.py         # Vector + Graph fusion
│   │
│   ├── retrievers/                  # NEW - Retriever abstractions
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract retriever interface
│   │   ├── qdrant_retriever.py      # Qdrant vector retriever
│   │   └── neo4j_retriever.py       # Neo4j graph retriever
│   │
│   ├── topics/
│   │   ├── __init__.py
│   │   ├── lda.py                   # Gensim LDA
│   │   ├── lsi.py                   # Gensim LSI
│   │   └── inference.py             # Topic inference
│   │
│   ├── indices/
│   │   ├── __init__.py
│   │   ├── manager.py               # Index lifecycle management
│   │   ├── storage.py               # Index persistence (local/S3)
│   │   └── chunks.py                # Chunk text storage
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── domain.py
│   │
│   └── main.py                      # FastAPI app entry point
│
├── tests/
│   ├── unit/
│   │   ├── test_embedding/
│   │   ├── test_search/
│   │   ├── test_graph/              # NEW - Graph traversal tests
│   │   ├── test_retrievers/         # NEW - Retriever tests
│   │   └── test_topics/
│   ├── integration/
│   │   ├── test_search_api.py
│   │   ├── test_embed_api.py
│   │   └── test_retrievers_integration.py  # NEW
│   ├── benchmark/                   # NEW - Performance benchmarks (WBS 6.1)
│   │   └── test_performance.py
│   ├── validation/                  # NEW - Validation tests (WBS 6.2, 6.3)
│   │   ├── test_spider_web_coverage.py
│   │   └── test_citation_accuracy.py
│   └── conftest.py
│
├── docs/
│   ├── ARCHITECTURE.md              # This file
│   ├── API.md
│   ├── GRAPH_RAG_POC.md             # Graph RAG design document
│   ├── INDEXING.md                  # How to build indices
│   └── reports/                     # NEW - Generated reports
│       ├── BENCHMARK_REPORT.md      # WBS 6.1 deliverable
│       ├── SPIDER_WEB_COVERAGE_REPORT.md  # WBS 6.2 deliverable
│       └── CITATION_ACCURACY_REPORT.md    # WBS 6.3 deliverable
│
├── scripts/
│   ├── start.sh
│   ├── build_index.py               # CLI for index building
│   ├── train_topics.py              # CLI for topic model training
│   ├── generate_benchmark_report.py # NEW - WBS 6.1
│   └── generate_citation_accuracy_report.py  # NEW - WBS 6.3
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## System Context

```
                          ┌─────────────────────────────────────────┐
                          │            CONSUMERS                     │
                          │                                          │
                          │  ┌────────────┐  ┌────────────────────┐ │
                          │  │ llm-gateway│  │ llm-doc-enhancer   │ │
                          │  │ (tools)    │  │ (pre-compute)      │ │
                          │  └─────┬──────┘  └─────────┬──────────┘ │
                          │        │                   │            │
                          │        │   ┌───────────────┘            │
                          │        │   │  ┌────────────────────┐   │
                          │        │   │  │ ai-agents          │   │
                          │        │   │  │ (code similarity)  │   │
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
│  │  POST /v1/embed  │  POST /v1/search  │  POST /v1/topics/*  │  GET /health│ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                        │
│  ┌──────────────┐  ┌──────────────┐  │  ┌──────────────┐  ┌──────────────┐   │
│  │  Embedding   │  │   Vector     │  │  │   Topic      │  │    Index     │   │
│  │   Engine     │  │   Search     │  │  │   Modeler    │  │   Manager    │   │
│  │              │  │              │  │  │              │  │              │   │
│  │ • SBERT      │  │ • FAISS      │  │  │ • LDA        │  │ • Create     │   │
│  │ • MiniLM     │  │ • Flat/IVF   │  │  │ • LSI        │  │ • Add/Delete │   │
│  │ • mpnet      │  │ • HNSW       │  │  │ • Doc2Vec    │  │ • Rebuild    │   │
│  └──────┬───────┘  └──────┬───────┘  │  └──────┬───────┘  └──────┬───────┘   │
│         │                 │          │         │                 │            │
└─────────┼─────────────────┼──────────┼─────────┼─────────────────┼────────────┘
          │                 │          │         │                 │
          ▼                 ▼          │         ▼                 ▼
┌──────────────────┐ ┌─────────────────┐│  ┌─────────────────┐ ┌─────────────────┐
│ HuggingFace Hub  │ │ FAISS Indices   ││  │ Gensim Models   │ │ Chunk Storage   │
│ (SBERT models)   │ │ (local/S3)      ││  │ (local/S3)      │ │ (local/S3)      │
└──────────────────┘ └─────────────────┘│  └─────────────────┘ └─────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/embed` | Generate embeddings for texts |
| POST | `/v1/embed/batch` | Async batch embedding job |
| POST | `/v1/search` | Semantic similarity search |
| POST | `/v1/search/vector` | Search by raw vector |
| POST | `/v1/search/hybrid` | Combined vector + graph search (accepts `taxonomy` param) |
| POST | `/v1/graph/traverse` | Spider web graph traversal |
| POST | `/v1/graph/query` | Raw Cypher query execution |
| POST | `/v1/topics/infer` | Infer topics for text |
| GET | `/v1/topics/{model_id}/topics` | List all topics |
| POST | `/v1/topics/similar` | Find docs with similar topics |
| POST | `/v1/indices` | Create new index |
| POST | `/v1/indices/{id}/vectors` | Add vectors to index |
| GET | `/v1/indices/{id}/stats` | Get index statistics |
| DELETE | `/v1/indices/{id}` | Delete index |
| GET | `/v1/chunks/{chunk_id}` | Get chunk text by ID |
| GET | `/v1/chunks/{chunk_id}/context` | Get surrounding chunks |
| GET | `/v1/taxonomies` | List available taxonomies |
| GET | `/health` | Health check |

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

## Components

### Embedding Engine (SBERT)
- Loads sentence-transformer models
- Generates dense vector embeddings
- Supports multiple models (MiniLM, mpnet)
- Batched inference for efficiency

### Vector Search (Qdrant)
- Cloud-native vector database
- Metadata filtering with search
- Top-k retrieval with scores
- Payload storage for chunk metadata

### Graph Engine (Neo4j) - NEW
- Taxonomy graph storage
- Cypher query execution
- Spider web traversal (PARALLEL, PERPENDICULAR, SKIP_TIER)
- Bidirectional relationship navigation

### Hybrid Search Engine - NEW
- Query planner (vector vs graph vs both)
- Result merger with score fusion
- Re-ranking based on tier relationships
- Deduplication across sources

### Graph Traversal Engine - NEW (WBS 6.4)

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

### Result Ranker - NEW (WBS 6.4)

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

### Topic Modeler (Gensim)
- LDA for topic discovery
- LSI for semantic similarity
- Topic inference for new documents

### Index Manager
- Index lifecycle (create, add, delete, rebuild)
- Persistence to local disk or S3
- Blue-green deployment for index updates

---

## Dependencies

| Dependency | Type | Purpose |
|------------|------|---------|
| HuggingFace Hub | External | SBERT model downloads |
| Qdrant | Infrastructure | Vector database (replaces FAISS) |
| Neo4j | Infrastructure | Taxonomy graph database |
| S3 (optional) | Infrastructure | Index/model storage |

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
# src/core/config.py
class Settings(BaseSettings):
    # Service
    service_name: str = "semantic-search-service"
    port: int = 8081
    
    # Embedding
    sbert_model: str = "all-mpnet-base-v2"
    embedding_batch_size: int = 32
    
    # Storage
    index_storage_path: str = "/data/indices"
    model_cache_path: str = "/data/models"
    topic_model_path: str = "/data/topics"
    
    # Qdrant (NEW)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chapters"
    
    # Neo4j (NEW)
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    
    # Hybrid Search (NEW)
    hybrid_default_vector_weight: float = 0.6
    hybrid_default_graph_weight: float = 0.4
    hybrid_max_traversal_hops: int = 5
    
    # Optional S3
    s3_bucket: Optional[str] = None
    
    class Config:
        env_prefix = "SEMANTIC_SEARCH_"
```

---

## See Also

- [GRAPH_RAG_POC.md](./GRAPH_RAG_POC.md) - Graph-augmented semantic search POC
- [API.md](./API.md) - Full API documentation
- [INDEXING.md](./INDEXING.md) - How to build indices
- [ai-agents/docs/ARCHITECTURE.md](/ai-agents/docs/ARCHITECTURE.md) - AI Agents service (primary consumer)
