# Semantic Search Microservice

## Overview

The Semantic Search Service is a **microservice** that provides embedding generation, vector search, and topic modeling capabilities. It exposes REST APIs consumed by other microservices and applications.

## Architecture Type

**Microservice** - Independently deployable, stateless (indices on disk/S3), horizontally scalable for search (single writer for indexing).

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
│   │   └── test_topics/
│   ├── integration/
│   │   ├── test_search_api.py
│   │   └── test_embed_api.py
│   └── conftest.py
│
├── data/
│   ├── indices/                     # FAISS indices (gitignored)
│   ├── models/                      # Cached SBERT models (gitignored)
│   └── topics/                      # Gensim topic models (gitignored)
│
├── docs/
│   ├── ARCHITECTURE.md              # This file
│   ├── API.md
│   └── INDEXING.md                  # How to build indices
│
├── scripts/
│   ├── start.sh
│   ├── build_index.py               # CLI for index building
│   └── train_topics.py              # CLI for topic model training
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
| POST | `/v1/search/hybrid` | **NEW** - Combined vector + graph search |
| POST | `/v1/graph/traverse` | **NEW** - Spider web graph traversal |
| POST | `/v1/graph/query` | **NEW** - Raw Cypher query execution |
| POST | `/v1/topics/infer` | Infer topics for text |
| GET | `/v1/topics/{model_id}/topics` | List all topics |
| POST | `/v1/topics/similar` | Find docs with similar topics |
| POST | `/v1/indices` | Create new index |
| POST | `/v1/indices/{id}/vectors` | Add vectors to index |
| GET | `/v1/indices/{id}/stats` | Get index statistics |
| DELETE | `/v1/indices/{id}` | Delete index |
| GET | `/v1/chunks/{chunk_id}` | Get chunk text by ID |
| GET | `/v1/chunks/{chunk_id}/context` | Get surrounding chunks |
| GET | `/health` | Health check |

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
