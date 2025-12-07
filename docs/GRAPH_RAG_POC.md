# Graph-Augmented Semantic Search POC

## Overview

This document outlines a **Proof of Concept (POC)** for extending the semantic-search-service to support **Graph RAG** (Retrieval-Augmented Generation). The goal is to combine vector similarity search (Qdrant) with graph-based relationship traversal (Neo4j) to enable the spider web taxonomy model.

> ⚠️ **POC Document** - This describes proposed architecture changes. The main `ARCHITECTURE.md` remains unchanged until this POC is validated.

---

## Problem Statement

The current semantic-search-service provides **flat vector search**:

```
Query → Embed → Vector Search → Top-K Results
```

This works well for similarity but **lacks**:
1. **Relationship awareness** - No concept of tiers, categories, or taxonomy structure
2. **Graph traversal** - Cannot follow PARALLEL/PERPENDICULAR/SKIP_TIER edges
3. **Multi-hop reasoning** - Each search is independent, no path-based retrieval
4. **Bidirectional navigation** - Cannot traverse from implementation → theory or vice versa

---

## Proposed Solution: Hybrid Architecture

Combine Qdrant (vector) + Neo4j (graph) with a unified query interface:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     SEMANTIC-SEARCH-SERVICE (Enhanced)                           │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                          API Layer (FastAPI)                                │ │
│  │                                                                             │ │
│  │  POST /v1/search          → Existing: Pure vector search                   │ │
│  │  POST /v1/search/hybrid   → NEW: Vector + graph combined                   │ │
│  │  POST /v1/graph/traverse  → NEW: Pure graph traversal                      │ │
│  │  POST /v1/graph/query     → NEW: Cypher queries                            │ │
│  │                                                                             │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                       │                                          │
│                    ┌──────────────────┼──────────────────┐                      │
│                    ▼                  ▼                  ▼                      │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐       │
│  │   Vector Engine      │  │   Graph Engine   │  │   Hybrid Engine      │       │
│  │                      │  │                  │  │                      │       │
│  │  • Qdrant client     │  │  • Neo4j client  │  │  • Query planner     │       │
│  │  • Embeddings        │  │  • Cypher gen    │  │  • Result merger     │       │
│  │  • Similarity        │  │  • Traversal     │  │  • Re-ranking        │       │
│  │  • Filtering         │  │  • Path finding  │  │  • Score fusion      │       │
│  └──────────────────────┘  └──────────────────┘  └──────────────────────────┘   │
│            │                        │                        │                  │
└────────────┼────────────────────────┼────────────────────────┼──────────────────┘
             │                        │                        │
             ▼                        ▼                        ▼
     ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
     │   Qdrant     │         │    Neo4j     │         │  Result Set  │
     │              │         │              │         │              │
     │  Vectors     │         │  Taxonomy    │         │  Unified     │
     │  Embeddings  │         │  Graph       │         │  Response    │
     │              │         │  Edges       │         │              │
     └──────────────┘         └──────────────┘         └──────────────┘
```

---

## Data Model

### Neo4j Graph Schema

```cypher
// Nodes
(:Book {
    id: string,
    title: string,
    tier: integer,
    category: string,
    author: string,
    year: integer
})

(:Chapter {
    id: string,
    book_id: string,
    number: integer,
    title: string,
    keywords: [string],
    concepts: [string],
    summary: string,
    page_range: string
})

(:Taxonomy {
    id: string,
    name: string,
    description: string
})

(:Tier {
    id: string,
    level: integer,
    name: string,
    description: string
})

// Relationships
(b:Book)-[:BELONGS_TO]->(t:Tier)
(b:Book)-[:IN_TAXONOMY]->(tx:Taxonomy)
(c:Chapter)-[:PART_OF]->(b:Book)
(c:Chapter)-[:DISCUSSES]->(concept:Concept)

// Cross-Reference Relationships (bidirectional in application logic)
(b1:Book)-[:PARALLEL {similarity: float}]->(b2:Book)
(b1:Book)-[:PERPENDICULAR {relevance: float}]->(b2:Book)
(b1:Book)-[:SKIP_TIER {hop_distance: int}]->(b2:Book)
(c1:Chapter)-[:SIMILAR_TO {score: float}]->(c2:Chapter)
```

### Qdrant Collection Schema

```json
{
    "collection_name": "chapters",
    "vectors": {
        "size": 1536,
        "distance": "Cosine"
    },
    "payload_schema": {
        "book_id": "keyword",
        "book_title": "text",
        "chapter_number": "integer",
        "chapter_title": "text",
        "tier": "integer",
        "taxonomy_id": "keyword",
        "keywords": "keyword[]",
        "concepts": "keyword[]"
    }
}
```

---

## New API Endpoints

### POST /v1/search/hybrid

Combines vector similarity with graph-aware filtering and re-ranking.

**Request:**
```json
{
    "query": "dependency injection patterns",
    "taxonomy_id": "software-engineering",
    "source_context": {
        "book": "Python Distilled",
        "chapter": 7,
        "tier": 1
    },
    "search_config": {
        "vector_top_k": 20,
        "graph_max_hops": 3,
        "relationship_types": ["PARALLEL", "PERPENDICULAR", "SKIP_TIER"],
        "min_similarity": 0.7
    },
    "ranking_config": {
        "vector_weight": 0.6,
        "graph_weight": 0.4,
        "tier_preference": "perpendicular"
    }
}
```

**Response:**
```json
{
    "results": [
        {
            "book": "Clean Code",
            "chapter": 11,
            "title": "Systems",
            "tier": 2,
            "relationship_to_source": "PERPENDICULAR",
            "scores": {
                "vector_similarity": 0.89,
                "graph_relevance": 0.92,
                "combined": 0.90
            },
            "path_from_source": [
                {"book": "Python Distilled", "chapter": 7, "tier": 1},
                {"book": "Clean Code", "chapter": 11, "tier": 2, "edge": "PERPENDICULAR"}
            ],
            "keywords": ["dependency injection", "factories", "separation of concerns"]
        }
    ],
    "metadata": {
        "vector_results_count": 20,
        "graph_filtered_count": 8,
        "tiers_covered": [1, 2, 3],
        "query_time_ms": 145
    }
}
```

### POST /v1/graph/traverse

Execute spider web traversal from a starting node.

**Request:**
```json
{
    "start_node": {
        "book": "Python Distilled",
        "chapter": 7,
        "tier": 1
    },
    "traversal_config": {
        "max_hops": 3,
        "relationship_types": ["PARALLEL", "PERPENDICULAR", "SKIP_TIER"],
        "direction": "BOTH",
        "allow_cycles": true
    },
    "filter": {
        "concepts": ["abstraction", "modularity"],
        "min_relevance": 0.5
    }
}
```

**Response:**
```json
{
    "paths": [
        {
            "nodes": [
                {"book": "Python Distilled", "chapter": 7, "tier": 1, "type": "START"},
                {"book": "Clean Code", "chapter": 3, "tier": 2, "edge": "PERPENDICULAR"},
                {"book": "Microservices", "chapter": 8, "tier": 3, "edge": "PERPENDICULAR"},
                {"book": "Philosophy of SW", "chapter": 9, "tier": 1, "edge": "SKIP_TIER"}
            ],
            "path_score": 0.78,
            "path_type": "non_linear",
            "hops": 3
        }
    ],
    "stats": {
        "nodes_visited": 24,
        "unique_books": 8,
        "paths_found": 5,
        "tiers_covered": [1, 2, 3]
    }
}
```

### POST /v1/graph/query

Execute raw Cypher queries for advanced use cases.

**Request:**
```json
{
    "query": "MATCH (b1:Book)-[r:PERPENDICULAR]->(b2:Book) WHERE b1.tier = $tier RETURN b2",
    "parameters": {
        "tier": 1
    }
}
```

---

## Implementation Phases

### Phase 1: Neo4j Integration (Week 1-2)

1. **Add Neo4j client** to semantic-search-service
   - `src/graph/neo4j_client.py`
   - Connection pooling, retry logic
   - Health check integration

2. **Create schema initialization**
   - `scripts/init_neo4j_schema.cypher`
   - Index creation for performance

3. **Add `/v1/graph/query` endpoint**
   - Basic Cypher execution
   - Parameter sanitization

### Phase 2: Traversal Engine (Week 3-4)

1. **Implement spider web traversal**
   - `src/graph/traversal.py`
   - BFS/DFS with relationship type filtering
   - Cycle detection (optional)

2. **Add `/v1/graph/traverse` endpoint**
   - Request validation
   - Path serialization

3. **Unit tests**
   - Mock Neo4j responses
   - Traversal algorithm tests

### Phase 3: Hybrid Search (Week 5-6)

1. **Implement query planner**
   - `src/search/hybrid.py`
   - Decide when to use vector vs graph vs both

2. **Implement result merger**
   - Score fusion algorithms (RRF, weighted)
   - Deduplication

3. **Add `/v1/search/hybrid` endpoint**
   - Full integration
   - Performance optimization

4. **Integration tests**
   - End-to-end hybrid search
   - Performance benchmarks

### Phase 4: AI Agents Integration (Week 7-8)

1. **Update ai-agents service**
   - Add `graph_client.py`
   - Register new tools

2. **Implement Cross-Reference Agent tools**
   - Wire up to semantic-search-service
   - Test with real taxonomy data

3. **End-to-end testing**
   - Full cross-referencing workflow
   - Citation generation

---

## Configuration

### Environment Variables (New)

```bash
# Neo4j Configuration
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secret
NEO4J_DATABASE=neo4j
NEO4J_MAX_CONNECTIONS=50
NEO4J_CONNECTION_TIMEOUT=30

# Hybrid Search
HYBRID_DEFAULT_VECTOR_WEIGHT=0.6
HYBRID_DEFAULT_GRAPH_WEIGHT=0.4
HYBRID_MAX_TRAVERSAL_HOPS=5
```

### Updated Pydantic Settings

```python
# src/core/config.py (additions)
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Neo4j
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_max_connections: int = 50
    neo4j_connection_timeout: int = 30
    
    # Hybrid Search
    hybrid_default_vector_weight: float = 0.6
    hybrid_default_graph_weight: float = 0.4
    hybrid_max_traversal_hops: int = 5
```

---

## Dependencies (New)

Add to `requirements.txt`:

```
# Graph Database
neo4j==5.18.0
neo4j-driver==5.18.0

# Optional: Graph algorithms
networkx==3.2.1
```

---

## Docker Compose (Development)

```yaml
# docker-compose.dev.yml (additions)
services:
  semantic-search:
    # ... existing config ...
    environment:
      - NEO4J_URL=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
    depends_on:
      - qdrant
      - neo4j  # NEW

  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"  # Browser UI
      - "7687:7687"  # Bolt protocol
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - ./scripts/init_neo4j_schema.cypher:/docker-entrypoint-initdb.d/init.cypher
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
      - NEO4J_PLUGINS=["apoc"]
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]
      interval: 10s
      timeout: 10s
      retries: 5

volumes:
  neo4j_data:
  neo4j_logs:
```

---

## Folder Structure (Additions)

```
semantic-search-service/
├── src/
│   ├── graph/                    # NEW
│   │   ├── __init__.py
│   │   ├── neo4j_client.py       # Connection management
│   │   ├── schema.py             # Graph schema definitions
│   │   └── traversal.py          # Spider web algorithms
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── vector.py             # Existing
│   │   └── hybrid.py             # NEW: Combined search
│   │
│   ├── api/
│   │   ├── routes/
│   │   │   ├── search.py         # Existing
│   │   │   └── graph.py          # NEW: /v1/graph/* endpoints
│   │   └── ...
│   │
│   └── models/
│       ├── ...
│       └── graph.py              # NEW: Graph request/response models
│
├── scripts/
│   └── init_neo4j_schema.cypher  # NEW
│
├── tests/
│   ├── unit/
│   │   └── test_graph/           # NEW
│   │       ├── test_traversal.py
│   │       └── test_neo4j_client.py
│   └── integration/
│       └── test_hybrid_search.py # NEW
```

---

## Success Criteria

### POC Complete When:

1. ✅ Neo4j container running with schema initialized
2. ✅ `/v1/graph/traverse` returns valid paths
3. ✅ `/v1/search/hybrid` combines vector + graph results
4. ✅ Unit tests pass (≥80% coverage on new code)
5. ✅ Integration test: Cross-Reference Agent can traverse taxonomy
6. ✅ Performance: Hybrid search < 500ms for typical queries

### Metrics to Track:

- Query latency (p50, p95, p99)
- Result relevance (manual evaluation)
- Tier coverage per query
- Path diversity

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Neo4j latency | High | Connection pooling, query caching |
| Complex Cypher queries | Medium | Pre-built query templates |
| Schema evolution | Medium | Migration scripts, versioning |
| Data sync (Qdrant ↔ Neo4j) | High | Single ingestion pipeline |

---

## Next Steps

1. **Review this POC** with team
2. **Provision Neo4j** (local Docker for dev)
3. **Create sample taxonomy data** in Neo4j
4. **Implement Phase 1** (Neo4j integration)
5. **Test with real textbook data**

---

## References

- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [LangChain Neo4j Integration](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher)
- [TIER_RELATIONSHIP_DIAGRAM.md](/textbooks/TIER_RELATIONSHIP_DIAGRAM.md)
- [ai-agents/docs/ARCHITECTURE.md](/ai-agents/docs/ARCHITECTURE.md)
