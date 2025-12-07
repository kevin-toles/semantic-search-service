# Graph RAG POC - Semantic Search Service

## Overview

This POC validates the integration of **Neo4j** (graph database) with **Qdrant** (vector database) to enable spider web taxonomy traversal for the Cross-Reference Agent.

## Quick Start

### 1. Start Infrastructure

```bash
cd poc
docker-compose -f docker-compose.poc.yml up -d
```

### 2. Initialize Schema

```bash
# Neo4j schema
docker exec -i poc-neo4j cypher-shell -u neo4j -p pocpassword < scripts/init_neo4j_schema.cypher

# Qdrant collection
python scripts/init_qdrant_collection.py
```

### 3. Load Sample Data

```bash
python scripts/load_sample_taxonomy.py
```

### 4. Verify Data

```bash
python scripts/verify_data.py
```

### 5. Run Tests

```bash
pytest tests/ -v
```

## Folder Structure

```
poc/
├── README.md                      # This file
├── docker-compose.poc.yml         # Neo4j + Qdrant containers
├── requirements-poc.txt           # POC dependencies
├── data/
│   └── sample_taxonomy.json       # Sample data
├── scripts/
│   ├── init_neo4j_schema.cypher   # Neo4j DDL
│   ├── init_qdrant_collection.py  # Qdrant setup
│   ├── load_sample_taxonomy.py    # Data loader
│   └── verify_data.py             # Validation
├── src/
│   ├── neo4j_client.py            # Neo4j wrapper
│   ├── qdrant_client.py           # Qdrant wrapper
│   ├── search/                    # Search implementations
│   ├── retrievers/                # LangChain retrievers
│   └── tools/                     # Agent tools
├── tests/                         # Unit tests
└── notebooks/                     # Jupyter exploration
```

## URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Neo4j Browser | http://localhost:7474 | neo4j / pocpassword |
| Qdrant Dashboard | http://localhost:6333/dashboard | None |

## Related Documentation

- [GRAPH_RAG_POC.md](../docs/GRAPH_RAG_POC.md) - Detailed design document
- [GRAPH_RAG_POC_PLAN.md](/textbooks/GRAPH_RAG_POC_PLAN.md) - WBS and project plan
- [ai-agents/poc/](../../ai-agents/poc/) - Cross-Reference Agent POC
