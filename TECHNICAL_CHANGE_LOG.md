# semantic-search-service: Technical Change Log

**Purpose**: Documents architectural decisions, API changes, and significant updates to the Semantic Search Service (embeddings, vector search, hybrid search).

---

## Changelog

### 2026-01-26: Lazy ServiceContainer Initialization Fix (CL-007)

**Summary**: Fixed production startup failure by allowing lazy ServiceContainer initialization in create_app.

**Commit:** `2b2dd61`

**Issue Fixed:**
- `ValueError` at startup when services=None passed to create_app
- Production apps instantiate at module level before lifespan starts

**Solution:**
- Remove ValueError when services=None
- Services injected during lifespan startup instead
- Added RuntimeError protection if services still None when dependency requested

**Files Changed:**

| File | Changes |
|------|---------|
| `src/api/app.py` | +18/-15 lines: Lazy initialization |

**Cross-References:**
- CL-006: Root cause (fakes refactor)

---

### 2026-01-23: Test Doubles Refactor (CL-006)

**Summary**: Moved fake test doubles from production code to tests directory, following best practices for dependency injection.

**Commit:** `caa31e6`

**Architectural Decision: No Test Doubles in Production Code**

Production code should not contain test doubles. All fakes moved to dedicated test module.

**New Structure:**

| Location | Contents |
|----------|----------|
| `tests/fakes.py` | FakeVectorClient, FakeGraphClient, FakeEmbeddingService |
| `tests/conftest.py` | Test fixtures for proper DI |

**Files Changed:**

| File | Changes |
|------|---------|
| `src/api/app.py` | Require explicit ServiceContainer |
| `src/api/dependencies.py` | -129 lines: Removed fake classes |
| `tests/fakes.py` | New: +137 lines test doubles |
| `tests/conftest.py` | New: +74 lines fixtures |
| `tests/integration/test_hybrid_api.py` | Updated imports |
| `tests/unit/test_embed_api.py` | Updated imports |

---

### 2026-01-18: WBS-LOG0 Structured JSON Logging (CL-005)

**Summary**: Added structured JSON logging with correlation ID support for distributed tracing.

**Files Changed:**

| File | Changes |
|------|---------|
| `src/core/logging.py` | Structured logging setup |
| `src/api/middleware.py` | Correlation ID propagation |

**Cross-References:**
- WBS-LOG0: Logging architecture

---

### 2026-01-15: SonarQube Remediation (CL-004)

**Summary**: Refactored routes.py and main.py to address SonarQube cognitive complexity findings.

**Improvements:**

| Metric | Before | After |
|--------|--------|-------|
| Cognitive Complexity | 20+ | <15 |
| Code Smells | 8 | 2 |

**Files Changed:**

| File | Changes |
|------|---------|
| `src/api/routes.py` | Extracted helper functions |
| `src/main.py` | Simplified startup logic |

---

### 2026-01-12: Collection and Embedding Defaults (CL-003)

**Summary**: Fixed default collection to 'chapters' and standardized on 384-dimension embeddings for all-MiniLM-L6-v2.

**Configuration:**

| Setting | Old | New |
|---------|-----|-----|
| Default Collection | `code_chunks` | `chapters` |
| Embedding Dimensions | Variable | 384 |
| Model | Variable | `all-MiniLM-L6-v2` |

**Files Changed:**

| File | Changes |
|------|---------|
| `src/config.py` | Updated defaults |
| `src/embedders/sbert.py` | Fixed dimension handling |

---

### 2026-01-10: Docker Network Fix (CL-002)

**Summary**: Removed data-network from docker-compose.yml to resolve network conflicts in hybrid mode.

**Files Changed:**

| File | Changes |
|------|---------|
| `docker-compose.yml` | Removed data-network |

---

### 2026-01-01: Initial Semantic Search Service (CL-001)

**Summary**: Initial service with hybrid search combining vector similarity and keyword matching.

**Core Components:**

| Component | Purpose |
|-----------|---------|
| `SBERTEmbedder` | Sentence-BERT embeddings |
| `HybridSearcher` | Combined vector + keyword search |
| `QdrantClient` | Vector database interface |
| `Neo4jClient` | Graph database for relationships |

**API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/v1/search/semantic` | POST | Vector similarity search |
| `/v1/search/hybrid` | POST | Combined search |
| `/v1/embed` | POST | Generate embeddings |

**Configuration:**

| Setting | Default | Purpose |
|---------|---------|---------|
| `SEMANTIC_SEARCH_PORT` | 8081 | Service port |
| `QDRANT_URL` | `http://localhost:6333` | Vector DB |
| `NEO4J_URI` | `bolt://localhost:7687` | Graph DB |
| `SBERT_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
