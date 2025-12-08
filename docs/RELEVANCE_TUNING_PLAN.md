# Relevance Tuning Implementation Plan

## Problem Statement

The semantic search is returning **surface-level keyword matches** without **domain/intent awareness**.

**Example**: Query about "LLM/RAG document chunking" returns:
- ❌ Effective Modern C++ copyright notice ("chunks of code")
- ❌ Vector memory reallocation ("chunk of memory") 
- ❌ Random C++ memory access patterns
- ✅ Should return: RAG pipelines, embedding strategies, document splitting

**Root Cause**: The retrieval layer cannot distinguish:
- "chunking for LLM/RAG pipelines" (intended domain)
- "generic chunks of memory in C++ internals" (wrong domain)

## Architecture Analysis

Current stack (per ARCHITECTURE.md):
```
Query → SBERT Embedding → Qdrant Vector Search
                              ↓
                         Neo4j Graph Traversal
                              ↓
                         ResultRanker (linear/rrf/max fusion)
                              ↓
                         Results
```

**Key Insight**: The LLM is NOT in this loop. Bad retrieval feeds bad context to ANY generator.

---

## Implementation Plan

### Phase 1: Taxonomy/Graph Hard Filter (Quick Win)

**Location**: `src/search/hybrid.py`, `src/retrievers/neo4j_retriever.py`

**Goal**: Use taxonomy tiers as a whitelist/blacklist based on query intent.

```python
# In neo4j_retriever.py - add domain filtering
DOMAIN_TAXONOMY = {
    "llm_rag": {
        "keywords": ["chunk", "embed", "RAG", "retrieval", "vector", "token", "context"],
        "tiers": ["architecture", "practices"],  # Whitelist
        "exclude_books": ["Effective Modern C++", "Game Programming Gems*"],
    },
    "cpp_performance": {
        "keywords": ["memory", "allocator", "cache", "thread"],
        "tiers": ["implementation"],
        "include_books": ["Effective Modern C++", "C++ Concurrency*"],
    },
}
```

**Implementation**:
```python
# Add to HybridSearchRequest model (already has tier_filter!)
focus_areas: list[str] | None = Field(
    default=None,
    description="Focus areas for domain-aware filtering (e.g., 'llm_rag', 'cpp_performance')"
)

# In hybrid_search, before fusion:
if focus_areas and "llm_rag" in focus_areas:
    # Hard filter: exclude C++/game dev books
    candidates = [c for c in candidates if c.book_id not in EXCLUDED_BOOKS]
    # Soft prior: boost LLM/RAG tier books
    for c in candidates:
        if c.tier in ["architecture", "practices"]:
            c.graph_score += 0.3
```

### Phase 2: Focus-Area Keyword Co-occurrence Filter

**Location**: `src/search/metadata_filter.py` (new), `src/search/ranker.py`

**Goal**: Require domain-specific term co-occurrence for relevance.

```python
# metadata_filter.py
FOCUS_AREA_TERMS = {
    "llm_rag_chunking": {
        "primary": ["chunk", "chunking", "split", "segment"],
        "domain": ["RAG", "retrieval", "vector", "embedding", "LLM", "token", 
                   "context window", "document", "corpus", "index"],
        "min_domain_matches": 1,  # Must have at least 1 domain term
    },
}

def filter_by_focus_area(
    passages: list[SearchResult],
    focus_area: str,
    query_text: str,
) -> list[SearchResult]:
    """Filter passages requiring domain term co-occurrence."""
    config = FOCUS_AREA_TERMS.get(focus_area, {})
    primary_terms = config.get("primary", [])
    domain_terms = config.get("domain", [])
    min_matches = config.get("min_domain_matches", 1)
    
    filtered = []
    for passage in passages:
        text_lower = passage.content.lower()
        
        # Check if passage has primary term
        has_primary = any(t in text_lower for t in primary_terms)
        
        # Count domain term matches
        domain_count = sum(1 for t in domain_terms if t.lower() in text_lower)
        
        # Keep if: has primary term AND meets domain threshold
        if has_primary and domain_count >= min_matches:
            passage.metadata["domain_match_count"] = domain_count
            filtered.append(passage)
        elif not has_primary:
            # No primary term - keep but lower score
            filtered.append(passage)
    
    return filtered
```

### Phase 3: Enhanced Ranker with Focus Overlap Score

**Location**: `src/search/ranker.py`

**Goal**: Add `focus_overlap_score` as a ranking signal.

```python
# Add to ResultRanker class
_DEFAULT_FOCUS_WEIGHT = 0.2  # Weight for focus area matching

def __init__(
    self,
    ...,
    focus_weight: float = _DEFAULT_FOCUS_WEIGHT,
    focus_keywords: list[str] | None = None,
) -> None:
    self._focus_weight = focus_weight
    self._focus_keywords = focus_keywords or []

def compute_focus_overlap(
    self,
    passage_text: str,
) -> float:
    """Compute overlap between passage and focus keywords."""
    if not self._focus_keywords:
        return 0.0
    
    text_lower = passage_text.lower()
    matches = sum(1 for kw in self._focus_keywords if kw.lower() in text_lower)
    return matches / len(self._focus_keywords)

def fuse_with_focus(
    self,
    vector_scores: dict[str, float],
    graph_scores: dict[str, float],
    passages: dict[str, str],  # doc_id -> text
) -> dict[str, float]:
    """Fuse scores including focus overlap."""
    base_fused = self.fuse(vector_scores, graph_scores)
    
    result = {}
    for doc_id, base_score in base_fused.items():
        focus_score = self.compute_focus_overlap(passages.get(doc_id, ""))
        
        # Adjust weights to include focus
        # final = (1-focus_weight) * base + focus_weight * focus_overlap
        final = (1 - self._focus_weight) * base_score + self._focus_weight * focus_score
        result[doc_id] = self._clamp_score(final)
    
    return result
```

### Phase 4: Topic Model as Domain Guardrail

**Location**: `src/topics/` (existing), integrate into `src/search/hybrid.py`

**Goal**: Use LDA/LSI topic similarity as a domain filter.

```python
# Pre-compute topic distributions for each chunk (offline)
# At search time:

def topic_similarity_filter(
    query_topics: list[float],  # Topic distribution for query
    passage_topics: dict[str, list[float]],  # doc_id -> topic distribution
    threshold: float = 0.3,
) -> set[str]:
    """Return doc_ids with sufficient topic similarity."""
    from scipy.spatial.distance import cosine
    
    valid_ids = set()
    for doc_id, topics in passage_topics.items():
        similarity = 1 - cosine(query_topics, topics)
        if similarity >= threshold:
            valid_ids.add(doc_id)
    
    return valid_ids

# Integrate into hybrid search:
# 1. Infer query topics
# 2. Filter candidates by topic similarity
# 3. Then apply vector+graph fusion
```

### Phase 5: Lightweight Relevance Classifier (Reranker)

**Location**: `src/search/reranker.py` (new)

**Goal**: Train a small model to classify relevance.

```python
# reranker.py
from dataclasses import dataclass
import numpy as np

@dataclass
class RerankerFeatures:
    """Features for relevance classification."""
    query_embedding: list[float]
    passage_embedding: list[float]
    topic_similarity: float
    focus_keyword_count: int
    domain_term_count: int
    tier: int
    book_in_target_domain: bool

class RelevanceReranker:
    """Second-stage reranker for domain-specific relevance."""
    
    def __init__(self, model_path: str | None = None):
        """Load trained model or use heuristic fallback."""
        self._model = self._load_model(model_path) if model_path else None
    
    def score(self, features: RerankerFeatures) -> float:
        """Score passage relevance (0-1)."""
        if self._model:
            return self._model.predict_proba(features.to_array())[0][1]
        else:
            # Heuristic fallback
            return self._heuristic_score(features)
    
    def _heuristic_score(self, features: RerankerFeatures) -> float:
        """Simple heuristic scoring before model is trained."""
        score = 0.5  # Base
        
        # Domain matching
        if features.book_in_target_domain:
            score += 0.2
        
        # Focus keyword presence
        score += min(0.2, features.focus_keyword_count * 0.05)
        
        # Domain term co-occurrence
        score += min(0.15, features.domain_term_count * 0.03)
        
        # Topic similarity
        score += features.topic_similarity * 0.15
        
        # Tier preference (lower tier = more foundational = better for concepts)
        if features.tier == 1:
            score += 0.1
        
        return min(1.0, score)
    
    def rerank(
        self,
        candidates: list[SearchResult],
        features_batch: list[RerankerFeatures],
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[SearchResult]:
        """Rerank candidates, filtering by threshold."""
        scored = []
        for candidate, features in zip(candidates, features_batch):
            relevance = self.score(features)
            if relevance >= threshold:
                candidate.metadata["reranker_score"] = relevance
                scored.append((relevance, candidate))
        
        # Sort by relevance descending
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[:top_k]]
```

### Phase 6: Graph Traversal Domain Constraints

**Location**: `src/graph/traversal.py`, `src/search/hybrid.py`

**Goal**: Keep graph traversal within relevant conceptual neighborhood.

```python
# In graph traversal, add domain-aware edge weighting
RELATIONSHIP_WEIGHTS_BY_DOMAIN = {
    "llm_rag": {
        "PARALLEL": 1.2,      # Same-tier, same-domain = boost
        "PERPENDICULAR": 1.0, # Cross-tier within domain = neutral
        "SKIP_TIER": 0.5,     # Large jump = penalize
    },
    "cpp_performance": {
        "PARALLEL": 1.2,
        "PERPENDICULAR": 1.0,
        "SKIP_TIER": 0.7,     # Less penalty for implementation details
    },
}

# Add to traversal query:
# - Include tier/domain in node properties
# - Penalize edges that leave the target domain subtree
cypher = """
MATCH path = (start)-[r*1..{max_depth}]-(related)
WHERE start.id = $start_id
  AND ALL(rel IN r WHERE 
      CASE 
        WHEN $target_domain = 'llm_rag' AND related.tier IN ['architecture', 'practices']
        THEN 1.0
        WHEN $target_domain = 'llm_rag' AND related.book_title CONTAINS 'C++'
        THEN 0.2  -- Heavy penalty for C++ when looking for LLM content
        ELSE 0.7
      END > 0.5
  )
RETURN related, r, length(path) as depth
"""
```

---

## Implementation Priority

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| 1. Taxonomy Hard Filter | Low | High | **P0** |
| 2. Focus Keyword Co-occurrence | Low | High | **P0** |
| 3. Enhanced Ranker | Medium | Medium | P1 |
| 4. Topic Model Guardrail | Medium | Medium | P1 |
| 5. Relevance Classifier | High | High | P2 |
| 6. Graph Domain Constraints | Medium | Medium | P2 |

---

## Validation Metrics

For the test query about LLM/RAG chunking:

| Metric | Current | Target |
|--------|---------|--------|
| Results from wrong domain (C++/Game) | ~70% | <10% |
| Results with `relevance_topic: "chunk"` but no LLM context | High | <5% |
| Results from taxonomy tier 1-2 (architecture/practices) | ~30% | >80% |
| Precision@10 for LLM/RAG content | Low | >0.8 |

---

## Files to Modify

1. **`src/api/models.py`** - Add `focus_areas` field to request ✅ (tier_filter exists)
2. **`src/search/metadata_filter.py`** - Create focus area co-occurrence filter
3. **`src/search/ranker.py`** - Add `focus_overlap_score` computation
4. **`src/search/hybrid.py`** - Integrate taxonomy filtering
5. **`src/search/reranker.py`** - New relevance classifier
6. **`src/retrievers/neo4j_retriever.py`** - Add domain-aware graph queries
7. **`config/domain_taxonomy.json`** - Domain definitions (keywords, tiers, books)

---

## Testing Strategy

1. **Unit Tests**: Each new function in isolation
2. **Integration Tests**: End-to-end hybrid search with focus areas
3. **Regression Tests**: Ensure existing queries still work
4. **Golden Set Tests**: Known good/bad passages for the chunking query

```python
# tests/golden/test_chunking_relevance.py
POSITIVE_PASSAGES = [
    "LLM-Engineers-Handbook - RAG ingestion pipeline chunks documents...",
    "AI Agents - semantic chunking for vector embeddings...",
]

NEGATIVE_PASSAGES = [
    "Effective Modern C++ - allocates a single chunk of memory...",
    "Game Programming Gems - chunk of code for threading...",
]

def test_reranker_separates_domains():
    reranker = RelevanceReranker()
    for passage in POSITIVE_PASSAGES:
        assert reranker.score(make_features(passage, "llm_rag")) > 0.7
    for passage in NEGATIVE_PASSAGES:
        assert reranker.score(make_features(passage, "llm_rag")) < 0.4
```
