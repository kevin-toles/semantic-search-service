"""
Unit tests for domain-aware metadata filtering.

Tests the MetadataFilter class with realistic examples from the
problematic cross-reference output (C++ memory chunks vs LLM RAG chunks).
"""

import pytest
from pathlib import Path

from src.search.metadata_filter import (
    MetadataFilter,
    DomainConfig,
    FilterResult,
    create_filter,
    filter_for_domain,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def filter_instance():
    """Create a MetadataFilter with loaded config."""
    f = MetadataFilter()
    f.load_config()
    return f


@pytest.fixture
def llm_rag_passages():
    """Passages that SHOULD be relevant for llm_rag domain."""
    return [
        {
            "id": "llm-handbook-ch14",
            "content": "The RAG ingestion pipeline extracts raw documents from various data sources. Then, it cleans, chunks (splits into smaller sections), and embeds the documents. Ultimately, it loads the embedded chunks into a vector DB.",
            "score": 0.85,
            "metadata": {
                "book_title": "LLM-Engineers-Handbook",
                "tier": "practices",
            },
        },
        {
            "id": "llm-handbook-ch17",
            "content": "Any RAG system is split into two independent components: The ingestion pipeline takes in raw data, cleans, chunks, embeds, and loads it into a vector DB. The inference pipeline queries the vector DB for relevant context.",
            "score": 0.82,
            "metadata": {
                "book_title": "LLM-Engineers-Handbook",
                "tier": "practices",
            },
        },
        {
            "id": "ai-agents-ch5",
            "content": "Chunk and embed the cleaned data. Store the vectorized data into a vector DB for RAG. For training, we have to fine-tune LLMs of various sizes.",
            "score": 0.78,
            "metadata": {
                "book_title": "AI Agents In Action",
                "tier": "architecture",
            },
        },
    ]


@pytest.fixture
def cpp_memory_passages():
    """Passages that should NOT be relevant for llm_rag domain (false positives)."""
    return [
        {
            "id": "cpp-ch10",
            "content": "std::vector lacks space for it, i.e., that the std::vector's size is equal to its capacity. When that happens, the std::vector allocates a new, larger, chunk of memory to hold its elements.",
            "score": 0.75,
            "metadata": {
                "book_title": "Effective Modern C++",
                "tier": "implementation",
            },
        },
        {
            "id": "cpp-ch15",
            "content": "std::make_shared allocates a single chunk of memory to hold both the Widget object and the control block. This optimization reduces the static size of the program.",
            "score": 0.72,
            "metadata": {
                "book_title": "Effective Modern C++",
                "tier": "implementation",
            },
        },
        {
            "id": "cpp-ch2",
            "content": "writing a program that uses several chunks of code from this book does not require permission. Selling or distributing a CD-ROM of examples from O'Reilly books does require permission.",
            "score": 0.70,
            "metadata": {
                "book_title": "Effective Modern C++",
                "tier": "implementation",
            },
        },
    ]


@pytest.fixture
def game_dev_passages():
    """Game dev passages that should NOT be relevant for llm_rag domain."""
    return [
        {
            "id": "gems-ch5",
            "content": "the victim page determination takes into account the amount of performance hit involved with actually filling a page with a given chunk of memory. This performance cost can come from a number of sources.",
            "score": 0.68,
            "metadata": {
                "book_title": "Game Programming Gems 7",
                "tier": "implementation",
            },
        },
        {
            "id": "gems-ch18",
            "content": "The kD-tree (k-dimensional tree) is a structure that recursively splits space in two halves. In this sense, it is a BSP tree. There is, however, a restriction—the splitting planes are axis-aligned.",
            "score": 0.65,
            "metadata": {
                "book_title": "Game Programming Gems 7",
                "tier": "implementation",
            },
        },
    ]


# =============================================================================
# Test Cases
# =============================================================================


class TestMetadataFilterInit:
    """Tests for MetadataFilter initialization."""

    def test_load_config_success(self, filter_instance):
        """Test that config loads successfully."""
        assert filter_instance._loaded is True
        assert "llm_rag" in filter_instance.available_domains

    def test_available_domains(self, filter_instance):
        """Test available domains are populated."""
        domains = filter_instance.available_domains
        assert "llm_rag" in domains
        assert "python_implementation" in domains
        assert "microservices_architecture" in domains

    def test_get_domain_returns_config(self, filter_instance):
        """Test get_domain returns DomainConfig."""
        config = filter_instance.get_domain("llm_rag")
        assert config is not None
        assert isinstance(config, DomainConfig)
        assert "chunk" in config.primary_keywords
        assert "RAG" in config.domain_keywords


class TestDomainFiltering:
    """Tests for domain-aware filtering."""

    def test_llm_rag_passages_kept(self, filter_instance, llm_rag_passages):
        """Test that legitimate LLM/RAG passages are kept."""
        results = filter_instance.apply(llm_rag_passages, "llm_rag")
        
        # All should be kept
        assert len(results) == len(llm_rag_passages)
        
        # Scores should be boosted (whitelist book + domain keywords)
        for result in results:
            assert result["score"] >= result["metadata"].get("original_score", 0.5)

    def test_cpp_passages_filtered_or_penalized(self, filter_instance, cpp_memory_passages):
        """Test that C++ memory passages are filtered or heavily penalized."""
        results = filter_instance.apply(cpp_memory_passages, "llm_rag", remove_filtered=True)
        
        # Some or all should be filtered out
        assert len(results) < len(cpp_memory_passages)

    def test_cpp_passages_penalized_when_kept(self, filter_instance, cpp_memory_passages):
        """Test that C++ passages get penalized scores when not removed."""
        results = filter_instance.apply(cpp_memory_passages, "llm_rag", remove_filtered=False)
        
        # All should have reduced scores
        for i, result in enumerate(results):
            original = cpp_memory_passages[i]["score"]
            # Score should be lower due to blacklist + no domain terms
            assert result["score"] < original, f"Expected {result['id']} to be penalized"

    def test_game_dev_passages_handled(self, filter_instance, game_dev_passages):
        """Test that game dev passages are filtered or penalized."""
        results = filter_instance.apply(game_dev_passages, "llm_rag", remove_filtered=False)
        
        for i, result in enumerate(results):
            original = game_dev_passages[i]["score"]
            # Should be penalized for blacklist pattern
            assert result["score"] <= original

    def test_mixed_passages_ranked_correctly(
        self, filter_instance, llm_rag_passages, cpp_memory_passages
    ):
        """Test that good passages outrank bad ones after filtering."""
        all_passages = llm_rag_passages + cpp_memory_passages
        results = filter_instance.apply(all_passages, "llm_rag", remove_filtered=False)
        
        # Sort by score
        results.sort(key=lambda x: -x["score"])
        
        # Top results should be LLM/RAG passages
        top_3_ids = [r["id"] for r in results[:3]]
        for passage in llm_rag_passages:
            assert passage["id"] in top_3_ids, f"Expected {passage['id']} in top 3"


class TestScoreAdjustments:
    """Tests for score adjustment logic."""

    def test_whitelist_book_boost(self, filter_instance):
        """Test that whitelist books get score boost."""
        passage = {
            "id": "test-1",
            "content": "RAG ingestion pipeline chunks documents for vector search.",
            "score": 0.5,
            "metadata": {"book_title": "LLM-Engineers-Handbook", "tier": "practices"},
        }
        
        adjustment, reasons = filter_instance.compute_score_adjustment(passage, "llm_rag")
        
        assert adjustment > 0
        assert any("whitelist_book" in r for r in reasons)

    def test_blacklist_book_penalty(self, filter_instance):
        """Test that blacklist books get score penalty."""
        passage = {
            "id": "test-2",
            "content": "allocates a chunk of memory",
            "score": 0.5,
            "metadata": {"book_title": "Effective Modern C++", "tier": "implementation"},
        }
        
        adjustment, reasons = filter_instance.compute_score_adjustment(passage, "llm_rag")
        
        assert adjustment < 0
        assert any("blacklist_book" in r for r in reasons)

    def test_primary_only_no_domain_penalty(self, filter_instance):
        """Test penalty when primary keyword present but no domain context."""
        passage = {
            "id": "test-3",
            "content": "The function processes the chunk of data efficiently.",
            "score": 0.5,
            "metadata": {"book_title": "Some Neutral Book", "tier": "implementation"},
        }
        
        adjustment, reasons = filter_instance.compute_score_adjustment(passage, "llm_rag")
        
        # Should have penalty for primary-only
        assert any("primary_only" in r for r in reasons)

    def test_domain_keywords_boost(self, filter_instance):
        """Test boost when domain keywords are present."""
        passage = {
            "id": "test-4",
            "content": "The RAG system uses vector embeddings for semantic search over documents.",
            "score": 0.5,
            "metadata": {"book_title": "Some Neutral Book", "tier": "architecture"},
        }
        
        adjustment, reasons = filter_instance.compute_score_adjustment(passage, "llm_rag")
        
        # Should have boost for domain keywords
        assert any("domain_keywords" in r for r in reasons)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_filter(self):
        """Test create_filter returns initialized filter."""
        f = create_filter()
        assert f._loaded is True
        assert len(f.available_domains) > 0

    def test_filter_for_domain(self, llm_rag_passages):
        """Test filter_for_domain convenience function."""
        results = filter_for_domain(llm_rag_passages, "llm_rag")
        assert len(results) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_domain_no_filter(self, filter_instance, llm_rag_passages):
        """Test that unknown domain passes through unchanged."""
        results = filter_instance.apply(llm_rag_passages, "unknown_domain")
        assert len(results) == len(llm_rag_passages)

    def test_empty_passages_list(self, filter_instance):
        """Test handling of empty passages list."""
        results = filter_instance.apply([], "llm_rag")
        assert results == []

    def test_missing_metadata(self, filter_instance):
        """Test handling of passages with missing metadata."""
        passage = {
            "id": "test-missing",
            "content": "Some content about chunks and RAG.",
            "score": 0.5,
            # No metadata
        }
        results = filter_instance.apply([passage], "llm_rag")
        assert len(results) == 1

    def test_empty_content(self, filter_instance):
        """Test handling of empty content."""
        passage = {
            "id": "test-empty",
            "content": "",
            "score": 0.5,
            "metadata": {"book_title": "Test Book"},
        }
        results = filter_instance.apply([passage], "llm_rag")
        assert len(results) == 1
