"""
Domain-aware metadata filtering for semantic search.

Implements focus-area keyword co-occurrence filtering to ensure
retrieved passages are relevant to the intended domain, not just
surface-level keyword matches.

Example:
    Query: "How to chunk documents for RAG?"
    - KEEP: Passages about RAG pipelines that mention "chunk"
    - REJECT: C++ passages about "chunk of memory" (no RAG context)

Design follows RELEVANCE_TUNING_PLAN.md Phase 2.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =============================================================================
# Constants
# =============================================================================

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "domain_taxonomy.json"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DomainConfig:
    """Configuration for a specific domain."""

    name: str
    description: str
    primary_keywords: list[str] = field(default_factory=list)
    domain_keywords: list[str] = field(default_factory=list)
    min_domain_matches: int = 1
    tier_whitelist: list[str] = field(default_factory=list)
    book_blacklist_patterns: list[str] = field(default_factory=list)
    book_whitelist_patterns: list[str] = field(default_factory=list)
    score_adjustments: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> DomainConfig:
        """Create DomainConfig from dictionary."""
        return cls(
            name=name,
            description=data.get("description", ""),
            primary_keywords=data.get("primary_keywords", []),
            domain_keywords=data.get("domain_keywords", []),
            min_domain_matches=data.get("min_domain_matches", 1),
            tier_whitelist=data.get("tier_whitelist", []),
            book_blacklist_patterns=data.get("book_blacklist_patterns", []),
            book_whitelist_patterns=data.get("book_whitelist_patterns", []),
            score_adjustments=data.get("score_adjustments", {}),
        )


@dataclass
class FilterResult:
    """Result of applying domain filter to a passage."""

    doc_id: str
    keep: bool
    original_score: float
    adjusted_score: float
    primary_matches: int
    domain_matches: int
    in_whitelist_book: bool
    in_blacklist_book: bool
    in_whitelist_tier: bool
    adjustment_reasons: list[str] = field(default_factory=list)


# =============================================================================
# MetadataFilter Class
# =============================================================================


class MetadataFilter:
    """Domain-aware metadata filter for semantic search results.

    Uses focus area configurations to filter and score passages
    based on keyword co-occurrence and taxonomy membership.

    Usage:
        filter = MetadataFilter()
        filter.load_config()

        # Filter for LLM/RAG domain
        results = filter.apply(
            passages=search_results,
            domain="llm_rag",
        )
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize filter with optional config path.

        Args:
            config_path: Path to domain_taxonomy.json (uses default if None)
        """
        self._config_path = config_path or _CONFIG_PATH
        self._domains: dict[str, DomainConfig] = {}
        self._default_settings: dict[str, Any] = {}
        self._loaded = False

    def load_config(self) -> None:
        """Load domain configuration from JSON file."""
        if not self._config_path.exists():
            # Use empty config if file doesn't exist
            self._domains = {}
            self._default_settings = {
                "min_domain_matches": 1,
                "primary_only_penalty": -0.3,
                "unknown_domain_behavior": "no_filter",
            }
            self._loaded = True
            return

        with open(self._config_path) as f:
            data = json.load(f)

        # Parse domains
        for name, domain_data in data.get("domains", {}).items():
            self._domains[name] = DomainConfig.from_dict(name, domain_data)

        # Parse default settings
        self._default_settings = data.get("default_settings", {})
        self._loaded = True

    @property
    def available_domains(self) -> list[str]:
        """Get list of available domain names."""
        if not self._loaded:
            self.load_config()
        return list(self._domains.keys())

    def get_domain(self, name: str) -> DomainConfig | None:
        """Get domain configuration by name."""
        if not self._loaded:
            self.load_config()
        return self._domains.get(name)

    def apply(
        self,
        passages: list[dict[str, Any]],
        domain: str,
        remove_filtered: bool = True,
    ) -> list[dict[str, Any]]:
        """Apply domain filter to passages.

        Args:
            passages: List of passage dicts with 'id', 'content', 'score', 'metadata'
            domain: Domain name to filter for (e.g., 'llm_rag')
            remove_filtered: If True, remove passages that don't pass; else adjust scores

        Returns:
            Filtered/adjusted list of passages
        """
        if not self._loaded:
            self.load_config()

        config = self._domains.get(domain)
        if config is None:
            # Unknown domain - check default behavior
            if self._default_settings.get("unknown_domain_behavior") == "no_filter":
                return passages
            # Otherwise, pass through unchanged
            return passages

        results = []
        for passage in passages:
            filter_result = self._evaluate_passage(passage, config)

            if filter_result.keep or not remove_filtered:
                # Update passage with adjusted score
                passage = passage.copy()
                passage["score"] = filter_result.adjusted_score
                passage["metadata"] = passage.get("metadata", {}).copy()
                passage["metadata"]["domain_filter"] = {
                    "domain": domain,
                    "primary_matches": filter_result.primary_matches,
                    "domain_matches": filter_result.domain_matches,
                    "in_whitelist_book": filter_result.in_whitelist_book,
                    "in_blacklist_book": filter_result.in_blacklist_book,
                    "adjustment_reasons": filter_result.adjustment_reasons,
                }
                results.append(passage)

        # Re-sort by adjusted score
        results.sort(key=lambda x: -x["score"])
        return results

    def compute_score_adjustment(
        self,
        passage: dict[str, Any],
        domain: str,
    ) -> tuple[float, list[str]]:
        """Compute score adjustment for a passage.

        Args:
            passage: Passage dict with content and metadata
            domain: Domain name

        Returns:
            Tuple of (adjustment_value, list_of_reasons)
        """
        if not self._loaded:
            self.load_config()

        config = self._domains.get(domain)
        if config is None:
            return 0.0, []

        filter_result = self._evaluate_passage(passage, config)
        adjustment = filter_result.adjusted_score - filter_result.original_score
        return adjustment, filter_result.adjustment_reasons

    def _evaluate_passage(
        self,
        passage: dict[str, Any],
        config: DomainConfig,
    ) -> FilterResult:
        """Evaluate a single passage against domain config.

        Args:
            passage: Passage dict
            config: Domain configuration

        Returns:
            FilterResult with evaluation details
        """
        doc_id = passage.get("id", "unknown")
        content = passage.get("content", "")
        original_score = passage.get("score", 0.5)
        metadata = passage.get("metadata", {})

        content_lower = content.lower()
        adjustments = config.score_adjustments

        # Count keyword matches
        primary_matches = sum(
            1 for kw in config.primary_keywords if kw.lower() in content_lower
        )
        domain_matches = sum(
            1 for kw in config.domain_keywords if kw.lower() in content_lower
        )

        # Check book membership
        book_title = metadata.get("book_title", metadata.get("title", ""))
        in_whitelist_book = self._matches_patterns(
            book_title, config.book_whitelist_patterns
        )
        in_blacklist_book = self._matches_patterns(
            book_title, config.book_blacklist_patterns
        )

        # Check tier membership
        tier = metadata.get("tier", metadata.get("tier_id", ""))
        in_whitelist_tier = tier in config.tier_whitelist if config.tier_whitelist else False

        # Compute score adjustment
        adjusted_score = original_score
        reasons: list[str] = []

        # Book whitelist/blacklist
        if in_whitelist_book:
            adj = adjustments.get("in_whitelist_book", 0.0)
            adjusted_score += adj
            if adj != 0:
                reasons.append(f"whitelist_book:{adj:+.2f}")

        if in_blacklist_book:
            adj = adjustments.get("in_blacklist_book", 0.0)
            adjusted_score += adj
            if adj != 0:
                reasons.append(f"blacklist_book:{adj:+.2f}")

        # Tier whitelist
        if in_whitelist_tier:
            adj = adjustments.get("in_whitelist_tier", 0.0)
            adjusted_score += adj
            if adj != 0:
                reasons.append(f"whitelist_tier:{adj:+.2f}")

        # Domain keyword presence bonus
        if domain_matches > 0:
            adj = adjustments.get("domain_keyword_present", 0.0) * min(domain_matches, 3)
            adjusted_score += adj
            if adj != 0:
                reasons.append(f"domain_keywords({domain_matches}):{adj:+.2f}")

        # Primary keyword without domain context penalty
        if primary_matches > 0 and domain_matches < config.min_domain_matches:
            adj = adjustments.get("primary_only_no_domain", -0.3)
            adjusted_score += adj
            reasons.append(f"primary_only_no_domain:{adj:+.2f}")

        # Clamp score
        adjusted_score = max(0.0, min(1.0, adjusted_score))

        # Determine if passage should be kept
        keep = True
        if in_blacklist_book and not in_whitelist_book:
            # In blacklist but not whitelist - more strict
            if domain_matches < config.min_domain_matches:
                keep = False
                reasons.append("rejected:blacklist_no_domain")

        if primary_matches > 0 and domain_matches < config.min_domain_matches:
            # Has primary keyword but no domain context - likely false positive
            if in_blacklist_book:
                keep = False
                reasons.append("rejected:primary_only_blacklist")

        return FilterResult(
            doc_id=doc_id,
            keep=keep,
            original_score=original_score,
            adjusted_score=adjusted_score,
            primary_matches=primary_matches,
            domain_matches=domain_matches,
            in_whitelist_book=in_whitelist_book,
            in_blacklist_book=in_blacklist_book,
            in_whitelist_tier=in_whitelist_tier,
            adjustment_reasons=reasons,
        )

    def _matches_patterns(self, text: str, patterns: list[str]) -> bool:
        """Check if text matches any of the glob patterns.

        Args:
            text: Text to check
            patterns: List of glob patterns (e.g., "Game Programming Gems*")

        Returns:
            True if text matches any pattern
        """
        if not text or not patterns:
            return False

        for pattern in patterns:
            if fnmatch.fnmatch(text, pattern):
                return True
            # Also try case-insensitive
            if fnmatch.fnmatch(text.lower(), pattern.lower()):
                return True

        return False


# =============================================================================
# Convenience Functions
# =============================================================================


def create_filter(config_path: Path | None = None) -> MetadataFilter:
    """Create and initialize a MetadataFilter.

    Args:
        config_path: Optional path to config file

    Returns:
        Initialized MetadataFilter
    """
    filter_instance = MetadataFilter(config_path)
    filter_instance.load_config()
    return filter_instance


def filter_for_domain(
    passages: list[dict[str, Any]],
    domain: str,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Convenience function to filter passages for a domain.

    Args:
        passages: List of passage dicts
        domain: Domain name (e.g., 'llm_rag')
        config_path: Optional path to config file

    Returns:
        Filtered passages
    """
    filter_instance = create_filter(config_path)
    return filter_instance.apply(passages, domain)
