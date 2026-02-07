"""
Query Expansion Module for Abbreviation ↔ Full Form Resolution.

This module provides bidirectional query expansion for CS abbreviations:
- When searching "llm", also search "large language model"
- When searching "large language model", also search "llm"

Usage:
    from src.search.query_expansion import QueryExpander
    
    expander = QueryExpander()
    expanded = expander.expand("How do LLMs handle context?")
    # Returns: ["How do LLMs handle context?", "How do large language models handle context?"]

Design Principles:
- Loads abbreviation dictionary at startup (O(1) lookups)
- Case-insensitive matching
- Returns original + expanded variants
- Supports both single terms and full queries
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default path to abbreviation dictionary
DEFAULT_DICT_PATH = Path(__file__).parent.parent.parent / "config" / "abbreviation_expansion.json"


class QueryExpander:
    """
    Expands queries by replacing abbreviations with full forms and vice versa.
    
    This enables finding content regardless of whether it uses abbreviations
    or full terminology.
    
    Example:
        >>> expander = QueryExpander()
        >>> expander.expand_term("llm")
        ['llm', 'large language model']
        >>> expander.expand_query("What is an API gateway?")
        ['What is an API gateway?', 'What is an application programming interface gateway?']
    """
    
    __slots__ = ("_abbreviations", "_full_forms", "_loaded", "_dict_path")
    
    def __init__(self, dict_path: Path | str | None = None) -> None:
        """
        Initialize the query expander.
        
        Args:
            dict_path: Path to abbreviation_expansion.json. If None, uses default.
        """
        self._dict_path = Path(dict_path) if dict_path else DEFAULT_DICT_PATH
        self._abbreviations: dict[str, str] = {}  # abbrev -> full_form
        self._full_forms: dict[str, str] = {}      # full_form -> abbrev
        self._loaded = False
        
    def _load_dictionary(self) -> None:
        """Lazily load the abbreviation dictionary."""
        if self._loaded:
            return
            
        if not self._dict_path.exists():
            logger.warning("Abbreviation dictionary not found: %s", self._dict_path)
            self._loaded = True
            return
            
        try:
            with open(self._dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Build lookup maps
            for abbrev, info in data.get("abbreviations", {}).items():
                self._abbreviations[abbrev.lower()] = info["full_form"].lower()
            
            for full_form, info in data.get("full_forms", {}).items():
                self._full_forms[full_form.lower()] = info["abbreviation"].lower()
            
            logger.info(
                "Loaded abbreviation dictionary: %d abbreviations, %d full forms",
                len(self._abbreviations),
                len(self._full_forms),
            )
            self._loaded = True
            
        except Exception as e:
            logger.error("Failed to load abbreviation dictionary: %s", e)
            self._loaded = True
    
    @property
    def abbreviation_count(self) -> int:
        """Get number of loaded abbreviations."""
        self._load_dictionary()
        return len(self._abbreviations)
    
    @property
    def is_loaded(self) -> bool:
        """Check if dictionary was successfully loaded."""
        self._load_dictionary()
        return len(self._abbreviations) > 0
    
    def get_full_form(self, abbreviation: str) -> str | None:
        """
        Get the full form for an abbreviation.
        
        Args:
            abbreviation: The abbreviation to look up (e.g., "llm")
            
        Returns:
            Full form if found (e.g., "large language model"), None otherwise
        """
        self._load_dictionary()
        return self._abbreviations.get(abbreviation.lower())
    
    def get_abbreviation(self, full_form: str) -> str | None:
        """
        Get the abbreviation for a full form.
        
        Args:
            full_form: The full form to look up (e.g., "large language model")
            
        Returns:
            Abbreviation if found (e.g., "llm"), None otherwise
        """
        self._load_dictionary()
        return self._full_forms.get(full_form.lower())
    
    def expand_term(self, term: str) -> list[str]:
        """
        Expand a single term to include its counterpart.
        
        Args:
            term: Single word or phrase to expand
            
        Returns:
            List containing original term and expansion (if found)
        """
        self._load_dictionary()
        term_lower = term.lower().strip()
        
        # Check if it's an abbreviation
        full_form = self._abbreviations.get(term_lower)
        if full_form:
            return [term, full_form]
        
        # Check if it's a full form
        abbreviation = self._full_forms.get(term_lower)
        if abbreviation:
            return [term, abbreviation]
        
        # No expansion found
        return [term]
    
    def expand_query(self, query: str, max_expansions: int = 3) -> list[str]:
        """
        Expand a full query by replacing abbreviations/full forms.
        
        Args:
            query: The search query to expand
            max_expansions: Maximum number of expanded variants to return
            
        Returns:
            List of query variants (original + expansions)
        """
        self._load_dictionary()
        
        if not query or not query.strip():
            return [query]
        
        results = [query]
        query_lower = query.lower()
        
        # Find all abbreviations in the query
        expansions_made = 0
        
        # Check for abbreviations (word boundaries)
        for abbrev, full_form in self._abbreviations.items():
            if expansions_made >= max_expansions:
                break
                
            # Use word boundary matching
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, query_lower, re.IGNORECASE):
                # Create expanded version preserving case
                expanded = re.sub(
                    pattern,
                    full_form,
                    query,
                    flags=re.IGNORECASE,
                )
                if expanded != query and expanded not in results:
                    results.append(expanded)
                    expansions_made += 1
        
        # Check for full forms (multi-word matching)
        for full_form, abbreviation in self._full_forms.items():
            if expansions_made >= max_expansions:
                break
                
            # Check if full form appears in query
            pattern = r'\b' + re.escape(full_form) + r'\b'
            if re.search(pattern, query_lower, re.IGNORECASE):
                # Create abbreviated version
                abbreviated = re.sub(
                    pattern,
                    abbreviation.upper(),  # Abbreviations typically uppercase
                    query,
                    flags=re.IGNORECASE,
                )
                if abbreviated != query and abbreviated not in results:
                    results.append(abbreviated)
                    expansions_made += 1
        
        return results
    
    def get_all_variants(self, term: str) -> dict[str, Any]:
        """
        Get all information about a term including variants.
        
        Args:
            term: Term to look up
            
        Returns:
            Dictionary with term info:
            {
                "original": "llm",
                "is_abbreviation": True,
                "is_full_form": False,
                "variants": ["llm", "large language model"],
                "full_form": "large language model",
                "abbreviation": None
            }
        """
        self._load_dictionary()
        term_lower = term.lower().strip()
        
        result = {
            "original": term,
            "is_abbreviation": False,
            "is_full_form": False,
            "variants": [term],
            "full_form": None,
            "abbreviation": None,
        }
        
        # Check if abbreviation
        full_form = self._abbreviations.get(term_lower)
        if full_form:
            result["is_abbreviation"] = True
            result["full_form"] = full_form
            result["variants"].append(full_form)
        
        # Check if full form
        abbreviation = self._full_forms.get(term_lower)
        if abbreviation:
            result["is_full_form"] = True
            result["abbreviation"] = abbreviation
            if abbreviation not in result["variants"]:
                result["variants"].append(abbreviation)
        
        return result


# Module-level singleton for convenience
_default_expander: QueryExpander | None = None


def get_query_expander() -> QueryExpander:
    """Get the default query expander instance (singleton)."""
    global _default_expander
    if _default_expander is None:
        _default_expander = QueryExpander()
    return _default_expander


def expand_query(query: str, max_expansions: int = 3) -> list[str]:
    """
    Convenience function to expand a query using the default expander.
    
    Args:
        query: The search query to expand
        max_expansions: Maximum number of expanded variants
        
    Returns:
        List of query variants
    """
    return get_query_expander().expand_query(query, max_expansions)


def expand_term(term: str) -> list[str]:
    """
    Convenience function to expand a term using the default expander.
    
    Args:
        term: Single term to expand
        
    Returns:
        List containing original and expansion (if found)
    """
    return get_query_expander().expand_term(term)
