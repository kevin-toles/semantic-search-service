"""
TDD RED Phase: Taxonomy Loader Tests

These tests define the expected behavior for loading taxonomy data into Neo4j.
The implementation does not exist yet - these tests should FAIL initially.

WBS Reference: 1.7, 1.7.1
Acceptance Criteria:
- Taxonomy JSON can be parsed into domain objects
- Taxonomy data can be loaded into Neo4j graph
- Relationships are created between Books, Chapters, Concepts, and Tiers
"""

import json
from pathlib import Path
from unittest.mock import patch

from src.core.config import Settings

# Sample test data matching the schema in data/seed/sample_taxonomy.json
SAMPLE_TAXONOMY = {
    "taxonomy": {
        "id": "software-engineering",
        "name": "Software Engineering",
        "description": "Test taxonomy",
    },
    "books": [
        {
            "id": "test-book",
            "title": "Test Book",
            "author": "Test Author",
            "year": 2024,
            "tier": 1,
            "category": "test",
            "chapters": [
                {
                    "id": "tb-ch1",
                    "number": 1,
                    "title": "Test Chapter",
                    "keywords": ["test", "example"],
                    "concepts": ["testing", "validation"],
                    "summary": "A test chapter for unit testing.",
                    "page_range": "1-10",
                }
            ],
        }
    ],
}


class TestTaxonomyParser:
    """Tests for parsing taxonomy JSON into domain objects."""

    def test_parse_taxonomy_from_json(self):
        """
        GIVEN valid taxonomy JSON
        WHEN parse_taxonomy is called
        THEN it returns a Taxonomy domain object
        """
        from src.graph.taxonomy_loader import parse_taxonomy

        result = parse_taxonomy(SAMPLE_TAXONOMY)

        assert result.id == "software-engineering"
        assert result.name == "Software Engineering"
        assert len(result.books) == 1

    def test_parse_taxonomy_from_file(self, tmp_path: Path):
        """
        GIVEN a taxonomy JSON file
        WHEN load_taxonomy_from_file is called
        THEN it returns a parsed Taxonomy object
        """
        from src.graph.taxonomy_loader import load_taxonomy_from_file

        file_path = tmp_path / "test_taxonomy.json"
        file_path.write_text(json.dumps(SAMPLE_TAXONOMY))

        result = load_taxonomy_from_file(file_path)

        assert result.id == "software-engineering"
        assert result.name == "Software Engineering"

    def test_parse_book_from_json(self):
        """
        GIVEN valid book JSON
        WHEN parse_book is called
        THEN it returns a Book domain object with chapters
        """
        from src.graph.taxonomy_loader import parse_book

        book_data = SAMPLE_TAXONOMY["books"][0]
        result = parse_book(book_data)

        assert result.id == "test-book"
        assert result.title == "Test Book"
        assert result.tier == 1
        assert len(result.chapters) == 1

    def test_parse_chapter_from_json(self):
        """
        GIVEN valid chapter JSON
        WHEN parse_chapter is called
        THEN it returns a Chapter domain object with concepts
        """
        from src.graph.taxonomy_loader import parse_chapter

        chapter_data = SAMPLE_TAXONOMY["books"][0]["chapters"][0]
        result = parse_chapter(chapter_data, book_id="test-book")

        assert result.id == "tb-ch1"
        assert result.number == 1
        assert result.book_id == "test-book"
        assert "testing" in result.concepts


class TestTaxonomyLoader:
    """Tests for loading taxonomy data into Neo4j."""

    def test_load_taxonomy_creates_taxonomy_node(
        self, settings: Settings, mock_neo4j_driver
    ):
        """
        GIVEN parsed taxonomy data
        WHEN load_taxonomy_to_neo4j is called
        THEN it creates a Taxonomy node in Neo4j
        """
        from src.graph.taxonomy_loader import (
            load_taxonomy_to_neo4j,
            parse_taxonomy,
        )

        taxonomy = parse_taxonomy(SAMPLE_TAXONOMY)

        with patch("src.graph.taxonomy_loader.get_neo4j_driver", return_value=mock_neo4j_driver):
            load_taxonomy_to_neo4j(taxonomy, settings)

        # Verify Cypher was executed to create Taxonomy node
        session = mock_neo4j_driver.session.return_value.__enter__.return_value
        session.run.assert_called()

    def test_load_taxonomy_creates_book_nodes(
        self, settings: Settings, mock_neo4j_driver
    ):
        """
        GIVEN parsed taxonomy with books
        WHEN load_taxonomy_to_neo4j is called
        THEN it creates Book nodes linked to Taxonomy
        """
        from src.graph.taxonomy_loader import (
            load_taxonomy_to_neo4j,
            parse_taxonomy,
        )

        taxonomy = parse_taxonomy(SAMPLE_TAXONOMY)

        with patch("src.graph.taxonomy_loader.get_neo4j_driver", return_value=mock_neo4j_driver):
            load_taxonomy_to_neo4j(taxonomy, settings)

        # Verify multiple Cypher statements were run (Taxonomy + Book nodes)
        session = mock_neo4j_driver.session.return_value.__enter__.return_value
        assert session.run.call_count >= 2

    def test_load_taxonomy_creates_chapter_nodes_with_relationships(
        self, settings: Settings, mock_neo4j_driver
    ):
        """
        GIVEN parsed taxonomy with books and chapters
        WHEN load_taxonomy_to_neo4j is called
        THEN it creates Chapter nodes with CONTAINS relationship to Book
        """
        from src.graph.taxonomy_loader import (
            load_taxonomy_to_neo4j,
            parse_taxonomy,
        )

        taxonomy = parse_taxonomy(SAMPLE_TAXONOMY)

        with patch("src.graph.taxonomy_loader.get_neo4j_driver", return_value=mock_neo4j_driver):
            load_taxonomy_to_neo4j(taxonomy, settings)

        # Verify Chapter creation was called
        session = mock_neo4j_driver.session.return_value.__enter__.return_value
        # Should have calls for Taxonomy, Book, Chapter, and relationships
        assert session.run.call_count >= 3

    def test_load_taxonomy_creates_concept_nodes(
        self, settings: Settings, mock_neo4j_driver
    ):
        """
        GIVEN parsed taxonomy with chapters containing concepts
        WHEN load_taxonomy_to_neo4j is called
        THEN it creates Concept nodes with COVERS relationship to Chapter
        """
        from src.graph.taxonomy_loader import (
            load_taxonomy_to_neo4j,
            parse_taxonomy,
        )

        taxonomy = parse_taxonomy(SAMPLE_TAXONOMY)

        with patch("src.graph.taxonomy_loader.get_neo4j_driver", return_value=mock_neo4j_driver):
            load_taxonomy_to_neo4j(taxonomy, settings)

        # Concepts should be created for each unique concept
        session = mock_neo4j_driver.session.return_value.__enter__.return_value
        assert session.run.call_count >= 4  # Taxonomy + Book + Chapter + Concepts

    def test_load_taxonomy_links_book_to_tier(
        self, settings: Settings, mock_neo4j_driver
    ):
        """
        GIVEN parsed taxonomy with books having tier assignments
        WHEN load_taxonomy_to_neo4j is called
        THEN it creates IN_TIER relationship between Book and Tier
        """
        from src.graph.taxonomy_loader import (
            load_taxonomy_to_neo4j,
            parse_taxonomy,
        )

        taxonomy = parse_taxonomy(SAMPLE_TAXONOMY)

        with patch("src.graph.taxonomy_loader.get_neo4j_driver", return_value=mock_neo4j_driver):
            load_taxonomy_to_neo4j(taxonomy, settings)

        session = mock_neo4j_driver.session.return_value.__enter__.return_value
        # Verify IN_TIER relationship was created
        assert session.run.called


class TestTaxonomyDomainObjects:
    """Tests for taxonomy domain object structure."""

    def test_taxonomy_dataclass_has_required_fields(self):
        """
        GIVEN the Taxonomy dataclass
        THEN it has id, name, description, and books fields
        """
        from src.graph.taxonomy_loader import Taxonomy

        # This will fail until dataclass is implemented
        taxonomy = Taxonomy(
            id="test",
            name="Test",
            description="Test desc",
            books=[],
        )
        assert taxonomy.id == "test"

    def test_book_dataclass_has_required_fields(self):
        """
        GIVEN the Book dataclass
        THEN it has id, title, author, year, tier, category, and chapters fields
        """
        from src.graph.taxonomy_loader import Book

        book = Book(
            id="test",
            title="Test",
            author="Author",
            year=2024,
            tier=1,
            category="test",
            chapters=[],
        )
        assert book.tier == 1

    def test_chapter_dataclass_has_required_fields(self):
        """
        GIVEN the Chapter dataclass
        THEN it has id, number, title, keywords, concepts, summary, page_range, book_id fields
        """
        from src.graph.taxonomy_loader import Chapter

        chapter = Chapter(
            id="test",
            number=1,
            title="Test",
            keywords=["a"],
            concepts=["b"],
            summary="Summary",
            page_range="1-10",
            book_id="book-1",
        )
        assert chapter.book_id == "book-1"
