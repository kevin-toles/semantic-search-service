"""
Taxonomy loader for Graph RAG.

Provides functionality to parse taxonomy JSON and load data into Neo4j graph database.
Supports the hierarchical structure: Taxonomy -> Books -> Chapters -> Concepts
with relationships to Tiers for categorization.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import Settings
from src.graph.health import get_neo4j_driver


@dataclass
class Chapter:
    """Domain object representing a book chapter."""

    id: str
    number: int
    title: str
    keywords: list[str]
    concepts: list[str]
    summary: str
    page_range: str
    book_id: str


@dataclass
class Book:
    """Domain object representing a book in the taxonomy."""

    id: str
    title: str
    author: str
    year: int
    tier: int
    category: str
    chapters: list[Chapter] = field(default_factory=list)


@dataclass
class Taxonomy:
    """Domain object representing the complete taxonomy structure."""

    id: str
    name: str
    description: str
    books: list[Book] = field(default_factory=list)


def parse_chapter(chapter_data: dict[str, Any], book_id: str) -> Chapter:
    """
    Parse chapter JSON into a Chapter domain object.

    Args:
        chapter_data: Dictionary containing chapter data
        book_id: ID of the parent book

    Returns:
        Chapter domain object
    """
    return Chapter(
        id=chapter_data["id"],
        number=chapter_data["number"],
        title=chapter_data["title"],
        keywords=chapter_data.get("keywords", []),
        concepts=chapter_data.get("concepts", []),
        summary=chapter_data.get("summary", ""),
        page_range=chapter_data.get("page_range", ""),
        book_id=book_id,
    )


def parse_book(book_data: dict[str, Any]) -> Book:
    """
    Parse book JSON into a Book domain object.

    Args:
        book_data: Dictionary containing book data

    Returns:
        Book domain object with parsed chapters
    """
    book_id = book_data["id"]
    chapters = [
        parse_chapter(ch, book_id) for ch in book_data.get("chapters", [])
    ]

    return Book(
        id=book_id,
        title=book_data["title"],
        author=book_data["author"],
        year=book_data["year"],
        tier=book_data["tier"],
        category=book_data.get("category", ""),
        chapters=chapters,
    )


def parse_taxonomy(data: dict[str, Any]) -> Taxonomy:
    """
    Parse taxonomy JSON into a Taxonomy domain object.

    Args:
        data: Dictionary containing full taxonomy data

    Returns:
        Taxonomy domain object with books and chapters
    """
    taxonomy_data = data["taxonomy"]
    books = [parse_book(b) for b in data.get("books", [])]

    return Taxonomy(
        id=taxonomy_data["id"],
        name=taxonomy_data["name"],
        description=taxonomy_data.get("description", ""),
        books=books,
    )


def load_taxonomy_from_file(file_path: Path) -> Taxonomy:
    """
    Load and parse taxonomy from a JSON file.

    Args:
        file_path: Path to the taxonomy JSON file

    Returns:
        Parsed Taxonomy domain object
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return parse_taxonomy(data)


def load_taxonomy_to_neo4j(taxonomy: Taxonomy, settings: Settings) -> None:
    """
    Load taxonomy data into Neo4j graph database.

    Creates nodes for:
    - Taxonomy (root node)
    - Books (linked to Taxonomy via CONTAINS)
    - Chapters (linked to Book via CONTAINS)
    - Concepts (linked to Chapter via COVERS)
    - Tier relationships (Book IN_TIER Tier)

    Args:
        taxonomy: Parsed Taxonomy domain object
        settings: Application settings with Neo4j configuration
    """
    driver = get_neo4j_driver(settings)

    try:
        with driver.session() as session:
            # Create/Update Taxonomy node
            session.run(
                """
                MERGE (t:Taxonomy {id: $id})
                SET t.name = $name, t.description = $description
                """,
                id=taxonomy.id,
                name=taxonomy.name,
                description=taxonomy.description,
            )

            # Create Book nodes and relationships
            for book in taxonomy.books:
                # Create Book node
                session.run(
                    """
                    MERGE (b:Book {id: $id})
                    SET b.title = $title,
                        b.author = $author,
                        b.year = $year,
                        b.tier = $tier,
                        b.category = $category
                    """,
                    id=book.id,
                    title=book.title,
                    author=book.author,
                    year=book.year,
                    tier=book.tier,
                    category=book.category,
                )

                # Link Book to Taxonomy
                session.run(
                    """
                    MATCH (t:Taxonomy {id: $taxonomy_id})
                    MATCH (b:Book {id: $book_id})
                    MERGE (t)-[:CONTAINS]->(b)
                    """,
                    taxonomy_id=taxonomy.id,
                    book_id=book.id,
                )

                # Link Book to Tier
                session.run(
                    """
                    MATCH (b:Book {id: $book_id})
                    MATCH (tier:Tier {level: $tier_level})
                    MERGE (b)-[:IN_TIER]->(tier)
                    """,
                    book_id=book.id,
                    tier_level=book.tier,
                )

                # Create Chapter nodes and relationships
                for chapter in book.chapters:
                    # Create Chapter node
                    session.run(
                        """
                        MERGE (c:Chapter {id: $id})
                        SET c.number = $number,
                            c.title = $title,
                            c.keywords = $keywords,
                            c.summary = $summary,
                            c.page_range = $page_range,
                            c.book_id = $book_id
                        """,
                        id=chapter.id,
                        number=chapter.number,
                        title=chapter.title,
                        keywords=chapter.keywords,
                        summary=chapter.summary,
                        page_range=chapter.page_range,
                        book_id=chapter.book_id,
                    )

                    # Link Chapter to Book
                    session.run(
                        """
                        MATCH (b:Book {id: $book_id})
                        MATCH (c:Chapter {id: $chapter_id})
                        MERGE (b)-[:CONTAINS]->(c)
                        """,
                        book_id=book.id,
                        chapter_id=chapter.id,
                    )

                    # Create Concept nodes and relationships
                    for concept_name in chapter.concepts:
                        session.run(
                            """
                            MERGE (concept:Concept {name: $name})
                            WITH concept
                            MATCH (c:Chapter {id: $chapter_id})
                            MERGE (c)-[:COVERS]->(concept)
                            """,
                            name=concept_name,
                            chapter_id=chapter.id,
                        )
    finally:
        driver.close()
