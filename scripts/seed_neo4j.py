#!/usr/bin/env python3
"""
WBS 3.2.2: Seed Neo4j Taxonomy Graph

Seeds the Neo4j graph database with:
- Tier nodes (T1, T2, T3)
- Book nodes with metadata
- Chapter nodes linked to books
- PARALLEL relationships (same tier)
- PERPENDICULAR relationships (adjacent tiers)
- BELONGS_TO relationships (chapter → book, book → tier)

Usage:
    python scripts/seed_neo4j.py --json /path/to/book.json
    python scripts/seed_neo4j.py --taxonomy /path/to/taxonomy.json
    python scripts/seed_neo4j.py --sample  # Use sample data

Environment Variables:
    NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
    NEO4J_USER: Neo4j username (default: neo4j)
    NEO4J_PASSWORD: Neo4j password (required)
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable


# =============================================================================
# Configuration
# =============================================================================

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "devpassword")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TierData:
    """Tier node data."""
    id: str
    name: str
    level: int
    description: str


@dataclass
class ChapterData:
    """Chapter node data."""
    id: str
    number: int
    title: str
    keywords: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    summary: str = ""
    page_range: str = ""


@dataclass
class BookData:
    """Book node data."""
    id: str
    title: str
    author: str
    year: int
    tier: int
    category: str
    chapters: list[ChapterData] = field(default_factory=list)


# =============================================================================
# Sample Data for Testing
# =============================================================================

SAMPLE_TIERS = [
    TierData(
        id="T1",
        name="Tier 1 - Architecture",
        level=1,
        description="High-level architecture patterns, domain modeling, strategic design"
    ),
    TierData(
        id="T2",
        name="Tier 2 - Practices",
        level=2,
        description="Implementation practices, patterns, tactical design"
    ),
    TierData(
        id="T3",
        name="Tier 3 - Implementation",
        level=3,
        description="Language-specific implementation, libraries, tools"
    ),
]

SAMPLE_BOOKS = [
    BookData(
        id="architecture-patterns-python",
        title="Architecture Patterns with Python",
        author="Harry Percival and Bob Gregory",
        year=2020,
        tier=1,
        category="architecture",
        chapters=[
            ChapterData(
                id="app-ch1",
                number=1,
                title="Domain Modeling",
                keywords=["domain model", "entity", "value object", "aggregate"],
                concepts=["Domain-Driven Design", "Ubiquitous Language"],
            ),
            ChapterData(
                id="app-ch2",
                number=2,
                title="Repository Pattern",
                keywords=["repository", "persistence", "abstraction", "ORM"],
                concepts=["Repository Pattern", "Data Mapper"],
            ),
        ]
    ),
    BookData(
        id="building-microservices",
        title="Building Microservices",
        author="Sam Newman",
        year=2021,
        tier=1,
        category="architecture",
        chapters=[
            ChapterData(
                id="bm-ch1",
                number=1,
                title="What Are Microservices?",
                keywords=["microservices", "distributed", "bounded context"],
                concepts=["Microservices Architecture", "Service Decomposition"],
            ),
            ChapterData(
                id="bm-ch2",
                number=2,
                title="Service Communication",
                keywords=["REST", "gRPC", "messaging", "async"],
                concepts=["Synchronous Communication", "Asynchronous Messaging"],
            ),
        ]
    ),
    BookData(
        id="fluent-python",
        title="Fluent Python",
        author="Luciano Ramalho",
        year=2022,
        tier=3,
        category="implementation",
        chapters=[
            ChapterData(
                id="fp-ch1",
                number=1,
                title="The Python Data Model",
                keywords=["dunder", "magic methods", "protocol", "duck typing"],
                concepts=["Python Data Model", "Duck Typing"],
            ),
            ChapterData(
                id="fp-ch2",
                number=2,
                title="Sequences",
                keywords=["list", "tuple", "slice", "comprehension"],
                concepts=["Sequence Protocol", "List Comprehension"],
            ),
        ]
    ),
]


# =============================================================================
# Neo4j Seeding Functions
# =============================================================================

async def clear_database(driver) -> None:
    """Clear all nodes and relationships from the database."""
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    print("✓ Cleared existing data")


async def create_constraints(driver) -> None:
    """Create uniqueness constraints for node IDs."""
    constraints = [
        "CREATE CONSTRAINT tier_id IF NOT EXISTS FOR (t:Tier) REQUIRE t.id IS UNIQUE",
        "CREATE CONSTRAINT book_id IF NOT EXISTS FOR (b:Book) REQUIRE b.id IS UNIQUE",
        "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.id IS UNIQUE",
    ]
    
    async with driver.session() as session:
        for constraint in constraints:
            try:
                await session.run(constraint)
            except Exception:
                pass  # Constraint may already exist
    print("✓ Created constraints")


async def create_tiers(driver, tiers: list[TierData]) -> None:
    """Create Tier nodes."""
    async with driver.session() as session:
        for tier in tiers:
            await session.run(
                """
                MERGE (t:Tier {id: $id})
                SET t.name = $name,
                    t.level = $level,
                    t.description = $description
                """,
                id=tier.id,
                name=tier.name,
                level=tier.level,
                description=tier.description,
            )
    print(f"✓ Created {len(tiers)} Tier nodes")


async def create_books(driver, books: list[BookData]) -> None:
    """Create Book nodes and link to Tiers."""
    async with driver.session() as session:
        for book in books:
            # Create book node
            await session.run(
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
            
            # Link book to tier
            tier_id = f"T{book.tier}"
            await session.run(
                """
                MATCH (b:Book {id: $book_id})
                MATCH (t:Tier {id: $tier_id})
                MERGE (b)-[:BELONGS_TO]->(t)
                """,
                book_id=book.id,
                tier_id=tier_id,
            )
    print(f"✓ Created {len(books)} Book nodes")


async def create_chapters(driver, books: list[BookData]) -> None:
    """Create Chapter nodes and link to Books."""
    total_chapters = 0
    
    async with driver.session() as session:
        for book in books:
            for chapter in book.chapters:
                # Create chapter node
                await session.run(
                    """
                    MERGE (c:Chapter {id: $id})
                    SET c.number = $number,
                        c.title = $title,
                        c.keywords = $keywords,
                        c.concepts = $concepts,
                        c.summary = $summary,
                        c.page_range = $page_range,
                        c.book_id = $book_id
                    """,
                    id=chapter.id,
                    number=chapter.number,
                    title=chapter.title,
                    keywords=chapter.keywords,
                    concepts=chapter.concepts,
                    summary=chapter.summary,
                    page_range=chapter.page_range,
                    book_id=book.id,
                )
                
                # Link chapter to book
                await session.run(
                    """
                    MATCH (c:Chapter {id: $chapter_id})
                    MATCH (b:Book {id: $book_id})
                    MERGE (c)-[:BELONGS_TO]->(b)
                    """,
                    chapter_id=chapter.id,
                    book_id=book.id,
                )
                total_chapters += 1
    
    print(f"✓ Created {total_chapters} Chapter nodes")


async def create_parallel_relationships(driver, books: list[BookData]) -> None:
    """Create PARALLEL relationships between books in the same tier."""
    async with driver.session() as session:
        # Group books by tier
        books_by_tier: dict[int, list[str]] = {}
        for book in books:
            books_by_tier.setdefault(book.tier, []).append(book.id)
        
        # Create PARALLEL relationships within each tier
        relationship_count = 0
        for tier, book_ids in books_by_tier.items():
            for i, book_id_1 in enumerate(book_ids):
                for book_id_2 in book_ids[i + 1:]:
                    await session.run(
                        """
                        MATCH (b1:Book {id: $id1})
                        MATCH (b2:Book {id: $id2})
                        MERGE (b1)-[:PARALLEL]->(b2)
                        MERGE (b2)-[:PARALLEL]->(b1)
                        """,
                        id1=book_id_1,
                        id2=book_id_2,
                    )
                    relationship_count += 2
        
        print(f"✓ Created {relationship_count} PARALLEL relationships")


async def create_perpendicular_relationships(driver, books: list[BookData]) -> None:
    """Create PERPENDICULAR relationships between books in adjacent tiers."""
    async with driver.session() as session:
        # Group books by tier
        books_by_tier: dict[int, list[str]] = {}
        for book in books:
            books_by_tier.setdefault(book.tier, []).append(book.id)
        
        # Create PERPENDICULAR relationships between adjacent tiers
        relationship_count = 0
        tiers = sorted(books_by_tier.keys())
        
        for i, tier in enumerate(tiers[:-1]):
            next_tier = tiers[i + 1]
            
            for book_id_1 in books_by_tier[tier]:
                for book_id_2 in books_by_tier[next_tier]:
                    await session.run(
                        """
                        MATCH (b1:Book {id: $id1})
                        MATCH (b2:Book {id: $id2})
                        MERGE (b1)-[:PERPENDICULAR]->(b2)
                        MERGE (b2)-[:PERPENDICULAR]->(b1)
                        """,
                        id1=book_id_1,
                        id2=book_id_2,
                    )
                    relationship_count += 2
        
        print(f"✓ Created {relationship_count} PERPENDICULAR relationships")


async def create_chapter_relationships(driver, books: list[BookData]) -> None:
    """Create relationships between related chapters based on shared keywords."""
    async with driver.session() as session:
        # Collect all chapters with their keywords
        all_chapters: list[tuple[str, set[str]]] = []
        for book in books:
            for chapter in book.chapters:
                keywords_set = set(kw.lower() for kw in chapter.keywords)
                all_chapters.append((chapter.id, keywords_set))
        
        # Create RELATED_TO relationships for chapters with shared keywords
        relationship_count = 0
        for i, (ch1_id, kw1) in enumerate(all_chapters):
            for ch2_id, kw2 in all_chapters[i + 1:]:
                shared = kw1 & kw2
                if len(shared) >= 1:  # At least one shared keyword
                    await session.run(
                        """
                        MATCH (c1:Chapter {id: $id1})
                        MATCH (c2:Chapter {id: $id2})
                        MERGE (c1)-[:RELATED_TO {shared_keywords: $shared}]->(c2)
                        MERGE (c2)-[:RELATED_TO {shared_keywords: $shared}]->(c1)
                        """,
                        id1=ch1_id,
                        id2=ch2_id,
                        shared=list(shared),
                    )
                    relationship_count += 2
        
        print(f"✓ Created {relationship_count} RELATED_TO relationships between chapters")


async def print_summary(driver) -> None:
    """Print summary of seeded data."""
    async with driver.session() as session:
        # Count nodes
        result = await session.run("MATCH (t:Tier) RETURN count(t) AS count")
        tier_count = (await result.single())["count"]
        
        result = await session.run("MATCH (b:Book) RETURN count(b) AS count")
        book_count = (await result.single())["count"]
        
        result = await session.run("MATCH (c:Chapter) RETURN count(c) AS count")
        chapter_count = (await result.single())["count"]
        
        # Count relationships
        result = await session.run(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC"
        )
        rel_counts = [(record["type"], record["count"]) async for record in result]
        
        print("\n" + "=" * 50)
        print("SEEDING SUMMARY")
        print("=" * 50)
        print(f"Tier nodes:    {tier_count}")
        print(f"Book nodes:    {book_count}")
        print(f"Chapter nodes: {chapter_count}")
        print("\nRelationships:")
        for rel_type, count in rel_counts:
            print(f"  {rel_type}: {count}")
        print("=" * 50)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_book_from_json(json_path: Path) -> BookData:
    """Load a single book from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    chapters = []
    for ch_data in data.get("chapters", []):
        chapters.append(ChapterData(
            id=ch_data.get("id", f"ch-{ch_data.get('number', 0)}"),
            number=ch_data.get("number", 0),
            title=ch_data.get("title", ""),
            keywords=ch_data.get("keywords", []),
            concepts=ch_data.get("concepts", []),
            summary=ch_data.get("summary", ""),
            page_range=ch_data.get("page_range", ""),
        ))
    
    return BookData(
        id=data.get("id", json_path.stem),
        title=data.get("title", data.get("book", {}).get("title", "Unknown")),
        author=data.get("author", data.get("book", {}).get("author", "Unknown")),
        year=data.get("year", data.get("book", {}).get("year", 2024)),
        tier=data.get("tier", data.get("book", {}).get("tier", 2)),
        category=data.get("category", "general"),
        chapters=chapters,
    )


def load_taxonomy_from_json(json_path: Path) -> tuple[list[TierData], list[BookData]]:
    """Load complete taxonomy from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    tiers = []
    books = []
    
    # Parse tiers if present
    if "tiers" in data:
        for t_data in data["tiers"]:
            tiers.append(TierData(
                id=t_data["id"],
                name=t_data["name"],
                level=t_data["level"],
                description=t_data.get("description", ""),
            ))
    else:
        tiers = SAMPLE_TIERS
    
    # Parse books
    for b_data in data.get("books", []):
        chapters = []
        for ch_data in b_data.get("chapters", []):
            chapters.append(ChapterData(
                id=ch_data.get("id", f"ch-{ch_data.get('number', 0)}"),
                number=ch_data.get("number", 0),
                title=ch_data.get("title", ""),
                keywords=ch_data.get("keywords", []),
                concepts=ch_data.get("concepts", []),
                summary=ch_data.get("summary", ""),
                page_range=ch_data.get("page_range", ""),
            ))
        
        books.append(BookData(
            id=b_data.get("id", b_data["title"].lower().replace(" ", "-")),
            title=b_data["title"],
            author=b_data.get("author", "Unknown"),
            year=b_data.get("year", 2024),
            tier=b_data.get("tier", 2),
            category=b_data.get("category", "general"),
            chapters=chapters,
        ))
    
    return tiers, books


# =============================================================================
# Main Entry Point
# =============================================================================

async def seed_database(
    tiers: list[TierData],
    books: list[BookData],
    clear: bool = True,
) -> None:
    """Seed the Neo4j database with taxonomy data."""
    print(f"\nConnecting to Neo4j at {NEO4J_URI}...")
    
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    
    try:
        # Verify connection
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS test")
            await result.single()
        print("✓ Connected to Neo4j")
        
        # Seed data
        if clear:
            await clear_database(driver)
        
        await create_constraints(driver)
        await create_tiers(driver, tiers)
        await create_books(driver, books)
        await create_chapters(driver, books)
        await create_parallel_relationships(driver, books)
        await create_perpendicular_relationships(driver, books)
        await create_chapter_relationships(driver, books)
        await print_summary(driver)
        
        print("\n✓ Seeding complete!")
        
    except ServiceUnavailable as e:
        print(f"\n✗ Failed to connect to Neo4j: {e}")
        print("  Make sure Neo4j is running on", NEO4J_URI)
        sys.exit(1)
    finally:
        await driver.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed Neo4j taxonomy graph database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--json",
        type=Path,
        help="Path to a single book JSON file",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        help="Path to a complete taxonomy JSON file",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample data for testing",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing data before seeding",
    )
    
    args = parser.parse_args()
    
    # Determine data source
    if args.sample:
        tiers = SAMPLE_TIERS
        books = SAMPLE_BOOKS
        print("Using sample data...")
    elif args.taxonomy:
        if not args.taxonomy.exists():
            print(f"✗ Taxonomy file not found: {args.taxonomy}")
            sys.exit(1)
        tiers, books = load_taxonomy_from_json(args.taxonomy)
        print(f"Loaded taxonomy from {args.taxonomy}")
    elif args.json:
        if not args.json.exists():
            print(f"✗ JSON file not found: {args.json}")
            sys.exit(1)
        tiers = SAMPLE_TIERS
        books = [load_book_from_json(args.json)]
        print(f"Loaded book from {args.json}")
    else:
        # Default to sample data
        tiers = SAMPLE_TIERS
        books = SAMPLE_BOOKS
        print("No input specified, using sample data...")
    
    # Run seeding
    asyncio.run(seed_database(tiers, books, clear=not args.no_clear))


if __name__ == "__main__":
    main()
