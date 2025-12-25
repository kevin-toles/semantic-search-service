#!/usr/bin/env python3
"""
WBS 3.2.1: Seed Qdrant with Test Data

Seeds the Qdrant vector database with:
- Chapter embeddings
- Metadata payloads (book, chapter_num, title, tier, keywords)

Usage:
    python scripts/seed_qdrant.py --json /path/to/book.json
    python scripts/seed_qdrant.py --sample  # Use sample data
    python scripts/seed_qdrant.py --dir /path/to/json/directory

Environment Variables:
    QDRANT_URL: Qdrant connection URL (default: http://localhost:6333)
    EMBEDDING_MODEL: SentenceTransformer model name (default: all-mpnet-base-v2)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "chapters")
VECTOR_SIZE = 768  # all-mpnet-base-v2 produces 768-dim vectors


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ChapterData:
    """Chapter data for embedding."""
    id: str
    number: int
    title: str
    content: str
    keywords: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    summary: str = ""
    page_range: str = ""
    book_id: str = ""
    book_title: str = ""
    book_author: str = ""
    tier: int = 2


# =============================================================================
# Sample Data for Testing
# =============================================================================

SAMPLE_CHAPTERS = [
    ChapterData(
        id="app-ch1",
        number=1,
        title="Domain Modeling",
        content="""
        Domain modeling is the process of creating a conceptual model of a system's domain.
        It involves identifying entities, value objects, and aggregates.
        Domain-Driven Design (DDD) provides patterns for effective domain modeling.
        The ubiquitous language ensures that business and technical teams communicate effectively.
        Entities have identity and lifecycle, while value objects are immutable and defined by their attributes.
        Aggregates define consistency boundaries and ensure transactional integrity.
        """,
        keywords=["domain model", "entity", "value object", "aggregate", "DDD"],
        concepts=["Domain-Driven Design", "Ubiquitous Language", "Aggregate Root"],
        book_id="architecture-patterns-python",
        book_title="Architecture Patterns with Python",
        book_author="Harry Percival and Bob Gregory",
        tier=1,
    ),
    ChapterData(
        id="app-ch2",
        number=2,
        title="Repository Pattern",
        content="""
        The Repository pattern provides an abstraction over data persistence.
        It allows the domain model to remain ignorant of data access concerns.
        Repositories can be implemented using various persistence technologies.
        SQLAlchemy provides a powerful ORM for Python applications.
        The Unit of Work pattern often works alongside repositories.
        Fake repositories enable easy testing without database dependencies.
        """,
        keywords=["repository", "persistence", "abstraction", "ORM", "SQLAlchemy"],
        concepts=["Repository Pattern", "Data Mapper", "Unit of Work"],
        book_id="architecture-patterns-python",
        book_title="Architecture Patterns with Python",
        book_author="Harry Percival and Bob Gregory",
        tier=1,
    ),
    ChapterData(
        id="bm-ch1",
        number=1,
        title="What Are Microservices?",
        content="""
        Microservices are independently deployable services modeled around business domains.
        They communicate via well-defined APIs, typically REST or messaging.
        Each service can be developed, deployed, and scaled independently.
        Bounded contexts from DDD help define service boundaries.
        Microservices enable teams to work autonomously on different services.
        However, they introduce complexity in terms of distributed systems challenges.
        """,
        keywords=["microservices", "distributed", "bounded context", "API", "deployment"],
        concepts=["Microservices Architecture", "Service Decomposition", "Bounded Context"],
        book_id="building-microservices",
        book_title="Building Microservices",
        book_author="Sam Newman",
        tier=1,
    ),
    ChapterData(
        id="bm-ch2",
        number=2,
        title="Service Communication",
        content="""
        Services communicate through synchronous or asynchronous mechanisms.
        REST APIs provide simple request-response communication patterns.
        gRPC offers high-performance binary communication with strong typing.
        Message queues enable asynchronous, decoupled communication.
        Event-driven architecture uses events to notify interested services.
        Circuit breakers prevent cascading failures in service communication.
        """,
        keywords=["REST", "gRPC", "messaging", "async", "circuit breaker"],
        concepts=["Synchronous Communication", "Asynchronous Messaging", "Event-Driven"],
        book_id="building-microservices",
        book_title="Building Microservices",
        book_author="Sam Newman",
        tier=1,
    ),
    ChapterData(
        id="fp-ch1",
        number=1,
        title="The Python Data Model",
        content="""
        Python's data model defines special methods that customize object behavior.
        Dunder methods like __len__ and __getitem__ enable sequence protocols.
        The data model allows objects to work with built-in functions and operators.
        Duck typing means objects are defined by their behavior, not their type.
        Implementing protocols makes objects integrate seamlessly with Python.
        Special methods should be called implicitly by the interpreter.
        """,
        keywords=["dunder", "magic methods", "protocol", "duck typing", "data model"],
        concepts=["Python Data Model", "Duck Typing", "Protocol Implementation"],
        book_id="fluent-python",
        book_title="Fluent Python",
        book_author="Luciano Ramalho",
        tier=3,
    ),
    ChapterData(
        id="fp-ch2",
        number=2,
        title="Sequences",
        content="""
        Python provides rich sequence types including lists, tuples, and strings.
        List comprehensions offer concise ways to create and transform lists.
        Slicing allows extracting portions of sequences with powerful notation.
        Generator expressions provide memory-efficient iteration.
        The bisect module enables efficient sorted sequence operations.
        Named tuples combine tuple efficiency with named field access.
        """,
        keywords=["list", "tuple", "slice", "comprehension", "generator"],
        concepts=["Sequence Protocol", "List Comprehension", "Generator Expression"],
        book_id="fluent-python",
        book_title="Fluent Python",
        book_author="Luciano Ramalho",
        tier=3,
    ),
]


# =============================================================================
# Embedding Functions
# =============================================================================

def get_embedding_model():
    """Load the SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("✗ sentence-transformers not installed")
        print("  Install with: pip install sentence-transformers")
        sys.exit(1)
    
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Model loaded (dimension: {model.get_sentence_embedding_dimension()})")
    return model


def generate_embedding(model, text: str) -> list[float]:
    """Generate embedding for a text string."""
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def generate_chapter_text(chapter: ChapterData) -> str:
    """Generate text for embedding from chapter data."""
    parts = [
        chapter.title,
        chapter.content,
    ]
    
    if chapter.keywords:
        parts.append("Keywords: " + ", ".join(chapter.keywords))
    
    if chapter.concepts:
        parts.append("Concepts: " + ", ".join(chapter.concepts))
    
    if chapter.summary:
        parts.append("Summary: " + chapter.summary)
    
    return "\n".join(parts)


# =============================================================================
# Qdrant Functions
# =============================================================================

def get_qdrant_client():
    """Create Qdrant client."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        print("✗ qdrant-client not installed")
        print("  Install with: pip install qdrant-client")
        sys.exit(1)
    
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)
    
    # Verify connection
    try:
        client.get_collections()
        print("✓ Connected to Qdrant")
    except Exception as e:
        print(f"✗ Failed to connect to Qdrant: {e}")
        sys.exit(1)
    
    return client


def create_collection(client, collection_name: str, vector_size: int) -> None:
    """Create or recreate the collection."""
    from qdrant_client.models import Distance, VectorParams
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name in collection_names:
        print(f"  Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    print(f"✓ Created collection '{collection_name}' (vector_size={vector_size})")


def upsert_chapters(
    client,
    model,
    chapters: list[ChapterData],
    collection_name: str,
) -> None:
    """Upsert chapter embeddings into Qdrant."""
    from qdrant_client.models import PointStruct
    
    points = []
    
    for i, chapter in enumerate(chapters):
        # Generate embedding
        text = generate_chapter_text(chapter)
        embedding = generate_embedding(model, text)
        
        # Create payload
        payload = {
            "chapter_id": chapter.id,
            "chapter_number": chapter.number,
            "title": chapter.title,
            "keywords": chapter.keywords,
            "concepts": chapter.concepts,
            "summary": chapter.summary,
            "page_range": chapter.page_range,
            "book_id": chapter.book_id,
            "book_title": chapter.book_title,
            "book_author": chapter.book_author,
            "tier": chapter.tier,
            "content_preview": chapter.content[:500] if chapter.content else "",
        }
        
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload=payload,
        ))
        
        print(f"  Embedded: {chapter.title}")
    
    # Upsert all points
    client.upsert(
        collection_name=collection_name,
        points=points,
    )
    print(f"✓ Upserted {len(points)} vectors")


def print_summary(client, collection_name: str) -> None:
    """Print summary of collection."""
    info = client.get_collection(collection_name)
    
    print("\n" + "=" * 50)
    print("QDRANT SEEDING SUMMARY")
    print("=" * 50)
    print(f"Collection:    {collection_name}")
    # Use points_count (newer API) with fallback to vectors_count (older API)
    vector_count = getattr(info, 'points_count', None) or getattr(info, 'vectors_count', 'N/A')
    print(f"Vector count:  {vector_count}")
    print(f"Vector size:   {info.config.params.vectors.size}")
    print(f"Distance:      {info.config.params.vectors.distance}")
    print(f"Status:        {info.status}")
    print("=" * 50)


def test_search(client, model, collection_name: str) -> None:
    """Test search functionality."""
    print("\nTesting search...")
    
    test_query = "domain driven design patterns"
    query_embedding = generate_embedding(model, test_query)
    
    # Use query_points for newer qdrant-client API, fallback to search for older versions
    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=3,
        ).points
    except AttributeError:
        # Fallback for older qdrant-client versions
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=3,
        )
    
    print(f"\nSearch results for: '{test_query}'")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.payload.get('title', 'Unknown')} (score: {result.score:.4f})")
        print(f"     Book: {result.payload.get('book_title', 'Unknown')}")
        print(f"     Tier: {result.payload.get('tier', 'Unknown')}")


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_chapters_from_json(json_path: Path) -> list[ChapterData]:
    """Load chapters from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    chapters = []
    book_info = data.get("book", {})
    book_id = data.get("id", json_path.stem)
    book_title = book_info.get("title", data.get("title", "Unknown"))
    book_author = book_info.get("author", data.get("author", "Unknown"))
    tier = book_info.get("tier", data.get("tier", 2))
    
    for ch_data in data.get("chapters", []):
        chapters.append(ChapterData(
            id=ch_data.get("id", f"{book_id}-ch{ch_data.get('number', 0)}"),
            number=ch_data.get("number", 0),
            title=ch_data.get("title", ""),
            content=ch_data.get("content", ch_data.get("text", "")),
            keywords=ch_data.get("keywords", []),
            concepts=ch_data.get("concepts", []),
            summary=ch_data.get("summary", ""),
            page_range=ch_data.get("page_range", ""),
            book_id=book_id,
            book_title=book_title,
            book_author=book_author,
            tier=tier,
        ))
    
    return chapters


def load_chapters_from_directory(dir_path: Path) -> list[ChapterData]:
    """Load chapters from all JSON files in a directory."""
    all_chapters = []
    
    json_files = list(dir_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {dir_path}")
    
    for json_file in json_files:
        try:
            chapters = load_chapters_from_json(json_file)
            all_chapters.extend(chapters)
            print(f"  Loaded {len(chapters)} chapters from {json_file.name}")
        except Exception as e:
            print(f"  ⚠ Failed to load {json_file.name}: {e}")
    
    return all_chapters


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed Qdrant vector database with chapter embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--json",
        type=Path,
        help="Path to a single book JSON file",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Path to a directory containing JSON files",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample data for testing",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"Collection name (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Don't recreate collection, append to existing",
    )
    parser.add_argument(
        "--test-search",
        action="store_true",
        help="Run a test search after seeding",
    )
    
    args = parser.parse_args()
    
    # Determine data source
    if args.sample:
        chapters = SAMPLE_CHAPTERS
        print("Using sample data...")
    elif args.dir:
        if not args.dir.exists():
            print(f"✗ Directory not found: {args.dir}")
            sys.exit(1)
        chapters = load_chapters_from_directory(args.dir)
    elif args.json:
        if not args.json.exists():
            print(f"✗ JSON file not found: {args.json}")
            sys.exit(1)
        chapters = load_chapters_from_json(args.json)
        print(f"Loaded {len(chapters)} chapters from {args.json}")
    else:
        # Default to sample data
        chapters = SAMPLE_CHAPTERS
        print("No input specified, using sample data...")
    
    if not chapters:
        print("✗ No chapters to seed")
        sys.exit(1)
    
    # Initialize clients
    model = get_embedding_model()
    client = get_qdrant_client()
    
    # Create/recreate collection
    if not args.no_recreate:
        create_collection(client, args.collection, VECTOR_SIZE)
    
    # Seed data
    upsert_chapters(client, model, chapters, args.collection)
    
    # Print summary
    print_summary(client, args.collection)
    
    # Optional test search
    if args.test_search:
        test_search(client, model, args.collection)
    
    print("\n✓ Seeding complete!")


if __name__ == "__main__":
    main()
