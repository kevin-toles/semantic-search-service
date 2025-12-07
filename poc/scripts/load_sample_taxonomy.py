#!/usr/bin/env python3
"""
Load sample taxonomy data into Neo4j and Qdrant.

Usage:
    python load_sample_taxonomy.py
"""

import json
from pathlib import Path
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer


# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "pocpassword"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "chapters"
DATA_FILE = Path(__file__).parent.parent / "data" / "sample_taxonomy.json"


def load_neo4j_data(driver, taxonomy_data: dict):
    """Load books, chapters, and relationships into Neo4j."""
    with driver.session() as session:
        # Load books
        for book in taxonomy_data["books"]:
            session.run("""
                MERGE (b:Book {id: $id})
                SET b.title = $title,
                    b.tier = $tier,
                    b.author = $author,
                    b.year = $year,
                    b.category = $category
                WITH b
                MATCH (t:Tier {level: $tier})
                MERGE (b)-[:BELONGS_TO]->(t)
                WITH b
                MATCH (tx:Taxonomy {id: 'software-engineering'})
                MERGE (b)-[:IN_TAXONOMY]->(tx)
            """, **book)
            print(f"  Loaded book: {book['title']}")
            
            # Load chapters
            for chapter in book.get("chapters", []):
                session.run("""
                    MERGE (c:Chapter {id: $id})
                    SET c.book_id = $book_id,
                        c.number = $number,
                        c.title = $title,
                        c.keywords = $keywords,
                        c.concepts = $concepts,
                        c.summary = $summary,
                        c.page_range = $page_range
                    WITH c
                    MATCH (b:Book {id: $book_id})
                    MERGE (c)-[:PART_OF]->(b)
                """, book_id=book["id"], **chapter)
        
        # Load relationships
        for rel in taxonomy_data.get("relationships", []):
            rel_type = rel["type"].upper()
            session.run(f"""
                MATCH (b1:Book {{id: $from_book}})
                MATCH (b2:Book {{id: $to_book}})
                MERGE (b1)-[r:{rel_type}]->(b2)
                SET r.similarity = $similarity
            """, 
                from_book=rel["from"],
                to_book=rel["to"],
                similarity=rel.get("similarity", 0.8)
            )
            print(f"  Created {rel_type} relationship: {rel['from']} -> {rel['to']}")


def load_qdrant_data(client: QdrantClient, model: SentenceTransformer, taxonomy_data: dict):
    """Load chapter embeddings into Qdrant."""
    points = []
    point_id = 0
    
    for book in taxonomy_data["books"]:
        for chapter in book.get("chapters", []):
            # Create embedding from title + summary
            text = f"{chapter['title']}. {chapter['summary']}"
            embedding = model.encode(text).tolist()
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "chapter_id": chapter["id"],
                    "book_id": book["id"],
                    "book_title": book["title"],
                    "chapter_number": chapter["number"],
                    "chapter_title": chapter["title"],
                    "tier": book["tier"],
                    "keywords": chapter["keywords"],
                    "concepts": chapter["concepts"],
                    "summary": chapter["summary"],
                }
            )
            points.append(point)
            point_id += 1
    
    # Upsert in batches
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    print(f"  Loaded {len(points)} chapter embeddings into Qdrant")


def main():
    """Main entry point."""
    # Load sample data
    if not DATA_FILE.exists():
        print(f"Error: Data file not found: {DATA_FILE}")
        print("Please create sample_taxonomy.json first")
        return
    
    with open(DATA_FILE) as f:
        taxonomy_data = json.load(f)
    
    print(f"Loaded {len(taxonomy_data['books'])} books from {DATA_FILE}")
    
    # Initialize clients
    print("\nConnecting to Neo4j...")
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    print("Connecting to Qdrant...")
    qdrant_client = QdrantClient(url=QDRANT_URL)
    
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load data
    print("\nLoading data into Neo4j...")
    load_neo4j_data(neo4j_driver, taxonomy_data)
    
    print("\nLoading embeddings into Qdrant...")
    load_qdrant_data(qdrant_client, model, taxonomy_data)
    
    # Cleanup
    neo4j_driver.close()
    
    print("\n✅ Sample taxonomy data loaded successfully!")


if __name__ == "__main__":
    main()
