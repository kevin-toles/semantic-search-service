#!/usr/bin/env python3
"""
Verify data integrity in Neo4j and Qdrant.

Usage:
    python verify_data.py
"""

from neo4j import GraphDatabase
from qdrant_client import QdrantClient


# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "pocpassword"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "chapters"


def verify_neo4j(driver):
    """Verify Neo4j data."""
    print("\n=== Neo4j Verification ===")
    
    with driver.session() as session:
        # Count nodes
        result = session.run("MATCH (b:Book) RETURN count(b) as count")
        book_count = result.single()["count"]
        print(f"Books: {book_count}")
        
        result = session.run("MATCH (c:Chapter) RETURN count(c) as count")
        chapter_count = result.single()["count"]
        print(f"Chapters: {chapter_count}")
        
        result = session.run("MATCH (t:Tier) RETURN count(t) as count")
        tier_count = result.single()["count"]
        print(f"Tiers: {tier_count}")
        
        # Count relationships
        result = session.run("MATCH ()-[r:PARALLEL]->() RETURN count(r) as count")
        parallel_count = result.single()["count"]
        print(f"PARALLEL relationships: {parallel_count}")
        
        result = session.run("MATCH ()-[r:PERPENDICULAR]->() RETURN count(r) as count")
        perpendicular_count = result.single()["count"]
        print(f"PERPENDICULAR relationships: {perpendicular_count}")
        
        result = session.run("MATCH ()-[r:SKIP_TIER]->() RETURN count(r) as count")
        skip_tier_count = result.single()["count"]
        print(f"SKIP_TIER relationships: {skip_tier_count}")
        
        # Sample books by tier
        print("\nBooks by Tier:")
        result = session.run("""
            MATCH (b:Book)-[:BELONGS_TO]->(t:Tier)
            RETURN t.level as tier, collect(b.title) as books
            ORDER BY t.level
        """)
        for record in result:
            print(f"  Tier {record['tier']}: {', '.join(record['books'])}")


def verify_qdrant(client: QdrantClient):
    """Verify Qdrant data."""
    print("\n=== Qdrant Verification ===")
    
    # Collection info
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Status: {collection_info.status}")
    print(f"Vectors count: {collection_info.vectors_count}")
    print(f"Points count: {collection_info.points_count}")
    
    # Sample search
    if collection_info.points_count > 0:
        print("\nSample search (query: 'clean code functions'):")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = model.encode("clean code functions").tolist()
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
        )
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.payload['book_title']} - Ch.{result.payload['chapter_number']}: {result.payload['chapter_title']} (score: {result.score:.3f})")


def main():
    """Main entry point."""
    # Initialize clients
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    qdrant_client = QdrantClient(url=QDRANT_URL)
    
    try:
        verify_neo4j(neo4j_driver)
        verify_qdrant(qdrant_client)
        print("\n✅ Data verification complete!")
    finally:
        neo4j_driver.close()


if __name__ == "__main__":
    main()
