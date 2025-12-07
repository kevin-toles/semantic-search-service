#!/usr/bin/env python3
"""
Initialize Qdrant collection for Graph RAG POC.

Usage:
    python init_qdrant_collection.py
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    CreateCollection,
    PayloadSchemaType,
)


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "chapters"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimensions


def main():
    """Create Qdrant collection with schema."""
    client = QdrantClient(url=QDRANT_URL)
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME in collection_names:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting...")
        client.delete_collection(COLLECTION_NAME)
    
    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    
    print(f"Created collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}")
    
    # Verify
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"Collection status: {collection_info.status}")
    print(f"Vectors count: {collection_info.vectors_count}")
    
    print("\nQdrant collection initialization complete!")


if __name__ == "__main__":
    main()
